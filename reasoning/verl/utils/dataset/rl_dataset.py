# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional
from pathlib import Path

import datasets
import numpy as np
import torch
import random
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


def _expand_dir_to_frame_list(base: Path, value):
    
    """If value is a directory path, expand it to a sorted list of image file paths.

    Keeps original value when it's already a list, a file path, or a URL.
    """
    try:
        if isinstance(value, str):
            # Skip URLs
            if value.startswith("http://") or value.startswith("https://"):
                return value
            p = Path(value)
            if not p.is_absolute():
                p = base / p
            if p.exists() and p.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
                files = [
                    f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts
                ]
                # Natural sort by filename to preserve frame order like 1,2,10
                files.sort(
                    key=lambda f: [
                        int(t) if t.isdigit() else t.lower()
                        for t in re.split(r"(\d+)", f.name)
                    ]
                )
                return [str(f.resolve()) for f in files]
    except Exception:
        # Fallback to original value on any issue
        return value
    return value


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        # self.processor.video_processor.size["longest_edge"]  = 10240*32*32
        # self.processor.video_processor.size["shortest_edge"] = 64*32*32
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "conversation")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 16)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        ### Qwen3-VL Arguments
        # 动态分辨率：根据图片数量自动调整
        self.image_max_pixels_high = config.get("image_max_pixels_high", 1024*32*32)  # 高分辨率（图片少时）
        self.image_max_pixels_low = config.get("image_max_pixels_low", 64*32*32)      # 低分辨率（图片多时）
        self.image_adaptive_threshold = config.get("image_adaptive_threshold", 5)     # 阈值：>=5张用低分辨率
        
        # 兼容旧配置
        self.image_max_pixels = config.get("image_max_pixels", self.image_max_pixels_high)
        self.image_min_pixels = config.get("image_min_pixels", 32*32*32)
        self.video_max_pixels = config.get("video_max_pixels", 10240*32*32)
        self.video_min_pixels = config.get("video_min_pixels", 64*32*32)
        self.max_images_per_sample = config.get("max_images_per_sample", 32)  # Limit images to avoid OOM
        
        # 默认使用高分辨率设置（会在 __getitem__ 中动态调整）
        self.processor.image_processor.size["longest_edge"] = self.image_max_pixels_high
        self.processor.image_processor.size["shortest_edge"] = self.image_min_pixels
        self.processor.video_processor.size["longest_edge"] = self.video_max_pixels
        self.processor.video_processor.size["shortest_edge"] = self.video_min_pixels

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        data_source = doc.get("data_source", "")
                        messages = self._build_messages(doc, data_source=data_source)
                        
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )

                        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                    image_patch_size=16,
                                                                    return_video_metadata=True)
                        
                        if video_inputs is not None:
                            video_inputs, video_metadatas = zip(*video_inputs)
                            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
                        else:
                            video_metadatas = None

                        # print(raw_prompt)

                        return len(
                            processor(text=[raw_prompt], images=image_inputs, videos=video_inputs, videos_kwargs=video_kwargs, video_metadata=video_metadatas, do_resize=True)["input_ids"][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)
    
    def _convert_conversation_rynnbrain(self, conversation):
        new_conversation = []
        for message in conversation:
            if message["role"] == "user":
                image_idx = 0
                new_contents = []
                for i, content in enumerate(message["content"]):
                    if content["type"] == "image":
                        last_content = message["content"][i - 1] if i > 0 else content
                        if last_content["type"] != "text" or "<frame " not in last_content["text"]:
                            new_contents.append({"type": "text", "text": f"<frame {image_idx}>: "})
                            image_idx += 1
                    new_contents.append(content)
                new_conversation.append({"role": message["role"], "content": new_contents})
            else:
                new_conversation.append(message)

        
        return new_conversation

    def _check_file_exists(self, file_path: str) -> bool:
        """Check if a file exists and is accessible."""
        if not file_path:
            return False
        # Skip URL checks - assume they are valid
        if isinstance(file_path, str) and (file_path.startswith("http://") or file_path.startswith("https://")):
            return True
        try:
            return os.path.isfile(file_path)
        except Exception:
            return False

    def _filter_existing_video_frames(self, frames: list) -> list:
        """Filter video frames to only include existing files."""
        if not frames:
            return frames
        existing_frames = [f for f in frames if self._check_file_exists(f)]
        if len(existing_frames) != len(frames):
            logger.warning(f"Filtered out {len(frames) - len(existing_frames)} non-existing video frames out of {len(frames)}")
        return existing_frames

    def _count_media_in_messages(self, messages: list) -> tuple:
        """Count the number of images and videos declared in messages."""
        num_images = 0
        num_videos = 0
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for seg in content:
                    if isinstance(seg, dict):
                        if seg.get("type") == "image":
                            num_images += 1
                        elif seg.get("type") == "video":
                            num_videos += 1
        return num_images, num_videos

    def _build_messages(self, example: dict, data_source: str = None):
        messages: list = example.pop(self.prompt_key)

        converted_messages = []
        for message in messages:
            role = message.get("role")
            if role == "user":
                new_content = []
                content = message.get("content", [])
                num_images = 0
                image_indices = []  # Track indices of image segments
                
                for idx, seg in enumerate(content):
                    for key in ["action", "navigation", "region", "affordance", "state", "trajectory", "image", "image_url", "video"]:
                        if key != seg["type"]:
                            seg.pop(key, None)
                    # Skip seg if all values (except 'type') are None
                    if all(v is None for k, v in seg.items() if k != "type"):
                        continue
                    # Expand folder path for video into list of frame paths
                    if seg["type"] == "video" and "video" in seg:
                        expanded = seg.get("video")
                        # Filter to only existing frames
                        if isinstance(expanded, list):
                            expanded = self._filter_existing_video_frames(expanded)
                            if not expanded:
                                logger.warning("No valid video frames found, converting to text placeholder")
                                seg = {"type": "text", "text": "[Video not available]"}
                                new_content.append(seg)
                                continue
                        seg["video"] = expanded
                        seg["sample_fps"] = 2.0
                        if len(expanded) > 1024:
                            seg["video"] = sorted(random.sample(expanded, 1024))
                        if isinstance(expanded, list):
                            seg["video_metadata"] = {"fps": 1.0, "total_frames": len(seg["video"])}
                    # Validate image exists before including it
                    if seg["type"] == "image":
                        image_path = seg.get("image")
                        if not self._check_file_exists(image_path):
                            logger.warning(f"Image file not found or not accessible: {image_path}, skipping...")
                            # Convert to text placeholder to maintain message structure
                            seg = {"type": "text", "text": f"[Image not available: {os.path.basename(str(image_path)) if image_path else 'unknown'}]"}
                        else:
                            num_images += 1
                            image_indices.append(len(new_content))  # Track this image's position
                    new_content.append(seg)
                
                # Limit images per sample to avoid OOM
                if num_images > self.max_images_per_sample and image_indices:
                    # Uniformly sample images to keep
                    keep_count = self.max_images_per_sample
                    step = len(image_indices) / keep_count
                    keep_indices = set(image_indices[int(i * step)] for i in range(keep_count))
                    
                    # Filter content, removing excess images and their preceding <frame N>: labels
                    filtered_content = []
                    skip_next_text = False
                    for i, seg in enumerate(new_content):
                        if i in image_indices and i not in keep_indices:
                            # Remove preceding frame label if exists
                            if filtered_content and filtered_content[-1].get("type") == "text":
                                last_text = filtered_content[-1].get("text", "")
                                if "<frame " in last_text and last_text.endswith(">: "):
                                    filtered_content.pop()
                            continue
                        filtered_content.append(seg)
                    new_content = filtered_content
                    logger.warning(f"Reduced images from {num_images} to {keep_count} to avoid OOM")
                
                converted_messages.append({"role": role, "content": new_content})

        # Only apply rynnbrain conversion for non-general data sources
        if data_source is None or "general" not in data_source.lower():
            converted_messages = self._convert_conversation_rynnbrain(converted_messages)

        return converted_messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        data_source = row_dict.get("data_source", "")
        messages = self._build_messages(row_dict, data_source=data_source)
        
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            # 动态分辨率：根据图片数量调整
            num_images, num_videos = self._count_media_in_messages(messages)
            if num_images >= self.image_adaptive_threshold:
                # 图片多时使用低分辨率
                self.processor.image_processor.size["longest_edge"] = self.image_max_pixels_low
            else:
                # 图片少时使用高分辨率
                self.processor.image_processor.size["longest_edge"] = self.image_max_pixels_high

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}


            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                    image_patch_size=16,
                                                                    return_video_metadata=True)
            
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None

            # Validate that the number of images/videos matches what's declared in messages
            # expected_images, expected_videos = self._count_media_in_messages(messages)
            # actual_images = len(image_inputs) if image_inputs else 0
            # actual_videos = len(video_inputs) if video_inputs else 0

            # print(expected_images, actual_images)
            
            # if expected_images != actual_images:
            # logger.warning(
            #     f"Image count mismatch: expected {expected_images} from messages, "
            #     f"but process_vision_info returned {actual_images}. "
            #     f"This may cause vision token mismatch errors."
            # )
            # # if expected_videos != actual_videos:
            # logger.warning(
            #     f"Video count mismatch: expected {expected_videos} from messages, "
            #     f"but process_vision_info returned {actual_videos}. "
            #     f"This may cause vision token mismatch errors."
            # )

            if image_inputs:
                multi_modal_data["image"] = image_inputs

            if video_inputs:
                multi_modal_data["video"] = [
                    (video.numpy(), metadata, video_kwargs) for video, metadata in zip(video_inputs, video_metadatas, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=image_inputs, videos=video_inputs, videos_kwargs=video_kwargs, video_metadata=video_metadatas,  return_tensors="pt", do_resize=True
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
