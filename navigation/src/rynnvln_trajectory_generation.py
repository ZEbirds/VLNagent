import os
import json
import random
import copy
import argparse
import multiprocessing as mp
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm


import torch
from torch.utils.data import Dataset

from transformers import PretrainedConfig, ProcessorMixin


class SFTDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def _preprocess(self, data_dict: Dict) -> Dict:
        return data_dict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError



class VLNDataArgs:
    def __init__(self, num_frames: int = 32, num_history: int = 8, num_future_steps: int = 4, remove_init_turns: bool = False):
        self.num_frames = num_frames
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.remove_init_turns = remove_init_turns

class VLN_SFT_Dataset(SFTDataset):
    def __init__(
        self,
        ann_path: List[str],
        data_root: str,
        vln_args: VLNDataArgs,
        model_config: Optional[PretrainedConfig] = None,
        processor: Optional[ProcessorMixin] = None,
        model_max_length: int = 2048,
        **kwargs,
    ):
        self.data_root = data_root
        self.vln_args = vln_args
        self.nav_data, self.data_list = self._load_and_slice_data(ann_path, data_root)
        self._dataset = self.data_list

        self.conjunctions = [
            'you can see ', 'in front of you is ', 'there is ', 'you can spot ',
            'you are toward the ', 'ahead of you is ', 'in your sight is '
        ]
        self.idx2actions = {'0': 'STOP', '1': "↑", '2': "←", '3': "→"}
        self.prompt_template = "You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←), TURN RIGHT (→), MOVE FORWARD (↑), or STOP."

    def _load_and_slice_data(self, ann_path: List[str], data_root: str):
        nav_data = []
        for path in ann_path:
            full_ann_path = os.path.join(data_root, path)
            dataset_folder = os.path.dirname(full_ann_path)
            with open(full_ann_path, 'r') as f:
                anno_json = json.load(f)
            for tdata in anno_json:
                tdata['video'] = os.path.join(dataset_folder, tdata['video'])
            nav_data.extend(anno_json)

        data_list = []
        for ep_id, item in enumerate(nav_data):
            actions = item['actions']
            actions_len = len(actions)
            if actions_len < self.vln_args.num_future_steps:
                continue

            instructions = item.get('instructions', [])
            if not isinstance(instructions, list):
                instructions = [instructions]
                
            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.vln_args.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)

                effective_len = actions_len - valid_idx
                if effective_len < self.vln_args.num_future_steps:
                    continue

                for start_idx in range(0, effective_len, self.vln_args.num_frames):
                    if start_idx < effective_len:
                        data_list.append((ep_id, ins_id, start_idx, valid_idx))
        return nav_data, data_list

    def __len__(self):
        return len(self.data_list)

    def actions2text(self, actions: List[int]) -> str:
        return ''.join([self.idx2actions.get(str(a), '') for a in actions])

    def clean_initial_rotations(self, instruction, actions):
        valid_idx = 0
        for act in actions:
            if act in [2, 3]: valid_idx += 1
            else: break
        if "turn" not in instruction.lower() and "rotate" not in instruction.lower():
            return valid_idx
        return 0

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        try:
            ep_id, ins_id, start_idx, valid_idx = self.data_list[index]
            data = self.nav_data[ep_id]
            # video_path = data['video']
            rgb_dir = os.path.join(self.data_root, 'rgb')
            video_frames = sorted(os.listdir(rgb_dir))
            instruction = data.get("instructions", [])[ins_id]


            actions_full = data['actions'][1+valid_idx:] + [0]
            actions_len = len(actions_full)
            time_ids = np.arange(start_idx, min(start_idx + self.vln_args.num_frames, actions_len))
            if len(time_ids) == 0:
                raise ValueError("Calculated empty time_ids, skipping sample.")

            target_actions = np.array(actions_full)[time_ids].tolist()

            start_t, end_t = time_ids[0] + valid_idx, time_ids[-1] + 1 + valid_idx
            
            sample_step_ids = np.arange(start_t, end_t, self.vln_args.num_future_steps, dtype=np.int32)
            sample_frames_paths = [os.path.join(rgb_dir, video_frames[i]) for i in sample_step_ids if i < len(video_frames)]
            
            history_frames_paths = []
            if time_ids[0] > 0:
                history_end_t = time_ids[0] + valid_idx
                step = max(history_end_t // self.vln_args.num_history, 1)
                history_step_ids = np.arange(valid_idx, history_end_t, step)
                history_frames_paths = [os.path.join(rgb_dir, video_frames[i]) for i in history_step_ids if i < len(video_frames)]

            conversations = []
            user_content = []
            prompt_text = self.prompt_template.replace('<instruction>', instruction)
            if history_frames_paths:
                prompt_text += " These are your historical observations: "
            user_content.append({"type": "text", "text": prompt_text})

            for img_path in history_frames_paths:
                user_content.append({"type": "image", "image": img_path})
            
            user_content.append({"type": "text", "text": " This is your current view: "})
            
            if not sample_frames_paths:
                 raise ValueError("sample_frames_paths is empty.")

            user_content.append({"type": "image", "image": sample_frames_paths[0]})
            step_actions = target_actions[0:self.vln_args.num_future_steps]

            conversation = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": self.actions2text(step_actions)}]}
            ]
            conversations.extend(conversation)

            for i in range(1, len(sample_frames_paths)):
                user_content = []
                prompt = random.choice(self.conjunctions)
                user_content.append({"type": "text", "text": f" {prompt}"})
                user_content.append({"type": "image", "image": sample_frames_paths[i]})
                
                start_action_idx = i * self.vln_args.num_future_steps
                end_action_idx = start_action_idx + self.vln_args.num_future_steps
                step_actions = target_actions[start_action_idx:end_action_idx]

                if not step_actions: continue

                conversation = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": self.actions2text(step_actions)}]}
                ]
                conversations.extend(conversation)
            
            return conversations

        except Exception as e:
            import sys
            print(f"Error when processing {index} : {e}", file=sys.stderr)
            return None

g_dataset = None

def init_worker(args_dict):
    global g_dataset
    
    vln_args = VLNDataArgs(
        num_frames=args_dict['num_frames'],
        num_history=args_dict['num_history'],
        num_future_steps=args_dict['num_future_steps'],
        remove_init_turns=args_dict['remove_init_turns']
    )
    
    g_dataset = VLN_SFT_Dataset(
        ann_path=args_dict['ann_path'],
        data_root=args_dict['data_root'],
        vln_args=vln_args
    )

def process_index(index: int) -> Optional[Dict[str, Any]]:
    global g_dataset
    if g_dataset is None:
        return None

    conversations = g_dataset[index]
    
    if conversations is None:
        return None
        
    record = {
        "id": f"vln_sample_{index}",
        "data_source": '', 
        "conversation": conversations
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Use multi-processing to generate SFT-format data files from the VLN dataset")
    parser.add_argument("--data-root", type=str, required=True, help="data root")
    parser.add_argument("--ann-path", type=str, nargs='+', required=True, help="Path(s) to annotation files, relative to the data root.")
    parser.add_argument("--output-file", type=str, default="output_vln_sft_data.jsonl", help="Path for the output JSON Lines file.")
    parser.add_argument("--num-frames", type=int, default=32, help="The number of action steps in each sample.")
    parser.add_argument("--num-history", type=int, default=8, help="The number of historical frames to include as context.")
    parser.add_argument("--num-future-steps", type=int, default=4, help="The number of future action steps the model should predict.")
    parser.add_argument("--remove-init-turns", action='store_true', help="If set, removes the initial turning actions from each trajectory")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(), help="The number of processes to use for data generation.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"Error: data-root does not exists: {args.data_root}"); return
    for ann_p in args.ann_path:
        if not os.path.isfile(os.path.join(args.data_root, ann_p)):
            print(f"Error file does not exists {os.path.join(args.data_root, ann_p)}"); return

    vln_args = VLNDataArgs(args.num_frames, args.num_history, args.num_future_steps, args.remove_init_turns)
    temp_dataset = VLN_SFT_Dataset(ann_path=args.ann_path, data_root=args.data_root, vln_args=vln_args)
    num_samples = len(temp_dataset)
    del temp_dataset
    



    
    indices = range(num_samples)
    init_args_dict = vars(args) 

    with open(args.output_file, 'w', encoding='utf-8') as f:
        with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(init_args_dict,)) as pool:
            results_iterator = pool.imap_unordered(process_index, indices)
            
            for record in tqdm(results_iterator, total=num_samples, desc="Processing samples"):
                if record:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n--- finished ---")
    print(f"Data successfully generated and saved to: {args.output_file}")

if __name__ == "__main__":
    if os.name != 'posix':
        mp.set_start_method('spawn', force=True)
    main()
