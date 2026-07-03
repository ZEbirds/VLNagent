from __future__ import annotations

import io
import json
import logging
import math
import os
from collections.abc import Iterable as IterableABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

Point = Tuple[float, float]
RawPoints = Union[str, Sequence[Sequence[Union[int, float]]]]
ImageLike = Union[str, Path, Image.Image, bytes, bytearray, Dict[str, Any]]


class _PromptRenderer:
    """Utility class that renders structured prompts on top of images."""

    SUPPORTED_TYPES = {"area", "affordance", "trajectory", "object"}

    def __init__(self, images: Optional[Iterable[ImageLike]] = None, *, auto_convert: bool = True) -> None:
        self._images: List[Image.Image] = []
        self._auto_convert = auto_convert

        if images is not None:
            for image in images:
                self.add_image(image)

        self._dispatch_table = {
            "area": self._apply_area_prompt,
            "affordance": self._apply_affordance_prompt,
            "trajectory": self._apply_trajectory_prompt,
            "object": self._apply_object_prompt,
        }

    def add_image(self, image: ImageLike) -> int:
        pil_image = self._to_pil_image(image)
        if self._auto_convert and pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        self._images.append(pil_image)
        return len(self._images) - 1

    def update_image(self, idx: int, image: ImageLike) -> None:
        if idx < 0 or idx >= len(self._images):
            raise IndexError(f"Image index {idx} is out of bounds.")
        pil_image = self._to_pil_image(image)
        if self._auto_convert and pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        self._images[idx] = pil_image

    def prompt(
        self,
        prompt_type: str,
        points: RawPoints,
        img_idx: int,
        *,
        image: Optional[ImageLike] = None,
        inplace: bool = False,
    ) -> Image.Image:
        prompt_type = prompt_type.lower()
        if prompt_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported prompt type: {prompt_type}. Expected one of {self.SUPPORTED_TYPES}.")

        parsed_points = self._parse_points(points)
        base_image = self._select_image(img_idx, image=image, inplace=inplace)
        handler = self._dispatch_table[prompt_type]
        return handler(base_image, parsed_points)

    def num_images(self) -> int:
        return len(self._images)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _select_image(self, idx: int, *, image: Optional[ImageLike], inplace: bool) -> Image.Image:
        if image is not None:
            pil_image = self._to_pil_image(image)
            if self._auto_convert and pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return pil_image

        if not self._images:
            raise ValueError("Image buffer is empty. Use `add_image` or provide `image` directly.")
        if idx < 0 or idx >= len(self._images):
            raise IndexError(f"Image index {idx} is out of bounds (buffer size: {len(self._images)}).")
        return self._images[idx] if inplace else self._images[idx].copy()

    @staticmethod
    def _to_pil_image(image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy()
        if isinstance(image, (bytes, bytearray)):
            with Image.open(io.BytesIO(image)) as img:
                return img.convert("RGB")
        if isinstance(image, dict):
            pil_image = fetch_image(image)
            return pil_image.convert("RGB")
        if isinstance(image, (str, Path)):
            try:
                pil_image = fetch_image({"image": str(image)})
            except Exception:
                with Image.open(image) as img:
                    pil_image = img.convert("RGB")
            return pil_image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}.")

    @staticmethod
    def _parse_points(points: RawPoints) -> List[Point]:
        if isinstance(points, str):
            try:
                parsed = json.loads(points)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse points JSON: {exc}") from exc
        else:
            parsed = points

        if not isinstance(parsed, Sequence) or not parsed:
            raise ValueError("Points must be a non-empty sequence.")

        parsed_points: List[Point] = []
        for idx, point in enumerate(parsed):
            if not isinstance(point, Sequence) or len(point) < 2:
                raise ValueError(f"Point #{idx} is invalid: {point}")
            try:
                raw_x, raw_y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Point #{idx} has non-numeric values: {point}") from exc

            if not raw_x.is_integer() or not raw_y.is_integer():
                raise ValueError(
                    f"Point #{idx} must contain integer coordinates in the range [0, 1000], got {point}."
                )
            x, y = int(raw_x), int(raw_y)
            parsed_points.append((x, y))

        return parsed_points

    @staticmethod
    def _maybe_scale_points(points: List[Point], size: Tuple[int, int]) -> List[Point]:
        width, height = size
        if width <= 0 or height <= 0:
            return points

        scaled_points: List[Point] = []
        for x, y in points:
            clamped_x = min(max(x, 0.0), 1000.0)
            clamped_y = min(max(y, 0.0), 1000.0)
            scaled_points.append(
                (
                    clamped_x / 1000.0 * width,
                    clamped_y / 1000.0 * height,
                )
            )
        return scaled_points

    @staticmethod
    def _clamp_points(points: List[Point], size: Tuple[int, int]) -> List[Point]:
        width, height = size
        return [
            (min(max(x, 0.0), width - 1), min(max(y, 0.0), height - 1))
            for x, y in points
        ]

    @staticmethod
    def _stroke_width(image: Image.Image, min_width: int = 3) -> int:
        return max(min_width, round(max(image.size) * 0.004))

    @staticmethod
    def _point_radius(image: Image.Image) -> int:
        return max(4, round(max(image.size) * 0.01))

    # ------------------------------------------------------------------ #
    # Prompt handlers
    # ------------------------------------------------------------------ #
    def _apply_area_prompt(self, image: Image.Image, points: List[Point]) -> Image.Image:
        scaled_points = self._clamp_points(self._maybe_scale_points(points, image.size), image.size)
        if len(scaled_points) == 1:
            raise ValueError("`area` prompt requires at least two points.")
        if len(scaled_points) == 2:
            p1, p2 = scaled_points
            polygon = [
                (p1[0], p1[1]),
                (p2[0], p1[1]),
                (p2[0], p2[1]),
                (p1[0], p2[1]),
            ]
        else:
            polygon = scaled_points

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.polygon(
            polygon,
            fill=(255, 99, 71, 90),
            outline=(255, 99, 71, 220),
            width=self._stroke_width(image),
        )

        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _apply_affordance_prompt(self, image: Image.Image, points: List[Point]) -> Image.Image:
        scaled_points = self._clamp_points(self._maybe_scale_points(points, image.size), image.size)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        radius = self._point_radius(image)
        total = max(1, len(scaled_points) - 1)
        for idx, (x, y) in enumerate(scaled_points):
            intensity = int(180 - idx * (120 / total))
            fill = (255, 165, 0, max(80, intensity))
            outline = (255, 140, 0, 200)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)

        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _apply_trajectory_prompt(self, image: Image.Image, points: List[Point]) -> Image.Image:
        scaled_points = self._clamp_points(self._maybe_scale_points(points, image.size), image.size)
        if len(scaled_points) < 2:
            raise ValueError("`trajectory` prompt requires at least two points.")

        annotated = image.convert("RGBA")
        draw = ImageDraw.Draw(annotated, "RGBA")
        stroke = self._stroke_width(image, min_width=4)

        draw.line(scaled_points, fill=(34, 197, 94, 255), width=stroke, joint="curve")
        self._draw_arrow_head(draw, scaled_points[-2], scaled_points[-1], stroke, color=(34, 197, 94, 255))
        self._draw_point_indices(draw, scaled_points, font_color=(255, 255, 255, 255), image=image)

        return annotated.convert("RGB")

    def _apply_object_prompt(self, image: Image.Image, points: List[Point]) -> Image.Image:
        scaled_points = self._clamp_points(self._maybe_scale_points(points, image.size), image.size)
        xs = [x for x, _ in scaled_points]
        ys = [y for _, y in scaled_points]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        annotated = image.convert("RGBA")
        draw = ImageDraw.Draw(annotated, "RGBA")
        stroke = self._stroke_width(image, min_width=3)

        draw.rectangle(bbox, outline=(66, 135, 245, 255), width=stroke)
        self._highlight_corners(draw, bbox, stroke)

        return annotated.convert("RGB")

    # ------------------------------------------------------------------ #
    # Drawing utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_arrow_head(
        draw: ImageDraw.ImageDraw,
        start: Point,
        end: Point,
        stroke: int,
        *,
        color: Tuple[int, int, int, int],
    ) -> None:
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        length = max(stroke * 2.5, 12)
        left = (
            end[0] - length * math.cos(angle - math.pi / 6),
            end[1] - length * math.sin(angle - math.pi / 6),
        )
        right = (
            end[0] - length * math.cos(angle + math.pi / 6),
            end[1] - length * math.sin(angle + math.pi / 6),
        )
        draw.polygon([end, left, right], fill=color)

    @staticmethod
    def _draw_point_indices(
        draw: ImageDraw.ImageDraw,
        points: Sequence[Point],
        *,
        font_color: Tuple[int, int, int, int],
        image: Image.Image,
    ) -> None:
        radius = max(4, round(max(image.size) * 0.01))
        for idx, (x, y) in enumerate(points):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(34, 197, 94, 220),
                outline=(16, 185, 129, 255),
            )
            draw.text((x + radius + 2, y - radius), str(idx), fill=font_color)

    @staticmethod
    def _highlight_corners(
        draw: ImageDraw.ImageDraw,
        bbox: Tuple[float, float, float, float],
        stroke: int,
    ) -> None:
        x1, y1, x2, y2 = bbox
        length = max(10, stroke * 3)
        corners = [
            ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
        ]
        for corner in corners:
            draw.line([corner[0], corner[1]], fill=(66, 135, 245, 255), width=stroke)
            draw.line([corner[0], corner[2]], fill=(66, 135, 245, 255), width=stroke)


class ImagePromptTool(BaseTool):
    """VERL-compatible tool wrapper built on top of the prompt renderer."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        tool_schema = tool_schema or self._build_default_schema()
        super().__init__(config, tool_schema)
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._default_auto_convert = config.get("auto_convert", True)
        self._max_images = config.get("max_images", 64)
        
        # Debug configuration for saving intermediate images
        self._debug_save_images = config.get("debug_save_images", True)
        self._debug_save_dir = Path(config.get("debug_save_dir", "./debug_tool_images"))
        self._debug_save_input = config.get("debug_save_input", True)
        self._debug_save_output = config.get("debug_save_output", True)
        
        if self._debug_save_images:
            self._debug_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ImagePromptTool] Debug mode enabled. Saving images to: {self._debug_save_dir}")

    def _build_default_schema(self) -> OpenAIFunctionToolSchema:
        schema_dict = {
            "type": "function",
            "function": {
                "name": "image_prompt_tool",
                "description": (
                    "Draw task-specific visual prompts (area, affordance, trajectory, object) on a buffered image. "
                    "Coordinates should be provided in 0-1000 normalized range."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt_type": {
                            "type": "string",
                            "description": "Prompt type. Supported values: area, affordance, trajectory, object.",
                            "enum": sorted(list(_PromptRenderer.SUPPORTED_TYPES)),
                        },
                        "points": {
                            "type": "string",
                            "description": "JSON string of [[x, y], ...] with each coordinate in [0, 1000].",
                        },
                        "img_idx": {
                            "type": "integer",
                            "description": "Index of the buffered image to annotate (0-based).",
                        },
                        "inplace": {
                            "type": "boolean",
                            "description": "If true, replace the stored image with the annotated version.",
                        },
                        "image": {
                            "type": "string",
                            "description": "Optional image URI/base64 payload to annotate instead of buffered image.",
                        },
                    },
                    "required": ["prompt_type", "points", "img_idx"],
                },
                "strict": False,
            },
        }
        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            from uuid import uuid4

            instance_id = str(uuid4())

        create_kwargs = kwargs.pop("create_kwargs", {})
        merged_kwargs = {**create_kwargs, **kwargs}

        images = merged_kwargs.get("images")
        if images is None and (single_image := merged_kwargs.get("image")) is not None:
            images = [single_image]

        auto_convert = merged_kwargs.get("auto_convert", self._default_auto_convert)

        renderer = _PromptRenderer(images=self._limit_images(images), auto_convert=auto_convert)
        
        # Create a short instance id for file naming (first 8 chars)
        short_id = instance_id[:8] if len(instance_id) > 8 else instance_id
        
        self._instances[instance_id] = {
            "renderer": renderer,
            "auto_convert": auto_convert,
            "call_count": 0,  # Track number of execute calls for debug naming
            "short_id": short_id,
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        
        # Save initial input images if debug mode is enabled
        if self._debug_save_images and self._debug_save_input and images:
            self._save_debug_images(
                instance_id=instance_id,
                images=renderer._images,
                stage="init",
                call_idx=0,
            )
        
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        instance = self._instances.get(instance_id)
        if instance is None:
            return ToolResponse(text="Error: tool instance not initialized."), -0.05, {"success": False}

        if not isinstance(parameters, dict):
            return ToolResponse(text="Error: parameters must be a JSON object."), -0.05, {"success": False}

        renderer: _PromptRenderer = instance["renderer"]
        prompt_type = parameters.get("prompt_type")
        points = parameters.get("points")
        img_idx = parameters.get("img_idx")
        inplace = bool(parameters.get("inplace", False))
        image_override = parameters.get("image")

        if prompt_type is None or points is None:
            return ToolResponse(text="Error: missing `prompt_type` or `points`."), -0.05, {"success": False}

        if image_override is None:
            if img_idx is None:
                return ToolResponse(text="Error: `img_idx` is required when no image is provided."), -0.05, {
                    "success": False
                }
            if not isinstance(img_idx, int):
                return ToolResponse(text="Error: `img_idx` must be an integer."), -0.05, {"success": False}
        else:
            img_idx = img_idx if isinstance(img_idx, int) else 0

        # Increment call count for debug naming
        instance["call_count"] += 1
        call_idx = instance["call_count"]

        # Save input image before annotation if debug mode is enabled
        if self._debug_save_images and self._debug_save_input:
            if image_override is None and 0 <= img_idx < renderer.num_images():
                self._save_debug_images(
                    instance_id=instance_id,
                    images=[renderer._images[img_idx]],
                    stage=f"call{call_idx}_input_{prompt_type}",
                    call_idx=call_idx,
                    extra_info={"img_idx": img_idx, "points": points},
                )

        try:
            annotated_image = renderer.prompt(
                prompt_type=prompt_type,
                points=points,
                img_idx=img_idx,
                image=image_override,
                inplace=False,
            )
        except (IndexError, ValueError, TypeError) as exc:
            return ToolResponse(text=f"Error: {exc}"), -0.05, {"success": False}

        if inplace and image_override is None:
            # Keep buffer aligned with the latest annotation.
            renderer.update_image(img_idx, annotated_image)

        # Save output image after annotation if debug mode is enabled
        if self._debug_save_images and self._debug_save_output:
            self._save_debug_images(
                instance_id=instance_id,
                images=[annotated_image],
                stage=f"call{call_idx}_output_{prompt_type}",
                call_idx=call_idx,
                extra_info={"img_idx": img_idx, "points": points, "inplace": inplace},
            )

        try:
            point_count = len(json.loads(points)) if isinstance(points, str) else len(points)
        except Exception:
            point_count = "unknown"

        response_text = (
            f"Rendered {prompt_type} prompt on image index {img_idx} "
            f"using {point_count} points."
        )
        
        # Build debug file path for metrics
        debug_image_path = None
        if self._debug_save_images:
            short_id = instance["short_id"]
            created_at = instance["created_at"]
            debug_image_path = str(
                self._debug_save_dir / f"{created_at}_{short_id}_call{call_idx}_output_{prompt_type}_0.png"
            )
        
        return (
            ToolResponse(
                image=[annotated_image],
                text=response_text,
            ),
            0.0,
            {
                "success": True,
                "img_idx": img_idx,
                "prompt_type": prompt_type,
                "point_count": point_count,
                "debug_image_path": debug_image_path,
            },
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

    def _save_debug_images(
        self,
        instance_id: str,
        images: List[Image.Image],
        stage: str,
        call_idx: int,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Save debug images to disk.
        
        Args:
            instance_id: The instance ID of the tool.
            images: List of PIL images to save.
            stage: Stage name for file naming (e.g., "init", "call1_input", "call1_output").
            call_idx: The call index for ordering.
            extra_info: Optional extra information to log.
            
        Returns:
            List of saved file paths.
        """
        saved_paths = []
        instance = self._instances.get(instance_id)
        if instance is None:
            return saved_paths
            
        short_id = instance["short_id"]
        created_at = instance["created_at"]
        
        for img_idx, image in enumerate(images):
            # Format: {timestamp}_{short_instance_id}_{stage}_{img_idx}.png
            filename = f"{created_at}_{short_id}_{stage}_{img_idx}.png"
            filepath = self._debug_save_dir / filename
            
            try:
                image.save(filepath, "PNG")
                saved_paths.append(str(filepath))
                logger.debug(f"[ImagePromptTool] Saved debug image: {filepath}")
            except Exception as e:
                logger.warning(f"[ImagePromptTool] Failed to save debug image {filepath}: {e}")
        
        # Save metadata as JSON if extra_info is provided
        if extra_info and saved_paths:
            meta_filename = f"{created_at}_{short_id}_{stage}_meta.json"
            meta_filepath = self._debug_save_dir / meta_filename
            try:
                meta_data = {
                    "instance_id": instance_id,
                    "stage": stage,
                    "call_idx": call_idx,
                    "saved_images": saved_paths,
                    **extra_info,
                }
                with open(meta_filepath, "w") as f:
                    json.dump(meta_data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"[ImagePromptTool] Failed to save metadata {meta_filepath}: {e}")
        
        return saved_paths

    def _limit_images(self, images: Optional[Iterable[ImageLike]]) -> List[ImageLike]:
        if not images:
            return []
        if isinstance(images, (str, Path, bytes, bytearray, Image.Image, dict)):
            return [images]
        if not isinstance(images, IterableABC):
            raise TypeError(f"Unsupported images container type: {type(images)}.")
        image_list = list(images)
        if len(image_list) <= self._max_images:
            return image_list
        # Keep the most recent images to bound memory usage.
        return image_list[-self._max_images :]


__all__ = ["ImagePromptTool"]

