# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Search-R1 Contributors
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

"""
RynnBrain Reward Scoring Module

Supports multiple task types:
- trajectory: Discrete Fréchet Distance (DFD)
- affordance: Nearest distance metric
- area: Points-in-polygon metric
- segment/referring: IoU for bounding boxes (object detection)
- counting: Exact match or relative error
- general: Substring exact match
"""

import json
import math
import random
import re
import string
from typing import Callable, Sequence

import numpy as np
from scipy.spatial.distance import cdist

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# =============================================================================
# Constants & Patterns
# =============================================================================

TASK_TYPE_KEYWORDS = {
    "counting": {"counting", "count", "number"},
    "segment": {"segment", "segmentation", "mask"},
    "trajectory": {"trajectory", "traj", "navigation", "path"},
    "affordance": {"affordance", "afford", "interaction"},
    "area": {"area", "region", "zone"},
    "referring": {"referring", "object_referring", "grounding", "detection", "bbox"},
    "general": {"general", "qa", "vqa", "general_qa"},
}

# Regex patterns for extracting structured content
PATTERNS = {
    "trajectory": re.compile(r"<trajectory>(.*?)</trajectory>", re.IGNORECASE | re.DOTALL),
    "area": re.compile(r"<area>(.*?)</area>", re.IGNORECASE | re.DOTALL),
    "affordance": re.compile(r"<affordance>(.*?)</affordance>", re.IGNORECASE | re.DOTALL),
    "bbox": re.compile(r"<bbox>(.*?)</bbox>", re.IGNORECASE | re.DOTALL),
    "segment": re.compile(r"<segment>(.*?)</segment>", re.IGNORECASE | re.DOTALL),
    "object": re.compile(r"<object>(.*?)</object>", re.IGNORECASE | re.DOTALL),
    "count": re.compile(r"<count>\s*(\d+)\s*</count>", re.IGNORECASE | re.DOTALL),
    "counting": re.compile(r"<counting>\s*(\d+)\s*</counting>", re.IGNORECASE | re.DOTALL),
    "coordinate": re.compile(r"[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]"),
    "frame": re.compile(r"<frame\s+(\d+)>", re.IGNORECASE),
    "number": re.compile(r"\b(\d+)\b"),
}


# =============================================================================
# Text Utilities
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize text by removing articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, answers: str | list[str]) -> bool:
    """Check if normalized prediction exactly matches any answer."""
    if isinstance(answers, str):
        answers = [answers]
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(ans) == pred_norm for ans in answers)


def substring_match(prediction: str, answers: str | list[str]) -> bool:
    """Check if any answer is a substring of the normalized prediction."""
    if isinstance(answers, str):
        answers = [answers]
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(ans) in pred_norm for ans in answers)


# =============================================================================
# Coordinate Parsing
# =============================================================================

def _extract_coordinates(text: str) -> list[list[float]]:
    """Extract (x, y) coordinate pairs from text."""
    points = []
    for x_str, y_str in PATTERNS["coordinate"].findall(text):
        try:
            points.append([float(x_str), float(y_str)])
        except ValueError:
            continue
    return points


def _normalize_scale(points: list[list[float]]) -> list[list[float]]:
    """Scale coordinates to 0-1000 range if they appear to be normalized (0-1)."""
    if not points:
        return []
    
    max_val = max(max(abs(p[0]), abs(p[1])) for p in points)
    if 0 < max_val <= 2:
        return [[p[0] * 1000, p[1] * 1000] for p in points]
    return points


def parse_points(source: str | Sequence | np.ndarray | dict | None, is_bbox: bool = False) -> list[list[float]]:
    """
    Parse coordinate points from various input formats.
    
    Args:
        source: Input data containing coordinates
        is_bbox: If True, treat 4-element lists as bounding boxes [x_min, y_min, x_max, y_max]
    """
    if source is None:
        return []
    
    if isinstance(source, str):
        # Try to extract from trajectory tags first
        matches = PATTERNS["trajectory"].findall(source)
        text = matches[-1] if matches else source
        return _normalize_scale(_extract_coordinates(text))
    
    if isinstance(source, dict):
        for key in ("ground_truth", "target", "answer"):
            if key in source:
                return parse_points(source[key], is_bbox=is_bbox)
        return []
    
    if isinstance(source, np.ndarray):
        source = source.tolist()
    
    if isinstance(source, Sequence):
        collected = []
        for entry in source:
            if isinstance(entry, (list, tuple)):
                # Handle bounding box format: [x_min, y_min, x_max, y_max]
                if is_bbox and len(entry) == 4:
                    try:
                        # Convert to two corner points
                        collected.append([float(entry[0]), float(entry[1])])
                        collected.append([float(entry[2]), float(entry[3])])
                    except (TypeError, ValueError):
                        continue
                elif len(entry) >= 2:
                    try:
                        collected.append([float(entry[0]), float(entry[1])])
                    except (TypeError, ValueError):
                        continue
            elif isinstance(entry, str):
                collected.extend(_extract_coordinates(entry))
            elif isinstance(entry, dict):
                for v in entry.values():
                    collected.extend(parse_points(v, is_bbox=is_bbox))
        return _normalize_scale(collected)
    
    return []


def parse_frame_id(source: str | None) -> int | None:
    """Extract frame ID from text (e.g., '<frame 5>')."""
    if not isinstance(source, str):
        return None
    matches = PATTERNS["frame"].findall(source)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass
    return None


def extract_frame_options(ground_truth) -> dict[int, list]:
    """Extract frame_id -> points mapping from ground truth list or JSON string."""
    options = {}
    
    # Handle JSON string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return options
    
    if not isinstance(ground_truth, list):
        return options
    
    for item in ground_truth:
        if isinstance(item, dict):
            for k, v in item.items():
                if k.startswith("frame"):
                    try:
                        fid = int(k.replace("frame", ""))
                        options[fid] = v
                    except ValueError:
                        pass
    return options


def extract_count(source: str | int | float | None) -> int | None:
    """Extract count number from various formats."""
    if source is None:
        return None
    if isinstance(source, (int, float)):
        return int(source)
    
    text = str(source)
    # Try tagged patterns first
    for pattern_name in ["count", "counting"]:
        matches = PATTERNS[pattern_name].findall(text)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass
    
    # Fallback to standalone numbers
    numbers = PATTERNS["number"].findall(text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    return None


def extract_solution(text: str, task_type: str | None = None) -> str | None:
    """Extract answer content from solution text based on task type."""
    if not text:
        return None
    
    # Task-specific pattern priority
    pattern_priority = {
        "counting": ["count", "counting"],
        "segment": ["segment", "object"],
        "area": ["area"],
        "affordance": ["affordance"],
        "referring": ["object"],
        "trajectory": ["trajectory"],
    }
    
    patterns_to_try = pattern_priority.get(task_type, [])
    patterns_to_try += ["trajectory", "segment", "object", "area", "affordance", "bbox", "count", "counting"]
    
    for pattern_name in patterns_to_try:
        if pattern_name in PATTERNS:
            matches = PATTERNS[pattern_name].findall(text)
            if matches:
                return matches[-1].strip()
    
    # Fallback: #### pattern
    answer_matches = list(re.finditer(r"####(.*)", text, re.DOTALL))
    if answer_matches:
        return answer_matches[-1].group(1).strip()
    
    return None


# =============================================================================
# Scoring Algorithms
# =============================================================================

def uniform_sampling(trajectory: np.ndarray, n_points: int) -> np.ndarray:
    """Uniformly sample points along a trajectory by arc length."""
    if len(trajectory) == 0:
        raise ValueError("Trajectory cannot be empty")
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    
    trajectory = np.array(trajectory, dtype=float)
    
    # Calculate cumulative arc length
    segment_lengths = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
    cumulative = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative[-1]
    
    if total_length == 0:
        return np.tile(trajectory[0], (n_points, 1))
    
    # Sample at uniform intervals
    sample_distances = np.linspace(0, total_length, n_points)
    sampled = np.zeros((n_points, trajectory.shape[1]))
    for i in range(trajectory.shape[1]):
        sampled[:, i] = np.interp(sample_distances, cumulative, trajectory[:, i])
    
    return sampled


def frechet_distance(seq1: list, seq2: list, n_samples: int = 15, scale: float = 5.0) -> float:
    """
    Calculate discrete Fréchet distance between two sequences.
    Returns exp(-distance * scale) as reward in (0, 1].
    
    Args:
        seq1, seq2: Point sequences (assumed to be in 0-1 normalized range)
        n_samples: Number of points to sample for comparison
        scale: Scaling factor for distance. Higher value = sharper reward curve.
               With normalized coords (0-1), typical Fréchet dist is 0-0.5.
               scale=5.0 means: dist=0.1 -> reward=0.61, dist=0.2 -> reward=0.37
    """
    if not seq1 or not seq2:
        return 0.0
    
    seq1 = uniform_sampling(np.array(seq1), n_samples)
    seq2 = uniform_sampling(np.array(seq2), n_samples)
    
    m, n = len(seq1), len(seq2)
    dist = lambda a, b: np.linalg.norm(a - b)
    
    # Dynamic programming for Fréchet distance
    C = np.zeros((m, n))
    C[0, 0] = dist(seq1[0], seq2[0])
    
    for i in range(1, m):
        C[i, 0] = max(C[i-1, 0], dist(seq1[i], seq2[0]))
    for j in range(1, n):
        C[0, j] = max(C[0, j-1], dist(seq1[0], seq2[j]))
    
    for i in range(1, m):
        for j in range(1, n):
            C[i, j] = max(min(C[i-1, j], C[i, j-1], C[i-1, j-1]), dist(seq1[i], seq2[j]))
    
    return math.exp(-C[m-1, n-1] * scale)


def iou_bbox(box1: list, box2: list) -> float:
    """Calculate IoU between two bounding boxes [(x1,y1), (x2,y2)]."""
    if not box1 or not box2 or len(box1) < 2 or len(box2) < 2:
        return 0.0
    
    (x1_a, y1_a), (x2_a, y2_a) = box1[0], box1[1]
    (x1_b, y1_b), (x2_b, y2_b) = box2[0], box2[1]
    
    # Intersection
    inter_x1, inter_y1 = max(x1_a, x1_b), max(y1_a, y1_b)
    inter_x2, inter_y2 = min(x2_a, x2_b), min(y2_a, y2_b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def nearest_distance_reward(pred_points: list, gt_points: list, scale: float = 5.0) -> float:
    """
    Calculate reward based on bidirectional average nearest distance.
    
    Computes both precision (pred→gt) and recall (gt→pred) to avoid
    rewarding models that only predict a few "safe" points.
    
    Args:
        pred_points: Predicted points (assumed to be in 0-1 normalized range)
        gt_points: Ground truth points
        scale: Scaling factor for distance. Higher value = sharper reward curve.
    
    Returns:
        exp(-mean_bidirectional_dist * scale) as reward in (0, 1]
    """
    if not pred_points or not gt_points:
        return 0.0
    
    dist_matrix = cdist(np.array(pred_points), np.array(gt_points), "euclidean")
    
    # Precision: average distance from each pred point to nearest gt point
    pred_to_gt = np.min(dist_matrix, axis=1).mean()
    # Recall: average distance from each gt point to nearest pred point
    gt_to_pred = np.min(dist_matrix, axis=0).mean()
    # Bidirectional mean (Chamfer-like distance)
    mean_dist = (pred_to_gt + gt_to_pred) / 2
    
    return math.exp(-mean_dist * scale)


def points_in_polygon(pred_points: list, polygon_points: list) -> float:
    """Calculate ratio of pred_points inside the polygon defined by polygon_points."""
    if not pred_points or not polygon_points:
        return 0.0
    
    if SHAPELY_AVAILABLE:
        try:
            polygon = Polygon(polygon_points)
            if not polygon.is_valid:
                return 0.0
            inside = sum(1 for p in pred_points if polygon.intersects(Point(p)))
            return inside / len(pred_points)
        except Exception:
            pass
    
    # Fallback: ray casting algorithm
    def _point_in_poly(x, y, poly):
        n, inside = len(poly), False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    inside = sum(1 for p in pred_points if _point_in_poly(p[0], p[1], polygon_points))
    return inside / len(pred_points)


# =============================================================================
# Task Type Inference
# =============================================================================

def infer_task_type(data_source: str | None) -> str | None:
    """Infer task type from data_source string."""
    if not data_source:
        return None
    
    source_lower = str(data_source).lower()
    
    # Check order matters: more specific types first
    check_order = ["counting", "referring", "segment", "trajectory", "affordance", "area", "general"]
    
    # Special case: object_referring should match referring, not segment
    if "object_referring" in source_lower:
        return "referring"
    
    for task_type in check_order:
        if any(kw in source_lower for kw in TASK_TYPE_KEYWORDS[task_type]):
            return task_type
    
    return None


# =============================================================================
# Task-Specific Scorers
# =============================================================================

def _normalize_points_1k(points: list) -> list:
    """Normalize points from 0-1000 to 0-1 range."""
    return [[p[0] / 1000, p[1] / 1000] for p in points]


def score_trajectory(pred_points: list, gt_points: list) -> float:
    """Score trajectory using Fréchet distance."""
    return frechet_distance(_normalize_points_1k(pred_points), _normalize_points_1k(gt_points))


def score_affordance(pred_points: list, gt_points: list) -> float:
    """Score affordance using nearest distance metric."""
    return nearest_distance_reward(_normalize_points_1k(pred_points), _normalize_points_1k(gt_points))


def score_area(pred_points: list, gt_points: list) -> float:
    """Score area/segment using points-in-polygon metric."""
    return points_in_polygon(_normalize_points_1k(pred_points), _normalize_points_1k(gt_points))


def score_referring(pred_points: list, gt_points: list) -> float:
    """
    Score referring using continuous IoU reward.
    
    Uses a smooth mapping that gives bonus for IoU > 0.5 while maintaining
    continuity for stable RL training:
    - IoU < 0.5: reward = IoU (linear)
    - IoU >= 0.5: reward = 0.5 + (IoU - 0.5) * 1.0 = IoU (stays linear but could be adjusted)
    
    Alternative: use sigmoid-like boost for high IoU values.
    """
    if len(pred_points) < 2 or len(gt_points) < 2:
        return 0.0
    
    pred_bbox = _normalize_points_1k(pred_points[-2:])
    gt_bbox = _normalize_points_1k(gt_points[-2:])
    iou = iou_bbox(pred_bbox, gt_bbox)
    
    # Continuous reward: linear mapping with slight boost for IoU > 0.5
    # This maintains gradient signal throughout the IoU range
    if iou < 0.5:
        return iou
    else:
        # Smooth acceleration: reward grows faster after 0.5 threshold
        # Maps IoU [0.5, 1.0] -> reward [0.5, 1.0] with quadratic boost
        excess = iou - 0.5
        return 0.5 + excess + excess * excess  # quadratic bonus for high IoU


def score_counting(solution_str: str, ground_truth) -> float:
    """Score counting task using exact match or relative error."""
    pred_count = extract_count(solution_str)
    
    # Parse ground truth
    gt_count = None
    if isinstance(ground_truth, str):
        try:
            gt_count = extract_count(json.loads(ground_truth))
        except json.JSONDecodeError:
            gt_count = extract_count(ground_truth)
    else:
        gt_count = extract_count(ground_truth)
    
    if pred_count is None or gt_count is None:
        return 0.0
    if pred_count == gt_count:
        return 1.0
    
    # Partial reward based on relative error
    relative_error = abs(pred_count - gt_count) / max(gt_count, 1)
    return math.exp(-relative_error)


def score_general(solution_str: str, ground_truth) -> float:
    """Score general QA using substring match."""
    answer = extract_solution(solution_str, "general")
    if not answer:
        return 0.0
    
    # Handle dict ground truth
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        return 1.0 if substring_match(answer, ground_truth["target"]) else 0.0
    
    gt_str = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    return 1.0 if substring_match(answer, gt_str) else 0.0


# Scorer registry
POINT_SCORERS: dict[str, Callable] = {
    "trajectory": score_trajectory,
    "affordance": score_affordance,
    "area": score_area,
    "segment": score_referring,  # Same as referring (bounding box IoU)
    "referring": score_referring,
}


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    method: str = "frechet",
    format_score: float = 0.0,
    score: float = 1.0,
    extra_info=None,
) -> float:
    """
    Compute reward score for various task types.
    
    Args:
        data_source: String identifier containing task type info
        solution_str: Model's response string
        ground_truth: Expected answer (format depends on task type)
        method: Scoring method (currently unused, kept for compatibility)
        format_score: Score to return when format is invalid
        score: Maximum score for correct answer
        extra_info: Additional info (unused, kept for compatibility)
    
    Returns:
        Reward score in [0, score] range
    """
    task_type = infer_task_type(data_source)
    do_print = random.randint(1, 64) == 32
    
    # Handle counting task
    if task_type == "counting":
        reward = score_counting(solution_str, ground_truth) * score
        if do_print:
            print(f"[Counting] GT: {ground_truth}, Reward: {reward:.4f}")
        return reward
    
    # Handle general QA task
    if task_type == "general":
        reward = score_general(solution_str, ground_truth) * score
        if do_print:
            print(f"[General] GT: {ground_truth}, Reward: {reward:.4f}")
        return reward
    
    # Handle point-based tasks (trajectory, affordance, area, segment, referring)
    pred_frame_id = parse_frame_id(solution_str)
    gt_frame_options = extract_frame_options(ground_truth)
    
    # Check if task uses bounding boxes (referring/segment with object detection)
    is_bbox_task = task_type in ("referring", "segment")
    
    # Determine ground truth points
    if gt_frame_options:
        # Multi-candidate: select based on predicted frame
        if pred_frame_id is not None and pred_frame_id in gt_frame_options:
            gt_points = parse_points(gt_frame_options[pred_frame_id], is_bbox=is_bbox_task)
        else:
            gt_points = []
    else:
        gt_points = parse_points(ground_truth, is_bbox=is_bbox_task)
    
    pred_points = parse_points(solution_str, is_bbox=is_bbox_task)
    
    if do_print:
        print(f"[{task_type}] GT frames: {list(gt_frame_options.keys())}, Pred frame: {pred_frame_id}")
        print(f" GT points: {gt_points}, Pred points: {pred_points}")
    
    if not gt_points or not pred_points:
        return format_score
    
    # Get scorer for task type (default to trajectory)
    scorer = POINT_SCORERS.get(task_type, score_trajectory)
    
    try:
        reward = scorer(pred_points, gt_points) * score
    except (ValueError, TypeError) as e:
        if do_print:
            print(f"  Error: {e}")
        return format_score
    
    if do_print:
        # # Show penalty info for affordance/referring
        # if task_type == "affordance" and len(pred_points) > 1:
        #     extra = len(pred_points) - 1
        #     print(f"  Expected 1 point, got {len(pred_points)} -> penalty: -{extra * 0.1:.1f}")
        # elif task_type in ("referring", "segment") and len(pred_points) > 2:
        #     extra = len(pred_points) - 2
        #     print(f"  Expected 2 points, got {len(pred_points)} -> penalty: -{extra * 0.1:.1f}")
        print(f"  Reward: {reward:.4f}")
    
    return reward
