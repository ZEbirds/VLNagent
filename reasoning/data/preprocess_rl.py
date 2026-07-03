"""
Preprocess RL dataset to parquet format.
Supports: trajectory, affordance, area, counting, segment, general
Input format: assistant.content = [{"frame1": [[x, y], ...]}, {"frame2": [[x, y], ...]}, ...]
"""

import argparse
import json
import os
import re

import datasets
from datasets import Dataset
from PIL import Image


TASK_CONFIGS = {
    "trajectory": {
        "tag": "trajectory",
        "system_prompt": (
            "You are an embodied agent. You are given a video to solve a trajectory prediction problem."
        ),
        "user_suffix": "\nFirst predict the frame containing the trajectory start point, then output up to 10 key trajectory points as a list of tuples.\nOutput format: `#### <answer><trajectory><frame i> (X_1, Y_1), (X_2, Y_2), ..., (X_N, Y_N) </trajectory></answer>`. Coordinates normalized to 0-1000.",
    },
    "affordance": {
        "tag": "affordance",
        "system_prompt": (
            "You are an embodied agent. You are given a video to solve an affordance prediction problem."
        ),
        "user_suffix": "\nFirst predict the key frame, then output a single affordance point as coordinates (x, y).\nOutput format: `#### <answer><affordance><frame i> (X, Y) </affordance></answer>`. Coordinates normalized to 0-1000.",
    },
    "area": {
        "tag": "area",
        "system_prompt": (
            "You are an embodied agent. You are given a video to solve an area prediction problem."
        ),
        "user_suffix": "\nFirst predict the key frame, then output coordinates as a series of tuples.\nOutput format: `#### <answer><area><frame i> (X_1, Y_1), ... </area></answer>`. Coordinates normalized to 0-1000.",
    },
    "counting": {
        "tag": "counting",
        "system_prompt": (
            "You are an embodied agent. You are given a video to solve a counting problem."
        ),
        "user_suffix": "\nOutput format: `#### <answer><counting>N</counting></answer>` where N is the count.",
    },
    "segment": {
        "tag": "object",
        "system_prompt": (
            "You are an embodied agent. You are given a video to solve an object detection problem."
        ),
        "user_suffix": "\nOutput format: `#### <answer><object><frame i> (X_min, Y_min), (X_max, Y_max) </object></answer>`. Coordinates normalized to 0-1000.",
    },
    "general": {
        "tag": None,
        "system_prompt": (
            "You are a helpful assistant. Think step by step and provide your final answer after `####`."
        ),
        "user_suffix": "",
    },
}

# Task alias mapping
TASK_ALIASES = {
    "segmentation": "segment",  # segmentation and segment are the same task
}


def remove_duplicate_prompts(text):
    """
    Remove duplicate prompt content from user content that overlaps with user_suffix.
    These prompt patterns come from raw data and would duplicate with the user_suffix added later.
    """
    # Define prompt patterns to remove
    patterns_to_remove = [
        # REFERRING_TYPES (segment/object)
        r"Output the bounding box in the format <object> <frame n>: \.\.\.;.*?\(x1,y1\), \(x2,y2\) </object>\.\s*n is the chosen frame index\.",
        r"First predict the key frame.*?Output format:\s*<object>.*?</object>.*?",
        
        # TRAJ_TYPES (trajectory)
        r"First predict the frame containing the trajectory start point,.*?<trajectory> <frame n>: \.\.\.;.*?\(x1, y1\), \(x2, y2\), \.\.\.\. </trajectory>.*?All coordinates must be normalized between 0 and 1000\.",
        r"Output format:\s*<trajectory>.*?</trajectory>.*?",
        
        # AFFORDANCE_TYPES (affordance)
        r"First predict the key frame,.*?output a single affordance point as coordinates.*?Output format:\s*<affordance> <frame n>: \.\.\.;.*?\(x, y\) </affordance>.*?Both x and y values must be normalized between 0 and 1000\.",
        r"Output format:\s*<affordance>.*?</affordance>.*?",
        
        # AREA_TYPES (area)
        r"First predict the key frame,.*?output coordinates as a series of tuples.*?Output format:\s*<area> <frame n>: \.\.\.;.*?\(x1, y1\), \(x2, y2\), \.\.\.\. </area>.*?All coordinates must be normalized between 0 and 1000\.",
        r"Output format:\s*<area>.*?</area>.*?",
        
        # Generic format hints (fallback)
        r"Output format:\s*`####\s*<answer><\w+>.*?</\w+></answer>`.*?Coordinates normalized to 0-1000\.",
        r"\nOutput format:.*?Coordinates normalized to 0-1000\.",
    ]
    
    # Apply cleaning rules one by one
    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def get_image_size_from_dataset(dataset):
    """Get image dimensions from the first example in the dataset"""
    for example in dataset:
        conversations = example.get("conversation", [])
        for conv in conversations:
            if conv.get("role") == "user":
                content = conv.get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Check if valid dimensions already exist
                        w, h = item.get("width"), item.get("height")
                        if w is not None and h is not None and w > 1 and h > 1:
                            return w, h
                        # Read image to get dimensions
                        image_path = item.get("image", "")
                        if image_path and os.path.exists(image_path):
                            try:
                                with Image.open(image_path) as img:
                                    return img.width, img.height
                            except Exception:
                                pass
        break  # Only check the first example
    return 1, 1  # Default value


def clean_user_content(content, img_width, img_height):
    """Clean user content list and normalize field structure"""
    if not isinstance(content, list):
        return content
    
    cleaned = []
    for item in content:
        if not isinstance(item, dict):
            cleaned.append({"type": "text", "text": str(item) if item else ""})
        elif item.get("type") == "text":
            cleaned.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "image":
            cleaned.append({
                "type": "image",
                "image": item.get("image", ""),
                "width": img_width,
                "height": img_height,
            })
        else:
            cleaned.append({"type": "text", "text": str(item.get("text", ""))})
    
    return cleaned


def parse_ground_truth(content, task_type):
    """
    Parse assistant content to ground_truth format.
    Input: [{"frame1": [[x, y], ...]}, {"frame2": [[x, y], ...]}, ...] or string/number
    Output: ground_truth (list, string, or number)
    """
    # general type: return text directly
    if task_type == "general":
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "".join(texts)
        return ""
    
    # counting type: extract number
    if task_type == "counting":
        # content format may be:
        # - int/float: direct number
        # - str: "4" or text containing a number
        # - dict: {"count": N} or {"type": "text", "text": "4"}
        # - list: [{"type": "text", "text": "4"}] or [N] or [{"count": N}]
        
        if isinstance(content, (int, float)):
            return int(content)
        
        if isinstance(content, str):
            # Try to parse number directly
            try:
                return int(content.strip())
            except ValueError:
                pass
            # Try to extract number from text
            match = re.search(r'\b(\d+)\b', content)
            if match:
                return int(match.group(1))
            return 0
        
        if isinstance(content, dict):
            # {"type": "text", "text": "4"} format
            if content.get("type") == "text" and "text" in content:
                text_val = content["text"]
                try:
                    return int(str(text_val).strip())
                except ValueError:
                    match = re.search(r'\b(\d+)\b', str(text_val))
                    if match:
                        return int(match.group(1))
            # {"count": N} format
            if "count" in content:
                return int(content["count"])
            # Other keys containing numbers
            for key, value in content.items():
                if isinstance(value, (int, float)):
                    return int(value)
        
        if isinstance(content, list):
            # [{"type": "text", "text": "4"}] format (most common)
            for item in content:
                if isinstance(item, (int, float)):
                    return int(item)
                if isinstance(item, str):
                    try:
                        return int(item.strip())
                    except ValueError:
                        pass
                if isinstance(item, dict):
                    # {"type": "text", "text": "4"} format
                    if item.get("type") == "text" and "text" in item:
                        text_val = item["text"]
                        try:
                            return int(str(text_val).strip())
                        except ValueError:
                            match = re.search(r'\b(\d+)\b', str(text_val))
                            if match:
                                return int(match.group(1))
                    # {"count": N} format
                    if "count" in item:
                        return int(item["count"])
                    # Other keys containing numeric values
                    for key, value in item.items():
                        if isinstance(value, (int, float)):
                            return int(value)
        
        return 0  # Default return 0
    
    if not isinstance(content, list):
        return []
    
    # Check if it's frame-coords format directly
    ground_truth = []
    for item in content:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str) and key.startswith("frame") and isinstance(value, list):
                    coords = [[float(c[0]), float(c[1])] for c in value if len(c) >= 2]
                    if coords:
                        ground_truth.append({key: coords})
    
    return ground_truth


def process_example(example, task_type, data_source, img_width, img_height):
    """Process a single example"""
    config = TASK_CONFIGS[task_type]
    conversations = example.get("conversation", [])
    
    user_content = None
    assistant_content = []
    
    for conv in conversations:
        role = conv.get("role", "")
        content = conv.get("content", [])
        
        if role == "user":
            user_content = list(content)
        elif role == "assistant":
            assistant_content = content
    
    if not user_content:
        user_content = [{"type": "text", "text": ""}]
    
    # Clean user content and remove extra fields
    user_content = clean_user_content(user_content, img_width, img_height)
    
    # Remove duplicate prompts
    for i in range(len(user_content)):
        if isinstance(user_content[i], dict) and user_content[i].get("type") == "text":
            user_content[i] = dict(user_content[i])
            user_content[i]["text"] = remove_duplicate_prompts(user_content[i]["text"])
    
    # Add format hint
    for i in range(len(user_content) - 1, -1, -1):
        if isinstance(user_content[i], dict) and user_content[i].get("type") == "text":
            user_content[i] = dict(user_content[i])
            if len(user_content[i]["text"]) > 0 and user_content[i]["text"][-1] != ".":
                user_content[i]["text"] += "."
            user_content[i]["text"] += config["user_suffix"]
            break
    
    # Parse ground_truth
    ground_truth = parse_ground_truth(assistant_content, task_type)
    
    return {
        "data_source": data_source,
        "conversation": [
            # {"role": "system", "content": [{"type": "text", "text": config["system_prompt"]}]},
            {"role": "user", "content": user_content},
        ],
        "reward_model": {
            "style": "rule", 
            "ground_truth": json.dumps(ground_truth)
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RL datasets.")
    parser.add_argument("--local_dataset_path", required=True, help="Path to input dataset (JSON/JSONL)")
    parser.add_argument("--local_save_dir", default="~/data/rl", help="Output directory")
    parser.add_argument("--task_type", required=True, help="Task type (trajectory/affordance/area/counting/segment/segmentation/general)")
    parser.add_argument("--data_source", default=None, help="Data source name")
    parser.add_argument("--train_split", default="100", help="Training split percentage")

    args = parser.parse_args()
    
    # Handle task aliases
    task_type = TASK_ALIASES.get(args.task_type, args.task_type)
    
    # Validate task type
    if task_type not in TASK_CONFIGS:
        print(f"Error: Unknown task type '{args.task_type}'. Valid types: {list(TASK_CONFIGS.keys()) + list(TASK_ALIASES.keys())}")
        exit(1)
    
    data_source = args.data_source or f"{task_type}_rl"
    local_save_dir = os.path.expanduser(args.local_save_dir)
    train_pct = int(args.train_split)

    print(f"Task: {task_type}, Source: {data_source}")
    print(f"Input: {args.local_dataset_path}")
    print(f"Output: {local_save_dir}")

    # Load data
    has_test = train_pct < 100
    train_dataset = datasets.load_dataset("json", data_files=args.local_dataset_path, split=f"train[:{train_pct}%]")
    
    if has_test:
        test_dataset = datasets.load_dataset("json", data_files=args.local_dataset_path, split=f"train[{train_pct}%:]")
        print(f"Processing {len(train_dataset)} train + {len(test_dataset)} test examples")
    else:
        print(f"Processing {len(train_dataset)} train examples")

    # Get image dimensions from dataset (read only once)
    img_width, img_height = get_image_size_from_dataset(train_dataset)
    print(f"Image size: {img_width} x {img_height}")

    # Process and save as parquet
    train_results = [process_example(ex, task_type, data_source, img_width, img_height) for ex in train_dataset]
    
    train_ds = Dataset.from_list(train_results)
    train_output = os.path.join(local_save_dir, f"train/{task_type}_rl_train.parquet")
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    train_ds.to_parquet(train_output)
    print(f"Saved: {train_output}")
    
    if has_test:
        test_results = [process_example(ex, task_type, data_source, img_width, img_height) for ex in test_dataset]
        
        test_ds = Dataset.from_list(test_results)
        test_output = os.path.join(local_save_dir, f"test/{task_type}_rl_test.parquet")
        os.makedirs(os.path.dirname(test_output), exist_ok=True)
        test_ds.to_parquet(test_output)
        print(f"Saved: {test_output}")
