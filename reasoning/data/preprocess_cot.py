
"""
Preprocess the rynnbrain COT dataset to jsonl format.
Supports multiple task types: trajectory, affordance, area, counting, segment, general
"""

import argparse
import json
import os
import re

import datasets

# from rynnbrain.src.train.verl.utils.hdfs_io import copy, makedirs


# ============== Task Type Definitions ==============

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
        "tag": "object",  # Keep object tag
        "system_prompt": (
            "You are an embodied agent. You are given a video to solve an object detection problem. "
        ),
        "user_suffix": "\nOutput format: `#### <answer><object><frame i> (X_min, Y_min), (X_max, Y_max) </object></answer>`. Coordinates normalized to 0-1000.",
    },
    "general": {
        "tag": None,  # No special tag
        "system_prompt": (
            "You are a helpful assistant. Think step by step and provide your final answer after `####`."
        ),
        "user_suffix": "",  # No extra suffix
    },
}

# Task alias mapping
TASK_ALIASES = {
    "segmentation": "segment",  # segmentation and segment are the same task
}


# ============== Helper Functions ==============

def get_text_from_content(content):
    """Extract all text from content (handles mixed image-text format)"""
    if isinstance(content, str):
        return content
    
    texts = []
    for seg in content:
        if isinstance(seg, dict) and seg.get("type") == "text":
            texts.append(seg.get("text", ""))
        elif isinstance(seg, str):
            texts.append(seg)
    return "".join(texts)


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


def extract_frame_and_coords(text, tag):
    """
    Extract frame and coordinates from text.
    
    Supported formats:
    - <tag><frame N> (x, y), (x, y) </tag>
    - <tag> <frame N>: [image]; (x, y) </tag>  (after merging image-text)
    """
    # print(text)
    # Extract frame number
    frame_match = re.search(r"<frame\s+(\d+)>", text, re.IGNORECASE)
    if not frame_match:
        return None, None
    frame_num = frame_match.group(1)
    
    # Extract all coordinate pairs (x, y)
    coord_pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    coords = re.findall(coord_pattern, text)
    
    if not coords:
        return None, None
    
    return frame_num, coords


def extract_solution(content, task_type):
    """
    Extract solution from assistant content.
    """
    tag = TASK_CONFIGS[task_type]["tag"]
    text = get_text_from_content(content)
    
    # general type: return original text directly
    if task_type == "general":
        return text.strip()
    
    # counting type
    if task_type == "counting":
        match = re.search(r"<counting>\s*(\d+)\s*</counting>", text, re.IGNORECASE)
        if match:
            return f"<counting>{match.group(1)}</counting>"
        match = re.search(r"^\s*(\d+)\s*$", text.strip())
        if match:
            return f"<counting>{match.group(1)}</counting>"
        return "<counting>0</counting>"
    
    # Other types: extract frame and coordinates
    frame_num, coords = extract_frame_and_coords(text, tag)
    
    if frame_num is None or not coords:
        print(f"Warning: Could not extract {tag} from: {text[:100]}...")
        return f"<{tag}></{tag}>"
    
    coords_str = ", ".join([f"({x}, {y})" for x, y in coords])
    return f"<{tag}><frame {frame_num}>, {coords_str}</{tag}>"


def extract_thinking_text(content):
    """Extract plain text from Thinking content"""
    if not content:
        return ""
    
    parts = []
    for seg in content:
        if seg.get("type") == "text":
            parts.append(seg["text"])
    
    return "".join(parts).replace(":;", " ")


# ============== Main Processing Functions ==============

def make_map_fn(split, task_type, data_source):
    """Create data processing function"""
    config = TASK_CONFIGS[task_type]
    
    def process_fn(example, idx):
        conversations = example.get("conversation", [])
        
        user_content = None
        thinking_content = []
        assistant_content = []
        
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", [])
            
            if role == "user":
                user_content = list(content)  # Make a copy
            elif role == "Thinking":
                thinking_content = content
            elif role == "assistant":
                assistant_content = content
        
        # Process user content
        if not user_content:
            user_content = [{"type": "text", "text": ""}]
        
        # Remove duplicate prompts
        for i in range(len(user_content)):
            if user_content[i].get("type") == "text":
                user_content[i] = dict(user_content[i])  # Copy
                user_content[i]["text"] = remove_duplicate_prompts(user_content[i]["text"])
        
        # Add user_suffix uniformly
        for i in range(len(user_content) - 1, -1, -1):
            if user_content[i].get("type") == "text":
                user_content[i] = dict(user_content[i])  # Copy
                if len(user_content[i]["text"]) > 0 and user_content[i]["text"][-1] != ".":
                    user_content[i]["text"] += "."
                user_content[i]["text"] += config["user_suffix"]
                break
        
        # Extract thinking and answer
        think_text = extract_thinking_text(thinking_content)
        solution = extract_solution(assistant_content, task_type)
        
        return {
                "data_source": data_source,
                "conversation": [
                    # {
                    #     "role": "system",
                    #     "content": [{"type": "text", "text": config["system_prompt"]}],
                    # },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"<think>{think_text}</think>#### <answer>{solution}</answer>"}]
                }
            ],
        }
    
    return process_fn


def make_map_fn_with_images(split, task_type, data_source):
    """Create data processing function that preserves images"""
    config = TASK_CONFIGS[task_type]
    
    def process_fn(example, idx):
        conversations = example.get("conversation", [])
        
        user_content = None
        thinking_content = []
        assistant_content = []
        
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", [])
            
            if role == "user":
                user_content = list(content)
            elif role == "Thinking":
                thinking_content = content
            elif role == "assistant":
                assistant_content = content
        
        if not user_content:
            user_content = [{"type": "text", "text": ""}]
        
        # Remove duplicate prompts
        for i in range(len(user_content)):
            if user_content[i].get("type") == "text":
                user_content[i] = dict(user_content[i])
                user_content[i]["text"] = remove_duplicate_prompts(user_content[i]["text"])
        
        # Add user_suffix uniformly
        for i in range(len(user_content) - 1, -1, -1):
            if user_content[i].get("type") == "text":
                user_content[i] = dict(user_content[i])
                if user_content[i]["text"][-1] != ".":
                    user_content[i]["text"] += "."
                user_content[i]["text"] += config["user_suffix"]
                break
        
        # Build assistant response with images
        response_content = [{"type": "text", "text": "<think>"}]
        for seg in thinking_content:
            if seg.get("type") == "text":
                response_content.append({"type": "text", "text": seg["text"].replace(":;", " ")})
            elif seg.get("type") == "image":
                response_content.append(seg)
        response_content.append({"type": "text", "text": "</think>"})
        
        # Extract solution
        solution = extract_solution(assistant_content, task_type)
        response_content.append({"type": "text", "text": f"#### <answer>{solution}</answer>"})
        
        return {
            "data_source": data_source,
            "conversation": [
                # {"role": "system", "content": [{"type": "text", "text": config["system_prompt"]}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_content}
            ],
        }
    
    return process_fn


def make_map_fn_simple(split, task_type, data_source):
    """Data processing function for simple format (no Thinking)"""
    config = TASK_CONFIGS[task_type]
    
    def process_fn(example, idx):
        conversations = example.get("conversation", example.get("conversations", []))
        
        user_content = None
        assistant_content = []
        
        for conv in conversations:
            role = conv.get("role", conv.get("from", ""))
            content = conv.get("content", conv.get("value", ""))
            
            if role in ["user", "human"]:
                if isinstance(content, str):
                    user_content = [{"type": "text", "text": content}]
                else:
                    user_content = list(content)
            elif role in ["assistant", "gpt"]:
                if isinstance(content, str):
                    assistant_content = [{"type": "text", "text": content}]
                else:
                    assistant_content = content
        
        if not user_content:
            user_content = [{"type": "text", "text": ""}]
        
        # Remove duplicate prompts
        for i in range(len(user_content)):
            if user_content[i].get("type") == "text":
                user_content[i] = dict(user_content[i])
                if user_content[i]["text"][-1] != ".":
                    user_content[i]["text"] += "."
                user_content[i]["text"] = remove_duplicate_prompts(user_content[i]["text"])
        
        # Add user_suffix uniformly
        for i in range(len(user_content) - 1, -1, -1):
            if user_content[i].get("type") == "text":
                user_content[i] = dict(user_content[i])
                user_content[i]["text"] += config["user_suffix"]
                break
        
        # Extract solution
        solution = extract_solution(assistant_content, task_type)
        
        return {
            "data_source": data_source,
            "conversation": [
                # {"role": "system", "content": [{"type": "text", "text": config["system_prompt"]}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": f"#### <answer>{solution}</answer>"}]}
            ],
        }
    
    return process_fn


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess COT datasets.")
    parser.add_argument("--local_dataset_path", required=True, help="Path to input dataset (JSON/JSONL)")
    parser.add_argument("--local_save_dir", default="~/data/cot", help="Output directory")
    parser.add_argument("--task_type", required=True, help="Task type (trajectory/affordance/area/counting/segment/segmentation/general)")
    parser.add_argument("--data_source", default=None, help="Data source name")
    parser.add_argument("--train_split", default="100", help="Training split percentage")
    parser.add_argument("--keep_images", action="store_true", help="Keep images in thinking")
    parser.add_argument("--simple", action="store_true", help="Simple format (no Thinking)")

    args = parser.parse_args()
    
    # Handle task aliases
    task_type = TASK_ALIASES.get(args.task_type, args.task_type)
    
    # Validate task type
    if task_type not in TASK_CONFIGS:
        print(f"Error: Unknown task type '{args.task_type}'. Valid types: {list(TASK_CONFIGS.keys()) + list(TASK_ALIASES.keys())}")
        exit(1)
    
    data_source = args.data_source or f"{task_type}_cot"
    local_save_dir = os.path.expanduser(args.local_save_dir)
    train_pct = int(args.train_split)

    print(f"Task: {task_type}, Source: {data_source}")
    print(f"Input: {args.local_dataset_path}")
    print(f"Output: {local_save_dir}")

    # Select processing function
    if args.simple:
        map_fn = make_map_fn_simple
    elif args.keep_images:
        map_fn = make_map_fn_with_images
    else:
        map_fn = make_map_fn

    # Load data
    has_test = train_pct < 100
    
    train_dataset = datasets.load_dataset("json", data_files=args.local_dataset_path, split=f"train[:{train_pct}%]")
    if has_test:
        test_dataset = datasets.load_dataset("json", data_files=args.local_dataset_path, split=f"train[{train_pct}%:]")
        print(f"Processing {len(train_dataset)} train + {len(test_dataset)} test examples")
    else:
        print(f"Processing {len(train_dataset)} train examples")

    # Process and save (manually write jsonl to avoid schema issues)
    process_fn = map_fn("train", task_type, data_source)
    
    os.makedirs(os.path.join(local_save_dir, "train"), exist_ok=True)
    train_output = os.path.join(local_save_dir, f"train/{task_type}_cot_train.jsonl")
    with open(train_output, "w", encoding="utf-8") as f:
        for idx, example in enumerate(train_dataset):
            result = process_fn(example, idx)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved: {train_output}")
    
    if has_test:
        os.makedirs(os.path.join(local_save_dir, "test"), exist_ok=True)
        test_output = os.path.join(local_save_dir, f"test/{task_type}_cot_test.jsonl")
        with open(test_output, "w", encoding="utf-8") as f:
            for idx, example in enumerate(test_dataset):
                result = process_fn(example, idx)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Saved: {test_output}")