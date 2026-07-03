# RynnBrain Reasoning

This module provides reasoning capability training for vision-language models, supporting various visual-spatial reasoning tasks. Training consists of two stages:
1. **COT/SFT Stage**: Supervised fine-tuning with chain-of-thought data to teach the model the reasoning format
2. **RL Stage**: Reinforcement learning to further optimize the model's reasoning capabilities

## Supported Task Types

| Task Type | Description | Output Format |
|-----------|-------------|---------------|
| `trajectory` | Trajectory prediction | `<trajectory><frame i> (X_1, Y_1), ..., (X_N, Y_N)</trajectory>` |
| `affordance` | Affordance point prediction | `<affordance><frame i> (X, Y)</affordance>` |
| `area` | Area prediction | `<area><frame i> (X_1, Y_1), ...</area>` |
| `counting` | Counting | `<counting>N</counting>` |
| `segment` | Object detection/segmentation | `<object><frame i> (X_min, Y_min), (X_max, Y_max)</object>` |
| `general` | General QA | Free-form text |

> Note: All coordinate values are normalized to 0-1000 range.

## Setup

### Dependencies

```bash
pip install datasets pillow
```

## Data Preparation

### Input Data Format

Raw data should be in JSON/JSONL format, with each sample containing a `conversation` field. The format supports multiple frames (images/video frames) as input:

```json
{
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/frame_001.jpg", "width": 1920, "height": 1080},
        {"type": "image", "image": "/path/to/frame_002.jpg", "width": 1920, "height": 1080},
        {"type": "image", "image": "/path/to/frame_003.jpg", "width": 1920, "height": 1080},
        {"type": "text", "text": "Describe the motion trajectory of this object"}
      ]
    },
    {
      "role": "Thinking",
      "content": [{"type": "text", "text": "Thinking process..."}]
    },
    {
      "role": "assistant", 
      "content": [
        {"frame1": [[x1, y1], [x2, y2], ...]},
        {"frame2": [[x1, y1], [x2, y2], ...]}
      ]
    }
  ]
}
```

**Content Types:**
- `image`: Image/frame input with optional `width` and `height` fields
- `text`: Text prompt

**Multi-frame Example (Video Input):**

```json
{
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/data/video_frames/001.jpg"},
        {"type": "image", "image": "/data/video_frames/002.jpg"},
        {"type": "image", "image": "/data/video_frames/003.jpg"},
        {"type": "image", "image": "/data/video_frames/004.jpg"},
        {"type": "image", "image": "/data/video_frames/005.jpg"},
        {"type": "text", "text": "Track the robot arm and predict its trajectory to grasp the red cube."}
      ]
    },
    {
      "role": "Thinking",
      "content": [{"type": "text", "text": "I can see the robot arm moving from left to right across the frames..."}]
    },
    {
      "role": "assistant",
      "content": [
        {"frame3": [[450, 320], [480, 350], [520, 400], [550, 420], [580, 450]]}
      ]
    }
  ]
}
```

**Single Image Example:**

```json
{
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/data/scene.jpg"},
        {"type": "text", "text": "Where should I click to pick up the apple?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"frame1": [[325, 480]]}
      ]
    }
  ]
}
```

---

## Stage 1: COT/SFT Training

### 1. Prepare COT Training Data

Use `preprocess_cot.py` to convert raw data to jsonl format:

```bash
python data/preprocess_cot.py \
    --local_dataset_path /path/to/raw_data.jsonl \
    --local_save_dir ~/data/cot \
    --task_type trajectory \
    --data_source my_trajectory_cot \
    --train_split 90
```

**Parameters:**
- `--local_dataset_path`: Input dataset path (JSON/JSONL)
- `--local_save_dir`: Output directory, default `~/data/cot`
- `--task_type`: Task type (`trajectory`/`affordance`/`area`/`counting`/`segment`/`general`)
- `--data_source`: Data source name (optional)
- `--train_split`: Training split percentage, default 100
- `--keep_images`: Keep images in thinking process
- `--simple`: Use simple format (no Thinking role)

**Output:**
```
~/data/cot/
├── train/
│   └── trajectory_cot_train.jsonl
└── test/
    └── trajectory_cot_test.jsonl
```

### 2. Format Data to Standard Format

Use `format_data.py` to convert the preprocessed data to a unified training format:

```bash
python data/format_data.py \
    --ann_path ~/data/cot/train/trajectory_cot_train.jsonl \
    --save_path ~/data/cot/train/trajectory_cot_train_formatted.jsonl \
    --tokenizer_path Qwen/Qwen2.5-VL-7B-Instruct \
    --data_root /path/to/media/root \
    -y
```

**Parameters:**
- `--ann_path`: Input annotation file(s), supports `json`, `jsonl`, and `parquet`
- `--save_path`: Output path for formatted data (jsonl)
- `--tokenizer_path`: Tokenizer path for text sequence length calculation
- `--data_root`: Root directory to resolve relative media paths
- `--nproc`: Number of parallel processes (default: CPU count)
- `--check`: Only validate data without formatting
- `--debug`: Debug mode (single process, no saving)
- `-y/--yes`: Overwrite existing files without prompt

**What it does:**
- Converts LLaVA-style conversations to ChatML format
- Extracts image/video metadata (width, height, duration)
- Normalizes content structure across datasets
- Calculates text sequence length for each sample

**Output format:**
```json
{
  "data_source": "trajectory_cot",
  "id": "0",
  "text_sequence_length": 256,
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "image", "text": "", "image": "/path/to/img.jpg", "video": "", "width": 1920, "height": 1080, "duration": 0.0},
        {"type": "text", "text": "Describe the trajectory.", "image": "", "video": "", "width": 0, "height": 0, "duration": 0.0}
      ]
    },
    {
      "role": "assistant",
      "content": [...]
    }
  ]
}
```

### 3. Launch SFT Training

For the Launch of SFT training please refer to the RynnScale project.
```

**Or use the full command:**

```bash
torchrun --nnodes 1 \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port 16667 \
    -m easy_vlm.api.train \
    --model_path /path/to/base_model \
    --data_mixture /path/to/cot_train.jsonl \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --llm_lr 1e-5 \
    --vision_encoder_lr 2e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir ./outputs/sft \
    --save_steps 2000
```

**Key SFT Parameters:**

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--llm_lr` | LLM learning rate | 2e-6 |
| `--vision_encoder_lr` | Vision encoder learning rate | 2e-6 |
| `--model_max_length` | Maximum sequence length | 16384 |
| `--per_device_train_batch_size` | Batch size per GPU | 8 |
| `--gradient_checkpointing` | Gradient checkpointing | True |
| `--use_multimodal_thinking` | Multimodal thinking | False |

---

## Stage 2: RL Training

### 1. Prepare RL Training Data

Use `preprocess_rl.py` to convert raw data to parquet format:

```bash
python data/preprocess_rl.py \
    --local_dataset_path /path/to/raw_data.jsonl \
    --local_save_dir ~/data/rl \
    --task_type trajectory \
    --data_source my_trajectory_data \
    --train_split 90
```

**Parameters:**
- `--local_dataset_path`: Input dataset path (JSON/JSONL)
- `--local_save_dir`: Output directory, default `~/data/rl`
- `--task_type`: Task type
- `--data_source`: Data source name (optional)
- `--train_split`: Training split percentage

**Output:**
```
~/data/rl/
├── train/
│   └── trajectory_rl_train.parquet
└── test/
    └── trajectory_rl_test.parquet
```

### 2. Merge Multiple Data Sources (Optional)

If you have multiple task types or data sources, process them separately and then merge:

```bash
# Process multiple data sources
python data/preprocess_rl.py --local_dataset_path /data/trajectory.jsonl --local_save_dir ~/data/rl --task_type trajectory
python data/preprocess_rl.py --local_dataset_path /data/affordance.jsonl --local_save_dir ~/data/rl --task_type affordance
python data/preprocess_rl.py --local_dataset_path /data/counting.jsonl --local_save_dir ~/data/rl --task_type counting

# Merge all parquet files
python data/merge_parquet.py --input_dir ~/data/rl --output_dir ~/data/rl
```

**Directory structure before merge:**
```
~/data/rl/
├── train/
│   ├── trajectory_rl_train.parquet
│   ├── affordance_rl_train.parquet
│   └── counting_rl_train.parquet
└── test/
    ├── trajectory_rl_test.parquet
    ├── affordance_rl_test.parquet
    └── counting_rl_test.parquet
```

**Directory structure after merge:**
```
~/data/rl/
├── train.parquet      # Merged train data
├── test.parquet       # Merged test data
├── train/             # Original files preserved
└── test/
```

### 3. Launch RL Training

#### Single Node Training

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=/path/to/sft_model \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=10
```

> **Note**: The `model.path` for RL training should point to the model from Stage 1 SFT training.

#### Multi-Node Training

Use `scripts/train_rl_multinode.sh` for distributed multi-node training.

**Step 1: Start Head Node**

```bash
./scripts/train_rl_multinode.sh \
    --head \
    --world-size 2 \
    --gpus-per-node 8 \
    --model-path /path/to/sft_model \
    --train-data /path/to/train.parquet \
    --test-data /path/to/test.parquet \
    --project-name my_project \
    --run-name my_experiment
```

**Step 2: Start Worker Nodes**

```bash
./scripts/train_rl_multinode.sh \
    --worker \
    --master-addr <head_node_ip> \
    --gpus-per-node 8
```

**Multi-Node Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--head` | Start as head node | - |
| `--worker` | Start as worker node | - |
| `--world-size` | Total number of nodes | 2 |
| `--gpus-per-node` | GPUs per node | 8 |
| `--master-addr` | Head node IP address | 127.0.0.1 |
| `--ray-port` | Ray port | 6379 |
| `--model-path` | SFT model path | - |
| `--train-data` | Training data path | - |
| `--test-data` | Test data path | - |
| `--project-name` | Project name | rynnbrain_trajectory_rl |
| `--run-name` | Experiment name | - |

### RL Training Configuration Details

```yaml
# Data Configuration
data.train_batch_size: 128          # Training batch size
data.max_prompt_length: 16384       # Maximum input length
data.max_response_length: 2048      # Maximum output length

# Actor Configuration
actor_rollout_ref.actor.optim.lr: 2e-6              # Learning rate
actor_rollout_ref.actor.ppo_mini_batch_size: 128    # PPO mini batch
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 2  # Micro batch per GPU
actor_rollout_ref.actor.ppo_epochs: 2               # PPO training epochs
actor_rollout_ref.actor.use_kl_loss: True           # Use KL divergence loss
actor_rollout_ref.actor.kl_loss_coef: 0.02          # KL loss coefficient

# Rollout Configuration
actor_rollout_ref.rollout.name: "sglang"            # Use sglang inference engine
actor_rollout_ref.rollout.gpu_memory_utilization: 0.6  # GPU memory utilization
actor_rollout_ref.rollout.n: 5                      # Samples per prompt

# Trainer Configuration
trainer.save_freq: 50               # Save frequency (steps)
trainer.test_freq: 10               # Test frequency (steps)
trainer.total_epochs: 10            # Total training epochs
```

### 4. Convert FSDP Checkpoint to HuggingFace Format

After RL training, the model is saved in FSDP format. Convert it to HuggingFace format for inference:

```bash
python -m easy_vlm.training.verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/rl_checkpoint/global_step_xxx/actor \
    --target_dir /path/to/output_hf_model
```

**Parameters:**
- `--backend`: Checkpoint backend, use `fsdp` for FSDP checkpoints
- `--local_dir`: Path to the FSDP checkpoint (contains `actor` subdirectory)
- `--target_dir`: Output path for the converted HuggingFace model

**Example with actual paths:**
```bash
# Convert the checkpoint at step 500
python -m easy_vlm.training.verl.model_merger merge \
    --backend fsdp \
    --local_dir ./outputs/rynnbrain_rl/trajectory_rl_v1/global_step_500/actor \
    --target_dir ./outputs/rynnbrain_rl/trajectory_rl_v1_hf
```

**For large models (distributed merge):**
```bash
torchrun --nproc_per_node 1 --nnodes 8 --node_rank ${RANK} \
    -m easy_vlm.training.verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/rl_checkpoint/global_step_xxx/actor \
    --target_dir /path/to/output_hf_model
```

---

## Complete Training Pipeline Example

```bash
# ========== Stage 1: COT/SFT ==========

# 1. Prepare COT data
python data/preprocess_cot.py \
    --local_dataset_path /data/raw/trajectory.jsonl \
    --local_save_dir /data/cot \
    --task_type trajectory \
    --train_split 90

# 2. Format data to standard format
python data/format_data.py \
    --ann_path /data/cot/train/trajectory_cot_train.jsonl \
    --save_path /data/cot/train/trajectory_cot_train_formatted.jsonl \
    --tokenizer_path Qwen/Qwen2.5-VL-7B-Instruct \
    --data_root /data/media \
    -y

# 3. SFT training
./scripts/train_sft.sh \
    /data/cot/train/trajectory_cot_train_formatted.jsonl \
    trajectory_sft_v1 \
    8 1e-5 2e-6

# ========== Stage 2: RL ==========

# 4. Prepare RL data (multiple data sources)
python data/preprocess_rl.py \
    --local_dataset_path /data/raw/trajectory.jsonl \
    --local_save_dir /data/rl \
    --task_type trajectory \
    --train_split 90

python data/preprocess_rl.py \
    --local_dataset_path /data/raw/affordance.jsonl \
    --local_save_dir /data/rl \
    --task_type affordance \
    --train_split 90

# 5. Merge all parquet files
python data/merge_parquet.py \
    --input_dir /data/rl \
    --output_dir /data/rl

# 6. RL training (using SFT model as starting point)
./scripts/train_rl_multinode.sh \
    --head \
    --world-size 1 \
    --gpus-per-node 8 \
    --model-path ./outputs/rynnbrain_sft/trajectory_sft_v1 \
    --train-data /data/rl/train.parquet \
    --test-data /data/rl/test.parquet \
    --run-name trajectory_rl_v1

# 7. Convert FSDP checkpoint to HuggingFace format
python -m easy_vlm.training.verl.model_merger merge \
    --backend fsdp \
    --local_dir ./outputs/rynnbrain_rl/trajectory_rl_v1/global_step_500/actor \
    --target_dir ./outputs/rynnbrain_rl/trajectory_rl_v1_hf
```

---

## Inference

After training and checkpoint conversion, the HuggingFace format model can be used for inference:

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("/path/to/trained_model")
processor = AutoProcessor.from_pretrained("/path/to/trained_model")

# System prompts for different task types
SYSTEM_PROMPTS = {
    "trajectory": "You are an embodied agent. You are given a video to solve a trajectory prediction problem.",
    "affordance": "You are an embodied agent. You are given a video to solve an affordance prediction problem.",
    "area": "You are an embodied agent. You are given a video to solve an area prediction problem.",
    "counting": "You are an embodied agent. You are given a video to solve a counting problem.",
    "segment": "You are an embodied agent. You are given a video to solve an object detection problem.",
    "general": "You are a helpful assistant. Think step by step and provide your final answer after `####`.",
}

# Prepare input with system prompt
task_type = "trajectory"
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPTS[task_type]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Predict the motion trajectory of the robot arm.\nFirst predict the frame containing the trajectory start point, then output up to 10 key trajectory points as a list of tuples.\nOutput format: `#### <answer><trajectory><frame i> (X_1, Y_1), (X_2, Y_2), ..., (X_N, Y_N) </trajectory></answer>`. Coordinates normalized to 0-1000."}
        ]
    }
]

# Inference
inputs = processor(messages, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

---

## Notes

1. **Training Order**: COT/SFT training must be completed before RL training
2. **Model Inheritance**: RL training should use the SFT-trained model as initialization
3. **Memory Requirements**: 8B models require 8x A100 80GB or equivalent
4. **Data Paths**: For multi-node training, ensure all nodes can access the same data paths
5. **Network Configuration**: Multi-node training requires NCCL environment variables (included in scripts)
6. **WandB Logging**: WandB is used by default for logging; ensure API key is configured
