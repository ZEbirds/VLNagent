#!/bin/bash
# =============================================================================
# COT/SFT Training Script
# 
# Usage:
#   ./train_sft.sh <data_path> <run_name> [batch_size] [llm_lr] [vision_lr]
#
# Examples:
#   ./train_sft.sh /data/cot/train/trajectory_cot_train.jsonl trajectory_sft 8 1e-5 2e-6
# =============================================================================

DATA_PATH=${1:-""}
RUN_NAME=${2:-"test"}
PER_DEVICE_BATCH_SIZE=${3:-8}
LLM_LR=${4:-1e-5}
VISION_LR=${5:-2e-6}
MM_THINK=${6:-True}
MODEL_PATH=${7:-"/path/to/base_model"}

# Single machine debug mode
if [[ -n $DEBUG && $DEBUG -eq 1 ]]; then
    WORLD_SIZE=1
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=16667
    RANK=0
fi

# Default values (if environment variables are not set)
WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-16667}
RANK=${RANK:-0}

echo "========================================"
echo "SFT Training Configuration:"
echo "  DATA_PATH: $DATA_PATH"
echo "  RUN_NAME: $RUN_NAME"
echo "  BATCH_SIZE: $PER_DEVICE_BATCH_SIZE"
echo "  LLM_LR: $LLM_LR"
echo "  VISION_LR: $VISION_LR"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "========================================"

# Environment configuration
export PYTHONPATH="$(pwd):$(pwd)/easy_vlm/training"
export WANDB_PROJECT=rynnbrain_sft

WORK_DIR=${WORK_DIR:-"./outputs"}
OUTPUT_DIR=$WORK_DIR/$WANDB_PROJECT/$RUN_NAME

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Model arguments
MODEL_ARGS=(
    --model_path $MODEL_PATH
    --gradient_checkpointing True
    --use_liger_kernel False
)

# Data arguments
DATA_ARGS=(
    --data_mixture $DATA_PATH
    --model_max_length 16384
    --mm_max_length 10240
    --fps 2
    --max_frames 2048
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE
    --gradient_accumulation_steps 1
    --num_train_epochs 1
    --remove_unused_columns False
    --use_multimodal_thinking $(($MM_THINK))
)

# Optimizer arguments
OPTIMIZER_ARGS=(
    --llm_lr $LLM_LR
    --projector_lr $LLM_LR
    --vision_encoder_lr $VISION_LR
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

# Training arguments
TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --bf16 True
    --tf32 True
    --fp16 False
    --dataloader_num_workers 8
    --decoder_load_balancing True
    --loss_reduction_scope sequence
    --average_tokens_across_devices True
)

# Logging arguments
LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --run_name $RUN_NAME
    --logging_steps 1
    --report_to wandb
    --save_strategy "steps"
    --save_steps 2000
    --save_total_limit 5
)

set -x

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    --rdzv_conf="timeout=7200,join_timeout=7200" \
    -m easy_vlm.api.train \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]} 2>&1 | tee -a $OUTPUT_DIR/$RANK.log
