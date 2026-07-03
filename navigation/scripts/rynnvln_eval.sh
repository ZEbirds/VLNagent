#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))


echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

CHECKPOINT='/public/hz_oss/ljp/rynn_brain_ckpt/rynn_nav_8b/v8_nav_finetune_bs128/checkpoint-6199/'
OUTPUT_PATH='./results/val_unseen/RynnBrain'
# WORLD_SIZE=8



torchrun --nproc_per_node=8 --master_port $MASTER_PORT src/rynnvln_eval.py \
        --model_path $CHECKPOINT \
        --output_path $OUTPUT_PATH \



