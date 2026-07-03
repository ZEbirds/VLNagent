#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))



export CHECKPOINT="/public/hz_oss/ljp/rynn_brain_ckpt/rynn_nav_8b/v7_nav_moe_new/checkpoint-6199/"
# export OUTPUT_PATH="./results/val_unseen/v7_moe_llmlr_2e-5"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"





torchrun --nproc_per_node=8 --master_port $MASTER_PORT src/rynnvln_moe_eval.py \
        --model_path $CHECKPOINT \
        --output_path $OUTPUT_PATH \





