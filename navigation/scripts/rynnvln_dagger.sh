#!/bin/bash
umask 000
set -x

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))


#change different config to generate dagger data for different dataset

DAGGER_DATASET=R2R
DAGGER_DATA_PATH=data/datasets/r2r/train/train.json.gz
DAGGER_GT_ANNOTATIONS_PATH=data/trajectory_data/R2R_V1-3/annotations.json

# DAGGER_DATASET=RxR
# DAGGER_DATA_PATH=data/datasets/rxr/train/train_guide_en.json.gz
# DAGGER_GT_ANNOTATIONS_PATH=data/trajectory_data/RxR_new/annotations.json

# DAGGER_DATASET=EnvDrop
# DAGGER_DATA_PATH=data/datasets/envdrop/envdrop.json.gz
# DAGGER_GT_ANNOTATIONS_PATH=data/trajectory_data/EnvDrop/annotations.json

CUDA_VISIBLE_DEVICES=0
DAGGER_UPDATE_SIZE=160000
DAGGER_COMMIT_FREQ=50 # dump data every DAGGER_COMMIT_FREQ updates
DAGGER_P=0 # allow model inference
DAGGER_DATA_IT=3 # not used if DAGGER_P=0

MID_RUN_NAME="/public/hz_oss/ljp/rynn_brain_ckpt/rynn_nav_8b/v8_nav_finetune_bs128_dagger/checkpoint-2651/"
#!/bin/bash
CHECKPOINT="${MID_RUN_NAME}"
echo "CHECKPOINT: ${CHECKPOINT}"

DAGGER_OUTPUT_PATH=data/dagger_data/${DAGGER_DATASET}

mkdir -p ${DAGGER_OUTPUT_PATH}



torchrun --nproc_per_node=8 --master_port $MASTER_PORT src/rynnvln_dagger.py \
    --model_path $CHECKPOINT \
    --dagger_dataset ${DAGGER_DATASET} \
    --dagger_data_path ${DAGGER_DATA_PATH} \
    --dagger_update_size ${DAGGER_UPDATE_SIZE} \
    --dagger_commit_freq ${DAGGER_COMMIT_FREQ} \
    --dagger_p ${DAGGER_P} \
    --dagger_data_it ${DAGGER_DATA_IT} \
    --dagger_output_path ${DAGGER_OUTPUT_PATH} \
    --dagger_gt_annotations_path ${DAGGER_GT_ANNOTATIONS_PATH} \
    # --dagger_save_video \