#!/bin/bash
# =============================================================================
# Multi-node RL Training Launch Script
# 
# Usage:
#   1. Manual mode (recommended): Start Ray on each node first, then run training only on head node
#   2. Auto mode: Use SLURM/Kubernetes or other cluster managers
#
# Examples:
#   # 2 nodes with 16 GPUs total
#   # Head node (192.168.1.100):
#   ./train_rl_multinode.sh --head --world-size 2 --gpus-per-node 8
#   
#   # Worker node (192.168.1.101):  
#   ./train_rl_multinode.sh --worker --master-addr 192.168.1.100
# =============================================================================

set -e

# Default parameters
MODE=""  # head or worker
WORLD_SIZE=2
GPUS_PER_NODE=8
MASTER_ADDR="127.0.0.1"
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265
MODEL_PATH=""
TRAIN_DATA_PATH=""
TEST_DATA_PATH=""
PROJECT_NAME="rynnbrain_rl"
RUN_NAME="rynnbrain_rl_8b"
OBJECT_STORE_MEMORY=$((50*1024*1024*1024))  # 50GB
WORK_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --head)
            MODE="head"
            shift
            ;;
        --worker)
            MODE="worker"
            shift
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --ray-port)
            RAY_PORT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA_PATH="$2"
            shift 2
            ;;
        --object-store-memory)
            OBJECT_STORE_MEMORY="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --head                  Start as head node"
            echo "  --worker                Start as worker node"
            echo "  --world-size N          Total number of nodes (default: 2)"
            echo "  --gpus-per-node N       GPUs per node (default: 8)"
            echo "  --master-addr ADDR      Head node IP address"
            echo "  --ray-port PORT         Ray port (default: 6379)"
            echo "  --model-path PATH       Model path"
            echo "  --train-data PATH       Training data path"
            echo "  --test-data PATH        Test data path"
            echo "  --project-name NAME     Project name (default: rynnbrain_rl)"
            echo "  --run-name NAME         Experiment name"
            echo "  --object-store-memory N Ray object store memory size (bytes)"
            echo ""
            echo "Examples:"
            echo "  # Head node:"
            echo "  $0 --head --world-size 2 --gpus-per-node 8"
            echo ""
            echo "  # Worker node:"
            echo "  $0 --worker --master-addr 192.168.1.100"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Error: Please specify --head or --worker"
    echo "Use --help to see usage"
    exit 1
fi

# =============================================================================
# Environment Configuration
# =============================================================================
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="YOUR_HF_HOME"
# Note: expandable_segments is incompatible with sglang's torch_memory_saver, disabled
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISABLE_DISTRIBUTED_DTENSOR=1
export PYTHONPATH="$(pwd):$(pwd)/easy_vlm/training"
export FORCE_QWENVL_VIDEO_READER="torchcodec"
export SGLANG_VLM_CACHE_SIZE_MB=2048

# NCCL multi-node configuration
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_NVLS_ENABLE=0  # Disable NVLink SHARP, fixes ncclNvlsBufferSetup SIGSEGV
export NCCL_TIMEOUT=1800  # 30 minute timeout
export GLOO_SOCKET_TIMEOUT_MS=1800000  # Gloo 30 minute timeout (milliseconds)

# =============================================================================
# Start Ray Node
# =============================================================================
echo "========================================"
echo "Multi-node Training Configuration:"
echo "  MODE: $MODE"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  RAY_PORT: $RAY_PORT"
echo "========================================"

# Stop existing Ray processes first
ray stop 2>/dev/null || true
sleep 2

if [ "$MODE" == "head" ]; then
    echo ">>> Starting Ray head node..."
    ray start --head \
        --port=$RAY_PORT \
        --num-gpus=$GPUS_PER_NODE \
        --object-store-memory=$OBJECT_STORE_MEMORY \
        --include-dashboard=False \
        --disable-usage-stats
    
    echo ">>> Head node started"
    echo ">>> Ray Head: $(hostname -I | awk '{print $1}'):$RAY_PORT"
    echo ""
    echo ">>> Waiting for worker nodes to join..."
    echo ">>> Worker nodes should run:"
    echo ">>>   $0 --worker --master-addr $(hostname -I | awk '{print $1}')"
    echo ""
    
    # Wait for all nodes to join
    EXPECTED_GPUS=$((WORLD_SIZE * GPUS_PER_NODE))
    while true; do
        CURRENT_GPUS=$(ray status 2>/dev/null | grep -oP '\d+(?=\.\d+ GPU)' || echo "0")
        if [ -z "$CURRENT_GPUS" ]; then
            CURRENT_GPUS=0
        fi
        echo "Current cluster GPUs: $CURRENT_GPUS / $EXPECTED_GPUS"
        if [ "$CURRENT_GPUS" -ge "$EXPECTED_GPUS" ]; then
            echo ">>> All nodes have joined the cluster!"
            break
        fi
        sleep 5
    done
    
    # Start training
    echo ">>> Starting training..."
    echo ">>> PROJECT_NAME: $PROJECT_NAME"
    echo ">>> RUN_NAME: $RUN_NAME"
    
    python3 -m verl.trainer.main_ppo \
        +ray_init.num_cpus=None \
        +ray_init.address="auto" \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA_PATH \
        data.val_files=$TEST_DATA_PATH \
        data.train_batch_size=128 \
        data.val_batch_size=128 \
        data.max_prompt_length=16384 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=False \
        data.dataloader_num_workers=8 \
        data.truncation='right' \
        data.prompt_key=conversation \
        data.image_key=images \
        data.video_key=video \
        data.return_raw_chat=True \
        data.return_multi_modal_inputs=False \
        custom_reward_function.path=$(pwd)/easy_vlm/training/verl/utils/reward_score/rynnbrain_reward.py  \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.ppo_epochs=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.02 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.multi_turn.enable=False \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
        actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name="sglang" \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        algorithm.use_kl_in_reward=True \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=2.0 \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='verl_grpo_rynnbrain_pointing' \
        trainer.experiment_name=$RUN_NAME \
        trainer.n_gpus_per_node=$GPUS_PER_NODE \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        trainer.default_local_dir=$WORK_DIR/$PROJECT_NAME/$RUN_NAME \
        trainer.total_epochs=10

elif [ "$MODE" == "worker" ]; then
    echo ">>> Starting Ray worker node, connecting to $MASTER_ADDR:$RAY_PORT ..."
    
    # Retry connection
    MAX_RETRIES=30
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if ray start --address=$MASTER_ADDR:$RAY_PORT \
            --num-gpus=$GPUS_PER_NODE \
            --object-store-memory=$OBJECT_STORE_MEMORY; then
            echo ">>> Worker node successfully joined cluster!"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo ">>> Connection failed, retrying $RETRY_COUNT/$MAX_RETRIES ..."
            sleep 5
        fi
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Error: Unable to connect to head node $MASTER_ADDR:$RAY_PORT"
        exit 1
    fi
    
    echo ""
    echo ">>> Worker node has joined cluster"
    echo ">>> Waiting for head node to start training task..."
    echo ">>> Press Ctrl+C to exit"
    
    # Keep process running
    while true; do
        if ! ray status >/dev/null 2>&1; then
            echo ">>> Ray cluster disconnected"
            exit 0
        fi
        sleep 30
    done
fi
