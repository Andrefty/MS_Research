#!/bin/bash
#SBATCH --job-name=verl-grpo
#SBATCH --gres=gpu:3             # 3 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --output=logs/verl_grpo_%j.out
#SBATCH --error=logs/verl_grpo_%j.err

# ============================================
# veRL GRPO Training Job for Qwen3-4B
# Uses apptainer container with veRL docker image
# ============================================

set -e

# Paths
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
TRAIN_DIR="$WORK_DIR/training_grpo"
SFT_CHECKPOINT="$WORK_DIR/checkpoints/sft_qwen3_4b"
OUTPUT_DIR="$WORK_DIR/checkpoints/grpo_qwen3_4b_verl"
DATA_DIR="$TRAIN_DIR/verl_data"
RAY_TEMP_DIR="/tmp/ray_$(whoami)" # Avoid conflict with other users' ray sessions

# veRL docker image via apptainer
VERL_IMAGE="$WORK_DIR/verl_vllm012_updated.sif"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$RAY_TEMP_DIR"
mkdir -p logs

echo "==========================================="
echo "veRL GRPO Training"
echo "Base Model: $SFT_CHECKPOINT"
echo "Data Dir: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Image: $VERL_IMAGE"
echo "==========================================="

# Check if veRL image exists
if [ ! -f "$VERL_IMAGE" ]; then
    echo "Error: veRL image not found at $VERL_IMAGE"
    echo "Pull it first with: apptainer pull docker://verlai/verl:vllm012.latest"
    exit 1
fi

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# WandB logging
export WANDB_PROJECT="vulnerability_grpo"
export WANDB_RUN_NAME="qwen3_4b_verl_grpo_$(date +%Y%m%d_%H%M)"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Completion logging for debugging (logs all completions during training)
export GRPO_COMPLETION_LOG="$OUTPUT_DIR/verl_completions_debug.jsonl"

# Number of GPUs
NUM_GPUS=3

# Unset ROCR_VISIBLE_DEVICES to avoid conflict with CUDA_VISIBLE_DEVICES in veRL
unset ROCR_VISIBLE_DEVICES

# ============================================
# veRL GRPO Training Configuration
# ============================================
# Settings based on:
# - Qwen3-4B README: Temperature=0.6, TopP=0.95, TopK=20
# - max_response_length=32768 (recommended output length)
# - KL divergence disabled (beta=0)
# - FSDP2 with offload for memory efficiency
# ============================================

cd "$TRAIN_DIR"

# --writable-tmpfs allows temporary writes in read-only container
# Note: Using updated container with transformers 4.57.6 and verl pre-installed
apptainer exec --nv \
    --writable-tmpfs \
    --env RAY_TMPDIR=$RAY_TEMP_DIR \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $HOME:$HOME \
    $VERL_IMAGE \
    bash -c "
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=$DATA_DIR/train.parquet \
            data.val_files=$DATA_DIR/val.parquet \
            data.train_batch_size=60 \
            data.max_prompt_length=28672 \
            data.max_response_length=32768 \
            data.truncation=left \
            data.prompt_key=prompt \
            actor_rollout_ref.model.path=$SFT_CHECKPOINT \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=12 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.0 \
            actor_rollout_ref.actor.strategy=fsdp2 \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.load_format=auto \
            actor_rollout_ref.rollout.dtype=bfloat16 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
            actor_rollout_ref.rollout.temperature=0.6 \
            actor_rollout_ref.rollout.top_p=0.95 \
            actor_rollout_ref.rollout.top_k=20 \
            actor_rollout_ref.rollout.n=4 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.rollout.skip_rollout=True \
            actor_rollout_ref.rollout.skip_dump_dir=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/training_grpo/rollout_cache \
            actor_rollout_ref.rollout.disable_log_stats=False \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
            algorithm.use_kl_in_reward=False \
            custom_reward_function.path=$TRAIN_DIR/reward_function.py \
            custom_reward_function.name=compute_score \
            'trainer.logger=[\"console\",\"wandb\"]' \
            trainer.project_name=$WANDB_PROJECT \
            trainer.experiment_name=$WANDB_RUN_NAME \
            trainer.default_local_dir=$OUTPUT_DIR \
            trainer.n_gpus_per_node=$NUM_GPUS \
            trainer.nnodes=1 \
            trainer.save_freq=100 \
            trainer.test_freq=20 \
            trainer.total_epochs=1
    "

echo "==========================================="
echo "veRL GRPO Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "==========================================="
