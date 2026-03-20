#!/bin/bash
#SBATCH --job-name=verl-grpo
#SBATCH --gres=gpu:3             # 3 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G               # Max: 382G (dgxh100), 430G (dgxa100) with 32 CPUs
#SBATCH --output=logs/verl_grpo_%j.out
#SBATCH --error=logs/verl_grpo_%j.err

# ============================================
# veRL GRPO Training Job for Qwen3-4B
# Uses conda environment: verl_env
# ============================================

set -e

# Paths
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
TRAIN_DIR="$WORK_DIR/training_grpo"
SFT_CHECKPOINT="$WORK_DIR/checkpoints/sft_qwen3_4b"
OUTPUT_DIR="$WORK_DIR/checkpoints/grpo_qwen3_4b_verl"
DATA_DIR="$TRAIN_DIR/verl_data"
RAY_TEMP_DIR="/tmp/ray_$(whoami)" # Avoid conflict with other users' ray sessions

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
echo "Using conda environment: verl_env"
echo "==========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl_env

# Expose pip-installed NVIDIA CUDA libraries to subprocesses (fixes libcudart.so.12 not found for SGLang env)
NVIDIA_LIBS="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# WandB logging
export WANDB_PROJECT="vuln_qwen3_4b_verl_grpo_sglang"
export WANDB_RUN_NAME="n_16_lr_5e-6_$(date +%Y%m%d_%H%M)"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Completion logging for debugging (logs all completions during training)
export GRPO_COMPLETION_LOG="$OUTPUT_DIR/verl_completions_debug.jsonl"

# VERL output directories
export VERL_VAL_OUTPUT_DIR="$OUTPUT_DIR/val_output"
export VERL_TRAIN_ROLLOUT_DIR="$OUTPUT_DIR/train_rollout"

# Number of GPUs
NUM_GPUS=3

# Set Ray temp directory
export RAY_TMPDIR=$RAY_TEMP_DIR

# Unset ROCR_VISIBLE_DEVICES to avoid conflict with CUDA_VISIBLE_DEVICES in veRL
unset ROCR_VISIBLE_DEVICES

# ============================================
# veRL GRPO Training Configuration
# ============================================
# Settings based on:
# - Qwen3-4B README: Temperature=0.6, TopP=0.95, TopK=20
# - max_prompt_length=28681, max_response_length=12279 (fits 40960 context)
# - KL divergence disabled (beta=0)
# - FSDP2 with offload for memory efficiency
# ============================================
# actor_rollout_ref.rollout.enable_prefix_caching=False \ - this doesn't do anything because verl seems to ignore disabling things set to false:
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=true \ - this apparently works
# # Line 309-311 in utils.py
# if isinstance(v, bool):
#     if v:
#         cli_args.append(f"--{k}")
#     # bool False: SKIPPED ENTIRELY — never emits --no-enable-prefix-caching!
# maybe:
# if isinstance(v, bool):
#     if v:
#         cli_args.append(f"--{k}")
#     else:
#         cli_args.append(f"--no-{k}")

# SGLang arguments tweaks:
# Disable rollout caching, causes batch size mismatching when running validation at step 20 for some reason
    # actor_rollout_ref.rollout.skip_rollout=True \
    # actor_rollout_ref.rollout.skip_dump_dir=$TRAIN_DIR/rollout_cache \

cd "$TRAIN_DIR"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=60 \
    data.max_prompt_length=28680 \
    data.max_response_length=12279 \
    data.filter_overlong_prompts=True \
    data.truncation=middle \
    actor_rollout_ref.model.path=$SFT_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=41960 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$TRAIN_DIR/reward_function.py \
    custom_reward_function.name=compute_score \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.resume_mode=auto \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.val_before_train=True \
    trainer.log_val_generations=50 \
    trainer.validation_data_dir=$VERL_VAL_OUTPUT_DIR \
    trainer.rollout_data_dir=$VERL_TRAIN_ROLLOUT_DIR \
    trainer.test_freq=20 \
    trainer.total_epochs=1

echo "==========================================="
echo "veRL GRPO Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "==========================================="
