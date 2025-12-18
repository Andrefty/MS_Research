#!/bin/bash
#SBATCH --job-name=grpo-qwen3-4b
#SBATCH --gres=gpu:3             # 3 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

# ============================================
# GRPO Training Job for Qwen3-4B
# Runs after SFT to refine with reward-based RL
# ============================================

set -e

# Paths
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
TRAIN_DIR="$WORK_DIR/training_grpo"
SFT_CHECKPOINT="$WORK_DIR/checkpoints/sft_qwen3_4b"  # From SFT phase
OUTPUT_DIR="$WORK_DIR/checkpoints/grpo_qwen3_4b"
TRAIN_FILE="$WORK_DIR/training_grpo/sft_dataset_train.jsonl"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Conda environment setup
# NOTE: Run setup_training_env.sh first to create the environment with GCC 11
source ~/miniconda3/bin/activate
conda activate SRI_training_standard_fa_probs2

# Set GCC for any JIT CUDA compilation (e.g., FlashInfer, custom ops)
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
echo "Using GCC: $($CC --version | head -1)"

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# WandB logging
export WANDB_PROJECT="qwen3-vuln-grpo"
export WANDB_RUN_NAME="grpo-qwen3-4b-$SLURM_JOB_ID"

# Number of GPUs
NUM_GPUS=3

echo "=========================================="
echo "Starting GRPO Training"
echo "Base Model: $SFT_CHECKPOINT"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run training with DeepSpeed
cd "$TRAIN_DIR"

# Qwen3 recommended params for thinking mode: Temp=0.6, TopP=0.95, TopK=20, MinP=0
deepspeed --num_gpus=$NUM_GPUS train_grpo.py \
    --model_name "$SFT_CHECKPOINT" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 32768 \
    --max_new_tokens 32768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --num_generations 4 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --beta 0.05 \
    --save_steps 100 \
    --logging_steps 10 \
    --deepspeed ds_config_zero3.json \
    --bf16

echo "=========================================="
echo "GRPO Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
