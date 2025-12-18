#!/bin/bash
#SBATCH --job-name=sft-qwen3-8b
#SBATCH --gres=gpu:3             # 3 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# ============================================
# SFT Training Job for Qwen3-4B
# Uses DeepSpeed ZeRO-3 for distributed training
# ============================================

set -e

# Paths
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
TRAIN_DIR="$WORK_DIR/training_grpo"
OUTPUT_DIR="$WORK_DIR/checkpoints/sft_qwen3_8b"
TRAIN_FILE="$WORK_DIR/training_grpo/sft_dataset_train.jsonl"
VAL_FILE="$WORK_DIR/training_grpo/sft_dataset_val.jsonl"

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

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WandB logging (optional)
export WANDB_PROJECT="qwen3-vuln-sft"
export WANDB_RUN_NAME="sft-qwen3-8b-$SLURM_JOB_ID"

# Number of GPUs
NUM_GPUS=3

echo "=========================================="
echo "Starting SFT Training"
echo "Model: Qwen/Qwen3-4B"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run training with DeepSpeed
cd "$TRAIN_DIR"

deepspeed --num_gpus=$NUM_GPUS train_sft.py \
    --model_name "Qwen/Qwen3-4B" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 32768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --save_steps 500 \
    --logging_steps 10 \
    --gradient_checkpointing \
    --deepspeed ds_config_zero3.json \
    --bf16

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
