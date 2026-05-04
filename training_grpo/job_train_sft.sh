#!/bin/bash
#SBATCH --job-name=sft-qwen3-4b
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
OUTPUT_DIR="$WORK_DIR/checkpoints/sft_qwen3_4b"
TRAIN_FILE="$WORK_DIR/training_grpo/sft_dataset_train.jsonl"
VAL_FILE="$WORK_DIR/training_grpo/sft_dataset_val.jsonl"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Conda environment setup
source ~/miniconda3/bin/activate
conda activate Res_sft_and_eval_env
export DS_SKIP_CUDA_CHECK=1 #On current fep, CUDA version is 13.1, but at this time PyTorch only comes prebuilt with up to CUDA 13.0, making this needed

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# WandB logging (optional)
export WANDB_PROJECT="qwen3-vuln-sft"
export WANDB_RUN_NAME="sft-qwen3-4b-$SLURM_JOB_ID"

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
