#!/bin/bash
#SBATCH --job-name=unsloth_grpo
#SBATCH --output=logs/unsloth_grpo_%j.out
#SBATCH --error=logs/unsloth_grpo_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G

# ============================================================================
# Unsloth GRPO Training Job
# Uses Unsloth with vLLM for efficient GRPO training
# ============================================================================

set -e

# Job info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

# Paths
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
SCRIPT_DIR="${WORK_DIR}/unsloth_grpo_train"
CONDA_ENV="/export/home/acs/stud/t/tudor.farcasanu/miniconda3/envs/res_unsloth_env"

# Create logs directory
mkdir -p ${SCRIPT_DIR}/logs

# Activate conda environment
source /export/home/acs/stud/t/tudor.farcasanu/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Environment variables for Unsloth (no vLLM standby mode)
# export UNSLOTH_VLLM_STANDBY=1  # Disabled - not using fast_inference
export WANDB_PROJECT="vulnerability-grpo-unsloth-no-vllm"

# Triton/CUDA fixes - DISABLED for this test
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TRITON_INTERPRET=0

# Completion logging for debugging - IMPORTANT for monitoring GRPO training
export GRPO_COMPLETION_LOG="${SCRIPT_DIR}/logs/completions_${SLURM_JOB_ID}.jsonl"

# GPU configuration - auto-detect from SLURM or default to visible GPUs
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Number of GPUs: ${NUM_GPUS}"

# Training arguments
MODEL_PATH="${WORK_DIR}/checkpoints/sft_qwen3_4b"
DATA_PATH="${WORK_DIR}/training_grpo/sft_dataset_train.jsonl"
OUTPUT_DIR="${WORK_DIR}/checkpoints/grpo_unsloth_${SLURM_JOB_ID}"

# Debug mode (uncomment for testing)
# DEBUG_FLAG="--debug --max_steps 10"
DEBUG_FLAG=""

echo "Conda env: ${CONDA_ENV}"
echo "Python: $(which python)"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Completion log: ${GRPO_COMPLETION_LOG}"

# Verify environment
python -c "import unsloth; print(f'Unsloth version: {unsloth.__version__}')" || echo "Could not get unsloth version"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || echo "Could not get vllm version"

cd ${SCRIPT_DIR}

# For multi-GPU: use torchrun (per Unsloth docs)
# https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp
# For single GPU: run directly
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Running with torchrun on ${NUM_GPUS} GPUs (DDP auto-enabled)..."
    torchrun --nproc_per_node ${NUM_GPUS} \
        train_grpo_unsloth.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        ${DEBUG_FLAG}
else
    echo "Running on single GPU..."
    python train_grpo_unsloth_no_fast_inference.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        ${DEBUG_FLAG}
fi

echo "=================================================="
echo "Job finished: $(date)"
echo "=================================================="
