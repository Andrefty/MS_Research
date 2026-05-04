#!/bin/bash
#SBATCH --job-name=setup_Res_sft_and_eval_env
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/setup_Res_sft_and_eval_env_%j.out
#SBATCH --error=logs/setup_Res_sft_and_eval_env_%j.err
# Setup script to create the SFT Training & Evaluation environment
# Runs on modern cluster with native GCC 14.3.1 and CUDA 13.1

set -e

ENV_NAME="Res_sft_and_eval_env"

echo "=============================================="
echo "Creating environment: $ENV_NAME"
echo "=============================================="

# Use miniconda3
source ~/miniconda3/bin/activate

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Removing..."
    conda deactivate 2>/dev/null || true
    conda env remove -n $ENV_NAME -y
fi

# Create new environment with Python 3.11
echo "Creating conda environment with Python 3.11..."
conda create -n $ENV_NAME python=3.11 -y

# Activate the environment
conda activate $ENV_NAME

# Install PyTorch
# The default pip install now uses the CUDA 13.x wheel, which aligns well with the cluster.
# Note: PyTorch wheels are statically linked with their own CUDA runtime to ensure they run anywhere. 
# However, when compiling extensions like DeepSpeed or FlashAttention, the cluster's system nvcc is used.
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install training packages
echo "Installing training packages..."
pip install "transformers>=4.51.0"
pip install accelerate
pip install deepspeed
pip install "trl>=0.15.0"  # GRPO support
pip install datasets
pip install wandb
pip install peft  # Optional, for LoRA if needed

# Install other utilities
pip install ninja
pip install packaging
pip install openai
pip install scikit-learn

# Install FlashAttention
echo "Installing FlashAttention (this may take a few minutes)..."
export MAX_JOBS=16
# We use the cluster's native gcc (14.3.1) and nvcc (13.1) here
pip install flash-attn --no-build-isolation

echo ""
echo "=============================================="
echo "Environment '$ENV_NAME' created successfully!"
echo "=============================================="
echo ""
echo "Verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
echo "  python -c 'import trl; print(f\"TRL: {trl.__version__}\")'"
echo "  python -c 'import flash_attn; print(f\"FlashAttention: {flash_attn.__version__}\")'"
