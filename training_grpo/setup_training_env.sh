#!/bin/bash
# Setup script to create the training environment for GRPO fine-tuning
# Uses conda to provide GCC 11 for building flash-attn CUDA extensions
# Run this once before submitting training jobs

set -e

ENV_NAME="SRI_training_standard_fa_probs2"

echo "=============================================="
echo "Creating training environment: $ENV_NAME"
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

# Install GCC 11 via conda (required for flash-attn build)
echo "Installing GCC 11 via conda (required for CUDA extension builds)..."
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y

# Set up compiler environment variables for CUDA builds
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
echo "Using GCC: $($CC --version | head -1)"

# Install PyTorch with CUDA 12.8 (compatible with H100)
echo "Installing PyTorch with CUDA 12.8..."
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch

# Install training packages
echo "Installing training packages..."
pip install "transformers>=4.51.0"  # Required for Qwen3
pip install accelerate
pip install deepspeed
pip install "trl>=0.15.0"  # GRPO support
pip install datasets
pip install wandb
pip install peft  # Optional, for LoRA if needed

# Install FlashAttention (with proper GCC)
echo "Installing FlashAttention (this may take a few minutes)..."
export MAX_JOBS=40
pip install flash-attn --no-build-isolation

# Install other utilities
pip install ninja
pip install packaging

echo ""
echo "=============================================="
echo "Environment '$ENV_NAME' created successfully!"
echo "=============================================="
echo ""
echo "IMPORTANT: Before running training jobs, you need to set up the"
echo "compiler environment. Add these lines to your job scripts:"
echo ""
echo '  source ~/miniconda3/bin/activate'
echo "  conda activate $ENV_NAME"
echo '  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"'
echo '  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"'
echo '  export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"'
echo ""
echo "Verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
echo "  python -c 'import trl; print(f\"TRL: {trl.__version__}\")'"
echo "  python -c 'import flash_attn; print(f\"FlashAttention: {flash_attn.__version__}\")'"
