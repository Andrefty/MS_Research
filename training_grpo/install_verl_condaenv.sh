#!/bin/bash
# verl Installation Script for SLURM Cluster (FSDP2 + vLLM)
# Tested on: H100 80GB, CUDA 13.1, Python 3.12
# Date: 2026-02-04

set -e  # Exit on error

echo "=== verl Installation Script ==="
echo "Target: FSDP2 training + vLLM rollouts"
echo ""

# Configuration
ENV_NAME="verl_env"
VERL_DIR="$HOME/SSL_research/verl"
VLLM_BUILD_DIR="$HOME/SSL_research/vllm_build"

# Step 1: Create conda environment
echo "=== Step 1: Creating conda environment ==="
source ~/miniconda3/bin/activate
conda create -n $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

# Step 2: Install PyTorch
echo "=== Step 2: Installing PyTorch 2.9.1 (cu129) ==="
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Step 3: Clone and build vLLM from source
echo "=== Step 3: Building vLLM 0.12.0 from source ==="
cd $(dirname $VLLM_BUILD_DIR)
rm -rf $VLLM_BUILD_DIR
git clone --depth 1 -b v0.12.0 https://github.com/vllm-project/vllm.git $(basename $VLLM_BUILD_DIR)
cd $VLLM_BUILD_DIR

# Remove torch pins from requirements
find requirements -name "*.txt" -print0 | xargs -0 sed -i '/torch/d'

# Install build deps and build
pip install -r requirements/build.txt

# Set CUDA paths for compilation
export CUDA_HOME=/usr/local/cuda-13.1  # Adjust if your CUDA path differs
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=32  # Adjust based on available CPUs

echo "Building vLLM (this takes ~1-2 hours)..."
pip install -e . --no-build-isolation --no-deps

# Install vLLM CUDA requirements
echo "=== Step 4: Installing vLLM CUDA requirements ==="
pip install -r requirements/cuda.txt

# Step 5: Install FlashAttention (prebuilt wheel)
echo "=== Step 5: Installing FlashAttention 2.8.1 ==="
cd $HOME/SSL_research
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
rm flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Step 6: Install core dependencies
echo "=== Step 6: Installing core dependencies ==="
pip install accelerate codetiming datasets hydra-core liger-kernel pandas peft \
    "pyarrow>=19.0.0" pybind11 pylatexenc \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    wandb packaging uvicorn fastapi tensorboard \
    mathruler qwen_vl_utils latex2sympy2_extended math_verify

# Step 7: Install verl
echo "=== Step 7: Installing verl ==="
cd $VERL_DIR
pip install --no-deps -e .

# Step 8: Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import vllm
print(f'vLLM: {vllm.__version__}')

import flash_attn
print(f'FlashAttn: {flash_attn.__version__}')

import ray
print(f'Ray: {ray.__version__}')

import verl
print(f'verl: {verl.__version__}')

print()
print('=== Installation successful! ===')
"

echo ""
echo "To use: conda activate $ENV_NAME"
echo "Optional: rm -rf $VLLM_BUILD_DIR  # to save ~2GB"
