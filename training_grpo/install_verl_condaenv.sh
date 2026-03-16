#!/bin/bash
# verl Installation Script (FSDP2 + SGLang)
# Based on: Dockerfile.stable.sglang and .github/workflows/sgl.yml
# Cluster: H100 80GB, CUDA 13.1, Python 3.12
# Date: 2026-03-11

set -e

ENV_NAME="verl_env"
VERL_DIR="$HOME/SSL_research/verl"

echo "=== verl Installation (FSDP2 + SGLang) ==="

# Step 1: Create conda environment
echo "=== Step 1: Creating conda environment ==="
source ~/miniconda3/bin/activate
conda create -n $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

# Step 2: Install SGLang (brings PyTorch 2.9.1+cu129)
echo "=== Step 2: Installing SGLang 0.5.6.post2 ==="
pip install "sglang[all]==0.5.6.post2" --no-cache-dir
pip install torch-memory-saver --no-cache-dir

# Step 3: Install FlashAttention (prebuilt wheel, no compilation)
echo "=== Step 3: Installing FlashAttention 2.8.1 ==="
cd /tmp
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
rm flash_attn-*.whl

# Step 4: Install core dependencies
echo "=== Step 4: Installing core dependencies ==="
pip install pybind11 accelerate codetiming datasets dill hydra-core liger-kernel \
    "numpy<2.0.0" pandas peft "pyarrow>=19.0.0" pylatexenc \
    ray[default] "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    transformers wandb packaging uvicorn fastapi tensorboard \
    mathruler qwen_vl_utils latex2sympy2_extended math_verify \
    hf_transfer cupy-cuda12x

# Step 5: Install verl from source
echo "=== Step 5: Installing verl ==="
cd $VERL_DIR
pip install --no-deps -e .

# Step 6: Fix cuDNN (MUST be last — flash_attn/ray revert it to 9.10)
echo "=== Step 6: Fixing cuDNN (must be last!) ==="
pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.16.0.29

# Step 7: Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang
print(f'SGLang: {sglang.__version__}')
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
echo "Usage: conda activate $ENV_NAME"
