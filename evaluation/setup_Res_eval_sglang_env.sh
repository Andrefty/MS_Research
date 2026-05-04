#!/bin/bash
# Setup script to create the SGLang Evaluation environment
# Runs on modern cluster with native GCC 14.3.1 and CUDA 13.1

set -e

ENV_NAME="Res_eval_sglang_transformersv5_env"

echo "=============================================="
echo "Creating environment: $ENV_NAME"
echo "=============================================="

source ~/miniconda3/bin/activate

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Removing..."
    conda deactivate 2>/dev/null || true
    conda env remove -n $ENV_NAME -y
fi

echo "Creating conda environment with Python 3.11..."
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME
pip install uv

# prerelease=allow is needed because sglang 0.5.10 which comes with transformers 5.3.0 depends on flash-attn-4 which is in prerelease
echo "Installing SGLang via uv..."
uv pip install sglang --prerelease=allow

echo ""
echo "=============================================="
echo "Environment '$ENV_NAME' created successfully!"
echo "=============================================="
