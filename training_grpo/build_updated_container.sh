#!/bin/bash
# Build updated veRL container with transformers 4.57.6 and verl pre-installed
#
# This script creates a new Apptainer/Singularity image with:
# - transformers 4.57.6 (fixes tokenizer regex bug)
# - verl 0.7.0 (pre-installed, no runtime install needed)
#
# Usage: Run this script on a machine with sufficient disk space and 
#        write access to the container storage location.

set -e

# Configuration
ORIGINAL_IMAGE="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/verl_vllm012.latest.sif"
SANDBOX_DIR="/tmp/verl_sandbox_$(whoami)_$$"
FINAL_IMAGE="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/verl_vllm012_updated.sif"

echo "============================================"
echo "Building Updated veRL Container"
echo "Original: $ORIGINAL_IMAGE"
echo "Output:   $FINAL_IMAGE"
echo "============================================"

# Step 1: Create sandbox from existing image
echo ""
echo "Step 1: Creating sandbox from original image..."
if [ -d "$SANDBOX_DIR" ]; then
    echo "Removing existing sandbox..."
    rm -rf "$SANDBOX_DIR"
fi
apptainer build --sandbox "$SANDBOX_DIR" "$ORIGINAL_IMAGE"

# Step 2: Install packages inside the sandbox
# Note: --no-home avoids mounting /export which doesn't exist in container
echo ""
echo "Step 2: Installing transformers and verl..."
apptainer exec --writable --no-home "$SANDBOX_DIR" pip install --upgrade transformers==4.57.6 verl

# Verify installations
echo ""
echo "Verifying installations..."
apptainer exec "$SANDBOX_DIR" python3 -c "
import transformers
import verl
print(f'transformers version: {transformers.__version__}')
print(f'verl version: {verl.__version__}')
"

# Step 3: Convert sandbox back to SIF
echo ""
echo "Step 3: Converting sandbox to SIF image..."
if [ -f "$FINAL_IMAGE" ]; then
    echo "Removing existing output image..."
    rm -f "$FINAL_IMAGE"
fi
apptainer build "$FINAL_IMAGE" "$SANDBOX_DIR"

# Cleanup
echo ""
echo "Step 4: Cleaning up sandbox..."
rm -rf "$SANDBOX_DIR"

echo ""
echo "============================================"
echo "Build complete!"
echo "New image: $FINAL_IMAGE"
echo ""
echo "Don't forget to update VERL_IMAGE in job_train_grpo.sh:"
echo "  VERL_IMAGE=\"$FINAL_IMAGE\""
echo "============================================"
