#!/bin/bash
#SBATCH --job-name=eval-unsloth-grpo
#SBATCH --output=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/logs/eval_%j.out
#SBATCH --error=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/logs/eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --gres=gpu:2

# =============================================================================
# Evaluation Job Script for UNSLOTH GRPO Checkpoint
# Waits for checkpoint, merges LoRA, then runs evaluation
# =============================================================================

# --- EASY CONFIGURATION ---
# Unsloth GRPO checkpoint directory (contains LoRA adapters after training)
GRPO_JOB_ID="${GRPO_JOB_ID:-5042975}"  # Default to latest, override with: GRPO_JOB_ID=XXXXX sbatch ...
CHECKPOINT_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/grpo_unsloth_${GRPO_JOB_ID}"

# Where to look for saved checkpoints (intermediate or final)
# Training saves at: checkpoint-100, checkpoint-200, etc. and final_lora/
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-true}"  # Set to false to skip waiting
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-8}"  # Max hours to wait for checkpoint

# Base model (SFT checkpoint that Unsloth GRPO was trained from)
BASE_MODEL_PATH="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/sft_qwen3_4b"

MODEL_NAME="grpo_unsloth_${GRPO_JOB_ID}"

# Number of GPUs - Qwen3-4B vocab size (151936) not divisible by 3, must use 2
NUM_GPUS=2

# --- Paths ---
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
EVAL_DIR="$WORK_DIR/evaluation"
LOG_DIR="$WORK_DIR/logs"
OUTPUT_DIR="$WORK_DIR/evaluation_results/${MODEL_NAME}"

SGLANG_ENV_PYTHON="$HOME/.conda/envs/sglangenv/bin/python"
UNSLOTH_ENV_PYTHON="$HOME/miniconda3/envs/res_unsloth_env/bin/python"
EVAL_ENV_NAME="SRI_training_standard_fa_probs2"

# Merge script
MERGE_SCRIPT="$WORK_DIR/unsloth_grpo_train/merge_lora.py"

# Pick a random port to avoid collisions
while true; do
    SGLANG_PORT=$(shuf -i 30000-40000 -n 1)
    if ! ss -lnt | grep -q ":$SGLANG_PORT "; then
        break
    fi
done
echo "Selected SGLang Port: $SGLANG_PORT"

SGLANG_HOST="127.0.0.1"

# --- Setup ---
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

SGLANG_LOG="$LOG_DIR/sglang_eval_${SLURM_JOB_ID}.log"
EVAL_LOG="$LOG_DIR/eval_run_${SLURM_JOB_ID}.log"

cd "$WORK_DIR"
echo "========================================"
echo "Unsloth GRPO Evaluation Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GRPO Job ID: $GRPO_JOB_ID"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Wait for checkpoint: $WAIT_FOR_CHECKPOINT"
echo "Max wait hours: $MAX_WAIT_HOURS"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"

# --- Function to find the best available checkpoint ---
find_lora_checkpoint() {
    local ckpt_dir="$1"
    
    # First choice: final_lora (training completed)
    if [[ -d "$ckpt_dir/final_lora" ]] && [[ -f "$ckpt_dir/final_lora/adapter_config.json" ]]; then
        echo "$ckpt_dir/final_lora"
        return 0
    fi
    
    # Second choice: latest intermediate checkpoint
    # Look for checkpoint-* directories with adapter files
    local latest=""
    for ckpt in $(ls -d "$ckpt_dir"/checkpoint-* 2>/dev/null | sort -V); do
        if [[ -f "$ckpt/adapter_config.json" ]]; then
            latest="$ckpt"
        fi
    done
    
    if [[ -n "$latest" ]]; then
        echo "$latest"
        return 0
    fi
    
    return 1
}

# --- Wait for checkpoint if requested ---
if [[ "$WAIT_FOR_CHECKPOINT" == "true" ]]; then
    echo ""
    echo "Waiting for checkpoint to be saved..."
    MAX_WAIT_SECONDS=$((MAX_WAIT_HOURS * 3600))
    SECONDS=0
    CHECKPOINT_FOUND=false
    
    while [[ $SECONDS -lt $MAX_WAIT_SECONDS ]]; do
        LORA_PATH=$(find_lora_checkpoint "$CHECKPOINT_DIR")
        if [[ -n "$LORA_PATH" ]]; then
            echo "Found checkpoint at: $LORA_PATH (after ${SECONDS}s)"
            CHECKPOINT_FOUND=true
            break
        fi
        
        # Print status every 5 minutes
        if [[ $((SECONDS % 300)) -eq 0 ]]; then
            echo "Still waiting for checkpoint... (${SECONDS}s / ${MAX_WAIT_SECONDS}s)"
            echo "Checking: $CHECKPOINT_DIR"
            ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "  Directory does not exist yet"
        fi
        
        sleep 60
    done
    
    if [[ "$CHECKPOINT_FOUND" != "true" ]]; then
        echo "ERROR: No checkpoint found within ${MAX_WAIT_HOURS} hours" >&2
        exit 1
    fi
else
    LORA_PATH=$(find_lora_checkpoint "$CHECKPOINT_DIR")
    if [[ -z "$LORA_PATH" ]]; then
        echo "ERROR: No checkpoint found at: $CHECKPOINT_DIR" >&2
        echo "Available contents:"
        ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "  Directory does not exist"
        exit 1
    fi
fi

echo ""
echo "Using LoRA checkpoint: $LORA_PATH"

# Determine checkpoint name for merged model
CKPT_NAME=$(basename "$LORA_PATH")
MERGED_MODEL_PATH="$CHECKPOINT_DIR/${CKPT_NAME}_merged"

# --- Merge LoRA with base model ---
if [[ ! -f "$MERGED_MODEL_PATH/config.json" ]]; then
    echo ""
    echo "Merging LoRA adapter with base model..."
    echo "This may take several minutes..."
    
    $UNSLOTH_ENV_PYTHON "$MERGE_SCRIPT" \
        --base_model "$BASE_MODEL_PATH" \
        --lora_adapter "$LORA_PATH" \
        --output "$MERGED_MODEL_PATH" \
        --max_seq_length 40960
    
    if [[ $? -ne 0 ]] || [[ ! -f "$MERGED_MODEL_PATH/config.json" ]]; then
        echo "ERROR: Failed to merge LoRA adapter" >&2
        exit 1
    fi
    echo "Successfully merged LoRA adapter!"
else
    echo "Merged model already exists at: $MERGED_MODEL_PATH"
fi

MODEL_PATH="$MERGED_MODEL_PATH"

SGLANG_PID=""

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [[ -n "$SGLANG_PID" ]] && ps -p "$SGLANG_PID" > /dev/null 2>&1; then
        echo "Stopping SGLang server (PID: $SGLANG_PID)..."
        kill "$SGLANG_PID"
        wait "$SGLANG_PID" 2>/dev/null
        echo "SGLang server stopped."
    fi
}
trap cleanup EXIT SIGINT SIGTERM

# --- Start SGLang Server ---
echo ""
echo "Starting SGLang server with $NUM_GPUS GPUs..."
echo "Model: $MODEL_PATH"

nohup $SGLANG_ENV_PYTHON -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$SGLANG_PORT" \
    --host "$SGLANG_HOST" \
    --tp "$NUM_GPUS" \
    --log-level info > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "SGLang server started with PID: $SGLANG_PID"
echo "Log: $SGLANG_LOG"

# Wait for server
TIMEOUT_SECONDS=1800
SECONDS=0
SERVER_READY=false

echo "Waiting for SGLang server (max ${TIMEOUT_SECONDS}s)..."

while [[ $SECONDS -lt $TIMEOUT_SECONDS ]]; do
    if grep -q "The server is fired up and ready to roll!" "$SGLANG_LOG" 2>/dev/null; then
        echo "SGLang server is ready! (${SECONDS}s)"
        SERVER_READY=true
        break
    fi
    sleep 10
done

if [[ "$SERVER_READY" != "true" ]]; then
    echo "ERROR: SGLang server did not become ready in ${TIMEOUT_SECONDS}s" >&2
    echo "Last 50 lines of log:" >&2
    tail -n 50 "$SGLANG_LOG" >&2
    exit 1
fi

# --- Run Evaluation ---
echo ""
echo "========================================"
echo "Running Evaluation Pipeline"
echo "========================================"

(
    # Initialize conda
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    
    conda activate "$EVAL_ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment: $EVAL_ENV_NAME"
        exit 1
    fi
    echo "Conda environment: $CONDA_DEFAULT_ENV"
    echo "Python: $(which python)"
    
    cd "$EVAL_DIR"
    
    # Step 1: Prepare eval dataset (if not exists)
    EVAL_DATASET="$OUTPUT_DIR/eval_dataset.jsonl"
    if [[ ! -f "$EVAL_DATASET" ]]; then
        echo ""
        echo "Step 1: Preparing evaluation dataset..."
        python prepare_eval_dataset.py \
            --primevul_test_path "$WORK_DIR/PrimeVul-v0.1-hf/paired/primevul_test_paired.jsonl" \
            --primevul_valid_path "$WORK_DIR/PrimeVul-v0.1-hf/paired/primevul_valid_paired.jsonl" \
            --sven_val_path "$WORK_DIR/sven/data/val-*.parquet" \
            --output_path "$EVAL_DATASET"
    else
        echo "Eval dataset already exists: $EVAL_DATASET"
    fi
    
    # Step 2: Run inference
    echo ""
    echo "Step 2: Running inference (both prompts)..."
    RESPONSES_FILE="$OUTPUT_DIR/eval_responses.jsonl"
    python run_eval.py \
        --eval_dataset "$EVAL_DATASET" \
        --output_file "$RESPONSES_FILE" \
        --sglang_host "$SGLANG_HOST" \
        --sglang_port "$SGLANG_PORT" \
        --prompts training std_cls \
        --concurrency 8 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 20 \
        --max_gen_length 32768
    
    # Step 3: Compute metrics
    echo ""
    echo "Step 3: Computing metrics..."
    METRICS_DIR="$OUTPUT_DIR/metrics"
    python compute_metrics.py \
        --input_file "$RESPONSES_FILE" \
        --output_dir "$METRICS_DIR"
    
    echo ""
    echo "========================================"
    echo "Evaluation Complete!"
    echo "========================================"
    echo "Results in: $OUTPUT_DIR"
    echo ""
    echo "Metrics files:"
    ls -la "$METRICS_DIR"
    
) > "$EVAL_LOG" 2>&1

EVAL_STATUS=$?

echo ""
echo "Evaluation finished with status: $EVAL_STATUS"
echo "Full log: $EVAL_LOG"
echo ""
echo "Last 30 lines of evaluation log:"
tail -n 30 "$EVAL_LOG"

exit $EVAL_STATUS
