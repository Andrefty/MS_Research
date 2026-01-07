#!/bin/bash
#SBATCH --job-name=eval-sft
#SBATCH --output=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/logs/eval_%j.out
#SBATCH --error=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/logs/eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --gres=gpu:2

# =============================================================================
# Evaluation Job Script
# Runs both training-format and std_cls prompts on test datasets
# =============================================================================

# --- EASY CONFIGURATION ---
# Change these to evaluate different checkpoints:

# Option 1: Stock Qwen3-4B (baseline)
# MODEL_PATH="Qwen/Qwen3-4B"
# MODEL_NAME="qwen3_4b_base"

# Option 2: SFT checkpoint
MODEL_PATH="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/sft_qwen3_4b"
MODEL_NAME="sft_qwen3_4b"

# Option 3: GRPO checkpoint
# MODEL_PATH="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/grpo_qwen3_4b"
# MODEL_NAME="grpo_qwen3_4b"

# Number of GPUs - Qwen3-4B vocab size (151936) not divisible by 3, must use 2
NUM_GPUS=2

# --- Paths ---
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
EVAL_DIR="$WORK_DIR/evaluation"
LOG_DIR="$WORK_DIR/logs"
OUTPUT_DIR="$WORK_DIR/evaluation_results/${MODEL_NAME}"

SGLANG_ENV_PYTHON="$HOME/.conda/envs/sglangenv/bin/python"
EVAL_ENV_NAME="SRI_training_standard_fa_probs2"

# Pick a random port to avoid collisions if multiple jobs run on the same node
# Try to find a free port between 30000 and 40000
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
echo "Evaluation Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Num GPUs: $NUM_GPUS"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"

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
