#!/bin/bash
#SBATCH --job-name=sglang-qwen-eval
#SBATCH --output=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/inference_experiment/logs/slurm_logs/sglang-qwen-eval-%j.log
#SBATCH --error=/export/home/acs/stud/t/tudor.farcasanu/SSL_research/inference_experiment/logs/slurm_logs/sglang-qwen-eval-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# --- Configuration ---
WORK_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research"
SGLANG_ENV_PYTHON="$HOME/.conda/envs/sglangenv/bin/python"
PRIMEVUL_ENV_NAME="primevul" # Conda environment name for the evaluation script

SGLANG_MODEL_PATH="Qwen/Qwen3-32B" # Model path for SGLang server
SGLANG_PORT="30000"
SGLANG_HOST="127.0.0.1"

LOG_DIR="$WORK_DIR/inference_experiment/logs" # Centralized log directory
SLURM_LOG_DIR="$LOG_DIR/slurm_logs"
SGLANG_LOG_FILE="$LOG_DIR/sglang_server-${SLURM_JOB_ID}.log"
EVAL_LOG_FILE="$LOG_DIR/eval_script-${SLURM_JOB_ID}.log"

# --- Setup ---
mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$(dirname "$SGLANG_LOG_FILE")"
mkdir -p "$(dirname "$EVAL_LOG_FILE")"

cd "$WORK_DIR"
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM Output Log: $SLURM_LOG_DIR/sglang-qwen-eval-$SLURM_JOB_ID.log"
echo "SLURM Error Log: $SLURM_LOG_DIR/sglang-qwen-eval-$SLURM_JOB_ID.err"
echo "SGLang Server Log: $SGLANG_LOG_FILE"
echo "Evaluation Script Log: $EVAL_LOG_FILE"

SGLANG_PID=""

# Trap to ensure SGLang server is killed on script exit/interruption
cleanup() {
    echo "Cleaning up..."
    if [[ -n "$SGLANG_PID" ]] && ps -p "$SGLANG_PID" > /dev/null; then
        echo "Stopping SGLang server (PID: $SGLANG_PID)..."
        kill "$SGLANG_PID"
        wait "$SGLANG_PID" 2>/dev/null # Wait for server to stop
        echo "SGLang server stopped."
    else
        echo "SGLang server (PID: $SGLANG_PID) not found or already stopped."
    fi
    
    # Deactivation is better handled within the subshell if it was activated there.
}
trap cleanup EXIT SIGINT SIGTERM

# --- Start SGLang Server ---
echo "Starting SGLang server..."
echo "Command: $SGLANG_ENV_PYTHON -m sglang.launch_server --model-path \"$SGLANG_MODEL_PATH\" --port \"$SGLANG_PORT\" --host \"$SGLANG_HOST\" --log-level info "
nohup $SGLANG_ENV_PYTHON -m sglang.launch_server \
    --model-path "$SGLANG_MODEL_PATH" \
    --port "$SGLANG_PORT" \
    --host "$SGLANG_HOST" \
    --log-level info > "$SGLANG_LOG_FILE" 2>&1 &
SGLANG_PID=$!
echo "SGLang server started with PID: $SGLANG_PID. Log: $SGLANG_LOG_FILE"

sleep 10

# --- Wait for SGLang Server Readiness ---
TIMEOUT_SECONDS=1800 
SECONDS=0 
SERVER_READY=false # Ensure this is initialized
echo "Waiting for SGLang server to be ready (max $TIMEOUT_SECONDS seconds)..."

while [[ $SECONDS -lt $TIMEOUT_SECONDS ]]; do
    # Check for the SGLang server readiness message
    if grep -q "The server is fired up and ready to roll!" "$SGLANG_LOG_FILE"; then
        echo "SGLang server is ready!"
        SERVER_READY=true
        break
    fi
    # # Fallback check for older/different SGLang versions or if the primary message changes
    # if $SGLANG_ENV_PYTHON -m sglang.check_server_status --host "$SGLANG_HOST" --port "$SGLANG_PORT" &> /dev/null; then
    #     echo "SGLang server is ready (checked via check_server_status)!"
    #     SERVER_READY=true
    #     break
    # fi
    echo "Still waiting for SGLang server... ($SECONDS s / $TIMEOUT_SECONDS s)"
    sleep 10 # Sleep for 10 seconds before next check
done

if [[ "$SERVER_READY" != "true" ]]; then
    echo "SGLang server did not become ready within $TIMEOUT_SECONDS seconds." >&2
    echo "Last 50 lines of SGLang server log ($SGLANG_LOG_FILE):" >&2
    tail -n 50 "$SGLANG_LOG_FILE" >&2
    exit 1
fi

# --- Run Dataset Generation Script ---
echo "SGLang server is up. Proceeding with dataset generation script."
# The command_generate_dataset.sh script is now responsible for calling generate_finetuning_dataset_merged.py
# It is assumed to be in the generate_dataset directory, relative to WORK_DIR
DATASET_GEN_SCRIPT_PATH="$WORK_DIR/generate_dataset/command_generate_dataset.sh"
DATASET_GEN_CWD="$WORK_DIR/generate_dataset" # Correct CWD for the generation script

echo "Running dataset generation script from $DATASET_GEN_SCRIPT_PATH in $DATASET_GEN_CWD with environment $PRIMEVUL_ENV_NAME..."
echo "Command will be executed in a subshell to manage environment activation."

(
    echo "Subshell CWD: $(pwd)"
    cd "$DATASET_GEN_CWD"
    echo "Subshell CWD after cd: $(pwd)"

    CONDA_BIN_DIR=""
    if [[ -d "$HOME/miniconda3/bin" ]]; then
        CONDA_BIN_DIR="$HOME/miniconda3/bin"
    elif [[ -d "$HOME/anaconda3/bin" ]]; then
        CONDA_BIN_DIR="$HOME/anaconda3/bin"
    elif [[ -d "$HOME/.conda/bin" ]]; then
        CONDA_BIN_DIR="$HOME/.conda/bin"
    fi

    if [[ -n "$CONDA_BIN_DIR" && ":$PATH:" != *":$CONDA_BIN_DIR:"* ]]; then
        echo "Adding $CONDA_BIN_DIR to PATH for subshell"
        export PATH="$CONDA_BIN_DIR:$PATH"
    fi

    if command -v conda &> /dev/null; then
        echo "Attempting: eval \"\$(conda shell.bash hook)\""
        eval "$(conda shell.bash hook)"
        HOOK_STATUS=$?
        if [ $HOOK_STATUS -ne 0 ]; then
            echo "conda shell.bash hook failed with status $HOOK_STATUS. Trying to source conda.sh."
            CONDA_BASE_DIR=$(conda info --base 2>/dev/null)
            if [[ -n "$CONDA_BASE_DIR" && -f "$CONDA_BASE_DIR/etc/profile.d/conda.sh" ]]; then
                echo "Sourcing $CONDA_BASE_DIR/etc/profile.d/conda.sh"
                # shellcheck source=/dev/null
                . "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
                SOURCE_STATUS=$?
                if [ $SOURCE_STATUS -ne 0 ]; then
                    echo "Sourcing conda.sh failed with status $SOURCE_STATUS."
                    exit 126 
                fi
            else
                echo "conda shell.bash hook failed and could not find/source conda.sh."
                echo "CONDA_BASE_DIR resolved to: '$CONDA_BASE_DIR'"
                exit 127 
            fi
        fi
    else
        echo "Conda command not found in subshell PATH even after attempting to add common paths. Cannot initialize conda."
        echo "Current PATH: $PATH"
        exit 127
    fi
    
    echo "Activating Conda environment: $PRIMEVUL_ENV_NAME"
    conda activate "$PRIMEVUL_ENV_NAME"
    ACTIVATION_STATUS=$?

    if [ $ACTIVATION_STATUS -eq 0 ]; then
        echo "Conda environment '$PRIMEVUL_ENV_NAME' activated successfully."
        echo "Current Conda environment: $CONDA_DEFAULT_ENV"
        echo "Python version: $(python --version 2>&1)"
        echo "Python path: $(which python)"
        echo "Running command_generate_dataset.sh from $DATASET_GEN_SCRIPT_PATH..."
        bash "$DATASET_GEN_SCRIPT_PATH" # Execute the updated command script
        SCRIPT_STATUS=$?
        echo "command_generate_dataset.sh finished with status: $SCRIPT_STATUS"
        echo "Deactivating Conda environment: $PRIMEVUL_ENV_NAME"
        conda deactivate
        exit $SCRIPT_STATUS
    else
        echo "Subshell: Failed to activate conda environment '$PRIMEVUL_ENV_NAME'. Activation status: $ACTIVATION_STATUS"
        if ! conda env list | grep -q "$PRIMEVUL_ENV_NAME"; then
            echo "Conda environment '$PRIMEVUL_ENV_NAME' does not seem to exist. Available environments:"
            conda env list
        fi
        exit $ACTIVATION_STATUS 
    fi
) > "$EVAL_LOG_FILE" 2>&1 

GEN_STATUS=$? # Changed from EVAL_STATUS

if [ $GEN_STATUS -ne 0 ]; then
    echo "Dataset generation script execution block finished with errors (status: $GEN_STATUS)."
else
    echo "Dataset generation script execution block finished successfully."
fi
echo "Full dataset generation log is available at: $EVAL_LOG_FILE" # Log file name can be kept or changed
echo "Last 20 lines of dataset generation log:"
tail -n 20 "$EVAL_LOG_FILE"

echo "Script finished. Cleanup will be handled by trap."
exit $GEN_STATUS # Exit with the status of the generation script

