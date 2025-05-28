#!/bin/bash
# This script runs the dataset generation script with recommended parameters.

# This script is expected to be in WORK_DIR/generate_dataset/
# and generate_finetuning_dataset_merged.py is in the same directory.

# Model and SGLang server parameters
SGLANG_HOST_PARAM="127.0.0.1"
SGLANG_PORT_PARAM="30000"
MODEL_ID_FOR_SGLANG_PARAM="Qwen/Qwen3-32B"
MODEL_ID_FOR_TOKENIZER_PARAM="Qwen/Qwen3-32B"
# LOCAL_MODEL_PATH_FOR_TOKENIZER_PARAM="" # Set if needed, e.g. if SGLANG_MODEL_PATH in job script is a local path

# Input/Output paths
# Using paired_train_sample.jsonl for generating the finetuning dataset.
INPUT_FILE_PARAM="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/PrimeVul-v0.1-hf/paired/primevul_train_paired.jsonl"
OUTPUT_BASE_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/generated_finetuning_data"
# Use a fixed filename to enable resuming - change date suffix manually if you want a new run
OUTPUT_FILENAME="qwen3_32b_finetune_dataset_resumable.jsonl"
OUTPUT_FILE_PARAM="$OUTPUT_BASE_DIR/$OUTPUT_FILENAME"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# LLM Generation Parameters (consistent with previous setup)
TEMPERATURE_PARAM="0.6"
TOP_P_PARAM="0.95"
TOP_K_PARAM="20"
MIN_P_PARAM="0.0"
MAX_GEN_LENGTH_PARAM="32768"
SEED_PARAM="1337"
LOGPROBS_PARAM="False" 
ENABLE_THINKING_PARAM="True"

echo "Running generate_finetuning_dataset_merged.py..."
echo "Input file: $INPUT_FILE_PARAM"
echo "Output file: $OUTPUT_FILE_PARAM"
echo "Model for SGLang: $MODEL_ID_FOR_SGLANG_PARAM"
echo "Tokenizer model: $MODEL_ID_FOR_TOKENIZER_PARAM"
echo "SGLang Host: $SGLANG_HOST_PARAM, Port: $SGLANG_PORT_PARAM"
echo "LLM Params: Temp=$TEMPERATURE_PARAM, TopP=$TOP_P_PARAM, TopK=$TOP_K_PARAM, MinP=$MIN_P_PARAM, MaxGenLen=$MAX_GEN_LENGTH_PARAM, Seed=$SEED_PARAM, Logprobs=$LOGPROBS_PARAM, EnableThinking=$ENABLE_THINKING_PARAM"
echo "Note: Script will automatically resume from existing output file if present"

# Ensure python from the correct environment is used (job_sglang script should handle activation)
# The './' assumes generate_finetuning_dataset_merged.py is in the same directory as this script
python ./generate_finetuning_dataset_merged.py \
    --input_file "$INPUT_FILE_PARAM" \
    --output_file "$OUTPUT_FILE_PARAM" \
    --sglang_host "$SGLANG_HOST_PARAM" \
    --sglang_port "$SGLANG_PORT_PARAM" \
    --model_id_for_sglang "$MODEL_ID_FOR_SGLANG_PARAM" \
    --model_id_for_tokenizer "$MODEL_ID_FOR_TOKENIZER_PARAM" \
    --temperature "$TEMPERATURE_PARAM" \
    --top_p "$TOP_P_PARAM" \
    --top_k "$TOP_K_PARAM" \
    --min_p "$MIN_P_PARAM" \
    --max_gen_length "$MAX_GEN_LENGTH_PARAM" \
    --seed "$SEED_PARAM" \
    --logprobs "$LOGPROBS_PARAM" \
    --enable_thinking "$ENABLE_THINKING_PARAM"
    # --local_model_path_for_tokenizer "$LOCAL_MODEL_PATH_FOR_TOKENIZER_PARAM" # Uncomment and set if a specific local path is needed for the tokenizer

echo "Dataset generation script finished. Output at: $OUTPUT_FILE_PARAM"