#!/bin/bash
# This script runs the GRPO dataset generation with the merged dataset.
# Uses generate_finetuning_dataset_grpo.py with unified schema.

# Model and SGLang server parameters
SGLANG_HOST_PARAM="127.0.0.1"
SGLANG_PORT_PARAM="30000"
MODEL_ID_FOR_SGLANG_PARAM="Qwen/Qwen3-32B"
MODEL_ID_FOR_TOKENIZER_PARAM="Qwen/Qwen3-32B"

# Input/Output paths - now using merged dataset
INPUT_FILE_PARAM="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/Research_merged_dataset/merged_train.jsonl"
OUTPUT_BASE_DIR="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/generated_finetuning_data"
OUTPUT_FILENAME="grpo_finetuning_dataset.jsonl"
OUTPUT_FILE_PARAM="$OUTPUT_BASE_DIR/$OUTPUT_FILENAME"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# LLM Generation Parameters (same as before)
TEMPERATURE_PARAM="0.6"
TOP_P_PARAM="0.95"
TOP_K_PARAM="20"
MIN_P_PARAM="0.0"
MAX_GEN_LENGTH_PARAM="32768"
SEED_PARAM="1337"
LOGPROBS_PARAM="False"
ENABLE_THINKING_PARAM="True"

# Concurrency for parallel request processing
CONCURRENCY_PARAM="8"

echo "Running generate_finetuning_dataset_grpo.py..."
echo "Input file: $INPUT_FILE_PARAM"
echo "Output file: $OUTPUT_FILE_PARAM"
echo "Model for SGLang: $MODEL_ID_FOR_SGLANG_PARAM"
echo "Tokenizer model: $MODEL_ID_FOR_TOKENIZER_PARAM"
echo "SGLang Host: $SGLANG_HOST_PARAM, Port: $SGLANG_PORT_PARAM"
echo "LLM Params: Temp=$TEMPERATURE_PARAM, TopP=$TOP_P_PARAM, TopK=$TOP_K_PARAM, MinP=$MIN_P_PARAM"
echo "Concurrency: $CONCURRENCY_PARAM parallel requests"
echo "Note: Samples with cve_desc are prioritized. Script will print when all cve_desc samples are done."
echo "Note: Script will automatically resume from existing output file if present."

python ./generate_finetuning_dataset_grpo.py \
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
    --enable_thinking "$ENABLE_THINKING_PARAM" \
    --concurrency "$CONCURRENCY_PARAM"

echo "Dataset generation script finished. Output at: $OUTPUT_FILE_PARAM"

