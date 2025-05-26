cd openai_expr;
MODEL="Qwen/Qwen3-32B" # gpt-4-0125-preview
PROMPT_STRATEGY="std_cls";
python run_prompting.py \
    --model $MODEL \
    --prompt_strategy $PROMPT_STRATEGY \
    --data_path /export/home/acs/stud/t/tudor.farcasanu/SSL_research/PrimeVul-v0.1-hf/paired/primevul_test_paired.jsonl \
    --output_folder ../output_dir \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --min_p 0 \
    --max_gen_length 32768 \
    --seed 1337 \
    --logprobs \
    # --fewshot_eg # Assuming enable_thinking=True (default in run_prompting.py for Qwen)
cd ..;

# --temperature 0.6 \ #recommended temperature for Qwen
# --max_gen_length 1024 \ #orig max length