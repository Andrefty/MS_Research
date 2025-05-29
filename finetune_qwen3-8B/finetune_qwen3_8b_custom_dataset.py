# -*- coding: utf-8 -*-
"""
Fine-tunes a Qwen3-8B model on a custom dataset using Unsloth.
Adapted from Unsloth's Qwen3 conversational reasoning notebook.
"""

from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling # Added import
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer # For inference streaming

# Configuration
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"  # Or your specific Qwen3-8B model
MAX_SEQ_LENGTH = 32768  # Adjust based on your data and GPU memory
LOAD_IN_4BIT = True
OUTPUT_DIR = "qwen3_8b_custom_finetuned" # Directory to save the fine-tuned model
CUSTOM_DATASET_PATH = "/export/home/acs/stud/t/tudor.farcasanu/SSL_research/generated_finetuning_data/qwen3_32b_finetune_dataset_resumable.jsonl"

# Training arguments (adjust as needed)
TRAIN_ARGS = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text", # This should match the output of format_custom_data
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    gradient_accumulation_steps=8,  # Adjust based on GPU memory
    warmup_steps=10,
    # max_steps=100,  # Set for a quick test, or use num_train_epochs
    num_train_epochs=1, # Set for a full run
    learning_rate=2e-4,
    logging_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="wandb", # or "wandb" if you have it configured
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False, # Important for Qwen3 with custom chat templates / long sequences
    dataset_num_proc=8, # Number of processes for dataset preprocessing
)

# LoRA configuration
LORA_R = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.05 # Slightly increased dropout
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_BIAS = "none"


def main():
    global CUSTOM_DATASET_PATH # Ensure it's accessible if defined globally

    # Ensure CUSTOM_DATASET_PATH is defined or passed appropriately
    if CUSTOM_DATASET_PATH is None:
        raise ValueError("CUSTOM_DATASET_PATH is not set. Please specify the path to your dataset.")

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        # token = "hf_...", # if using a gated repo
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing="unsloth", # True or "unsloth" for Unsloth version
        random_state=TRAIN_ARGS.seed,
        use_rslora=False,
        loftq_config=None,
    )
    print("LoRA adapters added.")
    model.print_trainable_parameters()

    # Load custom dataset
    # Assuming format_custom_data and load_custom_dataset are defined correctly elsewhere
    # and that load_custom_dataset returns a dataset compatible with SFTTrainer
    # The dataset should have a column named "text" (or whatever is in TRAIN_ARGS.dataset_text_field)
    # containing the formatted chat strings.
    print(f"Loading custom dataset from: {CUSTOM_DATASET_PATH}")
    raw_dataset = load_dataset("json", data_files=CUSTOM_DATASET_PATH, split="train")
    
    # We need a function to format each example into the chat template
    # and ensure the output is a dataset with a "text" field.
    def format_dataset_for_sft(examples):
        # This assumes your format_custom_data function takes a list of examples
        # and returns a list of formatted strings or a dictionary with a "text" key.
        # For SFTTrainer, we typically need a "text" field with the fully formatted chat.
        
        # Based on your previous dataset generation, each item in raw_dataset
        # will have 'prompt' and 'generated_response'.
        # We need to apply the chat template to these.
        
        formatted_texts = []
        for prompt, response in zip(examples['prompt'], examples['generated_response']):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            formatted_chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False, # Important for training
                enable_thinking=True # Consistent with Qwen3
            )
            formatted_texts.append(formatted_chat)
        return {"text": formatted_texts}

    # Apply the formatting
    # Note: If your dataset is very large, consider using .map() with batched=True
    # and potentially num_proc from TRAIN_ARGS.dataset_num_proc
    train_dataset = raw_dataset.map(
        format_dataset_for_sft,
        batched=True,
        # num_proc=TRAIN_ARGS.dataset_num_proc, # Can enable for faster processing
        remove_columns=raw_dataset.column_names # Remove old columns
    )
    print(f"Custom dataset loaded and formatted. First example:\\n{train_dataset[0]['text']}")


    # Explicitly create the data collator with mlm=False
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # Crucial for preventing the TypeError
    )

    print("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # dataset_text_field is already in TRAIN_ARGS
        # max_seq_length is already in TRAIN_ARGS
        # packing is already in TRAIN_ARGS
        data_collator=data_collator, # Pass the explicit collator
        args=TRAIN_ARGS, # Pass the pre-configured SFTConfig object
    )
    print("SFTTrainer setup complete.")

    # --- Training ---
    print("Starting training...")
    # @title Show current memory stats (if you have torch imported and a GPU)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training.")
    
    trainer_stats = trainer.train()
    
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    
    print("Training finished.")
    if hasattr(trainer_stats, 'metrics'):
        print(f"Training stats: {trainer_stats.metrics}")


    # --- Inference Example (Optional) ---
    print("\n--- Running Inference Example ---")
    FastLanguageModel.for_inference(model) # Enable native Unsloth inference

    messages = [
        {"role": "user", "content": "Analyze the following C code snippet for vulnerabilities: ... (your test code here) ... Hint: It's vulnerable. Changed lines: ..."}
    ]
    
    # Apply chat template for inference, enable_thinking=True, add_generation_prompt=True
    text_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # Tokenize manually after templating
        add_generation_prompt=True, # Crucial for inference
        enable_thinking=True # Consistent with training
    )
    
    tokenized_inputs = tokenizer(text_inputs, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating response...")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **tokenized_inputs,
        max_new_tokens=1024, # Adjust as needed
        temperature=0.6,    # Recommended for reasoning
        top_p=0.95,         # Recommended for reasoning
        top_k=20,           # Recommended for reasoning
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id # Important for generation
    )
    print("\n--- Inference Example Finished ---")

    # --- Saving the Model (Optional) ---
    print(f"Saving LoRA adapters to ./{OUTPUT_DIR}/lora_model")
    model.save_pretrained_merged(f"./{OUTPUT_DIR}/lora_model_merged_16bit", tokenizer, save_method = "merged_16bit",)
    # model.save_pretrained_merged(f"./{OUTPUT_DIR}/lora_model_merged_4bit", tokenizer, save_method = "merged_4bit",)
    # model.save_pretrained(f"./{OUTPUT_DIR}/lora_model") # Saves LoRA adapters only
    # tokenizer.save_pretrained(f"./{OUTPUT_DIR}/lora_model")
    print("Model saving process initiated. Check console for completion.")
    
    # To push to Hugging Face Hub:
    # model.push_to_hub_merged("your_hf_username/your_model_name_merged_16bit", tokenizer, save_method = "merged_16bit", token = "hf_...")
    # model.push_to_hub("your_hf_username/your_model_name_lora", tokenizer, save_method = "lora", token = "hf_...")

if __name__ == "__main__":
    main()
