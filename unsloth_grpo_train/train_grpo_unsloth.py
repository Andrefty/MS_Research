#!/usr/bin/env python3
"""
Unsloth GRPO Training Script for Vulnerability Detection

Uses:
- Unsloth FastLanguageModel for full BF16 fine-tuning with vLLM
- TRL GRPOTrainer with custom reward function
- Qwen3-4B SFT checkpoint with its DEFAULT chat template
"""

import os
import sys
import gc

# Enable Unsloth Standby mode for memory-efficient RL
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import torch


def load_unsloth_model(
    model_name: str,
    max_seq_length: int = 32768,
    gpu_memory_utilization: float = 0.95,
):
    """Load model with Unsloth for full BF16 fine-tuning with vLLM."""
    print(f"Loading model: {model_name}")
    print(f"Max seq length: {max_seq_length}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        full_finetuning=True,       # Full fine-tuning, not LoRA
        load_in_4bit=False,         # BF16 precision
        fast_inference=True,        # Enable vLLM, incompatible with full_finetuning
        gpu_memory_utilization=gpu_memory_utilization, # gpu_memory_utilization only used with fast_inference=True
    )
    
    # Ensure pad token (but DO NOT modify chat_template - use model's default!)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print chat template info for verification
    print(f"\n[Chat Template Info]")
    print(f"Using model's default chat template: {tokenizer.chat_template is not None}")
    if tokenizer.chat_template:
        print(f"Template preview: {tokenizer.chat_template[:200]}...")
    
    return model, tokenizer


def create_grpo_config(output_dir: str, tokenizer, max_steps: int = -1):
    """Create GRPOConfig with Qwen3 thinking mode settings and vLLM params."""
    from vllm import SamplingParams
    
    # vLLM sampling params with stop token
    # Using Qwen3 "thinking mode" settings: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
    vllm_sampling_params = SamplingParams(
        min_p=0.0,    # Qwen3 thinking mode: MinP=0
        top_p=0.95,   # Qwen3 thinking mode
        top_k=20,     # Qwen3 thinking mode
        seed=42,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    config = GRPOConfig(
        output_dir=output_dir,
        
        # vLLM sampling params
        vllm_sampling_params=vllm_sampling_params,
        
        # Sequence lengths (matching original veRL config)
        max_prompt_length=28672,
        max_completion_length=32768,
        
        # Generation settings (Qwen3 thinking mode: Temperature=0.6)
        num_generations=4,
        temperature=0.6,
        # top_p=0.95, #Only when without vLLM
        # top_k=20, #Only when without vLLM
        
        # Training params
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=15,
        num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        
        # GRPO specific
        loss_type="grpo",
        beta=0.0,  # No KL penalty
        
        # Logging
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        
        # Mixed precision
        bf16=True,
        
        # Other
        seed=42,
        report_to="wandb",
    )
    
    if max_steps > 0:
        config.max_steps = max_steps
    
    return config


def reward_function(completions, **kwargs):
    """
    Wrapper for the existing reward function with TRL format handling.
    
    TRL passes completions as list of list of dicts:
    completions = [[{"role": "assistant", "content": "..."}], ...]
    
    Expects kwargs to contain:
    - is_vulnerable: List[bool]
    - ground_truth_lines: List[List[int]]
    """
    from reward_function import reward_function_for_grpo
    
    # Extract text content from TRL's completion format
    completion_texts = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            # TRL format: [[{"role": "assistant", "content": "..."}], ...]
            content = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        elif isinstance(completion, str):
            content = completion
        else:
            content = str(completion) if completion else ""
        completion_texts.append(content)
    
    # Call the actual reward function (it handles logging internally)
    return reward_function_for_grpo(completion_texts, **kwargs)


def load_training_dataset(data_path: str, tokenizer):
    """
    Load and prepare the training dataset.
    
    Uses the model's default chat template for formatting.
    Expected format: JSONL with 'prompt', 'is_vulnerable', 'ground_truth_lines'
    """
    print(f"Loading dataset from: {data_path}")
    
    # Load raw JSONL
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Convert to conversational format for TRL
    def format_example(example):
        # Build prompt as chat messages - using model's default template
        messages = [
            {"role": "user", "content": example["prompt"]}
        ]
        return {
            "prompt": messages,
            "is_vulnerable": example.get("is_vulnerable", True),
            "ground_truth_lines": example.get("ground_truth_lines", []),
        }
    
    dataset = dataset.map(format_example)
    
    # Verify chat template works correctly
    print(f"\n[Dataset Info]")
    print(f"Loaded {len(dataset)} examples")
    if len(dataset) > 0:
        sample_formatted = tokenizer.apply_chat_template(
            dataset[0]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"Sample formatted prompt (first 500 chars):\n{sample_formatted[:500]}...")
    
    return dataset


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Unsloth GRPO Training")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/sft_qwen3_4b",
        help="Path to SFT checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/training_grpo/sft_dataset_train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/grpo_unsloth",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug dataset for quick testing"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epoch)"
    )
    args = parser.parse_args()
    
    # Use debug dataset if specified
    if args.debug:
        args.data_path = args.data_path.replace(".jsonl", "_debug.jsonl")
        print(f"DEBUG MODE: Using {args.data_path}")
    
    # Load model and tokenizer
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=32768,
    )
    
    # Load dataset
    dataset = load_training_dataset(args.data_path, tokenizer)
    
    # Create config
    config = create_grpo_config(
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        max_steps=args.max_steps,
    )
    
    # Free up memory before training (from example)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_function,
        args=config,
        train_dataset=dataset,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting GRPO training...")
    print("="*60 + "\n")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
