#!/usr/bin/env python3
"""
Unsloth GRPO Training Script for Vulnerability Detection

Uses:
- Unsloth FastLanguageModel with high-rank LoRA (r=256) for near full-finetuning quality
- vLLM for fast inference (required for long context GRPO)
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


# =============================================================================
# Model Loading with LoRA
# =============================================================================

def load_unsloth_model(
    model_name: str,
    max_seq_length: int = 40960,  # Qwen3's max_position_embeddings (native limit)
    lora_rank: int = 64,  # Balance between memory and model quality
    gpu_memory_utilization: float = 0.75,  # Lower to leave room for training gradients
):
    """
    Load model with Unsloth using high-rank LoRA for near full-finetuning quality.
    
    NOTE: Full fine-tuning is incompatible with vLLM (fast_inference=True).
    
    Args:
        model_name: Path to SFT checkpoint or HuggingFace model
        max_seq_length: Maximum sequence length (can go very high with vLLM)
        lora_rank: LoRA rank (256 is near full-finetuning)
        gpu_memory_utilization: GPU memory fraction for vLLM
    """
    print(f"Loading model: {model_name}")
    print(f"Max seq length: {max_seq_length}")
    print(f"LoRA rank: {lora_rank}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,         # BF16 LoRA (not 4-bit QLoRA)
        max_lora_rank=lora_rank,
        # fast_inference disabled - use standard generation instead of vLLM
        # This avoids memory contention between vLLM and training
    )
    
    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # Standard: alpha = 2 * rank
        lora_dropout=0,             # No dropout for RL
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
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


# =============================================================================
# GRPO Configuration
# =============================================================================

def create_grpo_config(output_dir: str, tokenizer, max_steps: int = -1):
    """Create GRPOConfig with Qwen3 thinking mode settings (no vLLM)."""
    # No vLLM - using standard generation
    
    config = GRPOConfig(
        output_dir=output_dir,
        
        # No vLLM sampling params - use regular generation
        
        # Sequence lengths
        # Model native limit: 40960 tokens (max_position_embeddings)
        # Max prompt in dataset: 28681 tokens -> leaves 40960-28681 = 12279 for completion
        max_prompt_length=28672,
        max_completion_length=12288,  # ~12K for reasoning (40960 - max_prompt ~28K)
        
        # Generation settings (Qwen3 thinking mode: Temperature=0.6)
        num_generations=2,  # Reduced for memory
        temperature=0.6,
        top_p=0.95,   # Enable for non-vLLM generation
        top_k=20,     # Enable for non-vLLM generation
        
        # Training params - reduced LR to prevent loss explosion
        learning_rate=8e-7,  # Reduced from 1e-6 to stabilize training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=15,
        num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,  # Gradient clipping to prevent explosion
        
        # GRPO specific
        loss_type="grpo",
        beta=0.0,  # No KL penalty
        
        # Logging and checkpointing - save frequently for safety
        logging_steps=1,
        save_steps=5,  # Save every 25 steps (~1.5h) for earlier checkpoints
        save_total_limit=3,  # Keep more checkpoints
        
        # Mixed precision
        bf16=True,
        
        # Other
        seed=42,
        report_to="wandb",
    )
    
    if max_steps > 0:
        config.max_steps = max_steps
    
    return config


# =============================================================================
# Reward Function
# =============================================================================

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


# =============================================================================
# Dataset Loading
# =============================================================================

def load_training_dataset(data_path: str, tokenizer):
    """
    Load and prepare the training dataset.
    
    Uses the model's default chat template for formatting.
    Expected format: JSONL with 'prompt', 'is_vulnerable', 'ground_truth_lines'
    """
    import numpy as np
    
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
    
    # Get token length statistics for info (no filtering - dataset already clean)
    print(f"\n[Dataset Token Length Stats]")
    
    def get_token_length(example):
        tokens = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        return {"token_length": len(tokens)}
    
    temp_dataset = dataset.map(get_token_length, desc="Calculating token lengths")
    lengths = np.array(temp_dataset["token_length"])
    print(f"Token length stats: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}, median={np.median(lengths):.1f}")
    print(f"90th percentile: {np.quantile(lengths, 0.9):.0f}, 95th: {np.quantile(lengths, 0.95):.0f}, 99th: {np.quantile(lengths, 0.99):.0f}")
    del temp_dataset
    
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
    
    parser = argparse.ArgumentParser(description="Unsloth GRPO Training with LoRA")
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
        "--lora_rank",
        type=int,
        default=64,  # Smaller rank saves memory while still training effectively
        help="LoRA rank (64 for memory-efficient training)"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,  # Reduced from 4 for memory
        help="Number of completions per prompt (reduces memory usage)"
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
    
    # Load model and tokenizer with LoRA
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=40960,  # Qwen3's max_position_embeddings
        lora_rank=args.lora_rank,
        gpu_memory_utilization=0.5,  # Very low to force Unsloth to not override
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
    print("Starting GRPO training with LoRA...")
    print(f"LoRA rank: {args.lora_rank}")
    print("="*60 + "\n")
    trainer.train()
    
    # Save LoRA adapters
    final_path = os.path.join(args.output_dir, "final_lora")
    print(f"\nSaving LoRA adapters to {final_path}")
    model.save_lora(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Option to merge and save full model
    merged_path = os.path.join(args.output_dir, "final_merged")
    print(f"\nMerging LoRA and saving full model to {merged_path}")
    model.save_pretrained_merged(merged_path, tokenizer)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
