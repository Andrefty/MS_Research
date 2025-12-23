#!/usr/bin/env python3
"""
GRPO Training Script for Qwen3-4B.
Uses trl GRPOTrainer with programmatic reward function.
"""

import os
import json
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import logging

from reward_function import compute_reward, parse_model_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_grpo_dataset(data_file: str) -> Dataset:
    """
    Load dataset for GRPO training.
    
    Expected input format (from prepare_sft_dataset.py):
    {
        "prompt": str,
        "response": str,  # Not used in GRPO (we generate new ones)
        "is_vulnerable": bool,
        "commit_id": str,
        "ground_truth_lines": list[int]
    }
    
    Returns HuggingFace Dataset with required fields.
    """
    samples = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append({
                    "prompt": data["prompt"],
                    "is_vulnerable": data["is_vulnerable"],
                    "ground_truth_lines": data.get("ground_truth_lines", []),
                })
    
    return Dataset.from_list(samples)


def create_reward_function(tokenizer):
    """
    Create reward function that works with GRPOTrainer.
    
    GRPOTrainer expects: reward_fn(completions, prompts, ...) -> List[float]
    """
    def reward_fn(completions, prompts=None, **kwargs):
        """
        Compute rewards for completions.
        
        Note: We need to pass ground truth info through the prompts or a side channel.
        For simplicity, we'll encode the ground truth in the prompt parsing.
        """
        rewards = []
        
        for i, completion in enumerate(completions):
            # Decode if tensor
            if hasattr(completion, 'tolist'):
                text = tokenizer.decode(completion, skip_special_tokens=True)
            else:
                text = completion
            
            # For now, use default values - will be overridden in custom trainer
            # In practice, you'd need to match completions back to their metadata
            is_vulnerable = kwargs.get('is_vulnerable', [True] * len(completions))[i] if 'is_vulnerable' in kwargs else True
            ground_truth_lines = kwargs.get('ground_truth_lines', [[]] * len(completions))[i] if 'ground_truth_lines' in kwargs else []
            
            reward = compute_reward(text, is_vulnerable, ground_truth_lines)
            rewards.append(reward)
        
        return rewards
    
    return reward_fn


class VulnGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer that passes ground truth info to reward function.
    """
    
    def compute_rewards(self, completions, prompts, ground_truth_info):
        """Override to pass ground truth info."""
        rewards = []
        
        for i, completion in enumerate(completions):
            # Decode completion
            if isinstance(completion, torch.Tensor):
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
            else:
                text = completion
            
            # Get ground truth for this sample
            is_vuln = ground_truth_info[i].get('is_vulnerable', True)
            gt_lines = ground_truth_info[i].get('ground_truth_lines', [])
            
            reward = compute_reward(text, is_vuln, gt_lines)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.model.device)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Qwen3-4B")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to SFT checkpoint or model name")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens to generate")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSONL)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate (typically lower for GRPO)")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    
    # GRPO specific - Qwen3 thinking mode: Temp=0.6, TopP=0.95, TopK=20
    # NOTE: beta=0.0 means no KL divergence penalty. Recent research (Open-Reasoner-Zero,
    # DAPO, Dr.GRPO) shows KL is not essential for GRPO and excluding it saves memory
    # (no reference model needed) and speeds up training. See TRL docs for details.
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL divergence coefficient (0.0 = disabled, saves memory)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (Qwen3 thinking: 0.6)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling (Qwen3 thinking: 0.95)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling (Qwen3 thinking: 20)")
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",  # Left padding for generation
        fix_mistral_regex=True,  # Fix regex pattern issue from transformers <= 4.57.2
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    # Load dataset
    logger.info(f"Loading training data: {args.train_file}")
    train_dataset = load_grpo_dataset(args.train_file)
    logger.info(f"Loaded {len(train_dataset)} samples")
    
    # Create reward function
    reward_fn = create_reward_function(tokenizer)
    
    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        seed=args.seed,
        report_to="wandb",
        run_name=f"grpo-qwen3-4b-vuln",
        # GRPO specific
        num_generations=args.num_generations,
        # CRITICAL: max_completion_length must match max_new_tokens!
        # Default is 256 which causes 1-token completions with long prompts
        max_completion_length=args.max_new_tokens,
        # KL divergence penalty (beta)
        beta=args.beta,
        # Enable completion logging for debugging
        log_completions=True,
        # num_completions_to_print=5, # Commented out to get all the completions
        # Generation params passed via generation_kwargs for Qwen3 thinking mode
        generation_kwargs={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
        },
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("GRPO training complete!")


if __name__ == "__main__":
    main()
