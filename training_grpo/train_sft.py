#!/usr/bin/env python3
"""
SFT Training Script for Qwen3-4B with DeepSpeed ZeRO-3.
Fine-tunes on vulnerability analysis teacher responses.
"""

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_chat_for_training(examples, tokenizer, max_length):
    """
    Format examples into chat template for training.
    Each example has 'prompt' and 'response' fields.
    """
    formatted_texts = []
    
    for prompt, response in zip(examples['prompt'], examples['response']):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True  # Qwen3 thinking mode
        )
        formatted_texts.append(formatted)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3-4B")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        help="Model name or path")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Maximum sequence length")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSONL)")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Path to validation data (JSONL)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"
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
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Load datasets
    logger.info(f"Loading training data: {args.train_file}")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    val_dataset = None
    if args.val_file:
        logger.info(f"Loading validation data: {args.val_file}")
        val_dataset = load_dataset("json", data_files=args.val_file, split="train")
    
    # Preprocess datasets
    logger.info("Tokenizing datasets...")
    
    train_dataset = train_dataset.map(
        lambda x: format_chat_for_training(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        desc="Tokenizing train data"
    )
    
    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: format_chat_for_training(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=val_dataset.column_names,
            num_proc=4,
            desc="Tokenizing val data"
        )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        seed=args.seed,
        report_to="wandb",  # or "none"
        run_name=f"sft-qwen3-8b-vuln",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
