#!/usr/bin/env python3
"""
Merge Unsloth LoRA adapter with base model.
Saves a fully merged HuggingFace model that can be loaded by SGLang/vLLM.

For checkpoints saved by TRL's GRPOTrainer (which saves PEFT format),
we use PEFT's merge_and_unload() then save_pretrained().
"""
import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/sft_qwen3_4b",
        help="Path to base model"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (containing adapter_config.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=40960,
        help="Max sequence length (should match training config)"
    )
    args = parser.parse_args()
    
    # Check if LoRA adapter exists
    adapter_config_path = os.path.join(args.lora_adapter, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"ERROR: adapter_config.json not found at: {args.lora_adapter}")
        print("This doesn't appear to be a valid LoRA checkpoint.")
        sys.exit(1)
    
    # Check if already merged
    if os.path.exists(os.path.join(args.output, "config.json")):
        print(f"Merged model already exists at: {args.output}")
        print("To re-merge, delete the output directory first.")
        sys.exit(0)
    
    print("="*60)
    print("Merging LoRA Adapter with Base Model")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"LoRA adapter: {args.lora_adapter}")
    print(f"Output: {args.output}")
    print(f"Max seq length: {args.max_seq_length}")
    print("="*60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    print(f"\nLoading LoRA adapter from: {args.lora_adapter}")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_adapter,
        torch_dtype=torch.bfloat16,
    )
    
    # Merge LoRA weights into base model
    print("\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"\nSaving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    
    print("\n" + "="*60)
    print("Merge complete!")
    print(f"Merged model saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
