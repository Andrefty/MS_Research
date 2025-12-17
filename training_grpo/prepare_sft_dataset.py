#!/usr/bin/env python3
"""
Prepare SFT dataset from GRPO generation output.
Converts the generated teacher responses into SFT training format.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def prepare_sft_dataset(input_file: str, output_file: str, max_samples: int = None):
    """
    Convert GRPO generation output to SFT training format.
    
    Input format (from generate_finetuning_dataset_grpo.py):
    {
        "commit_id": str,
        "source": str,
        "is_vulnerable": bool,
        "code": str,
        "prompt": str,
        "generated_response": str,  # Teacher response from Qwen3-32B
        "ground_truth_lines": list[int],
        "parsed_classification": str,
        "parsed_vulnerable_lines": list[int]
    }
    
    Output format for SFT training:
    {
        "prompt": str,
        "response": str,
        "is_vulnerable": bool,
        "commit_id": str,
        "ground_truth_lines": list[int]
    }
    """
    
    print(f"Loading input file: {input_file}")
    
    samples = []
    skipped_errors = 0
    skipped_no_response = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing samples"):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped_errors += 1
                continue
            
            # Skip samples with error responses
            response = data.get('generated_response', '')
            if not response or response.startswith('ERROR_'):
                skipped_no_response += 1
                continue
            
            # Create SFT training sample
            sft_sample = {
                "prompt": data['prompt'],
                "response": response,
                "is_vulnerable": data['is_vulnerable'],
                "commit_id": data['commit_id'],
                "ground_truth_lines": data.get('ground_truth_lines', [])
            }
            
            samples.append(sft_sample)
            
            if max_samples and len(samples) >= max_samples:
                break
    
    print(f"\nTotal samples processed: {len(samples)}")
    print(f"Skipped (JSON errors): {skipped_errors}")
    print(f"Skipped (no/error response): {skipped_no_response}")
    
    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved SFT dataset to: {output_file}")
    
    # Also create a train/val split
    val_ratio = 0.05
    val_size = int(len(samples) * val_ratio)
    train_size = len(samples) - val_size
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    train_file = output_file.replace('.jsonl', '_train.jsonl')
    val_file = output_file.replace('.jsonl', '_val.jsonl')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Train set: {len(train_samples)} samples -> {train_file}")
    print(f"Val set: {len(val_samples)} samples -> {val_file}")
    
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset from GRPO generation output")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to GRPO generation output (grpo_finetuning_dataset.jsonl)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save SFT dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    prepare_sft_dataset(args.input_file, args.output_file, args.max_samples)


if __name__ == "__main__":
    main()
