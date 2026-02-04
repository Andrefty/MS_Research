#!/usr/bin/env python3
"""
Prepare dataset for Unsloth GRPO training.

Converts SFT JSONL format to the format expected by TRL GRPOTrainer:
- prompt: List of chat messages
- is_vulnerable: bool
- ground_truth_lines: List[int]
"""

import json
import argparse
from pathlib import Path


def convert_sft_to_grpo(input_path: str, output_path: str):
    """
    Convert SFT JSONL to GRPO format.
    
    Input format: {"prompt": str, "response": str (optional), ...}
    Output format: {"prompt": [{"role": "user", "content": str}], "is_vulnerable": bool, "ground_truth_lines": list}
    """
    converted = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Skipping line {i}: {e}")
                continue
            
            # Extract the prompt
            prompt = item.get("prompt", "")
            
            # Extract ground truth info
            is_vulnerable = item.get("is_vulnerable", True)
            ground_truth_lines = item.get("ground_truth_lines", [])
            
            # Format as chat messages
            converted_item = {
                "prompt": [
                    {"role": "user", "content": prompt}
                ],
                "is_vulnerable": is_vulnerable,
                "ground_truth_lines": ground_truth_lines,
            }
            converted.append(converted_item)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SFT dataset to GRPO format")
    parser.add_argument("--input", type=str, required=True, help="Input SFT JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output GRPO JSONL file")
    args = parser.parse_args()
    
    convert_sft_to_grpo(args.input, args.output)


if __name__ == "__main__":
    main()
