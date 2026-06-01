#!/usr/bin/env python3
"""
Prepare SFT dataset from GRPO generation output.
Converts the generated teacher responses into SFT training format.

Hint mode (--include_hints / --no_include_hints):
  WITH hints (--include_hints):
    Uses the teacher's original prompt directly (which includes CVE/CWE context,
    vulnerability hints, and commit messages with dynamic token-aware truncation).
    Best for SFT training so the student sees the same prompt the teacher used.

  WITHOUT hints (--no_include_hints, default):
    Reconstructs hint-free prompts from the merged dataset (identical to eval
    format). The merged dataset is loaded by pair_id to regenerate hint-free prompts
    from the original code, rather than copying the teacher's hint-rich prompt.
    Requires --merged_dataset. Used for GRPO training where the model
    must learn to reason without extra context.
"""

import json
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_dataset.generate_finetuning_dataset_grpo import format_prompt_for_model_grpo


def load_jsonl(file_path: str) -> list:
    """Load JSONL file into list of dicts."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def build_merged_index(merged_file: str) -> dict:
    """Build a pair_id -> sample lookup index from the merged dataset.
    
    pair_id is guaranteed unique (includes vuln_hash since the uniqueness fix).
    """
    index = {}
    samples = load_jsonl(merged_file)
    for sample in samples:
        pair_id = sample['pair_id']
        if pair_id in index:
            raise ValueError(f"Duplicate pair_id in merged dataset: {pair_id}. This should be unique!")
        index[pair_id] = sample
    print(f"Built merged dataset index: {len(index)} entries")
    return index


def prepare_sft_dataset(input_file: str, output_file: str, merged_file: str = None,
                        max_samples: int = None, include_hints: bool = False):
    """
    Convert GRPO generation output to SFT training format.
    
    Args:
        include_hints: If True, uses the teacher's original prompt (WITH hints)
            directly — no reconstruction needed, merged_file is not required.
            If False (default), reconstructs hint-free prompts from the merged
            dataset (requires merged_file).
    
    Input format (from generate_finetuning_dataset_grpo.py):
    {
        "pair_id": str,
        "commit_id": str,
        "source": str,
        "is_vulnerable": bool,
        "code": str,
        "prompt": str,              # Teacher prompt (WITH hints)
        "generated_response": str,  # Teacher response from Qwen3-32B — kept as SFT target
        "ground_truth_lines": list[int],
        "parsed_classification": str,
        "parsed_important_lines": list[int]
    }
    
    Output format for SFT training:
    {
        "prompt": str,              # With or without hints (per include_hints flag)
        "response": str,            # Teacher's response (generated WITH hints)
        "is_vulnerable": bool,
        "pair_id": str,
        "commit_id": str,
        "ground_truth_lines": list[int]
    }
    """
    
    hint_mode = "WITH hints (teacher prompt)" if include_hints else "HINT-FREE (eval format)"
    print(f"Prompt mode: {hint_mode}")
    
    # Only need merged dataset for hint-free prompt reconstruction
    merged_index = None
    if not include_hints:
        if not merged_file:
            raise ValueError("--merged_dataset is required when not including hints (hint-free mode)")
        print(f"Loading merged dataset: {merged_file}")
        merged_index = build_merged_index(merged_file)
    
    print(f"Loading teacher generation output: {input_file}")
    
    samples = []
    skipped_errors = 0
    skipped_no_response = 0
    skipped_incomplete = 0  # Missing </think> or failed parsing
    skipped_no_pair = 0     # pair_id not found in merged dataset
    
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
            
            # Skip samples where model didn't complete thinking (no </think> tag)
            # or where parsing failed (parsed_classification is None)
            # These indicate truncated/stuck generation
            has_think_end = '</think>' in response
            parsed_ok = data.get('parsed_classification') is not None
            
            if not has_think_end or not parsed_ok:
                skipped_incomplete += 1
                continue

            pair_id = data.get('pair_id', '')
            is_vulnerable = data['is_vulnerable']
            
            if include_hints:
                # Use the teacher's original prompt directly — it already has
                # CVE/CWE context, hints, and dynamically truncated commit messages
                prompt = data['prompt']
            else:
                # Reconstruct hint-free prompt from merged dataset
                if pair_id not in merged_index:
                    skipped_no_pair += 1
                    continue

                # Look up original sample from merged dataset
                source_sample = merged_index[pair_id]
                # Get the code from the source sample
                code = source_sample['vuln_func'] if is_vulnerable else source_sample['patched_func']
                
                prompt = format_prompt_for_model_grpo(
                    code_snippet=code,
                    sample=source_sample,
                    is_vulnerable=None,  # Unknown = no hints
                    ground_truth_lines=None,
                    include_hints=False
                )
            
            # Create SFT training sample
            sft_sample = {
                "prompt": prompt,
                "response": response,
                "is_vulnerable": is_vulnerable,
                "pair_id": pair_id,
                "commit_id": data['commit_id'],
                "ground_truth_lines": data.get('ground_truth_lines', [])
            }
            
            samples.append(sft_sample)
            
            if max_samples and len(samples) >= max_samples:
                break
    
    print(f"\nTotal samples processed: {len(samples)}")
    print(f"Skipped (JSON errors): {skipped_errors}")
    print(f"Skipped (no/error response): {skipped_no_response}")
    print(f"Skipped (incomplete/unparseable): {skipped_incomplete}")
    print(f"Skipped (pair_id not found): {skipped_no_pair}")
    
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
    parser.add_argument("--merged_dataset", type=str, default=None,
                        help="Path to merged dataset (required when NOT using --include_hints)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    # Hint toggle: default OFF (hint-free, current behavior)
    parser.add_argument("--include_hints", action="store_true", dest="include_hints", default=False,
                        help="Use teacher's original prompt WITH hints (for SFT training)")
    parser.add_argument("--no_include_hints", action="store_false", dest="include_hints",
                        help="Reconstruct hint-free prompts from merged dataset (default, for GRPO)")
    
    args = parser.parse_args()
    
    prepare_sft_dataset(args.input_file, args.output_file, args.merged_dataset,
                        args.max_samples, args.include_hints)


if __name__ == "__main__":
    main()
