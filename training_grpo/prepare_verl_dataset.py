#!/usr/bin/env python3
"""
Convert SFT dataset (JSONL) to veRL-compatible parquet format.

Supports toggleable scoring of important_lines for patched (non-vulnerable) samples:
- --score_patched_lines: When enabled, uses added_lines from merged dataset as
  ground_truth_lines for patched samples (so GRPO rewards line identification in fixes).
- Default (disabled): ground_truth_lines for patched samples is [] (current behavior).

Usage:
    python prepare_verl_dataset.py \\
        --input_file sft_dataset_train.jsonl \\
        --output_dir verl_data \\
        --merged_dataset Research_merged_dataset/merged_train.jsonl \\
        --score_patched_lines \\
        --val_ratio 0.05
"""

import json
import argparse
import sys
import os
import math
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow not installed. Run: pip install pyarrow")
    exit(1)

try:
    from datasets import Dataset
except ImportError:
    Dataset = None
    print("Warning: datasets library not installed, using direct parquet write")


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


def get_added_line_numbers(sample: dict) -> list:
    """Extract line numbers from added_lines (fix lines in patched code).
    
    Filters out entries with empty text.
    """
    added_lines = sample.get('added_lines', [])
    if added_lines:
        return sorted(set(
            item['line_no'] for item in added_lines
            if 'line_no' in item and item.get('text', '').strip()
        ))
    return []


def convert_sample(sample: dict, idx: int, merged_index: dict = None,
                   score_patched_lines: bool = False) -> dict:
    """
    Convert a single sample to veRL format.
    
    veRL expects:
    - data_source: str - dataset identifier
    - prompt: List[Dict] - HF chat template format
    - ability: str - task category
    - reward_model: Dict with ground_truth
    - extra_info: Dict with metadata
    """
    # The prompt field already contains the formatted user message
    prompt_text = sample.get("prompt", "")
    is_vulnerable = sample.get("is_vulnerable", True)
    
    # Determine ground_truth_lines based on toggle
    ground_truth_lines = sample.get("ground_truth_lines", [])
    
    if not is_vulnerable:
        if score_patched_lines:
            # If the jsonl already has them (new format), use them.
            # If empty, fallback to merged dataset lookup (old format).
            if not ground_truth_lines and merged_index:
                pair_id = sample.get("pair_id", "")
                if pair_id in merged_index:
                    ground_truth_lines = get_added_line_numbers(merged_index[pair_id])
        else:
            # Patched + toggle OFF: empty (current behavior)
            ground_truth_lines = []
    
    # Ground truth for reward function
    ground_truth = json.dumps({
        "is_vulnerable": is_vulnerable,
        "ground_truth_lines": ground_truth_lines
    })
    
    return {
        "data_source": "vulnerability_detection",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "security",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            "pair_id": sample.get("pair_id", ""),
            "commit_id": sample.get("commit_id", ""),
            "idx": idx,
            "is_vulnerable": is_vulnerable
        }
    }


def flatten_for_parquet(sample: dict) -> dict:
    """
    Flatten nested structures for parquet storage.
    veRL reads these fields directly.
    Note: extra_info and reward_model must remain as dicts as veRL calls .get() on them.
    """
    return {
        "data_source": sample["data_source"],
        "prompt": sample["prompt"],  # Keep as list of dicts, NOT JSON string! veRL expects this.
        "ability": sample["ability"],
        "reward_model": sample["reward_model"],  # Keep as dict, veRL expects ["ground_truth"]
        "extra_info": sample["extra_info"]  # Keep as dict, veRL expects .get() to work
    }


def save_parquet_with_datasets(samples: list, output_path: str):
    """Save using HuggingFace datasets library (preferred)."""
    if Dataset is None:
        raise ImportError("datasets library not available")
    
    dataset = Dataset.from_list(samples)
    dataset.to_parquet(output_path)
    print(f"Saved {len(samples)} samples to {output_path}")


def save_parquet_direct(samples: list, output_path: str):
    """Save directly using pyarrow. Note: This flattens dicts to JSON strings."""
    # Convert dicts to JSON strings for pyarrow compatibility
    # (HuggingFace datasets handles dicts natively, use that method if available)
    columns = {
        "data_source": [s["data_source"] for s in samples],
        "prompt": [s["prompt"] for s in samples],
        "ability": [s["ability"] for s in samples],
        "reward_model": [json.dumps(s["reward_model"]) for s in samples],
        "extra_info": [json.dumps(s["extra_info"]) for s in samples]
    }
    
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.string()),
        ("ability", pa.string()),
        ("reward_model", pa.string()),  # JSON string
        ("extra_info", pa.string())  # JSON string
    ])
    
    table = pa.table(columns, schema=schema)
    pq.write_table(table, output_path)
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SFT JSONL dataset to veRL parquet format"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to input JSONL file (e.g., sft_dataset_train.jsonl)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="verl_data",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--merged_dataset", type=str, default=None,
        help="Path to merged dataset (required when --score_patched_lines is set)"
    )
    parser.add_argument(
        "--score_patched_lines", action="store_true",
        help="If set, use added_lines from merged dataset as ground_truth for patched samples"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Validation split ratio (default: 0.05)"
    )
    parser.add_argument(
        "--use_existing_split", action="store_true",
        help="If set, expects separate train/val JSONL files"
    )
    parser.add_argument(
        "--val_file", type=str, default=None,
        help="Path to validation JSONL file (if using existing split)"
    )
    parser.add_argument(
        "--max_gt_lines", type=int, default=None,
        help="Manual threshold: zero-out ground_truth_lines for samples with more GT lines than this value"
    )
    parser.add_argument(
        "--gt_percentile_cap", type=float, default=None,
        help="Auto-calculate threshold at this percentile (e.g. 99). "
             "Uses vuln-only percentile when patched GT is empty/disabled, all-samples otherwise."
    )
    parser.add_argument(
        "--gt_percentile_multiplier", type=float, default=1,
        help="Multiplier on the percentile-based threshold (default: 1). "
             "Final threshold = ceil(multiplier * percentile_value)"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if args.score_patched_lines and not args.merged_dataset:
        parser.error("--merged_dataset is required when --score_patched_lines is set")
    
    # Load merged dataset index if needed
    merged_index = None
    if args.merged_dataset:
        print(f"Loading merged dataset: {args.merged_dataset}")
        merged_index = build_merged_index(args.merged_dataset)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples")
    
    if args.score_patched_lines:
        print("Score patched lines: ENABLED (using added_lines as ground_truth for patched samples)")
    else:
        print("Score patched lines: DISABLED (ground_truth_lines=[] for patched samples)")
    
    # Convert to veRL format
    print("Converting to veRL format...")
    converted = [
        flatten_for_parquet(convert_sample(s, i, merged_index, args.score_patched_lines))
        for i, s in enumerate(samples)
    ]
    
    # ================================================================
    # GT Line Capping: zero-out GT lines for samples exceeding threshold
    # ================================================================
    gt_cap_threshold = None
    
    if args.max_gt_lines is not None and args.gt_percentile_cap is not None:
        parser.error("Cannot use both --max_gt_lines and --gt_percentile_cap")
    
    if args.gt_percentile_cap is not None:
        # Auto-calculate threshold based on percentile
        import numpy as np
        
        # Collect GT line counts per class
        vuln_gt_counts = []
        patched_gt_counts = []
        for sample in converted:
            gt_str = sample['reward_model']['ground_truth']
            gt = json.loads(gt_str)
            gt_lines = gt.get('ground_truth_lines', [])
            if gt.get('is_vulnerable', True):
                vuln_gt_counts.append(len(gt_lines))
            else:
                patched_gt_counts.append(len(gt_lines))
        
        # Decide which distribution to use for percentile:
        # If patched samples have no GT lines (all zeros) or --score_patched_lines is OFF,
        # use vulnerable-only percentile.
        # Otherwise use all samples.
        patched_has_gt = any(c > 0 for c in patched_gt_counts)
        
        if not patched_has_gt or not args.score_patched_lines:
            base_counts = vuln_gt_counts
            percentile_source = "vulnerable-only"
        else:
            base_counts = vuln_gt_counts + patched_gt_counts
            percentile_source = "all samples"
        
        # Filter to non-zero counts for percentile (samples with 0 GT lines are classification-only anyway)
        non_zero_counts = [c for c in base_counts if c > 0]
        
        if non_zero_counts:
            pctl_value = np.percentile(non_zero_counts, args.gt_percentile_cap)
            gt_cap_threshold = math.ceil(args.gt_percentile_multiplier * pctl_value)
            print(f"\nGT Line Capping: AUTO (percentile-based)")
            print(f"  Source: {percentile_source} ({len(non_zero_counts)} non-zero samples)")
            print(f"  {args.gt_percentile_cap}th percentile: {pctl_value:.1f}")
            print(f"  Multiplier: {args.gt_percentile_multiplier}")
            print(f"  Final threshold: {gt_cap_threshold}")
        else:
            print("\nWARNING: No non-zero GT line counts found, skipping GT line capping.")
    
    elif args.max_gt_lines is not None:
        gt_cap_threshold = args.max_gt_lines
        print(f"\nGT Line Capping: MANUAL (threshold = {gt_cap_threshold})")
    
    # Apply capping if threshold is set
    if gt_cap_threshold is not None:
        capped_count = 0
        for sample in converted:
            gt_str = sample['reward_model']['ground_truth']
            gt = json.loads(gt_str)
            gt_lines = gt.get('ground_truth_lines', [])
            if len(gt_lines) > gt_cap_threshold:
                capped_count += 1
                gt['ground_truth_lines'] = []
                sample['reward_model']['ground_truth'] = json.dumps(gt)
        print(f"  Zeroed-out GT lines for {capped_count}/{len(converted)} samples "
              f"({100*capped_count/len(converted):.2f}%) exceeding {gt_cap_threshold} lines")
        print(f"  (These samples will be classification-only during GRPO)")
        
        # Also print the recommended spray threshold for the job script
        print(f"\n  >>> Recommended GRPO_SPRAY_THRESHOLD env var: {gt_cap_threshold} <<<")
    
    # Split into train/val if needed
    if args.use_existing_split and args.val_file:
        train_samples = converted
        val_raw = load_jsonl(args.val_file)
        val_samples = [
            flatten_for_parquet(convert_sample(s, i, merged_index, args.score_patched_lines))
            for i, s in enumerate(val_raw)
        ]
    else:
        val_size = int(len(converted) * args.val_ratio)
        train_samples = converted[:-val_size] if val_size > 0 else converted
        val_samples = converted[-val_size:] if val_size > 0 else []
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Save parquet files
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    
    save_fn = save_parquet_with_datasets if Dataset else save_parquet_direct
    
    save_fn(train_samples, str(train_path))
    if val_samples:
        save_fn(val_samples, str(val_path))
    
    print(f"\nDone! Files saved to {output_dir}/")
    print(f"  - train.parquet: {len(train_samples)} samples")
    if val_samples:
        print(f"  - val.parquet: {len(val_samples)} samples")


if __name__ == "__main__":
    main()
