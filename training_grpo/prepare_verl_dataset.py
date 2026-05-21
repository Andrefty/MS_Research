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
    if is_vulnerable:
        # Vulnerable: use deleted_lines (already in ground_truth_lines from SFT prep)
        ground_truth_lines = sample.get("ground_truth_lines", [])
    elif score_patched_lines and merged_index:
        # Patched + toggle ON: look up added_lines from merged dataset
        pair_id = sample.get("pair_id", "")
        if pair_id in merged_index:
            ground_truth_lines = get_added_line_numbers(merged_index[pair_id])
        else:
            ground_truth_lines = []
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
