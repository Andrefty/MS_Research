#!/usr/bin/env python3
"""
Convert SFT dataset (JSONL) to veRL-compatible parquet format.

Usage:
    python prepare_verl_dataset.py \
        --input_file sft_dataset_train.jsonl \
        --output_dir verl_data \
        --val_ratio 0.05
"""

import json
import argparse
import os
from pathlib import Path

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


def convert_sample(sample: dict, idx: int) -> dict:
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
    
    # Ground truth for reward function
    ground_truth = json.dumps({
        "is_vulnerable": sample.get("is_vulnerable", True),
        "ground_truth_lines": sample.get("ground_truth_lines", [])
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
            "commit_id": sample.get("commit_id", ""),
            "idx": idx,
            "is_vulnerable": sample.get("is_vulnerable", True)
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples")
    
    # Convert to veRL format
    print("Converting to veRL format...")
    converted = [
        flatten_for_parquet(convert_sample(s, i))
        for i, s in enumerate(samples)
    ]
    
    # Split into train/val if needed
    if args.use_existing_split and args.val_file:
        train_samples = converted
        val_raw = load_jsonl(args.val_file)
        val_samples = [
            flatten_for_parquet(convert_sample(s, i))
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
