#!/usr/bin/env python3
"""
prepare_eval_dataset.py - Prepare evaluation dataset by merging test splits.

Reuses loading functions from merge_datasets.py for consistency.
Combines:
  - PrimeVul paired test
  - PrimeVul paired valid
  - SVEN val

Usage:
    python prepare_eval_dataset.py \
        --primevul_test_path ../PrimeVul-v0.1-hf/paired/primevul_test_paired.jsonl \
        --primevul_valid_path ../PrimeVul-v0.1-hf/paired/primevul_valid_paired.jsonl \
        --sven_val_path ../sven/data/val-*.parquet \
        --output_path eval_dataset.jsonl
"""

import json
import argparse
import os
import sys
from glob import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generate_dataset'))

from merge_datasets import load_primevul, load_sven


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation dataset from test splits.")
    parser.add_argument("--primevul_test_path", type=str, required=True,
                        help="Path to PrimeVul test paired .jsonl file")
    parser.add_argument("--primevul_valid_path", type=str, required=True,
                        help="Path to PrimeVul valid paired .jsonl file")
    parser.add_argument("--sven_val_path", type=str, required=True,
                        help="Path pattern for SVEN val parquet files (supports glob)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save merged eval dataset")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_samples = []
    
    # Load PrimeVul test
    print("=" * 60)
    print("Loading PrimeVul test...")
    primevul_test = load_primevul(args.primevul_test_path)
    # Tag with split
    for s in primevul_test:
        s['split'] = 'primevul_test'
    all_samples.extend(primevul_test)
    print(f"  Loaded {len(primevul_test)} pairs from PrimeVul test")
    
    # Load PrimeVul valid
    print("\nLoading PrimeVul valid...")
    primevul_valid = load_primevul(args.primevul_valid_path)
    for s in primevul_valid:
        s['split'] = 'primevul_valid'
    all_samples.extend(primevul_valid)
    print(f"  Loaded {len(primevul_valid)} pairs from PrimeVul valid")
    
    # Load SVEN val
    print("\nLoading SVEN val...")
    sven_files = glob(args.sven_val_path)
    if sven_files:
        sven_val = load_sven(sven_files)
        for s in sven_val:
            s['split'] = 'sven_val'
        all_samples.extend(sven_val)
        print(f"  Loaded {len(sven_val)} pairs from SVEN val")
    else:
        print(f"  Warning: No files found matching {args.sven_val_path}")
    
    # Filter samples where vuln_func == patched_func (data quality)
    before_filter = len(all_samples)
    all_samples = [s for s in all_samples if s['vuln_func'] != s['patched_func']]
    filtered_count = before_filter - len(all_samples)
    if filtered_count > 0:
        print(f"\nRemoved {filtered_count} samples with identical vuln/patched functions")
    
    # Stats
    print("\n" + "=" * 60)
    print(f"Total evaluation samples: {len(all_samples)} pairs")
    print("\nBreakdown by split:")
    from collections import Counter
    split_counts = Counter(s['split'] for s in all_samples)
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count} pairs")
    
    print("\nBreakdown by source:")
    source_counts = Counter(s['source'] for s in all_samples)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} pairs")
    
    # Write output
    print(f"\nWriting to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print("Done!")


if __name__ == "__main__":
    main()
