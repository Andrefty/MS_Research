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

from merge_datasets import load_primevul, load_sven, deduplicate_by_function, build_commit_metadata_pool, funcs_are_effectively_identical


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
    
    print(f"\nTotal before dedup: {len(all_samples)}")
    
    # Build metadata pool and deduplicate (same logic as training pipeline)
    print("\nBuilding commit-level metadata pool...")
    metadata_pool = build_commit_metadata_pool(all_samples)
    print(f"  Pooled metadata for {len(metadata_pool)} unique commits")
    
    print("\nDeduplicating by (commit_id, func_name, vuln_hash)...")
    all_samples = deduplicate_by_function(all_samples)
    print(f"Total after dedup: {len(all_samples)}")
    
    # Enrich with pooled metadata
    print("\nEnriching entries with pooled commit metadata...")
    for s in all_samples:
        cid = s['commit_id'].lower()
        if cid in metadata_pool:
            pooled = metadata_pool[cid]
            s['cve'] = pooled['cve']
            s['cwe'] = pooled['cwe']
            if pooled['cve_desc']:
                s['cve_desc'] = pooled['cve_desc']
            if pooled['commit_message'] and len(pooled['commit_message']) > len(s.get('commit_message', '')):
                s['commit_message'] = pooled['commit_message']
    
    # Filter samples where vuln_func is effectively identical to patched_func
    # (exact match OR only trailing whitespace differences — not real fixes)
    before_filter = len(all_samples)
    all_samples = [s for s in all_samples if not funcs_are_effectively_identical(s['vuln_func'], s['patched_func'])]
    filtered_count = before_filter - len(all_samples)
    if filtered_count > 0:
        print(f"\nRemoved {filtered_count} samples with identical/whitespace-only vuln/patched functions")
    
    # Stats
    from collections import Counter
    print("\n" + "=" * 60)
    print(f"Total evaluation samples: {len(all_samples)} pairs")
    print("\nBreakdown by split:")
    split_counts = Counter(s['split'] for s in all_samples)
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count} pairs")
    
    print("\nBreakdown by source:")
    source_counts = Counter(s['source'] for s in all_samples)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} pairs")
    
    # Field coverage
    has_func_name = sum(1 for s in all_samples if s.get('func_name'))
    has_vuln_hash = sum(1 for s in all_samples if s.get('vuln_hash'))
    has_cve_desc = sum(1 for s in all_samples if s.get('cve_desc'))
    has_commit_msg = sum(1 for s in all_samples if s.get('commit_message'))
    print(f"\n  Field coverage:")
    print(f"    func_name: {has_func_name}/{len(all_samples)}")
    print(f"    vuln_hash: {has_vuln_hash}/{len(all_samples)}")
    print(f"    cve_desc:  {has_cve_desc}/{len(all_samples)}")
    print(f"    commit_message: {has_commit_msg}/{len(all_samples)}")
    
    # Write output
    print(f"\nWriting to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print("Done!")


if __name__ == "__main__":
    main()
