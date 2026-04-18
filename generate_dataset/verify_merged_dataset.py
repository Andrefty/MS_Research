#!/usr/bin/env python3
"""
verify_merged_dataset.py - Verify that merged_train.jsonl has no OOB or misindexed line numbers.

Run this AFTER merge_datasets.py but BEFORE the expensive teacher regeneration
to confirm the fix worked.

Usage:
    python verify_merged_dataset.py --input_file Research_merged_dataset/merged_train.jsonl
    python verify_merged_dataset.py --input_file Research_merged_dataset/merged_train.jsonl --verbose
"""

import json
import argparse
from collections import Counter, defaultdict


def verify_sample(sample: dict) -> list[dict]:
    """
    Verify a single sample's deleted_lines and added_lines are correctly indexed.
    
    Returns list of issues found (empty if clean).
    """
    issues = []
    source = sample.get('source', 'unknown')
    commit_id = sample.get('commit_id', '')
    
    for field, func_key in [('deleted_lines', 'vuln_func'), ('added_lines', 'patched_func')]:
        lines = sample.get(field, [])
        func_body = sample.get(func_key, '')
        
        if not lines or not func_body:
            continue
        
        func_lines = func_body.splitlines()
        func_line_count = len(func_lines)
        
        for entry in lines:
            ln = entry.get('line_no', 0)
            text = entry.get('text', '').strip()
            
            # OOB check
            if ln > func_line_count or ln < 1:
                issues.append({
                    'type': 'oob',
                    'field': field,
                    'line_no': ln,
                    'func_lines': func_line_count,
                    'source': source,
                    'commit_id': commit_id,
                    'text': text[:80]
                })
            # Misindexing check (only for lines with text and within bounds)
            elif text:
                actual_text = func_lines[ln - 1].strip()
                if actual_text != text:
                    # Allow substring matches (some texts are truncated)
                    if text not in actual_text and actual_text not in text:
                        issues.append({
                            'type': 'misindexed',
                            'field': field,
                            'line_no': ln,
                            'expected': text[:80],
                            'actual': actual_text[:80],
                            'source': source,
                            'commit_id': commit_id
                        })
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify merged dataset for OOB/misindexed line numbers")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to merged_train.jsonl")
    parser.add_argument("--verbose", action="store_true",
                        help="Show individual problematic samples")
    parser.add_argument("--max_examples", type=int, default=10,
                        help="Max examples to show per issue type in verbose mode")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.input_file}...")
    samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    
    # Source distribution
    source_counts = Counter(s['source'] for s in samples)
    print(f"\nSource distribution:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")
    
    # Verify all samples
    total_issues = []
    issue_counts = Counter()
    source_issue_counts = defaultdict(Counter)
    samples_with_issues = 0
    
    for sample in samples:
        issues = verify_sample(sample)
        if issues:
            samples_with_issues += 1
            total_issues.extend(issues)
            for issue in issues:
                issue_counts[issue['type']] += 1
                source_issue_counts[issue['source']][issue['type']] += 1
    
    # Results
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION RESULTS")
    print(f"{'=' * 60}")
    
    if not total_issues:
        print(f"\n✅ ALL CLEAR — 0 issues found across {len(samples)} samples")
        print(f"   No OOB line numbers, no misindexed lines")
    else:
        print(f"\n❌ ISSUES FOUND")
        print(f"   Samples with issues: {samples_with_issues} ({100*samples_with_issues/len(samples):.1f}%)")
        print(f"   Total issues: {len(total_issues)}")
        print(f"     OOB:        {issue_counts.get('oob', 0)}")
        print(f"     Misindexed: {issue_counts.get('misindexed', 0)}")
        
        print(f"\n   By source:")
        for src in sorted(source_issue_counts.keys()):
            counts = source_issue_counts[src]
            print(f"     {src}: OOB={counts.get('oob', 0)}, misindexed={counts.get('misindexed', 0)}")
        
        if args.verbose:
            print(f"\n--- OOB examples ---")
            oob_examples = [i for i in total_issues if i['type'] == 'oob'][:args.max_examples]
            for ex in oob_examples:
                print(f"  [{ex['source']}] commit={ex['commit_id'][:12]}... "
                      f"{ex['field']} line_no={ex['line_no']} > func_lines={ex['func_lines']}")
                print(f"    text: {ex['text']}")
            
            print(f"\n--- Misindexed examples ---")
            mis_examples = [i for i in total_issues if i['type'] == 'misindexed'][:args.max_examples]
            for ex in mis_examples:
                print(f"  [{ex['source']}] commit={ex['commit_id'][:12]}... "
                      f"{ex['field']} line_no={ex['line_no']}")
                print(f"    expected: {ex['expected']}")
                print(f"    actual:   {ex['actual']}")
    
    # Also check: samples with deleted_lines that have no entries
    has_deleted = sum(1 for s in samples if s.get('deleted_lines'))
    no_deleted_vuln = sum(1 for s in samples if not s.get('deleted_lines'))
    print(f"\n--- Additional stats ---")
    print(f"  Samples with deleted_lines: {has_deleted}")
    print(f"  Samples without deleted_lines: {no_deleted_vuln}")
    
    # Check new fields
    has_func_name = sum(1 for s in samples if s.get('func_name'))
    has_filepath = sum(1 for s in samples if s.get('filepath'))
    print(f"  Samples with func_name: {has_func_name}")
    print(f"  Samples with filepath: {has_filepath}")
    
    return 0 if not total_issues else 1


if __name__ == "__main__":
    exit(main())
