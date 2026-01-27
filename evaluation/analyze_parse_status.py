#!/usr/bin/env python3
"""
Analyze datasets for parse status and extract non-valid samples.

This script analyzes JSONL datasets (generated datasets, SFT datasets) and:
1. Reports parse status breakdown (VALID, REGEX_FALLBACK, INCOMPLETE, etc.)
2. Saves full non-valid samples to a separate file for review

Complements analyze_response_schema.py by adding the ability to extract
and save the actual non-valid samples, not just statistics.

Usage:
    python analyze_parse_status.py --input dataset.jsonl [--save_nonvalid output.jsonl]
    
Examples:
    # Analyze and report status breakdown
    python analyze_parse_status.py --input training_grpo/sft_dataset_train.jsonl
    
    # Analyze and save non-valid samples
    python analyze_parse_status.py --input training_grpo/sft_dataset_train.jsonl \
        --save_nonvalid analysis/nonvalid_samples.jsonl
"""
import argparse
import json
import os
import sys

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.response_parser import parse_model_response


def get_response_field(data: dict) -> str:
    """Extract response text from data, checking multiple possible field names."""
    return (data.get('response') or 
            data.get('generated_response') or 
            data.get('chosen') or 
            data.get('rejected') or '')


def analyze_file(filepath: str, save_path: str = None, verbose: bool = False):
    """
    Analyze a JSONL file for parse status distribution.
    
    Args:
        filepath: Path to input JSONL file
        save_path: Optional path to save non-valid samples
        verbose: If True, print sample previews
    
    Returns:
        Dict with analysis results
    """
    status_counts = {}
    total = 0
    non_valid_samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            total += 1
            data = json.loads(line)
            
            response = get_response_field(data)
            result = parse_model_response(response)
            status = result.status
            
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Track non-VALID samples with full data
            if status != "VALID":
                # Add parse metadata to the sample
                data['_parse_status'] = result.status
                data['_parse_classification'] = result.classification
                data['_parse_lines'] = result.vulnerable_lines
                data['_line_number'] = i + 1
                non_valid_samples.append(data)
    
    # Save non-valid samples if requested
    if save_path and non_valid_samples:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            for sample in non_valid_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"💾 Saved {len(non_valid_samples)} full non-valid samples to: {save_path}")
    
    return {
        'total': total,
        'status_counts': status_counts,
        'non_valid_count': len(non_valid_samples),
        'non_valid_samples': non_valid_samples if verbose else None
    }


def print_report(filepath: str, results: dict, verbose: bool = False):
    """Print formatted analysis report."""
    print(f"\n{'=' * 70}")
    print(f"📁 {filepath}")
    print(f"{'=' * 70}")
    print(f"Total samples: {results['total']}")
    print(f"\nStatus breakdown:")
    
    for status, count in sorted(results['status_counts'].items(), key=lambda x: -x[1]):
        pct = count / results['total'] * 100 if results['total'] > 0 else 0
        marker = "✅" if status == "VALID" else "⚠️"
        print(f"  {marker} {status}: {count} ({pct:.2f}%)")
    
    if verbose and results.get('non_valid_samples'):
        print(f"\nFirst 5 non-VALID samples:")
        for sample in results['non_valid_samples'][:5]:
            print(f"  Line {sample['_line_number']}: {sample['_parse_status']}")
            print(f"    commit: {sample.get('commit_id', 'N/A')[:12]}...")
            print(f"    Classification: {sample['_parse_classification']}")
            print(f"    GT is_vulnerable: {sample.get('is_vulnerable')}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze dataset parse status and extract non-valid samples.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input JSONL file to analyze')
    parser.add_argument('--save_nonvalid', '-s',
                        help='Path to save non-valid samples (full data)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show sample previews in output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)
    
    results = analyze_file(args.input, args.save_nonvalid, args.verbose)
    print_report(args.input, results, args.verbose)
    
    # Exit with non-zero if there are non-valid samples
    if results['non_valid_count'] > 0:
        print(f"\n⚠️  Found {results['non_valid_count']} non-valid samples")
        sys.exit(0)  # Still success, just informative
    else:
        print(f"\n✅ All samples have VALID parse status")


if __name__ == "__main__":
    main()
