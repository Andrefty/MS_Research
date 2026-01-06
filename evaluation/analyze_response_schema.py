#!/usr/bin/env python3
"""
Analyze GRPO evaluation/training responses for JSON schema validity.
Validates that responses contain expected JSON format.

Works with:
- Evaluation responses (from run_eval.py)
- GRPO training completions (from reward_function.py logging)
"""

import json
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.response_parser import parse_model_response as _parse, ParseResult


def parse_model_response(response_text: str):
    """
    Parse model response to extract classification and vulnerable lines.
    Returns (classification, vulnerable_lines, parse_status)
    """
    result = _parse(response_text)
    return result.classification, result.vulnerable_lines, result.status


def analyze_responses(input_file: str, prompt_type_filter: str = None):
    """Analyze responses from evaluation JSONL file."""
    
    results = {
        "total": 0,
        "by_prompt_type": {},
        "parse_status_counts": Counter(),
        "classification_counts": Counter(),
        "samples_by_status": {},
    }
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            prompt_type = data.get('prompt_type', 'unknown')
            
            # Filter by prompt type if specified
            if prompt_type_filter and prompt_type != prompt_type_filter:
                continue
            
            results["total"] += 1
            
            if prompt_type not in results["by_prompt_type"]:
                results["by_prompt_type"][prompt_type] = {
                    "total": 0,
                    "valid": 0,
                    "parse_status": Counter(),
                }
            
            results["by_prompt_type"][prompt_type]["total"] += 1
            
            response = data.get('response', '')
            classification, lines, status = parse_model_response(response)
            
            results["parse_status_counts"][status] += 1
            results["by_prompt_type"][prompt_type]["parse_status"][status] += 1
            
            if status == "VALID":
                results["by_prompt_type"][prompt_type]["valid"] += 1
                results["classification_counts"][classification] += 1
            
            # Store sample for each status type
            if status not in results["samples_by_status"]:
                results["samples_by_status"][status] = []
            if len(results["samples_by_status"][status]) < 3:
                results["samples_by_status"][status].append({
                    "commit_id": data.get('commit_id', '')[:20],
                    "is_vulnerable": data.get('is_vulnerable'),
                    "response_preview": response[:200] if response else "",
                    "response_end": response[-200:] if len(response) > 200 else "",
                })
    
    return results


def print_report(results: dict, verbose: bool = False):
    """Print analysis report."""
    print("=" * 60)
    print("GRPO RESPONSE VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nTotal responses analyzed: {results['total']}")
    
    print("\n--- By Prompt Type ---")
    for prompt_type, stats in results["by_prompt_type"].items():
        valid_pct = (stats["valid"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"\n{prompt_type}:")
        print(f"  Total: {stats['total']}, Valid JSON: {stats['valid']} ({valid_pct:.1f}%)")
        print(f"  Parse status breakdown:")
        for status, count in stats["parse_status"].most_common():
            pct = count / stats["total"] * 100
            print(f"    {status}: {count} ({pct:.1f}%)")
    
    print("\n--- Classification Distribution (valid responses only) ---")
    for cls, count in results["classification_counts"].most_common():
        print(f"  {cls}: {count}")
    
    if verbose:
        print("\n--- Sample Responses by Status ---")
        for status, samples in results["samples_by_status"].items():
            print(f"\n{status}:")
            for i, sample in enumerate(samples[:2]):
                print(f"  Sample {i+1}: commit={sample['commit_id']}, is_vuln={sample['is_vulnerable']}")
                print(f"    Start: {sample['response_preview'][:100]}...")
                print(f"    End: ...{sample['response_end'][-100:]}")


def main():
    parser = argparse.ArgumentParser(description="Validate GRPO evaluation responses")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to eval_responses.jsonl")
    parser.add_argument("--prompt-type", "-p", type=str, default=None,
                        choices=["training", "std_cls"],
                        help="Filter by prompt type")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show sample responses")
    parser.add_argument("--output-json", "-o", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--extract-broken", "-b", type=str, default=None,
                        help="Extract all non-VALID/non-SKIPPED samples to JSONL file")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    results = analyze_responses(args.input, args.prompt_type)
    print_report(results, args.verbose)
    
    if args.output_json:
        # Convert Counter objects for JSON serialization
        output = {
            "total": results["total"],
            "by_prompt_type": {
                k: {
                    "total": v["total"],
                    "valid": v["valid"],
                    "parse_status": dict(v["parse_status"])
                }
                for k, v in results["by_prompt_type"].items()
            },
            "parse_status_counts": dict(results["parse_status_counts"]),
            "classification_counts": dict(results["classification_counts"]),
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    if args.extract_broken:
        # Extract all non-VALID/non-SKIPPED samples
        broken_samples = []
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                prompt_type = data.get('prompt_type', 'unknown')
                if args.prompt_type and prompt_type != args.prompt_type:
                    continue
                
                response = data.get('response', '')
                classification, lines, status = parse_model_response(response)
                
                # Extract if not VALID and not SKIPPED
                if status not in ["VALID", "SKIPPED"]:
                    broken_samples.append({
                        "commit_id": data.get('commit_id', ''),
                        "is_vulnerable": data.get('is_vulnerable'),
                        "prompt_type": prompt_type,
                        "subset": data.get('subset', ''),
                        "parse_status": status,
                        "response_length": len(response) if response else 0,
                        "response": response,
                    })
        
        with open(args.extract_broken, 'w', encoding='utf-8') as f:
            for sample in broken_samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"\nExtracted {len(broken_samples)} broken samples to: {args.extract_broken}")


if __name__ == "__main__":
    main()

