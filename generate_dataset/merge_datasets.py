#!/usr/bin/env python3
"""
merge_datasets.py - Merge PrimeVul, SVEN, and SecVulEval into a unified dataset.

Usage:
    python merge_datasets.py \
        --primevul_path PrimeVul-v0.1-hf/paired/primevul_train_paired.jsonl \
        --sven_path sven/data/train-*.parquet \
        --secvuleval_path SecVulEval/data/train-*.parquet \
        --output_path merged_dataset/merged_train.jsonl
"""

import json
import argparse
import os
import difflib
import re
from glob import glob
from collections import defaultdict
from typing import Optional
import pandas as pd
from tqdm import tqdm


# --- PrimeVul Loading ---

def extract_changed_lines_difflib(vuln_func: str, patched_func: str) -> tuple[list[dict], list[dict]]:
    """
    Use difflib to compute deleted and added lines between vuln and patched versions.
    Returns (deleted_lines, added_lines) where each is a list of {"line_no": int, "text": str}.
    """
    vuln_lines = vuln_func.splitlines()
    patched_lines = patched_func.splitlines()
    
    s = difflib.SequenceMatcher(None, vuln_lines, patched_lines)
    
    deleted_lines = []
    added_lines = []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'delete' or tag == 'replace':
            # Lines in vuln_lines[i1:i2] were deleted/replaced
            for k in range(i1, i2):
                deleted_lines.append({
                    "line_no": k + 1,  # 1-indexed
                    "text": vuln_lines[k]
                })
        if tag == 'insert' or tag == 'replace':
            # Lines in patched_lines[j1:j2] were added/replaced
            for k in range(j1, j2):
                added_lines.append({
                    "line_no": k + 1,  # 1-indexed
                    "text": patched_lines[k]
                })
    
    return deleted_lines, added_lines


def load_primevul(path: str) -> list[dict]:
    """Load PrimeVul paired dataset and transform to unified schema."""
    samples = []
    
    # Load all samples
    raw_samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line))
    
    # Group by commit_id
    by_commit = defaultdict(list)
    for sample in raw_samples:
        by_commit[sample['commit_id']].append(sample)
    
    # Process pairs
    for commit_id, items in tqdm(by_commit.items(), desc="Processing PrimeVul"):
        if len(items) != 2:
            continue
        
        targets = [item['target'] for item in items]
        if not (targets.count(0) == 1 and targets.count(1) == 1):
            continue
        
        vuln_sample = next(item for item in items if item['target'] == 1)
        patched_sample = next(item for item in items if item['target'] == 0)
        
        # Extract line changes
        deleted_lines, added_lines = extract_changed_lines_difflib(
            vuln_sample['func'], patched_sample['func']
        )
        
        # Get project from project_url if available
        project = vuln_sample.get('project', '')
        project_url = vuln_sample.get('project_url', '')
        
        samples.append({
            "source": "primevul",
            "pair_id": f"primevul_{project}_{commit_id}",
            "commit_id": commit_id,
            "project": project,
            "project_url": project_url,
            "vuln_func": vuln_sample['func'],
            "patched_func": patched_sample['func'],
            "cve": vuln_sample.get('cve'),
            "cve_desc": vuln_sample.get('cve_desc'),
            "cwe": vuln_sample.get('cwe') if isinstance(vuln_sample.get('cwe'), list) else [vuln_sample.get('cwe')] if vuln_sample.get('cwe') else None,
            "deleted_lines": deleted_lines,
            "added_lines": added_lines,
            "source_row_idx": vuln_sample.get('idx')
        })
    
    return samples


# --- SVEN Loading ---

def parse_sven_commit_link(commit_link: str) -> tuple[str, str]:
    """
    Extract project and commit_id from SVEN commit_link.
    Example: "github.com/abrt/libreport/commit/239c4f7..." -> ("abrt/libreport", "239c4f7...")
    """
    # Pattern: github.com/{org}/{repo}/commit/{hash}
    match = re.search(r'github\.com/([^/]+/[^/]+)/commit/([a-f0-9]+)', commit_link)
    if match:
        return match.group(1), match.group(2)
    return "", commit_link


def extract_sven_line_changes(line_changes: dict) -> tuple[list[dict], list[dict]]:
    """
    Extract deleted and added lines from SVEN's line_changes structure.
    Keeps all fields: line_no, char_start, char_end, text (renamed from 'line').
    """
    deleted_lines = []
    added_lines = []
    
    if 'deleted' in line_changes and line_changes['deleted'] is not None:
        for item in line_changes['deleted']:
            deleted_lines.append({
                "line_no": item['line_no'],
                "char_start": item.get('char_start'),
                "char_end": item.get('char_end'),
                "text": item['line'].rstrip('\n\r')
            })
    
    if 'added' in line_changes and line_changes['added'] is not None:
        for item in line_changes['added']:
            added_lines.append({
                "line_no": item['line_no'],
                "char_start": item.get('char_start'),
                "char_end": item.get('char_end'),
                "text": item['line'].rstrip('\n\r')
            })
    
    return deleted_lines, added_lines


def load_sven(paths: list[str]) -> list[dict]:
    """Load SVEN dataset and transform to unified schema."""
    samples = []
    
    for path in paths:
        df = pd.read_parquet(path)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing SVEN {os.path.basename(path)}"):
            project, commit_id = parse_sven_commit_link(row['commit_link'])
            deleted_lines, added_lines = extract_sven_line_changes(row['line_changes'])
            
            # Map vul_type to CWE format
            vul_type = row.get('vul_type', '')
            cwe = [vul_type.upper()] if vul_type else None
            
            samples.append({
                "source": "sven",
                "pair_id": f"sven_{project}_{commit_id}",
                "commit_id": commit_id,
                "project": project,
                "project_url": f"https://{row['commit_link'].split('/commit/')[0]}" if '/commit/' in row['commit_link'] else None,
                "vuln_func": row['func_src_before'],
                "patched_func": row['func_src_after'],
                "cve": None,  # SVEN doesn't have CVE
                "cve_desc": None,
                "cwe": cwe,
                "deleted_lines": deleted_lines,
                "added_lines": added_lines,
                "source_row_idx": idx
            })
    
    return samples


# --- SecVulEval Loading ---

def parse_secvuleval_changed_lines(changed_lines_str: str) -> list[dict]:
    """
    Parse SecVulEval's changed_lines JSON string.
    Format: '[[line_no, "text"], ...]'
    """
    if not changed_lines_str or changed_lines_str in ['None', 'null', '[]']:
        return []
    
    try:
        parsed = json.loads(changed_lines_str)
        return [{"line_no": item[0], "text": item[1].strip()} for item in parsed]
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def load_secvuleval(paths: list[str]) -> list[dict]:
    """Load SecVulEval dataset and transform to unified schema."""
    samples = []
    
    for path in paths:
        df = pd.read_parquet(path)
        
        # Filter to vulnerable samples with valid pairs
        # fixed_func_idx must not be null AND must be different from idx
        valid_vuln = df[
            (df['is_vulnerable'] == True) & 
            (df['fixed_func_idx'].notna()) & 
            (df['idx'] != df['fixed_func_idx'])
        ]
        
        for _, vuln_row in tqdm(valid_vuln.iterrows(), total=len(valid_vuln), desc=f"Processing SecVulEval {os.path.basename(path)}"):
            # Get the paired patched sample
            fixed_idx = int(vuln_row['fixed_func_idx'])
            patched_rows = df[df['idx'] == fixed_idx]
            
            if len(patched_rows) == 0:
                continue
            
            patched_row = patched_rows.iloc[0]
            
            # Parse line changes
            # Vulnerable sample has deleted lines, patched sample has added lines
            deleted_lines = parse_secvuleval_changed_lines(vuln_row['changed_lines'])
            added_lines = parse_secvuleval_changed_lines(patched_row['changed_lines'])
            
            # Extract project info
            project = vuln_row.get('project', '')
            commit_id = vuln_row.get('commit_id', '')
            
            # Get CVE/CWE as lists (handle numpy arrays)
            cve_list = vuln_row.get('cve_list', [])
            cwe_list = vuln_row.get('cwe_list', [])
            
            # Convert to Python list if numpy array
            if hasattr(cve_list, 'tolist'):
                cve_list = cve_list.tolist()
            if hasattr(cwe_list, 'tolist'):
                cwe_list = cwe_list.tolist()
            
            # Handle None/NaN
            if cve_list is None or (hasattr(cve_list, '__len__') and len(cve_list) == 0):
                cve_list = []
            if cwe_list is None or (hasattr(cwe_list, '__len__') and len(cwe_list) == 0):
                cwe_list = []
            
            # Extract Explanation from context field as cve_desc
            cve_desc = None
            context = vuln_row.get('context')
            if context is not None and isinstance(context, dict):
                explanations = context.get('Explanation', [])
                # Convert numpy array to list if needed
                if hasattr(explanations, 'tolist'):
                    explanations = explanations.tolist()
                if explanations is not None and len(explanations) > 0:
                    cve_desc = ' '.join(str(e) for e in explanations)
            
            samples.append({
                "source": "secvuleval",
                "pair_id": f"secvuleval_{project}_{commit_id}",
                "commit_id": commit_id,
                "project": project,
                "project_url": vuln_row.get('project_url', ''),
                "vuln_func": vuln_row['func_body'],
                "patched_func": patched_row['func_body'],
                "cve": cve_list[0] if len(cve_list) > 0 else None,
                "cve_desc": cve_desc,
                "cwe": list(cwe_list) if len(cwe_list) > 0 else None,
                "deleted_lines": deleted_lines,
                "added_lines": added_lines,
                "source_row_idx": int(vuln_row['idx'])
            })
    
    return samples


# --- Deduplication ---

def deduplicate_by_commit_id(samples: list[dict]) -> list[dict]:
    """
    Deduplicate samples by commit_id only (commit hashes are globally unique).
    Prefer SVEN (most accurate) > PrimeVul (has CVE desc) > SecVulEval.
    BUT: Skip samples where vuln_func == patched_func (data quality issue).
    Also merges CVE description from any source if the preferred source lacks it.
    """
    seen = {}
    # SVEN is most accurate, prioritize it
    source_priority = {"sven": 0, "primevul": 1, "secvuleval": 2}
    
    def is_valid_pair(sample):
        """Check if sample has different vuln and patched functions."""
        return sample['vuln_func'] != sample['patched_func']
    
    for sample in samples:
        key = sample['commit_id'].lower()
        
        if key not in seen:
            seen[key] = sample
        else:
            existing = seen[key]
            existing_priority = source_priority.get(existing['source'], 99)
            sample_priority = source_priority.get(sample['source'], 99)
            
            existing_valid = is_valid_pair(existing)
            sample_valid = is_valid_pair(sample)
            
            # Prefer valid samples (vuln != patched) over invalid ones
            should_replace = False
            if sample_valid and not existing_valid:
                # New sample is valid, existing is not - replace
                should_replace = True
            elif not sample_valid and existing_valid:
                # Existing is valid, new is not - keep existing
                should_replace = False
            elif sample_priority < existing_priority:
                # Both valid or both invalid - use source priority
                should_replace = True
            
            if should_replace:
                # Replace but try to keep cve_desc from existing
                if not sample.get('cve_desc') and existing.get('cve_desc'):
                    sample['cve_desc'] = existing['cve_desc']
                if not sample.get('cve') and existing.get('cve'):
                    sample['cve'] = existing['cve']
                seen[key] = sample
            else:
                # Keep existing, but try to get cve_desc from new sample if missing
                if not existing.get('cve_desc') and sample.get('cve_desc'):
                    existing['cve_desc'] = sample['cve_desc']
                if not existing.get('cve') and sample.get('cve'):
                    existing['cve'] = sample['cve']
    
    return list(seen.values())


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Merge vulnerability datasets into unified format.")
    parser.add_argument("--primevul_path", type=str, required=True,
                        help="Path to PrimeVul paired .jsonl file")
    parser.add_argument("--sven_path", type=str, required=True,
                        help="Path pattern for SVEN parquet files (supports glob)")
    parser.add_argument("--secvuleval_path", type=str, required=True,
                        help="Path pattern for SecVulEval parquet files (supports glob)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save merged .jsonl dataset")
    parser.add_argument("--skip_dedup", action="store_true",
                        help="Skip deduplication step")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load datasets
    print("=" * 60)
    print("Loading PrimeVul...")
    primevul_samples = load_primevul(args.primevul_path)
    print(f"  Loaded {len(primevul_samples)} pairs from PrimeVul")
    
    print("\nLoading SVEN...")
    sven_files = glob(args.sven_path)
    sven_samples = load_sven(sven_files) if sven_files else []
    print(f"  Loaded {len(sven_samples)} pairs from SVEN")
    
    print("\nLoading SecVulEval...")
    secvuleval_files = glob(args.secvuleval_path)
    secvuleval_samples = load_secvuleval(secvuleval_files) if secvuleval_files else []
    print(f"  Loaded {len(secvuleval_samples)} pairs from SecVulEval")
    
    # Merge
    all_samples = primevul_samples + sven_samples + secvuleval_samples
    print(f"\nTotal before dedup: {len(all_samples)}")
    
    # Deduplicate
    if not args.skip_dedup:
        print("\nDeduplicating by commit_id...")
        all_samples = deduplicate_by_commit_id(all_samples)
        print(f"Total after dedup: {len(all_samples)}")
    
    # Final filter: remove samples where vuln_func == patched_func (no valid source exists)
    before_filter = len(all_samples)
    all_samples = [s for s in all_samples if s['vuln_func'] != s['patched_func']]
    filtered_count = before_filter - len(all_samples)
    if filtered_count > 0:
        print(f"Removed {filtered_count} samples with identical vuln/patched functions")
    print(f"Final count: {len(all_samples)}")
    
    # Stats
    print("\n" + "=" * 60)
    print("Final dataset statistics:")
    source_counts = defaultdict(int)
    for s in all_samples:
        source_counts[s['source']] += 1
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} pairs")
    
    # Write output
    print(f"\nWriting to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print("Done!")
    
    # Show sample from each source
    print("\n" + "=" * 60)
    print("Sample entries:")
    for source in ['primevul', 'sven', 'secvuleval']:
        source_samples = [s for s in all_samples if s['source'] == source]
        if source_samples:
            sample = source_samples[0]
            print(f"\n--- {source} ---")
            print(f"  pair_id: {sample['pair_id']}")
            print(f"  project: {sample['project']}")
            print(f"  cve: {sample['cve']}")
            print(f"  cwe: {sample['cwe']}")
            print(f"  deleted_lines: {sample['deleted_lines'][:2]}..." if sample['deleted_lines'] else "  deleted_lines: []")
            print(f"  added_lines: {sample['added_lines'][:2]}..." if sample['added_lines'] else "  added_lines: []")


if __name__ == "__main__":
    main()
