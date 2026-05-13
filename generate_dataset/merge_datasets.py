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
import hashlib
import re
from glob import glob
from collections import defaultdict, Counter
from typing import Optional
import pandas as pd
from tqdm import tqdm


# --- Utility functions for function-level deduplication ---

def normalize_code(code: str) -> str:
    """Normalize code for hashing AND diffing: strip each line, remove empty, lowercase."""
    lines = code.splitlines()
    stripped = [l.strip() for l in lines if l.strip()]
    return '\n'.join(stripped).lower()


def compute_vuln_hash(code: str) -> str:
    """Compute SHA-256 hash of normalized code."""
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()


def extract_func_name_from_body(func_body: str) -> str:
    """
    Extract function name from C/C++ function body.
    Parses the first line looking for the function name before '('.
    """
    first_line = func_body.split('\n')[0].strip()
    m = re.match(r'(?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:\w+[\s*]+)*?(\w+)\s*\(', first_line)
    if m:
        name = m.group(1)
        # Filter out keywords that could be falsely matched
        if name not in ['if', 'else', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof', 'do']:
            return name
    return first_line[:40]


def diff_file_smart(file_a: str, file_b: str, proj_a: str = '', proj_b: str = '') -> bool:
    """
    Smart file comparison for deduplication.
    
    Handles:
    - PrimeVul storing just filenames (e.g. 'tx.c') vs SecVulEval full paths
    - Same basename but different directories (arch/x86/mm/fault.c vs arch/mips/mm/fault.c)
    - Fork detection: same path but different project = different file
    
    Returns True if files are DIFFERENT, False if same or can't tell.
    """
    fa = file_a if file_a and str(file_a) != 'None' else ''
    fb = file_b if file_b and str(file_b) != 'None' else ''
    if fa == '' or fb == '':
        return False  # Can't tell → treat as same
    # If one is just a basename (no '/'), fall back to basename comparison
    if '/' not in fa or '/' not in fb:
        return fa.split('/')[-1] != fb.split('/')[-1]
    # Both have directory structure → compare full paths
    # But also check: same path + different project = fork (e.g. iortcw vs OpenJK)
    if fa == fb and proj_a and proj_b and proj_a != proj_b:
        return True  # Same path, different project → fork
    return fa != fb


def compute_normalized_diff(code_a: str, code_b: str) -> tuple[int, int, float]:
    """
    Compute diff metrics on NORMALIZED code to avoid whitespace-only false positives.
    
    Returns (added, removed, similarity).
    """
    a_norm = normalize_code(code_a).splitlines()
    b_norm = normalize_code(code_b).splitlines()
    diff = list(difflib.unified_diff(a_norm, b_norm, lineterm=''))
    added = sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))
    removed = sum(1 for d in diff if d.startswith('-') and not d.startswith('---'))
    sim = difflib.SequenceMatcher(None, '\n'.join(a_norm), '\n'.join(b_norm)).ratio()
    return added, removed, sim


def funcs_are_effectively_identical(vuln_func: str, patched_func: str) -> bool:
    """
    Check if two functions are effectively identical, ignoring trailing whitespace.
    
    Trailing whitespace is never semantically meaningful in any programming language
    (even Python only cares about leading/indentation whitespace). This catches
    entries where the only "fix" is removing trailing spaces/tabs — which are
    dataset artifacts, not real vulnerability fixes.
    """
    vuln_stripped = '\n'.join(line.rstrip() for line in vuln_func.splitlines())
    patch_stripped = '\n'.join(line.rstrip() for line in patched_func.splitlines())
    return vuln_stripped == patch_stripped


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
    """
    Load PrimeVul paired dataset and transform to unified schema.
    
    Handles multi-pair commits: groups by func_name within each commit,
    then pairs vuln↔patched by best difflib similarity.
    """
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
    
    skipped_commits = 0
    multi_func_commits = 0
    
    for commit_id, items in tqdm(by_commit.items(), desc="Processing PrimeVul"):
        # Separate vuln and patched
        vuln_items = [it for it in items if it['target'] == 1]
        patched_items = [it for it in items if it['target'] == 0]
        
        if not vuln_items or not patched_items:
            skipped_commits += 1
            continue
        
        # Group by extracted func_name
        vuln_by_func = defaultdict(list)
        for it in vuln_items:
            fn = extract_func_name_from_body(it['func'])
            vuln_by_func[fn].append(it)
        
        patched_by_func = defaultdict(list)
        for it in patched_items:
            fn = extract_func_name_from_body(it['func'])
            patched_by_func[fn].append(it)
        
        if len(vuln_by_func) > 1 or len(patched_by_func) > 1:
            multi_func_commits += 1
        
        # For each func_name that appears in BOTH vuln and patched, pair 1:1
        all_func_names = set(vuln_by_func.keys()) | set(patched_by_func.keys())
        
        for fn in all_func_names:
            fn_vulns = vuln_by_func.get(fn, [])
            fn_patches = patched_by_func.get(fn, [])
            
            if not fn_vulns or not fn_patches:
                continue
            
            # Pair by best similarity (greedy 1:1 matching)
            used_patch = set()
            pairs = []
            for vi in fn_vulns:
                best_sim = -1
                best_pi_idx = -1
                for pi_idx, pi in enumerate(fn_patches):
                    if pi_idx in used_patch:
                        continue
                    sim = difflib.SequenceMatcher(None, vi['func'], pi['func']).ratio()
                    if sim > best_sim:
                        best_sim = sim
                        best_pi_idx = pi_idx
                if best_pi_idx >= 0:
                    used_patch.add(best_pi_idx)
                    pairs.append((vi, fn_patches[best_pi_idx]))
            
            for vuln_sample, patched_sample in pairs:
                func_name = extract_func_name_from_body(vuln_sample['func'])
                
                # Extract line changes
                deleted_lines, added_lines = extract_changed_lines_difflib(
                    vuln_sample['func'], patched_sample['func']
                )
                
                project = vuln_sample.get('project', '')
                project_url = vuln_sample.get('project_url', '')
                filepath = vuln_sample.get('file_name', '')
                
                # Commit message
                cm = vuln_sample.get('commit_message', '')
                commit_message = cm if cm and str(cm).strip().lower() not in ['none', 'null', ''] else ''
                
                # CVE/CWE as lists for later merging
                cve_raw = vuln_sample.get('cve')
                cve_list = [cve_raw] if cve_raw and str(cve_raw).strip().lower() not in ['none', 'null', ''] else []
                
                cwe_raw = vuln_sample.get('cwe')
                if isinstance(cwe_raw, list):
                    cwe_list = [c for c in cwe_raw if c and str(c).strip().lower() not in ['none', 'null', '']]
                elif cwe_raw and str(cwe_raw).strip().lower() not in ['none', 'null', '']:
                    cwe_list = [cwe_raw]
                else:
                    cwe_list = []
                
                samples.append({
                    "source": "primevul",
                    "pair_id": f"primevul_{project}_{commit_id}_{func_name}",
                    "commit_id": commit_id,
                    "project": project,
                    "project_url": project_url,
                    "func_name": func_name,
                    "filepath": filepath,
                    "vuln_hash": compute_vuln_hash(vuln_sample['func']),
                    "vuln_func": vuln_sample['func'],
                    "patched_func": patched_sample['func'],
                    "cve": cve_list,
                    "cve_desc": vuln_sample.get('cve_desc'),
                    "cwe": cwe_list,
                    "commit_message": commit_message,
                    "deleted_lines": deleted_lines,
                    "added_lines": added_lines,
                    "source_row_idx": vuln_sample.get('idx')
                })
    
    print(f"  Multi-function commits: {multi_func_commits}")
    print(f"  Skipped (no vuln or no patched): {skipped_commits}")
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
            
            func_name = row.get('func_name', '')
            
            # Map vul_type to CWE format (as list)
            vul_type = row.get('vul_type', '')
            cwe_list = [vul_type.upper()] if vul_type else []
            
            samples.append({
                "source": "sven",
                "pair_id": f"sven_{project}_{commit_id}_{func_name}",
                "commit_id": commit_id,
                "project": project,
                "project_url": f"https://{row['commit_link'].split('/commit/')[0]}" if '/commit/' in row['commit_link'] else None,
                "func_name": func_name,
                "filepath": row.get('file_name', ''),
                "vuln_hash": compute_vuln_hash(row['func_src_before']),
                "vuln_func": row['func_src_before'],
                "patched_func": row['func_src_after'],
                "cve": [],  # SVEN doesn't have CVE
                "cve_desc": None,
                "cwe": cwe_list,
                "commit_message": '',  # SVEN doesn't have commit messages
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


def _detect_line_problems(native_lines: list[dict], func_lines: list[str]) -> bool:
    """
    Detect if any ground truth line numbers are OOB or misindexed
    relative to the function body.
    
    Returns True if problems are detected, False if all lines are clean.
    """
    func_line_count = len(func_lines)
    for entry in native_lines:
        ln = entry['line_no']
        text = entry.get('text', '').strip()
        # OOB check
        if ln > func_line_count or ln < 1:
            return True
        # Misindexing check: line is in bounds but text doesn't match
        if text and func_lines[ln - 1].strip() != text:
            return True
    return False


def _try_text_reindex(native_lines: list[dict], func_lines: list[str]) -> Optional[list[dict]]:
    """
    Attempt to reindex line numbers by searching for each GT text in the function body.
    
    Returns reindexed list if successful, None if any entry is unresolvable.
    """
    reindexed = []
    # Group GT entries by their text to handle duplicates
    from collections import Counter
    gt_text_counts = Counter(e['text'].strip() for e in native_lines if e.get('text', '').strip())
    
    # Build a map: text -> list of line positions in func_body
    text_to_positions = defaultdict(list)
    for i, fl in enumerate(func_lines, 1):
        text_to_positions[fl.strip()].append(i)
    
    # Track which positions have been assigned (for ordered assignment of duplicates)
    assigned_positions = {}  # text -> index into its positions list (for ordered assignment)
    
    func_line_count = len(func_lines)
    
    for entry in native_lines:
        text = entry.get('text', '').strip()
        if not text:
            # Empty text lines cannot be reindexed (nothing to search for).
            # Drop them entirely — their line_no is file-relative and unreliable,
            # and a blank line carries no useful signal for vulnerability detection.
            continue
        
        positions = text_to_positions.get(text, [])
        
        if len(positions) == 0:
            # Text not found in function body at all → unresolvable
            return None
        elif len(positions) == 1:
            # Unique match → reindex
            reindexed.append({**entry, 'line_no': positions[0]})
        else:
            # Multiple matches — check if count matches GT occurrences
            gt_count = gt_text_counts[text]
            if gt_count == len(positions):
                # Same count: assign by positional order
                if text not in assigned_positions:
                    assigned_positions[text] = 0
                pos_idx = assigned_positions[text]
                if pos_idx < len(positions):
                    reindexed.append({**entry, 'line_no': positions[pos_idx]})
                    assigned_positions[text] = pos_idx + 1
                else:
                    return None  # Shouldn't happen, but safety
            else:
                # Ambiguous: different counts → unresolvable
                return None
    
    return reindexed


def reindex_secvuleval_lines(native_lines: list[dict], func_body: str,
                              fallback_difflib_lines: list[dict]) -> tuple[list[dict], str]:
    """
    Reindex SecVulEval line numbers to be function-relative.
    Only modifies samples where OOB or misindexing is detected.
    
    Strategy:
    1. If no problems detected → return native lines unchanged
    2. Try text-search reindex → return if successful
    3. Fall back to difflib-computed lines
    
    Args:
        native_lines: Original parsed [{line_no, text}, ...] from changed_lines
        func_body: The function body to reindex against
        fallback_difflib_lines: Lines computed by extract_changed_lines_difflib()
    
    Returns:
        (reindexed_lines, method) where method is 'native', 'text_reindex', or 'difflib'
    """
    if not native_lines:
        # Source had no changed_lines — use difflib if it found changes
        if fallback_difflib_lines:
            return fallback_difflib_lines, 'difflib'
        return native_lines, 'native'
    
    func_lines = func_body.splitlines()
    
    # Step 1: Detect problems
    if not _detect_line_problems(native_lines, func_lines):
        return native_lines, 'native'
    
    # Step 2: Try text-search reindex
    reindexed = _try_text_reindex(native_lines, func_lines)
    if reindexed is not None:
        return reindexed, 'text_reindex'
    
    # Step 3: Fall back to difflib
    return fallback_difflib_lines, 'difflib'


def load_secvuleval(paths: list[str]) -> list[dict]:
    """
    Load SecVulEval dataset and transform to unified schema.
    
    Applies conservative reindexing of ground truth line numbers:
    - Detects OOB (line_no > function line count) and misindexed samples
    - Tries text-search reindexing first (matching GT text to function lines)
    - Falls back to difflib if text search fails
    - Leaves clean samples untouched
    """
    samples = []
    reindex_stats = defaultdict(int)
    
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
            
            # Parse native line changes from SecVulEval
            # These may have file-absolute line numbers (OOB) — will be fixed below
            native_deleted = parse_secvuleval_changed_lines(vuln_row['changed_lines'])
            native_added = parse_secvuleval_changed_lines(patched_row['changed_lines'])
            
            vuln_func = vuln_row['func_body']
            patched_func = patched_row['func_body']
            
            # Compute difflib fallback (snippet-relative, always correct)
            difflib_deleted, difflib_added = extract_changed_lines_difflib(
                vuln_func, patched_func
            )
            
            # Reindex: detect problems → text search → difflib fallback
            deleted_lines, del_method = reindex_secvuleval_lines(
                native_deleted, vuln_func, difflib_deleted
            )
            added_lines, add_method = reindex_secvuleval_lines(
                native_added, patched_func, difflib_added
            )
            
            # Track reindexing stats
            reindex_stats[del_method] += 1
            reindex_stats[f'added_{add_method}'] += 1
            
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
                    desc_text = ' '.join(str(e) for e in explanations).strip()
                    # Only use if it has actual content (>5 chars, not just whitespace)
                    if len(desc_text) > 5:
                        cve_desc = desc_text
            
            # func_name and filepath for function-level dedup
            func_name = vuln_row.get('func_name', '')
            filepath = vuln_row.get('filepath', '')
            
            # Commit message — SecVulEval has commit_message field
            cm = vuln_row.get('commit_message', '')
            if isinstance(cm, float) or cm is None:
                commit_message = ''
            else:
                cm = str(cm).strip()
                commit_message = cm if cm.lower() not in ['none', 'null', ''] else ''
            
            samples.append({
                "source": "secvuleval",
                "pair_id": f"secvuleval_{project}_{commit_id}_{func_name}",
                "commit_id": commit_id,
                "project": project,
                "project_url": vuln_row.get('project_url', ''),
                "func_name": func_name,
                "filepath": filepath,
                "vuln_hash": compute_vuln_hash(vuln_func),
                "vuln_func": vuln_func,
                "patched_func": patched_func,
                "cve": list(cve_list),
                "cve_desc": cve_desc,
                "cwe": list(cwe_list),
                "commit_message": commit_message,
                "deleted_lines": deleted_lines,
                "added_lines": added_lines,
                "source_row_idx": int(vuln_row['idx'])
            })
    
    # Report reindexing stats
    if reindex_stats:
        print(f"  SecVulEval reindexing stats (deleted_lines):")
        for method in ['native', 'text_reindex', 'difflib']:
            count = reindex_stats.get(method, 0)
            if count > 0:
                print(f"    {method}: {count}")
        print(f"  SecVulEval reindexing stats (added_lines):")
        for method in ['native', 'text_reindex', 'difflib']:
            count = reindex_stats.get(f'added_{method}', 0)
            if count > 0:
                print(f"    {method}: {count}")
    
    return samples


# --- Metadata Pooling and Function-Level Deduplication ---

def build_commit_metadata_pool(samples: list[dict]) -> dict:
    """
    Build a commit-level metadata pool from ALL sources.
    
    For each commit_id, merges:
    - cve: union of all CVE IDs from all sources
    - cwe: union of all CWE IDs from all sources
    - cve_desc: prefer PrimeVul; fall back to SecVulEval if >5 chars
    - commit_message: prefer longest non-empty message
    
    Returns dict mapping commit_id -> {cve, cwe, cve_desc, commit_message}
    """
    pool = defaultdict(lambda: {
        'cve_set': set(),
        'cwe_set': set(),
        'cve_desc': None,
        'commit_message': '',
    })
    
    # Priority for cve_desc: PV > SE > SVEN
    desc_priority = {'primevul': 0, 'secvuleval': 1, 'sven': 2}
    
    for s in samples:
        cid = s['commit_id'].lower()
        p = pool[cid]
        
        # Merge CVEs
        for cve in (s.get('cve') or []):
            if cve and str(cve).strip().lower() not in ['none', 'null', '']:
                p['cve_set'].add(str(cve).strip())
        
        # Merge CWEs (filter out NVD placeholders that carry no signal)
        for cwe in (s.get('cwe') or []):
            cwe_str = str(cwe).strip()
            if (cwe_str and cwe_str.lower() not in ['none', 'null', '']
                    and cwe_str not in ['NVD-CWE-noinfo', 'NVD-CWE-Other']):
                p['cwe_set'].add(cwe_str)
        
        # cve_desc: prefer PV, fall back to SE
        desc = s.get('cve_desc')
        if desc and str(desc).strip().lower() not in ['none', 'null', ''] and len(str(desc).strip()) > 5:
            src_prio = desc_priority.get(s['source'], 99)
            if p['cve_desc'] is None:
                p['cve_desc'] = str(desc).strip()
                p['_desc_prio'] = src_prio
            elif src_prio < p.get('_desc_prio', 99):
                p['cve_desc'] = str(desc).strip()
                p['_desc_prio'] = src_prio
        
        # commit_message: keep longest
        msg = s.get('commit_message', '')
        if msg and not isinstance(msg, float):
            msg = str(msg)
            if len(msg) > len(p['commit_message']):
                p['commit_message'] = msg
    
    # Convert sets to sorted lists
    result = {}
    for cid, p in pool.items():
        result[cid] = {
            'cve': sorted(p['cve_set']),
            'cwe': sorted(p['cwe_set']),
            'cve_desc': p['cve_desc'],
            'commit_message': p['commit_message'],
        }
    return result


def deduplicate_by_function(samples: list[dict]) -> list[dict]:
    """
    Function-level deduplication with validated 3-tier conflict resolution.
    
    Strategy (validated against 457 conflict pairs, 0 wrong collapses):
    1. Group by (commit_id, func_name)
    2. Within each group, entries with same vuln_hash = true duplicates → keep best by priority
    3. Entries with different vuln_hash = potential conflicts → resolve pairwise:
       a. one_contains (removed==0):
          - diff_file → keep both (genuinely different)
          - same/unknown file → collapse (keep bigger)
       b. sim > 0.7 OR added_ratio >= 2:
          - diff_file → keep both (genuinely different)
          - same/unknown file → collapse (keep version with more added lines)
       c. low similarity AND low added_ratio:
          → keep both (conservatively assume genuinely different)
    """
    source_priority = {"sven": 0, "secvuleval": 1, "primevul": 2}
    
    def is_valid_pair(sample):
        return sample['vuln_func'] != sample['patched_func']
    
    def pick_best(entries):
        """Pick best entry from a list of same-hash entries by priority."""
        best = entries[0]
        for e in entries[1:]:
            e_valid = is_valid_pair(e)
            b_valid = is_valid_pair(best)
            e_prio = source_priority.get(e['source'], 99)
            b_prio = source_priority.get(best['source'], 99)
            
            if e_valid and not b_valid:
                best = e
            elif not e_valid and b_valid:
                pass
            elif e_prio < b_prio:
                best = e
        return best
    
    def collapse_pair(a, b):
        """
        Collapse two entries into one. A is shorter, B is longer.
        Returns the winning entry.
        """
        added, removed, sim = compute_normalized_diff(a['vuln_func'], b['vuln_func'])
        one_contains = removed == 0
        
        if one_contains:
            # B wholly contains A → keep B
            return b
        
        # Keep version with more added lines
        if added > removed:
            return b
        else:
            # Tie-break: prefer entry with commit_message, then priority
            a_has_msg = bool(a.get('commit_message', '').strip())
            b_has_msg = bool(b.get('commit_message', '').strip())
            if b_has_msg and not a_has_msg:
                return b
            elif a_has_msg and not b_has_msg:
                return a
            # Final tie-break: source priority
            a_prio = source_priority.get(a['source'], 99)
            b_prio = source_priority.get(b['source'], 99)
            return a if a_prio <= b_prio else b
    
    # Step 1: Group by (commit_id, func_name)
    by_cf = defaultdict(list)
    for s in samples:
        key = (s['commit_id'].lower(), s['func_name'])
        by_cf[key].append(s)
    
    survivors = []
    stats = Counter()
    
    for (cid, fname), group in by_cf.items():
        # Step 2: Sub-group by vuln_hash
        by_hash = defaultdict(list)
        for e in group:
            by_hash[e['vuln_hash']].append(e)
        
        if len(by_hash) == 1:
            # All same hash → true duplicate → pick best
            stats['same_hash_dedup'] += len(group) - 1
            survivors.append(pick_best(group))
            continue
        
        # Step 3: Multiple hashes → pairwise conflict resolution
        # Start with all variants as candidates
        variants = [pick_best(entries) for entries in by_hash.values()]
        
        # Iteratively collapse pairs
        changed = True
        while changed and len(variants) > 1:
            changed = False
            new_variants = []
            collapsed_indices = set()
            
            for i in range(len(variants)):
                if i in collapsed_indices:
                    continue
                a = variants[i]
                was_collapsed = False
                
                for j in range(i + 1, len(variants)):
                    if j in collapsed_indices:
                        continue
                    b = variants[j]
                    
                    # Ensure a is shorter
                    if len(a['vuln_func'].splitlines()) > len(b['vuln_func'].splitlines()):
                        a, b = b, a
                    
                    added, removed, sim = compute_normalized_diff(a['vuln_func'], b['vuln_func'])
                    one_contains = removed == 0
                    added_ratio = added / removed if removed > 0 else float('inf')
                    
                    diff_file = diff_file_smart(
                        a.get('filepath', ''), b.get('filepath', ''),
                        a.get('project', ''), b.get('project', '')
                    )
                    
                    should_collapse = False
                    if one_contains and not diff_file:
                        should_collapse = True
                    elif (sim > 0.7 or added_ratio >= 2) and not diff_file:
                        should_collapse = True
                    
                    if should_collapse:
                        winner = collapse_pair(a, b)
                        new_variants.append(winner)
                        collapsed_indices.add(i)
                        collapsed_indices.add(j)
                        was_collapsed = True
                        changed = True
                        stats['conflicts_collapsed'] += 1
                        break
                
                if not was_collapsed and i not in collapsed_indices:
                    new_variants.append(variants[i])
            
            variants = new_variants
        
        survivors.extend(variants)
        if len(by_hash) > 1:
            stats['conflicts_kept_both'] += len(variants)
    
    print(f"  Function-level dedup stats:")
    print(f"    Same-hash dedup removed: {stats['same_hash_dedup']}")
    print(f"    Different-hash conflicts collapsed: {stats['conflicts_collapsed']}")
    print(f"    Different-hash entries kept: {stats['conflicts_kept_both']}")
    
    return survivors


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
    
    # Build commit-level metadata pool (BEFORE dedup, using ALL sources)
    print("\nBuilding commit-level metadata pool...")
    metadata_pool = build_commit_metadata_pool(all_samples)
    print(f"  Pooled metadata for {len(metadata_pool)} unique commits")
    
    # Count metadata coverage
    pool_has_cve = sum(1 for v in metadata_pool.values() if v['cve'])
    pool_has_cwe = sum(1 for v in metadata_pool.values() if v['cwe'])
    pool_has_desc = sum(1 for v in metadata_pool.values() if v['cve_desc'])
    pool_has_msg = sum(1 for v in metadata_pool.values() if v['commit_message'])
    print(f"  Commits with CVE: {pool_has_cve}")
    print(f"  Commits with CWE: {pool_has_cwe}")
    print(f"  Commits with cve_desc: {pool_has_desc}")
    print(f"  Commits with commit_message: {pool_has_msg}")
    
    # Deduplicate
    if not args.skip_dedup:
        print("\nDeduplicating by (commit_id, func_name, vuln_hash)...")
        all_samples = deduplicate_by_function(all_samples)
        print(f"Total after dedup: {len(all_samples)}")
    
    # Enrich ALL surviving entries with pooled metadata
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
    
    # Final filter: remove samples where vuln_func is effectively identical to patched_func
    # (exact match OR only trailing whitespace differences — not real fixes)
    before_filter = len(all_samples)
    all_samples = [s for s in all_samples if not funcs_are_effectively_identical(s['vuln_func'], s['patched_func'])]
    filtered_count = before_filter - len(all_samples)
    if filtered_count > 0:
        print(f"Removed {filtered_count} samples with identical/whitespace-only vuln/patched functions")
    print(f"Final count: {len(all_samples)}")
    
    # Stats
    print("\n" + "=" * 60)
    print("Final dataset statistics:")
    source_counts = defaultdict(int)
    for s in all_samples:
        source_counts[s['source']] += 1
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} pairs")
    
    # New field coverage
    has_func_name = sum(1 for s in all_samples if s.get('func_name'))
    has_vuln_hash = sum(1 for s in all_samples if s.get('vuln_hash'))
    has_filepath = sum(1 for s in all_samples if s.get('filepath') and s['filepath'] != 'None')
    has_cve_desc = sum(1 for s in all_samples if s.get('cve_desc'))
    has_commit_msg = sum(1 for s in all_samples if s.get('commit_message'))
    has_cve = sum(1 for s in all_samples if s.get('cve'))
    has_cwe = sum(1 for s in all_samples if s.get('cwe'))
    print(f"\n  Field coverage:")
    print(f"    func_name: {has_func_name}/{len(all_samples)}")
    print(f"    vuln_hash: {has_vuln_hash}/{len(all_samples)}")
    print(f"    filepath:  {has_filepath}/{len(all_samples)}")
    print(f"    cve:       {has_cve}/{len(all_samples)}")
    print(f"    cwe:       {has_cwe}/{len(all_samples)}")
    print(f"    cve_desc:  {has_cve_desc}/{len(all_samples)}")
    print(f"    commit_message: {has_commit_msg}/{len(all_samples)}")
    
    # Check for duplicate keys
    key_counts = Counter()
    for s in all_samples:
        key = (s['commit_id'].lower(), s.get('func_name', ''), s.get('vuln_hash', ''))
        key_counts[key] += 1
    dup_keys = {k: v for k, v in key_counts.items() if v > 1}
    if dup_keys:
        print(f"\n  ⚠ WARNING: {len(dup_keys)} duplicate (commit_id, func_name, vuln_hash) keys found!")
        for k, v in list(dup_keys.items())[:5]:
            print(f"    {k}: {v} entries")
    else:
        print(f"\n  ✅ No duplicate (commit_id, func_name, vuln_hash) keys")
    
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
            print(f"  func_name: {sample.get('func_name', '')}")
            print(f"  vuln_hash: {sample.get('vuln_hash', '')[:16]}...")
            print(f"  project: {sample['project']}")
            print(f"  cve: {sample['cve']}")
            print(f"  cwe: {sample['cwe']}")
            print(f"  commit_message: {sample.get('commit_message', '')[:60]}..." if sample.get('commit_message') else "  commit_message: ''")
            print(f"  deleted_lines: {sample['deleted_lines'][:2]}..." if sample['deleted_lines'] else "  deleted_lines: []")
            print(f"  added_lines: {sample['added_lines'][:2]}..." if sample['added_lines'] else "  added_lines: []")


if __name__ == "__main__":
    main()
