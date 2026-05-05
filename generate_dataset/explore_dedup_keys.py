#!/usr/bin/env python3
"""
explore_dedup_keys.py - Explore dedup key candidates for multi-file commit support.

Investigates:
1. Functions with same (commit_id, func_name) but different code hash
2. Functions with same name conflicting in same commit (different code hashes)
3. Cross-dataset entries with same (commit_id, func_name) but different code hashes

Usage:
    python explore_dedup_keys.py
"""

import pandas as pd
import json
import hashlib
import re
import numpy as np
from collections import defaultdict, Counter


def normalize_code(code: str) -> str:
    """Normalize code for hashing: strip whitespace, normalize newlines."""
    lines = code.splitlines()
    # Strip each line, remove empty lines, lowercase
    stripped = [l.strip() for l in lines if l.strip()]
    return '\n'.join(stripped).lower()


def compute_code_hash(code: str) -> str:
    """Compute hash of normalized code."""
    return hashlib.md5(normalize_code(code).encode()).hexdigest()


def extract_func_name_from_body(func_body: str) -> str:
    """Extract function name from C/C++ function body's first line."""
    first_line = func_body.split('\n')[0].strip()
    # Match: [qualifiers] return_type [*] func_name(
    m = re.match(r'(?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:\w+[\s*]+)*?(\w+)\s*\(', first_line)
    if m:
        name = m.group(1)
        # Filter out false positives
        if name not in ['if', 'else', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof', 'do']:
            return name
    return first_line[:40]


# === Load all datasets ===
print("Loading datasets...")

# PrimeVul
pv = []
with open('PrimeVul-v0.1-hf/paired/primevul_train_paired.jsonl') as f:
    for line in f:
        if line.strip():
            pv.append(json.loads(line))

# SecVulEval
se = pd.read_parquet('SecVulEval/data/train-00000-of-00001.parquet')
se_vuln = se[
    (se['is_vulnerable'] == True) & 
    (se['fixed_func_idx'].notna()) & 
    (se['idx'] != se['fixed_func_idx'])
]

# SVEN
sv = pd.read_parquet('sven/data/train-00000-of-00001-23ea0a39e451d835.parquet')

# === Build unified function registry ===
# Key: (commit_id_lower, func_name, source) -> {vuln_hash, patched_hash, entry}
print("Building function registry...")

entries = []  # list of (commit_id, func_name, source, vuln_code_hash, patched_code_hash, entry_ref)

# PrimeVul: group by commit, pair vuln<->patched
pv_by_commit = defaultdict(list)
for e in pv:
    pv_by_commit[e['commit_id']].append(e)

for cid, items in pv_by_commit.items():
    vuln = [i for i in items if i['target'] == 1]
    patched = [i for i in items if i['target'] == 0]
    
    # Try to pair by func_name
    vuln_by_name = defaultdict(list)
    patched_by_name = defaultdict(list)
    for v in vuln:
        fname = extract_func_name_from_body(v['func'])
        vuln_by_name[fname].append(v)
    for p in patched:
        fname = extract_func_name_from_body(p['func'])
        patched_by_name[fname].append(p)
    
    for fname in vuln_by_name:
        for v in vuln_by_name[fname]:
            v_hash = compute_code_hash(v['func'])
            # Find best patched match
            p_hash = ''
            if fname in patched_by_name and patched_by_name[fname]:
                p_hash = compute_code_hash(patched_by_name[fname][0]['func'])
            
            entries.append({
                'commit_id': cid.lower(),
                'func_name': fname,
                'source': 'primevul',
                'vuln_hash': v_hash,
                'patched_hash': p_hash,
                'file_name': v.get('file_name', ''),
                'code_preview': v['func'][:100].replace('\n', ' '),
            })

# SecVulEval
for _, r in se_vuln.iterrows():
    fixed_idx = int(r['fixed_func_idx'])
    patched_rows = se[se['idx'] == fixed_idx]
    if len(patched_rows) == 0:
        continue
    patched_row = patched_rows.iloc[0]
    
    v_hash = compute_code_hash(r['func_body'])
    p_hash = compute_code_hash(patched_row['func_body'])
    
    entries.append({
        'commit_id': r['commit_id'].lower(),
        'func_name': r['func_name'],
        'source': 'secvuleval',
        'vuln_hash': v_hash,
        'patched_hash': p_hash,
        'file_name': r.get('filepath', ''),
        'code_preview': r['func_body'][:100].replace('\n', ' '),
    })

# SVEN
for _, r in sv.iterrows():
    link = r['commit_link']
    cid = link.split('/commit/')[-1].lower() if '/commit/' in link else link.lower()
    v_hash = compute_code_hash(r['func_src_before'])
    p_hash = compute_code_hash(r['func_src_after'])
    
    entries.append({
        'commit_id': cid,
        'func_name': r['func_name'],
        'source': 'sven',
        'vuln_hash': v_hash,
        'patched_hash': p_hash,
        'file_name': r.get('file_name', ''),
        'code_preview': r['func_src_before'][:100].replace('\n', ' '),
    })

print(f"Total entries: {len(entries)}")

# === Analysis 1: Same (commit_id, func_name) but different vuln_hash WITHIN same source ===
print("\n" + "="*70)
print("ANALYSIS 1: Same (commit_id, func_name) with different vuln_hash WITHIN same source")
print("="*70)

by_key_source = defaultdict(list)
for e in entries:
    key = (e['commit_id'], e['func_name'], e['source'])
    by_key_source[key].append(e)

intra_conflicts = []
for (cid, fname, src), group in by_key_source.items():
    if len(group) > 1:
        hashes = set(e['vuln_hash'] for e in group)
        if len(hashes) > 1:
            intra_conflicts.append((cid, fname, src, group))

print(f"Conflicts found: {len(intra_conflicts)}")
for cid, fname, src, group in intra_conflicts[:5]:
    print(f"\n  [{src}] commit={cid[:16]}... func_name={fname}")
    for e in group:
        print(f"    vuln_hash={e['vuln_hash'][:12]}... file={e['file_name']}")
        print(f"    preview: {e['code_preview'][:80]}")


# === Analysis 2: Same (commit_id, func_name) with different vuln_hash ACROSS sources ===
print("\n" + "="*70)
print("ANALYSIS 2: Same (commit_id, func_name) with different vuln_hash ACROSS sources")
print("="*70)

by_key = defaultdict(list)
for e in entries:
    key = (e['commit_id'], e['func_name'])
    by_key[key].append(e)

cross_conflicts = []
for (cid, fname), group in by_key.items():
    sources = set(e['source'] for e in group)
    if len(sources) > 1:
        hashes = set(e['vuln_hash'] for e in group)
        if len(hashes) > 1:
            cross_conflicts.append((cid, fname, group))

print(f"Cross-source conflicts (same commit+func, different vuln hash): {len(cross_conflicts)}")
for cid, fname, group in cross_conflicts[:8]:
    print(f"\n  commit={cid[:16]}... func_name={fname}")
    for e in group:
        print(f"    [{e['source']:12}] vuln_hash={e['vuln_hash'][:12]}... file={e['file_name']}")
        print(f"      preview: {e['code_preview'][:80]}")


# === Analysis 3: Same (commit_id, func_name) with SAME vuln_hash ACROSS sources ===
print("\n" + "="*70)
print("ANALYSIS 3: Same (commit_id, func_name) with SAME vuln_hash across sources (true duplicates)")
print("="*70)

cross_same = 0
for (cid, fname), group in by_key.items():
    sources = set(e['source'] for e in group)
    if len(sources) > 1:
        hashes = set(e['vuln_hash'] for e in group)
        if len(hashes) == 1:
            cross_same += 1

print(f"True duplicates (same commit+func+hash across sources): {cross_same}")


# === Analysis 4: Same func_name appearing multiple times in same commit (within one source) ===
print("\n" + "="*70)
print("ANALYSIS 4: Same func_name appearing multiple times in same commit within one source")
print("="*70)

same_name_same_commit = defaultdict(list)
for (cid, fname, src), group in by_key_source.items():
    if len(group) > 1:
        same_name_same_commit[src].append((cid, fname, group))

for src in ['primevul', 'secvuleval', 'sven']:
    conflicts = same_name_same_commit.get(src, [])
    same_hash = sum(1 for _, _, g in conflicts if len(set(e['vuln_hash'] for e in g)) == 1)
    diff_hash = sum(1 for _, _, g in conflicts if len(set(e['vuln_hash'] for e in g)) > 1)
    print(f"  {src}: {len(conflicts)} cases ({same_hash} same hash = true duplicates, {diff_hash} different hash = different code)")


# === Summary ===
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

total_unique_commit_func = len(by_key)
total_cross_source = sum(1 for g in by_key.values() if len(set(e['source'] for e in g)) > 1)
print(f"Total unique (commit_id, func_name) keys: {total_unique_commit_func}")
print(f"Keys appearing in >1 source: {total_cross_source}")
print(f"  Same hash (true duplicates, safe to dedup): {cross_same}")
print(f"  Different hash (need care): {len(cross_conflicts)}")
print(f"Intra-source same-key conflicts: {len(intra_conflicts)}")