import pandas as pd, re, json
from collections import Counter, defaultdict

df = pd.read_parquet('training_grpo/verl_data/train.parquet')
dfv = pd.read_parquet('training_grpo/verl_data/val.parquet')

# Load commit_id -> source mapping
commit_to_source = {}
mapping_file = '/export/home/acs/stud/t/tudor.farcasanu/SSL_research/generated_finetuning_data/grpo_finetuning_dataset.jsonl'
try:
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                d = json.loads(line)
                if 'commit_id' in d and 'source' in d:
                    commit_to_source[d['commit_id']] = d['source']
            except json.JSONDecodeError:
                pass
except Exception as e:
    print(f"Warning: Could not load source mapping: {e}")

issues = defaultdict(list)
stats = Counter()

def get_prompt_text(prompt):
    # prompt is a numpy array of dicts
    return ' '.join(m['content'] for m in prompt)

def analyze_sample(row, split, idx):
    gt = json.loads(row['reward_model']['ground_truth'])
    is_vulnerable = gt.get('is_vulnerable', False)
    gt_lines = gt.get('ground_truth_lines', [])
    
    extra_info = row['extra_info']
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    commit_id = extra_info.get('commit_id', '')
    source = commit_to_source.get(commit_id, 'Unknown')
    
    prompt_text = get_prompt_text(row['prompt'])

    # Count code lines via numbered prefix (1: ..., 2: ...)
    code_match = re.search(r'```[^\n]*\n(.*?)```', prompt_text, re.DOTALL)
    code_line_count = 0
    if code_match:
        code = code_match.group(1)
        numbered = re.findall(r'^\s*(\d+):', code, re.MULTILINE)
        if numbered:
            code_line_count = max(int(x) for x in numbered)
        else:
            code_line_count = len([l for l in code.split('\n') if l.strip()])

    stats['total'] += 1
    stats[f'source_{source}'] += 1

    # ISSUE 1: OOB line numbers
    if gt_lines and code_line_count > 0:
        oob = [l for l in gt_lines if l > code_line_count]
        if oob:
            stats['oob'] += 1
            issues['oob'].append({
                'split': split, 'idx': idx,
                'gt_lines': gt_lines, 'code_lines': code_line_count,
                'is_vulnerable': is_vulnerable,
                'source': source
            })

    # ISSUE 2: VULNERABLE with NO ground_truth_lines
    if is_vulnerable and not gt_lines:
        stats['vuln_no_lines'] += 1
        issues['vuln_no_lines'].append({'split': split, 'idx': idx, 'source': source})

    # ISSUE 3: Snippet too short for meaningful analysis (<=3 lines)
    if code_line_count > 0 and code_line_count <= 3:
        stats['tiny_snippet'] += 1
        issues['tiny'].append({'split': split, 'idx': idx, 'lines': code_line_count, 'is_vulnerable': is_vulnerable, 'gt_lines': gt_lines, 'source': source})

    # ISSUE 4: Hint-GT conflict: vulnerable_hint says VULNERABLE but gt says NOT
    has_vuln_hint = 'This code contains a security vulnerability' in prompt_text
    has_patched_hint = 'patched (non-vulnerable)' in prompt_text
    if has_vuln_hint and not is_vulnerable:
        stats['hint_gt_conflict'] += 1
    if has_patched_hint and is_vulnerable:
        stats['hint_gt_conflict_patched'] += 1

for idx, row in df.iterrows():
    analyze_sample(row, 'train', idx)
for idx, row in dfv.iterrows():
    analyze_sample(row, 'val', idx)

print(f"=== DATA QUALITY AUDIT ===")
print(f"Train: {len(df)}, Val: {len(dfv)}, Total: {stats['total']}")
print()
print(f"[OOB LINE]   GT line > snippet lines:          {stats['oob']:>5} ({100*stats['oob']/stats['total']:.1f}%)")
print(f"[VULN-NOLN]  VULNERABLE but no gt_lines:       {stats['vuln_no_lines']:>5} ({100*stats['vuln_no_lines']/stats['total']:.1f}%)")
print(f"[TINY]       Snippet <=3 lines:                {stats['tiny_snippet']:>5} ({100*stats['tiny_snippet']/stats['total']:.1f}%)")
print(f"[HINT_CFLCT] hint says VULN but GT says NOT:   {stats['hint_gt_conflict']:>5}")
print(f"[HINT_CFLCT] hint says PATCHED but GT is VULN: {stats['hint_gt_conflict_patched']:>5}")

# Print problem distribution across source datasets
print(f"\n--- Problem distribution across source datasets ---")

# Overall source distribution
sources = [k.replace('source_', '') for k in stats.keys() if k.startswith('source_')]
print("\n  Overall Source Distribution:")
for s in sorted(sources):
    count = stats[f'source_{s}']
    print(f"    {s}: {count:>6} ({100*count/stats['total']:.1f}%)")

# OOB breakdown
oob_train = sum(1 for x in issues['oob'] if x['split']=='train')
oob_val = sum(1 for x in issues['oob'] if x['split']=='val')
print(f"\n  OOB breakdown: train={oob_train}, val={oob_val}")
oob_sources = Counter([x['source'] for x in issues['oob']])
print("  OOB sources:")
for s, c in oob_sources.most_common():
    print(f"    {s}: {c}")

# Tiny snippet breakdown
tiny_train = sum(1 for x in issues['tiny'] if x['split']=='train')
tiny_val = sum(1 for x in issues['tiny'] if x['split']=='val')
tiny_vuln = sum(1 for x in issues['tiny'] if x['is_vulnerable'])
print(f"\n  Tiny breakdown: train={tiny_train}, val={tiny_val}")
print(f"  Tiny: vulnerable={tiny_vuln}, not_vulnerable={stats['tiny_snippet']-tiny_vuln}")
print(f"  Tiny with OOB gt_lines: {sum(1 for x in issues['tiny'] if x['gt_lines'])}")
tiny_sources = Counter([x['source'] for x in issues['tiny']])
print("  Tiny sources:")
for s, c in tiny_sources.most_common():
    print(f"    {s}: {c}")

# Vuln no lines breakdown
vuln_sources = Counter([x['source'] for x in issues['vuln_no_lines']])
print(f"\n  Vuln but no lines sources:")
for s, c in vuln_sources.most_common():
    print(f"    {s}: {c}")

# Show OOB examples
print(f"\n--- OOB examples ---")
for ex in issues['oob'][:8]:
    print(f"  [{ex['split']}:{ex['idx']}] source={ex['source']} | gt_lines={ex['gt_lines']} | code_lines={ex['code_lines']} | vuln={ex['is_vulnerable']}")

# VULN with no lines examples breakdown
print(f"\n--- 'VULNERABLE but no lines' samples ---")
print(f"  These {stats['vuln_no_lines']} samples have no specific line to point to")
print(f"  (model is supposed to classify VULNERABLE but can't identify any specific vulnerable_lines)")
print(f"  This is ambiguous for the reward function — it may only check classification, not lines")