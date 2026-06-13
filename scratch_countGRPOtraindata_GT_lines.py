import pandas as pd
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

df = pd.read_parquet('/export/home/acs/stud/t/tudor.farcasanu/SSL_research/training_grpo/verl_data/train.parquet')

vuln_lines_counts = []
patched_lines_counts = []

vuln_pair_ids = defaultdict(list)
patched_pair_ids = defaultdict(list)

for idx, row in df.iterrows():
    rm = row['reward_model']
    gt_str = rm.get('ground_truth', '{}')
    
    extra = row.get('extra_info', '{}')
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except:
            extra = ast.literal_eval(extra)
            
    try:
        if isinstance(gt_str, dict):
            gt = gt_str
        else:
            try:
                gt = json.loads(gt_str)
            except:
                gt = ast.literal_eval(gt_str)
    except Exception as e:
        print(f'Error parsing {gt_str}: {e}')
        continue
    
    is_vuln = extra.get('is_vulnerable', False)
    pair_id = extra.get('pair_id', 'unknown')
    lines = gt.get('ground_truth_lines', [])
    num_lines = len(lines)
    
    if is_vuln:
        vuln_lines_counts.append(num_lines)
        vuln_pair_ids[num_lines].append(pair_id)
    else:
        patched_lines_counts.append(num_lines)
        patched_pair_ids[num_lines].append(pair_id)

def print_stats(name, counts, pair_ids_map, threshold=100):
    if not counts:
        print(f"\n--- {name} Samples ---")
        print("No data available.")
        return
        
    print(f"\n{'='*40}")
    print(f"{name.upper()} SAMPLES SUMMARY")
    print(f"{'='*40}")
    print(f"{'Total Count:':<20} {len(counts)}")
    print(f"{'Mean Lines:':<20} {np.mean(counts):.2f}")
    print(f"{'Median Lines:':<20} {np.median(counts):.2f}")
    print(f"{'Min Lines:':<20} {np.min(counts)}")
    print(f"{'Max Lines:':<20} {np.max(counts)}")
    p75 = np.percentile(counts, 75)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)
    
    print(f"{'25th Percentile:':<20} {np.percentile(counts, 25):.2f}")
    print(f"{'75th Percentile:':<20} {p75:.2f}")
    print(f"{'95th Percentile:':<20} {p95:.2f}")
    print(f"{'99th Percentile:':<20} {p99:.2f}")
    
    print("\nSample Loss if Capped (Discarding > Threshold):")
    # Include percentiles and fixed values
    thresholds = [int(p75), int(p95), int(p99), 50, 75, 100]
    thresholds = sorted(list(set(thresholds)))
    
    print(f"{'Threshold':<15} | {'Lost Samples':<15} | {'% of Total Lost'}")
    print("-" * 55)
    for t in thresholds:
        lost = sum(1 for c in counts if c > t)
        pct_lost = (lost / len(counts)) * 100
        
        # Determine labels for context
        labels = []
        if t == int(p75): labels.append("75th Pctl")
        if t == int(p95): labels.append("95th Pctl")
        if t == int(p99): labels.append("99th Pctl")
        label_str = f" ({', '.join(labels)})" if labels else ""
        
        print(f"{str(t) + label_str:<15} | {lost:<15} | {pct_lost:.2f}%")
    
    print("\nLine Count Distribution:")
    print(f"{'Lines':<10} | {'Frequency':<10} | {'%' :<8} | {'Sample pair_ids (if > ' + str(threshold) + ' lines)'}")
    print("-" * 80)
    
    unique_counts, freqs = np.unique(counts, return_counts=True)
    total = len(counts)
    
    for val, freq in zip(unique_counts, freqs):
        pct = (freq / total) * 100
        samples_str = ""
        if val > threshold:
            # Get up to 3 random pair_ids for this high count
            samples = pair_ids_map[val][:3]
            samples_str = f"[{', '.join(samples)}]"
            
        print(f"{val:<10} | {freq:<10} | {pct:>5.1f}% | {samples_str}")

print_stats("Vulnerable", vuln_lines_counts, vuln_pair_ids, threshold=50)
print_stats("Patched", patched_lines_counts, patched_pair_ids, threshold=50)

all_counts = vuln_lines_counts + patched_lines_counts
all_pair_ids = defaultdict(list)
for k, v in vuln_pair_ids.items():
    all_pair_ids[k].extend(v)
for k, v in patched_pair_ids.items():
    all_pair_ids[k].extend(v)

print_stats("All", all_counts, all_pair_ids, threshold=50)

# Generate Graphs
plt.style.use('dark_background')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_dist(ax, data, title, color):
    if not data:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.set_title(title)
        return
        
    unique_counts, freqs = np.unique(data, return_counts=True)
    
    # Use categorical X-axis to place non-zero counts side-by-side (hiding empty gaps)
    x_positions = np.arange(len(unique_counts))
    ax.bar(x_positions, freqs, color=color, width=0.8, log=True)
    
    ax.set_title(title)
    ax.set_xlabel("Number of Ground Truth Lines")
    ax.set_ylabel("Frequency (Log Scale)")
    
    # Set x-ticks to the actual line counts
    if len(x_positions) > 20:
        step = max(1, len(x_positions) // 15)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels(unique_counts[::step], rotation=45)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_counts)
    
    # Add grid for easier reading of precise numbers
    ax.grid(True, which="both", axis="y", ls="-", alpha=0.2)

plot_dist(axes[0], vuln_lines_counts, "Vulnerable Samples", "red")
plot_dist(axes[1], patched_lines_counts, "Patched Samples", "green")
plot_dist(axes[2], all_counts, "All Samples", "blue")

plt.tight_layout()
out_path = '/export/home/acs/stud/t/tudor.farcasanu/SSL_research/gt_lines_distribution.png'
plt.savefig(out_path, dpi=150)
print(f"\nGraphs saved to: {out_path}")
