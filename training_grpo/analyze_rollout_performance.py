#!/usr/bin/env python3
"""
Analyze veRL GRPO rollout performance from completion debug logs.

Highlights the "fast-then-slow" rollout pattern and correlates it with
completion lengths, max_num_seqs, and max_num_batched_tokens settings.

Usage:
    # Analyze a single completions file
    python analyze_rollout_performance.py --completions path/to/verl_completions_debug.jsonl

    # Analyze and compare multiple runs (pass labels for legend)
    python analyze_rollout_performance.py \
        --completions run1.jsonl run2.jsonl run3.jsonl \
        --labels "seqs=1024,bt=8192" "seqs=1,bt=8192" "seqs=1,bt=40960" \
        --save-plots --output-dir ./rollout_analysis
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def parse_completions(filepath):
    """Parse a verl_completions_debug.jsonl file."""
    entries = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts_str = d.get('timestamp', '')
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                else:
                    ts = None
                entries.append({
                    'idx': i,
                    'timestamp': ts,
                    'call_num': d.get('call_num', 0),
                    'completion_length': d.get('completion_length', len(d.get('completion', ''))),
                    'reward': d.get('reward', 0.0),
                })
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: skipping line {i} in {filepath}: {e}")
                continue
    return entries


def compute_rates(entries, window_size=20):
    """Compute completion rate (completions/min) using a sliding window."""
    if len(entries) < 2:
        return [], [], []

    timestamps = [e['timestamp'] for e in entries if e['timestamp']]
    if len(timestamps) < 2:
        return [], [], []

    t0 = timestamps[0]
    elapsed_min = [(t - t0).total_seconds() / 60.0 for t in timestamps]

    rates = []
    rate_times = []
    rate_elapsed = []

    for i in range(window_size, len(timestamps)):
        dt = (timestamps[i] - timestamps[i - window_size]).total_seconds() / 60.0
        if dt > 0:
            rate = window_size / dt
        else:
            rate = float('inf')
        rates.append(rate)
        rate_times.append(timestamps[i])
        rate_elapsed.append(elapsed_min[i])

    return rate_elapsed, rates, rate_times


def compute_inter_completion_time(entries):
    """Compute time between consecutive completions."""
    timestamps = [e['timestamp'] for e in entries if e['timestamp']]
    if len(timestamps) < 2:
        return [], []

    t0 = timestamps[0]
    elapsed = [(t - t0).total_seconds() / 60.0 for t in timestamps[1:]]
    deltas = [(timestamps[i+1] - timestamps[i]).total_seconds()
              for i in range(len(timestamps) - 1)]
    return elapsed, deltas


def print_summary(entries, label=""):
    """Print text summary of completion performance."""
    if not entries:
        print(f"  No entries for {label}")
        return

    timestamps = [e['timestamp'] for e in entries if e['timestamp']]
    lengths = [e['completion_length'] for e in entries]
    rewards = [e['reward'] for e in entries]

    print(f"\n{'='*70}")
    print(f"  Run: {label}")
    print(f"{'='*70}")
    print(f"  Total completions: {len(entries)}")
    if timestamps:
        total_time = (timestamps[-1] - timestamps[0]).total_seconds()
        print(f"  Time span: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
        print(f"  Overall rate: {len(entries) / (total_time/60):.2f} completions/min")
    print(f"  Completion length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
    print(f"  Reward: min={min(rewards):.2f}, max={max(rewards):.2f}, "
          f"mean={np.mean(rewards):.2f}")

    # Show rate in buckets to highlight slowdown
    if timestamps and len(timestamps) >= 2:
        bucket_size = max(1, len(entries) // 10)
        print(f"\n  Rate breakdown (buckets of ~{bucket_size}):")
        print(f"  {'Completions':>15s} {'Time (min)':>12s} {'Rate (/min)':>12s} "
              f"{'Avg Length':>12s} {'Avg Reward':>12s}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for b_start in range(0, len(entries), bucket_size):
            b_end = min(b_start + bucket_size, len(entries)) - 1
            if b_start >= len(timestamps) or b_end >= len(timestamps):
                break
            dt = (timestamps[b_end] - timestamps[b_start]).total_seconds() / 60.0
            avg_len = np.mean([entries[j]['completion_length']
                               for j in range(b_start, b_end + 1)])
            avg_rew = np.mean([entries[j]['reward']
                               for j in range(b_start, b_end + 1)])
            rate = (b_end - b_start + 1) / dt if dt > 0 else float('inf')
            print(f"  [{b_start:5d}-{b_end:5d}] {dt:10.1f}m {rate:10.2f}/m "
                  f"{avg_len:10.0f}ch {avg_rew:10.3f}")


def print_batch_analysis(entries, label=""):
    """Print batch-level analysis: groups completions by gaps, identifies anomalous batches."""
    if not entries or len(entries) < 2:
        return

    timestamps = [e['timestamp'] for e in entries if e['timestamp']]
    if len(timestamps) < 2:
        return

    # Group into batches (gap > 30s between consecutive completions)
    batches = [[0]]
    for i in range(1, len(entries)):
        gap = (timestamps[i] - timestamps[i-1]).total_seconds()
        if gap > 30:
            batches.append([i])
        else:
            batches[-1].append(i)

    # Compute batch durations
    batch_info = []
    for bi, idxs in enumerate(batches):
        if bi == 0:
            dur = (timestamps[idxs[-1]] - timestamps[idxs[0]]).total_seconds()
        else:
            dur = (timestamps[idxs[-1]] - timestamps[batches[bi-1][-1]]).total_seconds()

        max_len = max(entries[i]['completion_length'] for i in idxs)
        avg_reward = np.mean([entries[i]['reward'] for i in idxs])
        tok_est = max_len / 4.5  # rough char-to-token estimate
        tps = tok_est / dur if dur > 0 else float('inf')
        batch_info.append({
            'batch_id': bi, 'size': len(idxs), 'duration': dur,
            'max_len': max_len, 'tps': tps, 'avg_reward': avg_reward,
        })

    durations = [b['duration'] for b in batch_info]

    print(f"\n  Batch Analysis (gap threshold: 30s)")
    print(f"  {'─'*65}")
    print(f"  Total batches: {len(batches)}")
    print(f"  Duration stats: median={np.median(durations):.1f}s, "
          f"mean={np.mean(durations):.1f}s, max={max(durations):.0f}s")

    # Slow batches (>300s)
    slow = [b for b in batch_info if b['duration'] > 300]
    anomalous = [b for b in batch_info if b['tps'] < 5]

    if slow:
        slow_total = sum(b['duration'] for b in slow)
        total = sum(durations)
        print(f"\n  Slow batches (>300s): {len(slow)}, "
              f"consuming {slow_total/60:.0f} min ({slow_total/total*100:.0f}% of total)")
        print(f"  {'Batch':>7s} {'Size':>5s} {'Duration':>10s} {'MaxLen':>8s} "
              f"{'tok/s':>8s} {'Note':>15s}")
        print(f"  {'─'*7} {'─'*5} {'─'*10} {'─'*8} {'─'*8} {'─'*15}")
        for b in slow:
            note = "⚠️ ANOMALOUS" if b['tps'] < 5 else ""
            print(f"  {b['batch_id']:7d} {b['size']:5d} {b['duration']:8.0f}s "
                  f"{b['max_len']:8d} {b['tps']:8.1f} {note:>15s}")

    if anomalous:
        print(f"\n  Anomalous batches (<5 tok/s): {len(anomalous)}")
        anom_total = sum(b['duration'] for b in anomalous)
        print(f"  Total anomalous time: {anom_total/60:.0f} min "
              f"({anom_total/sum(durations)*100:.0f}% of total)")

    # Long-completion stats
    long_50k = sum(1 for e in entries if e['completion_length'] > 50000)
    long_30k = sum(1 for e in entries if e['completion_length'] > 30000)
    if long_50k or long_30k:
        print(f"\n  Long completions: {long_30k} >30k chars, {long_50k} >50k chars")


def plot_single_run(entries, label, output_dir):
    """Generate detailed plots for a single run."""
    timestamps = [e['timestamp'] for e in entries if e['timestamp']]
    lengths = [e['completion_length'] for e in entries]

    if len(timestamps) < 3:
        print(f"  Not enough data points for plots ({len(timestamps)})")
        return

    t0 = timestamps[0]
    elapsed_min = [(t - t0).total_seconds() / 60.0 for t in timestamps]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Rollout Performance Analysis\n{label}', fontsize=14, fontweight='bold')

    # --- Plot 1: Cumulative completions over time ---
    ax = axes[0, 0]
    ax.plot(elapsed_min, range(len(elapsed_min)), 'b-', linewidth=1.5)
    ax.set_xlabel('Elapsed Time (minutes)')
    ax.set_ylabel('Cumulative Completions')
    ax.set_title('Cumulative Completions Over Time')
    ax.grid(True, alpha=0.3)
    # Mark where slowdown begins (rate drops below 50% of initial rate)
    rate_elapsed, rates, _ = compute_rates(entries, window_size=min(20, len(entries) // 4 + 1))
    if rates:
        initial_rate = np.mean(rates[:max(1, len(rates)//10)])
        slowdown_idx = None
        for i, r in enumerate(rates):
            if r < initial_rate * 0.3:
                slowdown_idx = i
                break
        if slowdown_idx is not None:
            ax.axvline(x=rate_elapsed[slowdown_idx], color='r', linestyle='--',
                       alpha=0.7, label=f'Slowdown @ {rate_elapsed[slowdown_idx]:.1f}min')
            ax.legend()

    # --- Plot 2: Completion rate over time ---
    ax = axes[0, 1]
    if rate_elapsed and rates:
        ax.plot(rate_elapsed, rates, 'g-', linewidth=1, alpha=0.6)
        # Smoothed rate
        if len(rates) > 5:
            smooth_window = min(20, len(rates) // 3)
            smoothed = np.convolve(rates, np.ones(smooth_window)/smooth_window, mode='valid')
            smooth_x = rate_elapsed[smooth_window-1:]
            if len(smooth_x) == len(smoothed):
                ax.plot(smooth_x, smoothed, 'g-', linewidth=2.5, label='Smoothed')
            ax.legend()
        ax.set_xlabel('Elapsed Time (minutes)')
        ax.set_ylabel('Completions / minute')
        ax.set_title('Completion Rate Over Time (sliding window)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)

    # --- Plot 3: Inter-completion time ---
    ax = axes[1, 0]
    ict_elapsed, ict_deltas = compute_inter_completion_time(entries)
    if ict_elapsed and ict_deltas:
        ax.scatter(ict_elapsed, ict_deltas, s=8, alpha=0.5, c='orange', edgecolors='none')
        # Rolling median
        if len(ict_deltas) > 10:
            win = min(20, len(ict_deltas) // 3)
            rolling_med = [np.median(ict_deltas[max(0,i-win):i+1])
                           for i in range(len(ict_deltas))]
            ax.plot(ict_elapsed, rolling_med, 'r-', linewidth=2, label='Rolling median')
            ax.legend()
        ax.set_xlabel('Elapsed Time (minutes)')
        ax.set_ylabel('Time Between Completions (seconds)')
        ax.set_title('Inter-Completion Time (⬆ = slower)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)

    # --- Plot 4: Completion length over time ---
    ax = axes[1, 1]
    ax.scatter(elapsed_min, lengths, s=8, alpha=0.5, c='purple', edgecolors='none')
    if len(lengths) > 10:
        win = min(20, len(lengths) // 3)
        rolling_med_len = [np.median(lengths[max(0,i-win):i+1])
                           for i in range(len(lengths))]
        ax.plot(elapsed_min, rolling_med_len, 'm-', linewidth=2, label='Rolling median')
        ax.legend()
    ax.set_xlabel('Elapsed Time (minutes)')
    ax.set_ylabel('Completion Length (chars)')
    ax.set_title('Completion Length Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_label = label.replace('/', '_').replace(' ', '_').replace('=', '')
    outpath = os.path.join(output_dir, f'rollout_perf_{safe_label}.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_comparison(all_entries, labels, output_dir):
    """Generate comparison plots across multiple runs."""
    if len(all_entries) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rollout Performance Comparison Across Configurations',
                 fontsize=14, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_entries)))

    for idx, (entries, label, color) in enumerate(zip(all_entries, labels, colors)):
        timestamps = [e['timestamp'] for e in entries if e['timestamp']]
        lengths = [e['completion_length'] for e in entries]

        if len(timestamps) < 3:
            continue

        t0 = timestamps[0]
        elapsed_min = [(t - t0).total_seconds() / 60.0 for t in timestamps]

        # Plot 1: Cumulative completions
        ax = axes[0, 0]
        ax.plot(elapsed_min, range(len(elapsed_min)), color=color,
                linewidth=2, label=label)

        # Plot 2: Rate over time
        ax = axes[0, 1]
        rate_elapsed, rates, _ = compute_rates(entries,
                                               window_size=min(20, len(entries)//4+1))
        if rates:
            if len(rates) > 5:
                smooth_window = min(20, len(rates) // 3)
                smoothed = np.convolve(rates, np.ones(smooth_window)/smooth_window,
                                       mode='valid')
                smooth_x = rate_elapsed[smooth_window-1:]
                if len(smooth_x) == len(smoothed):
                    ax.plot(smooth_x, smoothed, color=color, linewidth=2, label=label)

        # Plot 3: Inter-completion time
        ax = axes[1, 0]
        ict_elapsed, ict_deltas = compute_inter_completion_time(entries)
        if ict_elapsed and ict_deltas and len(ict_deltas) > 5:
            win = min(20, len(ict_deltas) // 3)
            rolling_med = [np.median(ict_deltas[max(0,i-win):i+1])
                           for i in range(len(ict_deltas))]
            ax.plot(ict_elapsed, rolling_med, color=color, linewidth=2, label=label)

        # Plot 4: Completion length distribution
        ax = axes[1, 1]
        ax.hist(lengths, bins=30, alpha=0.4, color=color, label=label, density=True)

    # Format all axes
    axes[0, 0].set_xlabel('Elapsed Time (minutes)')
    axes[0, 0].set_ylabel('Cumulative Completions')
    axes[0, 0].set_title('Cumulative Completions')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Elapsed Time (minutes)')
    axes[0, 1].set_ylabel('Completions / minute')
    axes[0, 1].set_title('Smoothed Completion Rate')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Elapsed Time (minutes)')
    axes[1, 0].set_ylabel('Time Between Completions (seconds)')
    axes[1, 0].set_title('Rolling Median Inter-Completion Time')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Completion Length (chars)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Completion Length Distribution')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'rollout_perf_comparison.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {outpath}")


def plot_length_vs_time_scatter(all_entries, labels, output_dir):
    """Scatter plot: completion length vs generation time for each run."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_entries)))

    for entries, label, color in zip(all_entries, labels, colors):
        timestamps = [e['timestamp'] for e in entries if e['timestamp']]
        lengths = [e['completion_length'] for e in entries if e['timestamp']]

        if len(timestamps) < 2:
            continue

        # Inter-completion times as proxy for generation time
        gen_times = [(timestamps[i+1] - timestamps[i]).total_seconds()
                     for i in range(len(timestamps) - 1)]
        gen_lengths = lengths[1:]  # lengths corresponding to gen_times

        ax.scatter(gen_lengths, gen_times, s=12, alpha=0.4, c=[color],
                   edgecolors='none', label=label)

    ax.set_xlabel('Completion Length (chars)', fontsize=12)
    ax.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax.set_title('Completion Length vs Generation Time\n'
                 '(longer completions → slower generation)', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'rollout_length_vs_time.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_simple(entries, label, output_dir):
    """Simple 2-panel plot: time per completion and length per completion vs completion #."""
    timestamps = [e['timestamp'] for e in entries if e['timestamp']]
    lengths = [e['completion_length'] for e in entries if e['timestamp']]

    if len(timestamps) < 3:
        print(f"  Not enough data points for simple plot ({len(timestamps)})")
        return

    t0 = timestamps[0]
    completion_nums = list(range(len(timestamps)))
    elapsed_min = [(t - t0).total_seconds() / 60.0 for t in timestamps]

    # Time per completion (seconds between consecutive completions)
    time_per = [(timestamps[i+1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'Rollout Performance — {label}\n'
                 f'{len(timestamps)} completions over {elapsed_min[-1]:.0f} min '
                 f'({elapsed_min[-1]/60:.1f}h)',
                 fontsize=13, fontweight='bold')

    # --- Top: Time per completion ---
    ax1.bar(completion_nums[1:], time_per, width=1.0, color='steelblue',
            alpha=0.7, edgecolor='none')
    # Rolling median overlay
    if len(time_per) > 10:
        win = min(20, len(time_per) // 4)
        rolling = [np.median(time_per[max(0, i-win):i+1])
                   for i in range(len(time_per))]
        ax1.plot(completion_nums[1:], rolling, 'r-', linewidth=2,
                 label=f'Rolling median (w={win})')
        ax1.legend(fontsize=9)
    ax1.set_ylabel('Time per Completion (seconds)')
    ax1.set_title('Generation Time per Completion')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add elapsed time as secondary x-axis on top
    ax1_top = ax1.twiny()
    # Place a few tick marks showing elapsed time
    n_ticks = min(8, len(completion_nums))
    tick_positions = np.linspace(0, len(completion_nums) - 1, n_ticks, dtype=int)
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(tick_positions)
    ax1_top.set_xticklabels([f'{elapsed_min[i]:.0f}m' for i in tick_positions],
                             fontsize=8)
    ax1_top.set_xlabel('Elapsed Time', fontsize=9)

    # --- Bottom: Length per completion ---
    ax2.bar(completion_nums, lengths, width=1.0, color='mediumpurple',
            alpha=0.7, edgecolor='none')
    if len(lengths) > 10:
        win = min(20, len(lengths) // 4)
        rolling_len = [np.median(lengths[max(0, i-win):i+1])
                       for i in range(len(lengths))]
        ax2.plot(completion_nums, rolling_len, 'darkred', linewidth=2,
                 label=f'Rolling median (w={win})')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Completion #')
    ax2.set_ylabel('Completion Length (chars)')
    ax2.set_title('Completion Length per Completion')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    safe_label = label.replace('/', '_').replace(' ', '_').replace('=', '').replace(',', '')
    outpath = os.path.join(output_dir, f'rollout_simple_{safe_label}.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze veRL GRPO rollout performance from completion debug logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single run analysis
    python analyze_rollout_performance.py \\
        --completions checkpoints/grpo_qwen3_4b_verl/verl_completions_debug.jsonl

    # Compare multiple runs
    python analyze_rollout_performance.py \\
        --completions run1.jsonl run2.jsonl \\
        --labels "max_num_seqs=1024" "max_num_seqs=1" \\
        --save-plots --output-dir ./rollout_analysis
        """)
    parser.add_argument('--completions', nargs='+', required=True,
                        help='Path(s) to verl_completions_debug.jsonl file(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each run (for legend). '
                             'If not provided, filenames are used.')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plot images')
    parser.add_argument('--output-dir', default='./rollout_analysis',
                        help='Output directory for plots (default: ./rollout_analysis)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (useful for headless)')

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.completions):
        print("Error: number of --labels must match number of --completions files")
        sys.exit(1)

    labels = args.labels or [os.path.basename(f) for f in args.completions]

    # Parse all files
    all_entries = []
    for filepath, label in zip(args.completions, labels):
        print(f"\nParsing: {filepath}")
        entries = parse_completions(filepath)
        all_entries.append(entries)
        print_summary(entries, label)
        print_batch_analysis(entries, label)

    # Generate plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Generating plots in {args.output_dir}")
        print(f"{'='*70}")

        # Individual run plots
        for entries, label in zip(all_entries, labels):
            print(f"\nPlotting: {label}")
            plot_simple(entries, label, args.output_dir)
            plot_single_run(entries, label, args.output_dir)

        # Comparison plots (if multiple runs)
        if len(all_entries) >= 2:
            print(f"\nPlotting comparison...")
            plot_comparison(all_entries, labels, args.output_dir)
            plot_length_vs_time_scatter(all_entries, labels, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
