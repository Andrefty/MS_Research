#!/usr/bin/env python3
"""
Analyze veRL GRPO training metrics from WandB.
Fetches training history from one or more runs and provides statistics + visualizations.

Usage:
    # Single run
    python analyze_wandb_training.py --run-ids 48hbazzc

    # Multiple runs (e.g. resumed training)
    python analyze_wandb_training.py --run-ids 48hbazzc hi7fv33g

    # With custom project
    python analyze_wandb_training.py --project vulnerability_grpo --entity andrefty-universitatea-politehnica-din-bucuresti --run-ids 48hbazzc

    # Save plots and JSON
    python analyze_wandb_training.py --run-ids 48hbazzc hi7fv33g --save-plots --save-json
"""

import argparse
import json
import os
import sys

try:
    import wandb
    import pandas as pd
    import numpy as np
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────
# veRL GRPO metric definitions
# ─────────────────────────────────────────────
METRIC_GROUPS = {
    "Reward & Score": {
        "critic/score/mean":    "Average reward across batch (your 0/0.3/0.6/1.0 scale)",
        "critic/score/max":     "Max reward in batch (should stay 1.0 if any sample gets full marks)",
        "critic/score/min":     "Min reward in batch (0.0 = some samples still fail completely)",
        "critic/rewards/mean":  "Same as score/mean (veRL logs both)",
    },
    "Policy Gradient": {
        "actor/pg_loss":           "PPO clipped policy gradient loss. Negative = policy improving on high-advantage tokens",
        "actor/pg_clipfrac":       "Fraction of tokens where ratio π_new/π_old exceeded clip range (ε=0.2). "
                                   "Healthy: 5-15%. Very low (<1%) = policy barely updating",
        "actor/pg_clipfrac_lower": "Fraction clipped on lower bound (ratio < 1-ε: tokens being suppressed)",
    },
    "Policy Health": {
        "actor/entropy":    "Token distribution entropy. Higher = more random, lower = more confident. "
                            "Decreasing over training = model sharpening (normal but watch for collapse)",
        "actor/ppo_kl":     "KL divergence vs reference policy (measured even if KL penalty disabled). "
                            "Shows how far policy has drifted from SFT checkpoint",
        "actor/kl_loss":    "KL penalty loss (0.0 if use_kl_loss=False)",
        "actor/grad_norm":  "L2 norm of gradients. Healthy: 0.1-10. Watch for spikes (instability)",
        "actor/lr":         "Current learning rate",
    },
    "Response Quality": {
        "response_length/mean":       "Average response length in tokens",
        "response_length/max":        "Max response length (should equal max_response_length if any hit cap)",
        "response_length/min":        "Min response length",
        "response_length/clip_ratio": "Fraction of responses that hit max_response_length (truncated). "
                                      "High = model tends to generate long outputs that get cut off",
    },
    "Validation": {
        "val-core/vulnerability_detection/acc/mean@1": "Greedy-decoded vulnerability detection accuracy on val set",
    },
    "GRPO Advantages": {
        "critic/advantages/mean": "Mean advantage (should be near 0 for normalized advantages)",
        "critic/advantages/max":  "Max advantage in batch",
        "critic/advantages/min":  "Min advantage (negative = worse than group average)",
    },
    "Performance": {
        "perf/throughput":                 "Tokens processed per second",
        "perf/time_per_step":              "Wall-clock seconds per training step",
        "perf/cpu_memory_used_gb":         "System RAM usage in GB",
        "perf/max_memory_reserved_gb":     "Peak GPU memory reserved in GB",
        "perf/max_memory_allocated_gb":    "Peak GPU memory allocated in GB",
        "timing_s/gen":                    "Time spent on rollout generation per step (s)",
        "timing_s/update_actor":           "Time spent on actor update per step (s)",
        "timing_s/old_log_prob":           "Time spent computing reference log probs per step (s)",
        "timing_s/reward":                 "Time spent computing reward per step (s)",
    },
}

# Metrics to plot (organized by subplot)
PLOT_PANELS = [
    ("Reward (score/mean)", ["critic/score/mean"], {}),
    ("Policy Gradient Loss", ["actor/pg_loss"], {}),
    ("Clip Fraction", ["actor/pg_clipfrac", "actor/pg_clipfrac_lower"], {}),
    ("Entropy", ["actor/entropy"], {}),
    ("KL Divergence (measured)", ["actor/ppo_kl"], {}),
    ("Grad Norm", ["actor/grad_norm"], {}),
    ("Response Length", ["response_length/mean"], {}),
    ("Response Clip Ratio", ["response_length/clip_ratio"], {}),
    ("Val Accuracy", ["val-core/vulnerability_detection/acc/mean@1"], {"marker": "o"}),
    ("Step Time (s)", ["perf/time_per_step"], {}),
    ("Gen / Update / LogProb Time", ["timing_s/gen", "timing_s/update_actor", "timing_s/old_log_prob"], {}),
    ("Memory (GB)", ["perf/cpu_memory_used_gb", "perf/max_memory_reserved_gb"], {}),
]


def fetch_and_merge_runs(entity, project, run_ids):
    """Fetch and concatenate history from multiple wandb runs."""
    api = wandb.Api()
    all_history = []
    all_configs = []

    for run_id in run_ids:
        run_path = f"{entity}/{project}/{run_id}"
        print(f"  Fetching: {run_path}")
        run = api.run(run_path)
        history = run.history(samples=2000)
        config = dict(run.config)
        summary = dict(run.summary)

        all_history.append(history)
        all_configs.append({"run_id": run_id, "config": config, "summary": summary,
                            "name": run.name, "state": run.state, "steps": len(history)})

    # Concatenate histories
    merged = pd.concat(all_history, ignore_index=True)

    # Sort by global_step if available
    if "training/global_step" in merged.columns:
        merged = merged.sort_values("training/global_step").reset_index(drop=True)

    return merged, all_configs


def print_metric_summary(history, metric_name, description=None, indent="  "):
    """Print summary statistics for a single metric."""
    if metric_name not in history.columns:
        return False

    data = history[metric_name].dropna()
    if len(data) == 0:
        return False

    print(f"\n{indent}{metric_name}:")
    if description:
        # Wrap description at ~80 chars
        desc_lines = []
        words = description.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 72:
                desc_lines.append(line)
                line = w
            else:
                line = f"{line} {w}" if line else w
        if line:
            desc_lines.append(line)
        for dl in desc_lines:
            print(f"{indent}  → {dl}")

    print(f"{indent}  points: {len(data)}")
    print(f"{indent}  range:  [{data.min():.6f}, {data.max():.6f}]")
    print(f"{indent}  mean:   {data.mean():.6f} ± {data.std():.6f}")
    print(f"{indent}  first → last: {data.iloc[0]:.6f} → {data.iloc[-1]:.6f}")

    # Show trajectory
    if len(data) >= 6:
        n = len(data)
        indices = [0, 1, n // 4, n // 2, 3 * n // 4, n - 2, n - 1]
        vals = [f"{data.iloc[i]:.4f}" for i in indices if i < n]
        print(f"{indent}  trajectory: [{', '.join(vals)}]")

    return True


def print_analysis(history, configs, verbose=False):
    """Print comprehensive training analysis."""
    print("\n" + "=" * 70)
    print("  veRL GRPO Training Analysis")
    print("=" * 70)

    # Run info
    print("\n── Run Info ──")
    total_steps = 0
    for c in configs:
        state_icon = "✅" if c["state"] == "finished" else "⏸️" if c["state"] == "crashed" else "🔄"
        print(f"  {state_icon} {c['run_id']} ({c['name']}): {c['steps']} steps [{c['state']}]")
        total_steps += c["steps"]
    print(f"  Total steps: {total_steps}")

    # Print each metric group
    for group_name, metrics in METRIC_GROUPS.items():
        printed_any = False
        header_printed = False
        for metric_name, description in metrics.items():
            desc = description if verbose else None
            if not header_printed:
                print(f"\n── {group_name} ──")
                header_printed = True
            ok = print_metric_summary(history, metric_name, desc)
            if ok:
                printed_any = True
        if not printed_any and header_printed:
            print("  (no data)")

    # Diagnostic checks
    print(f"\n── Diagnostics ──")
    issues = []
    suggestions = []

    # Check clipfrac
    if "actor/pg_clipfrac" in history.columns:
        cf = history["actor/pg_clipfrac"].dropna()
        if len(cf) > 0:
            mean_cf = cf.mean()
            if mean_cf < 0.01:
                issues.append(f"⚠️  pg_clipfrac very low ({mean_cf:.4f}) — policy barely updating per step")
                suggestions.append("Consider increasing LR (e.g. 1e-6 → 5e-6)")
                suggestions.append("Consider increasing n (e.g. 4 → 8) for better advantage estimation")
            elif mean_cf > 0.20:
                issues.append(f"⚠️  pg_clipfrac high ({mean_cf:.4f}) — updates may be too aggressive")
                suggestions.append("Consider lowering LR or increasing clip_range")

    # Check reward trend
    if "critic/score/mean" in history.columns:
        scores = history["critic/score/mean"].dropna()
        if len(scores) >= 20:
            first_q = scores.iloc[:len(scores)//4].mean()
            last_q = scores.iloc[-len(scores)//4:].mean()
            delta = last_q - first_q
            if abs(delta) < 0.02:
                issues.append(f"⚠️  Reward essentially flat (Δ={delta:+.4f} over training)")
            elif delta > 0:
                print(f"  ✅ Reward improving: {first_q:.4f} → {last_q:.4f} (Δ={delta:+.4f})")
            else:
                issues.append(f"⚠️  Reward declining: {first_q:.4f} → {last_q:.4f} (Δ={delta:+.4f})")

    # Check entropy collapse
    if "actor/entropy" in history.columns:
        ent = history["actor/entropy"].dropna()
        if len(ent) >= 10:
            if ent.iloc[-1] < ent.iloc[0] * 0.5:
                issues.append("⚠️  Entropy collapsed >50% — possible mode collapse")
            elif ent.iloc[-1] < ent.iloc[0] * 0.8:
                print(f"  ℹ️  Entropy decreased {ent.iloc[0]:.4f} → {ent.iloc[-1]:.4f} (normal sharpening)")

    # Check response length trend
    if "response_length/mean" in history.columns:
        rl = history["response_length/mean"].dropna()
        if len(rl) >= 10:
            rl_delta = rl.iloc[-1] - rl.iloc[0]
            if rl_delta < -500:
                print(f"  ✅ Response length decreasing: {rl.iloc[0]:.0f} → {rl.iloc[-1]:.0f} (model learning conciseness)")
            elif rl_delta > 500:
                issues.append(f"⚠️  Response length increasing: {rl.iloc[0]:.0f} → {rl.iloc[-1]:.0f} (verbose drift)")

    # Check val accuracy
    for val_key in ["val-core/vulnerability_detection/acc/mean@1"]:
        if val_key in history.columns:
            val = history[val_key].dropna()
            if len(val) >= 2:
                print(f"  ℹ️  Val accuracy: {val.iloc[0]:.4f} → {val.iloc[-1]:.4f} (Δ={val.iloc[-1]-val.iloc[0]:+.4f})")

    if issues:
        print()
        for issue in issues:
            print(f"  {issue}")
    if suggestions:
        print()
        for s in suggestions:
            print(f"  💡 {s}")

    if not issues:
        print("  ✅ No issues detected")


def plot_training(history, configs, output_dir):
    """Generate training curve plots."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    # Get step axis
    if "training/global_step" in history.columns:
        steps = history["training/global_step"]
    else:
        steps = history.index

    n_panels = len(PLOT_PANELS)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    run_label = ", ".join(c["run_id"][:8] for c in configs)
    fig.suptitle(f"veRL GRPO Training — {run_label}", fontsize=14, fontweight='bold')

    for idx, (title, metric_keys, kwargs) in enumerate(PLOT_PANELS):
        ax = axes[idx]
        has_data = False
        for mk in metric_keys:
            if mk in history.columns:
                data = history[mk].dropna()
                if len(data) > 0:
                    x = steps.iloc[data.index] if len(steps) == len(history) else data.index
                    label = mk.split("/")[-1] if len(metric_keys) > 1 else None
                    ax.plot(x, data.values, label=label, alpha=0.8, linewidth=1.2, **kwargs)
                    has_data = True

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if len(metric_keys) > 1 and has_data:
            ax.legend(fontsize=7)
        if not has_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, color='gray')

    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "grpo_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved: {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Analyze veRL GRPO training from WandB")
    parser.add_argument("--run-ids", nargs="+", required=True,
                        help="WandB run ID(s). Multiple for resumed runs.")
    parser.add_argument("--entity", default="andrefty-universitatea-politehnica-din-bucuresti",
                        help="WandB entity/team")
    parser.add_argument("--project", default="vulnerability_grpo",
                        help="WandB project name")
    parser.add_argument("--save-plots", action="store_true", help="Save training curve plots")
    parser.add_argument("--save-json", action="store_true", help="Save metrics to JSON")
    parser.add_argument("--output-dir", default="training_grpo/training_analysis",
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show metric descriptions")
    args = parser.parse_args()

    if not HAS_WANDB:
        print("Error: wandb/pandas not installed. Install with: pip install wandb pandas")
        sys.exit(1)

    print(f"Fetching {len(args.run_ids)} run(s) from {args.entity}/{args.project}...")
    history, configs = fetch_and_merge_runs(args.entity, args.project, args.run_ids)

    print_analysis(history, configs, verbose=args.verbose)

    if args.save_plots:
        plot_training(history, configs, args.output_dir)

    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        output = {
            "runs": [{"run_id": c["run_id"], "name": c["name"], "state": c["state"],
                       "steps": c["steps"]} for c in configs],
            "total_steps": sum(c["steps"] for c in configs),
            "final_summary": configs[-1].get("summary", {}),
        }
        # Add trajectory data for key metrics
        key_metrics = ["critic/score/mean", "actor/entropy", "actor/pg_clipfrac",
                        "response_length/mean", "response_length/clip_ratio"]
        output["trajectories"] = {}
        for m in key_metrics:
            if m in history.columns:
                data = history[m].dropna()
                output["trajectories"][m] = {
                    "values": data.tolist(),
                    "steps": history["training/global_step"].iloc[data.index].tolist()
                    if "training/global_step" in history.columns else data.index.tolist(),
                }

        # Remove non-serializable values from summary
        filtered_summary = {}
        for k, v in output["final_summary"].items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                filtered_summary[k] = v
            except (TypeError, ValueError):
                filtered_summary[k] = str(v)
        output["final_summary"] = filtered_summary

        json_path = os.path.join(args.output_dir, "grpo_training_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  📄 JSON saved: {json_path}")


if __name__ == "__main__":
    main()
