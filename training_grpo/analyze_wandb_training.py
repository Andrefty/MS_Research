#!/usr/bin/env python3
"""
Analyze GRPO training metrics from WandB.
Fetches training history and provides statistics/visualizations.

Usage:
    python analyze_wandb_training.py [--save-plots]
"""

import argparse
import json
try:
    import wandb
    import pandas as pd
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Note: wandb/pandas not installed. Install with: pip install wandb pandas")

def fetch_run_data(run_path):
    """Fetch run data from WandB API."""
    api = wandb.Api()
    run = api.run(run_path)
    
    # Get config
    config = dict(run.config)
    
    # Get history (all logged metrics)
    history = run.history()
    
    # Get summary (final values)
    summary = dict(run.summary)
    
    return config, history, summary

def analyze_rewards(history):
    """Analyze reward metrics from training history."""
    # Key metrics for GRPO
    metrics_to_analyze = [
        'train/reward',
        'train/rewards/reward_fn/mean',
        'train/rewards/reward_fn/std',
        'train/loss',
        'train/entropy',
        'train/learning_rate',
        'train/grad_norm',
    ]
    
    results = {}
    for metric in metrics_to_analyze:
        if metric in history.columns:
            data = history[metric].dropna()
            if len(data) > 0:
                results[metric] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'first': float(data.iloc[0]),
                    'last': float(data.iloc[-1]),
                    'count': len(data),
                }
    
    return results

def explain_grpo_metrics():
    """Explain what the GRPO metrics mean."""
    explanation = """
=== GRPO Training Metrics Explanation ===

UNDERSTANDING THE REWARD VALUES:
--------------------------------
The reward you see (~0.17) is the AVERAGE reward across all samples in a batch.
This is NOT the raw reward values (0.0, 0.3, 0.6, 1.0) directly.

Here's why:
1. Each sample gets one of 4 rewards: 0.0, 0.3, 0.6, or 1.0
2. GRPO generates multiple completions per prompt (num_generations=4)
3. The logged 'reward' is the mean across the batch

For example, if in one batch:
- 2 samples got 0.0 (wrong classification, no line matches)
- 1 sample got 0.3 (wrong classification but some lines correct)  
- 1 sample got 0.6 (correct classification but <50% lines)
- 1 sample got 1.0 (correct classification + >50% lines)

Average = (0 + 0 + 0.3 + 0.6 + 1.0) / 5 = 0.38

So seeing 0.17 means roughly:
- If samples rarely get 1.0 or 0.6, and mostly get 0.0 with occasional 0.3
- (0.0 * 4 + 0.3 * 1) / 5 ≈ 0.06 would be very bad
- (0.0 * 3 + 0.3 * 2) / 5 ≈ 0.12 still poor
- (0.0 * 2 + 0.3 * 2 + 0.6 * 1) / 5 ≈ 0.24 getting better

A reward of 0.17 suggests:
- Most samples are getting 0.0 or 0.3
- Very few are achieving the full 1.0 reward
- The model struggles with either classification OR line localization

WHY THE REWARD IS FLAT:
-----------------------
1. Task is hard: Vulnerability detection requires semantic understanding
2. Sparse reward: Only 4 discrete values, hard to get incremental signal
3. One epoch only: Model may need more training to improve
4. Cold start: SFT model may not have learned task well enough

WHAT WOULD "GOOD" LOOK LIKE:
----------------------------
- Reward increasing from ~0.17 to ~0.4-0.5 over training
- Would indicate model learning to get more 0.6 and 1.0 rewards
- Final reward of 0.7+ would suggest strong task performance
"""
    return explanation

def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO training metrics from WandB")
    parser.add_argument("--run-path",
                        help="WandB run path")
    parser.add_argument("--save-json", action="store_true", help="Save metrics to JSON")
    parser.add_argument("--output-dir", default=".", help="Output directory for files")
    args = parser.parse_args()
    
    if not HAS_WANDB:
        print("WandB not available. Printing explanation only.\n")
        print(explain_grpo_metrics())
        return
    
    print(f"Fetching data from WandB run: {args.run_path}")
    print("-" * 60)
    
    try:
        config, history, summary = fetch_run_data(args.run_path)
    except Exception as e:
        print(f"Error fetching WandB data: {e}")
        print("\nMake sure you're logged in: wandb login")
        print(explain_grpo_metrics())
        return
    
    # Print config
    print("\n=== Training Configuration ===")
    important_config = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size',
                        'gradient_accumulation_steps', 'num_generations', 'beta', 
                        'temperature', 'top_p', 'top_k', 'max_length']
    for key in important_config:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # Analyze rewards
    print("\n=== Reward Statistics ===")
    reward_stats = analyze_rewards(history)
    
    for metric, stats in reward_stats.items():
        print(f"\n{metric}:")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  First → Last: {stats['first']:.4f} → {stats['last']:.4f}")
        print(f"  Count: {stats['count']}")
    
    # Print explanation
    print(explain_grpo_metrics())
    
    # Save to JSON if requested
    if args.save_json:
        output = {
            'config': config,
            'reward_stats': reward_stats,
            'summary': {k: v for k, v in summary.items() if not k.startswith('_')}
        }
        output_path = f"{args.output_dir}/grpo_training_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved analysis to {output_path}")

if __name__ == "__main__":
    main()
