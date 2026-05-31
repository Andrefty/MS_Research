import json
import os
import glob
from pathlib import Path

def get_metrics(base_dir):
    results = []
    
    # Pattern to match sft_qwen3_4b directories
    search_pattern = os.path.join(base_dir, "grpo_qwen3_4b*")
    directories = glob.glob(search_pattern)

    # # Get all subdirectories in base_dir
    # directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for d in sorted(directories):
        # Look for any metrics folder, including multitemp ones like metrics_temp0p2
        metrics_dirs = glob.glob(os.path.join(d, "metrics*"))
        for m_dir in sorted(metrics_dirs):
            metrics_path = os.path.join(m_dir, "metrics_combined_training.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        data = json.load(f)
                    
                    run_name = os.path.basename(d)
                    metrics_dirname = os.path.basename(m_dir)
                    if metrics_dirname != "metrics":
                        run_name += f" ({metrics_dirname.replace('metrics_', '')})"
                    
                    # Extracting relevant metrics
                    metrics = {
                        "Run": run_name,
                        "Acc": f"{data.get('standard_metrics', {}).get('accuracy', 0):.4f}",
                        "F1": f"{data.get('standard_metrics', {}).get('f1_score', 0):.4f}",
                        "P-C": f"{data.get('pairwise_metrics', {}).get('P-C_ratio', 0):.4f}",
                        "P-V": f"{data.get('pairwise_metrics', {}).get('P-V_ratio', 0):.4f}",
                        "P-B": f"{data.get('pairwise_metrics', {}).get('P-B_ratio', 0):.4f}",
                        "P-R": f"{data.get('pairwise_metrics', {}).get('P-R_ratio', 0):.4f}",
                        "Loc F1": f"{data.get('line_localization_metrics', {}).get('avg_f1', 0):.4f}",
                        "Loc P/R": f"{data.get('line_localization_metrics', {}).get('avg_precision', 0):.4f}/{data.get('line_localization_metrics', {}).get('avg_recall', 0):.4f}",
                        "Parseable": f"{data.get('parseable_samples', 0)}/{data.get('total_samples', 0)}"
                    }
                    results.append(metrics)
                except Exception as e:
                    print(f"Error reading {metrics_path}: {e}")
                    
    return results

def print_table(results):
    if not results:
        print("No results found.")
        return
    
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="github"))
    except ImportError:
        # Fallback to simple markdown formatting if tabulate is not available
        headers = results[0].keys()
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        rows = []
        for r in results:
            row = "| " + " | ".join(str(r[h]) for h in headers) + " |"
            rows.append(row)
        
        print("\n".join([header_row, separator_row] + rows))

if __name__ == "__main__":
    base_dir = "/export/home/acs/stud/t/tudor.farcasanu/SSL_research/evaluation_results"
    results = get_metrics(base_dir)
    print_table(results)
