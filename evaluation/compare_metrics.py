import json
import os
import glob
import argparse

def get_metrics(base_dir, run_type="all", filter_str=None):
    results = []
    
    # Normalize run_type to a list of strings
    if isinstance(run_type, str):
        run_type = [t.strip() for t in run_type.split(",")]
    
    # Determine search patterns
    patterns = []
    if "all" in run_type:
        patterns.append("*")
    else:
        if "sft" in run_type:
            patterns.append("sft_*")
        if "grpo" in run_type:
            patterns.append("grpo_*")
        if "base" in run_type:
            patterns.append("*base*")
        
    directories = []
    for pattern in patterns:
        search_pattern = os.path.join(base_dir, pattern)
        directories.extend(glob.glob(search_pattern))

    for d in sorted(directories):
        run_name_base = os.path.basename(d)
        
        # Apply filter on run name if specified
        if filter_str and filter_str.lower() not in run_name_base.lower():
            continue
            
        # Look for any metrics folder, including multitemp ones like metrics_temp0p2
        metrics_dirs = glob.glob(os.path.join(d, "metrics*"))
        for m_dir in sorted(metrics_dirs):
            metrics_path = os.path.join(m_dir, "metrics_combined_training.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        data = json.load(f)
                    
                    run_name = run_name_base
                    metrics_dirname = os.path.basename(m_dir)
                    if metrics_dirname != "metrics":
                        run_name += f" ({metrics_dirname.replace('metrics_', '')})"
                        
                    # Also apply filter on combined run_name just in case
                    if filter_str and filter_str.lower() not in run_name.lower():
                        continue
                    
                    # Extracting relevant metrics
                    metrics = {
                        "Run": run_name,
                        "Acc": data.get('standard_metrics', {}).get('accuracy', 0),
                        "F1": data.get('standard_metrics', {}).get('f1_score', 0),
                        "P-C": data.get('pairwise_metrics', {}).get('P-C_ratio', 0),
                        "P-V": data.get('pairwise_metrics', {}).get('P-V_ratio', 0),
                        "P-B": data.get('pairwise_metrics', {}).get('P-B_ratio', 0),
                        "P-R": data.get('pairwise_metrics', {}).get('P-R_ratio', 0),
                        "Loc F1": data.get('line_localization_metrics', {}).get('avg_f1', 0),
                        #"Loc P/R": f"{data.get('line_localization_metrics', {}).get('avg_precision', 0):.6f}/{data.get('line_localization_metrics', {}).get('avg_recall', 0):.6f}",
                        "Parseable": f"{data.get('parseable_samples', 0)}/{data.get('total_samples', 0)}"
                    }
                    results.append(metrics)
                except Exception as e:
                    print(f"Error reading {metrics_path}: {e}")
                    
    return results

def print_table(results, sort_by=None, limit=None, reverse=False):
    if not results:
        print("No results found.")
        return
        
    if sort_by:
        if sort_by not in results[0]:
            print(f"Warning: Sort metric '{sort_by}' not found. Valid metrics are: {', '.join(results[0].keys())}")
        else:
            def sort_key(row):
                val = row.get(sort_by, 0)
                # Handle sorting for string fractions like '100/100' or '0.2/0.3'
                if isinstance(val, str) and "/" in val and sort_by != "Run":
                    parts = val.split("/")
                    try:
                        return float(parts[0])
                    except ValueError:
                        pass
                return val if val is not None else 0
                
            results = sorted(results, key=sort_key, reverse=reverse)
            
    if limit is not None:
        results = results[:limit]
        
    # Format floats for display
    formatted_results = []
    for r in results:
        formatted_row = {}
        for k, v in r.items():
            if isinstance(v, float):
                formatted_row[k] = f"{v:.6f}"
            else:
                formatted_row[k] = v
        formatted_results.append(formatted_row)
        
    try:
        from tabulate import tabulate
        print(tabulate(formatted_results, headers="keys", tablefmt="github"))
    except ImportError:
        # Fallback to simple markdown formatting if tabulate is not available
        headers = formatted_results[0].keys()
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        rows = []
        for r in formatted_results:
            row = "| " + " | ".join(str(r[h]) for h in headers) + " |"
            rows.append(row)
        
        print("\n".join([header_row, separator_row] + rows))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across SFT and GRPO models.")
    parser.add_argument("--base_dir", type=str, default="/export/home/acs/stud/t/tudor.farcasanu/SSL_research/evaluation_results", help="Base directory containing results")
    parser.add_argument("--type", action="append", help="Type of runs to include (sft, grpo, base, all). Can be specified multiple times or comma-separated (default: all)")
    parser.add_argument("--filter", type=str, default=None, help="String to filter run names by (e.g. 'qwen3')")
    parser.add_argument("--sort", type=str, default=None, help="Metric to sort by (e.g., 'Acc', 'F1', 'P-C', 'Run')")
    parser.add_argument("--desc", action="store_true", help="Sort in descending order (useful for sorting by best metric first)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of results to show")
    
    args = parser.parse_args()
    
    # Normalize types
    types = []
    if args.type:
        for t in args.type:
            types.extend([x.strip() for x in t.split(",")])
    else:
        types = ["all"]
        
    results = get_metrics(args.base_dir, run_type=types, filter_str=args.filter)
    print_table(results, sort_by=args.sort, limit=args.limit, reverse=args.desc)
