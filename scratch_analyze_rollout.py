import pandas as pd
import numpy as np
import os
import glob
import json
import ast
import sys
import argparse
sys.path.insert(0, '/export/home/acs/stud/t/tudor.farcasanu/SSL_research')
from utils.response_parser import parse_model_response

def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO rollout line prediction distributions.")
    parser.add_argument('--rollout_dir', type=str, required=True, help="Path to the train_rollout directory.")
    args = parser.parse_args()
    
    print(f"\nAnalyzing Rollout from: {args.rollout_dir}")
    jsonl_files = glob.glob(os.path.join(args.rollout_dir, '*.jsonl'))
    
    if not jsonl_files:
        print("No rollout jsonl files found.")
        return
        
    def get_step_num(f):
        try: return int(os.path.basename(f).split('.')[0])
        except: return -1
        
    jsonl_files.sort(key=get_step_num)
    print(f"Found {len(jsonl_files)} rollout files. Aggregating data across all steps...")
    
    vuln_data = []
    patched_data = []
    
    total_files_processed = 0
    
    for file_path in jsonl_files:
        total_files_processed += 1
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                
                gt_str = row.get('gts', '{}')
                try:
                    gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
                except:
                    gt = {}
                    
                is_vuln = gt.get('is_vulnerable', True)
                
                gt_lines_list = gt.get('ground_truth_lines', [])
                if gt_lines_list is None: gt_lines_list = []
                num_gt_lines = len(gt_lines_list)
                
                response = row.get('output', '')
                if isinstance(response, (list, np.ndarray)) and len(response) > 0:
                    response = response[0]
                    
                res = parse_model_response(response, include_raw_json=True)
                
                lines = []
                if res.status == 'VALID' and res.raw_json:
                    lines = res.raw_json.get('important_lines', res.raw_json.get('vulnerable_lines', []))
                
                num_pred_lines = len(lines) if isinstance(lines, list) else 0
                
                if is_vuln:
                    vuln_data.append({'gt_lines': num_gt_lines, 'pred_lines': num_pred_lines})
                else:
                    patched_data.append({'gt_lines': num_gt_lines, 'pred_lines': num_pred_lines})
                        
    total_samples = len(vuln_data) + len(patched_data)
    if total_samples == 0:
        print("No valid data parsed.")
        return
        
    vuln_gt_array = [d['gt_lines'] for d in vuln_data]
    patched_gt_array = [d['gt_lines'] for d in patched_data]
    
    vuln_99th = np.percentile(vuln_gt_array, 99) if vuln_gt_array else 0
    patched_99th = np.percentile(patched_gt_array, 99) if patched_gt_array else 0
    all_99th = np.percentile(vuln_gt_array + patched_gt_array, 99)
    
    # Calculate spray counts using the run's own 99th percentile threshold
    vuln_spray_count = sum(1 for d in vuln_data if d['pred_lines'] > all_99th)
    patched_spray_count = sum(1 for d in patched_data if d['pred_lines'] > all_99th)
                
    print(f"\nData-specific 99th Percentile GT Lines - Vuln: {vuln_99th:.2f}, Patched: {patched_99th:.2f}, All: {all_99th:.2f}")
    
    print(f"\nRollout Stats (Across {total_files_processed} files):")
    print(f"Total Rollout Samples: {total_samples}")
    
    if vuln_data:
        vuln_preds = [d['pred_lines'] for d in vuln_data]
        print(f"\nVulnerable Samples (n={len(vuln_data)}):")
        print(f"Average Predicted Lines: {np.mean(vuln_preds):.2f}")
        print(f"Max Predicted Lines: {np.max(vuln_preds)}")
        print(f"Sprayed (> {all_99th:.2f} lines): {vuln_spray_count} ({(vuln_spray_count/len(vuln_data))*100:.2f}%)")
        
    if patched_data:
        patched_preds = [d['pred_lines'] for d in patched_data]
        print(f"\nPatched Samples (n={len(patched_data)}):")
        print(f"Average Predicted Lines: {np.mean(patched_preds):.2f}")
        print(f"Max Predicted Lines: {np.max(patched_preds)}")
        print(f"Sprayed (> {all_99th:.2f} lines): {patched_spray_count} ({(patched_spray_count/len(patched_data))*100:.2f}%)")

if __name__ == "__main__":
    main()
