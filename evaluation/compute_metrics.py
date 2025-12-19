#!/usr/bin/env python3
"""
compute_metrics.py - Compute evaluation metrics from inference responses.

Computes:
- Standard metrics: Accuracy, Precision, Recall, F1
- Pair-wise metrics: P-C, P-V, P-B, P-R
- Line localization metrics (for training prompt)

Outputs separate metrics for:
- PrimeVul test only (comparison with Semester 2)
- Combined dataset (full evaluation)

Usage:
    python compute_metrics.py \
        --input_file eval_responses.jsonl \
        --output_dir metrics/
"""

import json
import argparse
import os
import re
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def parse_prediction_training(response_text):
    """Parse training-format response (JSON output)."""
    if not response_text or response_text.startswith("ERROR:"):
        return None, None
    
    response = response_text.lower()
    
    # Remove thinking block
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    
    # Try to find JSON
    json_match = re.search(r'\{[^{}]*"classification"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            classification = data.get('classification', '').upper()
            vulnerable_lines = data.get('vulnerable_lines', [])
            
            if 'NOT_VULNERABLE' in classification or 'NOT VULNERABLE' in classification:
                return 0, vulnerable_lines
            elif 'VULNERABLE' in classification:
                return 1, vulnerable_lines
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: text patterns
    if 'not_vulnerable' in response or 'not vulnerable' in response:
        return 0, []
    elif 'vulnerable' in response:
        return 1, []
    
    return None, None


def parse_prediction_std_cls(response_text):
    """Parse std_cls format response (YES/NO)."""
    if not response_text or response_text.startswith("ERROR:"):
        return None
    
    response = response_text.lower().strip()
    
    # Remove thinking block
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    
    if not response:
        return None
    
    # Check for patterns
    if "(1) yes" in response:
        return 1
    elif "(2) no" in response:
        return 0
    elif "yes" in response:
        return 1
    elif "no" in response:
        return 0
    
    return None


def compute_line_metrics(ground_truth_lines, predicted_lines):
    """Compute precision/recall for line localization."""
    if not ground_truth_lines:
        return {'precision': None, 'recall': None, 'f1': None}
    
    # Extract line numbers from ground truth (which has dicts)
    gt_line_nos = set()
    for item in ground_truth_lines:
        if isinstance(item, dict) and 'line_no' in item:
            gt_line_nos.add(item['line_no'])
        elif isinstance(item, (int, float)):
            gt_line_nos.add(int(item))
    
    if not gt_line_nos:
        return {'precision': None, 'recall': None, 'f1': None}
    
    # Handle predicted lines
    pred_line_nos = set()
    if predicted_lines:
        for line in predicted_lines:
            if isinstance(line, (int, float)):
                pred_line_nos.add(int(line))
    
    if not pred_line_nos:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Compute metrics
    true_positives = len(gt_line_nos & pred_line_nos)
    precision = true_positives / len(pred_line_nos) if pred_line_nos else 0
    recall = true_positives / len(gt_line_nos) if gt_line_nos else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_metrics_for_subset(responses, prompt_type, subset_name):
    """Compute all metrics for a subset of responses."""
    
    # Parse predictions
    true_labels = []
    predicted_labels = []
    
    # For pair-wise metrics
    samples_by_pair = defaultdict(list)
    
    # For line metrics
    line_metrics_list = []
    
    unparseable = 0
    
    for resp in responses:
        target = resp['target']
        response_text = resp['response']
        
        if prompt_type == "training":
            pred, pred_lines = parse_prediction_training(response_text)
        else:
            pred = parse_prediction_std_cls(response_text)
            pred_lines = None
        
        if pred is None:
            unparseable += 1
            continue
        
        true_labels.append(target)
        predicted_labels.append(pred)
        
        # For pair-wise
        pair_id = resp.get('pair_id', resp['commit_id'])
        samples_by_pair[pair_id].append({
            'target': target,
            'predicted': pred,
            'is_vulnerable': resp['is_vulnerable']
        })
        
        # For line metrics (only training prompt, only vulnerable samples)
        if prompt_type == "training" and resp['is_vulnerable'] and pred == 1:
            gt_lines = resp.get('ground_truth_lines', [])
            if gt_lines:
                lm = compute_line_metrics(gt_lines, pred_lines or [])
                if lm['precision'] is not None:
                    line_metrics_list.append(lm)
    
    # Standard metrics
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    else:
        accuracy = precision = recall = f1 = 0.0
    
    # Pair-wise metrics
    pc_count, pv_count, pb_count, pr_count = 0, 0, 0, 0
    valid_pairs = 0
    
    for pair_id, items in samples_by_pair.items():
        # Need exactly one vuln and one patched
        vuln_items = [i for i in items if i['is_vulnerable']]
        benign_items = [i for i in items if not i['is_vulnerable']]
        
        if len(vuln_items) == 1 and len(benign_items) == 1:
            pred_vuln = vuln_items[0]['predicted']
            pred_benign = benign_items[0]['predicted']
            valid_pairs += 1
            
            if pred_vuln == 1 and pred_benign == 0:
                pc_count += 1
            elif pred_vuln == 1 and pred_benign == 1:
                pv_count += 1
            elif pred_vuln == 0 and pred_benign == 0:
                pb_count += 1
            elif pred_vuln == 0 and pred_benign == 1:
                pr_count += 1
    
    # Line localization metrics (average)
    avg_line_precision = None
    avg_line_recall = None
    avg_line_f1 = None
    if line_metrics_list:
        avg_line_precision = sum(m['precision'] for m in line_metrics_list) / len(line_metrics_list)
        avg_line_recall = sum(m['recall'] for m in line_metrics_list) / len(line_metrics_list)
        avg_line_f1 = sum(m['f1'] for m in line_metrics_list) / len(line_metrics_list)
    
    return {
        "subset": subset_name,
        "prompt_type": prompt_type,
        "total_samples": len(responses),
        "parseable_samples": len(true_labels),
        "unparseable_samples": unparseable,
        "standard_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "pairwise_metrics": {
            "total_valid_pairs": valid_pairs,
            "P-C_count": pc_count,
            "P-C_ratio": pc_count / valid_pairs if valid_pairs > 0 else 0,
            "P-V_count": pv_count,
            "P-V_ratio": pv_count / valid_pairs if valid_pairs > 0 else 0,
            "P-B_count": pb_count,
            "P-B_ratio": pb_count / valid_pairs if valid_pairs > 0 else 0,
            "P-R_count": pr_count,
            "P-R_ratio": pr_count / valid_pairs if valid_pairs > 0 else 0
        },
        "line_localization_metrics": {
            "num_samples": len(line_metrics_list),
            "avg_precision": avg_line_precision,
            "avg_recall": avg_line_recall,
            "avg_f1": avg_line_f1
        } if prompt_type == "training" else None
    }


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to eval responses from run_eval.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save metrics JSON files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load responses
    print(f"Loading responses: {args.input_file}")
    responses = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    print(f"Loaded {len(responses)} responses")
    
    # Group by prompt type
    by_prompt = defaultdict(list)
    for resp in responses:
        by_prompt[resp['prompt_type']].append(resp)
    
    print(f"Prompt types: {list(by_prompt.keys())}")
    
    # Compute metrics for each prompt type
    for prompt_type in by_prompt:
        prompt_responses = by_prompt[prompt_type]
        
        # 1. PrimeVul test only
        primevul_test = [r for r in prompt_responses if r['split'] == 'primevul_test']
        if primevul_test:
            metrics = compute_metrics_for_subset(primevul_test, prompt_type, "primevul_test")
            output_file = os.path.join(args.output_dir, f"metrics_primevul_test_{prompt_type}.json")
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved: {output_file}")
            print(f"  Accuracy: {metrics['standard_metrics']['accuracy']:.4f}")
            print(f"  F1: {metrics['standard_metrics']['f1_score']:.4f}")
            print(f"  P-C: {metrics['pairwise_metrics']['P-C_ratio']:.4f}")
            print(f"  P-V: {metrics['pairwise_metrics']['P-V_ratio']:.4f}")
            print(f"  P-B: {metrics['pairwise_metrics']['P-B_ratio']:.4f}")
            print(f"  P-R: {metrics['pairwise_metrics']['P-R_ratio']:.4f}")
        
        # 2. Combined (all data)
        metrics = compute_metrics_for_subset(prompt_responses, prompt_type, "combined")
        output_file = os.path.join(args.output_dir, f"metrics_combined_{prompt_type}.json")
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {output_file}")
        print(f"  Accuracy: {metrics['standard_metrics']['accuracy']:.4f}")
        print(f"  F1: {metrics['standard_metrics']['f1_score']:.4f}")
        print(f"  P-C: {metrics['pairwise_metrics']['P-C_ratio']:.4f}")
        print(f"  P-V: {metrics['pairwise_metrics']['P-V_ratio']:.4f}")
        print(f"  P-B: {metrics['pairwise_metrics']['P-B_ratio']:.4f}")
        print(f"  P-R: {metrics['pairwise_metrics']['P-R_ratio']:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
