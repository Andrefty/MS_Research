#!/usr/bin/env python3
"""
Reward function for GRPO training.
Computes multi-level rewards based on classification accuracy and line matching.

Supports both:
- TRL GRPOTrainer (reward_function_for_grpo)
- veRL trainer (compute_score)
"""

import sys
import os
import json

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, List, Optional
from utils.response_parser import parse_for_reward as parse_model_response
from utils.response_parser import parse_model_response as parse_model_response_full



def compute_line_accuracy(predicted_lines: List[int], ground_truth_lines: List[int]) -> float:
    """
    Compute overlap accuracy between predicted and ground truth lines.
    
    Returns:
        Accuracy as float between 0.0 and 1.0
    """
    if not ground_truth_lines:
        # No ground truth lines (e.g., patched sample)
        # If model also predicts no lines, that's correct
        return 1.0 if not predicted_lines else 0.0
    
    if not predicted_lines:
        return 0.0
    
    overlap = len(set(predicted_lines) & set(ground_truth_lines))
    return overlap / len(ground_truth_lines)


def compute_reward(
    response: str,
    is_vulnerable: bool,
    ground_truth_lines: List[int],
    _log_first: list = [True],  # Mutable default to track first call
) -> float:
    """
    Compute multi-level reward for GRPO training.
    
    Reward tiers:
    - 1.0: Correct classification + ≥50% vulnerable lines correct
    - 0.6: Correct classification only
    - 0.3: Some correct lines but wrong classification
    - 0.05: Response has parsable JSON but wrong classification and no correct lines
    - 0.0: No parsable JSON / completely wrong format
    
    Args:
        response: Model's generated response text
        is_vulnerable: Ground truth - whether the code is vulnerable
        ground_truth_lines: List of line numbers that are vulnerable
        
    Returns:
        Reward value between 0.0 and 1.0
    """
    # Parse response - get full result for status check
    result = parse_model_response_full(response)
    classification = result.classification
    predicted_lines = result.vulnerable_lines
    
    # Debug: log first few calls to show exactly what we're parsing
    if _log_first[0]:
        _log_first[0] = False
        print(f"[compute_reward DEBUG] Response length: {len(response)}", flush=True)
        print(f"[compute_reward DEBUG] Response repr: {repr(response[:200])}", flush=True)
        print(f"[compute_reward DEBUG] Parse status: {result.status}", flush=True)
        print(f"[compute_reward DEBUG] Parsed classification: {classification}", flush=True)
        print(f"[compute_reward DEBUG] Parsed lines: {predicted_lines}", flush=True)
    
    # If no parsable classification at all
    if classification is None:
        # Check if there's at least valid JSON structure
        if result.status in ["VALID", "INVALID_CLASSIFICATION"]:
            return 0.05  # Has JSON but classification is invalid
        return 0.0  # No JSON at all or incomplete
    
    # Check classification correctness
    expected_class = "VULNERABLE" if is_vulnerable else "NOT_VULNERABLE"
    correct_classification = (classification == expected_class)
    
    # Compute line accuracy
    line_accuracy = compute_line_accuracy(predicted_lines, ground_truth_lines)
    
    # Compute reward based on tier
    if correct_classification and line_accuracy >= 0.5:
        return 1.0
    elif correct_classification:
        return 0.6
    elif line_accuracy > 0:
        return 0.3
    else:
        return 0.05  # Valid format but wrong classification and no correct lines


def batch_compute_rewards(
    responses: List[str],
    is_vulnerable_list: List[bool],
    ground_truth_lines_list: List[List[int]],
) -> List[float]:
    """
    Compute rewards for a batch of responses.
    
    Args:
        responses: List of model responses
        is_vulnerable_list: List of ground truth vulnerability labels
        ground_truth_lines_list: List of ground truth line lists
        
    Returns:
        List of reward values
    """
    return [
        compute_reward(resp, is_vuln, gt_lines)
        for resp, is_vuln, gt_lines in zip(responses, is_vulnerable_list, ground_truth_lines_list)
    ]


# For use with trl GRPOTrainer
# Completion logging for debugging
import os
import threading
import json as json_module  # Alias to avoid conflict
from datetime import datetime

_completion_log_lock = threading.Lock()
_step_counter = [0]  # Mutable to track steps across calls
_first_call_logged = [False]  # Track if we've logged first call


def get_log_path():
    """Get log path from env var, checked each time."""
    return os.environ.get('GRPO_COMPLETION_LOG', None)


def log_completions_batch(completions: List[str], rewards: List[float], 
                          is_vulnerable_list: List[bool], ground_truth_lines_list: List[List[int]]):
    """Log a batch of completions to file for debugging."""
    log_path = get_log_path()
    
    # Always print first call info for debugging
    if not _first_call_logged[0]:
        _first_call_logged[0] = True
        print(f"[reward_function] First call received {len(completions)} completions")
        print(f"[reward_function] GRPO_COMPLETION_LOG = {log_path}")
        if completions:
            print(f"[reward_function] First completion length: {len(completions[0])}")
            print(f"[reward_function] First completion preview: {completions[0][:200]}...")
    
    if not log_path:
        return
    
    _step_counter[0] += 1
    current_step = _step_counter[0]
    
    try:
        with _completion_log_lock:
            with open(log_path, 'a', encoding='utf-8') as f:
                for i, (comp, reward, is_vuln, gt_lines) in enumerate(
                    zip(completions, rewards, is_vulnerable_list, ground_truth_lines_list)
                ):
                    log_entry = {
                        "step": current_step,
                        "idx": i,
                        "timestamp": datetime.now().isoformat(),
                        "is_vulnerable": is_vuln,
                        "ground_truth_lines": gt_lines,
                        "completion_length": len(comp) if comp else 0,
                        "completion": comp,
                        "reward": reward,
                    }
                    f.write(json_module.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[reward_function] Warning: Failed to log completion: {e}")


def reward_function_for_grpo(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function compatible with trl GRPOTrainer.
    
    Expects kwargs to contain:
    - is_vulnerable: List[bool]
    - ground_truth_lines: List[List[int]]
    
    Returns:
        List of reward floats
    """
    is_vulnerable_list = kwargs.get('is_vulnerable', [True] * len(completions))
    ground_truth_lines_list = kwargs.get('ground_truth_lines', [[]] * len(completions))
    
    rewards = batch_compute_rewards(completions, is_vulnerable_list, ground_truth_lines_list)
    
    # Log completions for debugging
    log_completions_batch(completions, rewards, is_vulnerable_list, ground_truth_lines_list)
    
    return rewards


# =============================================================================
# veRL-compatible reward function
# =============================================================================

_verl_call_counter = [0]
_verl_first_call_logged = [False]


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    veRL-compatible reward function.
    
    This is the interface veRL expects for custom reward functions.
    
    Args:
        data_source: Dataset identifier (e.g., "vulnerability_detection")
        solution_str: Model's generated response text
        ground_truth: JSON string containing is_vulnerable and ground_truth_lines
        extra_info: Optional additional metadata
    
    Returns:
        float: Reward value (0.0, 0.05, 0.3, 0.6, or 1.0)
    """
    # Parse ground truth
    try:
        gt = json.loads(ground_truth)
        is_vulnerable = gt.get("is_vulnerable", True)
        gt_lines = gt.get("ground_truth_lines", [])
    except (json.JSONDecodeError, TypeError):
        # Fallback if ground truth parsing fails
        is_vulnerable = True
        gt_lines = []
    
    # Compute reward
    reward = compute_reward(solution_str, is_vulnerable, gt_lines)
    
    # Log completion for debugging
    _verl_call_counter[0] += 1
    log_path = get_log_path()
    
    # Log first call info
    if not _verl_first_call_logged[0]:
        _verl_first_call_logged[0] = True
        print(f"[veRL compute_score] First call!", flush=True)
        print(f"[veRL compute_score] GRPO_COMPLETION_LOG = {log_path}", flush=True)
        print(f"[veRL compute_score] Completion length: {len(solution_str)}", flush=True)
        print(f"[veRL compute_score] Completion preview: {solution_str[:300]}...", flush=True)
        print(f"[veRL compute_score] Reward: {reward}", flush=True)
    
    # Write to log file if configured
    if log_path:
        try:
            with _completion_log_lock:
                with open(log_path, 'a', encoding='utf-8') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    safe_gt_lines = [int(x) for x in gt_lines] if gt_lines else []
                    safe_reward = float(reward)
                    log_entry = {
                        "call_num": _verl_call_counter[0],
                        "timestamp": datetime.now().isoformat(),
                        "data_source": data_source,
                        "is_vulnerable": bool(is_vulnerable),
                        "ground_truth_lines": safe_gt_lines,
                        "completion_length": len(solution_str) if solution_str else 0,
                        "completion": solution_str,
                        "reward": safe_reward,
                        "extra_info": str(extra_info) if extra_info else None,
                    }
                    f.write(json_module.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            if _verl_call_counter[0] <= 5:  # Only warn first few times
                print(f"[veRL compute_score] Warning: Failed to log: {e}", flush=True)
    
    return reward


if __name__ == "__main__":
    # Test cases for tiered reward system
    test_cases = [
        # Correct classification + correct lines -> 1.0
        ('<think>analysis</think>{"classification": "VULNERABLE", "vulnerable_lines": [5, 7, 10], "reasoning_summary": "..."}',
         True, [5, 7], 1.0),
        
        # Correct classification + partial lines -> 0.6
        ('<think>analysis</think>{"classification": "VULNERABLE", "vulnerable_lines": [5], "reasoning_summary": "..."}',
         True, [5, 7, 10], 0.6),
        
        # Correct classification, no lines needed -> 1.0
        ('<think>analysis</think>{"classification": "NOT_VULNERABLE", "vulnerable_lines": [], "reasoning_summary": "..."}',
         False, [], 1.0),
        
        # Wrong classification, some lines correct -> 0.3
        ('<think>analysis</think>{"classification": "NOT_VULNERABLE", "vulnerable_lines": [5, 7], "reasoning_summary": "..."}',
         True, [5, 7, 10], 0.3),
        
        # Wrong classification, no correct lines, but valid JSON -> 0.05
        ('<think>analysis</think>{"classification": "VULNERABLE", "vulnerable_lines": [1, 2], "reasoning_summary": "..."}',
         False, [], 0.05),
        
        # No JSON, incomplete thinking -> 0.0
        ('<think>still thinking about VULNERABLE code...',
         True, [5], 0.0),
    ]
    
    print("Testing reward function:")
    for response, is_vuln, gt_lines, expected in test_cases:
        reward = compute_reward(response, is_vuln, gt_lines)
        status = "✓" if abs(reward - expected) < 0.01 else "✗"
        print(f"  {status} Expected {expected}, got {reward}")
    
    # Test veRL compute_score interface
    print("\nTesting veRL compute_score interface:")
    ground_truth = json.dumps({"is_vulnerable": True, "ground_truth_lines": [5, 7]})
    response = '<think>done</think>{"classification": "VULNERABLE", "vulnerable_lines": [5, 7], "reasoning_summary": "test"}'
    score = compute_score("vulnerability_detection", response, ground_truth)
    print(f"  Score: {score} (expected 1.0)")

