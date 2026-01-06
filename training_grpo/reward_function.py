#!/usr/bin/env python3
"""
Reward function for GRPO training.
Computes multi-level rewards based on classification accuracy and line matching.
"""

import sys
import os

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, List, Optional
from utils.response_parser import parse_for_reward as parse_model_response



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
) -> float:
    """
    Compute multi-level reward for GRPO training.
    
    Reward tiers:
    - 1.0: Correct classification + ≥50% vulnerable lines correct
    - 0.6: Correct classification only
    - 0.3: Some correct lines but wrong classification
    - 0.0: Wrong classification + no correct lines
    
    Args:
        response: Model's generated response text
        is_vulnerable: Ground truth - whether the code is vulnerable
        ground_truth_lines: List of line numbers that are vulnerable
        
    Returns:
        Reward value between 0.0 and 1.0
    """
    # Parse response
    classification, predicted_lines = parse_model_response(response)
    
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
        return 0.0


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

_completion_log_lock = threading.Lock()
_completion_log_path = os.environ.get('GRPO_COMPLETION_LOG', None)
_step_counter = [0]  # Mutable to track steps across calls


def log_completion(step: int, prompt: str, completion: str, reward: float, is_vulnerable: bool, gt_lines: List[int]):
    """Log a single completion to file for debugging."""
    if not _completion_log_path:
        return
    
    try:
        with _completion_log_lock:
            with open(_completion_log_path, 'a', encoding='utf-8') as f:
                log_entry = {
                    "step": step,
                    "timestamp": __import__('datetime').datetime.now().isoformat(),
                    "is_vulnerable": is_vulnerable,
                    "ground_truth_lines": gt_lines,
                    "prompt_preview": prompt[:500] if prompt else "",
                    "completion_length": len(completion) if completion else 0,
                    "completion": completion,
                    "reward": reward,
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Warning: Failed to log completion: {e}")


def reward_function_for_grpo(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function compatible with trl GRPOTrainer.
    
    Expects kwargs to contain:
    - is_vulnerable: List[bool]
    - ground_truth_lines: List[List[int]]
    - prompts (optional): List[str] for logging
    
    Returns:
        List of reward floats
    """
    is_vulnerable_list = kwargs.get('is_vulnerable', [True] * len(completions))
    ground_truth_lines_list = kwargs.get('ground_truth_lines', [[]] * len(completions))
    prompts = kwargs.get('prompts', [''] * len(completions))
    
    rewards = batch_compute_rewards(completions, is_vulnerable_list, ground_truth_lines_list)
    
    # Log completions for debugging if GRPO_COMPLETION_LOG is set
    if _completion_log_path:
        _step_counter[0] += 1
        current_step = _step_counter[0]
        for i, (comp, reward, is_vuln, gt_lines, prompt) in enumerate(
            zip(completions, rewards, is_vulnerable_list, ground_truth_lines_list, prompts)
        ):
            log_completion(current_step, prompt, comp, reward, is_vuln, gt_lines)
    
    return rewards



if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Correct classification + correct lines
        ('{"classification": "VULNERABLE", "vulnerable_lines": [5, 7, 10], "reasoning_summary": "..."}',
         True, [5, 7], 1.0),
        
        # Correct classification + partial lines  
        ('{"classification": "VULNERABLE", "vulnerable_lines": [5], "reasoning_summary": "..."}',
         True, [5, 7, 10], 0.6),  # 33% < 50%
        
        # Correct classification, no lines needed
        ('{"classification": "NOT_VULNERABLE", "vulnerable_lines": [], "reasoning_summary": "..."}',
         False, [], 1.0),
        
        # Wrong classification, some lines correct
        ('{"classification": "NOT_VULNERABLE", "vulnerable_lines": [5, 7], "reasoning_summary": "..."}',
         True, [5, 7, 10], 0.3),
        
        # Wrong classification, no correct lines
        ('{"classification": "VULNERABLE", "vulnerable_lines": [1, 2], "reasoning_summary": "..."}',
         False, [], 0.0),
    ]
    
    print("Testing reward function:")
    for response, is_vuln, gt_lines, expected in test_cases:
        reward = compute_reward(response, is_vuln, gt_lines)
        status = "✓" if abs(reward - expected) < 0.01 else "✗"
        print(f"  {status} Expected {expected}, got {reward}")
