#!/usr/bin/env python3
"""
response_parser.py - Centralized response parsing for vulnerability classification.

This module provides a single, robust implementation for parsing model responses
across all scripts (training, evaluation, dataset generation).

Expected response format (after thinking):
{
    "classification": "VULNERABLE" or "NOT_VULNERABLE",
    "vulnerable_lines": [line_numbers...],
    "reasoning_summary": "..."
}
"""

import json
import re
from typing import Tuple, List, Optional, NamedTuple


class ParseResult(NamedTuple):
    """Result of parsing a model response."""
    classification: Optional[str]  # "VULNERABLE", "NOT_VULNERABLE", or None
    vulnerable_lines: List[int]
    status: str  # "VALID", "NO_JSON", "INVALID_JSON", "NO_CLASSIFICATION", etc.
    raw_json: Optional[dict]  # The parsed JSON dict if successful


def extract_json_from_response(response_text: str) -> Optional[dict]:
    """
    Extract JSON object from model response using Python's built-in JSON decoder.
    
    Uses json.JSONDecoder().raw_decode() which is specifically designed to parse
    JSON from a string that has extra content before/after the JSON.
    
    This handles:
    - JSON with curly braces inside string values (e.g., "{N, x}")
    - JSON appearing after </think> block
    - Multiple JSON objects (returns the one with "classification")
    
    Returns the parsed dict or None if no valid JSON found.
    """
    if not response_text:
        return None
    
    # Remove thinking block if present
    text = response_text
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    
    # Use Python's built-in JSON decoder with raw_decode
    # This properly handles all JSON edge cases including nested braces in strings
    decoder = json.JSONDecoder()
    
    # Find all JSON objects and look for one with "classification"
    json_objects = []
    pos = 0
    while pos < len(text):
        # Find next potential JSON start
        start = text.find('{', pos)
        if start == -1:
            break
        
        try:
            # raw_decode parses JSON from position and returns (obj, end_position)
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                json_objects.append(obj)
                if 'classification' in obj:
                    return obj  # Found our target
            pos = end
        except json.JSONDecodeError:
            # Not valid JSON starting here, move past this '{'
            pos = start + 1
    
    # Return first dict if any were found (might not have "classification")
    if json_objects:
        return json_objects[0]
    
    # Fallback: Try regex extraction for severely malformed responses
    class_match = re.search(
        r'"classification"\s*:\s*"([^"]+)"',
        response_text,
        re.IGNORECASE
    )
    if class_match:
        classification = class_match.group(1).upper()
        
        # Try to find vulnerable_lines
        lines_match = re.search(
            r'"vulnerable_lines"\s*:\s*\[([\d,\s]*)\]',
            response_text
        )
        lines = []
        if lines_match:
            lines_str = lines_match.group(1)
            lines = [int(x.strip()) for x in lines_str.split(',') if x.strip().isdigit()]
        
        return {
            'classification': classification,
            'vulnerable_lines': lines,
            'reasoning_summary': '[extracted via regex fallback]'
        }
    
    return None


def normalize_classification(classification: str) -> Optional[str]:
    """
    Normalize classification string to standard values.
    
    Returns:
        "VULNERABLE", "NOT_VULNERABLE", or None if invalid
    """
    if not classification:
        return None
    
    upper = classification.upper().strip()
    
    # Check for NOT_VULNERABLE variants first (more specific)
    if 'NOT_VULNERABLE' in upper or 'NOT VULNERABLE' in upper:
        return 'NOT_VULNERABLE'
    elif 'VULNERABLE' in upper:
        return 'VULNERABLE'
    
    return None


def parse_model_response(
    response_text: str,
    include_raw_json: bool = False
) -> ParseResult:
    """
    Parse model response to extract classification and vulnerable lines.
    
    This is the main entry point for parsing responses.
    
    Args:
        response_text: The full model response (may include <think> block)
        include_raw_json: If True, include the parsed JSON dict in result
    
    Returns:
        ParseResult with classification, lines, status, and optionally raw_json
    """
    if not response_text:
        return ParseResult(None, [], "EMPTY", None)
    
    if response_text.startswith("ERROR:"):
        return ParseResult(None, [], "API_ERROR", None)
    
    if response_text.startswith("SKIPPED:"):
        return ParseResult(None, [], "SKIPPED", None)
    
    # Try to extract JSON
    json_data = extract_json_from_response(response_text)
    
    if json_data is None:
        # No JSON found - check for keywords as last resort
        # BUT only if model finished thinking (has </think> tag)
        # Otherwise the model was stuck/truncated mid-thinking
        has_think_end = '</think>' in response_text.lower()
        
        if has_think_end:
            response_upper = response_text.upper()
            if 'NOT_VULNERABLE' in response_upper or 'NOT VULNERABLE' in response_upper:
                return ParseResult('NOT_VULNERABLE', [], "KEYWORD_ONLY", None)
            elif 'VULNERABLE' in response_upper:
                return ParseResult('VULNERABLE', [], "KEYWORD_ONLY", None)
            return ParseResult(None, [], "NO_JSON", None)
        else:
            # Model didn't finish thinking - incomplete response
            return ParseResult(None, [], "INCOMPLETE", None)
    
    # JSON found - extract fields
    raw_classification = json_data.get('classification', '')
    classification = normalize_classification(raw_classification)
    
    if classification is None:
        return ParseResult(
            None, [], "INVALID_CLASSIFICATION",
            json_data if include_raw_json else None
        )
    
    # Extract vulnerable lines
    vulnerable_lines = []
    raw_lines = json_data.get('vulnerable_lines', [])
    
    if isinstance(raw_lines, list):
        for item in raw_lines:
            if isinstance(item, (int, float)):
                vulnerable_lines.append(int(item))
            elif isinstance(item, str) and item.isdigit():
                vulnerable_lines.append(int(item))
    
    return ParseResult(
        classification,
        vulnerable_lines,
        "VALID",
        json_data if include_raw_json else None
    )


def parse_for_reward(response_text: str) -> Tuple[Optional[str], List[int]]:
    """
    Simple interface for reward function (backward compatible).
    
    Returns:
        (classification, vulnerable_lines) tuple
    """
    result = parse_model_response(response_text)
    return result.classification, result.vulnerable_lines


def parse_for_metrics(response_text: str) -> Tuple[Optional[int], Optional[List[int]]]:
    """
    Interface for compute_metrics.py (backward compatible).
    
    Returns:
        (prediction, vulnerable_lines) where prediction is 0, 1, or None
    """
    result = parse_model_response(response_text)
    
    if result.classification is None:
        return None, None
    
    prediction = 1 if result.classification == "VULNERABLE" else 0
    return prediction, result.vulnerable_lines


# Alias for backward compatibility
def get_classification_and_lines(response_text: str) -> Tuple[Optional[str], List[int]]:
    """Backward compatible alias for parse_for_reward."""
    return parse_for_reward(response_text)


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Standard valid JSON
        (
            '<think>Analysis...</think>\n{"classification": "VULNERABLE", "vulnerable_lines": [57], "reasoning_summary": "Buffer overflow"}',
            "VULNERABLE", [57], "VALID"
        ),
        # JSON with curly braces in string (the bug case)
        (
            '<think>...</think>\n{"classification": "VULNERABLE", "vulnerable_lines": [57], "reasoning_summary": "tensor shape {N, num_updates / N}"}',
            "VULNERABLE", [57], "VALID"
        ),
        # NOT_VULNERABLE
        (
            '{"classification": "NOT_VULNERABLE", "vulnerable_lines": [], "reasoning_summary": "Safe code"}',
            "NOT_VULNERABLE", [], "VALID"
        ),
        # No JSON but has keyword
        (
            '<think>The code is VULNERABLE because...</think>',
            "VULNERABLE", [], "KEYWORD_ONLY"
        ),
        # Empty
        ("", None, [], "EMPTY"),
        # Error
        ("ERROR: timeout", None, [], "API_ERROR"),
    ]
    
    print("Testing response_parser.py:")
    print("=" * 60)
    
    all_passed = True
    for i, (text, exp_class, exp_lines, exp_status) in enumerate(test_cases):
        result = parse_model_response(text)
        
        passed = (
            result.classification == exp_class and
            result.vulnerable_lines == exp_lines and
            result.status == exp_status
        )
        
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        
        print(f"{status} Test {i+1}:")
        print(f"  Input: {text[:80]}...")
        print(f"  Expected: class={exp_class}, lines={exp_lines}, status={exp_status}")
        print(f"  Got:      class={result.classification}, lines={result.vulnerable_lines}, status={result.status}")
        print()
    
    print("=" * 60)
    print(f"All tests passed: {all_passed}")
