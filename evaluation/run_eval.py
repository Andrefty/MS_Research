#!/usr/bin/env python3
"""
run_eval.py - Run evaluation inference on test datasets with parallel requests.

Runs both training-format and std_cls prompts on all samples.
Reuses parallel SGLang pattern from dataset generation.

Usage:
    python run_eval.py \
        --eval_dataset eval_dataset.jsonl \
        --output_file eval_responses.jsonl \
        --sglang_host 127.0.0.1 \
        --sglang_port 30000 \
        --concurrency 8
"""

import json
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import threading

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generate_dataset'))

from generate_finetuning_dataset_grpo import format_prompt_for_model_grpo

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    print("Warning: transformers not installed")


# --- Constants ---
MIN_REASONABLE_OUTPUT_SIZE = 4096  # Ensure model always has at least this much space to generate
DEFAULT_MODEL_CAPACITY = 32768


# --- Tokenizer Utilities ---
_tokenizer_cache = {}
_tokenizer_type_cache = {}


def get_tokenizer_for_model(model_identifier, local_model_path=None):
    """Load tokenizer for token counting."""
    if model_identifier in _tokenizer_cache:
        return _tokenizer_cache[model_identifier], _tokenizer_type_cache[model_identifier]

    tokenizer_obj = None
    is_hf_tokenizer = False

    if AutoTokenizer is not None:
        try:
            path_to_try = local_model_path if local_model_path else model_identifier
            tokenizer_obj = AutoTokenizer.from_pretrained(path_to_try, trust_remote_code=True)
            is_hf_tokenizer = True
            print(f"Successfully loaded Hugging Face tokenizer for: {path_to_try}")
        except Exception as e:
            print(f"Warning: Failed to load Hugging Face tokenizer for '{path_to_try}': {e}")
            tokenizer_obj = None
            is_hf_tokenizer = False
    
    if tokenizer_obj is None:
        try:
            import tiktoken
            tokenizer_obj = tiktoken.get_encoding("cl100k_base")
            is_hf_tokenizer = False
            print(f"Using tiktoken cl100k_base as fallback for: {model_identifier}")
        except ImportError:
            print("Warning: `tiktoken` library not installed.")
        except Exception as e:
            print(f"Warning: Failed to load tiktoken: {e}")

    _tokenizer_cache[model_identifier] = tokenizer_obj
    _tokenizer_type_cache[model_identifier] = is_hf_tokenizer
    return tokenizer_obj, is_hf_tokenizer


def count_tokens(text_string, tokenizer_obj, is_hf_tokenizer):
    """Count tokens in a string."""
    if not text_string:
        return 0
    
    if tokenizer_obj and is_hf_tokenizer:
        return len(tokenizer_obj.encode(text_string, add_special_tokens=False))
    elif tokenizer_obj:
        try:
            return len(tokenizer_obj.encode(text_string))
        except Exception:
            return len(text_string.split())
    else:
        return len(text_string.split())


def get_total_model_capacity(model_name):
    """Get total context length for a model."""
    if "32B" in model_name or "qwen3" in model_name.lower() or "4B" in model_name:
        return 32768
    if "8B" in model_name:
        return 32768
    print(f"Warning: Using default model capacity of {DEFAULT_MODEL_CAPACITY} for {model_name}")
    return DEFAULT_MODEL_CAPACITY


# --- Prompt Formats ---

def format_prompt_training(code_snippet, sample):
    """Training-style prompt without hints (for evaluation)."""
    return format_prompt_for_model_grpo(
        code_snippet=code_snippet,
        sample=sample,
        is_vulnerable=None,  # Unknown for evaluation
        ground_truth_lines=None,
        include_hints=False
    )


def format_prompt_std_cls(code_snippet):
    """Standard classification prompt from Semester 2."""
    return f"""Please analyze the following code:
```
{code_snippet}
```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information."""


# --- Thread-safe Writer ---

class ThreadSafeWriter:
    def __init__(self, output_file_path):
        self.output_file_path = output_file_path
        self.lock = threading.Lock()
        self.buffer = []
        self.buffer_size = 10
        self.total_written = 0
    
    def write(self, result):
        with self.lock:
            self.buffer.append(result)
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        if self.buffer:
            with open(self.output_file_path, 'a', encoding='utf-8') as f:
                for result in self.buffer:
                    f.write(json.dumps(result) + '\n')
            self.total_written += len(self.buffer)
            self.buffer.clear()
    
    def flush(self):
        with self.lock:
            self._flush_buffer()


# --- Request Processing ---

def process_single_request(client, args, sample, is_vuln, prompt, prompt_type, max_tokens):
    """Process a single evaluation request.
    
    Args:
        max_tokens: Dynamic max tokens calculated based on prompt length and model capacity.
    """
    code = sample['vuln_func'] if is_vuln else sample['patched_func']
    
    try:
        extra_body = {"enable_thinking": True}
        if args.top_k is not None:
            extra_body["top_k"] = args.top_k
        if args.min_p is not None:
            extra_body["min_p"] = args.min_p
        
        response = client.chat.completions.create(
            model=args.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,  # Dynamic, based on prompt length
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body=extra_body
        )
        
        response_text = response.choices[0].message.content if response.choices else ""
        
        return {
            "commit_id": sample['commit_id'],
            "pair_id": sample.get('pair_id', sample['commit_id']),
            "source": sample['source'],
            "split": sample['split'],
            "is_vulnerable": is_vuln,
            "target": 1 if is_vuln else 0,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "response": response_text,
            "ground_truth_lines": sample.get('deleted_lines', []) if is_vuln else []
        }
    except Exception as e:
        return {
            "commit_id": sample['commit_id'],
            "pair_id": sample.get('pair_id', sample['commit_id']),
            "source": sample['source'],
            "split": sample['split'],
            "is_vulnerable": is_vuln,
            "target": 1 if is_vuln else 0,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "response": f"ERROR: {str(e)}",
            "ground_truth_lines": []
        }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation inference")
    
    # Input/Output
    parser.add_argument("--eval_dataset", type=str, required=True,
                        help="Path to merged eval dataset (from prepare_eval_dataset.py)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save responses")
    
    # SGLang
    parser.add_argument("--sglang_host", type=str, default="127.0.0.1")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--model_id", type=str, default="default",
                        help="Model ID for SGLang (usually 'default' for single model)")
    
    # Generation params (Qwen3 thinking mode defaults)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0)
    parser.add_argument("--max_gen_length", type=int, default=32768)
    
    # Concurrency
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of parallel requests")
    
    # Prompt selection
    parser.add_argument("--prompts", nargs='+', default=["training", "std_cls"],
                        choices=["training", "std_cls"],
                        help="Which prompt types to run")
    
    # Tokenizer for token counting (to ensure prompts leave room for response)
    parser.add_argument("--tokenizer_model", type=str, default="Qwen/Qwen3-4B",
                        help="Model name or path for tokenizer (for token counting)")
    
    args = parser.parse_args()
    
    # Setup tokenizer for token counting
    tokenizer_obj, is_hf_tokenizer = get_tokenizer_for_model(args.tokenizer_model)
    total_model_capacity = get_total_model_capacity(args.tokenizer_model)
    print(f"Model capacity: {total_model_capacity} tokens, min output size: {MIN_REASONABLE_OUTPUT_SIZE}")
    
    # Setup client
    base_url = f"http://{args.sglang_host}:{args.sglang_port}/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    
    # Load eval dataset
    print(f"Loading eval dataset: {args.eval_dataset}")
    samples = []
    with open(args.eval_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} pairs")
    
    # Prepare all requests (each sample -> 2 versions (vuln/patched) x N prompts)
    # With token length validation to ensure classification instructions aren't truncated
    requests = []
    skipped_too_long = 0
    
    for sample in samples:
        for is_vuln in [True, False]:
            code = sample['vuln_func'] if is_vuln else sample['patched_func']
            
            for prompt_type in args.prompts:
                if prompt_type == "training":
                    prompt = format_prompt_training(code, sample)
                else:  # std_cls
                    prompt = format_prompt_std_cls(code)
                
                # Check token count to ensure enough space for response
                prompt_tokens = count_tokens(prompt, tokenizer_obj, is_hf_tokenizer)
                available_for_response = total_model_capacity - prompt_tokens
                
                # Calculate dynamic max_tokens: min of what's available and requested max
                dynamic_max_tokens = min(args.max_gen_length, available_for_response)
                
                if available_for_response < MIN_REASONABLE_OUTPUT_SIZE:
                    skipped_too_long += 1
                    # Still add the request but mark it as skipped (for tracking)
                    requests.append({
                        "sample": sample,
                        "is_vuln": is_vuln,
                        "prompt": prompt,
                        "prompt_type": prompt_type,
                        "max_tokens": 0,  # Won't be used since it's skipped
                        "skip_reason": f"prompt_too_long:{prompt_tokens}_tokens"
                    })
                else:
                    requests.append({
                        "sample": sample,
                        "is_vuln": is_vuln,
                        "prompt": prompt,
                        "prompt_type": prompt_type,
                        "max_tokens": dynamic_max_tokens,
                        "skip_reason": None
                    })
    
    print(f"Total requests: {len(requests)} ({len(samples)} pairs x 2 versions x {len(args.prompts)} prompts)")
    if skipped_too_long > 0:
        print(f"Warning: {skipped_too_long} requests have prompts too long (< {MIN_REASONABLE_OUTPUT_SIZE} tokens available for response)")
    
    # Clear output file
    open(args.output_file, 'w').close()
    
    # Setup writer
    writer = ThreadSafeWriter(args.output_file)
    
    # Separate requests into valid and skipped
    valid_requests = [r for r in requests if r["skip_reason"] is None]
    skipped_requests = [r for r in requests if r["skip_reason"] is not None]
    
    # Write skipped requests immediately
    for req in skipped_requests:
        sample = req["sample"]
        is_vuln = req["is_vuln"]
        skipped_result = {
            "commit_id": sample['commit_id'],
            "pair_id": sample.get('pair_id', sample['commit_id']),
            "source": sample['source'],
            "split": sample['split'],
            "is_vulnerable": is_vuln,
            "target": 1 if is_vuln else 0,
            "prompt_type": req["prompt_type"],
            "prompt": req["prompt"],
            "response": f"SKIPPED: {req['skip_reason']}",
            "ground_truth_lines": sample.get('deleted_lines', []) if is_vuln else []
        }
        writer.write(skipped_result)
    
    print(f"Skipped {len(skipped_requests)} requests (prompt too long)")
    print(f"Processing {len(valid_requests)} valid requests...")
    
    # Process valid requests with thread pool
    processed = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for req in valid_requests:
            future = executor.submit(
                process_single_request,
                client, args,
                req["sample"], req["is_vuln"], req["prompt"], req["prompt_type"],
                req["max_tokens"]  # Pass dynamic max_tokens
            )
            futures[future] = req
        
        with tqdm(total=len(valid_requests), desc="Running evaluation") as pbar:
            for future in as_completed(futures):
                result = future.result()
                writer.write(result)
                
                if result['response'].startswith("ERROR:"):
                    errors += 1
                processed += 1
                pbar.update(1)
    
    writer.flush()
    
    print(f"\nDone! Processed {processed} requests, {errors} errors, {len(skipped_requests)} skipped")
    print(f"Responses saved to: {args.output_file}")


if __name__ == "__main__":
    main()
