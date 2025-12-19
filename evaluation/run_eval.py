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

def process_single_request(client, args, sample, is_vuln, prompt, prompt_type):
    """Process a single evaluation request."""
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
            max_tokens=args.max_gen_length,
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
    
    args = parser.parse_args()
    
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
    requests = []
    for sample in samples:
        for is_vuln in [True, False]:
            code = sample['vuln_func'] if is_vuln else sample['patched_func']
            
            for prompt_type in args.prompts:
                if prompt_type == "training":
                    prompt = format_prompt_training(code, sample)
                else:  # std_cls
                    prompt = format_prompt_std_cls(code)
                
                requests.append({
                    "sample": sample,
                    "is_vuln": is_vuln,
                    "prompt": prompt,
                    "prompt_type": prompt_type
                })
    
    print(f"Total requests: {len(requests)} ({len(samples)} pairs x 2 versions x {len(args.prompts)} prompts)")
    
    # Clear output file
    open(args.output_file, 'w').close()
    
    # Setup writer
    writer = ThreadSafeWriter(args.output_file)
    
    # Process with thread pool
    processed = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for req in requests:
            future = executor.submit(
                process_single_request,
                client, args,
                req["sample"], req["is_vuln"], req["prompt"], req["prompt_type"]
            )
            futures[future] = req
        
        with tqdm(total=len(requests), desc="Running evaluation") as pbar:
            for future in as_completed(futures):
                result = future.result()
                writer.write(result)
                
                if result['response'].startswith("ERROR:"):
                    errors += 1
                processed += 1
                pbar.update(1)
    
    writer.flush()
    
    print(f"\nDone! Processed {processed} requests, {errors} errors")
    print(f"Responses saved to: {args.output_file}")


if __name__ == "__main__":
    main()
