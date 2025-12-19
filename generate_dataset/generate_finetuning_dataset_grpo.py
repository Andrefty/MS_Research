import json
import argparse
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai._types import NOT_GIVEN
from tqdm import tqdm
import threading

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    print("Warning: `transformers` library not installed.")

# --- Constants ---
MIN_REASONABLE_OUTPUT_SIZE = 4096
DEFAULT_CONCURRENCY = 8  # Number of concurrent requests

# --- Tokenizer Utilities ---
_tokenizer_cache = {}
_tokenizer_type_cache = {}

def get_tokenizer_for_model(model_identifier, local_model_path_for_tokenizer_if_any):
    if model_identifier in _tokenizer_cache:
        return _tokenizer_cache[model_identifier], _tokenizer_type_cache[model_identifier]

    tokenizer_obj = None
    is_hf_tokenizer = False

    if AutoTokenizer is not None:
        try:
            path_to_try = local_model_path_for_tokenizer_if_any if local_model_path_for_tokenizer_if_any else model_identifier
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

def get_total_model_capacity(model_name):
    if "32B" in model_name or "qwen3" in model_name.lower():
        return 32768
    if "8B" in model_name:
        return 32768
    print(f"Warning: Using default model capacity of 32768 for {model_name}")
    return 32768

def count_tokens(text_string, tokenizer_obj, is_hf_tokenizer):
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


# --- GRPO Prompt Formatting ---

def build_context_section(sample):
    """
    Build context section based on available CVE/CWE information.
    | Scenario | Context Provided |
    |----------|------------------|
    | Has CVE + cve_desc | CVE: {cve}, Description: {cve_desc} |
    | Has CVE only | CVE: {cve}, Type: {cwe} |
    | No CVE | Vulnerability type: {cwe} |
    | No CVE or CWE | (omit context section) |
    """
    cve = sample.get('cve')
    cve_desc = sample.get('cve_desc')
    cwe = sample.get('cwe')
    
    # Clean up cve_desc
    if cve_desc and cve_desc.strip().lower() in ["none", "null", "na", "n/a", ""]:
        cve_desc = None
    
    # Format CWE list
    cwe_str = None
    if cwe:
        if isinstance(cwe, list):
            cwe_str = ", ".join(cwe)
        else:
            cwe_str = str(cwe)
    
    context_parts = []
    
    if cve and cve_desc:
        context_parts.append(f"CVE: {cve}")
        context_parts.append(f"Description: {cve_desc}")
    elif cve and cwe_str:
        context_parts.append(f"CVE: {cve}")
        context_parts.append(f"Vulnerability type: {cwe_str}")
    elif cwe_str:
        context_parts.append(f"Vulnerability type: {cwe_str}")
    
    if context_parts:
        return "Context:\n- " + "\n- ".join(context_parts)
    return None


def get_ground_truth_lines(sample):
    """Extract line numbers from deleted_lines for ground truth."""
    deleted_lines = sample.get('deleted_lines', [])
    if deleted_lines:
        return sorted(set(item['line_no'] for item in deleted_lines if 'line_no' in item))
    return []


def format_changed_lines_hint(ground_truth_lines):
    """Format line numbers as hint string."""
    if ground_truth_lines:
        return "Lines: " + ", ".join(map(str, ground_truth_lines))
    return None


def format_prompt_for_model_grpo(code_snippet, sample, is_vulnerable, 
                                  ground_truth_lines=None, line_number_threshold=0,
                                  include_hints=True):
    """
    Format prompt for GRPO training with JSON output format.
    
    Args:
        code_snippet: The code to analyze
        sample: Sample dict with metadata (CVE, CWE, etc.)
        is_vulnerable: True if vulnerable, False if patched, None if unknown (eval mode)
        ground_truth_lines: List of line numbers that were changed
        line_number_threshold: Unused, kept for compatibility
        include_hints: If False, skip CVE/CWE context and vulnerability hints (for evaluation)
    
    For training (include_hints=True):
        - Vulnerable samples: includes CVE context and changed lines hint
        - Patched samples: includes hint that code is patched
    
    For evaluation (include_hints=False):
        - No CVE/CWE context
        - No vulnerability/patched hints
    """
    # Always add line numbers for GRPO (need for line matching)
    lines = code_snippet.splitlines()
    numbered_code = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    
    prompt_parts = [
        "You are a security expert analyzing code for vulnerabilities.",
        "",
        "Analyze the following code snippet:",
        "",
        "```",
        numbered_code,
        "```",
        ""
    ]
    
    if include_hints:
        # Add context section (CVE/CWE) - only when hints enabled
        context = build_context_section(sample)
        if context:
            prompt_parts.append(context)
            prompt_parts.append("")
        
        # Add hint about vulnerability status
        if is_vulnerable:
            prompt_parts.append("Hint: This code contains a security vulnerability.")
            # Add changed lines hint for vulnerable samples
            lines_hint = format_changed_lines_hint(ground_truth_lines)
            if lines_hint:
                prompt_parts.append(f"The changed lines that might be related to the vulnerability are: {lines_hint}.")
        elif is_vulnerable is False:  # Explicitly False, not None
            prompt_parts.append("Hint: This code is the patched (non-vulnerable) version.")
        
        prompt_parts.append("")
    
    # Add JSON output instruction
    prompt_parts.append(
        'Task: After your reasoning, provide your analysis in JSON format: '
        '{"classification": "VULNERABLE" or "NOT_VULNERABLE", "vulnerable_lines": [...], "reasoning_summary": "..."}'
    )
    
    return "\n".join(prompt_parts)


def parse_model_response(response_text):
    """
    Parse model response to extract classification and vulnerable lines.
    Returns: (classification, vulnerable_lines) or (None, None) if parsing fails.
    """
    if not response_text:
        return None, None
    
    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*"classification"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            classification = data.get('classification', '').upper()
            vulnerable_lines = data.get('vulnerable_lines', [])
            
            # Normalize classification
            if 'VULNERABLE' in classification and 'NOT' not in classification:
                classification = 'VULNERABLE'
            elif 'NOT' in classification or classification == 'NOT_VULNERABLE':
                classification = 'NOT_VULNERABLE'
            else:
                classification = None
            
            # Ensure vulnerable_lines is a list of ints
            if isinstance(vulnerable_lines, list):
                vulnerable_lines = [int(x) for x in vulnerable_lines if isinstance(x, (int, float))]
            else:
                vulnerable_lines = []
            
            return classification, vulnerable_lines
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: try to extract from text patterns
    response_upper = response_text.upper()
    if 'NOT_VULNERABLE' in response_upper or 'NOT VULNERABLE' in response_upper or '(2) NO' in response_upper:
        return 'NOT_VULNERABLE', []
    elif 'VULNERABLE' in response_upper or '(1) YES' in response_upper:
        return 'VULNERABLE', []
    
    return None, None


# --- Resume and Live Writing Utilities ---

def get_already_processed_ids(output_file_path):
    """Check existing output file to find already processed IDs."""
    processed_ids = set()
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Create unique key for each sample (commit_id + is_vulnerable)
                        key = f"{data.get('commit_id')}_{data.get('is_vulnerable')}"
                        processed_ids.add(key)
            print(f"Found {len(processed_ids)} already processed samples in existing output file.")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    return processed_ids


# --- Thread-safe file writer ---

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


# --- Request processing ---

def process_single_request(client, args, prompt, sample, is_vulnerable, 
                           ground_truth_lines, api_max_tokens, tokenizer_obj, is_hf_tokenizer):
    """Process a single request to the LLM."""
    try:
        extra_body = {}
        if args.model_id_for_sglang.startswith("Qwen"):
            extra_body["enable_thinking"] = args.enable_thinking
            if args.top_k is not None:
                extra_body["top_k"] = args.top_k
            if args.min_p is not None:
                extra_body["min_p"] = args.min_p

        response = client.chat.completions.create(
            model=args.model_id_for_sglang,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=api_max_tokens,
            seed=args.seed,
            logprobs=args.logprobs,
            extra_body=extra_body if extra_body else NOT_GIVEN
        )
        generated_text = response.choices[0].message.content
        parsed_class, parsed_lines = parse_model_response(generated_text)
        
        return {
            "commit_id": sample['commit_id'],
            "source": sample.get('source'),
            "is_vulnerable": is_vulnerable,
            "code": sample['vuln_func'] if is_vulnerable else sample['patched_func'],
            "prompt": prompt,
            "generated_response": generated_text,
            "ground_truth_lines": ground_truth_lines if is_vulnerable else [],
            "parsed_classification": parsed_class,
            "parsed_vulnerable_lines": parsed_lines
        }
    except Exception as e:
        return {
            "commit_id": sample['commit_id'],
            "source": sample.get('source'),
            "is_vulnerable": is_vulnerable,
            "code": sample['vuln_func'] if is_vulnerable else sample['patched_func'],
            "prompt": prompt,
            "generated_response": f"ERROR_SGLANG_CALL: {str(e)}",
            "ground_truth_lines": ground_truth_lines if is_vulnerable else [],
            "parsed_classification": None,
            "parsed_vulnerable_lines": None
        }


def prepare_request(sample, is_vulnerable, tokenizer_obj, is_hf_tokenizer, 
                    total_model_capacity, max_gen_length):
    """Prepare a request dict without making the API call."""
    if is_vulnerable:
        code = sample['vuln_func']
        ground_truth_lines = get_ground_truth_lines(sample)
    else:
        code = sample['patched_func']
        ground_truth_lines = []
    
    prompt = format_prompt_for_model_grpo(
        code, sample, is_vulnerable=is_vulnerable,
        ground_truth_lines=ground_truth_lines
    )
    
    prompt_tokens = count_tokens(prompt, tokenizer_obj, is_hf_tokenizer)
    max_new_tokens = total_model_capacity - prompt_tokens
    api_max_tokens = min(max_gen_length, max_new_tokens)
    
    if api_max_tokens < MIN_REASONABLE_OUTPUT_SIZE:
        return None  # Skip due to token limit
    
    return {
        'sample': sample,
        'is_vulnerable': is_vulnerable,
        'prompt': prompt,
        'ground_truth_lines': ground_truth_lines,
        'api_max_tokens': api_max_tokens
    }


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate GRPO finetuning dataset from merged vulnerability data.")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to merged .jsonl file (Research_merged_dataset/merged_train.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to save the generated .jsonl dataset.")
    parser.add_argument("--sglang_host", type=str, default="127.0.0.1")
    parser.add_argument("--sglang_port", type=int, default=30000)
    
    parser.add_argument('--model_id_for_sglang', type=str, default="Qwen/Qwen3-32B")
    parser.add_argument('--model_id_for_tokenizer', type=str, default="Qwen/Qwen3-32B")
    parser.add_argument('--local_model_path_for_tokenizer', type=str, default=None)

    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--max_gen_length', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--logprobs', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--enable_thinking', type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # Concurrency settings
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Number of concurrent requests (default: {DEFAULT_CONCURRENCY})")

    args = parser.parse_args()

    # Setup SGLang client
    sglang_api_url = f"http://{args.sglang_host}:{args.sglang_port}/v1"
    client = OpenAI(api_key="EMPTY", base_url=sglang_api_url)

    tokenizer_obj, is_hf_tokenizer = get_tokenizer_for_model(
        args.model_id_for_tokenizer, args.local_model_path_for_tokenizer
    )
    total_model_capacity = get_total_model_capacity(args.model_id_for_sglang)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Check for already processed samples
    already_processed_ids = get_already_processed_ids(args.output_file)

    # Load merged dataset
    print("Loading merged dataset...")
    all_samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))
    
    print(f"Loaded {len(all_samples)} pairs from merged dataset")
    
    # Sort samples: cve_desc first, then others
    def has_cve_desc(sample):
        desc = sample.get('cve_desc')
        if desc and str(desc).strip().lower() not in ["none", "null", "na", "n/a", ""]:
            return 0 # Lower value = higher priority
        return 1
    
    all_samples.sort(key=has_cve_desc)
    
    samples_with_desc = sum(1 for s in all_samples if has_cve_desc(s) == 0)
    print(f"Samples with cve_desc: {samples_with_desc}")
    print(f"Samples without cve_desc: {len(all_samples) - samples_with_desc}")
    print(f"Using concurrency: {args.concurrency}")

    # Prepare all requests
    print("Preparing requests...")
    pending_requests = []
    skipped_due_to_token_limit = 0
    
    for sample in all_samples:
        commit_id = sample['commit_id']
        
        # Check vulnerable version
        vuln_key = f"{commit_id}_True"
        if vuln_key not in already_processed_ids:
            req = prepare_request(sample, True, tokenizer_obj, is_hf_tokenizer,
                                  total_model_capacity, args.max_gen_length)
            if req:
                pending_requests.append(req)
            else:
                skipped_due_to_token_limit += 1
        
        # Check patched version
        patched_key = f"{commit_id}_False"
        if patched_key not in already_processed_ids:
            req = prepare_request(sample, False, tokenizer_obj, is_hf_tokenizer,
                                  total_model_capacity, args.max_gen_length)
            if req:
                pending_requests.append(req)
            else:
                skipped_due_to_token_limit += 1
    
    print(f"Prepared {len(pending_requests)} requests to process")
    print(f"Skipped due to token limit: {skipped_due_to_token_limit}")

    if not pending_requests:
        print("No new requests to process. Exiting.")
        return

    # Setup thread-safe writer
    writer = ThreadSafeWriter(args.output_file)
    
    # Process requests concurrently
    processed_count = 0
    error_count = 0
    
    def process_request(req):
        nonlocal processed_count, error_count
        result = process_single_request(
            client, args, req['prompt'], req['sample'], req['is_vulnerable'],
            req['ground_truth_lines'], req['api_max_tokens'], tokenizer_obj, is_hf_tokenizer
        )
        writer.write(result)
        if result['parsed_classification'] is None:
            error_count += 1
        return result
    
    print(f"\nProcessing {len(pending_requests)} requests with concurrency={args.concurrency}...")
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = list(tqdm(
            executor.map(process_request, pending_requests),
            total=len(pending_requests),
            desc="Processing samples"
        ))
        processed_count = len(futures)
    
    # Final flush
    writer.flush()
    
    print(f"\nFinished processing.")
    print(f"Total pairs in input: {len(all_samples)}")
    print(f"Already processed samples (skipped): {len(already_processed_ids)}")
    print(f"Samples processed: {processed_count}")
    print(f"Samples with errors: {error_count}")
    print(f"Samples skipped due to token limit: {skipped_due_to_token_limit}")
    print(f"Total written to output: {writer.total_written}")
    print(f"Generated dataset saved to {args.output_file}")


if __name__ == "__main__":
    main()
