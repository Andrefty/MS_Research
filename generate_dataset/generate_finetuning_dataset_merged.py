import json
import argparse
import os
import difflib
from openai import OpenAI
from openai._types import NOT_GIVEN # Import NOT_GIVEN
from tqdm import tqdm
from collections import defaultdict

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizerBase = None
    print("Warning: `transformers` library not installed. Token counting may rely on basic methods or tiktoken if Qwen HF tokenizer loading fails.")

# --- Constants ---
MIN_REASONABLE_OUTPUT_SIZE = 4096

# --- Tokenizer and Model Capacity Utilities ---
_tokenizer_cache = {}
_tokenizer_type_cache = {}  # To store if it's HF or not

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
            print(f"Warning: Failed to load Hugging Face tokenizer for '{path_to_try}': {e}. Falling back if possible.")
            tokenizer_obj = None
            is_hf_tokenizer = False
    
    if tokenizer_obj is None:
        if "qwen" in model_identifier.lower():
            try:
                import tiktoken # Import tiktoken here
                tokenizer_obj = tiktoken.get_encoding("cl100k_base")
                is_hf_tokenizer = False
                print(f"Using tiktoken cl100k_base as fallback for Qwen model: {model_identifier}")
            except ImportError:
                print("Warning: `tiktoken` library not installed. Cannot use tiktoken fallback for Qwen.")
            except Exception as e:
                print(f"Warning: Failed to load tiktoken cl100k_base for Qwen: {e}")
        
        if tokenizer_obj is None:
            print(f"Warning: Could not load any tokenizer for {model_identifier}. Token counting will be very basic (split by space).")

    _tokenizer_cache[model_identifier] = tokenizer_obj
    _tokenizer_type_cache[model_identifier] = is_hf_tokenizer
    return tokenizer_obj, is_hf_tokenizer

def get_total_model_capacity(model_name):
    if "Qwen3-32B" in model_name or "qwen3-32b" in model_name.lower():
        return 32768
    if "32B" in model_name or "qwen3" in model_name.lower(): # General Qwen3 32k
        return 32768
    if "8B" in model_name: # Qwen/Qwen3-8B
        return 32768
    # Fallback, can be refined
    print(f"Warning: Using default model capacity of 32768 for {model_name}. Please verify.")
    return 32768

def count_tokens(text_string, tokenizer_obj, is_hf_tokenizer):
    if not text_string:
        return 0
    
    if tokenizer_obj and is_hf_tokenizer:
        return len(tokenizer_obj.encode(text_string, add_special_tokens=False))
    elif tokenizer_obj: # Tiktoken or other non-HF with encode
        try:
            return len(tokenizer_obj.encode(text_string))
        except Exception as e:
            print(f"Warning: Tokenizer ({type(tokenizer_obj)}) failed to encode: {e}. Falling back to space split.")
            return len(text_string.split())
    else: # No tokenizer
        return len(text_string.split())

# --- Diff and Prompt Formatting ---
def get_changed_lines_info_str(non_vuln_code_str, vuln_code_str):
    non_vuln_lines = non_vuln_code_str.splitlines()
    vuln_lines = vuln_code_str.splitlines()
    s = difflib.SequenceMatcher(None, non_vuln_lines, vuln_lines)
    
    changed_vuln_line_numbers = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != 'equal':
            # Add line numbers from the vulnerable version (j1 to j2)
            # difflib line numbers are 0-indexed for list, convert to 1-indexed for display
            for k in range(j1, j2):
                changed_vuln_line_numbers.append(k + 1)

    if changed_vuln_line_numbers:
        return "Lines: " + ", ".join(map(str, sorted(list(set(changed_vuln_line_numbers)))))
    return None

def format_prompt_for_model(code_snippet_str, cve_desc_str, is_vulnerable_ground_truth,
                            vulnerable_lines_info_str=None, line_number_threshold=20):
    lines = code_snippet_str.splitlines()
    use_line_numbers = len(lines) > line_number_threshold
    
    numbered_code = ""
    if use_line_numbers:
        for i, line_content in enumerate(lines):
            numbered_code += f"{i+1}: {line_content}\\n" # Use \\n for explicit newline in f-string
        numbered_code = numbered_code.rstrip('\\n')
    else:
        numbered_code = code_snippet_str

    prompt = f"Please analyze the following code snippet:\\n\\n```\\n{numbered_code}\\n```\\n\\n"
    
    cleaned_cve_desc = cve_desc_str.strip() if cve_desc_str else ""
    if cleaned_cve_desc and cleaned_cve_desc.lower() not in ["none", "null", "na", "n/a", ""]:
        prompt += f"The associated CVE description is: {cleaned_cve_desc}\\n\\n"

    hint_status = "vulnerable" if is_vulnerable_ground_truth else "not vulnerable"
    prompt += f"Hint: According to the ground truth, this code snippet is considered {hint_status}.\\n"

    if is_vulnerable_ground_truth and vulnerable_lines_info_str:
        prompt += f"The changed lines that might be related to the vulnerability are: {vulnerable_lines_info_str}.\\n"
    elif not is_vulnerable_ground_truth:
        prompt += "This code snippet is the non-vulnerable version.\\n"
        
    ## First prompt
    # task_description = f"Please provide a detailed explanation of why the code snippet is {hint_status}"
    # if is_vulnerable_ground_truth and vulnerable_lines_info_str:
    #      task_description += ", focusing on how the changes at the mentioned lines might contribute to its state"
    # # task_description += ". Conclude with a clear and concise summary."
    # task_description += ". After your reasoning, provide a final answer explicitly stating either '(1) YES: A security vulnerability detected.' or '(2) NO: No security vulnerability.'."
    ## Second prompt
    # task_description = f"Please provide your detailed reasoning on why the code snippet is {hint_status}"
    # if is_vulnerable_ground_truth and vulnerable_lines_info_str:
    #      task_description += ", particularly considering how the changes at the mentioned lines might contribute to its state"
    # task_description += ". After you have presented your reasoning, you must conclude your response *only* with one of the following exact statements, without any additional explanatory text: '(1) YES: A security vulnerability detected.' or '(2) NO: No security vulnerability.'."
    task_description = "Please indicate your analysis result *only* with one of the following exact statements, without any additional explanatory text: '(1) YES: A security vulnerability detected.' or '(2) NO: No security vulnerability.'."

    prompt += f"\\nTask: {task_description}"
    return prompt

# --- Resume and Live Writing Utilities ---
def get_already_processed_commit_ids(output_file_path):
    """Check existing output file to find already processed commit IDs."""
    processed_commit_ids = set()
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'commit_id' in data:
                            processed_commit_ids.add(data['commit_id'])
            print(f"Found {len(processed_commit_ids)} already processed commit IDs in existing output file.")
        except Exception as e:
            print(f"Warning: Could not read existing output file {output_file_path}: {e}")
    return processed_commit_ids

def write_results_to_file(results_list, output_file_path, mode='a'):
    """Write results to file and clear the list to save memory."""
    if results_list:
        with open(output_file_path, mode, encoding='utf-8') as f_out:
            for result in results_list:
                f_out.write(json.dumps(result) + '\n')
        results_list.clear()

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate finetuning dataset from PrimeVul paired data using SGLang.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to PrimeVul paired .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated .jsonl dataset.")
    parser.add_argument("--sglang_host", type=str, default="127.0.0.1", help="SGLang server host.")
    parser.add_argument("--sglang_port", type=int, default=30000, help="SGLang server port.")
    
    parser.add_argument('--model_id_for_sglang', type=str, default="Qwen/Qwen3-32B",
                        help="Model identifier for SGLang API calls.")
    parser.add_argument('--model_id_for_tokenizer', type=str, default="Qwen/Qwen3-32B",
                        help="Model identifier for loading the tokenizer.")
    parser.add_argument('--local_model_path_for_tokenizer', type=str, default=None,
                        help="Optional local path to the model files for the tokenizer.")

    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20) 
    parser.add_argument('--max_gen_length', type=int, default=32768, help="Desired maximum generation length for the model's response part.") # Changed default from 8192 to 32768
    parser.add_argument('--seed', type=int, default=1337, help="Seed for model generation if supported by SGLang endpoint.")
    parser.add_argument('--min_p', type=float, default=0.0, help="min_p sampling parameter (note: may not be supported by standard OpenAI API endpoint).")
    parser.add_argument('--logprobs', type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to request logprobs from the model. Pass 'true' or 'false'.")
    parser.add_argument('--enable_thinking', type=lambda x: (str(x).lower() == 'true'), default=True, help='(For Qwen models) Enable thinking mode. Default: True')

    parser.add_argument('--line_number_threshold', type=int, default=20, help="Threshold for adding line numbers to code.")
    args = parser.parse_args()

    # Corrected SGLang API URL
    sglang_api_url = f"http://{args.sglang_host}:{args.sglang_port}/v1" 
    client = OpenAI(api_key="EMPTY", base_url=sglang_api_url)

    tokenizer_obj, is_hf_tokenizer = get_tokenizer_for_model(args.model_id_for_tokenizer, args.local_model_path_for_tokenizer)
    if tokenizer_obj is None and not ("qwen" in args.model_id_for_tokenizer.lower() and _tokenizer_type_cache.get(args.model_id_for_tokenizer) is False): # Check if tiktoken was successfully loaded for qwen
        print(f"Error: Could not load a proper tokenizer for {args.model_id_for_tokenizer}. Exiting.")
        return

    total_model_capacity = get_total_model_capacity(args.model_id_for_sglang)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Check for already processed commit IDs to enable resume functionality
    already_processed_commit_ids = get_already_processed_commit_ids(args.output_file)

    all_samples_raw = []
    with open(args.input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            all_samples_raw.append(json.loads(line))

    samples_by_commit_id = defaultdict(list)
    for sample in all_samples_raw:
        samples_by_commit_id[sample['commit_id']].append(sample)

    processed_results = []
    skipped_due_to_pairing_issue_count = 0
    skipped_due_to_token_limit_count = 0
    processed_pair_count = 0
    skipped_due_to_already_processed_count = 0
    
    commit_ids_to_process_potentially = list(samples_by_commit_id.keys())
    
    # Filter out already processed commit IDs
    commit_ids_to_process = [cid for cid in commit_ids_to_process_potentially if cid not in already_processed_commit_ids]
    
    print(f"Total commit_ids in input: {len(commit_ids_to_process_potentially)}")
    print(f"Already processed commit_ids: {len(already_processed_commit_ids)}")
    print(f"Remaining commit_ids to process: {len(commit_ids_to_process)}")

    for commit_id in tqdm(commit_ids_to_process, desc="Processing commit_ids"):
        items = samples_by_commit_id[commit_id]
        
        if len(items) != 2:
            skipped_due_to_pairing_issue_count += 1
            continue
            
        targets = [item['target'] for item in items]
        if not (targets.count(0) == 1 and targets.count(1) == 1):
            skipped_due_to_pairing_issue_count += 1
            continue
        
        processed_pair_count +=1
        vuln_sample = next(item for item in items if item['target'] == 1)
        non_vuln_sample = next(item for item in items if item['target'] == 0)

        # Process Vulnerable Sample
        vulnerable_lines_info_str = get_changed_lines_info_str(non_vuln_sample['func'], vuln_sample['func'])
        prompt_vuln = format_prompt_for_model(
            vuln_sample['func'], vuln_sample.get('cve_desc'),
            is_vulnerable_ground_truth=True, vulnerable_lines_info_str=vulnerable_lines_info_str,
            line_number_threshold=args.line_number_threshold
        )
        prompt_vuln_tokens = count_tokens(prompt_vuln, tokenizer_obj, is_hf_tokenizer)
        
        max_new_tokens_for_model_call_vuln = total_model_capacity - prompt_vuln_tokens
        api_max_tokens_vuln = min(args.max_gen_length, max_new_tokens_for_model_call_vuln)

        if api_max_tokens_vuln < MIN_REASONABLE_OUTPUT_SIZE:
            skipped_due_to_token_limit_count += 1
        else:
            try:
                extra_body_params = {}
                if args.model_id_for_sglang.startswith("Qwen"):
                    extra_body_params["enable_thinking"] = args.enable_thinking
                    if args.top_k is not None:
                        extra_body_params["top_k"] = args.top_k
                    if args.min_p is not None:
                        extra_body_params["min_p"] = args.min_p

                completion_params = {
                    "model": args.model_id_for_sglang,
                    "messages": [{"role": "user", "content": prompt_vuln}],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": api_max_tokens_vuln,
                    "seed": args.seed,
                    "logprobs": args.logprobs,
                    "extra_body": extra_body_params if extra_body_params else NOT_GIVEN
                }
                
                response_vuln = client.chat.completions.create(**completion_params)
                generated_text_vuln = response_vuln.choices[0].message.content
                processed_results.append({
                    "original_sample_idx": vuln_sample.get("idx"), 
                    "commit_id": commit_id, 
                    "target": 1,
                    "cve_desc_used": vuln_sample.get('cve_desc', "") if vuln_sample.get('cve_desc', "").lower() not in ["none", "null", "na", "n/a", ""] else None,
                    "vulnerable_lines_hint_provided": vulnerable_lines_info_str,
                    "prompt": prompt_vuln, 
                    "generated_response": generated_text_vuln,
                    # "original_func_for_reference": vuln_sample['func'],
                    # "model_params": {
                    #     "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, 
                    #     "max_new_tokens_requested": api_max_tokens_vuln, "seed": args.seed
                    # }
                })
            except Exception as e:
                print(f"Error (Vuln) commit_id {commit_id}, idx {vuln_sample.get('idx')}: {e}")
                processed_results.append({
                    "original_sample_idx": vuln_sample.get("idx"), "commit_id": commit_id, "target": 1,
                    "cve_desc_used": vuln_sample.get('cve_desc', "") if vuln_sample.get('cve_desc', "").lower() not in ["none", "null", "na", "n/a", ""] else None,
                    "vulnerable_lines_hint_provided": vulnerable_lines_info_str,
                    "prompt": prompt_vuln, "generated_response": f"ERROR_SGLANG_CALL: {str(e)}",
                    # "original_func_for_reference": vuln_sample['func'],
                    # "model_params": {
                    #     "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, 
                    #     "max_new_tokens_requested": api_max_tokens_vuln, "seed": args.seed
                    # }
                })

        # Process Non-Vulnerable Sample
        prompt_non_vuln = format_prompt_for_model(
            non_vuln_sample['func'], non_vuln_sample.get('cve_desc'),
            is_vulnerable_ground_truth=False, vulnerable_lines_info_str=None,
            line_number_threshold=args.line_number_threshold
        )
        prompt_non_vuln_tokens = count_tokens(prompt_non_vuln, tokenizer_obj, is_hf_tokenizer)

        max_new_tokens_for_model_call_non_vuln = total_model_capacity - prompt_non_vuln_tokens
        api_max_tokens_non_vuln = min(args.max_gen_length, max_new_tokens_for_model_call_non_vuln)

        if api_max_tokens_non_vuln < MIN_REASONABLE_OUTPUT_SIZE:
            skipped_due_to_token_limit_count += 1
        else:
            try:
                extra_body_params_non_vuln = {}
                if args.model_id_for_sglang.startswith("Qwen"):
                    extra_body_params_non_vuln["enable_thinking"] = args.enable_thinking
                    if args.top_k is not None:
                        extra_body_params_non_vuln["top_k"] = args.top_k
                    if args.min_p is not None:
                        extra_body_params_non_vuln["min_p"] = args.min_p

                completion_params_non_vuln = {
                    "model": args.model_id_for_sglang,
                    "messages": [{"role": "user", "content": prompt_non_vuln}],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": api_max_tokens_non_vuln,
                    "seed": args.seed,
                    "logprobs": args.logprobs,
                    "extra_body": extra_body_params_non_vuln if extra_body_params_non_vuln else NOT_GIVEN
                }

                response_non_vuln = client.chat.completions.create(**completion_params_non_vuln)
                generated_text_non_vuln = response_non_vuln.choices[0].message.content
                processed_results.append({
                    "original_sample_idx": non_vuln_sample.get("idx"), 
                    "commit_id": commit_id, 
                    "target": 0,
                    "cve_desc_used": non_vuln_sample.get('cve_desc', "") if non_vuln_sample.get('cve_desc', "").lower() not in ["none", "null", "na", "n/a", ""] else None,
                    "vulnerable_lines_hint_provided": None, # No specific lines hint for non-vuln
                    "prompt": prompt_non_vuln, 
                    "generated_response": generated_text_non_vuln,
                    # "original_func_for_reference": non_vuln_sample['func'],
                    # "model_params": {
                    #     "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, 
                    #     "max_new_tokens_requested": api_max_tokens_non_vuln, "seed": args.seed
                    # }
                })
            except Exception as e:
                print(f"Error (Non-Vuln) commit_id {commit_id}, idx {non_vuln_sample.get('idx')}: {e}")
                processed_results.append({
                    "original_sample_idx": non_vuln_sample.get("idx"), "commit_id": commit_id, "target": 0,
                    "cve_desc_used": non_vuln_sample.get('cve_desc', "") if non_vuln_sample.get('cve_desc', "").lower() not in ["none", "null", "na", "n/a", ""] else None,
                    "vulnerable_lines_hint_provided": None,
                    "prompt": prompt_non_vuln, "generated_response": f"ERROR_SGLANG_CALL: {str(e)}",
                    # "original_func_for_reference": non_vuln_sample['func'],
                    # "model_params": {
                    #     "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                    #     "max_new_tokens_requested": api_max_tokens_non_vuln, "seed": args.seed
                    # }
                })
        
        # Write results incrementally to save memory and enable live updates
        # Write after each commit_id is fully processed (both vuln and non-vuln)
        if len(processed_results) >= 2:  # Should have exactly 2 results per commit_id
            write_results_to_file(processed_results, args.output_file, mode='a')
    
    # Write any remaining results
    if processed_results:
        write_results_to_file(processed_results, args.output_file, mode='a')
    
    print(f"Finished processing.")
    print(f"Total commit_ids in input: {len(samples_by_commit_id)}")
    print(f"Already processed commit_ids (skipped): {len(already_processed_commit_ids)}")
    print(f"Commit_ids processed as valid pairs: {processed_pair_count}")
    print(f"Commit_ids skipped due to pairing issues (not 2 items, or not one vuln/non-vuln): {skipped_due_to_pairing_issue_count}")
    print(f"Individual samples skipped due to token limit: {skipped_due_to_token_limit_count}")
    print(f"Generated dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
