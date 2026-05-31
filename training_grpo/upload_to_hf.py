#!/usr/bin/env python3
"""
Upload GRPO checkpoints to HuggingFace: Andrefty/qwen3-4b-vuln-grpo-verl

global_step_<step> model weights + train_rollout + val_output + eval results + README.md + (optional) debug jsonl
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, create_repo

REPO_ID = "Andrefty/qwen3-4b-vuln-grpo-verl"
BASE_DIR = Path("/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/grpo_qwen3_4b_verl")

def get_model_files(step_dir: Path):
    """Get (local_path, repo_path) pairs for model files from a step's huggingface dir."""
    hf_dir = step_dir / "actor" / "huggingface"
    pairs = []
    for f in sorted(hf_dir.iterdir()):
        if f.is_file():
            pairs.append((f, f.name))
    return pairs

def get_rollout_files(rollout_dir: Path, repo_prefix: str):
    """Get (local_path, repo_path) pairs for rollout jsonl files."""
    pairs = []
    for f in sorted(rollout_dir.iterdir()):
        if f.is_file() and f.suffix == '.jsonl':
            pairs.append((f, f"{repo_prefix}/{f.name}"))
    return pairs

def upload_commit(api, repo_id, operations, commit_msg):
    print(f"\n📤 Committing: {commit_msg}")
    print(f"   Files: {len(operations)}")
    total_bytes = sum(os.path.getsize(op.path_or_fileobj) for op in operations)
    print(f"   Total size: {total_bytes / 1e9:.1f} GB")
    
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_msg,
    )
    print(f"   ✅ Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=209, help="Global step of the checkpoint to upload")
    parser.add_argument("-m", "--message", type=str, help="Override commit message")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo exists or error: {e}")

    # ── COMMIT: step + rollouts + eval results ──────────────────────
    step_dir = BASE_DIR / f"global_step_{args.step}"
    model_files = get_model_files(step_dir)
    rollout_files = get_rollout_files(BASE_DIR / "train_rollout", "train_rollout")
    val_files = get_rollout_files(BASE_DIR / "val_output", "val_output")
    
    all_files = model_files + rollout_files + val_files
    
    debug_file = BASE_DIR / "verl_completions_debug.jsonl"
    if debug_file.exists():
        all_files.append((debug_file, "verl_completions_debug.jsonl"))
        
    readme_file = step_dir / "README.md"
    # if not readme_file.exists():
    #     readme_file = BASE_DIR / "README.md"
    if readme_file.exists():
        all_files.append((readme_file, "README.md"))
        
    eval_results_dir = step_dir / "grpo_qwen3_4b_eval_results"
    if eval_results_dir.exists():
        for f in eval_results_dir.rglob('*'):
            if f.is_file():
                all_files.append((f, f"grpo_qwen3_4b_eval_results/{f.relative_to(eval_results_dir)}"))
    
    print(f"\n=== Uploading step {args.step} model + rollouts + eval ({len(all_files)} files) ===")
    for local, repo in all_files:
        size = os.path.getsize(local) / 1e6
        print(f"  {repo} ({size:.1f} MB)")
    
    if not args.dry_run:
        operations = []
        uploading_evals = any(repo_path.startswith("grpo_qwen3_4b_eval_results/") for _, repo_path in all_files)
        
        if uploading_evals:
            try:
                repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
                if any(f.startswith("grpo_qwen3_4b_eval_results/") for f in repo_files):
                    operations.append(CommitOperationDelete(path_in_repo="grpo_qwen3_4b_eval_results", is_folder=True))
                    print("  🗑️ Scheduled deletion of existing 'grpo_qwen3_4b_eval_results' in the remote repo.")
            except Exception as e:
                print(f"  ⚠️ Could not check remote files for deletion: {e}")

        operations.extend([
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
            for local_path, repo_path in all_files
        ])
        commit_msg = args.message if args.message else f"Checkpoint step {args.step} upload"
        upload_commit(api, REPO_ID, operations, commit_msg)

    print("\n🎉 Upload complete!" if not args.dry_run else "\n✅ Dry run complete.")

if __name__ == "__main__":
    main()
