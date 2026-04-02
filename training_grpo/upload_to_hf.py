#!/usr/bin/env python3
"""
Upload GRPO checkpoints to HuggingFace: Andrefty/qwen3-4b-vuln-grpo-verl

Commit 1: global_step_100 model weights (best checkpoint by val accuracy)
Commit 2: global_step_209 model weights + train_rollout + val_output + debug jsonl
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

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
    parser.add_argument("--commit", choices=["step100", "step209", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo exists or error: {e}")

    # ── COMMIT 1: step 100 ──────────────────────────────────────────────
    if args.commit in ("step100", "both"):
        step100_dir = BASE_DIR / "global_step_100"
        model_files = get_model_files(step100_dir)
        
        print(f"\n=== Commit 1: step 100 model ({len(model_files)} files) ===")
        for local, repo in model_files:
            print(f"  {repo} ({os.path.getsize(local)/1e9:.2f} GB)")
        
        if not args.dry_run:
            operations = [
                CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
                for local_path, repo_path in model_files
            ]
            upload_commit(api, REPO_ID, operations,
                "Add step 100 checkpoint (best val accuracy: 89.9%, LR=5e-6, n=16)")

    # ── COMMIT 2: step 209 + rollouts ───────────────────────────────────
    if args.commit in ("step209", "both"):
        step209_dir = BASE_DIR / "global_step_209"
        model_files = get_model_files(step209_dir)
        rollout_files = get_rollout_files(BASE_DIR / "train_rollout", "train_rollout")
        val_files = get_rollout_files(BASE_DIR / "val_output", "val_output")
        debug_file = BASE_DIR / "verl_completions_debug.jsonl"
        
        all_files = model_files + rollout_files + val_files + [(debug_file, "verl_completions_debug.jsonl")]
        
        print(f"\n=== Commit 2: step 209 model + rollouts ({len(all_files)} files) ===")
        for local, repo in all_files:
            size = os.path.getsize(local) / 1e6
            print(f"  {repo} ({size:.1f} MB)")
        
        if not args.dry_run:
            operations = [
                CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
                for local_path, repo_path in all_files
            ]
            upload_commit(api, REPO_ID, operations,
                "Add step 209 final checkpoint + train_rollout + val_output + debug completions")

    print("\n🎉 Upload complete!" if not args.dry_run else "\n✅ Dry run complete.")

if __name__ == "__main__":
    main()
