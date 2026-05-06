#!/usr/bin/env python3
"""
Upload a specified directory to HuggingFace: Andrefty/qwen3-4b-vuln-sft-research
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

REPO_ID = "Andrefty/qwen3-4b-vuln-sft-research"

def get_model_files(step_dir: Path):
    """Get (local_path, repo_path) pairs for model files from a step's dir."""
    pairs = []
    for f in sorted(step_dir.iterdir()):
        if f.is_file():
            pairs.append((f, f.name))
    return pairs

def upload_commit(api, repo_id, operations, commit_msg):
    print(f"\n📤 Committing: {commit_msg}")
    print(f"   Files: {len(operations)}")
    total_bytes = sum(os.path.getsize(op.path_or_fileobj) for op in operations)
    print(f"   Total size: {total_bytes / 1e9:.2f} GB")
    
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_msg,
    )
    print(f"   ✅ Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory to upload")
    parser.add_argument("--message", type=str, required=True, help="Commit message")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo exists or error: {e}")

    step_dir = Path(args.dir)
    if not step_dir.exists() or not step_dir.is_dir():
        print(f"❌ Directory {step_dir} does not exist or is not a directory.")
        return

    model_files = get_model_files(step_dir)
    if not model_files:
        print(f"❌ No files found in {step_dir}")
        return

    print(f"\n=== Uploading {step_dir.name} ({len(model_files)} files) ===")
    for local, repo in model_files:
        size = os.path.getsize(local) / 1e9
        if size > 0.01:
            print(f"  {repo} ({size:.2f} GB)")
        else:
            print(f"  {repo} ({os.path.getsize(local)/1e6:.2f} MB)")
    
    if not args.dry_run:
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
            for local_path, repo_path in model_files
        ]
        upload_commit(api, REPO_ID, operations, args.message)

    print("\n🎉 Upload complete!" if not args.dry_run else "\n✅ Dry run complete.")

if __name__ == "__main__":
    main()
