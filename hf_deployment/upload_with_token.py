#!/usr/bin/env python3
"""
upload_with_token.py - Upload to HF Hub using token (easier than CLI)

Usage:
    python upload_with_token.py --token YOUR_HF_TOKEN --username YOUR_USERNAME
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
load_dotenv()

def upload_with_token(token: str, username: str):
    """Upload using token instead of CLI."""
    
    print("\n" + "="*70)
    print("  UPLOADING SLM TO HUGGING FACE")
    print("="*70 + "\n")
    
    # Setup API
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_id = f"{os.getenv('username')}/slm"
    
    # Test token
    try:
        user_info = api.whoami()
        print(f"✓ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\nMake sure your token is valid!")
        print("Get a new one at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Create repo
    print(f"\n📦 Creating repository: {repo_id}")
    try:
        repo_url = create_repo(repo_id, private=False, exist_ok=True, token=token)
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"⚠ Repository: {e}")
    
    # Files to upload
    files = [
        ("stage0_best_1st_trial.pt", "stage0_best_1st_trial.pt"),
        ("stage2_best.pt", "stage2_best.pt"),
        ("tokenizer_50k.json", "tokenizer_50k.json"),
        ("tokenizer_corpus.json", "tokenizer_corpus.json"),
        ("config_stage0.yaml", "config_stage0.yaml"),
        ("config_stage2.yaml", "config_stage2.yaml"),
        ("model.py", "model.py"),
        ("tokenizer.py", "tokenizer.py"),
        ("inference.py", "inference.py"),
        ("requirements.txt", "requirements.txt"),
        ("README.md", "README.md"),
    ]
    
    print(f"\n📤 Uploading files ({len(files)} total)...\n")
    
    uploaded = 0
    failed = []
    
    for local_file, remote_name in files:
        if not Path(local_file).exists():
            print(f"  ⊘ {remote_name} — file not found, skipping")
            continue
        
        try:
            print(f"  ↑ {remote_name}...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model",
            )
            size = Path(local_file).stat().st_size / (1024*1024)
            print(f"✓ ({size:.1f}MB)")
            uploaded += 1
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failed.append(remote_name)
    
    print("\n" + "="*70)
    print(f"✅ UPLOAD COMPLETE: {uploaded}/{len(files)} files")
    if failed:
        print(f"⚠ Failed: {', '.join(failed)}")
    print("="*70 + "\n")
    
    print(f"🎉 Your model is now at:")
    print(f"   https://huggingface.co/{repo_id}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Upload SLM to HF Hub with token")
    parser.add_argument("--token", type=str, required=True, help="HF API token")
    parser.add_argument("--username", type=str, required=True, help="HF username")
    
    args = parser.parse_args()
    upload_with_token(args.token, args.username)


if __name__ == "__main__":
    main()
