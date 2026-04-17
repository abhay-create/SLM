#!/usr/bin/env python3
"""
upload_to_hf.py — Upload SLM checkpoints and files to Hugging Face Hub.

Usage:
    # First, authenticate with HF
    huggingface-cli login
    
    # Then run this script
    python upload_to_hf.py \
        --repo_id your-username/slm \
        --stage0_ckpt /path/to/stage0_best_1st_trial.pt \
        --stage2_ckpt /path/to/stage2_best.pt \
        --create
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    print("✓ huggingface_hub imported successfully")
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)


def upload_to_hub(
    repo_id: str,
    stage0_ckpt: Path,
    stage2_ckpt: Path,
    tokenizer_50k: Path,
    tokenizer_corpus: Path,
    config_stage0: Path,
    config_stage2: Path,
    readme: Path,
    requirements: Path,
    inference_script: Path,
    create: bool = False,
) -> None:
    """Upload model files to HF Hub."""
    
    print(f"\n{'='*70}")
    print(f"  UPLOADING TO HUGGING FACE HUB")
    print(f"  Repository: {repo_id}")
    print(f"{'='*70}\n")
    
    api = HfApi()
    
    # Create repo if needed
    if create:
        print("📦 Creating repository...")
        try:
            repo_url = create_repo(repo_id, private=False, exist_ok=True)
            print(f"✓ Repository created: {repo_url}")
        except Exception as e:
            print(f"⚠ Repository creation: {e}")
    
    # Upload files
    files_to_upload = [
        (stage0_ckpt, "stage0_best_1st_trial.pt"),
        (stage2_ckpt, "stage2_best.pt"),
        (tokenizer_50k, "tokenizer_50k.json"),
        (tokenizer_corpus, "tokenizer_corpus.json"),
        (config_stage0, "config_stage0.yaml"),
        (config_stage2, "config_stage2.yaml"),
        (readme, "README.md"),
        (requirements, "requirements.txt"),
        (inference_script, "inference.py"),
    ]
    
    print("\n📤 Uploading files...\n")
    
    for local_path, hf_name in files_to_upload:
        if not local_path.exists():
            print(f"⚠ Skipping {hf_name}: file not found at {local_path}")
            continue
        
        try:
            print(f"  Uploading {hf_name}...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_name,
                repo_id=repo_id,
                repo_type="model",
            )
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✓ ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"✓ Upload complete!")
    print(f"\nModel available at: https://huggingface.co/{repo_id}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Upload SLM checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HF repo ID (e.g., your-username/slm)"
    )
    parser.add_argument(
        "--stage0_ckpt",
        type=str,
        default="/home/user20/NLP/slm/checkpoints/stage0_best _1st_trial.pt",
        help="Path to stage0 checkpoint"
    )
    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        default="/home/user20/NLP/slm/checkpoints/stage2_best.pt",
        help="Path to stage2 checkpoint"
    )
    parser.add_argument(
        "--tokenizer_50k",
        type=str,
        default="/home/user20/NLP/slm/tokenizers/tokenizer_50k.json",
        help="Path to 50k tokenizer"
    )
    parser.add_argument(
        "--tokenizer_corpus",
        type=str,
        default="/home/user20/NLP/slm/tokenizers/tokenizer_corpus.json",
        help="Path to corpus tokenizer"
    )
    parser.add_argument(
        "--config_stage0",
        type=str,
        default="/home/user20/NLP/slm/configs/stage0.yaml",
        help="Path to stage0 config"
    )
    parser.add_argument(
        "--config_stage2",
        type=str,
        default="/home/user20/NLP/slm/configs/stage2.yaml",
        help="Path to stage2 config"
    )
    parser.add_argument(
        "--readme",
        type=str,
        default="./README.md",
        help="Path to README.md"
    )
    parser.add_argument(
        "--requirements",
        type=str,
        default="./requirements.txt",
        help="Path to requirements.txt"
    )
    parser.add_argument(
        "--inference_script",
        type=str,
        default="./inference.py",
        help="Path to inference.py"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create repository if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    stage0_ckpt = Path(args.stage0_ckpt)
    stage2_ckpt = Path(args.stage2_ckpt)
    tokenizer_50k = Path(args.tokenizer_50k)
    tokenizer_corpus = Path(args.tokenizer_corpus)
    config_stage0 = Path(args.config_stage0)
    config_stage2 = Path(args.config_stage2)
    readme = Path(args.readme)
    requirements = Path(args.requirements)
    inference_script = Path(args.inference_script)
    
    # Verify files exist
    for path, name in [
        (stage0_ckpt, "Stage 0 checkpoint"),
        (stage2_ckpt, "Stage 2 checkpoint"),
    ]:
        if not path.exists():
            print(f"❌ ERROR: {name} not found at {path}")
            sys.exit(1)
    
    # Upload
    upload_to_hub(
        args.repo_id,
        stage0_ckpt,
        stage2_ckpt,
        tokenizer_50k,
        tokenizer_corpus,
        config_stage0,
        config_stage2,
        readme,
        requirements,
        inference_script,
        create=args.create,
    )


if __name__ == "__main__":
    main()
