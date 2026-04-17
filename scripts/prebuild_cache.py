"""
Pre-build and cache all datasets for all stages.
Run this ONCE before any training to avoid waiting during training.

All datasets are cached at EVERY seq_len they will be used at to ensure
exact matches during training (no truncation/padding of replay buffers).

Usage:
  python prebuild_cache.py
"""

from tokenizers import Tokenizer
from src.dataset import StreamingStageDataset
import os

TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
CACHE_DIR = "cache"

# ALL datasets × ALL seq_lens they're used at
# Format: (dataset_name, seq_len, max_tokens)
CACHE_CONFIGS = [
    # Stage 0: TinyStories main
    ("tinystories", 256, 50_000_000),
    
    # Stage 0b: BabyLM Easy main + TinyStories replay
    ("babylm_easy", 384, 50_000_000),
    ("tinystories", 384, 50_000_000),     # Replay source for 0b
    
    # Stage 0c: BabyLM Hard main + replay sources
    ("babylm_hard", 512, 60_000_000),
    ("babylm_easy", 512, 50_000_000),     # Replay source for 0c
    ("tinystories", 512, 50_000_000),     # Replay source for 0c
    
    # Stage 1: SimpleWiki main + BabyLM Hard replay
    ("simplewiki", 512, 220_000_000),
    ("babylm_hard", 512, 60_000_000),     # Replay source for stage 1
    
    # Stage 2: FineWeb-Edu main + SimpleWiki replay
    ("fineweb_edu", 512, 500_000_000),
    ("simplewiki", 512, 220_000_000),     # Replay source for stage 2
]

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print("=" * 70)
print("PRE-BUILDING DATASET CACHES FOR ALL STAGES")
print("=" * 70)

for dataset_name, seq_len, max_tokens in CACHE_CONFIGS:
    cache_path = os.path.join(CACHE_DIR, f"train_{dataset_name}_seq{seq_len}.pkl")
    
    # Skip if already cached
    if os.path.exists(cache_path):
        size_mb = os.path.getsize(cache_path) / (1024**2)
        print(f"✓ {dataset_name:15} @ seq_len={seq_len:3}: {size_mb:6.0f} MB (cached)")
        continue
    
    print(f"⏳ {dataset_name:15} @ seq_len={seq_len:3}...", end=" ", flush=True)
    
    ds = StreamingStageDataset().build(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_tokens=max_tokens,
        cache_dir=CACHE_DIR,
    )
    
    size_mb = os.path.getsize(cache_path) / (1024**2)
    print(f"{len(ds.chunks):8,} chunks → {size_mb:6.0f} MB ✓")

print("\n" + "=" * 70)
print("✅ ALL CACHES COMPLETE! Training will start instantly.")
print("=" * 70)
