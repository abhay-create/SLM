"""
dataset_overlap.py
Per-stage PyTorch datasets + val set builders with REPLAY support to prevent catastrophic forgetting.

Each stage has:
  - A train dataset (tokenized, chunked to seq_len, streamed or cached)
  - Multiple sources: current stage + replay buffer from previous stages
  - A val dataset   (fixed held-out split, loaded once into memory)

Val sets for ALL three stages are always available so the training loop
can log cross-stage val losses at every eval step.
"""

import os
import pickle
import random
import numpy as np
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizers import Tokenizer


# ─── Token-level chunking ─────────────────────────────────────────────────────

def tokenize_and_chunk(
    text_iter  : Iterator[str],
    tokenizer  : Tokenizer,
    seq_len    : int,
    max_tokens : Optional[int] = None,
    bos_id     : int = 2,    # [BOS] index in special_tokens
    eos_id     : int = 3,    # [EOS]
) -> list[list[int]]:
    """
    Streams text, tokenizes, concatenates with BOS/EOS, then chunks into
    non-overlapping windows of seq_len+1 tokens (input + target shift).
    Returns list of token id lists, each of length seq_len+1.
    """
    buffer  = []
    chunks  = []
    tokens_seen = 0

    with tqdm(total=max_tokens, unit="tok", desc="Tokenizing & chunking", disable=(max_tokens is None)) as pbar:
        for text in text_iter:
            if not text or not text.strip():
                continue
            ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
            buffer.extend(ids)

            while len(buffer) >= seq_len + 1:
                chunks.append(buffer[: seq_len + 1])
                buffer = buffer[seq_len + 1 :]
                tokens_seen += seq_len + 1
                pbar.update(seq_len + 1)
                if max_tokens and tokens_seen >= max_tokens:
                    pbar.close()
                    return chunks

    return chunks


# ─── Cached val set builder ───────────────────────────────────────────────────

def build_val_set(
    stage      : int,
    tokenizer  : Tokenizer,
    seq_len    : int,
    n_docs     : int,
    cache_dir  : str = "cache",
) -> list[list[int]]:
    """
    Loads or builds + caches the val set for a given stage.
    Always uses the same held-out seed so val sets are reproducible.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"val_stage{stage}_seq{seq_len}.pkl")

    if os.path.exists(cache_path):
        print(f"[dataset] Loading cached val_stage{stage} from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[dataset] Building val_stage{stage} ({n_docs} docs)...")
    text_iter = _val_iter(stage, n_docs)
    chunks    = tokenize_and_chunk(text_iter, tokenizer, seq_len)

    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[dataset] val_stage{stage} saved: {len(chunks)} chunks → {cache_path}")
    return chunks


def _val_iter(stage: int, n_docs: int) -> Iterator[str]:
    """Returns a deterministic held-out slice for each stage."""
    from datasets import load_dataset
    rng = random.Random(42)   # fixed seed for reproducibility

    if stage == 0:
        ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        for i, ex in enumerate(ds):
            if i >= n_docs: break
            yield ex["text"]

    elif stage == 1:
        # Mix of SimpleWiki + BabyLM
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple",
                          split="train", streaming=True)
        count = 0
        for ex in ds:
            if rng.random() < 0.05:    # ~5% holdout sample
                yield ex["text"]
                count += 1
                if count >= n_docs: break

    elif stage == 2:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        count = 0
        for ex in ds:
            if ex.get("score", 0) >= 3 and rng.random() < 0.01:
                yield ex["text"]
                count += 1
                if count >= n_docs: break


# ─── In-memory val dataset ────────────────────────────────────────────────────

class ChunkedTokenDataset(Dataset):
    """Simple dataset from a list of token id chunks."""
    def __init__(self, chunks: list[list[int]]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        x     = chunk[:-1]    # input
        y     = chunk[1:]     # target (shifted by 1)
        return x, y


# ─── Streaming train dataset with REPLAY ──────────────────────────────────────

class StreamingStageDataset(Dataset):
    """
    Pre-tokenizes + chunks a stage's training data into memory.
    Supports REPLAY: mixes current stage data with previous stage data
    to prevent catastrophic forgetting.

    Call build() before using in a DataLoader.
    """
    def __init__(self):
        self.chunks: list[list[int]] = []

    def build(
        self,
        stage      : int,
        tokenizer  : Tokenizer,
        seq_len    : int,
        max_tokens : int,
        cache_dir  : str = "cache",
        replay_ratio: float = 0.0,  # % of tokens from previous stage(s)
    ):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"train_stage{stage}_seq{seq_len}_replay{replay_ratio:.1f}.pkl")

        if os.path.exists(cache_path):
            print(f"[dataset] Loading cached train_stage{stage} (replay={replay_ratio:.1f})")
            with open(cache_path, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"[dataset] Loaded {len(self.chunks):,} chunks")
            return self

        print(f"[dataset] Building train_stage{stage} (max_tokens={max_tokens:,}, replay={replay_ratio:.1f})...")
        text_iter    = _train_iter(stage, replay_ratio=replay_ratio)
        self.chunks  = tokenize_and_chunk(text_iter, tokenizer, seq_len, max_tokens)
        random.shuffle(self.chunks)   # shuffle once before caching

        with open(cache_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[dataset] train_stage{stage}: {len(self.chunks):,} chunks → {cache_path}")
        return self

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


def _train_iter(stage: int, replay_ratio: float = 0.0) -> Iterator[str]:
    """
    Yields text from current stage, optionally interleaved with previous stage(s)
    based on replay_ratio.
    
    replay_ratio: float in [0, 1]
      - 0.0: only current stage
      - 0.2: 20% previous stage(s), 80% current
      - 1.0: 100% previous stage(s) (pathological)
    """
    from datasets import load_dataset

    rng = random.Random(42)  # fixed seed for reproducibility
    
    # Get current stage source(s)
    if stage == 0:
        current_sources = [iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))]
        
    elif stage == 1:
        wiki  = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
        try:
            baby  = load_dataset("babylm/babylm_10M", split="train", streaming=True)
            current_sources = [iter(wiki), iter(baby)]
        except Exception:
            current_sources = [iter(wiki)]
    
    elif stage == 2:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        current_sources = [iter(ds)]
    
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Get replay source(s) if replay_ratio > 0
    replay_sources = []
    if replay_ratio > 0 and stage > 0:
        if stage >= 1:
            # Add TinyStories (stage 0 data)
            replay_sources.append(iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True)))
        if stage >= 2:
            # Add SimpleWiki + BabyLM (stage 1 data)
            wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
            try:
                baby = load_dataset("babylm/babylm_10M", split="train", streaming=True)
                replay_sources.extend([iter(wiki), iter(baby)])
            except Exception:
                replay_sources.append(iter(wiki))
    
    # Interleave: use rng.random() to decide whether to draw from replay or current
    current_active = list(range(len(current_sources)))
    replay_active  = list(range(len(replay_sources)))
    
    while current_active or replay_active:
        # Choose which pool to draw from based on replay_ratio
        if replay_active and rng.random() < replay_ratio:
            idx = rng.choice(replay_active)
            try:
                ex   = next(replay_sources[idx])
                text = ex.get("text") or ex.get("sentence") or ""
                if text: yield text
            except StopIteration:
                replay_active.remove(idx)
        elif current_active:
            idx = rng.choice(current_active)
            try:
                ex   = next(current_sources[idx])
                text = ex.get("text") or ex.get("sentence") or ""
                if text: yield text
            except StopIteration:
                current_active.remove(idx)
        else:
            break


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 2,
        pin_memory  = True,
    )


# ─── Load all three val sets ──────────────────────────────────────────────────

def load_all_val_sets(
    tokenizer : Tokenizer,
    cache_dir : str = "cache",
) -> dict:
    """
    Returns a dict of three pre-built val DataLoaders:
      {"s0": loader, "s1": loader, "s2": loader}
    Used in train.py to log all cross-stage val losses.
    """
    configs = {
        "s0": (0, 256, 5000),
        "s1": (1, 384, 3000),
        "s2": (2, 512, 2000),
    }
    loaders = {}
    for key, (stage, seq_len, n_docs) in configs.items():
        chunks  = build_val_set(stage, tokenizer, seq_len, n_docs, cache_dir)
        ds      = ChunkedTokenDataset(chunks)
        loaders[key] = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    return loaders
