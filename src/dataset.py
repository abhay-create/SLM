"""
dataset.py — Per-stage datasets, tokenization utilities, and val set builders.

This module exposes the dataset iteration and caching primitives used by
both curriculum training and expansion stages. Key capabilities:
    - Text iterators for multiple corpora (TinyStories, BabyLM, SimpleWiki, ROC, etc.)
    - Tokenization and chunking utilities (`tokenize_and_chunk`) that produce
        fixed-size token windows used for language model training
    - Streaming and cached train dataset construction with optional replay mixing
        (see `_load_replay_chunks` and `StreamingStageDataset.build`)
    - Val set building and caching (`build_val_set`) and a convenient
        `load_all_val_sets` helper returning DataLoaders used during evaluation

Cache keys use `dataset_name + seq_len` so each stage gets a dedicated cache
file. Val sets are built/cached once and reused across stages to allow
consistent cross-domain tracking.
"""

import glob
import hashlib
import os
import pickle
import random
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


# ─── BabyLM source lists ──────────────────────────────────────────────────────

# Simpler sources: child-directed speech, conversational, simple dialogue
BABYLM_EASY = {
    "switchboard",          # Spoken dialogue transcripts
    "bnc",                  # British National Corpus (many spoken examples)
    "childes",              # Child-directed speech
}

# Harder sources: books, educational, structured content
BABYLM_HARD = {
    "project gutenberg",    # Books and longer texts
    "wikipedia",            # Encyclopedic content
    "qed",                  # Educational subtitles
}

# Fallback field names to check for source in BabyLM HF dataset
_BABYLM_SOURCE_FIELDS = ("data-source", "source", "dataset", "domain", "file")


# ─── Text iterators ───────────────────────────────────────────────────────────

def _iter_tinystories(split: str = "train") -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for ex in ds:
        yield ex["text"]


def _iter_babylm(allowed_sources: set, seed: int = 0) -> Iterator[str]:
    """
    Streams BabyLM 100M, filtering by source name.
    If source metadata isn't available in the dataset, yields all text
    (graceful fallback — log a warning so user knows).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "BabyLM-community/babylm-eng",
        split    = "train",
        streaming= True,
    )

    source_field = None    # will be detected on first example
    warned       = False

    for ex in ds:
        # Detect source field once
        if source_field is None:
            for field in _BABYLM_SOURCE_FIELDS:
                if field in ex:
                    source_field = field
                    break
            if source_field is None and not warned:
                print("[dataset] WARNING: BabyLM has no source field — yielding all text")
                warned = True

        # Filter by source if possible
        if source_field is not None:
            src = str(ex.get(source_field, "")).lower()
            # Check if any of our target sources appear in the source string
            if not any(s in src for s in allowed_sources):
                continue

        text = ex.get("text") or ex.get("sentence") or ""
        if text.strip():
            yield text


def _iter_simplewiki() -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.simple",
        split    = "train",
        streaming= True,
    )
    for ex in ds:
        yield ex["text"]


def _iter_fineweb(min_score: float = 3.0) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name     = "sample-10BT",
        split    = "train",
        streaming= True,
    )
    for ex in ds:
        if ex.get("score", 0) >= min_score:
            yield ex["text"]


# ─── Hash-based deterministic val split (S-5 fix for H-1) ──────────────────

def _hash_is_val(text: str, val_fraction: float, seed: str = "slm_val") -> bool:
    """
    Returns True if this document belongs in the held-out validation set.
    Uses MD5 of a fixed seed + first 64 chars, so the split is deterministic
    on document content (not on iterator order), making it resume-safe.
    """
    key = (seed + text[:64]).encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    return (h % 10000) < int(val_fraction * 10000)


# ─────────────────────────────────────────────────────────────────────────────

def _iter_rocstories() -> Iterator[str]:
    """Loops over ROCStories training docs (excludes val-split documents)."""
    from datasets import load_dataset
    ds = load_dataset("mintujupally/ROCStories", split="train")
    while True:
        for ex in ds:
            text = ex.get("text", ex.get("story", ""))
            if text and not _hash_is_val(text, val_fraction=0.10, seed="slm_roc_val"):
                yield text


def _iter_rocstories_val(n_docs: int) -> Iterator[str]:
    """One-pass iterator over the held-out ROCStories val docs."""
    from datasets import load_dataset
    ds = load_dataset("mintujupally/ROCStories", split="train")
    count = 0
    for ex in ds:
        text = ex.get("text", ex.get("story", ""))
        if text and _hash_is_val(text, val_fraction=0.10, seed="slm_roc_val"):
            yield text
            count += 1
            if count >= n_docs:
                break


def _iter_simplestories() -> Iterator[str]:
    """Loops over SimpleStories training docs (excludes val-split documents)."""
    from datasets import load_dataset
    ds = load_dataset("SimpleStories/SimpleStories", split="train")
    while True:
        for ex in ds:
            text = ex.get("story", ex.get("text", ""))
            if text and not _hash_is_val(text, val_fraction=0.10, seed="slm_simple_val"):
                yield text


def _iter_simplestories_val(n_docs: int) -> Iterator[str]:
    """One-pass iterator over the held-out SimpleStories val docs."""
    from datasets import load_dataset
    ds = load_dataset("SimpleStories/SimpleStories", split="train")
    count = 0
    for ex in ds:
        text = ex.get("story", ex.get("text", ""))
        if text and _hash_is_val(text, val_fraction=0.10, seed="slm_simple_val"):
            yield text
            count += 1
            if count >= n_docs:
                break


def _iter_children_stories() -> Iterator[str]:
    """Loops over Children-Stories training docs (excludes val-split documents)."""
    from datasets import load_dataset
    ds = load_dataset("ajibawa-2023/Children-Stories-Collection", split="train")
    while True:
        for ex in ds:
            text = ex.get("text", ex.get("story", ""))
            if text and not _hash_is_val(text, val_fraction=0.10, seed="slm_child_val"):
                yield text


def _iter_children_stories_val(n_docs: int) -> Iterator[str]:
    """One-pass iterator over the held-out Children-Stories val docs."""
    from datasets import load_dataset
    ds = load_dataset("ajibawa-2023/Children-Stories-Collection", split="train")
    count = 0
    for ex in ds:
        text = ex.get("text", ex.get("story", ""))
        if text and _hash_is_val(text, val_fraction=0.10, seed="slm_child_val"):
            yield text
            count += 1
            if count >= n_docs:
                break


# ─── Train iterator dispatcher ────────────────────────────────────────────────

def get_train_iter(dataset_name) -> Iterator[str]:
    """
    Maps dataset_name (from yaml config) to its text iterator.
    If dataset_name is a string, returns that iterator.
    If dataset_name is a dict mapping names to weights, returns a mixed iterator.

    Imports are performed lazily inside each lambda so that optional
    modules (writingprompts_dataset, legal_dataset) are only loaded when
    the corresponding dataset is actually requested.
    """
    if isinstance(dataset_name, dict):
        return get_mixed_train_iter(dataset_name)

    dispatch = {
        "tinystories"        : lambda: _iter_tinystories("train"),
        "legal_documents"    : lambda: __import__(
                                    "src.legal_dataset", fromlist=["iter_legal_documents"]
                                ).iter_legal_documents(),
        "babylm_easy"        : lambda: _iter_babylm(BABYLM_EASY),
        "babylm_hard"        : lambda: _iter_babylm(BABYLM_HARD),
        "simplewiki"         : _iter_simplewiki,
        "fineweb_edu"        : _iter_fineweb,
        "writingprompts"     : lambda: __import__(
                                    "src.writingprompts_dataset",
                                    fromlist=["iter_writingprompts"]
                                ).iter_writingprompts("train"),
        "writingprompts_easy": lambda: __import__(
                                    "src.writingprompts_dataset",
                                    fromlist=["iter_writingprompts"]
                                ).iter_writingprompts("train", max_length_filter=500),
        "rocstories"         : _iter_rocstories,
        "simplestories"      : _iter_simplestories,
        "children_stories"   : _iter_children_stories,
    }

    if dataset_name not in dispatch:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Valid options: {list(dispatch.keys())}"
        )
    return dispatch[dataset_name]()


def get_mixed_train_iter(dataset_dict: dict) -> Iterator[str]:
    """
    Takes a dictionary of {dataset_name: weight} and yields texts
    sampled according to the weights. Loops endlessly.
    """
    # Normalize weights
    names = list(dataset_dict.keys())
    weights = [dataset_dict[name] for name in names]
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    
    # Initialize iterators
    iterators = {name: get_train_iter(name) for name in names}
    
    while True:
        # Choose a dataset based on probabilities
        chosen_name = random.choices(names, weights=probs, k=1)[0]
        try:
            yield next(iterators[chosen_name])
        except StopIteration:
            # If an iterator is exhausted, restart it
            iterators[chosen_name] = get_train_iter(chosen_name)
            yield next(iterators[chosen_name])


# ─── Val iterator dispatcher ──────────────────────────────────────────────────

def get_val_iter(dataset_name: str, n_docs: int, seed: int = 42) -> Iterator[str]:
    """
    Returns a small held-out val split for each dataset.
    Uses a fixed seed so val sets are always identical across runs.
    """
    rng = random.Random(seed)

    if dataset_name == "tinystories":
        # TinyStories has an official validation split
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        for i, ex in enumerate(ds):
            if i >= n_docs: break
            yield ex["text"]

    elif dataset_name in ("babylm_easy", "babylm_hard"):
        allowed = BABYLM_EASY if dataset_name == "babylm_easy" else BABYLM_HARD
        count   = 0
        for text in _iter_babylm(allowed, seed=seed):
            if rng.random() < 0.05:    # ~5% holdout sample
                yield text
                count += 1
                if count >= n_docs: break

    elif dataset_name == "simplewiki":
        from datasets import load_dataset
        ds    = load_dataset("wikimedia/wikipedia", "20231101.simple",
                             split="train", streaming=True)
        count = 0
        for ex in ds:
            if rng.random() < 0.05:
                yield ex["text"]
                count += 1
                if count >= n_docs: break

    elif dataset_name == "fineweb_edu":
        from datasets import load_dataset
        ds    = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                             split="train", streaming=True)
        count = 0
        for ex in ds:
            if ex.get("score", 0) >= 3 and rng.random() < 0.01:
                yield ex["text"]
                count += 1
                if count >= n_docs: break

    elif dataset_name == "legal_documents":
        from src.legal_dataset import iter_legal_documents
        count = 0
        for text in iter_legal_documents(max_samples=None):
            if rng.random() < 0.05:  # ~5% holdout sample
                yield text
                count += 1
                if count >= n_docs: break

    elif dataset_name == "writingprompts":
        from src.writingprompts_dataset import iter_writingprompts
        count = 0
        for text in iter_writingprompts(split="validation"):
            if count >= n_docs: break
            yield text
            count += 1
            
    elif dataset_name == "rocstories":
        # H-1 fix: use hash-based held-out val split, not the training iterator
        yield from _iter_rocstories_val(n_docs)

    elif dataset_name == "simplestories":
        # H-1 fix: use hash-based held-out val split
        yield from _iter_simplestories_val(n_docs)

    elif dataset_name == "children_stories":
        # H-1 fix: use hash-based held-out val split
        yield from _iter_children_stories_val(n_docs)

    else:
        raise ValueError(f"Unknown dataset_name for val: {dataset_name}")


# ─── Token-level chunking ─────────────────────────────────────────────────────

def tokenize_and_chunk(
    text_iter  : Iterator[str],
    tokenizer  : Tokenizer,
    seq_len    : int,
    max_tokens : Optional[int] = None,
    bos_id     : int = 2,
    eos_id     : int = 3,
) -> list[list[int]]:
    """
    Streams text → tokenizes → concatenates with BOS/EOS → chunks into
    non-overlapping windows of (seq_len + 1) tokens (input + target shift).
    """
    buffer      = []
    chunks      = []
    tokens_seen = 0

    for text in text_iter:
        if not text or not text.strip():
            continue
        ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
        buffer.extend(ids)

        while len(buffer) >= seq_len + 1:
            chunks.append(buffer[: seq_len + 1])
            buffer       = buffer[seq_len + 1 :]
            tokens_seen += seq_len + 1
            if max_tokens and tokens_seen >= max_tokens:
                return chunks

    return chunks


# ─── Val set builder (cached) ─────────────────────────────────────────────────

def build_val_set(
    dataset_name : str,
    tokenizer    : Tokenizer,
    seq_len      : int,
    n_docs       : int,
    cache_dir    : str = "cache",
) -> list[list[int]]:
    """Builds and caches a val chunk list. Loads from disk on subsequent calls."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"val_{dataset_name}_seq{seq_len}.pkl")

    if os.path.exists(cache_path):
        print(f"[dataset] Loading cached val '{dataset_name}' from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[dataset] Building val '{dataset_name}' ({n_docs} docs)...")
    chunks = tokenize_and_chunk(
        get_val_iter(dataset_name, n_docs),
        tokenizer,
        seq_len,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[dataset] Saved: {len(chunks)} chunks → {cache_path}")
    return chunks


# ─── PyTorch datasets ─────────────────────────────────────────────────────────

class ChunkedTokenDataset(Dataset):
    """In-memory dataset from a list of token id chunks."""
    def __init__(self, chunks: list[list[int]]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


def _load_replay_chunks(
    replay_from  : list,
    replay_ratio : float,
    n_current    : int,
    target_seq_len : int,
    cache_dir    : str,
) -> list:
    """
    Loads chunks from previous stage caches and samples a fraction of them.
    Automatically handles seq_len mismatches by truncating/padding to target_seq_len.

    replay_ratio is applied to the FINAL dataset size, so:
      n_replay = (replay_ratio / (1 - replay_ratio)) * n_current

    If multiple sources are in replay_from, the replay budget is split
    equally across them.

    Only uses already-cached files — never re-downloads or re-tokenizes.
    If a cache file doesn't exist yet, that source is skipped with a warning.
    """
    # How many replay chunks we need in total
    n_replay_total = int((replay_ratio / max(1 - replay_ratio, 1e-6)) * n_current)
    n_per_source   = max(1, n_replay_total // len(replay_from))

    all_replay = []
    for ds_name in replay_from:
        # Try to load cache at exact target seq_len first
        import glob
        exact_cache = os.path.join(cache_dir, f"train_{ds_name}_seq{target_seq_len}.pkl")
        
        if os.path.exists(exact_cache):
            print(f"[dataset] Loading replay source '{ds_name}' from {exact_cache} (exact match)")
            with open(exact_cache, "rb") as f:
                source_chunks = pickle.load(f)
        else:
            # Fall back to any cached version and truncate/pad to target_seq_len
            pattern   = os.path.join(cache_dir, f"train_{ds_name}_seq*.pkl")
            matches   = sorted(glob.glob(pattern))

            if not matches:
                print(f"[dataset] WARNING: No cache found for replay source '{ds_name}' "
                      f"(pattern: {pattern}) — skipping")
                continue

            # H-9 fix: sort numerically (not lexicographically) so seq512 > seq64
            def _extract_seqlen(p: str) -> int:
                try:
                    return int(p.split('seq')[-1].split('.pkl')[0])
                except (ValueError, IndexError):
                    return 0

            matches_by_seqlen = sorted(matches, key=_extract_seqlen)
            cache_path = matches_by_seqlen[-1]  # now truly the largest seq_len
            source_seq_len = _extract_seqlen(cache_path)
            print(f"[dataset] Loading replay source '{ds_name}' from {cache_path} "
                  f"(source: seq_len={source_seq_len}, target: seq_len={target_seq_len})")
            with open(cache_path, "rb") as f:
                source_chunks = pickle.load(f)

            # Adjust chunks to target_seq_len: truncate long, skip short
            adjusted_chunks = []
            n_truncated = 0
            for chunk in source_chunks:
                if len(chunk) > target_seq_len + 1:
                    # Truncate to target_seq_len + 1 (input + target)
                    adjusted_chunks.append(chunk[: target_seq_len + 1])
                    n_truncated += 1
                elif len(chunk) == target_seq_len + 1:
                    # Perfect match
                    adjusted_chunks.append(chunk)
                # Skip chunks that are too short (don't pad with zeros — corrupts loss)
            
            if n_truncated > 0:
                print(f"[dataset]   → Truncated: {n_truncated:,} chunks to seq_len={target_seq_len}")
            n_skipped = len(source_chunks) - len(adjusted_chunks)
            if n_skipped > 0:
                print(f"[dataset]   → Skipped: {n_skipped:,} chunks (too short)")
            
            source_chunks = adjusted_chunks

        # Random sample without replacement (or with, if cache is smaller than needed)
        rng = random.Random(42)   # fixed seed → reproducible replay selection
        if len(source_chunks) >= n_per_source:
            sampled = rng.sample(source_chunks, n_per_source)
        else:
            # Cache smaller than needed — use all of it and warn
            sampled = source_chunks
            print(f"[dataset] WARNING: '{ds_name}' cache has {len(source_chunks):,} chunks "
                  f"but {n_per_source:,} requested — using all available")

        all_replay.extend(sampled)
        print(f"[dataset] Replay '{ds_name}': {len(sampled):,} chunks sampled")

    return all_replay


class StreamingStageDataset(Dataset):
    """
    Tokenizes + chunks a stage's training data and caches to disk.
    Cache key = dataset_name + seq_len, so each stage gets its own file.

    Supports replay buffer mixing: a fraction of chunks from previous stage
    caches are sampled and interleaved with the current stage chunks.
    This prevents catastrophic forgetting of earlier distributions.
    """
    def __init__(self):
        self.chunks: list[list[int]] = []

    def build(
        self,
        dataset_name  : str,
        tokenizer     : Tokenizer,
        seq_len       : int,
        max_tokens    : int,
        cache_dir     : str = "cache",
        replay_from   : list = None,   # list of dataset_names to replay
        replay_ratio  : float = 0.0,   # fraction of final dataset from replay
    ) -> "StreamingStageDataset":
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"train_{dataset_name}_seq{seq_len}.pkl"
        )

        # ── Build or load current stage chunks ────────────────────────────────
        if os.path.exists(cache_path):
            print(f"[dataset] Loading cached train '{dataset_name}' @ seq={seq_len}")
            with open(cache_path, "rb") as f:
                current_chunks = pickle.load(f)
            print(f"[dataset] Loaded {len(current_chunks):,} chunks")
        else:
            print(f"[dataset] Building train '{dataset_name}' "
                  f"(seq={seq_len}, max_tokens={max_tokens:,})...")
            current_chunks = tokenize_and_chunk(
                get_train_iter(dataset_name),
                tokenizer,
                seq_len,
                max_tokens,
            )
            with open(cache_path, "wb") as f:
                pickle.dump(current_chunks, f)
            print(f"[dataset] Saved {len(current_chunks):,} chunks → {cache_path}")

        # ── Replay buffer mixing ───────────────────────────────────────────────
        if replay_from and replay_ratio > 0.0:
            replay_chunks = _load_replay_chunks(
                replay_from     = replay_from,
                replay_ratio    = replay_ratio,
                n_current       = len(current_chunks),
                target_seq_len  = seq_len,
                cache_dir       = cache_dir,
            )
            self.chunks = current_chunks + replay_chunks
            print(f"[dataset] Mixed: {len(current_chunks):,} current  "
                  f"+ {len(replay_chunks):,} replay  "
                  f"= {len(self.chunks):,} total  "
                  f"({replay_ratio*100:.0f}% replay)")
        else:
            self.chunks = current_chunks

        random.shuffle(self.chunks)
        return self

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 2,
        pin_memory  = True,
    )


# ─── All val sets (called once, used throughout all stages) ───────────────────

# One val set per dataset, each at the seq_len that stage uses.
# All are logged at every eval step regardless of current stage.
# VAL_CONFIGS = {
#     "s0"  : ("tinystories", 256,  5000),
#     "s0b" : ("babylm_easy", 384,  3000),
#     "s0c" : ("babylm_hard", 512,  2000),
#     "s1"  : ("simplewiki",  512,  2000),
#     "s2"  : ("fineweb_edu", 512,  1500),
# }

VAL_CONFIGS = {
    "s0"    : ("tinystories",     256, 5000),
    "s1"    : ("simplewiki",      512, 2000),
    "s2"    : ("fineweb_edu",     512, 1500),
    "roc"   : ("rocstories",      384, 2000),
    "simple": ("simplestories",   384, 2000),
    "child" : ("children_stories",512, 2000),
    "wp"    : ("writingprompts",  384, 2000),
}

def load_all_val_sets(
    tokenizer : Tokenizer,
    cache_dir : str = "cache",
) -> dict:
    """
    Builds (or loads cached) val DataLoaders for all 5 stages.
    Returns dict: {"s0": loader, "s0b": loader, ..., "s2": loader}
    """
    loaders = {}
    for key, (dataset_name, seq_len, n_docs) in VAL_CONFIGS.items():
        chunks = build_val_set(dataset_name, tokenizer, seq_len, n_docs, cache_dir)
        ds     = ChunkedTokenDataset(chunks)
        loaders[key] = DataLoader(
            ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
        )
        print(f"[dataset] val '{key}' ({dataset_name}@{seq_len}): {len(ds)} chunks")
    return loaders
