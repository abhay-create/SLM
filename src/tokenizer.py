"""
tokenizer.py
Trains two BPE tokenizers on a sample of all stage data combined:
  - Tokenizer A : corpus-derived vocab (natural cutoff, ~32-40k)
  - Tokenizer B : fixed 50k vocab

Uses HuggingFace tokenizers library (fast, battle-tested BPE trainer).

Usage:
  python tokenizer.py --output_dir tokenizers/ --sample_size 5000000
"""

import os
import argparse
import random
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSequence


# ─── Text iterators per data source ───────────────────────────────────────────

def iter_tinystories(split: str = "train", max_docs: int = None) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if max_docs and i >= max_docs: break
        yield ex["text"]


def iter_simplewiki(max_docs: int = None) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if max_docs and i >= max_docs: break
        yield ex["text"]


def iter_babylm(max_docs: int = None) -> Iterator[str]:
    """BabyLM 100M word split — text files streamed line-by-line."""
    # try:
    #     from datasets import load_dataset
    #     ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", streaming=True)
    #     # fallback: BabyLM via text files if HF version unavailable
    # except Exception:
    #     pass
    try:
        from datasets import load_dataset
        ds = load_dataset("BabyLM-community/babylm-eng", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if max_docs and i >= max_docs: break
            text = ex.get("text", "") or ex.get("sentence", "")
            if text: yield text
    except Exception as e:
        print(f"[tokenizer] BabyLM load warning: {e} — skipping BabyLM")


def iter_fineweb(max_docs: int = None) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for i, ex in enumerate(ds):
        if max_docs and i >= max_docs: break
        if ex.get("score", 0) >= 3:
            yield ex["text"]


def combined_iterator(sample_size: int) -> Iterator[str]:
    """
    Interleave all four sources for balanced tokenizer training.
    sample_size = total target number of text examples across all sources.
    """
    per_source = sample_size // 4
    print(f"[tokenizer] Sampling {per_source:,} docs from each source ({sample_size:,} total)")

    sources = [
        iter_tinystories(max_docs=per_source),
        iter_simplewiki(max_docs=per_source),
        iter_babylm(max_docs=per_source),
        iter_fineweb(max_docs=per_source),
    ]

    # Round-robin interleave
    iters = [iter(s) for s in sources]
    active = list(range(len(iters)))
    while active:
        random.shuffle(active)
        still_active = []
        for idx in active:
            try:
                yield next(iters[idx])
                still_active.append(idx)
            except StopIteration:
                pass
        active = still_active


# ─── Build + train tokenizer ──────────────────────────────────────────────────

def build_tokenizer(vocab_size: int) -> Tokenizer:
    """Returns an untrained BPE tokenizer with standard settings."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder       = ByteLevelDecoder()
    return tokenizer


def train_tokenizer(
    vocab_size  : int,
    sample_size : int,
    output_path : str,
    name        : str,
) -> Tokenizer:
    print(f"\n[tokenizer] Training {name} | vocab_size={vocab_size:,}")
    tokenizer = build_tokenizer(vocab_size)
    trainer   = BpeTrainer(
        vocab_size        = vocab_size,
        min_frequency     = 2,
        special_tokens    = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress     = True,
    )

    # Collect texts — stream directly into trainer
    def text_gen():
        for text in combined_iterator(sample_size):
            yield text

    tokenizer.train_from_iterator(text_gen(), trainer=trainer)
    tokenizer.save(output_path)
    print(f"[tokenizer] Saved {name} → {output_path}")
    print(f"[tokenizer] Actual vocab size: {tokenizer.get_vocab_size():,}")
    return tokenizer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  type=str, default="tokenizers")
    parser.add_argument("--sample_size", type=int, default=5_000_000,
                        help="Total number of text samples for training")
    parser.add_argument("--which", type=str, default="both",
                        choices=["corpus", "fixed", "both"],
                        help="corpus=natural cutoff, fixed=50k, both=train both")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.which in ("corpus", "both"):
        # Tokenizer A: let trainer find natural cutoff (~32-40k)
        train_tokenizer(
            vocab_size  = 40000,          # generous upper bound; actual will be <= this
            sample_size = args.sample_size,
            output_path = os.path.join(args.output_dir, "tokenizer_corpus.json"),
            name        = "TokenizerA (corpus)",
        )

    if args.which in ("fixed", "both"):
        # Tokenizer B: fixed 50k
        train_tokenizer(
            vocab_size  = 50000,
            sample_size = args.sample_size,
            output_path = os.path.join(args.output_dir, "tokenizer_50k.json"),
            name        = "TokenizerB (50k)",
        )

    print("\n[tokenizer] Done. Files saved in:", args.output_dir)


if __name__ == "__main__":
    main()
