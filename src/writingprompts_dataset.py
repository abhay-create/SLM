"""
writingprompts_dataset.py — WritingPrompts data pipeline for model expansion.

Provides:
  - Dataset streaming from HuggingFace (euclaise/writingprompts)
  - Difficulty scoring (same composite as TinyStories: PPL + TTR + length + coherence)
  - Easy/medium/hard bucket splitting
  - Curriculum-compatible data iteration

Usage:
  # Stream training data
  from src.writingprompts_dataset import iter_writingprompts
  for text in iter_writingprompts(split="train"):
      ...

  # Score difficulty (one-time)
  python -m src.writingprompts_dataset --score --output writingprompts_scores.npy

  # Build bucketed cache
  python -m src.writingprompts_dataset --build_cache \
    --tokenizer tokenizers/tokenizer_corpus.json \
    --scores writingprompts_scores.npy
"""

import os
import argparse
import numpy as np
from typing import Iterator, Optional


# ─── Dataset name on HuggingFace ──────────────────────────────────────────────
WP_DATASET = "euclaise/writingprompts"


# ─── Text iterators ──────────────────────────────────────────────────────────

def iter_writingprompts(
    split: str = "train",
    bucket: str = None,
    scores_path: str = None,
    bucket_ranges: dict = None,
) -> Iterator[str]:
    """
    Stream WritingPrompts text from HuggingFace.

    If bucket and scores_path are provided, only yields stories in that
    difficulty tier.

    Args:
        split: "train", "validation", or "test"
        bucket: Optional — "easy", "medium", or "hard"
        scores_path: Path to pre-computed difficulty scores (required for bucket filtering)
        bucket_ranges: Optional dict of {bucket: (start_pct, end_pct)} overrides

    Yields:
        Combined prompt + story text strings.
    """
    from datasets import load_dataset

    ds = load_dataset(WP_DATASET, split=split, streaming=True)

    # If bucket filtering is requested, load scores
    allowed_indices = None
    if bucket and scores_path and os.path.exists(scores_path):
        allowed_indices = _get_bucket_indices(scores_path, bucket, bucket_ranges)
        print(f"[writingprompts] Bucket '{bucket}': {len(allowed_indices):,} stories")

    for i, example in enumerate(ds):
        # Filter by bucket if needed
        if allowed_indices is not None and i not in allowed_indices:
            continue

        # Combine prompt and story
        prompt = example.get("prompt", example.get("title", ""))
        story = example.get("story", example.get("text", ""))

        if not story or not story.strip():
            continue

        # Clean up common WritingPrompts artifacts
        text = _clean_wp_text(prompt, story)
        if text:
            yield text


def _clean_wp_text(prompt: str, story: str) -> str:
    """Clean WritingPrompts text: remove [WP] tags, excess whitespace."""
    # Remove common WritingPrompts tags
    prompt = prompt.strip()
    for tag in ["[WP]", "[EU]", "[CW]", "[TT]", "[PI]", "[OT]", "[RF]", "[MP]"]:
        prompt = prompt.replace(tag, "").strip()

    story = story.strip()

    # Skip very short stories (likely metadata or broken entries)
    if len(story.split()) < 20:
        return ""

    # Combine prompt and story with a separator
    if prompt:
        return f"{prompt}\n\n{story}"
    return story


def _get_bucket_indices(
    scores_path: str,
    bucket: str,
    bucket_ranges: dict = None,
) -> set:
    """
    Get story indices belonging to a difficulty bucket.

    Default bucket ranges (by percentile of sorted difficulty):
      easy:   0% - 30%
      medium: 30% - 70%
      hard:   70% - 100%
    """
    if bucket_ranges is None:
        bucket_ranges = {
            "easy": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "hard": (0.7, 1.0),
        }

    scores = np.load(scores_path)
    n = len(scores)
    start_pct, end_pct = bucket_ranges[bucket]
    start_idx = int(n * start_pct)
    end_idx = int(n * end_pct)

    # scores[:, 0] contains original story indices
    # scores are already sorted by difficulty (ascending)
    return set(int(scores[i, 0]) for i in range(start_idx, end_idx))


# ─── Difficulty Scoring ───────────────────────────────────────────────────────

def score_writingprompts(
    output_path: str,
    max_stories: int = None,
    batch_size: int = 16,
    tokenizer_path: str = "tokenizers/tokenizer_corpus.json",
    weights: tuple = (0.55, 0.20, 0.15, 0.10),
):
    """
    Score WritingPrompts difficulty using the same composite as TinyStories.

    Produces a .npy file with shape (N, 5):
      [story_index, difficulty, perplexity, ttr, length]

    Uses the same scoring infrastructure as src/score_difficulty.py.
    """
    import torch
    from tqdm.auto import tqdm

    w_ppl, w_ttr, w_len, w_coh = weights
    print(f"[score_wp] Weights: PPL={w_ppl}, TTR={w_ttr}, len={w_len}, coh={w_coh}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[score_wp] Device: {device}")

    # Import scoring functions from existing infrastructure
    from src.score_difficulty import (
        compute_perplexity_batch,
        compute_ttr,
        compute_token_length,
        compute_coherence_bonus,
    )

    # Load reference model
    print("[score_wp] Loading GPT-2 reference model...")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    ref_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
    print("[score_wp] GPT-2 loaded.")

    # Project tokenizer for length counting
    from tokenizers import Tokenizer as HFTokenizer
    proj_tokenizer = HFTokenizer.from_file(tokenizer_path)

    # Stream and score
    from datasets import load_dataset
    ds = load_dataset(WP_DATASET, split="train", streaming=True)

    scores = []
    text_batch = []
    index_batch = []

    # Resume support
    start_index = 0
    if os.path.exists(output_path):
        existing = np.load(output_path)
        start_index = int(np.max(existing[:, 0])) + 1
        scores = existing.tolist()
        print(f"[score_wp] Resuming from index {start_index} "
              f"({len(scores)} existing scores)")

    total = max_stories or 300000
    for i, example in enumerate(tqdm(ds, desc="Scoring WP", total=total)):
        if max_stories and i >= max_stories:
            break
        if i < start_index:
            continue

        prompt = example.get("prompt", example.get("title", ""))
        story = example.get("story", example.get("text", ""))
        text = _clean_wp_text(prompt, story)
        if not text:
            continue

        text_batch.append(text)
        index_batch.append(i)

        if len(text_batch) >= batch_size:
            ppls = compute_perplexity_batch(
                text_batch, ref_model, ref_tokenizer, device
            )
            for j, (txt, ppl) in enumerate(zip(text_batch, ppls)):
                ttr = compute_ttr(txt)
                length = compute_token_length(txt, proj_tokenizer)
                coh_bonus = compute_coherence_bonus(txt, length)
                difficulty = (
                    w_ppl * ppl
                    + w_ttr * ttr * 100
                    + w_len * (length / 100)
                    - w_coh * coh_bonus
                )
                scores.append((index_batch[j], difficulty, ppl, ttr, length))
            text_batch = []
            index_batch = []

            # Periodic save
            if len(scores) % 5000 < batch_size:
                _save_scores(scores, output_path)

    # Process remainder
    if text_batch:
        ppls = compute_perplexity_batch(
            text_batch, ref_model, ref_tokenizer, device
        )
        for j, (txt, ppl) in enumerate(zip(text_batch, ppls)):
            ttr = compute_ttr(txt)
            length = compute_token_length(txt, proj_tokenizer)
            coh_bonus = compute_coherence_bonus(txt, length)
            difficulty = (
                w_ppl * ppl + w_ttr * ttr * 100
                + w_len * (length / 100) - w_coh * coh_bonus
            )
            scores.append((index_batch[j], difficulty, ppl, ttr, length))

    _save_scores(scores, output_path)
    print(f"\n[score_wp] Done. {len(scores):,} stories scored → {output_path}")


def _save_scores(scores: list, output_path: str):
    """Sort by difficulty and save as .npy."""
    arr = np.array(scores, dtype=np.float64)
    sorted_idx = np.argsort(arr[:, 1])
    np.save(output_path, arr[sorted_idx])


# ─── Bucket Statistics ────────────────────────────────────────────────────────

def print_bucket_stats(scores_path: str):
    """Print difficulty distribution statistics for WritingPrompts."""
    scores = np.load(scores_path)
    n = len(scores)

    print(f"\n{'='*60}")
    print(f"WritingPrompts Difficulty Statistics")
    print(f"{'='*60}")
    print(f"  Total stories: {n:,}")
    print(f"  Difficulty: {scores[0, 1]:.2f} — {scores[-1, 1]:.2f}")
    print(f"  Mean: {np.mean(scores[:, 1]):.2f}  Std: {np.std(scores[:, 1]):.2f}")

    tiers = {
        "EASY (0-30%)":   (0.0, 0.3),
        "MEDIUM (30-70%)": (0.3, 0.7),
        "HARD (70-100%)":  (0.7, 1.0),
    }
    for label, (lo, hi) in tiers.items():
        start, end = int(n * lo), int(n * hi)
        tier_scores = scores[start:end]
        print(f"\n  {label}:")
        print(f"    Count: {len(tier_scores):,}")
        print(f"    Difficulty: {tier_scores[0, 1]:.2f} — {tier_scores[-1, 1]:.2f}")
        print(f"    Avg PPL: {np.mean(tier_scores[:, 2]):.2f}")
        print(f"    Avg TTR: {np.mean(tier_scores[:, 3]):.3f}")
        print(f"    Avg Length: {np.mean(tier_scores[:, 4]):.0f} tokens")

    print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WritingPrompts data pipeline"
    )
    parser.add_argument("--score", action="store_true",
                        help="Score difficulty for WritingPrompts")
    parser.add_argument("--stats", action="store_true",
                        help="Print bucket statistics")
    parser.add_argument("--output", type=str,
                        default="writingprompts_scores.npy",
                        help="Output path for scores")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Limit stories to score")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for scoring")
    parser.add_argument("--tokenizer", type=str,
                        default="tokenizers/tokenizer_corpus.json",
                        help="Path to project tokenizer")
    args = parser.parse_args()

    if args.score:
        score_writingprompts(
            output_path=args.output,
            max_stories=args.max_stories,
            batch_size=args.batch_size,
            tokenizer_path=args.tokenizer,
        )
    if args.stats:
        print_bucket_stats(args.output)
