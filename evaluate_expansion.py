"""
evaluate_expansion.py — Expansion-specific evaluation and comparison.

Evaluates expanded model checkpoints with focus on:
  - Parameter count verification at each stage
  - Cross-dataset performance (TinyStories + WritingPrompts)
  - Forgetting detection (TinyStories val loss vs baseline)
  - Layer-wise gradient norm analysis
  - Stage-over-stage comparison

Usage:
  # Evaluate a single expanded checkpoint
  python evaluate_expansion.py \
    --checkpoint checkpoints/stage_A_best.pt \
    --baseline checkpoints/stagefull_dataset_curriculum_adaptive_best.pt \
    --tokenizer tokenizers/tokenizer_corpus.json

  # Compare all expansion stages
  python evaluate_expansion.py \
    --compare_stages \
    --tokenizer tokenizers/tokenizer_corpus.json
"""

import os
import argparse
import pickle
import glob

import numpy as np
import torch
from tokenizers import Tokenizer

from src.model import SLM, SLMConfig
from evaluate_curriculum import (
    evaluate_on_chunks,
    load_model_from_checkpoint,
    generate_sample,
)


def verify_param_count(model: SLM, expected_range: tuple = None):
    """Print and verify parameter count breakdown."""
    breakdown = model.param_breakdown()
    total = model.num_params()

    print(f"\n  Parameter Breakdown:")
    for k, v in breakdown.items():
        print(f"    {k:<10}: {v/1e6:>8.2f}M")

    if expected_range:
        lo, hi = expected_range
        status = "✓" if lo <= total <= hi else "✗"
        print(f"\n  Expected: {lo/1e6:.0f}M – {hi/1e6:.0f}M")
        print(f"  Actual:   {total/1e6:.1f}M  [{status}]")

    return total


def evaluate_forgetting(
    model: SLM,
    baseline_loss: float,
    chunks: list,
    device: str,
    vocab_size: int,
    max_samples: int = 200,
) -> dict:
    """
    Measure how much the expanded model has forgotten TinyStories.

    Returns dict with forgetting metrics.
    """
    current_loss, current_ppl = evaluate_on_chunks(
        model, chunks, device, vocab_size, max_samples
    )

    forgetting_pct = ((current_loss - baseline_loss) / max(baseline_loss, 1e-6)) * 100
    status = "OK" if forgetting_pct < 5.0 else "WARNING" if forgetting_pct < 10.0 else "CRITICAL"

    return {
        "baseline_loss": baseline_loss,
        "current_loss": current_loss,
        "current_ppl": current_ppl,
        "forgetting_pct": forgetting_pct,
        "status": status,
    }


def compare_stages(args):
    """Compare all available expansion stage checkpoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # Find all stage checkpoints
    stage_patterns = [
        ("Baseline (50M)", "checkpoints/stagefull_dataset_curriculum_adaptive_best.pt"),
        ("Stage A (58M)", "checkpoints/stage_A_best.pt"),
        ("Stage B (71M)", "checkpoints/stage_B_best.pt"),
        ("Stage C (99M)", "checkpoints/stage_C_best.pt"),
    ]

    available = []
    for name, path in stage_patterns:
        if os.path.exists(path):
            available.append((name, path))

    if len(available) < 2:
        print("Need at least 2 checkpoints for comparison.")
        print(f"Found: {[n for n, _ in available]}")
        return

    # Load TinyStories chunks for forgetting evaluation
    ts_chunks = _load_tinystories_chunks(args.cache_dir)

    print(f"\n{'='*70}")
    print(f"  EXPANSION STAGE COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Stage':<20} {'Params':>8} {'Layers':>7} {'d_ff':>6} "
          f"{'Val Loss':>10} {'TS Loss':>10} {'Forget%':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*7} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")

    baseline_ts_loss = None

    for name, path in available:
        model, cfg, info = load_model_from_checkpoint(path, device)
        total_params = model.num_params()

        # Evaluate on TinyStories
        ts_loss, ts_ppl = evaluate_on_chunks(
            model, ts_chunks, device, vocab_size, 200
        )

        if baseline_ts_loss is None:
            baseline_ts_loss = ts_loss
            forget_pct = 0.0
        else:
            forget_pct = ((ts_loss - baseline_ts_loss) / max(baseline_ts_loss, 1e-6)) * 100

        val_loss = info.get("best_val_loss", "?")
        if isinstance(val_loss, float):
            val_str = f"{val_loss:.4f}"
        else:
            val_str = str(val_loss)

        print(f"  {name:<20} {total_params/1e6:>6.1f}M {cfg.n_layers:>7} "
              f"{cfg.d_ff:>6} {val_str:>10} {ts_loss:>10.4f} "
              f"{forget_pct:>+9.1f}%")

        del model

    # Generation comparison
    print(f"\n{'='*70}")
    print(f"  GENERATION COMPARISON")
    print(f"{'='*70}")

    prompts = [
        "Once upon a time",
        "The little girl wanted to",
        "There was a mysterious cave",
    ]

    for name, path in available:
        model, cfg, info = load_model_from_checkpoint(path, device)
        print(f"\n  --- {name} ---")

        for prompt in prompts:
            text = generate_sample(model, tokenizer, prompt, device, max_tokens=80)
            text = text.replace("\n", " ").strip()
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"  Prompt: \"{prompt}\"")
            print(f"  Output: \"{text}\"\n")

        del model

    print(f"{'='*70}\n")


def _load_tinystories_chunks(cache_dir: str) -> list:
    """Load TinyStories chunks from cache."""
    pattern = os.path.join(cache_dir, "train_tinystories_seq*.pkl")
    cache_files = sorted(glob.glob(pattern))
    if not cache_files:
        print(f"WARNING: No TinyStories cache found at {pattern}")
        return []

    with open(cache_files[0], "rb") as f:
        chunks = pickle.load(f)
    print(f"[eval] Loaded {len(chunks):,} TinyStories chunks from {cache_files[0]}")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Evaluate expanded SLM")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to expanded checkpoint")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline (pre-expansion) checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer JSON file")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory for dataset caches")
    parser.add_argument("--compare_stages", action="store_true",
                        help="Compare all available expansion stages")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples per evaluation")
    args = parser.parse_args()

    if args.compare_stages:
        compare_stages(args)
        return

    if not args.checkpoint:
        print("Provide --checkpoint or use --compare_stages")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # ── Load expanded model ──────────────────────────────────────────────────
    print(f"\nLoading expanded checkpoint: {args.checkpoint}")
    model, cfg, info = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Step: {info['step']}, Tokens: {info['tokens_seen']}, "
          f"Val: {info['best_val_loss']}")

    # ── Check expansion metadata ─────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "expansion_meta" in ckpt:
        meta = ckpt["expansion_meta"]
        print(f"\n  Expansion metadata:")
        print(f"    Type: {meta.get('type', '?')}")
        print(f"    Source: {meta.get('source_checkpoint', '?')}")
        if "details" in meta:
            for k, v in meta["details"].items():
                print(f"    {k}: {v}")

    # ── Parameter verification ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PARAMETER VERIFICATION")
    print(f"{'='*70}")
    verify_param_count(model)

    # ── TinyStories forgetting ───────────────────────────────────────────────
    ts_chunks = _load_tinystories_chunks(args.cache_dir)
    if ts_chunks:
        print(f"\n{'='*70}")
        print(f"  TINYSTORIES FORGETTING ANALYSIS")
        print(f"{'='*70}")

        baseline_loss = None
        if args.baseline:
            bl_model, _, bl_info = load_model_from_checkpoint(
                args.baseline, device
            )
            baseline_loss = bl_info.get("best_val_loss", None)
            if baseline_loss:
                bl_loss, _ = evaluate_on_chunks(
                    bl_model, ts_chunks, device, vocab_size, args.max_samples
                )
                baseline_loss = bl_loss
            del bl_model

        if baseline_loss:
            result = evaluate_forgetting(
                model, baseline_loss, ts_chunks, device,
                vocab_size, args.max_samples,
            )
            print(f"  Baseline TS loss: {result['baseline_loss']:.4f}")
            print(f"  Current TS loss:  {result['current_loss']:.4f}")
            print(f"  Current TS PPL:   {result['current_ppl']:.2f}")
            print(f"  Forgetting:       {result['forgetting_pct']:+.1f}%  "
                  f"[{result['status']}]")

    # ── Tier evaluation ──────────────────────────────────────────────────────
    if ts_chunks:
        scores_path = "curriculum_scores.npy"
        n = len(ts_chunks)

        if os.path.exists(scores_path):
            scores = np.load(scores_path)
            n_scores = len(scores)
            story_to_diff = {int(r[0]): r[1] for r in scores}
            chunk_diffs = []
            for i in range(n):
                s = min(int(i * n_scores / max(n, 1)), n_scores - 1)
                chunk_diffs.append(
                    story_to_diff.get(s, scores[s, 1])
                )
            sorted_idx = sorted(range(n), key=lambda i: chunk_diffs[i])
        else:
            sorted_idx = list(range(n))

        tiers = {
            "EASY (0-30%)": [ts_chunks[sorted_idx[i]] for i in range(int(n*0.3))],
            "MEDIUM (30-70%)": [ts_chunks[sorted_idx[i]] for i in range(int(n*0.3), int(n*0.7))],
            "HARD (70-100%)": [ts_chunks[sorted_idx[i]] for i in range(int(n*0.7), n)],
        }

        print(f"\n{'='*70}")
        print(f"  PER-TIER EVALUATION (TinyStories)")
        print(f"{'='*70}")

        for tier_name, tier_chunks in tiers.items():
            loss, ppl = evaluate_on_chunks(
                model, tier_chunks, device, vocab_size, args.max_samples
            )
            print(f"  {tier_name:<25}  loss={loss:.4f}  PPL={ppl:.2f}")

    # ── Sample generation ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SAMPLE GENERATIONS")
    print(f"{'='*70}")

    prompts = [
        "Once upon a time",
        "The little boy was very",
        "She went to the park and",
        "There was a big",
    ]

    for prompt in prompts:
        text = generate_sample(model, tokenizer, prompt, device)
        text = text.replace("\n", " ").strip()
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: \"{text}\"")

    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
