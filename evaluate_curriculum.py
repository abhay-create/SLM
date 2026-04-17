"""
evaluate_curriculum.py — Per-tier evaluation and ablation comparison.

Evaluates a trained SLM checkpoint on easy/medium/hard story tiers separately,
and generates sample text at each tier to qualitatively assess quality.

Usage:
  # Evaluate a checkpoint
  python evaluate_curriculum.py \\
    --checkpoint checkpoints/stage0_curriculum_adaptive_best.pt \\
    --tokenizer tokenizers/tokenizer_corpus.json

  # Compare two checkpoints (e.g., baseline vs curriculum)
  python evaluate_curriculum.py \\
    --checkpoint checkpoints/stage0_curriculum_adaptive_best.pt \\
    --baseline checkpoints/stage0_best.pt \\
    --tokenizer tokenizers/tokenizer_corpus.json
"""

import os
import argparse
import numpy as np
import torch
from tokenizers import Tokenizer

from src.model import SLM, SLMConfig


@torch.no_grad()
def evaluate_on_chunks(model, chunks, device, vocab_size, max_samples=200):
    """Evaluate model on a list of token chunks. Returns average loss and PPL."""
    model.eval()
    losses = []

    indices = (
        np.random.choice(len(chunks), min(max_samples, len(chunks)), replace=False)
        if len(chunks) > max_samples
        else range(len(chunks))
    )

    for idx in indices:
        chunk = torch.tensor(chunks[idx], dtype=torch.long).unsqueeze(0).to(device)
        x = chunk[:, :-1].clamp(0, vocab_size - 1)
        y = chunk[:, 1:]
        y = torch.where(y == -1, y, y.clamp(0, vocab_size - 1))

        if x.shape[1] > model.cfg.ctx_len:
            x = x[:, :model.cfg.ctx_len]
            y = y[:, :model.cfg.ctx_len]

        _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = sum(losses) / max(len(losses), 1)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


def load_model_from_checkpoint(ckpt_path, device):
    """Load SLM model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    info = {
        "step": ckpt.get("step", "?"),
        "tokens_seen": ckpt.get("tokens_seen", "?"),
        "best_val_loss": ckpt.get("best_val_loss", "?"),
    }
    return model, cfg, info


def generate_sample(model, tokenizer, prompt, device, max_tokens=100,
                    temperature=0.7, top_k=30):
    """Generate text from a prompt."""
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    output = model.generate(
        idx, max_new=max_tokens, temperature=temperature, top_k=top_k,
        use_cache=True,
    )
    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Evaluate curriculum-trained SLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Optional baseline checkpoint for comparison")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer JSON file")
    parser.add_argument("--scores_path", type=str, default="curriculum_scores.npy",
                        help="Path to difficulty scores file")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory for dataset caches")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples per tier for evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, cfg, info = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Step: {info['step']}, Tokens: {info['tokens_seen']}, "
          f"Val: {info['best_val_loss']}")

    baseline_model = None
    if args.baseline:
        print(f"\nLoading baseline: {args.baseline}")
        baseline_model, _, baseline_info = load_model_from_checkpoint(
            args.baseline, device
        )
        print(f"  Step: {baseline_info['step']}, "
              f"Tokens: {baseline_info['tokens_seen']}, "
              f"Val: {baseline_info['best_val_loss']}")

    # ── Load chunks for tier evaluation ───────────────────────────────────────
    import pickle
    import glob

    # Find the TinyStories cache
    pattern = os.path.join(args.cache_dir, "train_tinystories_seq*.pkl")
    cache_files = sorted(glob.glob(pattern))
    if not cache_files:
        print(f"ERROR: No TinyStories cache found at {pattern}")
        print("Run training first to build the cache.")
        return

    cache_path = cache_files[0]  # Use first available
    print(f"\nLoading chunks from: {cache_path}")
    with open(cache_path, "rb") as f:
        all_chunks = pickle.load(f)
    print(f"  Total chunks: {len(all_chunks):,}")

    # ── Load difficulty scores and split into tiers ───────────────────────────
    n = len(all_chunks)
    if os.path.exists(args.scores_path):
        scores = np.load(args.scores_path)
        print(f"  Loaded {len(scores):,} difficulty scores")

        # Map scores to chunk order (same logic as curriculum_dataset)
        n_scores = len(scores)
        story_to_difficulty = {}
        for row in scores:
            story_to_difficulty[int(row[0])] = row[1]

        chunk_difficulties = []
        for i in range(n):
            approx_story = int(i * n_scores / max(n, 1))
            approx_story = min(approx_story, n_scores - 1)
            if approx_story in story_to_difficulty:
                chunk_difficulties.append(story_to_difficulty[approx_story])
            else:
                chunk_difficulties.append(scores[approx_story, 1])

        sorted_idx = sorted(range(n), key=lambda i: chunk_difficulties[i])
    else:
        print(f"  No scores file found — using sequential order for tiers")
        sorted_idx = list(range(n))

    # Split into tiers
    tiers = {
        "EASY (bottom 30%)": [all_chunks[sorted_idx[i]] for i in range(int(n * 0.3))],
        "MEDIUM (30-70%)": [
            all_chunks[sorted_idx[i]]
            for i in range(int(n * 0.3), int(n * 0.7))
        ],
        "HARD (top 30%)": [
            all_chunks[sorted_idx[i]]
            for i in range(int(n * 0.7), n)
        ],
        "ALL": all_chunks,
    }

    # ── Evaluate per tier ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PER-TIER EVALUATION")
    print(f"{'='*70}")

    results = {}
    for tier_name, tier_chunks in tiers.items():
        loss, ppl = evaluate_on_chunks(
            model, tier_chunks, device, vocab_size, args.max_samples,
        )
        results[tier_name] = {"loss": loss, "ppl": ppl}
        print(f"  {tier_name:<25}  loss={loss:.4f}  PPL={ppl:.2f}")

    # ── Baseline comparison ───────────────────────────────────────────────────
    if baseline_model:
        print(f"\n{'='*70}")
        print(f"  BASELINE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Tier':<25}  {'Curriculum':>12}  {'Baseline':>12}  {'Delta':>10}")
        print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}")

        for tier_name, tier_chunks in tiers.items():
            bl_loss, bl_ppl = evaluate_on_chunks(
                baseline_model, tier_chunks, device, vocab_size, args.max_samples,
            )
            curr = results[tier_name]
            delta_loss = curr["loss"] - bl_loss
            delta_pct = (delta_loss / bl_loss) * 100 if bl_loss > 0 else 0

            sign = "↓" if delta_loss < 0 else "↑"
            print(
                f"  {tier_name:<25}  "
                f"{curr['loss']:>10.4f}  "
                f"{bl_loss:>10.4f}  "
                f"{sign} {abs(delta_pct):>6.2f}%"
            )

    # ── Sample generation ─────────────────────────────────────────────────────
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
        print(f"\n  Prompt: \"{prompt}\"")
        text = generate_sample(model, tokenizer, prompt, device)
        # Clean up and truncate
        text = text.replace("\n", " ").strip()
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"  Output: \"{text}\"")

        if baseline_model:
            bl_text = generate_sample(baseline_model, tokenizer, prompt, device)
            bl_text = bl_text.replace("\n", " ").strip()
            if len(bl_text) > 300:
                bl_text = bl_text[:300] + "..."
            print(f"  Baseline: \"{bl_text}\"")

    # ── Curriculum trajectory (if available) ──────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "curriculum_state" in ckpt:
        state = ckpt["curriculum_state"]
        print(f"\n{'='*70}")
        print(f"  CURRICULUM TRAJECTORY")
        print(f"{'='*70}")
        print(f"  Final fraction: {state['current_fraction']*100:.1f}%")
        print(f"  Val loss history length: {len(state['val_loss_history'])}")
        if state.get("fraction_history"):
            fh = state["fraction_history"]
            print(f"  Fraction history: {fh[0]*100:.0f}% → {fh[-1]*100:.1f}% "
                  f"({len(fh)} updates)")

    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
