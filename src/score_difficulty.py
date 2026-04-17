"""
score_difficulty.py — One-time difficulty pre-scoring for TinyStories.

Computes a composite difficulty score for each story using:
  - GPT-2 reference model perplexity (primary: 60% weight)
  - Type-Token Ratio / lexical diversity (20% weight)
  - Token length (minor: 20% weight)

Outputs:
  curriculum_scores.npy — shape (N, 5):
    [chunk_index, difficulty, perplexity, ttr, length]

Usage:
  python score_difficulty.py --output curriculum_scores.npy
  python score_difficulty.py --output curriculum_scores.npy --max_stories 1000  # quick test
"""

import os
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm


def compute_perplexity_batch(texts, model, tokenizer, device, max_length=512):
    """Compute perplexity for a batch of texts using true vectorized PyTorch operations."""
    try:
        # Encode entire batch structurally
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Utilize native backend unrecorded modes saving massive memory overhead
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits

        # Shift geometry exactly mapping HuggingFace native architecture mappings
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Unreduced Loss (to isolate per-sequence loss dynamically)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), shift_labels.size(1))

        # Dynamically ignore padded zero-fields
        attention_mask = inputs["attention_mask"][..., 1:].contiguous()
        seq_lengths = attention_mask.sum(dim=1)

        # Average standard length dynamically
        per_seq_loss = (loss * attention_mask).sum(dim=1) / seq_lengths.clamp(min=1)
        
        return [min(p, 10000.0) for p in torch.exp(per_seq_loss).tolist()]

    except Exception as e:
        print(f"[warning] Batch logic collapse: {e}")
        return [500.0] * len(texts)


def compute_ttr(text):
    """Compute Type-Token Ratio (lexical diversity)."""
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)


def compute_token_length(text, tokenizer_enc):
    """Compute token length using the project's tokenizer."""
    return len(tokenizer_enc.encode(text).ids)


def compute_coherence_bonus(text, token_length):
    """
    Compute a coherence bonus for long stories.
    Stories that are long BUT coherent (strong discourse markers)
    should come earlier in the curriculum.
    """
    if token_length < 150:
        return 0.0

    import re
    temporal = r'\b(first|then|next|finally|after|before|when|suddenly)\b'
    causal = r'\b(because|so|therefore|thus|as a result)\b'
    adversative = r'\b(but|however|although|even though|despite)\b'

    text_lower = text.lower()
    marker_count = (
        len(re.findall(temporal, text_lower)) +
        len(re.findall(causal, text_lower)) +
        len(re.findall(adversative, text_lower))
    )

    # Normalize by token length (markers per 100 tokens)
    return (marker_count / max(token_length, 1)) * 100



def main():
    parser = argparse.ArgumentParser(description="Score TinyStories difficulty")
    parser.add_argument("--output", type=str, default="curriculum_scores.npy",
                        help="Output path for scores array")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Limit number of stories (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume scoring from existing output file avoiding redundant compute")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for GPT-2 inference")
    parser.add_argument("--tokenizer_path", type=str,
                        default="tokenizers/tokenizer_corpus.json",
                        help="Path to project tokenizer for length counting")
    parser.add_argument(
        "--weights", type=str, default="0.55,0.20,0.15,0.10",
        help="Comma-separated weights for perplexity, TTR, length, coherence_bonus"
    )
    args = parser.parse_args()

    w_ppl, w_ttr, w_len, w_coh = [float(x) for x in args.weights.split(",")]
    print(f"[score] Weights: perplexity={w_ppl}, TTR={w_ttr}, length={w_len}, coherence={w_coh}")

    # ── Device setup ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[score] Device: {device}")

    # ── Load reference model (GPT-2 small) ────────────────────────────────────
    print("[score] Loading GPT-2 reference model...")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    ref_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT-2 tokenizer has no pad token by default
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
    print("[score] GPT-2 loaded.")

    # ── Load project tokenizer (for length counting) ──────────────────────────
    from tokenizers import Tokenizer as HFTokenizer

    proj_tokenizer = HFTokenizer.from_file(args.tokenizer_path)
    print(f"[score] Project tokenizer loaded: vocab={proj_tokenizer.get_vocab_size()}")

    # ── Load TinyStories dataset ──────────────────────────────────────────────
    print("[score] Loading TinyStories dataset...")
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # ── Score all stories ─────────────────────────────────────────────────────
    print("[score] Scoring stories...")
    scores = []
    start_index = 0
    if args.resume and os.path.exists(args.output):
        print(f"[score] Resuming from existing {args.output}...")
        existing_scores = np.load(args.output)
        start_index = int(np.max(existing_scores[:, 0])) + 1
        scores = existing_scores.tolist()
        print(f"[score] Found {len(scores)} existing scores. Fast-forwarding dataset to index {start_index}...")

    text_batch = []
    index_batch = []

    for i, example in enumerate(tqdm(ds, desc="Scoring", total=args.max_stories or 2119719)):
        if args.max_stories and i >= args.max_stories:
            break
            
        if i < start_index:
            continue

        text = example["text"]
        if not text or not text.strip():
            continue

        text_batch.append(text)
        index_batch.append(i)

        # Process in batches for GPU efficiency
        if len(text_batch) >= args.batch_size:
            # Compute perplexities for the batch
            ppls = compute_perplexity_batch(
                text_batch, ref_model, ref_tokenizer, device
            )

            for j, (txt, ppl) in enumerate(zip(text_batch, ppls)):
                ttr = compute_ttr(txt)
                length = compute_token_length(txt, proj_tokenizer)
                coh_bonus = compute_coherence_bonus(txt, length)

                # Composite difficulty score
                difficulty = (
                    w_ppl * ppl
                    + w_ttr * ttr * 100
                    + w_len * (length / 100)
                    - w_coh * coh_bonus
                )

                scores.append((
                    index_batch[j],  # story index in dataset
                    difficulty,      # composite score
                    ppl,             # raw perplexity
                    ttr,             # type-token ratio
                    length,          # token length
                ))

            text_batch = []
            index_batch = []

    # Process remaining batch
    if text_batch:
        ppls = compute_perplexity_batch(
            text_batch, ref_model, ref_tokenizer, device
        )
        for j, (txt, ppl) in enumerate(zip(text_batch, ppls)):
            ttr = compute_ttr(txt)
            length = compute_token_length(txt, proj_tokenizer)
            coh_bonus = compute_coherence_bonus(txt, length)
            difficulty = (
                w_ppl * ppl + w_ttr * ttr * 100 + w_len * (length / 100) - w_coh * coh_bonus
            )
            scores.append((index_batch[j], difficulty, ppl, ttr, length))

    # ── Sort by difficulty (ascending) and save ───────────────────────────────
    scores_array = np.array(scores, dtype=np.float64)
    sorted_indices = np.argsort(scores_array[:, 1])  # sort by difficulty column
    scores_sorted = scores_array[sorted_indices]

    np.save(args.output, scores_sorted)
    print(f"\n[score] Saved {len(scores_sorted)} scores → {args.output}")

    # ── Print statistics ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DIFFICULTY SCORING STATISTICS")
    print(f"{'='*60}")
    print(f"  Total stories scored: {len(scores_sorted):,}")
    print(f"  Difficulty range: {scores_sorted[0, 1]:.2f} — {scores_sorted[-1, 1]:.2f}")
    print(f"  Difficulty median: {np.median(scores_sorted[:, 1]):.2f}")
    print(f"  Difficulty mean:   {np.mean(scores_sorted[:, 1]):.2f}")
    print(f"  Difficulty std:    {np.std(scores_sorted[:, 1]):.2f}")

    print(f"\n  Perplexity range: {scores_sorted[:, 2].min():.2f} — {scores_sorted[:, 2].max():.2f}")
    print(f"  Perplexity mean:  {np.mean(scores_sorted[:, 2]):.2f}")

    print(f"\n  TTR range: {scores_sorted[:, 3].min():.3f} — {scores_sorted[:, 3].max():.3f}")
    print(f"  TTR mean:  {np.mean(scores_sorted[:, 3]):.3f}")

    print(f"\n  Length range: {scores_sorted[:, 4].min():.0f} — {scores_sorted[:, 4].max():.0f}")
    print(f"  Length mean:  {np.mean(scores_sorted[:, 4]):.0f}")

    # ── Correlation analysis ──────────────────────────────────────────────────
    ppl_vals = scores_sorted[:, 2]
    ttr_vals = scores_sorted[:, 3]
    len_vals = scores_sorted[:, 4]
    diff_vals = scores_sorted[:, 1]

    corr_len_diff = np.corrcoef(len_vals, diff_vals)[0, 1]
    corr_ppl_diff = np.corrcoef(ppl_vals, diff_vals)[0, 1]
    corr_ttr_diff = np.corrcoef(ttr_vals, diff_vals)[0, 1]
    corr_len_ppl = np.corrcoef(len_vals, ppl_vals)[0, 1]

    print(f"\n  Correlations:")
    print(f"    length  ↔ difficulty:  {corr_len_diff:.3f}")
    print(f"    PPL     ↔ difficulty:  {corr_ppl_diff:.3f}")
    print(f"    TTR     ↔ difficulty:  {corr_ttr_diff:.3f}")
    print(f"    length  ↔ perplexity: {corr_len_ppl:.3f}")

    if corr_len_ppl < 0.7:
        print(f"\n  ⚡ length↔perplexity correlation ({corr_len_ppl:.3f}) < 0.7")
        print(f"     → Perplexity adds significant signal beyond length alone!")
    else:
        print(f"\n  ℹ  length↔perplexity are highly correlated ({corr_len_ppl:.3f})")

    # ── Show sample stories at different difficulty levels ─────────────────────
    print(f"\n{'='*60}")
    print("SAMPLE STORIES BY DIFFICULTY TIER")
    print(f"{'='*60}")

    n = len(scores_sorted)
    tiers = {
        "EASY (5th percentile)": int(n * 0.05),
        "MEDIUM (50th percentile)": int(n * 0.50),
        "HARD (95th percentile)": int(n * 0.95),
    }

    # Re-load a few examples for display
    # ds_display = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    # target_indices = {int(scores_sorted[idx, 0]) for idx in tiers.values()}
    story_cache = {}

    # for i, ex in enumerate(ds_display):
    #     if i in target_indices:
    #         story_cache[i] = ex["text"][:200]
    #     if len(story_cache) == len(target_indices):
    #         break

    for label, tier_idx in tiers.items():
        row = scores_sorted[tier_idx]
        orig_idx = int(row[0])
        # text_preview = story_cache.get(orig_idx, "[Text unavailable]") + "..."
        text_preview = "[Text hidden to prevent streaming HTTP lock bypass in subset optimization]..."
        
        print(f"\n  {label}:")
        print(f"    Index: {orig_idx}")
        print(f"    Difficulty: {scores_sorted[tier_idx, 1]:.2f}")
        print(f"    PPL: {scores_sorted[tier_idx, 2]:.2f}")
        print(f"    TTR: {scores_sorted[tier_idx, 3]:.3f}")
        print(f"    Length: {scores_sorted[tier_idx, 4]:.0f} tokens")
        print(f"    Text: \"{text_preview}\"")

    print(f"\n{'='*60}")
    print("[score] Done.")


if __name__ == "__main__":
    main()
