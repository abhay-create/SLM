"""
Deep comparative analysis of TinyStories vs WritingPrompts datasets.
Examines: token lengths, vocabulary usage, difficulty distributions,
tokenizer coverage, and what the model actually sees during training.
"""
import os, sys, pickle, json, random
import numpy as np
from collections import Counter

# ── 1. Load cached training chunks ──────────────────────────────────────────
print("=" * 70)
print("  DATASET COMPARISON: TinyStories vs WritingPrompts")
print("=" * 70)

# Find all cached chunks
ts_caches = sorted([f for f in os.listdir("cache") if f.startswith("train_tinystories")])
wp_caches = sorted([f for f in os.listdir("cache") if f.startswith("train_writingprompts")])

print(f"\nTinyStories caches: {ts_caches}")
print(f"WritingPrompts caches: {wp_caches}")

# Load TinyStories (seq256)
ts_chunks = []
for f in ts_caches:
    if "seq256" in f:
        with open(f"cache/{f}", "rb") as fh:
            ts_chunks = pickle.load(fh)
        print(f"  Loaded {len(ts_chunks):,} TS chunks from {f}")
        break

# Load WritingPrompts (seq384)
wp_chunks = []
for f in wp_caches:
    if "seq384" in f:
        with open(f"cache/{f}", "rb") as fh:
            wp_chunks = pickle.load(fh)
        print(f"  Loaded {len(wp_chunks):,} WP chunks from {f}")
        break

if not ts_chunks or not wp_chunks:
    print("ERROR: Could not load both datasets")
    sys.exit(1)

# ── 2. Basic statistics ─────────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  BASIC STATISTICS")
print(f"{'─' * 70}")

def chunk_stats(chunks, name):
    lengths = [len(c) for c in chunks]
    # Count unique tokens
    all_tokens = Counter()
    sample = random.sample(chunks, min(50000, len(chunks)))
    for c in sample:
        all_tokens.update(c)
    
    # Token frequency analysis
    total_tokens_sampled = sum(all_tokens.values())
    unique_tokens = len(all_tokens)
    top_20 = all_tokens.most_common(20)
    bottom_coverage = sum(v for _, v in all_tokens.most_common()[-1000:]) / total_tokens_sampled * 100
    top_coverage = sum(v for _, v in all_tokens.most_common(1000)) / total_tokens_sampled * 100
    
    print(f"\n  {name}:")
    print(f"    Total chunks:       {len(chunks):>12,}")
    print(f"    Chunk length:       {lengths[0]:>12} tokens (fixed)")
    print(f"    Total tokens:       {len(chunks) * lengths[0] / 1e6:>12.1f}M")
    print(f"    Unique tokens used: {unique_tokens:>12,} (from {len(sample):,} sample chunks)")
    print(f"    Top 1000 tokens:    {top_coverage:>11.1f}% of all tokens")
    print(f"    Bottom 1000 tokens: {bottom_coverage:>11.1f}% of all tokens")
    
    return all_tokens, lengths

ts_tokens, ts_lengths = chunk_stats(ts_chunks, "TinyStories (seq=256)")
wp_tokens, wp_lengths = chunk_stats(wp_chunks, "WritingPrompts (seq=384)")

# ── 3. Vocabulary overlap ───────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  VOCABULARY OVERLAP")
print(f"{'─' * 70}")

ts_vocab = set(ts_tokens.keys())
wp_vocab = set(wp_tokens.keys())
shared = ts_vocab & wp_vocab
ts_only = ts_vocab - wp_vocab
wp_only = wp_vocab - ts_vocab

print(f"  TS unique token IDs:  {len(ts_vocab):,}")
print(f"  WP unique token IDs: {len(wp_vocab):,}")
print(f"  Shared:              {len(shared):,}  ({len(shared)/max(len(ts_vocab|wp_vocab),1)*100:.1f}%)")
print(f"  TS-only:             {len(ts_only):,}")
print(f"  WP-only:             {len(wp_only):,}")

# ── 4. Token distribution comparison ────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  TOKEN FREQUENCY DISTRIBUTION")
print(f"{'─' * 70}")

# Load tokenizer for decoding
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizers/tokenizer_corpus.json")

print(f"\n  Top 15 tokens in TinyStories:")
for tok_id, count in ts_tokens.most_common(15):
    try:
        decoded = tokenizer.decode([tok_id]).replace('\n', '\\n')
    except:
        decoded = f"<id:{tok_id}>"
    pct = count / sum(ts_tokens.values()) * 100
    print(f"    [{tok_id:>5}] '{decoded:>15}' : {count:>8,}  ({pct:.2f}%)")

print(f"\n  Top 15 tokens in WritingPrompts:")
for tok_id, count in wp_tokens.most_common(15):
    try:
        decoded = tokenizer.decode([tok_id]).replace('\n', '\\n')
    except:
        decoded = f"<id:{tok_id}>"
    pct = count / sum(wp_tokens.values()) * 100
    print(f"    [{tok_id:>5}] '{decoded:>15}' : {count:>8,}  ({pct:.2f}%)")

# ── 5. WP-only tokens (what TS model never saw) ────────────────────────────
print(f"\n{'─' * 70}")
print(f"  TOKENS UNIQUE TO WRITINGPROMPTS (never seen in TinyStories)")
print(f"{'─' * 70}")

# Sort WP-only tokens by frequency in WP
wp_only_freq = [(tid, wp_tokens[tid]) for tid in wp_only]
wp_only_freq.sort(key=lambda x: -x[1])

print(f"\n  Top 30 WP-only tokens (by frequency):")
for tok_id, count in wp_only_freq[:30]:
    try:
        decoded = tokenizer.decode([tok_id]).replace('\n', '\\n')
    except:
        decoded = f"<id:{tok_id}>"
    pct = count / sum(wp_tokens.values()) * 100
    print(f"    [{tok_id:>5}] '{decoded:>20}' : {count:>6,}  ({pct:.3f}%)")

# ── 6. Difficulty score distributions ────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  DIFFICULTY SCORE DISTRIBUTIONS")
print(f"{'─' * 70}")

ts_scores_path = "Curriculum_scores/tiny_stories.npy"
wp_scores_path = "writingprompts_scores.npy"

if os.path.exists(ts_scores_path):
    ts_scores = np.load(ts_scores_path)
    print(f"\n  TinyStories scores shape: {ts_scores.shape}")
    if ts_scores.ndim == 2 and ts_scores.shape[1] >= 2:
        diffs = ts_scores[:, 1]
    else:
        diffs = ts_scores
    print(f"    min={diffs.min():.3f}  max={diffs.max():.3f}  mean={diffs.mean():.3f}  std={diffs.std():.3f}")
    print(f"    p10={np.percentile(diffs, 10):.3f}  p50={np.percentile(diffs, 50):.3f}  p90={np.percentile(diffs, 90):.3f}")
else:
    print(f"  TinyStories scores not found at {ts_scores_path}")

if os.path.exists(wp_scores_path):
    wp_scores = np.load(wp_scores_path)
    print(f"\n  WritingPrompts scores shape: {wp_scores.shape}")
    if wp_scores.ndim == 2 and wp_scores.shape[1] >= 2:
        diffs = wp_scores[:, 1]
    else:
        diffs = wp_scores
    print(f"    min={diffs.min():.3f}  max={diffs.max():.3f}  mean={diffs.mean():.3f}  std={diffs.std():.3f}")
    print(f"    p10={np.percentile(diffs, 10):.3f}  p50={np.percentile(diffs, 50):.3f}  p90={np.percentile(diffs, 90):.3f}")
else:
    print(f"  WritingPrompts scores not found at {wp_scores_path}")

# ── 7. Sample content comparison ────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  SAMPLE CONTENT COMPARISON")
print(f"{'─' * 70}")

print(f"\n  --- 3 Random TinyStories chunks (decoded) ---")
for i in random.sample(range(len(ts_chunks)), 3):
    text = tokenizer.decode(ts_chunks[i].tolist() if hasattr(ts_chunks[i], 'tolist') else list(ts_chunks[i]))
    text = text[:300].replace('\n', ' ')
    print(f"\n  [{i}] {text}...")

print(f"\n  --- 3 Random WritingPrompts chunks (decoded) ---")
for i in random.sample(range(len(wp_chunks)), 3):
    text = tokenizer.decode(wp_chunks[i].tolist() if hasattr(wp_chunks[i], 'tolist') else list(wp_chunks[i]))
    text = text[:400].replace('\n', ' ')
    print(f"\n  [{i}] {text}...")

# ── 8. Entropy analysis ─────────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  TOKEN ENTROPY ANALYSIS")
print(f"{'─' * 70}")

def token_entropy(counter):
    total = sum(counter.values())
    probs = np.array([c / total for c in counter.values()])
    return -np.sum(probs * np.log2(probs + 1e-12))

ts_entropy = token_entropy(ts_tokens)
wp_entropy = token_entropy(wp_tokens)

print(f"  TinyStories token entropy:    {ts_entropy:.2f} bits")
print(f"  WritingPrompts token entropy: {wp_entropy:.2f} bits")
print(f"  Difference:                   {wp_entropy - ts_entropy:+.2f} bits")
print(f"  (Higher entropy = more uniform/diverse token usage)")

# ── 9. Repetition analysis ──────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  WITHIN-CHUNK REPETITION ANALYSIS")
print(f"{'─' * 70}")

def repetition_rate(chunks, n_samples=5000):
    """Measure n-gram repetition within individual chunks."""
    bigram_reps = []
    trigram_reps = []
    
    sample = random.sample(range(len(chunks)), min(n_samples, len(chunks)))
    for idx in sample:
        c = chunks[idx]
        if hasattr(c, 'tolist'):
            c = c.tolist()
        
        # Bigram repetition
        bigrams = [(c[i], c[i+1]) for i in range(len(c)-1)]
        if bigrams:
            unique_ratio = len(set(bigrams)) / len(bigrams)
            bigram_reps.append(1 - unique_ratio)
        
        # Trigram repetition
        trigrams = [(c[i], c[i+1], c[i+2]) for i in range(len(c)-2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            trigram_reps.append(1 - unique_ratio)
    
    return np.mean(bigram_reps), np.mean(trigram_reps)

ts_bi, ts_tri = repetition_rate(ts_chunks)
wp_bi, wp_tri = repetition_rate(wp_chunks)

print(f"  TinyStories:    bigram rep={ts_bi:.4f}  trigram rep={ts_tri:.4f}")
print(f"  WritingPrompts: bigram rep={wp_bi:.4f}  trigram rep={wp_tri:.4f}")

# ── 10. Scale comparison ────────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f"  SCALE COMPARISON")
print(f"{'─' * 70}")

ts_total_tokens = len(ts_chunks) * ts_lengths[0]
wp_total_tokens = len(wp_chunks) * wp_lengths[0]

print(f"  TinyStories:    {len(ts_chunks):>10,} chunks × {ts_lengths[0]} = {ts_total_tokens/1e6:>8.1f}M tokens")
print(f"  WritingPrompts: {len(wp_chunks):>10,} chunks × {wp_lengths[0]} = {wp_total_tokens/1e6:>8.1f}M tokens")
print(f"  Ratio:          WP is {wp_total_tokens/ts_total_tokens:.2f}× the size of TS")
print(f"")
print(f"  Training budget consumed:")
print(f"    Baseline on TS:      376.8M tokens ({376.8/ts_total_tokens*1e6:.0f}% of TS)")
print(f"    Stage B on WP:        27.7M tokens ({27.7/wp_total_tokens*1e6:.1f}% of WP)")
print(f"    Stage C on WP:        28.7M tokens ({28.7/wp_total_tokens*1e6:.1f}% of WP)")

print(f"\n{'=' * 70}")
print(f"  ANALYSIS COMPLETE")
print(f"{'=' * 70}")
