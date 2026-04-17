import os
# Suppress huggingface symlink warnings and limit threads
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from src.score_difficulty import compute_perplexity_batch, compute_coherence_bonus

print("Loading GPT-2 model (this may take a moment to download)...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.", flush=True)

texts = [
    "Once upon a time there was a little dog. He liked to play with older dogs.",
    "First, the quantum discriminator calibrates the asymptotic singularity, because orthogonal matrix inversion halts spontaneously.",
    "sdjkfhakjsdfhgkjdhfkgjhdsfkjghdfkjghdkfjhgkdfjhgkjdfhgkjdfhg",
]

print("\nComputing PPL...", flush=True)
ppls = compute_perplexity_batch(texts, model, tokenizer, "cpu", max_length=512)
print("\n=== GPT-2 Perplexity Test ===")
print("Text 1 (Simple story):")
print(f"  PPL: {ppls[0]:.2f} (Expected low, < 100)")
print("Text 2 (Complex scientific):")
print(f"  PPL: {ppls[1]:.2f} (Expected medium-high, ~200-800)")
print("Text 3 (Garbage random text):")
print(f"  PPL: {ppls[2]:.2f} (Expected maxed out, ~1000s or fallback 500)")

print("\n=== Coherence Bonus Test ===")
t1 = "first " * 50
print(f"Short text bonus (< 150 len): {compute_coherence_bonus(t1, 100)}")

t2 = "First he went home. Then he ate dinner. Because he was hungry. After that he slept. However he woke up. Finally it was morning. " * 30
t2_len = 200 
print(f"Long repetitive with markers: {compute_coherence_bonus(t2, t2_len):.2f}")

t3 = "The quick brown fox jumps over the lazy dog. " * 30
print(f"Long text without markers: {compute_coherence_bonus(t3, 200):.2f}")

print("\nAll tests ran successfully!")
