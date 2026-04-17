"""
memorization_vs_generalization_test.py

Compares two checkpoints on:
1. MEMORIZATION SIGNALS: Does it regurgitate training data?
2. GENERALIZATION SIGNALS: Does it understand patterns?

Key tests:
- In-distribution (common TinyStories patterns) vs Out-of-distribution
- Novel names/entities never seen in training
- Prompt sensitivity (does input affect output?)
- OOD domain adaptation
"""

import os
import torch
from tokenizers import Tokenizer
from src.model import SLM
import difflib
import re
import glob

TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Find checkpoints
all_ckpts = glob.glob(os.path.join("checkpoints/", "stage0_best*.pt"))
if len(all_ckpts) < 2:
    print("❌ Need at least 2 checkpoints!")
    exit(1)

ckpt1_path = all_ckpts[0]
ckpt2_path = all_ckpts[1]

print("="*80)
print("MEMORIZATION vs GENERALIZATION ANALYSIS")
print("="*80)
print(f"Checkpoint 1: {ckpt1_path}")
print(f"Checkpoint 2: {ckpt2_path}\n")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = SLM(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def generate(model, tokenizer, prompt, temp=0.01, max_new=60):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new=max_new, temperature=temp, top_k=1, use_cache=True)
    return tokenizer.decode(output_ids[0].tolist())


model1 = load_model(ckpt1_path)
model2 = load_model(ckpt2_path)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


# ─── TEST 1: MEMORIZATION - Common Training Phrases ────────────────────────────

print("\n" + "="*80)
print("TEST 1: MEMORIZATION - Common TinyStories Phrases")
print("="*80)
print("(If outputs are identical to training data → MEMORIZATION signal)\n")

# These are VERY common in TinyStories
common_phrases = [
    "Once upon a time there was a little",
    "The little boy loved to",
    "One day she decided to",
]

for phrase in common_phrases:
    out1 = generate(model1, tokenizer, phrase)
    out2 = generate(model2, tokenizer, phrase)
    
    # Check similarity
    similarity = difflib.SequenceMatcher(None, out1, out2).ratio()
    
    print(f"Prompt: {repr(phrase)}")
    print(f"  Model 1: {out1[:70]}...")
    print(f"  Model 2: {out2[:70]}...")
    print(f"  Similarity: {100*similarity:.0f}% {'🔴 HIGH (memorization)' if similarity > 0.8 else '🟢 LOW (generalization)'}")
    print()


# ─── TEST 2: GENERALIZATION - Rare/Novel Words ────────────────────────────────

print("\n" + "="*80)
print("TEST 2: GENERALIZATION - Rare/Novel Entities")
print("="*80)
print("(If model handles rare names well → GENERALIZATION signal)\n")

# Uncommon names (unlikely in TinyStories)
rare_names = [
    ("Zephyr", "Once upon a time there was a boy named Zephyr"),
    ("Indigo", "Once upon a time there was a girl named Indigo"),
    ("Cipher", "Once upon a time there was a robot named Cipher"),
]

for name, prompt in rare_names:
    out1 = generate(model1, tokenizer, prompt)
    out2 = generate(model2, tokenizer, prompt)
    
    # Check if name is preserved
    name_in_1 = name.lower() in out1.lower()
    name_in_2 = name.lower() in out2.lower()
    
    print(f"Novel name: {repr(name)}")
    print(f"  Model 1: {out1[:70]}... {'✓ Preserves' if name_in_1 else '✗ Loses'}")
    print(f"  Model 2: {out2[:70]}... {'✓ Preserves' if name_in_2 else '✗ Loses'}")
    print()


# ─── TEST 3: GENERALIZATION - OOD Domain Prompts ─────────────────────────────

print("\n" + "="*80)
print("TEST 3: GENERALIZATION - Out-of-Domain (OOD) Prompts")
print("="*80)
print("(If model handles non-story domains → GENERALIZATION signal)\n")

ood_prompts = [
    "The scientist discovered",
    "The algorithm computed",
    "The crystal formation showed",
]

for prompt in ood_prompts:
    out1 = generate(model1, tokenizer, prompt)
    out2 = generate(model2, tokenizer, prompt)
    
    print(f"OOD Prompt: {repr(prompt)}")
    print(f"  Model 1: {out1[:70]}...")
    print(f"  Model 2: {out2[:70]}...")
    print()


# ─── TEST 4: GENERALIZATION - Prompt Sensitivity ──────────────────────────────

print("\n" + "="*80)
print("TEST 4: GENERALIZATION - Prompt Sensitivity")
print("="*80)
print("(If small prompt changes → large output changes → GENERALIZATION)\n")

base_prompt = "The girl was"
variants = [
    "The girl was happy",
    "The girl was sad",
    "The girl was angry",
]

print("Same beginning, different adjectives:\n")
print(f"{'Model':<10} {'happy':<35} {'sad':<35} {'angry':<35} Variance")
print("-"*120)

for model_num, model in enumerate([model1, model2], 1):
    outputs = [generate(model, tokenizer, v) for v in variants]
    
    # Calculate pairwise differences
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff = 1 - difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            diffs.append(diff)
    
    variance = sum(diffs) / len(diffs) if diffs else 0
    status = "🟢 HIGH (good)" if variance > 0.3 else "🔴 LOW (memorizing)"
    
    print(f"M{model_num:<8} {outputs[0][:33]:<35} {outputs[1][:33]:<35} {outputs[2][:33]:<35} {100*variance:>5.0f}% {status}")

print()


# ─── TEST 5: ROBUSTNESS - Handling Variations ────────────────────────────────

print("\n" + "="*80)
print("TEST 5: ROBUSTNESS - Input Variations")
print("="*80)
print("(If model is robust to phrasing → GENERALIZATION)\n")

# Same meaning, different phrasing
similar_prompts = [
    "A girl named Sarah walked",
    "Sarah, a girl, walked",
    "Walking through the forest was a girl named Sarah",
]

print("Same scenario, different phrasings:\n")

for model_num, model in enumerate([model1, model2], 1):
    outputs = [generate(model, tokenizer, p, max_new=40) for p in similar_prompts]
    
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            diffs.append(diff)
    
    avg_similarity = sum(diffs) / len(diffs) if diffs else 0
    robustness = 1 - avg_similarity
    status = "🟢 ROBUST (generalizes)" if robustness > 0.4 else "🔴 FRAGILE (memorizes)"
    
    print(f"Model {model_num}:")
    print(f"  Output 1: {outputs[0][:60]}...")
    print(f"  Output 2: {outputs[1][:60]}...")
    print(f"  Output 3: {outputs[2][:60]}...")
    print(f"  Robustness Score: {100*robustness:.0f}% {status}\n")


# ─── SUMMARY ──────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("MEMORIZATION vs GENERALIZATION SUMMARY")
print("="*80)

print("""
MEMORIZATION INDICATORS (🔴 BAD):
  ✗ Identical outputs for common phrases
  ✗ Fails on rare/novel words
  ✗ Cannot handle OOD domains
  ✗ Low prompt sensitivity (inputs don't affect output)
  ✗ Fragile to phrasing changes

GENERALIZATION INDICATORS (🟢 GOOD):
  ✓ Diverse outputs for common phrases
  ✓ Handles rare/novel words well
  ✓ Adapts to OOD domains
  ✓ High prompt sensitivity (inputs matter)
  ✓ Robust to paraphrasing/variants

INTERPRETATION:
  - Model with more 🟢 signals = Better generalization
  - Model with more 🔴 signals = More memorization
  - Fully trained models often show BOTH (memorize + generalize)
  - Early exit models may have stopped before memorization
  
INSIGHT:
  If early-exit model scores higher on generalization tests
  → Early stopping prevented overfitting (good!)
  
  If fully-trained model scores higher
  → Extra training helped (good!)
""")

print("="*80)
