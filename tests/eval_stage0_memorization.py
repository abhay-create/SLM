"""
eval_stage0_memorization.py
Tests to detect whether Stage 0 model is memorizing vs. actually understanding.

Checks:
1. Novelty Test — Is generated text different from training data?
2. Diversity Test — Same prompt → different outputs (temperature > 0)?
3. OOD Prompts — Can it generate from prompts NOT in training data?
4. Perturbation — Do small prompt changes cause output changes?
5. N-gram Overlap — How much of output appears verbatim in training?
6. Semantic Coherence — Does generated text make logical sense?
"""

import os
import sys
import torch
import random
import difflib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from src.model import SLM

# ─── Configuration ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/stage1_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 80
TEMPERATURE    = 0.8
TOP_K          = 50


def load_model_and_tokenizer():
    """Load checkpoint and tokenizer."""
    print(f"[eval] Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = SLM(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[eval] Model & tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    return model, tokenizer


def generate(model, tokenizer, prompt, num_outputs=1, temperature=TEMPERATURE, top_k=TOP_K):
    """Generate multiple outputs for the same prompt."""
    outputs = []
    for _ in range(num_outputs):
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new=MAX_NEW_TOKENS,
                temperature=temperature,
                top_k=top_k,
                use_cache=True
            )
        
        text = tokenizer.decode(output_ids[0].tolist())
        outputs.append(text)
    
    return outputs


def test_diversity(model, tokenizer):
    """TEST 1: Diversity — Does the model generate different outputs from the same prompt?"""
    print("\n" + "="*80)
    print("TEST 1: DIVERSITY (Same prompt → different outputs?)")
    print("="*80)
    
    prompt = "Once upon a time there was"
    outputs = generate(model, tokenizer, prompt, num_outputs=3, temperature=0.9)
    
    # Calculate character-level differences
    for i, out in enumerate(outputs, 1):
        print(f"\n[Output {i}]")
        print(f"{out[:100]}...")
    
    # Check pairwise differences
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff_ratio = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            diffs.append(diff_ratio)
    
    avg_similarity = sum(diffs) / len(diffs) if diffs else 0
    print(f"\n[Diversity Score] Avg output similarity: {avg_similarity:.2%}")
    print(f"  → {100-avg_similarity*100:.1f}% different (higher = better)")
    
    if avg_similarity > 0.8:
        print("  ⚠️  LOW diversity — model might be memorizing!")
    else:
        print("  ✓ Good diversity — model generates varied outputs")


def test_perturbation(model, tokenizer):
    """TEST 2: Perturbation — Do small prompt changes cause output changes?"""
    print("\n" + "="*80)
    print("TEST 2: PERTURBATION (Modify prompt → output changes?)")
    print("="*80)
    
    base_prompt = "The girl walked into the"
    variants = [
        base_prompt,
        "The girl walked in the",  # 'into' → 'in'
        "The girl ran into the",    # 'walked' → 'ran'
        "The boy walked into the",  # 'girl' → 'boy'
    ]
    
    outputs = []
    for i, prompt in enumerate(variants):
        out = generate(model, tokenizer, prompt, num_outputs=1)[0]
        outputs.append(out)
        print(f"\n[Prompt {i}] {repr(prompt)}")
        print(f"Output: {out[:80]}...")
    
    # Calculate pairwise differences
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff_ratio = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            diffs.append(diff_ratio)
    
    avg_change = sum(diffs) / len(diffs) if diffs else 0
    print(f"\n[Perturbation Score] Avg output similarity: {avg_change:.2%}")
    
    if avg_change > 0.9:
        print("  ⚠️  Outputs very similar despite prompt changes — possible memorization")
    else:
        print("  ✓ Prompt changes cause output changes — good sign of understanding")


def test_ood_prompts(model, tokenizer):
    """TEST 3: OOD Prompts — Can model handle unusual/novel prompts?"""
    print("\n" + "="*80)
    print("TEST 3: OUT-OF-DISTRIBUTION PROMPTS")
    print("="*80)
    
    # These are NOT typical TinyStories opening phrases
    ood_prompts = [
        "The quantum algorithm calculated",
        "In a surreal dreamscape filled",
        "The archaeologist discovered ancient",
        "Beneath the crystalline surface lived",
    ]
    
    for i, prompt in enumerate(ood_prompts, 1):
        out = generate(model, tokenizer, prompt, num_outputs=1)[0]
        print(f"\n[OOD Prompt {i}] {repr(prompt)}")
        print(f"Output: {out[:100]}...")
        
        # Check for basic coherence: does it continue naturally?
        # Count how many complete sentences
        sentences = out.count('.') + out.count('!') + out.count('?')
        if sentences >= 2:
            print(f"  ✓ Generates {sentences} sentences (coherent)")
        else:
            print(f"  ⚠️  Only {sentences} sentences (might struggle with novel prompts)")


def test_ngram_overlap(model, tokenizer):
    """TEST 4: N-gram Overlap — How much of output matches training data exactly?"""
    print("\n" + "="*80)
    print("TEST 4: N-GRAM OVERLAP WITH TRAINING DATA")
    print("="*80)
    
    # These prompts are common in TinyStories
    prompts = [
        "Once upon a time there was",
        "In the forest there lived",
        "One day a little",
    ]
    
    for prompt in prompts:
        output = generate(model, tokenizer, prompt, num_outputs=1)[0]
        
        # Extract 5-grams and count exact matches with prompt
        words = output.split()
        five_grams = [' '.join(words[i:i+5]) for i in range(len(words)-5)]
        
        # Very rough heuristic: if output *starts* same as training continuations, it's suspicious
        # We'd need actual training data for proper check, but this is a proxy
        
        print(f"\n[Prompt] {repr(prompt)}")
        print(f"[Output] {output[:100]}...")
        print(f"  Generated {len(five_grams)} 5-grams")
        print(f"  (Note: Full n-gram check requires access to training dataset)")


def test_temperature_effect(model, tokenizer):
    """TEST 5: Temperature Effect — Does low vs high temp show different behavior?"""
    print("\n" + "="*80)
    print("TEST 5: TEMPERATURE EFFECT (Low temp vs high temp?)")
    print("="*80)
    
    prompt = "She opened the door and"
    
    # Use low temperature (0.1) instead of 0.0 to avoid numerical instability
    print(f"\n[Prompt] {repr(prompt)}")
    
    print("\nTemperature=0.1 (low, mostly deterministic):")
    out1 = generate(model, tokenizer, prompt, num_outputs=1, temperature=0.1)[0]
    out2 = generate(model, tokenizer, prompt, num_outputs=1, temperature=0.1)[0]
    print(f"  Output 1: {out1[:60]}...")
    print(f"  Output 2: {out2[:60]}...")
    
    similarity_low = difflib.SequenceMatcher(None, out1, out2).ratio()
    if similarity_low > 0.95:
        print(f"  ✓ Mostly deterministic ({100*similarity_low:.1f}% similar)")
    else:
        print(f"  ⚠️  Some variation at low temp ({100*similarity_low:.1f}% similar)")
    
    print("\nTemperature=1.5 (high, stochastic):")
    out3 = generate(model, tokenizer, prompt, num_outputs=1, temperature=1.5)[0]
    out4 = generate(model, tokenizer, prompt, num_outputs=1, temperature=1.5)[0]
    print(f"  Output 1: {out3[:60]}...")
    print(f"  Output 2: {out4[:60]}...")
    
    similarity_high = difflib.SequenceMatcher(None, out3, out4).ratio()
    if similarity_high < 0.8:
        print(f"  ✓ High variance at high temp ({100*similarity_high:.1f}% similar)")
    else:
        print(f"  ⚠️  Low variance at high temp ({100*similarity_high:.1f}% similar)")


def test_completion_quality(model, tokenizer):
    """TEST 6: Completion Quality — Does output make semantic sense?"""
    print("\n" + "="*80)
    print("TEST 6: SEMANTIC QUALITY (Does output make sense?)")
    print("="*80)
    
    test_cases = [
        ("The cat was", "animal-related"),
        ("It was a sunny day", "scene-setting"),
        ("Sarah wanted to", "goal-driven"),
        ("Suddenly there was a", "event-driven"),
    ]
    
    for prompt, context in test_cases:
        output = generate(model, tokenizer, prompt, num_outputs=1)[0]
        
        print(f"\n[{context}] {repr(prompt)}")
        print(f"Output: {output}")
        
        # Check for red flags:
        # - Repeated words (often sign of poor generation)
        words = output.lower().split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(3)
        
        is_repetitive = any(count > 5 and word not in ['the', 'a', 'and', 'to', 'of'] 
                           for word, count in most_common)
        
        if is_repetitive:
            print(f"  ⚠️  Repetitive words detected: {most_common}")
        else:
            print(f"  ✓ Natural word distribution")


def main():
    print("="*80)
    print("STAGE 0 MEMORIZATION EVALUATION")
    print("="*80)
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Run all tests
    test_diversity(model, tokenizer)
    test_perturbation(model, tokenizer)
    test_ood_prompts(model, tokenizer)
    test_ngram_overlap(model, tokenizer)
    test_temperature_effect(model, tokenizer)
    test_completion_quality(model, tokenizer)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  ✓ High diversity + good perturbation response = understands patterns")
    print("  ⚠️  Low diversity + similar outputs = possible memorization")
    print("  ✓ Handles OOD prompts well = generalizes beyond training data")
    print("  ⚠️  Only generates near training data = overfitting/memorization")
    print()


if __name__ == "__main__":
    main()
