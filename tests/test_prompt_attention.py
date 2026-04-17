"""
test_prompt_attention.py
Tests whether the model actually READS and respects input tokens,
or if it just ignores them and generates default continuations.

Red flags for overfitting:
- Model ignores explicit names in prompt
- Model generates generic "Once upon a time" continuations regardless of input
- Model doesn't preserve entities from prompt
"""

import os
import torch
from tokenizers import Tokenizer
from src.model import SLM

CHECKPOINT_PATH = "checkpoints/stage0_best _1st_trial.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 60
TEMPERATURE    = 0.7
TOP_K          = 50


def load_model_and_tokenizer():
    """Load checkpoint and tokenizer."""
    print(f"[test] Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = SLM(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    return model, tokenizer


def generate(model, tokenizer, prompt):
    """Generate text from prompt."""
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            use_cache=True
        )
    
    return tokenizer.decode(output_ids[0].tolist())


def test_name_preservation(model, tokenizer):
    """TEST 1: Does model preserve NAMES from the prompt?"""
    print("\n" + "="*80)
    print("TEST 1: NAME PRESERVATION (Does model use names from prompt?)")
    print("="*80)
    
    # Test with different names
    test_cases = [
        ("Once upon a time there was a girl named Alice", "Alice"),
        ("Once upon a time there was a boy named Charlie", "Charlie"),
        ("Once upon a time there was a mouse named Guddu", "Guddu"),
        ("Once upon a time there was a cat named Whiskers", "Whiskers"),
    ]
    
    preserved_count = 0
    
    for prompt, target_name in test_cases:
        output = generate(model, tokenizer, prompt)
        
        # Check if target name appears in output (case-insensitive)
        name_present = target_name.lower() in output.lower()
        status = "✓" if name_present else "✗"
        
        print(f"\n{status} Prompt name: {target_name}")
        print(f"   Output: {output[:80]}...")
        
        if name_present:
            preserved_count += 1
            print(f"   ✓ {target_name} used in output")
        else:
            print(f"   ⚠️  {target_name} IGNORED — model generated different names")
    
    preservation_rate = preserved_count / len(test_cases)
    print(f"\n[Name Preservation Rate] {preserved_count}/{len(test_cases)} = {100*preservation_rate:.0f}%")
    
    if preservation_rate < 0.5:
        print("  🚨 LOW preservation — model likely ignoring input tokens!")
    elif preservation_rate > 0.75:
        print("  ✓ Good preservation — model respects input names")
    else:
        print("  ⚠️  Moderate preservation — room for improvement")


def test_attribute_consistency(model, tokenizer):
    """TEST 2: Does model use ADJECTIVES/ATTRIBUTES from prompt?"""
    print("\n" + "="*80)
    print("TEST 2: ATTRIBUTE CONSISTENCY (Does model use adjectives from prompt?)")
    print("="*80)
    
    test_cases = [
        ("Once upon a time there was a very brave girl", ["brave", "courageous", "fearless"]),
        ("Once upon a time there was a very lazy cat", ["lazy", "sleepy", "tired"]),
        ("Once upon a time there was a very clever boy", ["clever", "smart", "intelligent", "wise"]),
    ]
    
    consistency_score = 0
    
    for prompt, target_attrs in test_cases:
        output = generate(model, tokenizer, prompt)
        
        # Check if any target attribute appears
        attr_found = any(attr.lower() in output.lower() for attr in target_attrs)
        status = "✓" if attr_found else "✗"
        
        print(f"\n{status} Prompt attribute: {target_attrs[0]}")
        print(f"   Output: {output[:100]}...")
        
        if attr_found:
            consistency_score += 1
            print(f"   ✓ Attributes preserved")
        else:
            print(f"   ⚠️  Attributes NOT reflected in output")
    
    consistency_rate = consistency_score / len(test_cases)
    print(f"\n[Attribute Consistency] {consistency_score}/{len(test_cases)} = {100*consistency_rate:.0f}%")
    
    if consistency_rate < 0.5:
        print("  🚨 Model ignores prompt context/attributes!")
    else:
        print("  ✓ Model respects prompt tone/attributes")


def test_sequence_dependence(model, tokenizer):
    """TEST 3: Does changing prompt ORDER change output?"""
    print("\n" + "="*80)
    print("TEST 3: SEQUENCE DEPENDENCE (Does prompt order matter?)")
    print("="*80)
    
    # Same elements, different order
    prompts = [
        "Once upon a time there was a girl named Sarah in a forest",
        "Once upon a time in a forest there was a girl named Sarah",
        "There was a girl named Sarah in a forest once upon a time",
    ]
    
    outputs = []
    for i, prompt in enumerate(prompts, 1):
        output = generate(model, tokenizer, prompt)
        outputs.append(output)
        print(f"\n[Prompt {i}] {prompt}")
        print(f"Output: {output[:80]}...")
    
    # Check if outputs are different
    import difflib
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            sim = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            diffs.append(sim)
    
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    print(f"\n[Sequence Dependence] Average similarity: {100*avg_diff:.1f}%")
    
    if avg_diff > 0.9:
        print("  🚨 Outputs almost identical — prompt order IGNORED!")
    elif avg_diff < 0.3:
        print("  ✓ Order matters — model is sequence-dependent")
    else:
        print("  ⚠️  Partial dependence — model somewhat ignores order")


def test_specific_continuation(model, tokenizer):
    """TEST 4: Can model continue SPECIFIC context?"""
    print("\n" + "="*80)
    print("TEST 4: SPECIFIC CONTEXT CONTINUATION")
    print("="*80)
    
    test_cases = [
        ("The dog was", "animal/pet context"),
        ("The king was", "royal context"),
        ("The robot was", "sci-fi context"),
        ("The witch was", "magical context"),
    ]
    
    print("\nChecking if model picks up on domain-specific words:\n")
    
    for prompt, context in test_cases:
        output = generate(model, tokenizer, prompt)
        print(f"[{context}] {prompt}")
        print(f"  → {output[:80]}...")


def main():
    print("="*80)
    print("PROMPT ATTENTION & OVERFITTING DIAGNOSTIC")
    print("="*80)
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Run diagnostic tests
    test_name_preservation(model, tokenizer)
    test_attribute_consistency(model, tokenizer)
    test_sequence_dependence(model, tokenizer)
    test_specific_continuation(model, tokenizer)
    
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print("""
If you see:
  🚨 LOW name preservation (<50%)  → Model ignoring input! Possible issues:
     - Model too small, attention not learning
     - Prompt tokens not influencing hidden states
     - Overfitting to generic "Once upon a time" patterns
     
  ✓ HIGH name preservation (>75%)  → Model paying attention properly
     - Input tokens directly influence generation
     - Good sign of actual language understanding

Recommendations if scores are low:
  1. Train longer (Stage 0 may need more steps)
  2. Increase model capacity (more layers/dims)
  3. Use dataset_overlap.py with replay_ratio to reduce overfitting
  4. Reduce max_tokens per stage (less token budget = less memorization)
""")


if __name__ == "__main__":
    main()
