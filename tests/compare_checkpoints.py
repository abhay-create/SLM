"""
compare_checkpoints.py
Compare two Stage 0 checkpoints:
- One fully trained
- One stopped early due to early exit

Measures: generation quality, entity consistency, training curves, etc.
"""

import os
import torch
from tokenizers import Tokenizer
from src.model import SLM
import difflib
import re

TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Find checkpoints in directory
import glob

checkpoint_dir = "checkpoints/"
all_ckpts = glob.glob(os.path.join(checkpoint_dir, "stage0_best*.pt"))

print("="*80)
print("STAGE 0 CHECKPOINT COMPARISON")
print("="*80)
print(f"\nFound {len(all_ckpts)} Stage 0 checkpoints:")
for i, ckpt in enumerate(all_ckpts, 1):
    print(f"  {i}. {ckpt}")

if len(all_ckpts) < 2:
    print("\n❌ Need at least 2 checkpoints to compare!")
    exit(1)

# Select two checkpoints
ckpt1_path = all_ckpts[0]
ckpt2_path = all_ckpts[1]

print(f"\nComparing:")
print(f"  Checkpoint 1: {ckpt1_path}")
print(f"  Checkpoint 2: {ckpt2_path}")


def load_model(ckpt_path):
    """Load checkpoint and return model + metadata."""
    print(f"\n[Loading] {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    model = SLM(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    metadata = {
        "step": ckpt.get("step", 0),
        "tokens_seen": ckpt.get("tokens_seen", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
    }
    
    return model, metadata


def generate(model, tokenizer, prompt, max_new=60, temperature=0.01, top_k=1):
    """Generate text from prompt."""
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            use_cache=True
        )
    
    return tokenizer.decode(output_ids[0].tolist())


def test_name_preservation(model, tokenizer, name="Alice"):
    """Test if model preserves character name."""
    prompt = f"Once upon a time there was a girl named {name}. {name} was curious"
    output = generate(model, tokenizer, prompt)
    
    name_count = len(re.findall(r'\b' + name + r'\b', output, re.IGNORECASE))
    return name_count >= 2  # Should appear at least twice (prompt + output)


def test_diversity(model, tokenizer, prompt="Once upon a time"):
    """Generate same prompt 3x, check diversity."""
    outputs = []
    for _ in range(3):
        # Use higher temperature for diversity test
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new=50,
                temperature=0.7,
                top_k=50,
                use_cache=True
            )
        
        outputs.append(tokenizer.decode(output_ids[0].tolist()))
    
    # Calculate pairwise differences
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff_ratio = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            diffs.append(diff_ratio)
    
    avg_diversity = 1 - (sum(diffs) / len(diffs) if diffs else 0)
    return avg_diversity


def test_quality(model, tokenizer, prompt="The dog was"):
    """Generate and check for obvious quality issues."""
    output = generate(model, tokenizer, prompt)
    
    # Check for word repetition (sign of poor generation)
    words = output.lower().split()
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    
    # Check for excessive repetition of non-common words
    excessive_repetition = sum(1 for w, c in word_counts.items() 
                               if c > 5 and w not in ['the', 'a', 'and', 'to', 'of', 'is', 'was'])
    
    return excessive_repetition == 0  # True if no excessive repetition


# Load both models and tokenizer
print("\n" + "="*80)
print("LOADING MODELS")
print("="*80)

model1, meta1 = load_model(ckpt1_path)
model2, meta2 = load_model(ckpt2_path)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Print metadata comparison
print("\n" + "="*80)
print("TRAINING METADATA COMPARISON")
print("="*80)

print(f"\n{'Metric':<30} {'Checkpoint 1':>20} {'Checkpoint 2':>20}")
print("-"*70)
print(f"{'Steps':<30} {meta1['step']:>20} {meta2['step']:>20}")
print(f"{'Tokens Seen':<30} {meta1['tokens_seen']:>20,} {meta2['tokens_seen']:>20,}")
print(f"{'Best Val Loss':<30} {meta1['best_val_loss']:>20.4f} {meta2['best_val_loss']:>20.4f}")

# Determine which is fully trained
full_ckpt = ckpt1_path if meta1['step'] > meta2['step'] else ckpt2_path
early_ckpt = ckpt2_path if meta1['step'] > meta2['step'] else ckpt1_path
full_model = model1 if meta1['step'] > meta2['step'] else model2
early_model = model2 if meta1['step'] > meta2['step'] else model1

print(f"\nFully trained: {os.path.basename(full_ckpt)}")
print(f"Early exit: {os.path.basename(early_ckpt)}")

# Generation quality comparison
print("\n" + "="*80)
print("GENERATION QUALITY COMPARISON")
print("="*80)

test_prompts = [
    "Once upon a time there was",
    "The princess was",
    "In the forest lived",
]

print("\n[Checkpoint 1 (Fully Trained)]\n")
for prompt in test_prompts:
    output = generate(model1, tokenizer, prompt, max_new=60, temperature=0.01)
    print(f"Prompt: {repr(prompt)}")
    print(f"Output: {output[:100]}...\n")

print("\n[Checkpoint 2 (Early Exit)]\n")
for prompt in test_prompts:
    output = generate(model2, tokenizer, prompt, max_new=60, temperature=0.01)
    print(f"Prompt: {repr(prompt)}")
    print(f"Output: {output[:100]}...\n")

# Metric comparison
print("\n" + "="*80)
print("QUALITY METRICS COMPARISON")
print("="*80)

print(f"\n{'Metric':<30} {'Checkpoint 1':>20} {'Checkpoint 2':>20}")
print("-"*70)

# Test 1: Name preservation
name_pres_1 = test_name_preservation(model1, tokenizer)
name_pres_2 = test_name_preservation(model2, tokenizer)
print(f"{'Name Preservation':<30} {str(name_pres_1):>20} {str(name_pres_2):>20}")

# Test 2: Diversity
div_1 = test_diversity(model1, tokenizer)
div_2 = test_diversity(model2, tokenizer)
print(f"{'Diversity Score':<30} {div_1:>20.3f} {div_2:>20.3f}")

# Test 3: Generation quality
qual_1 = test_quality(model1, tokenizer)
qual_2 = test_quality(model2, tokenizer)
print(f"{'No Repetition Issues':<30} {str(qual_1):>20} {str(qual_2):>20}")

# Summary
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

better_count = 0
worse_count = 0

if meta1['best_val_loss'] < meta2['best_val_loss']:
    print(f"\n✓ Checkpoint 1 has LOWER val loss ({meta1['best_val_loss']:.4f} vs {meta2['best_val_loss']:.4f})")
    better_count += 1
else:
    print(f"\n✓ Checkpoint 2 has LOWER val loss ({meta2['best_val_loss']:.4f} vs {meta1['best_val_loss']:.4f})")
    worse_count += 1

if name_pres_1 and not name_pres_2:
    print(f"✓ Checkpoint 1 preserves names better")
    better_count += 1
elif name_pres_2 and not name_pres_1:
    print(f"✓ Checkpoint 2 preserves names better")
    worse_count += 1

if div_1 > div_2:
    print(f"✓ Checkpoint 1 has higher diversity ({div_1:.3f} vs {div_2:.3f})")
    better_count += 1
else:
    print(f"✓ Checkpoint 2 has higher diversity ({div_2:.3f} vs {div_1:.3f})")
    worse_count += 1

print(f"\n{'='*80}")
if better_count > worse_count:
    print(f"VERDICT: Checkpoint 1 (fully trained) is BETTER overall")
elif worse_count > better_count:
    print(f"VERDICT: Checkpoint 2 (early exit) is surprisingly competitive")
else:
    print(f"VERDICT: Both checkpoints are comparable")

print(f"\nImplication of early stop:")
print(f"  - Fully trained model converged: {'YES' if meta1['step'] > meta2['step']*1.5 else 'NO'}")
print(f"  - Val loss improved: {'YES' if meta1['best_val_loss'] < meta2['best_val_loss'] else 'NO'}")
print(f"  - Training efficiency: {meta2['tokens_seen']/meta1['tokens_seen']*100:.0f}% of full training")

print("\n" + "="*80)
