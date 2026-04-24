"""
Comprehensive generation + perplexity evaluation across all checkpoints.
Tests both TS-style and WP-style prompts to see style transfer effects.
"""
import torch, os, pickle, sys
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0, '.')
from evaluate_curriculum import load_model_from_checkpoint, evaluate_on_chunks, generate_sample

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizers/tokenizer_corpus.json")
vocab_size = tokenizer.get_vocab_size()

# Load val chunks
with open("cache/val_tinystories_seq256.pkl", "rb") as f:
    ts_val = pickle.load(f)
print(f"TS val chunks: {len(ts_val)}")

wp_val_path = "cache/val_writingprompts_seq384.pkl"
if os.path.exists(wp_val_path):
    with open(wp_val_path, "rb") as f:
        wp_val = pickle.load(f)
    print(f"WP val chunks: {len(wp_val)}")
else:
    wp_val = []
    print("No WP val chunks found")

checkpoints = [
    ("Baseline (50M, 6L)", "checkpoints/stagefull_dataset_curriculum_adaptive_best.pt"),
    ("Stage A (58M, 9L)", "checkpoints/stage_A_best.pt"),
    ("Stage B (71M, 12L)", "checkpoints/stage_B_best.pt"),
    ("Stage C (99M, 12L)", "checkpoints/stage_C_best.pt"),
]

# Diverse prompts covering both domains
prompts = {
    "TinyStories-style": [
        "Once upon a time, there was a little",
        "The cat sat on the mat and",
        "Mom said to the boy,",
        "The dog ran to the park because",
    ],
    "WritingPrompts-style": [
        "The year is 2157. Humanity has",
        "She opened the door to find",
        "Death appeared before him and said",
        "The last human on Earth sat alone",
    ],
    "Neutral": [
        "It was a dark and stormy night",
        "The old man looked at the sky",
        "I never thought this would happen",
    ],
}

print(f"\n{'=' * 80}")
print(f"  COMPREHENSIVE EVALUATION ACROSS ALL CHECKPOINTS")
print(f"{'=' * 80}")

for ckpt_name, ckpt_path in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f"\n  SKIP: {ckpt_name} not found")
        continue
    
    print(f"\n{'─' * 80}")
    print(f"  {ckpt_name}")
    print(f"{'─' * 80}")
    
    model, cfg, info = load_model_from_checkpoint(ckpt_path, device)
    print(f"  Config: n_layers={cfg.n_layers}, d_ff={cfg.d_ff}, ctx={cfg.ctx_len}")
    print(f"  Params: {model.num_params()/1e6:.1f}M")
    print(f"  Best val: {info.get('best_val_loss', '?')}")
    
    # Evaluate on TS val
    ts_loss, ts_ppl = evaluate_on_chunks(model, ts_val, device, vocab_size, 500)
    print(f"\n  TinyStories val:    loss={ts_loss:.4f}  PPL={ts_ppl:.2f}")
    
    # Evaluate on WP val
    if wp_val:
        wp_loss, wp_ppl = evaluate_on_chunks(model, wp_val, device, vocab_size, 500)
        print(f"  WritingPrompts val: loss={wp_loss:.4f}  PPL={wp_ppl:.2f}")
    
    # Generation tests
    for category, prompt_list in prompts.items():
        print(f"\n  [{category}]")
        for prompt in prompt_list:
            text = generate_sample(model, tokenizer, prompt, device, max_tokens=100, temperature=0.7, top_k=30)
            text = text.replace('\n', ' ').strip()
            if len(text) > 250:
                text = text[:250] + "..."
            print(f"    > \"{prompt}\"")
            print(f"      {text}")
            print()
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'=' * 80}")
print(f"  ALL EVALUATIONS COMPLETE")
print(f"{'=' * 80}")
