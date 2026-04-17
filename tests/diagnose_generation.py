"""
diagnose_generation.py
Analyze why Stage 1 model generates repetitive text.
Checks logits distribution, token probabilities, and checkpoint integrity.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from src.model import SLM

CHECKPOINT_PATH = "checkpoints/stage1_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Test prompts
TEST_PROMPTS = [
    "Lions are large carnivorous",
    "The Earth is the third planet from",
    "Water is made of hydrogen and",
    "William Shakespeare was an English"
]


def load_model_and_tokenizer():
    """Load checkpoint and tokenizer."""
    print(f"[LOG] Loading checkpoint from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] {CHECKPOINT_PATH} not found!")
        sys.exit(1)
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model_config = ckpt["config"]
    
    print(f"[LOG] Model config: {model_config}")
    print(f"[LOG] Checkpoint keys: {ckpt.keys()}")
    print(f"[LOG] Device: {DEVICE}")
    
    # Check if checkpoint has training_step info
    training_step = ckpt.get("training_step", "unknown")
    tokens_seen = ckpt.get("tokens_seen", "unknown")
    print(f"[LOG] Training step: {training_step}, Tokens seen: {tokens_seen}")
    print()
    
    # Create model and load weights
    model = SLM(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Check if model weights are non-zero
    total_norm = sum(p.norm().item() for p in model.parameters() if p.requires_grad)
    print(f"[LOG] Model weight norm: {total_norm:.4f}")
    print(f"[LOG] Loading tokenizer from {TOKENIZER_PATH}...")
    
    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ERROR] {TOKENIZER_PATH} not found!")
        sys.exit(1)
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[LOG] Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print()
    
    return model, tokenizer, model_config


def diagnose_logits(model, tokenizer, prompt):
    """Generate text with logits analysis."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {repr(prompt)}")
    print(f"{'='*80}")
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    
    print(f"[INFO] Prompt tokens: {encoded.ids}")
    print(f"[INFO] Prompt length: {len(encoded.ids)}")
    print()
    
    # Generate with logits analysis using model.generate
    max_new = 50
    temperature = 1.0
    top_k = 200
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            use_cache=True
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(f"[GENERATED TEXT ({max_new} tokens)]")
    print(generated_text)
    print()
    
    # Now analyze the logits for the first few steps
    print(f"[LOGITS ANALYSIS]")
    with torch.no_grad():
        logits, _ = model(input_ids)
        
        # Look at first 5 steps
        for step in range(min(5, max_new)):
            logits_current = logits[:, -1, :]
            logits_normed = logits_current / temperature
            
            # Top-k filtering
            if top_k:
                v, _ = torch.topk(logits_normed, min(top_k, logits_normed.size(-1)))
                logits_filtered = logits_normed.clone()
                logits_filtered[logits_filtered < v[:, [-1]]] = float("-inf")
            else:
                logits_filtered = logits_normed
            
            # Get probabilities
            probs = F.softmax(logits_filtered, dim=-1)
            
            # Top 5 tokens
            top5_probs, top5_ids = torch.topk(probs, 5, dim=-1)
            top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_ids[0]]
            top5_id_values = top5_ids[0].tolist()
            top5_prob_values = top5_probs[0].tolist()
            
            # Get actual next token
            next_actual_token = output_ids[0, len(input_ids[0]) + step].item()
            next_actual_str = tokenizer.decode([next_actual_token])
            next_actual_prob = probs[0, next_actual_token].item()
            
            print(f"[Step {step+1}] Next token in output: '{next_actual_str}' (ID: {next_actual_token}, prob: {next_actual_prob:.6f})")
            print(f"  Top-5 probabilities:")
            for i, (token_str, prob_val, token_id) in enumerate(zip(top5_tokens, top5_prob_values, top5_id_values)):
                print(f"    {i+1}. '{token_str}' [{token_id}]: {prob_val:.6f}")
            
            # Check for mode collapse
            max_prob = probs.max().item()
            second_prob = torch.topk(probs, 2, dim=-1)[0][0, 1].item()
            prob_ratio = max_prob / (second_prob + 1e-10)
            
            if prob_ratio > 5:
                print(f"  ⚠️  CONCENTRATION: {prob_ratio:.1f}x")
            
            print()
            
            # For next step, we need the actual logits after sampling the real next token
            # This requires re-running forward pass - skip for efficiency, just show one step
            break


def main():
    print("="*80)
    print("Stage 1 Generation Diagnostics")
    print("="*80)
    print()
    
    model, tokenizer, config = load_model_and_tokenizer()
    
    # Test each prompt
    for prompt in TEST_PROMPTS[:2]:  # Test just first 2 for detailed output
        diagnose_logits(model, tokenizer, prompt)
        print("\n")


if __name__ == "__main__":
    main()
