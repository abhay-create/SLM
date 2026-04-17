"""
test_stage0_inference.py — Text generation inference test for SLM checkpoints.

Loads a trained checkpoint and generates text from test prompts.
Results are saved to a timestamped JSON file.
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from src.model import SLM

# ─── Configuration ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/stage2_best.pt"
TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test prompts
TEST_PROMPTS = [
    "Plants grow when",
    "Once upon a time",
    "The universe is",
]

# Inference hyperparameters to test
MAX_NEW_TOKENS_VALUES = [20, 40, 60]
TEMPERATURE_VALUES = [0.5, 0.7, 1.0]
TOP_K_VALUES = [10, 30, 50]

# Results directory
RESULTS_DIR = Path("inference_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Model Loading ────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> Tuple[object, dict]:
    """Load model checkpoint and return config + state dict."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[load] Loading checkpoint: {checkpoint_path.name}")
    
    # Load checkpoint (weights_only=False to support SLMConfig objects)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model_state = ckpt["model_state"]
    
    return config, model_state


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load tokenizer from JSON file."""
    tokenizer_path = Path(tokenizer_path)
    
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    print(f"[load] Loading tokenizer: {tokenizer_path.name}")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"       Vocab size: {vocab_size}")
    
    return tokenizer


def setup_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: str,
) -> Tuple[SLM, Tokenizer]:
    """Load both model and tokenizer, and create model instance."""
    # Load checkpoint config and weights
    config, model_state = load_checkpoint(checkpoint_path)
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Create model and load weights
    print(f"[model] Creating SLM with config...")
    model = SLM(config).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"[model] Device: {DEVICE}")
    print(f"[model] Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
    
    return model, tokenizer


# ─── Text Generation ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: SLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Generate text from a prompt using the model."""
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids.copy()
    input_ids = torch.tensor([token_ids], device=DEVICE)
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
            next_logits.fill_(float('-inf'))
            next_logits[top_k_indices] = top_k_logits
        
        # Sample next token
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Append to sequence
        token_ids.append(next_token)
        input_ids = torch.tensor([token_ids], device=DEVICE)
    
    # Decode full sequence
    generated_text = tokenizer.decode(token_ids)
    return generated_text


# ─── Results Saving ───────────────────────────────────────────────────────────

def save_results(results: List[Dict], filename: str = None) -> Path:
    """Save inference results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_results_{timestamp}.json"
    
    filepath = RESULTS_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")
    return filepath


# ─── Main Test ────────────────────────────────────────────────────────────────

def run_inference_test(
    checkpoint_path: str,
    tokenizer_path: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    model=None,
    tokenizer=None,
) -> List[Dict]:
    """Run inference test on multiple prompts and return results."""
    # Setup model and tokenizer on first call
    if model is None or tokenizer is None:
        print("=" * 80)
        print("  SLM INFERENCE TEST")
        print("=" * 80 + "\n")
        model, tokenizer = setup_model_and_tokenizer(checkpoint_path, tokenizer_path)
    
    # Test generation
    print("=" * 80)
    print(f"Generating text (temp={temperature}, top_k={top_k}, max_len={max_new_tokens})")
    print("=" * 80 + "\n")
    
    test_results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[Prompt {i}] {prompt!r}")
        print("-" * 80)
        
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        
        print(generated)
        print()
        
        # Save result
        test_results.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "generated_text": generated,
        })
    
    print("=" * 80)
    print("✓ Inference test complete!")
    print("=" * 80)
    
    return test_results


# ─── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print(f"  Testing {len(MAX_NEW_TOKENS_VALUES) * len(TEMPERATURE_VALUES) * len(TOP_K_VALUES)} hyperparameter combinations")
        print("=" * 80 + "\n")
        
        combo_num = 0
        total_combos = len(MAX_NEW_TOKENS_VALUES) * len(TEMPERATURE_VALUES) * len(TOP_K_VALUES)
        all_results = []
        
        # Load model once and reuse
        model, tokenizer = setup_model_and_tokenizer(CHECKPOINT_PATH, TOKENIZER_PATH)
        
        for max_tok in MAX_NEW_TOKENS_VALUES:
            for temp in TEMPERATURE_VALUES:
                for tk in TOP_K_VALUES:
                    combo_num += 1
                    print(f"\n[{combo_num}/{total_combos}] max_tokens={max_tok}, temperature={temp}, top_k={tk}\n")
                    
                    results = run_inference_test(
                        checkpoint_path=CHECKPOINT_PATH,
                        tokenizer_path=TOKENIZER_PATH,
                        max_new_tokens=max_tok,
                        temperature=temp,
                        top_k=tk,
                        model=model,
                        tokenizer=tokenizer,
                    )
                    all_results.extend(results)
        
        # Save all results to file
        results_file = save_results(all_results)
        
        print("\n" + "=" * 80)
        print(f"✅ All {total_combos} test combinations completed!")
        print(f"📊 Total outputs: {len(all_results)}")
        print(f"📁 Results saved to: {results_file}")
        print("=" * 80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
