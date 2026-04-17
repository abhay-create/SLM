"""
test_with_seed.py
Run inference with fixed seed to check if repetition is deterministic
"""
import os
import sys
import torch
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from src.model import SLM

CHECKPOINT_PATH = "checkpoints/stage1_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    "Lions are large carnivorous",
    "The Earth is the third planet from",
    "Water is made of hydrogen and",
    "William Shakespeare was an English"
]


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer():
    """Load checkpoint and tokenizer."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model_config = ckpt["config"]
    
    model = SLM(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, use_cache=True):
    """Generate text from a prompt."""
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new=100,
            temperature=1.0,
            top_k=200,
            use_cache=use_cache
        )
    
    return tokenizer.decode(output_ids[0].tolist())


def main():
    print("="*80)
    print("Testing with FIXED SEED (seed=42)")
    print("="*80)
    print()
    
    # Run 3 times with same seed to verify determinism
    for run_num in range(1, 4):
        print(f"\n[RUN {run_num}]")
        print("-"*80)
        
        set_seed(42)  # Always use same seed
        model, tokenizer = load_model_and_tokenizer()
        
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\n[Prompt {i}] {repr(prompt)}")
            generated = generate_text(model, tokenizer, prompt, use_cache=True)
            print(generated)
            
            if run_num == 1 and i == 1:
                # Check for repetition patterns in first prompt
                token_list = tokenizer.encode(generated).ids
                print(f"\n[ANALYSIS] Token list (first 30): {token_list[:30]}")
                
                # Check for repeated sequences
                for window_size in [2, 3, 5]:
                    max_repeat = 0
                    max_repeat_token = None
                    token_counts = {}
                    for j in range(len(token_list) - window_size):
                        window = tuple(token_list[j:j+window_size])
                        token_counts[window] = token_counts.get(window, 0) + 1
                        if token_counts[window] > max_repeat:
                            max_repeat = token_counts[window]
                            max_repeat_token = window
                    
                    if max_repeat > 1:
                        decoded = tokenizer.decode(max_repeat_token)
                        print(f"  Window size {window_size}: '{decoded}' repeated {max_repeat}x")


if __name__ == "__main__":
    main()
