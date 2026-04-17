"""
test_without_cache.py
Test inference with use_cache=False to isolate KV cache bug
"""
import os
import sys
import torch

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


def load_model_and_tokenizer():
    """Load checkpoint and tokenizer."""
    print(f"[test] Loading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model_config = ckpt["config"]
    
    print(f"[test] Model config: {model_config}")
    print(f"[test] Device: {DEVICE}")
    
    # Create model and load weights
    model = SLM(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"[test] Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[test] Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
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
    
    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text


def main():
    print("=" * 80)
    print("Testing Stage 1 Inference: WITH vs WITHOUT KV Cache")
    print("=" * 80)
    print()
    
    model, tokenizer = load_model_and_tokenizer()
    print()
    
    print("=" * 80)
    print("GENERATION WITH KV CACHE (use_cache=True)")
    print("=" * 80)
    print()
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[Prompt {i}] {repr(prompt)}")
        print("-" * 80)
        
        generated = generate_text(model, tokenizer, prompt, use_cache=True)
        print(generated)
        print()
    
    print("\n\n")
    print("=" * 80)
    print("GENERATION WITHOUT KV CACHE (use_cache=False)")
    print("=" * 80)
    print()
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[Prompt {i}] {repr(prompt)}")
        print("-" * 80)
        
        generated = generate_text(model, tokenizer, prompt, use_cache=False)
        print(generated)
        print()


if __name__ == "__main__":
    main()
