"""
Test inference on legal model (stage 0 best checkpoint)
"""

import os
import torch
from tokenizers import Tokenizer
from src.model import SLM

# Configuration
CHECKPOINT = "checkpoints_legal/stage0_best.pt"
TOKENIZER_PATH = "tokenizers/tokenizer_corpus.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Legal prompts to test
PROMPTS = [
    "This agreement shall",
    "Indemnification:",
    "Confidential information",
    "The party agrees to",
    "Limitation of liability",
    "Termination:",
    "Governing law",
    "Payment terms",
    "Warranties and representations",
    "Non-disclosure",
]

def load_model(checkpoint_path, tokenizer_path, device):
    """Load checkpoint and tokenizer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = SLM(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"✓ Model loaded | Params: {model.num_params()/1e6:.1f}M")
    print(f"✓ Tokenizer loaded | Vocab size: {tokenizer.get_vocab_size()}")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new=100, temperature=0.7, top_k=50, device="cuda"):
    """Generate text from prompt."""
    ids = tokenizer.encode(prompt).ids
    inp = torch.tensor([ids], device=device)
    
    with torch.no_grad():
        out = model.generate(
            inp, 
            max_new=max_new, 
            temperature=temperature, 
            top_k=top_k, 
            use_cache=True
        )
    
    return tokenizer.decode(out[0].tolist())

def main():
    print("=" * 80)
    print("Legal SLM Inference Test")
    print("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer = load_model(CHECKPOINT, TOKENIZER_PATH, DEVICE)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Generation params: max_new=100, temperature=0.7, top_k=50\n")
    
    # Generate from prompts
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i:2d}] Prompt: '{prompt}'")
        print("-" * 80)
        
        result = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_new=100, 
            temperature=0.7, 
            top_k=50,
            device=DEVICE
        )
        
        print(result)
        print()
    
    # Comparison: Different temperatures
    print("\n" + "=" * 80)
    print("Temperature Comparison (same prompt)")
    print("=" * 80)
    
    test_prompt = "The party agrees to"
    for temp in [0.3, 0.7, 1.0]:
        print(f"\n[Temperature: {temp}]")
        print("-" * 80)
        result = generate_text(
            model, 
            tokenizer, 
            test_prompt, 
            max_new=80, 
            temperature=temp, 
            top_k=50,
            device=DEVICE
        )
        print(result)
    
    print("\n" + "=" * 80)
    print("✓ Inference test complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
