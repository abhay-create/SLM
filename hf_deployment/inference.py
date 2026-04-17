"""
inference.py — Simple inference script for SLM models on Hugging Face Hub.

Usage:
    python inference.py \
        --checkpoint stage2_best.pt \
        --tokenizer tokenizer_50k.json \
        --prompt "Once upon a time" \
        --max_len 100
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SLM, SLMConfig
from src.tokenizer import Tokenizer


def load_model(checkpoint_path: str, pos_type: str = "rope", device: str = "cuda") -> SLM:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    config = ckpt["config"]
    
    # Create model with loaded config
    model = SLM(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Config: {config}")
    return model


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load tokenizer."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"✓ Loaded tokenizer from {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


@torch.no_grad()
def generate(
    model: SLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_len: int = 100,
    temperature: float = 1.0,
    top_k: int = 40,
    device: str = "cuda",
) -> str:
    """
    Generate text from prompt.
    
    Args:
        model: SLM model
        tokenizer: Tokenizer
        prompt: Input prompt text
        max_len: Maximum tokens to generate
        temperature: Sampling temperature (>1 = more random, <1 = more deterministic)
        top_k: Keep only top-k tokens for sampling
        device: Device to run on
    
    Returns:
        Generated text
    """
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    input_ids = torch.tensor([token_ids], device=device)
    
    print(f"\n[Input] {prompt}")
    generated_ids = token_ids.copy()
    
    # Generation loop
    for step in range(max_len):
        # Forward pass
        logits, _ = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated_ids.append(next_token)
        input_ids = torch.tensor([generated_ids], device=device)
        
        # Stop on EOS (if tokenizer has one)
        if next_token == tokenizer.get_vocab_size() - 1:  # Assuming last token is EOS
            break
    
    # Decode
    full_text = tokenizer.decode(generated_ids)
    return full_text


@torch.no_grad()
def get_embeddings(
    model: SLM,
    tokenizer: Tokenizer,
    text: str,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Get last-layer embeddings for input text.
    
    Returns:
        Tensor of shape (seq_len, d_model)
    """
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids], device=device)
    
    # Forward pass (model outputs are (logits, hidden_states))
    # We need to modify model to return embeddings - for now just logits
    logits, loss = model(input_ids)
    
    return logits[0]  # Return last layer logits as proxy


def main():
    parser = argparse.ArgumentParser(description="SLM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max_len", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load model and tokenizer
    model = load_model(args.checkpoint, device=args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Generate
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    
    print(f"\n[Output]\n{output}\n")


if __name__ == "__main__":
    main()
