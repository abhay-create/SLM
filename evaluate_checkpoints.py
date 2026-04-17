"""
Evaluate Stage 0 checkpoints on perplexity, language modeling, and inference metrics.
Compares: stage0_best_1st_trial.pt vs stage0_best_1stfull.pt
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, List

from src.model import SLM, SLMConfig
from src.dataset import StreamingStageDataset, load_all_val_sets
from tokenizers import Tokenizer


def load_checkpoint(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[SLM, dict]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer config if not stored
    cfg = SLMConfig()
    
    # Create model and load state
    model = SLM(cfg).to(device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model, {"config": cfg, "device": device}


@torch.no_grad()
def compute_perplexity(model: SLM, dataloader, device: str, max_batches: int = 100) -> Dict[str, float]:
    """
    Compute perplexity on validation set.
    Perplexity = exp(loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Accumulate loss
        batch_size, seq_len = y.shape
        total_loss += loss.item() * (batch_size * seq_len)
        total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "tokens_evaluated": total_tokens,
        "batches_evaluated": min(i + 1, max_batches)
    }


@torch.no_grad()
def compute_lm_score(model: SLM, dataloader, device: str, max_batches: int = 100) -> Dict[str, float]:
    """
    Compute language modeling metrics:
    - Accuracy: % of correct next-token predictions
    - Confidence: avg softmax probability of true tokens
    """
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_confidence = 0.0
    
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, _ = model(x, y)  # logits: (B, T, vocab_size)
        
        # Get predictions
        predictions = logits.argmax(dim=-1)  # (B, T)
        
        # Mask for valid positions
        mask = y != -100  # Assuming -100 is padding/ignored
        
        # Compute accuracy
        correct = (predictions == y) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        
        # Compute confidence (probability of true token)
        probs = F.softmax(logits, dim=-1)  # (B, T, vocab_size)
        true_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
        total_confidence += (true_probs[mask].log().sum().item())
    
    accuracy = total_correct / max(total_tokens, 1) if total_tokens > 0 else 0.0
    avg_log_prob = total_confidence / max(total_tokens, 1) if total_tokens > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "avg_log_prob": avg_log_prob,
        "tokens_evaluated": total_tokens,
        "batches_evaluated": min(i + 1, max_batches)
    }


@torch.no_grad()
def inference(model: SLM, tokenizer: Tokenizer, prompt: str, max_length: int = 100, 
              device: str = "cuda" if torch.cuda.is_available() else "cpu") -> str:
    """
    Generate text from prompt (inference test).
    """
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt).ids
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    for _ in range(max_length):
        with torch.no_grad():
            if tokens.shape[1] > model.cfg.ctx_len:
                # Keep only last ctx_len tokens
                tokens = tokens[:, -model.cfg.ctx_len:]
            
            logits, _ = model(tokens, targets=None)
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode
    generated_ids = tokens[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def evaluate_model(checkpoint_path: str, tokenizer_path: str, stage: int = 0) -> Dict:
    """
    Comprehensive evaluation of a checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"Evaluating: {Path(checkpoint_path).name}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    # Load model
    model, meta = load_checkpoint(checkpoint_path, device)
    cfg = meta["config"]
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load validation dataloaders
    print("Loading validation datasets...")
    val_loaders = load_all_val_sets(
        tokenizer_path=tokenizer_path,
        batch_size=32,
        seq_len=cfg.ctx_len,
        num_workers=0,
        pin_memory=True
    )
    
    results = {
        "checkpoint": Path(checkpoint_path).name,
        "stage": stage,
        "device": device,
        "config": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "vocab_size": cfg.vocab_size,
        }
    }
    
    # Evaluate on each validation set
    for val_name, val_loader in val_loaders.items():
        print(f"\nEvaluating on {val_name}...")
        
        # Perplexity
        perp_metrics = compute_perplexity(model, val_loader, device, max_batches=50)
        print(f"  Loss: {perp_metrics['avg_loss']:.4f}")
        print(f"  Perplexity: {perp_metrics['perplexity']:.2f}")
        
        # Language Modeling Score
        lm_metrics = compute_lm_score(model, val_loader, device, max_batches=50)
        print(f"  Accuracy: {lm_metrics['accuracy']*100:.2f}%")
        print(f"  Avg Log Prob: {lm_metrics['avg_log_prob']:.4f}")
        
        results[val_name] = {
            **perp_metrics,
            **lm_metrics
        }
    
    # Inference test
    print(f"\nInference Test:")
    test_prompts = [
        "Once upon a time",
        "The cat sat on",
        "Sarah walked to the"
    ]
    
    inference_results = []
    for prompt in test_prompts:
        print(f"  Prompt: '{prompt}'")
        generated = inference(model, tokenizer, prompt, max_length=50, device=device)
        print(f"  Generated: '{generated[:100]}...'")
        inference_results.append({
            "prompt": prompt,
            "generated": generated[:200]
        })
    
    results["inference_samples"] = inference_results
    
    return results


def main():
    """Evaluate both checkpoints."""
    
    # Paths
    checkpoint_dir = Path("/home/user20/NLP/slm/checkpoints")
    tokenizer_path = "/home/user20/NLP/slm/tokenizers/tokenizer_50k.json"
    
    # Checkpoints to evaluate
    checkpoints = [
        ("stage0_best_1st_trial.pt", "First Trial (Early Exit - 107.7M tokens)"),
        ("stage0_best_1stfull.pt", "Full Training (200M tokens)"),
    ]
    
    all_results = {}
    
    for checkpoint_name, description in checkpoints:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            print(f"\n\n{'#'*70}")
            print(f"# {description}")
            print(f"# {checkpoint_path}")
            print(f"{'#'*70}")
            
            try:
                results = evaluate_model(str(checkpoint_path), tokenizer_path, stage=0)
                all_results[checkpoint_name] = results
            except Exception as e:
                print(f"ERROR evaluating {checkpoint_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Save results
    output_file = checkpoint_dir.parent / "EVALUATION_RESULTS.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if len(all_results) >= 2:
        checkpoint_names = list(all_results.keys())
        for val_name in ["val_s0", "val_s1", "val_s2"]:
            print(f"\n{val_name} (TinyStories val)" if val_name == "val_s0" else f"\n{val_name}")
            for ckpt_name in checkpoint_names:
                if val_name in all_results[ckpt_name]:
                    metrics = all_results[ckpt_name][val_name]
                    print(f"\n  {ckpt_name}:")
                    print(f"    Loss: {metrics.get('avg_loss', 'N/A'):.4f}")
                    print(f"    Perplexity: {metrics.get('perplexity', 'N/A'):.2f}")
                    print(f"    Accuracy: {metrics.get('accuracy', 'N/A')*100:.2f}%")


if __name__ == "__main__":
    main()
