"""
analyze_logit_distribution.py
Analyze the actual logit ranges and distributions from the stage1 model
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from src.model import SLM

CHECKPOINT_PATH = "checkpoints/stage1_best.pt"
TOKENIZER_PATH  = "tokenizers/tokenizer_corpus.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    "Lions are large carnivorous",
    "The Earth is the third planet from",
]


def load_model_and_tokenizer():
    """Load checkpoint and tokenizer."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model_config = ckpt["config"]
    
    # Create model and load weights
    model = SLM(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    return model, tokenizer


def analyze_logits(model, tokenizer, prompt):
    """Analyze logit distribution."""
    print(f"\n{'='*80}")
    print(f"Prompt: {repr(prompt)}")
    print(f"{'='*80}")
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=DEVICE)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        logits_current = logits[:, -1, :]  # shape: [1, 40000]
        
        print(f"\n[LOGIT STATISTICS]")
        print(f"  Mean: {logits_current.mean().item():.6f}")
        print(f"  Std: {logits_current.std().item():.6f}")
        print(f"  Min: {logits_current.min().item():.6f}")
        print(f"  Max: {logits_current.max().item():.6f}")
        print(f"  Median: {logits_current.median().item():.6f}")
        print()
        
        # Show percentiles
        print(f"[PERCENTILES]")
        for p in [50, 75, 90, 95, 99]:
            val = torch.quantile(logits_current.flatten(), p / 100.0).item()
            print(f"  {p}th percentile: {val:.6f}")
        print()
        
        # Simulate top-k filtering with k=200
        top_k = 200
        v, indices = torch.topk(logits_current, min(top_k, logits_current.size(-1)))
        threshold = v[:, [-1]].item()
        
        print(f"[TOP-{top_k} FILTERING]")
        print(f"  Top-1 logit: {v[0, 0].item():.6f}")
        print(f"  Top-10 logit: {v[0, 9].item():.6f}")
        print(f"  Top-50 logit: {v[0, 49].item():.6f}")
        print(f"  Top-200 logit (threshold): {threshold:.6f}")
        print(f"  Bottom of distribution (0%): {logits_current.min().item():.6f}")
        print()
        
        # Check position of specific tokens
        print(f"[SPECIFIC TOKEN LOGITS]")
        for token_id, token_name in [(15, "comma"), (17, "period"), (238, " and"), (3145, " related")]:
            logit_val = logits_current[0, token_id].item()
            rank_val = (logits_current[0] > logit_val).sum().item() + 1
            below_threshold = logit_val < threshold
            print(f"  {token_name:15} [ID {token_id:5}]: logit={logit_val:8.6f}, rank={rank_val:5}, below_threshold={below_threshold}")
        print()
        
        #Now check what multinomial does with top-k filtering
        temperature = 1.0
        logits_scaled = logits_current / temperature
        mask = logits_scaled < v[:, [-1]]
        logits_scaled[mask] = float("-inf")
        probs = F.softmax(logits_scaled, dim=-1)
        
        print(f"[AFTER TOP-K + SOFTMAX]")
        for token_id, token_name in [(15, "comma"), (17, "period"), (238, " and"), (3145, " related")]:
            prob_val = probs[0, token_id].item()
            print(f"  {token_name:15}: prob={prob_val:.10f}")
        print()
        
        # Top-5 actual probabilities
        top5_probs, top5_ids = torch.topk(probs, 5, dim=-1)
        top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_ids[0]]
        
        print(f"[TOP-5 PROBABILITIES]")
        for i, (token_str, prob_val, token_id) in enumerate(zip(top5_tokens, top5_probs[0].tolist(), top5_ids[0].tolist())):
            print(f"  {i+1}. '{token_str:15}' [{token_id:5}]: {prob_val:.6f}")


def main():
    print("="*80)
    print("Logit Distribution Analysis")
    print("="*80)
    
    model, tokenizer = load_model_and_tokenizer()
    
    for prompt in TEST_PROMPTS:
        analyze_logits(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
