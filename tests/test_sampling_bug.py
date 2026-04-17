"""
test_sampling_bug.py
Test the _sample function to understand the bug
"""
import torch
import torch.nn.functional as F


def _sample_original(logits, temperature, top_k):
    """Original _sample function from model.py"""
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def test_sampling():
    """Test sampling with a simple example"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate logits: shape [1, 40000]
    # Create a logits tensor where:
    # - Position 15 (comma) has logit ~0.19
    # - Position 17 (period) has logit ~0.19
    # - Other high positions have lower values
    
    # Create realistic logits
    batch_logits = torch.randn(1, 40000, device=device)
    
    # Set some specific values (manually set high logits for testing)
    batch_logits[0, 15] = 0.5  # comma: high
    batch_logits[0, 17] = 0.48  # period: high
    batch_logits[0, 238] = 0.3  # and: medium
    batch_logits[0, 3145] = -5.0  # related: VERY low
    
    print(f"Original logits shapes: {batch_logits.shape}")
    print(f"Logits[0, 15] (comma): {batch_logits[0, 15].item():.6f}")
    print(f"Logits[0, 17] (period): {batch_logits[0, 17].item():.6f}")
    print(f"Logits[0, 238] (and): {batch_logits[0, 238].item():.6f}")
    print(f"Logits[0, 3145] (related): {batch_logits[0, 3145].item():.6f}")
    print()
    
    # Test with temperature = 1.0, top_k = 200
    temperature = 1.0
    top_k = 200
    
    # Copy logits for _sample
    logits_copy = batch_logits.clone()
    
    # Trace through _sample step by step
    print(f"[Step 1] Divide by temperature ({temperature})")
    logits_scaled = logits_copy / temperature
    print(f"  Logits[0, 15]: {logits_scaled[0, 15].item():.6f}")
    print(f"  Logits[0, 3145]: {logits_scaled[0, 3145].item():.6f}")
    print()
    
    print(f"[Step 2] Top-k filtering (k={top_k})")
    v, indices = torch.topk(logits_scaled, min(top_k, logits_scaled.size(-1)))
    print(f"  v shape: {v.shape}")
    print(f"  indices shape: {indices.shape}")
    print(f"  v[:, -1]: {v[:, [-1]]}")  # Threshold value
    print(f"  Indices in top-k: {indices[0, :10].tolist()}... (showing first 10)")
    print()
    
    print(f"[Step 3] Apply masking")
    mask = logits_scaled < v[:, [-1]]
    print(f"  Mask shape: {mask.shape}")
    print(f"  Number of masked positions: {mask.sum().item()}")
    print(f"  Position 15 masked? {mask[0, 15].item()}")
    print(f"  Position 3145 masked? {mask[0, 3145].item()}")
    logits_scaled[mask] = float("-inf")
    print()
    
    print(f"[Step 4] Softmax + multinomial")
    probs = F.softmax(logits_scaled, dim=-1)
    print(f"  Probs shape: {probs.shape}")
    print(f"  Prob[0, 15]: {probs[0, 15].item():.6f}")
    print(f"  Prob[0, 17]: {probs[0, 17].item():.6f}")
    print(f"  Prob[0, 238]: {probs[0, 238].item():.6f}")
    print(f"  Prob[0, 3145] (after -inf): {probs[0, 3145].item():.10f}")
    print()
    
    # Sample
    sampled = torch.multinomial(probs, 1)
    print(f"Sampled token: {sampled[0, 0].item()}")
    print()


if __name__ == "__main__":
    test_sampling()
