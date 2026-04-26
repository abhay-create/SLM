"""
100M Parameter Small Language Model Architecture
Using RMSNorm, SwiGLU, and RoPE for efficient training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len: int):
        return (
            self.cos_cached[:seq_len, :].to(x.device),
            self.sin_cached[:seq_len, :].to(x.device)
        )


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function for FFN"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # down

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE"""
    def __init__(self, dim: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.o_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    def __init__(self, dim: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Pre-norm attention with residual
        x = x + self.dropout(self.attn(self.attn_norm(x), mask))
        # Pre-norm FFN with residual
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class SLM100M(nn.Module):
    """100M Parameter Small Language Model"""
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 640,
        n_layers: int = 10,
        n_heads: int = 10,
        d_ff: int = 2560,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm and output projection
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(
            1, 1, seq_len, seq_len
        )
        
        # Embed tokens
        x = self.tok_emb(input_ids)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def create_model(config: dict) -> SLM100M:
    """Factory function to create model from config"""
    return SLM100M(
        vocab_size=config.get('vocab_size', 50257),
        d_model=config.get('d_model', 640),
        n_layers=config.get('n_layers', 10),
        n_heads=config.get('n_heads', 10),
        d_ff=config.get('d_ff', 2560),
        max_seq_len=config.get('max_seq_len', 512),
        dropout=config.get('dropout', 0.1),
    )


if __name__ == "__main__":
    # Test model creation
    model = SLM100M()
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    dummy_input = torch.randint(0, 50257, (2, 128))
    dummy_target = torch.randint(0, 50257, (2, 128))
    
    logits, loss = model(dummy_input, dummy_target)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")