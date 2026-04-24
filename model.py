"""
Decoder-only SLM — ~47-51M params depending on tokenizer + pos config.
Supports: pos_type = "learnable" | "rope", KV cache for inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class SLMConfig:
    vocab_size   : int   = 50000
    d_model      : int   = 512
    n_layers     : int   = 6
    n_heads      : int   = 8
    d_ff         : int   = 2048
    ctx_len      : int   = 512
    dropout      : float = 0.0
    bias         : bool  = False
    weight_tying : bool  = True
    pos_type     : str   = "learnable"   # "learnable" | "rope"
    rope_base    : int   = 10000
    head_dim     : int   = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.pos_type in ("learnable", "rope")
        self.head_dim = self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()) * self.scale


class RotaryEmbedding(nn.Module):
    """RoPE — no learnable parameters, shared across all layers."""
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, offset: int = 0):
        T   = q.shape[2]
        cos = self.cos_cache[offset: offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset: offset + T].unsqueeze(0).unsqueeze(0)
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)


class SwiGLU(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SLMConfig, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model  = cfg.d_model
        self.use_rope = cfg.pos_type == "rope"
        self.rope     = rope
        self.attn_drop = cfg.dropout
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)
        def reshape(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        if self.use_rope and self.rope is not None:
            offset = kv_cache[0].shape[2] if kv_cache is not None else 0
            q, k   = self.rope(q, k, offset=offset)

        new_cache = None
        if use_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            new_cache = (k, v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=not use_cache,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)), new_cache


class DecoderBlock(nn.Module):
    def __init__(self, cfg: SLMConfig, rope=None):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg, rope=rope)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLU(cfg)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class SLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model) if cfg.pos_type == "learnable" else None
        self.rope    = RotaryEmbedding(cfg.head_dim, cfg.ctx_len, cfg.rope_base) if cfg.pos_type == "rope" else None
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers  = nn.ModuleList([DecoderBlock(cfg, rope=self.rope) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        for name, p in m.named_parameters():
            if name in ("out_proj.weight", "w_down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.ctx_len
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.emb_drop(x)
        for layer in self.layers:
            x, _ = layer(x, kv_cache=None, use_cache=False)
        logits = self.lm_head(self.norm_out(x))
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) \
                 if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=1.0, top_k=None, use_cache=True):
        kv_caches = [None] * self.cfg.n_layers

        if use_cache:
            B, T_prompt = idx.shape
            x = self.tok_emb(idx)
            if self.pos_emb is not None:
                x = x + self.pos_emb(torch.arange(T_prompt, device=idx.device))
            for i, layer in enumerate(self.layers):
                x, kv_caches[i] = layer(x, kv_cache=None, use_cache=True)
            logits = self.lm_head(self.norm_out(x))[:, -1, :]
            generated = []
            for step in range(max_new):
                next_id = self._sample(logits, temperature, top_k)
                generated.append(next_id)
                x_new = self.tok_emb(next_id)
                if self.pos_emb is not None:
                    x_new = x_new + self.pos_emb(torch.tensor([T_prompt + step], device=idx.device))
                for i, layer in enumerate(self.layers):
                    x_new, kv_caches[i] = layer(x_new, kv_cache=kv_caches[i], use_cache=True)
                logits = self.lm_head(self.norm_out(x_new))[:, -1, :]
            return torch.cat([idx] + generated, dim=1)
        else:
            for _ in range(max_new):
                idx_c   = idx[:, -self.cfg.ctx_len:]
                logits, _ = self(idx_c)
                next_id = self._sample(logits[:, -1, :], temperature, top_k)
                idx     = torch.cat([idx, next_id], dim=1)
            return idx

    @staticmethod
    def _sample(logits, temperature, top_k):
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        return torch.multinomial(F.softmax(logits, dim=-1), 1)

    def num_params(self, exclude_embeddings=False):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb: n -= self.pos_emb.weight.numel()
        return n

    def param_breakdown(self):
        def c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "tok_emb" : c(self.tok_emb),
            "pos_emb" : c(self.pos_emb) if self.pos_emb else 0,
            "rope"    : 0,
            "layers"  : c(self.layers),
            "norm_out": c(self.norm_out),
            "lm_head" : 0 if self.cfg.weight_tying else c(self.lm_head),
            "total"   : self.num_params(),
        }


if __name__ == "__main__":
    import math
    for pos in ("learnable", "rope"):
        cfg   = SLMConfig(vocab_size=50000, pos_type=pos)
        model = SLM(cfg)
        print(f"\npos={pos}")
        for k, v in model.param_breakdown().items():
            print(f"  {k:<10}: {v/1e6:.2f}M")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        x = torch.randint(0, 50000, (2, 64), device=device)
        y = torch.randint(0, 50000, (2, 64), device=device)
        _, loss = model(x, y)
        print(f"  loss={loss.item():.3f} (expect ~{math.log(50000):.2f})")
        p = torch.randint(0, 50000, (1, 8), device=device)
        print(f"  cached gen shape   : {tuple(model.generate(p, max_new=4, use_cache=True).shape)}")
        print(f"  uncached gen shape : {tuple(model.generate(p, max_new=4, use_cache=False).shape)}")
