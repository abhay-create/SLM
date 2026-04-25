"""
expand_model.py — Model expansion utilities for SLM.

Implements model growth utilities:
  - Depthwise warm-start expansion (layer cloning with symmetry-breaking noise)
  - FFN widening (zero-padded w_down for exact function preservation)
  - Context length extension (learnable pos_emb interpolation)
  - Expansion validation (output similarity checks)

Usage:
  # Expand depth: 6 → 9 layers
  python expand_model.py \
    --checkpoint checkpoints/stagefull_dataset_curriculum_adaptive_best.pt \
    --action expand_depth \
    --clone_layers 3,4,5 \
    --noise_std 0.01 \
    --output checkpoints/expanded_9L.pt

  # Widen FFN: d_ff 2048 → 3584
  python expand_model.py \
    --checkpoint checkpoints/expanded_12L.pt \
    --action expand_ffn \
    --new_d_ff 3584 \
    --output checkpoints/expanded_12L_ff3584.pt

  # Extend context: 256 → 384
  python expand_model.py \
    --checkpoint checkpoints/expanded_12L.pt \
    --action expand_context \
    --new_ctx_len 384 \
    --output checkpoints/expanded_12L_ctx384.pt
"""

import copy
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import SLM, SLMConfig, DecoderBlock


def _copy_non_layer_parameters(source: SLM, target: SLM, skip_pos_emb: bool = False):
    """Copy embeddings, output norm, and untied LM head parameters.

    Args:
        skip_pos_emb: If True, do not copy positional embeddings. Use this
            when the caller handles pos_emb separately (e.g. interpolation
            during context-length expansion where shapes differ).
    """
    target.tok_emb.load_state_dict(source.tok_emb.state_dict())
    if not skip_pos_emb and source.pos_emb is not None and target.pos_emb is not None:
        target.pos_emb.load_state_dict(source.pos_emb.state_dict())
    target.norm_out.load_state_dict(source.norm_out.state_dict())
    if not source.cfg.weight_tying:
        target.lm_head.load_state_dict(source.lm_head.state_dict())


# ─── Depth Expansion ─────────────────────────────────────────────────────────

def expand_depth(
    model: SLM,
    clone_layer_indices: list[int],
    noise_std: float = 0.01,
) -> SLM:
    """
    Expand model depth by cloning specified layers and appending them.

    The cloned layers are deep-copied, then symmetry-breaking noise is added
    to all weight parameters. This ensures the cloned layers can learn
    different functions from their sources via gradient differentiation.

    This is a warm-start expansion, not an exact function-preserving transform:
    appending active Transformer blocks changes the model output. Use
    `validate_expansion()` to measure the initial output drift.

    Args:
        model: Trained SLM model.
        clone_layer_indices: 0-indexed layer indices to clone (e.g., [3, 4, 5]).
        noise_std: Standard deviation of Gaussian noise for symmetry breaking.
                   Must be > 0 — without noise, cloned layers produce identical
                   gradients and provide no training benefit.

    Returns:
        New SLM model with expanded depth. Original model is not modified.
    """
    assert noise_std > 0, (
        "noise_std must be > 0. Without symmetry-breaking noise, "
        "cloned layers produce identical gradients."
    )
    assert all(0 <= i < model.cfg.n_layers for i in clone_layer_indices), (
        f"clone_layer_indices {clone_layer_indices} out of range "
        f"for {model.cfg.n_layers}-layer model"
    )

    old_cfg = model.cfg
    new_n_layers = old_cfg.n_layers + len(clone_layer_indices)

    # Build new config
    new_cfg = SLMConfig(
        vocab_size=old_cfg.vocab_size,
        d_model=old_cfg.d_model,
        n_layers=new_n_layers,
        n_heads=old_cfg.n_heads,
        d_ff=old_cfg.d_ff,
        ctx_len=old_cfg.ctx_len,
        dropout=old_cfg.dropout,
        bias=old_cfg.bias,
        weight_tying=old_cfg.weight_tying,
        pos_type=old_cfg.pos_type,
        rope_base=old_cfg.rope_base,
    )

    # Create new model (random init)
    new_model = SLM(new_cfg)

    # Copy all non-layer parameters from old model
    _copy_non_layer_parameters(model, new_model)

    # Copy original layers
    for i in range(old_cfg.n_layers):
        new_model.layers[i].load_state_dict(model.layers[i].state_dict())

    # Clone specified layers and append with noise
    for j, src_idx in enumerate(clone_layer_indices):
        dest_idx = old_cfg.n_layers + j
        # Deep copy the source layer's state dict
        src_state = copy.deepcopy(model.layers[src_idx].state_dict())

        # Add symmetry-breaking noise to all weight tensors
        noised_state = {}
        for key, tensor in src_state.items():
            if tensor.dim() >= 2:  # Weight matrices
                noise = torch.randn_like(tensor) * noise_std
                noised_state[key] = tensor + noise
            else:  # Biases, norms — keep exact
                noised_state[key] = tensor

        new_model.layers[dest_idx].load_state_dict(noised_state)

    print(f"[expand] Depth expansion: {old_cfg.n_layers} → {new_n_layers} layers")
    print(f"[expand] Cloned layers: {clone_layer_indices} → "
          f"[{old_cfg.n_layers}..{new_n_layers - 1}]")
    print(f"[expand] Symmetry noise: σ={noise_std}")
    print(f"[expand] Parameters: {model.num_params()/1e6:.1f}M → "
          f"{new_model.num_params()/1e6:.1f}M")

    return new_model


# ─── FFN Width Expansion ─────────────────────────────────────────────────────

def expand_ffn_width(model: SLM, new_d_ff: int) -> SLM:
    """
    Widen the FFN (SwiGLU) in all layers with function-preserving initialization.

    For each SwiGLU layer:
      - w_gate: (old_ff, d_model) -> (new_ff, d_model); new rows = small random
      - w_up:   (old_ff, d_model) -> (new_ff, d_model); new rows = small random
      - w_down: (d_model, old_ff) -> (d_model, new_ff); new columns = ZEROS

    The zero-initialized w_down columns ensure the expanded neurons are "dormant"
    at init — they don't affect model output. This makes the expansion exactly
    function-preserving: f(x; θ_expanded) = f(x; θ_original) for all x.

    Args:
        model: Trained SLM model.
        new_d_ff: Target FFN width (must be > current d_ff).

    Returns:
        New SLM model with wider FFN. Original model is not modified.
    """
    old_d_ff = model.cfg.d_ff
    assert new_d_ff > old_d_ff, (
        f"new_d_ff ({new_d_ff}) must be > current d_ff ({old_d_ff})"
    )

    old_cfg = model.cfg
    new_cfg = SLMConfig(
        vocab_size=old_cfg.vocab_size,
        d_model=old_cfg.d_model,
        n_layers=old_cfg.n_layers,
        n_heads=old_cfg.n_heads,
        d_ff=new_d_ff,
        ctx_len=old_cfg.ctx_len,
        dropout=old_cfg.dropout,
        bias=old_cfg.bias,
        weight_tying=old_cfg.weight_tying,
        pos_type=old_cfg.pos_type,
        rope_base=old_cfg.rope_base,
    )

    new_model = SLM(new_cfg)

    # Copy non-layer parameters
    _copy_non_layer_parameters(model, new_model)

    d_model = old_cfg.d_model
    init_std = 0.02 / math.sqrt(2 * old_cfg.n_layers)

    # Expand each layer's FFN
    for i in range(old_cfg.n_layers):
        old_layer = model.layers[i]
        new_layer = new_model.layers[i]

        # Copy attention weights exactly (unchanged)
        new_layer.norm1.load_state_dict(old_layer.norm1.state_dict())
        new_layer.attn.load_state_dict(old_layer.attn.state_dict())
        new_layer.norm2.load_state_dict(old_layer.norm2.state_dict())

        # Expand w_gate: (old_ff, d_model) -> (new_ff, d_model)
        old_gate = old_layer.ffn.w_gate.weight.data  # (old_ff, d_model)
        new_gate = new_layer.ffn.w_gate.weight.data   # (new_ff, d_model)
        new_gate[:old_d_ff, :] = old_gate
        nn.init.normal_(new_gate[old_d_ff:, :], 0.0, init_std)
        if old_layer.ffn.w_gate.bias is not None:
            new_layer.ffn.w_gate.bias.data[:old_d_ff] = old_layer.ffn.w_gate.bias.data
            new_layer.ffn.w_gate.bias.data[old_d_ff:] = 0.0

        # Expand w_up: (old_ff, d_model) -> (new_ff, d_model)
        old_up = old_layer.ffn.w_up.weight.data
        new_up = new_layer.ffn.w_up.weight.data
        new_up[:old_d_ff, :] = old_up
        nn.init.normal_(new_up[old_d_ff:, :], 0.0, init_std)
        if old_layer.ffn.w_up.bias is not None:
            new_layer.ffn.w_up.bias.data[:old_d_ff] = old_layer.ffn.w_up.bias.data
            new_layer.ffn.w_up.bias.data[old_d_ff:] = 0.0

        # Expand w_down: (d_model, old_ff) -> (d_model, new_ff)
        # New columns = ZEROS (function-preserving: dormant neurons)
        old_down = old_layer.ffn.w_down.weight.data  # (d_model, old_ff)
        new_down = new_layer.ffn.w_down.weight.data   # (d_model, new_ff)
        new_down[:, :old_d_ff] = old_down
        new_down[:, old_d_ff:] = 0.0  # Dormant neurons
        if old_layer.ffn.w_down.bias is not None:
            new_layer.ffn.w_down.bias.data.copy_(old_layer.ffn.w_down.bias.data)

    print(f"[expand] FFN widening: d_ff {old_d_ff} → {new_d_ff}")
    print(f"[expand] New neurons per layer: {new_d_ff - old_d_ff} "
          f"(w_down columns initialized to ZERO)")
    print(f"[expand] Parameters: {model.num_params()/1e6:.1f}M → "
          f"{new_model.num_params()/1e6:.1f}M")

    return new_model


# ─── Context Length Expansion ─────────────────────────────────────────────────

def expand_context_length(model: SLM, new_ctx_len: int) -> SLM:
    """
    Extend learnable positional embeddings via linear interpolation.

    For models with pos_type="learnable", the positional embedding matrix
    is (ctx_len, d_model). We interpolate it to (new_ctx_len, d_model) so
    that existing positions retain near-identical values while new positions
    get smooth estimates.

    For RoPE models, context extension is automatic (no learned positions),
    so only the config ctx_len is updated.

    Args:
        model: Trained SLM model.
        new_ctx_len: Target context length (must be > current ctx_len).

    Returns:
        New SLM model with extended context. Original model is not modified.
    """
    old_ctx_len = model.cfg.ctx_len
    assert new_ctx_len > old_ctx_len, (
        f"new_ctx_len ({new_ctx_len}) must be > current ctx_len ({old_ctx_len})"
    )

    old_cfg = model.cfg
    new_cfg = SLMConfig(
        vocab_size=old_cfg.vocab_size,
        d_model=old_cfg.d_model,
        n_layers=old_cfg.n_layers,
        n_heads=old_cfg.n_heads,
        d_ff=old_cfg.d_ff,
        ctx_len=new_ctx_len,
        dropout=old_cfg.dropout,
        bias=old_cfg.bias,
        weight_tying=old_cfg.weight_tying,
        pos_type=old_cfg.pos_type,
        rope_base=old_cfg.rope_base,
    )

    new_model = SLM(new_cfg)

    # Copy all parameters first (layers, embeddings, norms)
    # Layers are identical — copy state dicts
    for i in range(old_cfg.n_layers):
        new_model.layers[i].load_state_dict(model.layers[i].state_dict())
    # skip_pos_emb=True because pos_emb sizes differ; interpolation below handles it
    _copy_non_layer_parameters(model, new_model, skip_pos_emb=True)

    # Handle positional embeddings
    if model.pos_emb is not None and old_cfg.pos_type == "learnable":
        # Interpolate: (old_ctx, d_model) → (new_ctx, d_model)
        old_pos = model.pos_emb.weight.data  # (old_ctx, d_model)

        # Use 1D interpolation along the sequence dimension
        # Reshape for F.interpolate: (1, d_model, old_ctx)
        old_pos_t = old_pos.T.unsqueeze(0)  # (1, d_model, old_ctx)
        new_pos_t = F.interpolate(
            old_pos_t,
            size=new_ctx_len,
            mode="linear",
            align_corners=True,
        )
        new_model.pos_emb.weight.data = new_pos_t.squeeze(0).T  # (new_ctx, d_model)

        print(f"[expand] Positional embeddings interpolated: "
              f"{old_ctx_len} → {new_ctx_len}")
    elif old_cfg.pos_type == "rope":
        # RoPE extends automatically — just update config
        print(f"[expand] RoPE: ctx_len updated {old_ctx_len} → {new_ctx_len} "
              f"(no weight changes needed)")

    print(f"[expand] Context expansion: {old_ctx_len} → {new_ctx_len}")
    print(f"[expand] Parameters: {model.num_params()/1e6:.1f}M → "
          f"{new_model.num_params()/1e6:.1f}M")

    return new_model


# ─── Expansion Validation ────────────────────────────────────────────────────

@torch.no_grad()
def validate_expansion(
    old_model: SLM,
    new_model: SLM,
    vocab_size: int,
    device: str = "cpu",
    n_samples: int = 5,
    seq_len: int = None,
) -> dict:
    """
    Validate that model expansion preserves function.

    Runs the same inputs through both models and checks:
      - Cosine similarity of output logits
      - Loss difference
      - Max absolute difference

    Returns a dict with validation metrics.
    """
    old_model.eval()
    new_model.eval()

    # Use the minimum context length
    if seq_len is None:
        seq_len = min(old_model.cfg.ctx_len, new_model.cfg.ctx_len)
    seq_len = min(seq_len, 64)  # Keep it short for validation

    cos_sims = []
    loss_diffs = []
    max_diffs = []

    for _ in range(n_samples):
        x = torch.randint(0, vocab_size, (1, seq_len), device=device)
        y = torch.randint(0, vocab_size, (1, seq_len), device=device)

        old_logits, old_loss = old_model(x, y)
        new_logits, new_loss = new_model(x, y)

        # Cosine similarity
        old_flat = old_logits.view(-1)
        new_flat = new_logits[:, :seq_len, :].reshape(-1)
        cos_sim = F.cosine_similarity(
            old_flat.unsqueeze(0), new_flat.unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)

        # Loss difference
        loss_diffs.append(abs(new_loss.item() - old_loss.item()))

        # Max absolute difference
        max_diff = (old_logits - new_logits[:, :seq_len, :]).abs().max().item()
        max_diffs.append(max_diff)

    result = {
        "cosine_similarity": sum(cos_sims) / len(cos_sims),
        "avg_loss_diff": sum(loss_diffs) / len(loss_diffs),
        "max_abs_diff": max(max_diffs),
        "n_samples": n_samples,
    }

    status = "PASS" if result["cosine_similarity"] > 0.90 else "FAIL"
    print(f"[validate] Expansion validation: {status}")
    print(f"[validate]   Cosine similarity: {result['cosine_similarity']:.4f}")
    print(f"[validate]   Avg loss diff:     {result['avg_loss_diff']:.4f}")
    print(f"[validate]   Max abs diff:      {result['max_abs_diff']:.4f}")

    return result


# ─── Checkpoint Creation ──────────────────────────────────────────────────────

def create_expansion_checkpoint(
    model: SLM,
    output_path: str,
    source_checkpoint_path: str,
    expansion_type: str,
    expansion_details: dict,
):
    """
    Save an expanded model as a new checkpoint.

    The checkpoint includes:
      - Model state dict and config
      - Expansion metadata (type, source, details)
      - Reset step/tokens counters (fresh training start)
      - No optimizer state (must be re-initialized)
    """
    data = {
        "model_state": model.state_dict(),
        "config": model.cfg,
        "step": 0,
        "tokens_seen": 0,
        "best_val_loss": float("inf"),
        "expansion_meta": {
            "type": expansion_type,
            "source_checkpoint": source_checkpoint_path,
            "details": expansion_details,
            "params_before": expansion_details.get("params_before"),
            "params_after": model.num_params(),
        },
    }

    torch.save(data, output_path)
    print(f"[expand] Checkpoint saved → {output_path}")
    print(f"[expand]   Params: {model.num_params()/1e6:.1f}M")


# ─── High-level stage expansion ──────────────────────────────────────────────

def run_stage_expansion(
    checkpoint_path: str,
    stage: str,
    output_path: str,
    clone_layers: list[int] = None,
    noise_std: float = 0.01,
    new_d_ff: int = None,
    new_ctx_len: int = None,
    device: str = "cpu",
):
    """
    Run a complete expansion stage — load, expand, validate, save.

    Handles single or combined expansions per stage.
    """
    print(f"\n{'='*70}")
    print(f"  MODEL EXPANSION — Stage {stage}")
    print(f"{'='*70}\n")

    # Load source checkpoint
    print(f"[expand] Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    old_cfg = ckpt["config"]
    model = SLM(old_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params_before = model.num_params()
    print(f"[expand] Source model: {params_before/1e6:.1f}M params, "
          f"{old_cfg.n_layers}L, d_ff={old_cfg.d_ff}, ctx={old_cfg.ctx_len}")

    expansion_details = {
        "stage": stage,
        "params_before": params_before,
        "source_config": {
            "n_layers": old_cfg.n_layers,
            "d_ff": old_cfg.d_ff,
            "ctx_len": old_cfg.ctx_len,
        },
    }

    # Apply expansions
    if clone_layers is not None:
        print(f"\n--- Depth Expansion ---")
        old_for_validation = model
        model = expand_depth(model, clone_layers, noise_std).to(device)
        expansion_details["depth"] = {
            "clone_from": clone_layers,
            "noise_std": noise_std,
            "new_n_layers": model.cfg.n_layers,
        }

        # Validate depth expansion
        result = validate_expansion(
            old_for_validation, model, old_cfg.vocab_size, device
        )
        if result["cosine_similarity"] < 0.90:
            print("[WARNING] Depth expansion cosine similarity < 0.90!")
            print("[WARNING] Consider reducing noise_std.")

    if new_d_ff is not None:
        print(f"\n--- FFN Widening ---")
        old_for_validation = model
        model = expand_ffn_width(model, new_d_ff).to(device)
        expansion_details["ffn_width"] = {
            "old_d_ff": old_for_validation.cfg.d_ff,
            "new_d_ff": new_d_ff,
        }

        # Validate FFN widening (should be exact)
        result = validate_expansion(
            old_for_validation, model, old_cfg.vocab_size, device
        )
        if result["max_abs_diff"] > 1e-4:
            print("[WARNING] FFN widening max diff > 1e-4!")
            print("[WARNING] Function preservation may be compromised.")

    if new_ctx_len is not None:
        print(f"\n--- Context Expansion ---")
        model = expand_context_length(model, new_ctx_len).to(device)
        expansion_details["context"] = {
            "old_ctx_len": old_cfg.ctx_len,
            "new_ctx_len": new_ctx_len,
        }

    # Save expanded checkpoint
    print(f"\n--- Saving ---")
    create_expansion_checkpoint(
        model, output_path, checkpoint_path,
        expansion_type=f"stage_{stage}",
        expansion_details=expansion_details,
    )

    print(f"\n{'='*70}")
    print(f"  EXPANSION COMPLETE")
    print(f"  {params_before/1e6:.1f}M → {model.num_params()/1e6:.1f}M")
    print(f"{'='*70}\n")

    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Expand SLM model size")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to source checkpoint")
    p.add_argument("--output", type=str, required=True,
                   help="Path for expanded checkpoint")
    p.add_argument("--action", type=str, required=True,
                   choices=["expand_depth", "expand_ffn", "expand_context",
                            "stage_a", "stage_b", "stage_c"],
                   help="Expansion action to perform")
    p.add_argument("--clone_layers", type=str, default="3,4,5",
                   help="Comma-separated 0-indexed layer indices to clone")
    p.add_argument("--noise_std", type=float, default=0.01,
                   help="Symmetry-breaking noise std for cloned layers")
    p.add_argument("--new_d_ff", type=int, default=3584,
                   help="Target FFN width for expand_ffn action")
    p.add_argument("--new_ctx_len", type=int, default=384,
                   help="Target context length for expand_context action")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clone_layers = [int(x) for x in args.clone_layers.split(",")]

    if args.action == "expand_depth":
        run_stage_expansion(
            args.checkpoint, "depth", args.output,
            clone_layers=clone_layers, noise_std=args.noise_std,
            device=device,
        )
    elif args.action == "expand_ffn":
        run_stage_expansion(
            args.checkpoint, "ffn", args.output,
            new_d_ff=args.new_d_ff, device=device,
        )
    elif args.action == "expand_context":
        run_stage_expansion(
            args.checkpoint, "context", args.output,
            new_ctx_len=args.new_ctx_len, device=device,
        )
    elif args.action == "stage_a":
        # Stage A: 6 → 9 layers
        run_stage_expansion(
            args.checkpoint, "A", args.output,
            clone_layers=[3, 4, 5], noise_std=0.01,
            device=device,
        )
    elif args.action == "stage_b":
        # Stage B: 9 → 12 layers + context 256→384
        run_stage_expansion(
            args.checkpoint, "B", args.output,
            clone_layers=[6, 7, 8], noise_std=0.01,
            new_ctx_len=384, device=device,
        )
    elif args.action == "stage_c":
        # Stage C: FFN widen + context 384→512
        run_stage_expansion(
            args.checkpoint, "C", args.output,
            new_d_ff=3584, new_ctx_len=512,
            device=device,
        )
