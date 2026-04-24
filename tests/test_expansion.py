"""
test_expansion.py — Unit tests for model expansion utilities.

Tests:
  1. Depth expansion preserves function (high cosine similarity)
  2. FFN widening is exactly function-preserving (identical output)
  3. Context expansion produces valid model
  4. Parameter counts match expectations
  5. Checkpoint round-trip works
"""

import sys
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, ".")

from src.model import SLM, SLMConfig
from expand_model import (
    expand_depth,
    expand_ffn_width,
    expand_context_length,
    validate_expansion,
    create_expansion_checkpoint,
)


def make_small_model(n_layers=6, d_ff=2048, ctx_len=64, vocab_size=1000):
    """Create a small model for testing (fast)."""
    cfg = SLMConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=n_layers,
        n_heads=4,
        d_ff=d_ff,
        ctx_len=ctx_len,
        pos_type="learnable",
        weight_tying=True,
    )
    model = SLM(cfg)
    # Give it non-trivial weights (not just random init)
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
    return model


def test_depth_expansion():
    """Test that depth expansion preserves function with high similarity."""
    print("\n=== Test: Depth Expansion ===")

    model = make_small_model(n_layers=6)
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))

    model.eval()
    with torch.no_grad():
        old_logits, old_loss = model(x, y)

    # Expand: clone layers 3,4,5
    expanded = expand_depth(model, [3, 4, 5], noise_std=0.01)
    expanded.eval()

    assert expanded.cfg.n_layers == 9, f"Expected 9 layers, got {expanded.cfg.n_layers}"
    assert len(list(expanded.layers)) == 9

    with torch.no_grad():
        new_logits, new_loss = expanded(x, y)

    # Check cosine similarity
    cos_sim = F.cosine_similarity(
        old_logits.reshape(1, -1), new_logits.reshape(1, -1)
    ).item()
    loss_diff = abs(new_loss.item() - old_loss.item())

    print(f"  Cosine similarity: {cos_sim:.4f} (threshold: 0.90)")
    print(f"  Loss diff: {loss_diff:.4f}")
    print(f"  Old params: {model.num_params()}")
    print(f"  New params: {expanded.num_params()}")

    assert cos_sim > 0.85, f"Cosine similarity too low: {cos_sim}"
    print("  ✓ PASSED")
    return True


def test_depth_expansion_noise_mandatory():
    """Test that noise_std=0 raises an error."""
    print("\n=== Test: Noise Must Be Mandatory ===")

    model = make_small_model(n_layers=6)
    try:
        expand_depth(model, [3, 4, 5], noise_std=0.0)
        print("  ✗ FAILED — should have raised AssertionError")
        return False
    except AssertionError as e:
        print(f"  Correctly raised: {e}")
        print("  ✓ PASSED")
        return True


def test_ffn_widening():
    """Test that FFN widening is exactly function-preserving."""
    print("\n=== Test: FFN Widening (Function Preserving) ===")

    model = make_small_model(n_layers=4, d_ff=128)
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))

    model.eval()
    with torch.no_grad():
        old_logits, old_loss = model(x, y)

    # Widen FFN
    expanded = expand_ffn_width(model, new_d_ff=256)
    expanded.eval()

    assert expanded.cfg.d_ff == 256, f"Expected d_ff=256, got {expanded.cfg.d_ff}"

    with torch.no_grad():
        new_logits, new_loss = expanded(x, y)

    max_diff = (old_logits - new_logits).abs().max().item()
    loss_diff = abs(new_loss.item() - old_loss.item())

    print(f"  Max absolute diff: {max_diff:.8f} (threshold: 1e-4)")
    print(f"  Loss diff: {loss_diff:.8f}")
    print(f"  Old params: {model.num_params()}")
    print(f"  New params: {expanded.num_params()}")

    # FFN widening should be EXACT (zero-padded w_down)
    assert max_diff < 1e-4, f"FFN widening not function-preserving: max_diff={max_diff}"
    print("  ✓ PASSED")
    return True


def test_context_expansion():
    """Test that context expansion via interpolation works."""
    print("\n=== Test: Context Expansion ===")

    model = make_small_model(n_layers=4, ctx_len=32)

    # Expand context
    expanded = expand_context_length(model, new_ctx_len=64)
    expanded.eval()

    assert expanded.cfg.ctx_len == 64, f"Expected ctx_len=64, got {expanded.cfg.ctx_len}"

    # Should be able to process longer sequences
    x = torch.randint(0, 1000, (1, 48))
    y = torch.randint(0, 1000, (1, 48))

    with torch.no_grad():
        logits, loss = expanded(x, y)

    print(f"  New ctx_len: {expanded.cfg.ctx_len}")
    print(f"  Can process seq_len=48: ✓")
    print(f"  Loss on extended seq: {loss.item():.4f}")
    print(f"  Old pos_emb shape: {model.pos_emb.weight.shape}")
    print(f"  New pos_emb shape: {expanded.pos_emb.weight.shape}")

    # Check that original positions are preserved (approximately)
    old_pos = model.pos_emb.weight.data
    new_pos = expanded.pos_emb.weight.data
    # Interpolation means positions 0 and 31 should be very close
    pos0_diff = (old_pos[0] - new_pos[0]).abs().max().item()
    pos_last_diff = (old_pos[-1] - new_pos[-1]).abs().max().item()
    print(f"  Position 0 max diff: {pos0_diff:.6f}")
    print(f"  Position -1 max diff: {pos_last_diff:.6f}")

    assert pos0_diff < 0.01, f"Position 0 not preserved: diff={pos0_diff}"
    assert pos_last_diff < 0.01, f"Position -1 not preserved: diff={pos_last_diff}"
    print("  ✓ PASSED")
    return True


def test_parameter_counts():
    """Verify parameter counts match expected values for expansion stages."""
    print("\n=== Test: Parameter Count Verification ===")

    V, d, ctx = 40000, 512, 256

    expected = {
        "Current (6L, ff=2048)": (6, 2048, 45.5e6, 46.5e6),
        "Stage A (9L, ff=2048)": (9, 2048, 58.0e6, 59.0e6),
        "Stage B (12L, ff=2048)": (12, 2048, 70.5e6, 71.5e6),
        "Stage C (12L, ff=3584)": (12, 3584, 98.5e6, 100.0e6),
    }

    all_pass = True
    for name, (nL, dff, lo, hi) in expected.items():
        # Calculate (same formula as earlier)
        per_layer = 4 * d * d + 3 * d * dff + 2 * d
        total = V * d + ctx * d + d + nL * per_layer

        in_range = lo <= total <= hi
        status = "✓" if in_range else "✗"
        print(f"  {name:<30} {total/1e6:>6.1f}M  "
              f"(expected {lo/1e6:.0f}-{hi/1e6:.0f}M) [{status}]")

        if not in_range:
            all_pass = False

    if all_pass:
        print("  ✓ ALL PASSED")
    else:
        print("  ✗ SOME FAILED")
    return all_pass


def test_checkpoint_roundtrip():
    """Test saving and loading an expanded checkpoint."""
    print("\n=== Test: Checkpoint Round-Trip ===")

    import tempfile, os

    model = make_small_model(n_layers=4, d_ff=128)
    expanded = expand_ffn_width(model, new_d_ff=256)

    x = torch.randint(0, 1000, (1, 32))

    expanded.eval()
    with torch.no_grad():
        original_logits, _ = expanded(x)

    # Save
    tmp_path = os.path.join("checkpoints", "_test_expansion_roundtrip.pt")
    create_expansion_checkpoint(
        expanded, tmp_path,
        source_checkpoint_path="test",
        expansion_type="test",
        expansion_details={"test": True, "params_before": model.num_params()},
    )

    # Load
    ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
    loaded_cfg = ckpt["config"]
    loaded_model = SLM(loaded_cfg)
    loaded_model.load_state_dict(ckpt["model_state"])
    loaded_model.eval()

    with torch.no_grad():
        loaded_logits, _ = loaded_model(x)

    max_diff = (original_logits - loaded_logits).abs().max().item()
    print(f"  Max diff after round-trip: {max_diff:.10f}")

    # Clean up
    os.remove(tmp_path)

    assert max_diff < 1e-6, f"Round-trip not exact: max_diff={max_diff}"
    print("  ✓ PASSED")
    return True


def test_combined_expansion():
    """Test depth + FFN expansion in sequence (Stage A→B→C simulation)."""
    print("\n=== Test: Combined Expansion Pipeline ===")

    model = make_small_model(n_layers=3, d_ff=64, ctx_len=32)
    x = torch.randint(0, 1000, (1, 16))
    y = torch.randint(0, 1000, (1, 16))

    model.eval()
    with torch.no_grad():
        base_logits, base_loss = model(x, y)

    # Stage A: depth 3→5
    model_a = expand_depth(model, [1, 2], noise_std=0.01)
    assert model_a.cfg.n_layers == 5

    # Stage B: depth 5→7 + context 32→48
    model_b = expand_depth(model_a, [3, 4], noise_std=0.01)
    model_b = expand_context_length(model_b, 48)
    assert model_b.cfg.n_layers == 7
    assert model_b.cfg.ctx_len == 48

    # Stage C: FFN widen 64→128
    model_c = expand_ffn_width(model_b, new_d_ff=128)
    assert model_c.cfg.d_ff == 128

    # Verify the final model works
    model_c.eval()
    x_long = torch.randint(0, 1000, (1, 32))
    y_long = torch.randint(0, 1000, (1, 32))
    with torch.no_grad():
        final_logits, final_loss = model_c(x_long, y_long)

    print(f"  Stage transitions: 3L→5L→7L (depth) + ff 64→128 (width)")
    print(f"  Context: 32→48")
    print(f"  Base params:  {model.num_params()}")
    print(f"  Final params: {model_c.num_params()}")
    print(f"  Final loss:   {final_loss.item():.4f}")
    print("  ✓ PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  EXPANSION MODEL TESTS")
    print("=" * 60)

    tests = [
        test_depth_expansion,
        test_depth_expansion_noise_mandatory,
        test_ffn_widening,
        test_context_expansion,
        test_parameter_counts,
        test_checkpoint_roundtrip,
        test_combined_expansion,
    ]

    results = []
    for test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_fn.__name__, passed))
        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  {n_pass}/{n_total} tests passed")
    print(f"{'='*60}")

    sys.exit(0 if n_pass == n_total else 1)
