"""
debug_stage1_loss.py — Comprehensive trace of Stage 1 data flow and loss computation.

Traces:
  1. Replay source loading (TinyStories)
  2. Current stage data loading (SimpleWiki)
  3. Mixed dataset composition
  4. Batch formation and shapes
  5. Model forward pass
  6. Loss computation
  7. Loss spike pattern
"""

import os
import yaml
import pickle
import torch
from pathlib import Path

# Import project modules
from src.model import SLM, SLMConfig
from src.dataset import StreamingStageDataset, make_dataloader
from tokenizers import Tokenizer

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_replay_cache():
    """Check if replay source cache exists and its dimensions."""
    print_section("1. REPLAY SOURCE CACHE CHECK")
    
    cache_dir = "cache"
    replay_sources = ["tinystories"]
    seq_len = 512
    
    for source in replay_sources:
        # Check exact match first
        exact_cache = os.path.join(cache_dir, f"train_{source}_seq{seq_len}.pkl")
        if os.path.exists(exact_cache):
            with open(exact_cache, "rb") as f:
                chunks = pickle.load(f)
            print(f"\n✓ Found: {exact_cache}")
            print(f"  - Chunks: {len(chunks):,}")
            print(f"  - First chunk length: {len(chunks[0])}")
            print(f"  - Expected: {seq_len + 1} (input + target shift)")
            print(f"  - Match: {'✓ YES' if len(chunks[0]) == seq_len + 1 else '✗ NO'}")
            
            # Check chunk consistency
            lengths = [len(c) for c in chunks]
            unique_lengths = set(lengths)
            print(f"  - Unique chunk lengths: {sorted(unique_lengths)}")
            if len(unique_lengths) == 1:
                print(f"    ✓ All chunks same length")
            else:
                print(f"    ✗ INCONSISTENT LENGTHS - This will cause issues!")
                print(f"      Count by length:")
                from collections import Counter
                for length, count in sorted(Counter(lengths).items()):
                    print(f"        {length}: {count:,} chunks")
        else:
            # Check for fallback caches
            import glob
            pattern = os.path.join(cache_dir, f"train_{source}_seq*.pkl")
            matches = sorted(glob.glob(pattern))
            if matches:
                print(f"\n⚠ No exact match at seq={seq_len}")
                print(f"  Found fallback caches:")
                for m in matches:
                    try:
                        with open(m, "rb") as f:
                            chunks = pickle.load(f)
                        cached_seq = int(m.split('seq')[-1].split('.pkl')[0])
                        print(f"    - {os.path.basename(m)}: {len(chunks):,} chunks @ seq={cached_seq}")
                    except Exception as e:
                        print(f"    - {os.path.basename(m)}: ERROR reading - {e}")
            else:
                print(f"\n✗ NO CACHE FOUND for '{source}'")
                print(f"  Pattern: {pattern}")
                print(f"  This source will be SKIPPED from replay!")

def check_current_stage_cache():
    """Check if current stage (SimpleWiki) cache exists."""
    print_section("2. CURRENT STAGE CACHE CHECK")
    
    cache_dir = "cache"
    dataset_name = "simplewiki"
    seq_len = 512
    
    cache_path = os.path.join(cache_dir, f"train_{dataset_name}_seq{seq_len}.pkl")
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"\n✓ Found: {cache_path}")
        print(f"  - Chunks: {len(chunks):,}")
        print(f"  - Memory: ~{len(chunks) * 513 * 8 / 1e9:.2f} GB (if all float64)")
        
        # Sample chunks
        sample_idx = 0
        print(f"\n  Sample chunk #{sample_idx}:")
        print(f"    - Length: {len(chunks[sample_idx])}")
        print(f"    - First 10 token IDs: {chunks[sample_idx][:10]}")
        print(f"    - Last 10 token IDs: {chunks[sample_idx][-10:]}")
        
        # Check for NaN/inf/invalid tokens
        all_tokens = []
        for chunk in chunks[:100]:  # Check first 100 chunks
            all_tokens.extend(chunk)
        min_tok = min(all_tokens)
        max_tok = max(all_tokens)
        print(f"\n  Token ID statistics (first 100 chunks):")
        print(f"    - Min: {min_tok}")
        print(f"    - Max: {max_tok}")
        print(f"    - Expected range: [0, vocab_size)")
    else:
        print(f"\n✗ NOT FOUND: {cache_path}")
        print(f"  Stage 1 will need to build this cache first!")

def check_tokenizer():
    """Check tokenizer capabilities."""
    print_section("3. TOKENIZER CHECK")
    
    tokenizer_path = "tokenizers/tokenizer_corpus.json"
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            vocab_size = tokenizer.get_vocab_size()
            print(f"\n✓ Loaded: {tokenizer_path}")
            print(f"  - Vocab size: {vocab_size:,}")
            
            # Test encoding
            test_text = "The quick brown fox jumps over the lazy dog."
            encoded = tokenizer.encode(test_text)
            print(f"\n  Test encoding:")
            print(f"    Text: '{test_text}'")
            print(f"    Tokens: {encoded.ids}")
            print(f"    Token count: {len(encoded.ids)}")
        except Exception as e:
            print(f"\n✗ ERROR loading tokenizer: {e}")
    else:
        print(f"\n✗ NOT FOUND: {tokenizer_path}")

def load_config():
    """Load stage1 config."""
    print_section("4. STAGE 1 CONFIG")
    
    config_path = "configs/stage1.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"\n✓ Loaded: {config_path}")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
        return cfg
    else:
        print(f"\n✗ NOT FOUND: {config_path}")
        return None

def build_dataset_and_check():
    """Build training dataset and check replay mixing."""
    print_section("5. DATASET BUILDING & REPLAY MIXING")
    
    # Load config
    with open("configs/stage1.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    
    dataset_name = cfg_dict["dataset"]
    seq_len = int(cfg_dict["seq_len"])
    max_tokens = int(str(cfg_dict["max_tokens"]).replace("_", ""))
    replay_ratio = float(cfg_dict.get("replay_ratio", 0.0))
    replay_from = cfg_dict.get("replay_from", []) or []
    batch_size = int(cfg_dict["batch_size"])
    cache_dir = "cache"
    
    print(f"\nDataset config:")
    print(f"  - Name: {dataset_name}")
    print(f"  - Seq length: {seq_len}")
    print(f"  - Max tokens: {max_tokens:,}")
    print(f"  - Replay ratio: {replay_ratio*100:.1f}%")
    print(f"  - Replay from: {replay_from}")
    print(f"  - Batch size: {batch_size}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizers/tokenizer_corpus.json")
    
    # Build dataset (this will load/create caches)
    print(f"\nBuilding dataset...")
    try:
        train_ds = StreamingStageDataset().build(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            replay_from=replay_from,
            replay_ratio=replay_ratio,
        )
        
        print(f"\n✓ Dataset built successfully")
        print(f"  - Total chunks: {len(train_ds):,}")
        print(f"  - Total tokens (approx): {len(train_ds) * seq_len:,}")
        
        # Estimate replay composition
        if replay_ratio > 0:
            n_replay = int((replay_ratio / max(1 - replay_ratio, 1e-6)) * (len(train_ds) / (1 + replay_ratio / (1 - replay_ratio))))
            n_current = len(train_ds) - n_replay
            print(f"  - Current (SimpleWiki): ~{n_current:,} chunks")
            print(f"  - Replay (TinyStories): ~{n_replay:,} chunks")
            print(f"  - Actual replay ratio: {n_replay / len(train_ds) * 100:.1f}%")
        
        return train_ds, tokenizer
    except Exception as e:
        print(f"\n✗ ERROR building dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def check_batch_flow(train_ds, tokenizer):
    """Inspect batch formation and shapes."""
    print_section("6. BATCH FORMATION")
    
    if train_ds is None:
        print("\n✗ Dataset unavailable (previous step failed)")
        return None
    
    try:
        dataloader = make_dataloader(train_ds, batch_size=32)
        
        print(f"\nDataLoader created:")
        print(f"  - Batch size: 32")
        print(f"  - Batches per epoch: {len(train_ds) // 32:,}")
        
        # Get first batch
        x, y = next(iter(dataloader))
        print(f"\nFirst batch shapes:")
        print(f"  - Input (x): {x.shape}")
        print(f"    Expected: [32, 512]")
        print(f"    Match: {'✓' if x.shape == (32, 512) else '✗'}")
        print(f"  - Target (y): {y.shape}")
        print(f"    Expected: [32, 512]")
        print(f"    Match: {'✓' if y.shape == (32, 512) else '✗'}")
        
        # Check for invalid tokens
        print(f"\nToken statistics (first batch):")
        print(f"  - Min token ID: {x.min().item()}")
        print(f"  - Max token ID: {x.max().item()}")
        print(f"  - Vocab size: {tokenizer.get_vocab_size():,}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(x).any() or torch.isnan(y).any()
        has_inf = torch.isinf(x).any() or torch.isinf(y).any()
        print(f"  - Has NaN: {'✗ YES' if has_nan else '✓ NO'}")
        print(f"  - Has Inf: {'✗ YES' if has_inf else '✓ NO'}")
        
        return x, y, dataloader
    except Exception as e:
        print(f"\n✗ ERROR in batch flow: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def check_model_forward(x, y, tokenizer):
    """Run model forward pass and trace loss."""
    print_section("7. MODEL FORWARD PASS")
    
    if x is None:
        print("\n✗ Batch unavailable (previous step failed)")
        return
    
    try:
        # Load stage0 checkpoint
        checkpoint_path = "checkpoints/stage0_best.pt"
        if not os.path.exists(checkpoint_path):
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
            return
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Create model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab_size = tokenizer.get_vocab_size()
        model_cfg = SLMConfig(
            vocab_size=vocab_size,
            pos_type="rope",
            ctx_len=512,
        )
        model = SLM(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        print(f"\n✓ Model loaded")
        print(f"  - Config: {model_cfg}")
        print(f"  - Device: {device}")
        print(f"  - Parameters: {model.num_params():,}")
        
        # Forward pass
        x_dev = x.to(device)
        y_dev = y.to(device)
        
        print(f"\nForward pass:")
        with torch.no_grad():
            logits, loss = model(x_dev, y_dev)
        
        print(f"  - Logits shape: {logits.shape}")
        print(f"    Expected: [32, 512, {vocab_size}]")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Check for NaN/Inf in logits
        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()
        print(f"  - Logits has NaN: {'✗ YES' if has_nan else '✓ NO'}")
        print(f"  - Logits has Inf: {'✗ YES' if has_inf else '✓ NO'}")
        print(f"  - Logits min: {logits.min().item():.4f}")
        print(f"  - Logits max: {logits.max().item():.4f}")
        
        return loss.item()
    except Exception as e:
        print(f"\n✗ ERROR in model forward: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_loss_spike_simulation():
    """Simulate loss spike detector behavior."""
    print_section("8. LOSS SPIKE SIMULATION")
    
    # Simulate a training run
    # Assuming high initial loss on SimpleWiki, some recovery with replay data
    simulated_losses = [
        9.2, 9.1, 9.0, 8.95, 8.98,      # steps 0-4 (basline period)
        8.99, 8.97, 8.96, 8.94, 9.01,   # steps 5-9 (baseline period)
        9.03, 9.05, 9.1, 9.15, 9.2,     # steps 10-14
        9.25, 9.3, 9.35, 9.35, 9.4,     # steps 15-19
        9.45,                             # step 20
    ]
    
    print(f"\nSimulated loss sequence (21 steps):")
    for i, loss in enumerate(simulated_losses):
        print(f"  Step {i:2d}: {loss:.3f}")
    
    # Simulate spike detector
    from collections import deque
    window = deque(maxlen=20)
    threshold = 0.5
    
    print(f"\nSpike detection (window=20, threshold={threshold}):")
    for step, train_loss in enumerate(simulated_losses):
        window.append(train_loss)
        
        if len(window) < window.maxlen:
            status = f"[Building window {len(window)}/20]"
        else:
            baseline = min(list(window)[: window.maxlen // 2])
            current = train_loss
            delta = current - baseline
            spike = delta > threshold
            status = f"baseline={baseline:.3f}, current={current:.3f}, Δ={delta:.3f} → {'SPIKE!' if spike else 'ok'}"
        
        print(f"  Step {step:2d}: {status}")

def main():
    print("\n" + "="*70)
    print("  STAGE 1 LOSS SPIKE DEBUG TRACE")
    print("="*70)
    
    # Run all checks
    check_replay_cache()
    check_current_stage_cache()
    check_tokenizer()
    cfg = load_config()
    
    # Build dataset and dataloader
    train_ds, tokenizer = build_dataset_and_check()
    
    # Check first batch
    x, y, dataloader = check_batch_flow(train_ds, tokenizer)
    
    # Run model forward on first batch
    loss = check_model_forward(x, y, tokenizer)
    
    # Simulate spike detection
    check_loss_spike_simulation()
    
    print_section("SUMMARY")
    print("""
This debug trace shows the complete data flow from config → dataset → batch → model → loss.

Common issues:
1. ✗ Replay cache missing or wrong seq_len → Skipped replay
2. ✗ Dataset cache not built → Slow first run
3. ✗ Batch shapes mismatch → Model forward fails
4. ✗ High initial loss + low warmup → Spike detector fires too early
5. ✗ Distribution shift too severe → Model can't handle SimpleWiki

Next steps:
- See LOSS_SPIKE_ANALYSIS.md for detailed explanation
- Modify stage1.yaml to increase warmup_steps or reduce spike_threshold
- Consider curriculum scheduling: use easier dataset first
""")

if __name__ == "__main__":
    main()
