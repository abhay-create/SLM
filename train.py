"""
Main training script for 100M parameter SLM on WritingPrompts dataset.

Usage:
    python train.py                         # Fresh training (curriculum mode)
    python train.py --mode direct           # Direct training (no curriculum)
    python train.py --resume checkpoints/curriculum/best.pt  # Resume from checkpoint
    python train.py --test                  # Run pre-training smoke test only

Launched via nohup:
    nohup python train.py > train.log 2>&1 &
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import sys
import os
import pickle
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import create_model
from dataset import create_curriculum_stages, TextDataset
from tokenizer_utils import create_tokenizer
from trainer import Trainer, CurriculumTrainer
from evaluator import Evaluator, print_evaluation_results
from visualizer import Visualizer


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_smoke_test(model, tokenizer, device, seq_len=64, batch_size=2):
    """
    Run a quick smoke test to verify the model and data pipeline work.
    Returns True if all tests pass, raises on failure.
    """
    print("\n" + "="*60)
    print("SMOKE TEST — verifying setup before full training")
    print("="*60)

    # Move model to device BEFORE creating any tensors on it
    model = model.to(device)
    model.eval()

    # 1. Test forward pass
    print("[1/5] Testing forward pass...", end=" ", flush=True)
    dummy_x = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    dummy_y = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        logits, loss = model(dummy_x, dummy_y)
    assert logits.shape == (batch_size, seq_len, tokenizer.vocab_size), \
        f"Bad logit shape: {logits.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    print(f"OK (loss={loss.item():.4f})")

    # 2. Test loss is reasonable at init (should be ~ln(vocab_size) ≈ 10.8)
    print("[2/5] Checking initial loss range...", end=" ", flush=True)
    expected_loss = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()
    assert abs(loss.item() - expected_loss) < 5.0, \
        f"Initial loss {loss.item():.4f} too far from expected {expected_loss:.4f}"
    print(f"OK (expected ~{expected_loss:.2f})")

    # 3. Test tokenizer
    print("[3/5] Testing tokenizer...", end=" ", flush=True)
    test_text = "Once upon a time, in a land far away,"
    tokens = tokenizer.encode(test_text)
    assert isinstance(tokens, list) and len(tokens) > 0, "Tokenizer returned empty"
    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str) and len(decoded) > 0, "Decode returned empty"
    print(f"OK ({len(tokens)} tokens)")

    # 4. Test data files exist and are loadable
    print("[4/5] Testing data files...", end=" ", flush=True)
    simple_path = Path("data/curriculum/simple_tokens.pkl")
    complex_path = Path("data/curriculum/complex_tokens.pkl")
    assert simple_path.exists(), f"Missing: {simple_path}"
    assert complex_path.exists(), f"Missing: {complex_path}"
    with open(simple_path, 'rb') as f:
        simple_tok = pickle.load(f)
    with open(complex_path, 'rb') as f:
        complex_tok = pickle.load(f)
    assert len(simple_tok) > seq_len, f"Simple data too short: {len(simple_tok)}"
    assert len(complex_tok) > seq_len, f"Complex data too short: {len(complex_tok)}"
    print(f"OK (simple={len(simple_tok):,}, complex={len(complex_tok):,} tokens)")

    # 5. Test DataLoader + backward pass
    print("[5/5] Testing DataLoader + backward pass...", end=" ", flush=True)
    dataset = TextDataset(simple_tok[:10000], seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    model.train()
    optimizer_test = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_test.zero_grad()
    _, test_loss = model(batch_x, batch_y)
    test_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_test.step()
    assert not torch.isnan(test_loss), "Training loss is NaN!"
    assert not torch.isnan(grad_norm), "Grad norm is NaN!"
    print(f"OK (loss={test_loss.item():.4f}, grad_norm={grad_norm:.4f})")

    print("\n✅ All smoke tests passed! Starting training...\n")
    return True


def setup_optimizer(model, config):
    """Setup optimizer and LR scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimization']['betas'],
        eps=config['optimization']['eps'],
    )

    # Cosine LR schedule with warmup
    # Count total steps across all curriculum stages
    steps_per_epoch = 1000  # approximate
    total_steps = config['training']['num_epochs'] * steps_per_epoch
    warmup_steps = config['training']['warmup_steps']

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_ratio = config['training']['min_lr_ratio']
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def ensure_data_exists(tokenizer, min_tokens=500_000):
    """
    Make sure training data exists with enough tokens.
    Downloads/generates if needed.
    """
    simple_path = Path("data/curriculum/simple_tokens.pkl")
    complex_path = Path("data/curriculum/complex_tokens.pkl")

    need_regen = False

    if not simple_path.exists() or not complex_path.exists():
        need_regen = True
    else:
        with open(simple_path, 'rb') as f:
            s = pickle.load(f)
        with open(complex_path, 'rb') as f:
            c = pickle.load(f)
        if len(s) < min_tokens or len(c) < min_tokens:
            print(f"⚠ Data files are too small (simple={len(s):,}, complex={len(c):,}).")
            print(f"  Need at least {min_tokens:,} tokens per file.")
            need_regen = True

    if need_regen:
        print("Regenerating training data...")
        # Import and run the prepare_data script
        try:
            from prepare_data import prepare_writingprompts
            prepare_writingprompts("data/curriculum", max_tokens=20_000_000)
        except Exception as e:
            print(f"Download failed ({e}), generating synthetic data...")
            from prepare_data import _generate_synthetic_writingprompts
            _generate_synthetic_writingprompts(Path("data/curriculum"), tokenizer)


def train_curriculum_mode(model, tokenizer, config, device, resume_path=None):
    """Train using curriculum learning."""
    print("\n" + "="*60)
    print("CURRICULUM LEARNING MODE")
    print("="*60)

    # Ensure data
    ensure_data_exists(tokenizer)

    # Build curriculum
    curriculum = create_curriculum_stages('data/curriculum', tokenizer)

    # Validation loader from complex data (held-out subset)
    with open('data/curriculum/complex_tokens.pkl', 'rb') as f:
        val_tokens_all = pickle.load(f)
    val_tokens = val_tokens_all[int(0.9 * len(val_tokens_all)):]  # last 10% for val
    seq_len = config['training']['seq_len']
    val_dataset = TextDataset(val_tokens, seq_len)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda'),
    )

    # Initial train loader (stage 0)
    train_loader = curriculum.get_stage_dataloader(0, config['training']['batch_size'])

    # Setup optimizer
    optimizer, scheduler = setup_optimizer(model, config)

    # Create trainer
    trainer = CurriculumTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir='logs/curriculum',
        checkpoint_dir='checkpoints/curriculum',
        eval_interval=config['training']['eval_interval'],
        save_interval=config['training']['save_interval'],
        gradient_clip=config['training']['gradient_clip'],
        use_amp=config['training'].get('mixed_precision', True),
        curriculum_dataset=curriculum,
        stage_epochs=config['curriculum']['stage_epochs'],
    )

    # Resume if requested
    if resume_path and Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)
    else:
        # Check for automatic resume from latest checkpoint
        ckpt_dir = Path('checkpoints/curriculum')
        checkpoints = sorted(ckpt_dir.glob('checkpoint_step_*.pt'))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Auto-resuming from latest checkpoint: {latest}")
            trainer.load_checkpoint(str(latest))

    trainer.train_curriculum()
    return trainer


def train_direct_mode(model, tokenizer, config, device, resume_path=None):
    """Train using direct learning (no curriculum)."""
    print("\n" + "="*60)
    print("DIRECT LEARNING MODE")
    print("="*60)

    # Ensure data
    ensure_data_exists(tokenizer)

    # Load and mix data
    with open('data/curriculum/simple_tokens.pkl', 'rb') as f:
        simple_tokens = pickle.load(f)
    with open('data/curriculum/complex_tokens.pkl', 'rb') as f:
        complex_tokens = pickle.load(f)

    # Mix 50-50
    all_tokens = simple_tokens + complex_tokens
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    seq_len = config['training']['seq_len']
    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda'),
    )

    optimizer, scheduler = setup_optimizer(model, config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir='logs/direct',
        checkpoint_dir='checkpoints/direct',
        eval_interval=config['training']['eval_interval'],
        save_interval=config['training']['save_interval'],
        gradient_clip=config['training']['gradient_clip'],
        use_amp=config['training'].get('mixed_precision', True),
    )

    if resume_path and Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)
    else:
        ckpt_dir = Path('checkpoints/direct')
        checkpoints = sorted(ckpt_dir.glob('checkpoint_step_*.pt'))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Auto-resuming from latest checkpoint: {latest}")
            trainer.load_checkpoint(str(latest))

    trainer.train(config['training']['num_epochs'])
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train 100M SLM on WritingPrompts')
    parser.add_argument('--mode', type=str, default='curriculum',
                        choices=['curriculum', 'direct'],
                        help='Training mode')
    parser.add_argument('--model_config', type=str,
                        default='configs/model_config.yaml')
    parser.add_argument('--train_config', type=str,
                        default='configs/training_config.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run smoke test only and exit')
    args = parser.parse_args()

    print("="*60)
    print("  100M SLM Training — WritingPrompts Dataset")
    print("="*60)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {args.mode}")
    print(f"  PID:  {os.getpid()}")

    # Load configs
    print("\nLoading configurations...")
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)

    if args.mode:
        train_config['training']['mode'] = args.mode
    mode = train_config['training']['mode']

    # Setup device
    device = train_config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = 'cpu'
        train_config['training']['mixed_precision'] = False

    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM: {total_mem / 1024**3:.1f} GB")

    # Create tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = create_tokenizer('gpt2', cache_dir='tokenizer_cache')
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Create model
    print("\nCreating model...")
    model = create_model(model_config['model'])
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Run smoke test
    try:
        # Temporarily init data if needed for smoke test
        ensure_data_exists(tokenizer, min_tokens=10000)
        run_smoke_test(model, tokenizer, device,
                       seq_len=model_config['model']['max_seq_len'],
                       batch_size=train_config['training']['batch_size'])
    except AssertionError as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ SMOKE TEST ERROR: {e}")
        raise

    if args.test:
        print("Test mode: exiting after smoke test.")
        sys.exit(0)

    # Ensure full data is present
    print("\nChecking/preparing data...")
    ensure_data_exists(tokenizer, min_tokens=500_000)

    # Print data stats
    with open('data/curriculum/simple_tokens.pkl', 'rb') as f:
        s = pickle.load(f)
    with open('data/curriculum/complex_tokens.pkl', 'rb') as f:
        c = pickle.load(f)
    print(f"  Simple tokens:  {len(s):,}")
    print(f"  Complex tokens: {len(c):,}")
    print(f"  Total tokens:   {len(s)+len(c):,}")

    # Train
    if mode == 'curriculum':
        trainer = train_curriculum_mode(
            model, tokenizer, train_config, device, args.resume
        )
        log_dir = 'logs/curriculum'
    else:
        trainer = train_direct_mode(
            model, tokenizer, train_config, device, args.resume
        )
        log_dir = 'logs/direct'

    # Final evaluation
    print("\nRunning final evaluation...")
    try:
        evaluator = Evaluator(model, tokenizer, device)
        val_loader = trainer.val_loader
        final_metrics = evaluator.evaluate_and_sample(
            val_loader,
            prompts=train_config.get('eval_prompts', [
                "Once upon a time",
                "The hero looked at the horizon",
                "In a world where magic",
            ])
        )
        print_evaluation_results(final_metrics, f"Final {mode.capitalize()} Results")
    except Exception as e:
        print(f"⚠ Final evaluation failed: {e}")

    # Create visualizations
    print("\nCreating visualizations...")
    try:
        viz = Visualizer('plots')
        viz.create_all_plots(log_dir, mode)
    except Exception as e:
        print(f"⚠ Visualization failed (non-critical): {e}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"  Mode:        {mode}")
    print(f"  Logs:        {log_dir}")
    print(f"  Checkpoints: checkpoints/{mode}")
    print(f"  Best val:    {trainer.best_val_loss:.4f}")
    print(f"  Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()