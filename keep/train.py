"""
train.py — Single-stage training loop.

Features:
  - Three early-exit conditions (plateau / token budget / loss spike)
  - All three val losses logged at every eval step
  - Best checkpoint saved immediately on improvement
  - Resume support (--resume flag)

Usage:
  python train.py --stage 0 --config configs/stage0.yaml \
                  --tokenizer tokenizers/tokenizer_50k.json \
                  --checkpoint_dir checkpoints/ \
                  --prev_checkpoint checkpoints/stage0_best.pt   # for stage 1+
"""

import os
import math
import time
import argparse
import yaml
from collections import deque
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from model    import SLM, SLMConfig
from dataset  import StreamingStageDataset, load_all_val_sets, make_dataloader
from logger   import TrainingLogger
from tokenizers import Tokenizer


# ─── Val loss computation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: SLM, loader, device: str, max_batches: int = 50) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ─── Early exit helpers ───────────────────────────────────────────────────────

class PlateauDetector:
    """Fires when val loss hasn't improved by min_delta over `patience` evals."""
    def __init__(self, patience: int, min_delta: float):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def update(self, val_loss: float) -> bool:
        """Returns True if plateau detected (exit signal)."""
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class SpikeDetector:
    """Fires when train loss increases by more than threshold over a window."""
    def __init__(self, window: int, threshold: float):
        self.window    = deque(maxlen=window)
        self.threshold = threshold

    def update(self, train_loss: float) -> bool:
        self.window.append(train_loss)
        if len(self.window) < self.window.maxlen:
            return False
        baseline = min(list(self.window)[: self.window.maxlen // 2])
        current  = train_loss
        return (current - baseline) > self.threshold


# ─── LR schedule (cosine with warmup) ────────────────────────────────────────

def get_lr(step: int, warmup: int, max_lr: float, min_lr: float,
           total_steps: int) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(path: str, model: SLM, optimizer, scheduler_state: dict,
                    step: int, tokens_seen: int, val_loss: float):
    torch.save({
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "step"           : step,
        "tokens_seen"    : tokens_seen,
        "best_val_loss"  : val_loss,
        "config"         : model.cfg,
    }, path)
    print(f"[train] Checkpoint saved → {path}  (val={val_loss:.4f})")


def load_checkpoint(path: str, model: SLM, optimizer) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[train] Resumed from {path}  (step={ckpt['step']}, val={ckpt['best_val_loss']:.4f})")
    return ckpt


# ─── Main training function ───────────────────────────────────────────────────

def train(args):
    # Load config
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    stage          = int(cfg_dict["stage"])
    seq_len        = int(cfg_dict["seq_len"])
    max_tokens     = int(str(cfg_dict["max_tokens"]).replace("_", ""))
    batch_size     = int(cfg_dict["batch_size"])
    eval_interval  = int(cfg_dict["eval_interval"])
    patience       = int(cfg_dict["patience"])
    min_delta      = float(cfg_dict["min_delta"])
    spike_thresh   = float(cfg_dict["spike_threshold"])
    spike_window   = int(cfg_dict["spike_window"])
    lr_max         = float(cfg_dict["learning_rate"])
    lr_min         = float(cfg_dict["lr_min"])
    warmup_steps   = int(cfg_dict["lr_warmup_steps"])
    weight_decay   = float(cfg_dict["weight_decay"])
    grad_clip      = float(cfg_dict["grad_clip"])
    # replay_ratio   = float(cfg_dict.get("replay_ratio", 0.0))  # default: no replay

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Stage {stage} | device={device} | seq_len={seq_len}")

    # Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # Model
    model_cfg = SLMConfig(
        vocab_size = vocab_size,
        pos_type   = args.pos_type,
        ctx_len    = 512,      # always build with max context
    )
    model = SLM(model_cfg).to(device)
    print(f"[train] Model params: {model.num_params()/1e6:.1f}M")

    # Optimizer
    # Separate weight decay: apply only to 2D params (not norms/biases)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.95), eps=1e-8)

    # AMP scaler (bf16 on modern CUDA, fp16 fallback)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float16
    scaler   = GradScaler(device=device, enabled=(not use_bf16))

    # Resume or load from previous stage
    start_step   = 0
    tokens_seen  = 0
    best_val     = float("inf")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"stage{stage}_best.pt")

    if args.resume and os.path.exists(best_ckpt_path):
        ckpt        = load_checkpoint(best_ckpt_path, model, optimizer)
        start_step  = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        best_val    = ckpt["best_val_loss"]
    elif args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
        print(f"[train] Loading weights from prev stage: {args.prev_checkpoint}")
        ckpt = torch.load(args.prev_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    # Dataset + loaders
    train_ds = StreamingStageDataset().build(
        stage     = stage,
        tokenizer = tokenizer,
        seq_len   = seq_len,
        max_tokens= max_tokens,
        cache_dir = args.cache_dir,
        # replay_ratio = replay_ratio,
    )
    train_loader = make_dataloader(train_ds, batch_size=batch_size)
    val_loaders  = load_all_val_sets(tokenizer, cache_dir=args.cache_dir)

    # Compute total steps for LR schedule
    tokens_per_step = batch_size * seq_len
    max_steps       = max_tokens // tokens_per_step
    print(f"[train] max_steps={max_steps:,}  tokens/step={tokens_per_step:,}")

    # Exit detectors
    plateau = PlateauDetector(patience=patience, min_delta=min_delta)
    spike   = SpikeDetector(window=spike_window, threshold=spike_thresh)
    logger  = TrainingLogger(stage=stage, log_dir=args.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step         = start_step
    exit_reason  = None

    pbar = tqdm(total=max_steps, initial=start_step, unit="step", desc=f"[Stage {stage}]")

    while True:
        for x, y in train_loader:
            if step >= max_steps:
                exit_reason = "token_budget"
                break

            x, y = x.to(device), y.to(device)

            # LR update
            lr = get_lr(step, warmup_steps, lr_max, lr_min, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=dtype):
                _, loss = model(x, y)

            if use_bf16:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            tokens_seen += tokens_per_step
            train_loss   = loss.item()

            # Spike check
            if spike.update(train_loss):
                exit_reason = "loss_spike"
                break

            # Eval
            if step % eval_interval == 0 and step > 0:
                val_losses = {
                    k: evaluate(model, loader, device)
                    for k, loader in val_loaders.items()
                }
                current_val = val_losses[f"s{stage}"]

                # Save best checkpoint
                if current_val < best_val:
                    best_val = current_val
                    save_checkpoint(
                        best_ckpt_path, model, optimizer,
                        {"lr": lr}, step, tokens_seen, best_val,
                    )

                logger.log(step, tokens_seen, train_loss, val_losses, lr)

                # Plateau check (on current stage's val loss)
                if plateau.update(current_val):
                    exit_reason = "plateau"
                    break

            pbar.update(1)
            step += 1

        if exit_reason:
            break

        # Epoch ended without hitting any exit — continue if tokens remain
        if tokens_seen >= max_tokens:
            exit_reason = "token_budget"
            break

    pbar.close()
    logger.log_exit(exit_reason, step, tokens_seen)
    print(f"[train] Stage {stage} complete. Best val: {best_val:.4f}")
    print(f"[train] Best checkpoint: {best_ckpt_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",            type=int, required=True)
    p.add_argument("--config",           type=str, required=True)
    p.add_argument("--tokenizer",        type=str, required=True)
    p.add_argument("--checkpoint_dir",   type=str, default="checkpoints")
    p.add_argument("--log_dir",          type=str, default="logs")
    p.add_argument("--cache_dir",        type=str, default="cache")
    p.add_argument("--prev_checkpoint",  type=str, default=None,
                   help="Path to best checkpoint from previous stage")
    p.add_argument("--resume",           action="store_true",
                   help="Resume current stage from its best checkpoint")
    p.add_argument("--pos_type",         type=str, default="learnable",
                   choices=["learnable", "rope"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
