"""
train_curriculum.py — Curriculum-driven training loop and helpers.

This module implements the curriculum training loop used both for base-stage
curriculum training and for expansion-stage fine-tuning. Key responsibilities:
    - Build model and optimizer, set up mixed precision
    - Construct curriculum datasets (`CurriculumStageDataset`) and optional replay
        mixing (replay pool support)
    - Manage an adaptive `CompetenceScheduler` that controls eligible-data fraction
    - Provide evaluation helpers: `evaluate`, `evaluate_by_tier`
    - Early-exit detectors: `PlateauDetector`, `SpikeDetector`
    - LR scheduler utilities and differential-LR support for expansion training
    - Checkpoint helpers (`save_checkpoint`, `load_checkpoint`) — `save_checkpoint`
        now stores an optional `anchor_val` (TinyStories baseline) to detect forgetting

New/modified features (compared to earlier versions):
    - Anchor baseline: compute and persist TinyStories (`s0`) validation loss at
        the start of training; use it to compute `ts_forgetting` during training and
        drive adaptive replay policies
    - Replay pool sampling: `CurriculumStageDataset` supports loading cached
        replay chunks and `set_replay_fraction()` to dynamically control how often
        replay samples are drawn
    - Extended logging: training logs include `replay_frac`, `ts_forgetting`, and
        `grad_norm` columns written by `src.logger.TrainingLogger`

Usage examples:
    python train_curriculum.py --config configs/stage0_full.yaml --tokenizer tokenizers/tokenizer_corpus.json

"""

import os
import math
import time
import argparse
import yaml
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import numpy as np

from src.model import SLM, SLMConfig
from src.curriculum_dataset import (
    CurriculumStageDataset,
    CompetenceScheduler,
    make_curriculum_dataloader,
)
from src.dataset import load_all_val_sets
from src.logger import TrainingLogger
from tokenizers import Tokenizer


# ─── Val loss computation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: SLM, loader, device: str, max_batches: int = 50,
    vocab_size: int = 50000,
) -> float:
    """Evaluate model on a validation DataLoader."""
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if x.shape[1] > model.cfg.ctx_len:
            x = x[:, :model.cfg.ctx_len]
            y = y[:, :model.cfg.ctx_len]
        x = x.clamp(0, vocab_size - 1)
        y = torch.where(y == -1, y, y.clamp(0, vocab_size - 1))
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ─── Early exit helpers ───────────────────────────────────────────────────────

class PlateauDetector:
    """Fires when val loss hasn't improved by min_delta over patience evals."""
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class SpikeDetector:
    """Fires when train loss increases by more than threshold over a window."""
    def __init__(self, window: int, threshold: float):
        self.window = deque(maxlen=window)
        self.threshold = threshold

    def update(self, train_loss: float) -> bool:
        self.window.append(train_loss)
        if len(self.window) < self.window.maxlen:
            return False
        baseline = min(list(self.window)[: self.window.maxlen // 2])
        return (train_loss - baseline) > self.threshold


# ─── Dynamic Block Size & Divergence (Addendum features) ──────────────────────

def get_dynamic_block_size(curriculum_fraction: float, max_block: int = 256) -> int:
    """Soft token budget: forces learning local syntax before long-context.
    
    Schedule scales with max_block to support expansion stages:
      - max_block <= 256: [64, 128, 192, 256]
      - max_block <= 384: [64, 128, 192, 256, 384]
      - max_block <= 512: [64, 128, 192, 256, 384, 512]
      - max_block <= 768: [64, 128, 256, 384, 512, 768]
    """
    if max_block <= 256:
        BUDGET_SCHEDULE = [64, 128, 192, 256]
    elif max_block <= 384:
        BUDGET_SCHEDULE = [64, 128, 192, 256, 384]
    elif max_block <= 512:
        BUDGET_SCHEDULE = [64, 128, 192, 256, 384, 512]
    else:
        BUDGET_SCHEDULE = [64, 128, 256, 384, 512, 768]
    idx = min(int(curriculum_fraction * len(BUDGET_SCHEDULE)), len(BUDGET_SCHEDULE) - 1)
    return min(BUDGET_SCHEDULE[idx], max_block)

def kv_divergence_metric(model: SLM, x: torch.Tensor) -> float:
    """K=V divergence on global attention block as uncertainty signal."""
    last_block = list(model.layers)[-1]
    kv_pairs = []

    def hook_fn(module, inp, out):
        # out is the QKV concatenated tensor
        C = out.size(-1) // 3
        k = out[..., C:2*C]
        v = out[..., 2*C:]
        kv_pairs.append((k.detach(), v.detach()))

    h = last_block.attn.qkv_proj.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        model(x)
    h.remove()
    model.train()

    if not kv_pairs:
        return 0.0

    k, v = kv_pairs[0]
    cos_sim = torch.nn.functional.cosine_similarity(
        k.reshape(-1, k.size(-1)), v.reshape(-1, v.size(-1)), dim=-1
    ).mean().item()
    return 1.0 - cos_sim


# ─── LR schedule (cosine with warmup) ────────────────────────────────────────

def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    path, model, optimizer, scheduler_state, step, tokens_seen,
    val_loss, curriculum_state=None, anchor_val=None, forgetting_ema=None,
):
    data = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": val_loss,
        "config": model.cfg,
    }
    if curriculum_state is not None:
        data["curriculum_state"] = curriculum_state
    if anchor_val is not None:
        data["anchor_val"] = float(anchor_val)
    if forgetting_ema is not None:
        data["forgetting_ema"] = float(forgetting_ema)
    torch.save(data, path)
    print(f"[train] Checkpoint saved → {path}  (val={val_loss:.4f})")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[train] Resumed from {path}  "
          f"(step={ckpt['step']}, val={ckpt['best_val_loss']:.4f})")
    return ckpt


# ─── Per-tier evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_by_tier(
    model: SLM, dataset: CurriculumStageDataset, device: str,
    vocab_size: int, n_samples: int = 100,
) -> dict:
    """
    Evaluate model on easy/medium/hard tiers separately.
    Returns dict: {"easy": loss, "medium": loss, "hard": loss}
    """
    model.eval()
    n_total = len(dataset.chunks)
    n_eligible = len(dataset.sorted_indices)

    tiers = {
        "easy": (0, int(n_eligible * 0.3)),
        "medium": (int(n_eligible * 0.3), int(n_eligible * 0.7)),
        "hard": (int(n_eligible * 0.7), n_eligible),
    }

    results = {}
    for tier_name, (start, end) in tiers.items():
        if end <= start:
            results[tier_name] = float("nan")
            continue

        tier_losses = []
        sample_indices = torch.randint(start, end, (min(n_samples, end - start),))

        for idx in sample_indices:
            chunk_idx = dataset.sorted_indices[idx.item()]
            chunk = torch.tensor(
                dataset.chunks[chunk_idx], dtype=torch.long
            ).unsqueeze(0).to(device)

            x = chunk[:, :-1].clamp(0, vocab_size - 1)
            y = chunk[:, 1:]
            y = torch.where(y == -1, y, y.clamp(0, vocab_size - 1))

            if x.shape[1] > model.cfg.ctx_len:
                x = x[:, :model.cfg.ctx_len]
                y = y[:, :model.cfg.ctx_len]

            _, loss = model(x, y)
            tier_losses.append(loss.item())

        results[tier_name] = sum(tier_losses) / max(len(tier_losses), 1)

    model.train()
    return results


# ─── Main training function ───────────────────────────────────────────────────

def train(args):
    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    stage = str(cfg_dict["stage"])
    dataset_name = cfg_dict["dataset"]
    val_key = cfg_dict.get("val_key", "s0")
    seq_len = int(cfg_dict["seq_len"])
    max_tokens = int(str(cfg_dict["max_tokens"]).replace("_", ""))
    batch_size = int(cfg_dict["batch_size"])
    eval_interval = int(cfg_dict["eval_interval"])
    patience = int(cfg_dict["patience"])
    min_delta = float(cfg_dict["min_delta"])
    spike_thresh = float(cfg_dict["spike_threshold"])
    spike_window = int(cfg_dict["spike_window"])
    lr_max = float(cfg_dict["learning_rate"])
    lr_min = float(cfg_dict["lr_min"])
    warmup_steps = int(cfg_dict["lr_warmup_steps"])
    weight_decay = float(cfg_dict["weight_decay"])
    grad_clip = float(cfg_dict["grad_clip"])

    # Curriculum settings (with defaults for non-curriculum configs)
    curriculum_mode = args.curriculum_mode or cfg_dict.get("curriculum_mode", "adaptive")
    initial_fraction = float(cfg_dict.get("initial_fraction", 0.15))
    scores_path = cfg_dict.get("scores_path", "curriculum_scores.npy")
    # Anti-forgetting / replay mapping parameters
    forgetting_ema_alpha = float(cfg_dict.get("forgetting_ema_alpha", 0.3))
    replay_cap = float(cfg_dict.get("replay_cap", 0.3))
    replay_scale = float(cfg_dict.get("replay_scale", 2.0))
    min_replay = float(cfg_dict.get("min_replay", 0.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  CURRICULUM LEARNING TRAINING — Stage {stage}")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Seq len:    {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Mode:       {curriculum_mode.upper()}")
    print(f"  Initial %:  {initial_fraction*100:.0f}%")
    print(f"{'='*70}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[train] Tokenizer: {args.tokenizer} (vocab={vocab_size:,})")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = SLMConfig(
        vocab_size=vocab_size,
        pos_type="learnable",
        ctx_len=seq_len,
    )
    model = SLM(model_cfg).to(device)
    print(f"[train] Model params: {model.num_params()/1e6:.1f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.dim() < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr_max,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device=device, enabled=(not use_bf16))

    # ── Curriculum dataset ────────────────────────────────────────────────────
    train_ds = CurriculumStageDataset().build(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_tokens=max_tokens,
        cache_dir=args.cache_dir,
        scores_path=scores_path,
        mode=curriculum_mode,
        initial_fraction=initial_fraction,
        # Optionally include replay sources configured in YAML
        replay_sources=cfg_dict.get("replay_sources", None),
        initial_replay_fraction=float(cfg_dict.get("initial_replay_fraction", 0.0)),
    )

    # ── Curriculum scheduler (only for adaptive mode) ─────────────────────────
    scheduler = None
    if curriculum_mode == "adaptive":
        scheduler = CompetenceScheduler(
            initial_fraction=initial_fraction,
        )
        print(f"[train] CompetenceScheduler initialized "
              f"(start={initial_fraction*100:.0f}%)")

    # ── Validation loaders ────────────────────────────────────────────────────
    val_loaders = load_all_val_sets(tokenizer, cache_dir=args.cache_dir)

    # ── Anchor baseline (for forgetting detection) ───────────────────────────
    initial_anchor_val = None
    if 's0' in val_loaders:
        try:
            initial_anchor_val = evaluate(model, val_loaders['s0'], device, vocab_size=vocab_size)
            print(f"[train] Initial anchor (s0) val: {initial_anchor_val:.4f}")
        except Exception as e:
            print(f"[train] Failed to compute initial anchor val: {e}")
    # EMA of forgetting signal (smoothed): initialized from checkpoint if present
    forgetting_ema = None

    # ── Compute total steps ───────────────────────────────────────────────────
    tokens_per_step = batch_size * seq_len
    max_steps = max_tokens // tokens_per_step
    print(f"[train] max_steps={max_steps:,}  tokens/step={tokens_per_step:,}")

    # ── Resume or load checkpoint ─────────────────────────────────────────────
    start_step = 0
    tokens_seen = 0
    best_val = float("inf")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(
        args.checkpoint_dir, f"stage{stage}_curriculum_{curriculum_mode}_best.pt"
    )

    if args.resume and os.path.exists(best_ckpt_path):
        ckpt = load_checkpoint(best_ckpt_path, model, optimizer)
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        best_val = ckpt["best_val_loss"]
        # Restore curriculum state
        if scheduler and "curriculum_state" in ckpt:
            scheduler.load_state_dict(ckpt["curriculum_state"])
            train_ds.set_eligible_fraction(scheduler.get_current_fraction())
            print(f"[train] Restored curriculum fraction: "
                  f"{scheduler.get_current_fraction()*100:.1f}%")
        # Restore anchor baseline for forgetting detection if present
        if "anchor_val" in ckpt:
            try:
                initial_anchor_val = float(ckpt.get("anchor_val"))
                print(f"[train] Restored anchor baseline (s0): {initial_anchor_val:.4f}")
            except Exception:
                pass
        # Restore forgetting EMA if persisted
        if "forgetting_ema" in ckpt:
            try:
                forgetting_ema = float(ckpt.get("forgetting_ema"))
                print(f"[train] Restored forgetting EMA: {forgetting_ema:.6f}")
            except Exception:
                pass
    elif args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
        print(f"[train] Loading weights from: {args.prev_checkpoint}")
        ckpt = torch.load(args.prev_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    # ── Exit detectors ────────────────────────────────────────────────────────
    plateau = PlateauDetector(patience=patience, min_delta=min_delta)
    spike = SpikeDetector(window=spike_window, threshold=spike_thresh)
    logger = TrainingLogger(stage=stage, log_dir=args.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    exit_reason = None
    loss_window = []
    pbar = tqdm(
        total=max_steps, initial=start_step,
        desc=f"Stage {stage} [{curriculum_mode}]", unit="step",
    )

    # Track curriculum metrics for logging
    curriculum_log = []
    deep_layer_grads = []

    while True:
        # Rebuild dataloader each "epoch" to respect updated eligible count
        train_loader = make_curriculum_dataloader(
            train_ds, batch_size=batch_size, shuffle=True,
        )

        for x, y in train_loader:
            if step >= max_steps:
                exit_reason = "token_budget"
                break

            x, y = x.to(device), y.to(device)

            fraction_pct = (
                scheduler.get_current_fraction()
                if scheduler
                else (train_ds.eligible_count / len(train_ds.chunks))
            )
            dyn_ctx = get_dynamic_block_size(fraction_pct, max_block=model.cfg.ctx_len)

            x = x[:, :dyn_ctx]
            y = y[:, :dyn_ctx]

            # Safeguard: clamp token IDs
            x = x.clamp(0, vocab_size - 1)
            y = torch.where(y == -1, y, y.clamp(0, vocab_size - 1))

            # LR update
            lr = get_lr(step, warmup_steps, lr_max, lr_min, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)

            if use_bf16:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Record deep layer grad norms for stability tracking
            blocks = list(model.layers)
            norms = [
                p.grad.norm().item()
                for layer in blocks[-2:]
                for p in layer.parameters()
                if getattr(p, 'grad', None) is not None
            ]
            if norms:
                deep_layer_grads.append(sum(norms) / len(norms))
            if len(deep_layer_grads) > 100:
                deep_layer_grads.pop(0)

            # Global grad norm for monitoring
            grads = [p.grad.norm().item() for p in model.parameters() if getattr(p, 'grad', None) is not None]
            global_grad_norm = sum(grads) / max(len(grads), 1) if grads else float('nan')

            if use_bf16:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            tokens_seen += tokens_per_step
            loss_window.append(loss.item())
            if len(loss_window) > 50:
                loss_window.pop(0)
            train_loss = sum(loss_window) / len(loss_window)

            # Update progress bar
            fraction_pct = (
                scheduler.get_current_fraction() * 100
                if scheduler
                else (train_ds.eligible_count / len(train_ds.chunks) * 100)
            )
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{train_loss:.3f}",
                "lr": f"{lr:.1e}",
                "data%": f"{fraction_pct:.0f}%",
            })

            # Spike check
            if spike.update(train_loss):
                exit_reason = "loss_spike"
                break

            # ── Evaluation ────────────────────────────────────────────────────
            if step % eval_interval == 0 and step > 0:
                # Compute val losses on all datasets
                val_losses = {
                    k: evaluate(model, loader, device, vocab_size=vocab_size)
                    for k, loader in val_loaders.items()
                }
                current_val = val_losses.get(val_key, val_losses.get("s0", 0))

                # Anchor (TinyStories) loss and forgetting metric placeholder
                ts_loss = val_losses.get('s0', None)
                forgetting = float('nan')

                # Assess deep layer stability
                deep_stable = True
                if len(deep_layer_grads) > 20:
                    cv = np.std(deep_layer_grads) / (np.mean(deep_layer_grads) + 1e-8)
                    deep_stable = cv < 0.15

                kv_div = float('nan')
                # ── Update curriculum schedule ────────────────────────────────
                if curriculum_mode == "adaptive" and scheduler:
                    # evaluate internal KV divergence 
                    kv_div = kv_divergence_metric(model, x)
                    if kv_div > 0.4:
                        print(f"  [curriculum] High K=V divergence ({kv_div:.2f}) -> Adding patience")
                        scheduler.add_patience(2)

                    info = scheduler.update_competence(current_val, deep_layers_stable=deep_stable)
                    train_ds.set_eligible_fraction(info["fraction"])

                    # Adaptive replay fraction policy:
                    # If forgetting observed on anchor domain (s0) increase replay
                    if ts_loss is not None and initial_anchor_val is not None:
                        # Raw relative forgetting: positive means worse than source
                        raw_forgetting = (ts_loss - initial_anchor_val) / max(initial_anchor_val, 1e-6)

                        # Update EMA of forgetting to smooth noisy small-val signals
                        if forgetting_ema is None:
                            forgetting_ema = float(raw_forgetting)
                        else:
                            forgetting_ema = (
                                forgetting_ema_alpha * float(raw_forgetting)
                                + (1.0 - forgetting_ema_alpha) * forgetting_ema
                            )

                        # Use the (non-negative) EMA to map to replay fraction
                        use_forgetting = max(0.0, forgetting_ema)
                        new_replay = min(replay_cap, max(min_replay, use_forgetting * replay_scale))
                        train_ds.set_replay_fraction(new_replay)
                        forgetting = float(raw_forgetting)
                        print(
                            f"[replay] Adjusted replay_frac → {train_ds.replay_frac:.3f} "
                            f"(raw={raw_forgetting:.3f}, ema={forgetting_ema:.3f})"
                        )

                    # Rebuild loader with new eligible count
                    curriculum_log.append({
                        "step": step,
                        "fraction": info["fraction"],
                        "status": info["status"],
                        "improvement_rate": info["improvement_rate"],
                    })

                    print(
                        f"\n  [curriculum] {info['status'].upper()} | "
                        f"fraction: {info['old_fraction']*100:.1f}% → "
                        f"{info['fraction']*100:.1f}% "
                        f"(+{info['expansion']*100:.1f}%) | "
                        f"improvement: {info['improvement_rate']*100:+.2f}% | "
                        f"eligible: {train_ds.eligible_count:,}/{len(train_ds.chunks):,}"
                    )

                elif curriculum_mode in ("length", "perplexity"):
                    train_ds.set_eligible_from_step(step, max_steps)

                # Per-tier evaluation (every eval interval for logging)
                tier_results = evaluate_by_tier(
                    model, train_ds, device, vocab_size
                )
                tier_str = "  ".join(
                    f"{k}={v:.3f}" for k, v in tier_results.items()
                )
                print(f"  [tiers] {tier_str}")

                # Save best checkpoint
                if current_val < best_val:
                    best_val = current_val
                    save_checkpoint(
                        best_ckpt_path, model, optimizer,
                        {"lr": lr}, step, tokens_seen, best_val,
                        curriculum_state=(
                            scheduler.state_dict() if scheduler else None
                        ),
                        anchor_val=initial_anchor_val,
                        forgetting_ema=forgetting_ema,
                    )

                curr_metrics = {
                    "tier_easy": tier_results.get("easy", float('nan')),
                    "tier_medium": tier_results.get("medium", float('nan')),
                    "tier_hard": tier_results.get("hard", float('nan')),
                    "fraction": scheduler.get_current_fraction() if scheduler else float('nan'),
                    "kv_div": kv_div,
                    "replay_frac": getattr(train_ds, 'replay_frac', float('nan')),
                    "anchor_reg": float('nan'),
                    "ts_forgetting": forgetting,
                    "ts_forgetting_ema": (forgetting_ema if forgetting_ema is not None else float('nan')),
                    "grad_norm": global_grad_norm,
                }
                logger.log(step, tokens_seen, train_loss, val_losses, lr, curriculum=curr_metrics)

                # Plateau check
                if plateau.update(current_val):
                    exit_reason = "plateau"
                    break

            # ── Non-adaptive curriculum update (per-step) ─────────────────────
            if curriculum_mode in ("length", "perplexity") and step % 100 == 0:
                train_ds.set_eligible_from_step(step, max_steps)

            step += 1

        if exit_reason:
            break

    pbar.close()
    logger.log_exit(exit_reason, step, tokens_seen)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Exit reason:  {exit_reason}")
    print(f"  Total steps:  {step:,}")
    print(f"  Tokens seen:  {tokens_seen/1e6:.1f}M")
    print(f"  Best val:     {best_val:.4f}")
    print(f"  Checkpoint:   {best_ckpt_path}")

    if curriculum_mode == "adaptive" and scheduler:
        print(f"\n  Curriculum trajectory:")
        print(f"    Initial fraction: {initial_fraction*100:.0f}%")
        print(f"    Final fraction:   {scheduler.get_current_fraction()*100:.1f}%")
        print(f"    Expansions:       {len(scheduler.fraction_history)}")

        if curriculum_log:
            print(f"\n  Curriculum log (last 10 updates):")
            for entry in curriculum_log[-10:]:
                print(
                    f"    step={entry['step']:>6}  "
                    f"fraction={entry['fraction']*100:>5.1f}%  "
                    f"status={entry['status']:<20}  "
                    f"improvement={entry['improvement_rate']*100:+.2f}%"
                )

    print(f"{'='*70}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SLM with curriculum learning on TinyStories"
    )
    p.add_argument("--config", type=str, required=True,
                    help="Path to YAML config file")
    p.add_argument("--tokenizer", type=str, required=True,
                    help="Path to tokenizer JSON file")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="Directory for saving checkpoints")
    p.add_argument("--log_dir", type=str, default="logs",
                    help="Directory for training logs")
    p.add_argument("--cache_dir", type=str, default="cache",
                    help="Directory for dataset caches")
    p.add_argument("--prev_checkpoint", type=str, default=None,
                    help="Path to checkpoint to initialize weights from")
    p.add_argument("--resume", action="store_true",
                    help="Resume training from last best checkpoint")
    p.add_argument("--curriculum_mode", type=str, default=None,
                    choices=["random", "length", "perplexity", "adaptive"],
                    help="Override curriculum mode from config")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
