"""
train_expansion.py — Orchestrates one expansion stage end-to-end.

This script runs a single expansion stage defined in `configs/expansion_stages.yaml`.
High-level flow implemented by this file:
    1) Parse CLI args (stage, config, tokenizer, dirs)
    2) Load stage config (`stage_<stage>` key) from YAML
    3) Load tokenizer and source checkpoint (torch.load) and instantiate `SLM` with the
         checkpoint `config` + `model_state`
    4) Apply function-preserving expansions as requested by the config using
         `expand_model.expand_depth`, `expand_model.expand_ffn_width`, and
         `expand_model.expand_context_length` — each expansion is validated via
         `expand_model.validate_expansion`
    5) Build training components: differential-LR optimizer (`build_expansion_optimizer`),
         mixed-precision (`GradScaler` / `autocast`), curriculum dataset (`CurriculumStageDataset`),
         and optional `CompetenceScheduler` for adaptive curricula
    6) Train with a curriculum-aware dataloader and dynamic context sizing
         (uses `get_dynamic_block_size`, per-step LR updates via `update_lr_groups`),
         perform periodic evaluation (`evaluate`) and per-tier analysis (`evaluate_by_tier`)
    7) Checkpoint best models using `save_checkpoint` (from `train_curriculum`) —
         checkpoints now optionally include an `anchor_val` (TinyStories baseline)
         which is used to detect forgetting on earlier domains
    8) After training completes, run capability logging (`src.capability_logger.run_capability_logging`)
         to append cross-domain metrics and stylized samples to `docs/curriculum_capabilities.md`

Notes / interactions with other modules:
- Curriculum and replay policies are implemented in `src.curriculum_dataset` and
    `train_curriculum` (CompetenceScheduler). The expansion script delegates curriculum
    training primitives (evaluation, tiered eval, LR schedules, detection utilities)
    to `train_curriculum` helper functions.
- Logging is handled by `src.logger.TrainingLogger` which writes a per-stage CSV with
    extended columns (replay_frac, ts_forgetting, grad_norm, etc.).

Usage examples (unchanged):
    python train_expansion.py --stage 3 --tokenizer tokenizers/tokenizer_corpus.json

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

from expand_model import (
    expand_depth,
    expand_ffn_width,
    expand_context_length,
    validate_expansion,
    create_expansion_checkpoint,
)
from train_curriculum import (
    evaluate,
    evaluate_by_tier,
    PlateauDetector,
    SpikeDetector,
    get_dynamic_block_size,
    kv_divergence_metric,
    get_lr,
    save_checkpoint,
)
from src.capability_logger import run_capability_logging



# ─── Differential Learning Rate Optimizer ─────────────────────────────────────

def build_expansion_optimizer(
    model: SLM,
    n_pretrained_layers: int,
    lr_max: float,
    pretrained_lr_mult: float,
    new_lr_mult: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Build optimizer with differential learning rates for expansion training.

    Pretrained layers get a lower learning rate (pretrained_lr_mult * lr_max).
    New layers get the full learning rate (new_lr_mult * lr_max).
    Non-layer parameters (embeddings, norms) use pretrained rate.
    """
    # Categorize parameters
    pretrained_decay = []
    pretrained_no_decay = []
    new_decay = []
    new_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if this param belongs to a new layer
        is_new = False
        if "layers." in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            if layer_idx >= n_pretrained_layers:
                is_new = True

        # Classify by decay/no-decay
        if param.dim() >= 2:
            if is_new:
                new_decay.append(param)
            else:
                pretrained_decay.append(param)
        else:
            if is_new:
                new_no_decay.append(param)
            else:
                pretrained_no_decay.append(param)

    param_groups = [
        {
            "params": pretrained_decay,
            "weight_decay": weight_decay,
            "lr": lr_max * pretrained_lr_mult,
            "group_name": "pretrained_decay",
        },
        {
            "params": pretrained_no_decay,
            "weight_decay": 0.0,
            "lr": lr_max * pretrained_lr_mult,
            "group_name": "pretrained_no_decay",
        },
        {
            "params": new_decay,
            "weight_decay": weight_decay,
            "lr": lr_max * new_lr_mult,
            "group_name": "new_decay",
        },
        {
            "params": new_no_decay,
            "weight_decay": 0.0,
            "lr": lr_max * new_lr_mult,
            "group_name": "new_no_decay",
        },
    ]

    # Remove empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr_max,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Log group sizes
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"[optimizer] {g['group_name']}: "
              f"{n_params/1e6:.2f}M params @ lr={g['lr']:.2e}")

    return optimizer


# ─── LR update with differential scaling ─────────────────────────────────────

def update_lr_groups(
    optimizer: torch.optim.AdamW,
    step: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float,
    total_steps: int,
    pretrained_lr_mult: float,
    new_lr_mult: float,
):
    """Update LR for each param group with differential scaling."""
    base_lr = get_lr(step, warmup_steps, lr_max, lr_min, total_steps)

    for group in optimizer.param_groups:
        name = group.get("group_name", "")
        if "new" in name:
            group["lr"] = base_lr * new_lr_mult
        else:
            group["lr"] = base_lr * pretrained_lr_mult

    return base_lr


# ─── Main expansion training function ────────────────────────────────────────

def train_expansion(args):
    # ── Load expansion config ────────────────────────────────────────────────
    with open(args.config) as f:
        all_configs = yaml.safe_load(f)

    stage_key = f"stage_{args.stage.lower()}"
    if stage_key not in all_configs:
        raise ValueError(
            f"Stage '{args.stage}' not found in config. "
            f"Available: {list(all_configs.keys())}"
        )

    cfg = all_configs[stage_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  EXPANSION TRAINING — Stage {args.stage}")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Source:     {cfg['source_checkpoint']}")
    print(f"  Target:     n_layers={cfg['n_layers']}, d_ff={cfg['d_ff']}, "
          f"ctx={cfg['ctx_len']}")
    print(f"{'='*70}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[train] Tokenizer: {args.tokenizer} (vocab={vocab_size:,})")

    # ── Load source checkpoint ───────────────────────────────────────────────
    source_path = cfg["source_checkpoint"]
    print(f"[train] Loading source: {source_path}")
    ckpt = torch.load(source_path, map_location=device, weights_only=False)
    old_cfg = ckpt["config"]
    model = SLM(old_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[train] Source model: {model.num_params()/1e6:.1f}M params, "
          f"{old_cfg.n_layers}L, d_ff={old_cfg.d_ff}")

    # ── Apply expansions ─────────────────────────────────────────────────────
    n_pretrained_layers = old_cfg.n_layers

    # Depth expansion
    if "clone_from" in cfg:
        clone_layers = cfg["clone_from"]
        noise_std = cfg.get("noise_std", 0.01)
        print(f"\n[expand] Depth: cloning layers {clone_layers}")
        old_model_for_val = model
        model = expand_depth(model, clone_layers, noise_std).to(device)
        result = validate_expansion(
            old_model_for_val, model, vocab_size, device
        )
        if result["cosine_similarity"] < 0.90:
            print("[WARNING] Low cosine similarity after depth expansion!")
        del old_model_for_val

    # FFN widening
    if "new_d_ff" in cfg:
        new_d_ff = cfg["new_d_ff"]
        print(f"\n[expand] FFN: widening d_ff → {new_d_ff}")
        old_model_for_val = model
        model = expand_ffn_width(model, new_d_ff).to(device)
        result = validate_expansion(
            old_model_for_val, model, vocab_size, device
        )
        if result["max_abs_diff"] > 1e-4:
            print("[WARNING] FFN widening not function-preserving!")
        del old_model_for_val

    # Context expansion
    if "new_ctx_len" in cfg:
        new_ctx_len = cfg["new_ctx_len"]
        if new_ctx_len > model.cfg.ctx_len:
            print(f"\n[expand] Context: {model.cfg.ctx_len} → {new_ctx_len}")
            model = expand_context_length(model, new_ctx_len).to(device)

    print(f"\n[train] Expanded model: {model.num_params()/1e6:.1f}M params, "
          f"{model.cfg.n_layers}L, d_ff={model.cfg.d_ff}, ctx={model.cfg.ctx_len}")

    # ── Training config ──────────────────────────────────────────────────────
    seq_len = int(cfg["seq_len"])
    max_tokens = int(str(cfg["max_tokens"]).replace("_", ""))
    batch_size = int(cfg["batch_size"])
    eval_interval = int(cfg["eval_interval"])
    patience = int(cfg["patience"])
    min_delta = float(cfg["min_delta"])
    spike_thresh = float(cfg["spike_threshold"])
    spike_window = int(cfg["spike_window"])
    lr_max = float(cfg["learning_rate"])
    lr_min = float(cfg["lr_min"])
    warmup_steps = int(cfg["lr_warmup_steps"])
    weight_decay = float(cfg["weight_decay"])
    grad_clip = float(cfg["grad_clip"])
    pretrained_lr_mult = float(cfg.get("pretrained_lr_mult", 0.5))
    new_lr_mult = float(cfg.get("new_lr_mult", 1.0))

    curriculum_mode = cfg.get("curriculum_mode", "adaptive")
    initial_fraction = float(cfg.get("initial_fraction", 0.3))
    scores_path = cfg.get("scores_path", "curriculum_scores.npy")
    dataset_name = cfg["dataset"]
    val_key = cfg.get("val_key", "s0")

    # ── Optimizer with differential LR ───────────────────────────────────────
    optimizer = build_expansion_optimizer(
        model,
        n_pretrained_layers=n_pretrained_layers,
        lr_max=lr_max,
        pretrained_lr_mult=pretrained_lr_mult,
        new_lr_mult=new_lr_mult,
        weight_decay=weight_decay,
    )

    # ── Mixed precision ──────────────────────────────────────────────────────
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device=device, enabled=(not use_bf16))

    # ── Curriculum dataset ───────────────────────────────────────────────────
    train_ds = CurriculumStageDataset().build(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_tokens=max_tokens,
        cache_dir=args.cache_dir,
        scores_path=scores_path,
        mode=curriculum_mode,
        initial_fraction=initial_fraction,
        stage_name=args.stage,
    )

    scheduler = None
    if curriculum_mode == "adaptive":
        scheduler = CompetenceScheduler(initial_fraction=initial_fraction)

    # ── Validation loaders ───────────────────────────────────────────────────
    val_loaders = load_all_val_sets(tokenizer, cache_dir=args.cache_dir)

    # ── Compute steps ────────────────────────────────────────────────────────
    tokens_per_step = batch_size * seq_len
    max_steps = max_tokens // tokens_per_step
    print(f"[train] max_steps={max_steps:,}  tokens/step={tokens_per_step:,}")

    # ── Checkpointing ────────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(
        args.checkpoint_dir, f"stage_{args.stage}_best.pt"
    )

    step = 0
    tokens_seen = 0
    best_val = float("inf")

    # ── Detectors ────────────────────────────────────────────────────────────
    plateau = PlateauDetector(patience=patience, min_delta=min_delta)
    spike = SpikeDetector(window=spike_window, threshold=spike_thresh)
    logger = TrainingLogger(
        stage=f"expansion_{args.stage}",
        log_dir=args.log_dir,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    exit_reason = None
    loss_window = []
    deep_layer_grads = []
    curriculum_log = []

    pbar = tqdm(
        total=max_steps,
        desc=f"Stage {args.stage} expansion",
        unit="step",
    )

    while True:
        train_loader = make_curriculum_dataloader(
            train_ds, batch_size=batch_size, shuffle=True,
        )

        for x, y in train_loader:
            if step >= max_steps:
                exit_reason = "token_budget"
                break

            x, y = x.to(device), y.to(device)

            # Dynamic block size
            fraction_pct = (
                scheduler.get_current_fraction()
                if scheduler
                else (train_ds.eligible_count / len(train_ds.chunks))
            )
            dyn_ctx = get_dynamic_block_size(
                fraction_pct, max_block=model.cfg.ctx_len
            )
            x = x[:, :dyn_ctx]
            y = y[:, :dyn_ctx]

            # Safeguard: clamp token IDs
            x = x.clamp(0, vocab_size - 1)
            y = torch.where(y == -1, y, y.clamp(0, vocab_size - 1))

            # LR update (differential)
            base_lr = update_lr_groups(
                optimizer, step, warmup_steps, lr_max, lr_min,
                max_steps, pretrained_lr_mult, new_lr_mult,
            )

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=device, dtype=dtype, enabled=(device == "cuda")
            ):
                _, loss = model(x, y)

            if use_bf16:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Track deep layer grad norms
            blocks = list(model.layers)
            norms = [
                p.grad.norm().item()
                for layer in blocks[-3:]
                for p in layer.parameters()
                if getattr(p, 'grad', None) is not None
            ]
            if norms:
                deep_layer_grads.append(sum(norms) / len(norms))
            if len(deep_layer_grads) > 100:
                deep_layer_grads.pop(0)

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
            frac_display = (
                scheduler.get_current_fraction() * 100
                if scheduler
                else (train_ds.eligible_count / len(train_ds.chunks) * 100)
            )
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{train_loss:.3f}",
                "lr": f"{base_lr:.1e}",
                "data%": f"{frac_display:.0f}%",
            })

            # Spike check
            if spike.update(train_loss):
                exit_reason = "loss_spike"
                break

            # ── Evaluation ───────────────────────────────────────────────────
            if step % eval_interval == 0 and step > 0:
                val_losses = {
                    k: evaluate(model, loader, device, vocab_size=vocab_size)
                    for k, loader in val_loaders.items()
                }
                current_val = val_losses.get(
                    val_key, val_losses.get("s0", 0)
                )

                # Check TinyStories forgetting
                ts_val = val_losses.get("s0", None)
                if ts_val is not None:
                    source_val = ckpt.get("best_val_loss", float("inf"))
                    forgetting = (ts_val - source_val) / max(source_val, 1e-6)
                    if forgetting > 0.05:
                        print(f"  [WARNING] TinyStories forgetting: "
                              f"{forgetting*100:.1f}% degradation")

                # Deep layer stability
                deep_stable = True
                if len(deep_layer_grads) > 20:
                    cv = np.std(deep_layer_grads) / (
                        np.mean(deep_layer_grads) + 1e-8
                    )
                    deep_stable = cv < 0.15

                kv_div = float('nan')

                # Curriculum update
                if curriculum_mode == "adaptive" and scheduler:
                    kv_div = kv_divergence_metric(model, x)
                    if kv_div > 0.4:
                        print(f"  [curriculum] High K=V divergence "
                              f"({kv_div:.2f}) -> Adding patience")
                        scheduler.add_patience(2)

                    info = scheduler.update_competence(
                        current_val, deep_layers_stable=deep_stable
                    )
                    train_ds.set_eligible_fraction(info["fraction"])

                    curriculum_log.append({
                        "step": step,
                        "fraction": info["fraction"],
                        "status": info["status"],
                    })

                    print(
                        f"\n  [curriculum] {info['status'].upper()} | "
                        f"fraction: {info['old_fraction']*100:.1f}% → "
                        f"{info['fraction']*100:.1f}% | "
                        f"eligible: {train_ds.eligible_count:,}/"
                        f"{len(train_ds.chunks):,}"
                    )

                # Per-tier evaluation
                tier_results = evaluate_by_tier(
                    model, train_ds, device, vocab_size
                )
                tier_str = "  ".join(
                    f"{k}={v:.3f}" for k, v in tier_results.items()
                )
                print(f"  [tiers] {tier_str}")

                # Save best
                if current_val < best_val:
                    best_val = current_val
                    save_checkpoint(
                        best_ckpt_path, model, optimizer,
                        {"lr": base_lr}, step, tokens_seen, best_val,
                        curriculum_state=(
                            scheduler.state_dict() if scheduler else None
                        ),
                    )

                # Log
                curr_metrics = {
                    "tier_easy": tier_results.get("easy", float('nan')),
                    "tier_medium": tier_results.get("medium", float('nan')),
                    "tier_hard": tier_results.get("hard", float('nan')),
                    "fraction": (
                        scheduler.get_current_fraction()
                        if scheduler else float('nan')
                    ),
                    "kv_div": kv_div,
                    "ts_forgetting": forgetting if ts_val else float('nan'),
                }
                logger.log(
                    step, tokens_seen, train_loss, val_losses,
                    base_lr, curriculum=curr_metrics,
                )

                # Plateau check
                if plateau.update(current_val):
                    exit_reason = "plateau"
                    break

            step += 1

        if exit_reason:
            break

    pbar.close()
    logger.log_exit(exit_reason, step, tokens_seen)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXPANSION TRAINING COMPLETE — Stage {args.stage}")
    print(f"{'='*70}")
    print(f"  Exit reason:  {exit_reason}")
    print(f"  Total steps:  {step:,}")
    print(f"  Tokens seen:  {tokens_seen/1e6:.1f}M")
    print(f"  Best val:     {best_val:.4f}")
    print(f"  Checkpoint:   {best_ckpt_path}")
    print(f"  Model:        {model.num_params()/1e6:.1f}M params")
    print(f"{'='*70}\n")

    # Run capability logger
    if os.path.exists(best_ckpt_path):
        # We need to free up GPU memory before running the evaluation
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        try:
            run_capability_logging(best_ckpt_path, args.tokenizer, stage_name=args.stage)
        except Exception as e:
            print(f"[logger] Capability logging failed: {e}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SLM with model expansion"
    )
    p.add_argument("--stage", type=str, required=True,
                   choices=["A", "B", "C", "2", "3", "4", "5", "6"],
                   help="Expansion stage to run")
    p.add_argument("--config", type=str,
                   default="configs/expansion_stages.yaml",
                   help="Path to expansion stages config")
    p.add_argument("--tokenizer", type=str, required=True,
                   help="Path to tokenizer JSON file")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory for saving checkpoints")
    p.add_argument("--log_dir", type=str, default="Logs",
                   help="Directory for training logs")
    p.add_argument("--cache_dir", type=str, default="cache",
                   help="Directory for dataset caches")
    return p.parse_args()


if __name__ == "__main__":
    train_expansion(parse_args())
