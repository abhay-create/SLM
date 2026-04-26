#!/usr/bin/env python3
"""
plot_training_metrics.py — Generate publication-quality training graphs.

Reads all stageexpansion_*.csv logs and produces a set of graphs covering:
  1. Training & validation loss across stages
  2. Per-domain validation loss evolution
  3. Per-domain perplexity evolution
  4. TinyStories forgetting signal
  5. Replay fraction over training
  6. Gradient norms (global + deep layers)
  7. Throughput (tokens/s) and GPU memory
  8. Per-tier difficulty breakdown
  9. Stage comparison bar charts (final metrics)

Usage:
  python scripts/plot_training_metrics.py                   # defaults
  python scripts/plot_training_metrics.py --log_dir Logs --output_dir Logs/plots
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

# ── Matplotlib setup (Agg backend for headless) ──────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Style ────────────────────────────────────────────────────────────────────

# Colour palette — one colour per stage, warm → cool gradient
STAGE_COLORS = {
    "expansion_2": "#E8575A",
    "expansion_3": "#F59E42",
    "expansion_4": "#43B581",
    "expansion_5": "#5291E0",
    "expansion_6": "#8B5CF6",
}
STAGE_LABELS = {
    "expansion_2": "Stage 2 – ROC+Simple",
    "expansion_3": "Stage 3 – Children's",
    "expansion_4": "Stage 4 – SimpleWiki",
    "expansion_5": "Stage 5 – WP Easy",
    "expansion_6": "Stage 6 – WP Full",
}

DOMAIN_COLORS = {
    "s0": "#E8575A",
    "s1": "#F59E42",
    "s2": "#F5C542",
    "roc": "#43B581",
    "simple": "#5291E0",
    "child": "#8B5CF6",
    "wp": "#EC4899",
}
DOMAIN_LABELS = {
    "s0": "TinyStories",
    "s1": "SimpleWiki",
    "s2": "FineWeb-Edu",
    "roc": "ROCStories",
    "simple": "SimpleStories",
    "child": "Children-Stories",
    "wp": "WritingPrompts",
}


def setup_style():
    """Apply a clean dark style."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#394867",
        "axes.labelcolor": "#e0e0e0",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "text.color": "#e0e0e0",
        "xtick.color": "#a0a0a0",
        "ytick.color": "#a0a0a0",
        "grid.color": "#2a3a5c",
        "grid.alpha": 0.5,
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#394867",
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1a1a2e",
        "font.size": 10,
    })


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_stage_csv(csv_path: str) -> dict:
    """Load a stage CSV into a dict of numpy arrays, skipping EXIT rows."""
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "EXIT:" in line:
                continue
            rows.append(line.split(","))

    data = {}
    for col_idx, col_name in enumerate(header):
        vals = []
        for row in rows:
            raw = row[col_idx] if col_idx < len(row) else ""
            try:
                vals.append(float(raw))
            except (ValueError, IndexError):
                vals.append(np.nan)
        data[col_name] = np.array(vals)
    return data


def load_all_stages(log_dir: str) -> dict:
    """Find and load all stage CSVs. Returns {stage_name: data_dict}."""
    pattern = os.path.join(log_dir, "stageexpansion_*.csv")
    paths = sorted(glob.glob(pattern))
    stages = {}
    for p in paths:
        data = load_stage_csv(p)
        if "stage" in data and len(data["stage"]) > 0:
            # Stage name from the data itself
            stage_name = None
            # Read from the meta.json sidecar
            meta_path = p.replace(".csv", ".meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                stage_name = meta.get("stage", None)
            if stage_name is None:
                # Fallback: extract from filename
                import re
                m = re.search(r"stageexpansion_(\d+)", os.path.basename(p))
                stage_name = f"expansion_{m.group(1)}" if m else os.path.basename(p)
            stages[stage_name] = data
    return stages


def load_meta(log_dir: str) -> dict:
    """Load all meta.json sidecars."""
    metas = {}
    for p in sorted(glob.glob(os.path.join(log_dir, "stageexpansion_*.meta.json"))):
        with open(p) as f:
            meta = json.load(f)
        metas[meta.get("stage", os.path.basename(p))] = meta
    return metas


# ─── Plotting Functions ──────────────────────────────────────────────────────

def plot_train_val_loss(stages: dict, output_dir: str):
    """Plot 1: Training loss + target val loss across all stages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss Across Stages", fontsize=15, fontweight="bold")

    for stage_name, data in stages.items():
        color = STAGE_COLORS.get(stage_name, "#888")
        label = STAGE_LABELS.get(stage_name, stage_name)
        steps = data.get("step", np.array([]))
        # Training loss
        train = data.get("train_loss", np.array([]))
        mask = ~np.isnan(train)
        if mask.any():
            ax1.plot(steps[mask], train[mask], color=color, label=label, linewidth=1.5, alpha=0.9)
        # Target val loss
        current_val = data.get("current_val", np.array([]))
        mask = ~np.isnan(current_val)
        if mask.any():
            ax2.plot(steps[mask], current_val[mask], color=color, label=label, linewidth=1.5, alpha=0.9, marker="o", markersize=3)

    for ax, title in [(ax1, "Training Loss"), (ax2, "Target Domain Val Loss")]:
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_train_val_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_domain_val_loss(stages: dict, output_dir: str):
    """Plot 2: All 7 domain val losses, one subplot per stage."""
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Per-Domain Validation Loss by Stage", fontsize=15, fontweight="bold")

    domains = ["s0", "s1", "s2", "roc", "simple", "child", "wp"]

    for ax, (stage_name, data) in zip(axes, stages.items()):
        steps = data.get("step", np.array([]))
        for dom in domains:
            col = f"val_{dom}"
            vals = data.get(col, np.array([]))
            mask = ~np.isnan(vals)
            if mask.any():
                ax.plot(steps[mask], vals[mask],
                        color=DOMAIN_COLORS.get(dom, "#888"),
                        label=DOMAIN_LABELS.get(dom, dom),
                        linewidth=1.3, alpha=0.85)
        ax.set_title(STAGE_LABELS.get(stage_name, stage_name), fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Val Loss")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_domain_val_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_domain_perplexity(stages: dict, output_dir: str):
    """Plot 3: Per-domain perplexity (log scale), final snapshot bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Domain Perplexity", fontsize=15, fontweight="bold")

    domains = ["s0", "roc", "simple", "child", "wp"]

    # Left: lines over training (all stages concatenated with offset)
    cum_offset = 0
    tick_positions, tick_labels = [], []
    for stage_name, data in stages.items():
        steps = data.get("step", np.array([]))
        if len(steps) == 0:
            continue
        shifted = steps + cum_offset
        for dom in domains:
            col = f"ppl_{dom}"
            vals = data.get(col, np.array([]))
            mask = ~np.isnan(vals) & (vals < 10000)
            if mask.any():
                ax1.plot(shifted[mask], vals[mask],
                         color=DOMAIN_COLORS.get(dom, "#888"),
                         linewidth=1.2, alpha=0.8)
        mid = cum_offset + (steps[-1] - steps[0]) / 2
        tick_positions.append(mid)
        tick_labels.append(stage_name.replace("expansion_", "S"))
        # Add stage separator
        ax1.axvline(cum_offset, color="#555", linestyle="--", alpha=0.4)
        cum_offset += steps[-1] + 1000

    ax1.set_yscale("log")
    ax1.set_xlabel("Step (concatenated)")
    ax1.set_ylabel("Perplexity (log)")
    ax1.set_title("Perplexity Over Training")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.grid(True, alpha=0.3)
    # Custom legend
    handles = [Line2D([0], [0], color=DOMAIN_COLORS[d], lw=2, label=DOMAIN_LABELS[d]) for d in domains]
    ax1.legend(handles=handles, fontsize=8, loc="upper right")

    # Right: bar chart of final perplexity per stage per domain
    x = np.arange(len(domains))
    width = 0.15
    for i, (stage_name, data) in enumerate(stages.items()):
        finals = []
        for dom in domains:
            col = f"ppl_{dom}"
            vals = data.get(col, np.array([]))
            valid = vals[~np.isnan(vals)]
            finals.append(valid[-1] if len(valid) > 0 else 0)
        color = STAGE_COLORS.get(stage_name, "#888")
        ax2.bar(x + i * width, finals, width, label=stage_name.replace("expansion_", "S"),
                color=color, alpha=0.85)

    ax2.set_xlabel("Domain")
    ax2.set_ylabel("Final Perplexity")
    ax2.set_title("Final Perplexity by Stage")
    ax2.set_xticks(x + width * (len(stages) - 1) / 2)
    ax2.set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=25, ha="right", fontsize=8)
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "03_domain_perplexity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_forgetting_and_replay(stages: dict, output_dir: str):
    """Plot 4: TinyStories forgetting EMA + replay fraction."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle("Forgetting & Replay Dynamics", fontsize=15, fontweight="bold")

    for stage_name, data in stages.items():
        color = STAGE_COLORS.get(stage_name, "#888")
        label = STAGE_LABELS.get(stage_name, stage_name)
        steps = data.get("step", np.array([]))

        # Forgetting
        fgt = data.get("ts_forgetting", np.array([]))
        ema = data.get("ts_forgetting_ema", np.array([]))
        mask_f = ~np.isnan(fgt)
        mask_e = ~np.isnan(ema)
        if mask_f.any():
            ax1.plot(steps[mask_f], fgt[mask_f] * 100, color=color, alpha=0.4, linewidth=1)
        if mask_e.any():
            ax1.plot(steps[mask_e], ema[mask_e] * 100, color=color, label=label, linewidth=2)

        # Replay
        replay = data.get("replay_frac", np.array([]))
        mask_r = ~np.isnan(replay)
        if mask_r.any():
            ax2.plot(steps[mask_r], replay[mask_r] * 100, color=color, label=label, linewidth=1.5)

    ax1.set_ylabel("Forgetting (%)")
    ax1.set_title("TinyStories Forgetting (raw + EMA)")
    ax1.axhline(5, color="#E8575A", linestyle="--", alpha=0.5, label="5% threshold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Replay Fraction (%)")
    ax2.set_title("Replay Fraction")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "04_forgetting_replay.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gradient_norms(stages: dict, output_dir: str):
    """Plot 5: Global and deep-layer gradient norms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gradient Norms", fontsize=15, fontweight="bold")

    for stage_name, data in stages.items():
        color = STAGE_COLORS.get(stage_name, "#888")
        label = STAGE_LABELS.get(stage_name, stage_name)
        steps = data.get("step", np.array([]))

        grad = data.get("grad_norm", np.array([]))
        mask = ~np.isnan(grad)
        if mask.any():
            ax1.plot(steps[mask], grad[mask], color=color, label=label, linewidth=1.3, alpha=0.85)

        deep = data.get("deep_grad_norm", np.array([]))
        mask = ~np.isnan(deep)
        if mask.any():
            ax2.plot(steps[mask], deep[mask], color=color, label=label, linewidth=1.3, alpha=0.85)

    ax1.set_title("Global Grad Norm")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Grad Norm")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Deep Layer Grad Norm")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Grad Norm")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "05_gradient_norms.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_throughput_memory(stages: dict, output_dir: str):
    """Plot 6: Tokens/s throughput + GPU memory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Throughput & GPU Memory", fontsize=15, fontweight="bold")

    for stage_name, data in stages.items():
        color = STAGE_COLORS.get(stage_name, "#888")
        label = STAGE_LABELS.get(stage_name, stage_name)
        steps = data.get("step", np.array([]))

        tps = data.get("tokens_per_sec", np.array([]))
        mask = ~np.isnan(tps) & (tps > 0)
        if mask.any():
            ax1.plot(steps[mask], tps[mask] / 1000, color=color, label=label, linewidth=1.5, alpha=0.85)

        mem = data.get("gpu_mem_alloc_mb", np.array([]))
        mask = ~np.isnan(mem)
        if mask.any():
            ax2.plot(steps[mask], mem[mask] / 1024, color=color, label=label, linewidth=1.5, alpha=0.85)

    ax1.set_title("Throughput")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("k tokens/s")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("GPU Memory Allocated")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("GB")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "06_throughput_memory.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_tier_breakdown(stages: dict, output_dir: str):
    """Plot 7: Easy/medium/hard tier loss per stage."""
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 5), sharey=False)
    if len(stages) == 1:
        axes = [axes]
    fig.suptitle("Per-Tier Difficulty Breakdown", fontsize=15, fontweight="bold")

    tier_colors = {"easy": "#43B581", "med": "#F59E42", "hard": "#E8575A"}

    for ax, (stage_name, data) in zip(axes, stages.items()):
        steps = data.get("step", np.array([]))
        for tier_key, tier_label in [("tier_easy", "Easy"), ("tier_med", "Medium"), ("tier_hard", "Hard")]:
            vals = data.get(tier_key, np.array([]))
            mask = ~np.isnan(vals)
            short_key = tier_key.replace("tier_", "")
            if mask.any():
                ax.plot(steps[mask], vals[mask],
                        color=tier_colors.get(short_key, "#888"),
                        label=tier_label, linewidth=1.5, alpha=0.85)
        ax.set_title(STAGE_LABELS.get(stage_name, stage_name), fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Tier Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "07_tier_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_stage_summary(stages: dict, metas: dict, output_dir: str):
    """Plot 8: Summary bar charts — final val loss, params, training time."""
    stage_names = list(stages.keys())
    n = len(stage_names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Stage Summary", fontsize=15, fontweight="bold")
    x = np.arange(n)
    colors = [STAGE_COLORS.get(s, "#888") for s in stage_names]
    short_labels = [s.replace("expansion_", "Stage ") for s in stage_names]

    # Final target val loss
    final_vals = []
    for s in stage_names:
        cv = stages[s].get("current_val", np.array([]))
        valid = cv[~np.isnan(cv)]
        final_vals.append(valid[-1] if len(valid) > 0 else 0)
    axes[0].bar(x, final_vals, color=colors, alpha=0.85)
    axes[0].set_title("Final Target Val Loss")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, rotation=20, ha="right", fontsize=9)
    for i, v in enumerate(final_vals):
        axes[0].text(i, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#e0e0e0")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Training time (from meta exit.elapsed_s)
    times = []
    for s in stage_names:
        meta = metas.get(s, {})
        elapsed = meta.get("exit", {}).get("elapsed_s", 0)
        times.append(elapsed / 60)  # minutes
    axes[1].bar(x, times, color=colors, alpha=0.85)
    axes[1].set_title("Training Time")
    axes[1].set_ylabel("Minutes")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=20, ha="right", fontsize=9)
    for i, v in enumerate(times):
        axes[1].text(i, v + 0.5, f"{v:.0f}m", ha="center", va="bottom", fontsize=8, color="#e0e0e0")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Tokens seen
    tokens = []
    for s in stage_names:
        meta = metas.get(s, {})
        t = meta.get("exit", {}).get("tokens_seen", 0)
        tokens.append(t / 1e6)
    axes[2].bar(x, tokens, color=colors, alpha=0.85)
    axes[2].set_title("Tokens Consumed")
    axes[2].set_ylabel("Millions")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, rotation=20, ha="right", fontsize=9)
    for i, v in enumerate(tokens):
        axes[2].text(i, v + 1, f"{v:.0f}M", ha="center", va="bottom", fontsize=8, color="#e0e0e0")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "08_stage_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_tinystories_retention(stages: dict, output_dir: str):
    """Plot 9: TinyStories (s0) loss across all stages — shows retention."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("TinyStories Retention Across Stages", fontsize=15, fontweight="bold")

    cum_offset = 0
    tick_positions, tick_labels = [], []

    for stage_name, data in stages.items():
        color = STAGE_COLORS.get(stage_name, "#888")
        steps = data.get("step", np.array([]))
        if len(steps) == 0:
            continue
        shifted = steps + cum_offset

        s0 = data.get("val_s0", np.array([]))
        mask = ~np.isnan(s0)
        if mask.any():
            ax.plot(shifted[mask], s0[mask], color=color,
                    label=STAGE_LABELS.get(stage_name, stage_name),
                    linewidth=2, alpha=0.9, marker="o", markersize=3)

        mid = cum_offset + (steps[-1] - steps[0]) / 2
        tick_positions.append(mid)
        tick_labels.append(stage_name.replace("expansion_", "S"))
        ax.axvline(cum_offset, color="#555", linestyle="--", alpha=0.3)
        cum_offset += steps[-1] + 1000

    ax.set_xlabel("Step (concatenated across stages)")
    ax.set_ylabel("TinyStories Val Loss")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "09_tinystories_retention.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot SLM training metrics")
    parser.add_argument("--log_dir", default="Logs", help="Directory containing stage CSV logs")
    parser.add_argument("--output_dir", default="Logs/plots", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    print(f"\n{'═'*60}")
    print(f"  SLM Training Metrics — Plot Generator")
    print(f"{'═'*60}")
    print(f"  Log dir:    {args.log_dir}")
    print(f"  Output dir: {args.output_dir}")

    stages = load_all_stages(args.log_dir)
    metas = load_meta(args.log_dir)

    if not stages:
        print(f"\n  ERROR: No stageexpansion_*.csv files found in {args.log_dir}/")
        return

    print(f"  Stages:     {len(stages)} loaded")
    for s in stages:
        n = len(stages[s].get("step", []))
        print(f"    • {s}: {n} eval points")
    print(f"{'═'*60}\n")

    print("Generating plots...")
    plot_train_val_loss(stages, args.output_dir)
    plot_domain_val_loss(stages, args.output_dir)
    plot_domain_perplexity(stages, args.output_dir)
    plot_forgetting_and_replay(stages, args.output_dir)
    plot_gradient_norms(stages, args.output_dir)
    plot_throughput_memory(stages, args.output_dir)
    plot_tier_breakdown(stages, args.output_dir)
    plot_stage_summary(stages, metas, args.output_dir)
    plot_tinystories_retention(stages, args.output_dir)

    print(f"\n{'═'*60}")
    print(f"  All plots saved to {args.output_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
