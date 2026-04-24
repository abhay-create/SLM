"""
Comprehensive expansion analysis plots.
Generates all figures for the expansion report.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("docs/figures", exist_ok=True)

# ── Load all CSV logs ────────────────────────────────────────────────────────
log_files = {
    "Baseline (50M)": "Logs/stagefull_dataset_20260416_133055.csv",
    "Stage A (58M)":  "Logs/stageexpansion_A_20260421_090025.csv",
    "Stage B (71M)":  "Logs/stageexpansion_B_20260421_092312.csv",
    "Stage C (99M)":  "Logs/stageexpansion_C_20260421_102040.csv",
}

dfs = {}
for name, path in log_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.dropna(subset=['step'])
        df = df[~df['note'].astype(str).str.contains('EXIT', na=False)]
        dfs[name] = df

# ── FIGURE 1: TinyStories val loss across all stages ────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
colors = {"Baseline (50M)": "#2563eb", "Stage A (58M)": "#f59e0b", 
          "Stage B (71M)": "#10b981", "Stage C (99M)": "#ef4444"}

cumulative_tokens = 0
stage_boundaries = []
for name, df in dfs.items():
    tokens_m = (df['tokens_seen'] + cumulative_tokens) / 1e6
    ax.plot(tokens_m, df['val_s0'], color=colors[name], linewidth=2.5, label=name)
    stage_boundaries.append(cumulative_tokens / 1e6)
    cumulative_tokens += df['tokens_seen'].max()

# Mark stage transitions
for i, (boundary, stage_name) in enumerate(zip(stage_boundaries[1:], list(dfs.keys())[1:])):
    ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    ax.text(boundary + 2, ax.get_ylim()[1] * 0.95, stage_name.split('(')[0].strip(), 
            fontsize=8, color='gray', rotation=90, va='top')

# Reference line: baseline best
baseline_best = dfs["Baseline (50M)"]["val_s0"].min()
ax.axhline(y=baseline_best, color='blue', linestyle=':', alpha=0.4, label=f'Baseline best ({baseline_best:.3f})')

ax.set_xlabel('Cumulative Tokens (Millions)', fontsize=12)
ax.set_ylabel('TinyStories Validation Loss', fontsize=12)
ax.set_title('TinyStories Validation Loss Across All Expansion Stages', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig('docs/figures/fig1_tinystories_trajectory.png', dpi=150)
print("Saved fig1_tinystories_trajectory.png")
plt.close()

# ── FIGURE 2: Per-stage training loss ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
expansion_stages = ["Stage A (58M)", "Stage B (71M)", "Stage C (99M)"]

for i, name in enumerate(expansion_stages):
    ax = axes[i]
    df = dfs[name]
    ax.plot(df['tokens_seen'] / 1e6, df['train_loss'], color=colors[name], linewidth=2, label='Train')
    ax.plot(df['tokens_seen'] / 1e6, df['val_s0'], color=colors[name], linewidth=2, linestyle='--', label='TS Val')
    
    # Show tier losses if available
    if 'tier_easy' in df.columns:
        tier_avg = (df['tier_easy'] + df['tier_med'] + df['tier_hard']) / 3
        ax.plot(df['tokens_seen'] / 1e6, tier_avg, color=colors[name], linewidth=2, linestyle=':', label='Tier Avg')
    
    ax.set_xlabel('Tokens (M)', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

fig.suptitle('Per-Stage Training Dynamics', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('docs/figures/fig2_per_stage_dynamics.png', dpi=150, bbox_inches='tight')
print("Saved fig2_per_stage_dynamics.png")
plt.close()

# ── FIGURE 3: Parameter growth vs. performance ──────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 6))

stage_names = ['Baseline\n(50M)', 'Stage A\n(58M)', 'Stage B\n(71M)', 'Stage C\n(99M)']
params = [45.8, 58.4, 71.0, 99.4]
ts_val = [1.5373, 1.6278, 2.9679, 2.7893]  # Final val_s0 at exit
ts_best = [1.5373, 1.5835, 2.1265, 2.2000]  # Best val_s0 during training

x = np.arange(len(stage_names))

# Bar chart for params
bars = ax1.bar(x, params, 0.6, color=['#2563eb', '#f59e0b', '#10b981', '#ef4444'], alpha=0.3)
ax1.set_ylabel('Parameters (Millions)', fontsize=12, color='gray')
ax1.set_ylim(0, 120)
ax1.tick_params(axis='y', labelcolor='gray')

# Line for TS val loss
ax2 = ax1.twinx()
ax2.plot(x, ts_best, 'b-o', linewidth=2.5, markersize=10, label='Best TS Val Loss')
ax2.plot(x, ts_val, 'b--s', linewidth=1.5, markersize=7, alpha=0.5, label='Exit TS Val Loss')
ax2.set_ylabel('TinyStories Validation Loss', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Forgetting annotations
for i in range(1, len(ts_best)):
    pct = ((ts_best[i] - ts_best[0]) / ts_best[0]) * 100
    ax2.annotate(f'+{pct:.0f}%', xy=(i, ts_best[i]), xytext=(i+0.15, ts_best[i]+0.08),
                fontsize=9, color='red', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(stage_names, fontsize=11)
ax2.legend(loc='upper left', fontsize=10)
ax1.set_title('Model Growth vs. TinyStories Retention', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('docs/figures/fig3_growth_vs_retention.png', dpi=150)
print("Saved fig3_growth_vs_retention.png")
plt.close()

# ── FIGURE 4: WritingPrompts learning curve (Stages B & C) ──────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for name in ["Stage B (71M)", "Stage C (99M)"]:
    df = dfs[name]
    if 'tier_easy' in df.columns:
        tier_avg = (df['tier_easy'] + df['tier_med'] + df['tier_hard']) / 3
        ax.plot(df['tokens_seen'] / 1e6, tier_avg, color=colors[name], 
                linewidth=2.5, marker='o', markersize=6, label=f'{name} Tier Avg')
        ax.fill_between(df['tokens_seen'] / 1e6, df['tier_easy'], df['tier_hard'], 
                        color=colors[name], alpha=0.1)

ax.set_xlabel('Tokens Seen (Millions)', fontsize=12)
ax.set_ylabel('WritingPrompts Tier Loss', fontsize=12)
ax.set_title('WritingPrompts Learning Progress', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig('docs/figures/fig4_wp_learning.png', dpi=150)
print("Saved fig4_wp_learning.png")
plt.close()

# ── FIGURE 5: Forgetting bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

stage_labels = ['Stage A\n(58M)', 'Stage B\n(71M)', 'Stage C\n(99M)']
forgetting = [
    ((1.5835 - 1.5373) / 1.5373) * 100,  # Stage A best
    ((2.1265 - 1.5373) / 1.5373) * 100,  # Stage B best
    ((2.2000 - 1.5373) / 1.5373) * 100,  # Stage C best
]
forgetting_exit = [
    ((1.6278 - 1.5373) / 1.5373) * 100,
    ((2.9679 - 1.5373) / 1.5373) * 100,
    ((2.7893 - 1.5373) / 1.5373) * 100,
]

x = np.arange(len(stage_labels))
w = 0.35

bars1 = ax.bar(x - w/2, forgetting, w, label='Best checkpoint', color=['#f59e0b', '#10b981', '#ef4444'], alpha=0.7)
bars2 = ax.bar(x + w/2, forgetting_exit, w, label='At exit', color=['#f59e0b', '#10b981', '#ef4444'], alpha=0.3, 
               edgecolor=['#f59e0b', '#10b981', '#ef4444'], linewidth=2)

# 5% threshold line
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='5% target threshold')
ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% warning threshold')

for bar, val in zip(bars1, forgetting):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
            ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, forgetting_exit):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
            ha='center', fontsize=9, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(stage_labels, fontsize=11)
ax.set_ylabel('TinyStories Forgetting (%)', fontsize=12)
ax.set_title('Catastrophic Forgetting by Stage', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, max(forgetting_exit) * 1.2)
fig.tight_layout()
fig.savefig('docs/figures/fig5_forgetting.png', dpi=150)
print("Saved fig5_forgetting.png")
plt.close()

# ── FIGURE 6: Summary dashboard ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Params
ax = axes[0, 0]
stage_short = ['Base', 'A', 'B', 'C']
ax.bar(stage_short, params, color=['#2563eb', '#f59e0b', '#10b981', '#ef4444'], alpha=0.8)
ax.set_ylabel('Parameters (M)')
ax.set_title('A. Parameter Count', fontweight='bold')
for i, v in enumerate(params):
    ax.text(i, v + 1, f'{v:.1f}M', ha='center', fontsize=10)

# Panel B: Training tokens
ax = axes[0, 1]
tokens = [376.8, 24.6, 27.7, 28.7]
ax.bar(stage_short, tokens, color=['#2563eb', '#f59e0b', '#10b981', '#ef4444'], alpha=0.8)
ax.set_ylabel('Tokens (M)')
ax.set_title('B. Training Tokens per Stage', fontweight='bold')
for i, v in enumerate(tokens):
    ax.text(i, v + 3, f'{v:.1f}M', ha='center', fontsize=10)

# Panel C: TS val loss
ax = axes[1, 0]
ax.plot(stage_short, ts_best, 'b-o', linewidth=2.5, markersize=10)
ax.axhline(y=ts_best[0], color='blue', linestyle=':', alpha=0.3)
ax.set_ylabel('TinyStories Val Loss')
ax.set_title('C. TinyStories Performance (lower=better)', fontweight='bold')
ax.grid(True, alpha=0.2)

# Panel D: WP tier loss
ax = axes[1, 1]
wp_final = [None, None, 4.64, 4.69]  # Tier averages at exit
wp_steps = ['Base', 'A', 'B', 'C']
ax.bar(['B', 'C'], [4.64, 4.69], color=['#10b981', '#ef4444'], alpha=0.8)
ax.set_ylabel('WritingPrompts Tier Avg Loss')
ax.set_title('D. WritingPrompts Performance', fontweight='bold')

fig.suptitle('SLM Expansion Summary Dashboard (50M → 99M)', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('docs/figures/fig6_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved fig6_dashboard.png")
plt.close()

print("\nAll figures saved to docs/figures/")
