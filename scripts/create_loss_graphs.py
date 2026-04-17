#!/usr/bin/env python3
"""
Generate loss & validation loss graphs for all training logs.
Creates one plot per CSV with training loss and all validation losses on same axes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Setup
logs_dir = Path("/home/user20/NLP/slm/logs")
output_dir = Path("/home/user20/NLP/slm/logs/graphs")
output_dir.mkdir(exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

# Get all CSV files
csv_files = sorted(logs_dir.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in logs/")

# Create graphs
successful = 0
skipped = 0

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        # Skip files with only headers or too little data
        if len(df) < 2:
            print(f"⊘ {csv_file.name} (insufficient data, {len(df)} rows)")
            skipped += 1
            continue
        
        # Check for required columns
        required_cols = {'train_loss', 'val_s0', 'val_s1', 'val_s2'}
        if not required_cols.issubset(df.columns):
            print(f"⊘ {csv_file.name} (missing columns)")
            skipped += 1
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get step or index for x-axis
        x = df.get('step', range(len(df)))
        
        # Plot training loss (primary axis)
        ax.plot(x, df['train_loss'], 'o-', label='Training Loss', 
                linewidth=2.5, markersize=5, color='#FF6B6B', alpha=0.8)
        
        # Plot validation losses
        ax.plot(x, df['val_s0'], 's--', label='Val S0 (TinyStories)', 
                linewidth=2, markersize=4, color='#4ECDC4', alpha=0.7)
        ax.plot(x, df['val_s1'], '^--', label='Val S1 (SimpleWiki)', 
                linewidth=2, markersize=4, color='#45B7D1', alpha=0.7)
        ax.plot(x, df['val_s2'], 'D--', label='Val S2 (FineWeb)', 
                linewidth=2, markersize=4, color='#96CEB4', alpha=0.7)
        
        # Formatting
        stage = df['stage'].iloc[0] if 'stage' in df.columns else "?"
        tokens = df['tokens_seen'].iloc[-1] / 1e6 if 'tokens_seen' in df.columns else "?"
        
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Stage {stage} Training - {csv_file.stem}\n'
                     f'Final Train Loss: {df["train_loss"].iloc[-1]:.4f} | '
                     f'Tokens: {tokens:.1f}M' if isinstance(tokens, float) else 
                     f'Final Train Loss: {df["train_loss"].iloc[-1]:.4f}',
                     fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Save
        output_file = output_dir / f"{csv_file.stem}_graph.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {csv_file.name} → {output_file.name}")
        successful += 1
        
    except Exception as e:
        print(f"✗ {csv_file.name} (error: {e})")
        skipped += 1

print(f"\n✓ Generated {successful} graphs")
print(f"⊘ Skipped {skipped} files")
print(f"📁 Output directory: {output_dir}")
