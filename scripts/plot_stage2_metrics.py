import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_stage2_metrics():
    # Find the latest Stage 2 log
    log_files = glob.glob('Logs/stageexpansion_2_*.csv')
    if not log_files:
        print("No Stage 2 log files found.")
        return
    
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Plotting from {latest_log}")
    
    df = pd.read_csv(latest_log)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Training Loss
    ax1.plot(df['step'], df['train_loss'], label='Train Loss', alpha=0.6)
    ax1.set_title('Stage 2 Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot Validation Losses
    val_cols = [c for c in df.columns if c.startswith('val_')]
    for col in val_cols:
        # Filter out s1, s2, wp if they are too high to see others
        if df[col].mean() < 5.0:
            ax2.plot(df['step'], df[col], label=col, marker='o', markersize=4)
    
    ax2.set_title('Stage 2 Validation Losses (Low Loss Domains)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    output_path = 'docs/figures/stage2_learning_curves.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    plot_stage2_metrics()
