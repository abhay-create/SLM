import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def main():
    os.makedirs("docs/figures", exist_ok=True)
    
    # Define the log files
    log_files = {
        "Baseline": "Logs/stagefull_dataset_20260416_133055.csv",
        "Stage A": "Logs/stageexpansion_A_20260421_090025.csv",
        "Stage B": "Logs/stageexpansion_B_20260421_092312.csv",
        "Stage C": "Logs/stageexpansion_C_20260421_102040.csv"
    }
    
    all_dfs = []
    last_tokens = 0
    
    # Collect Baseline
    df_base = pd.read_csv(log_files["Baseline"])
    df_base = df_base[df_base['note'] != 'EXIT:plateau']
    df_base['global_tokens'] = df_base['tokens_seen']
    df_base['stage_name'] = 'Baseline'
    all_dfs.append(df_base)
    
    last_tokens = df_base['tokens_seen'].max()
    
    # Collect expansions
    for stage in ["Stage A", "Stage B", "Stage C"]:
        if os.path.exists(log_files[stage]):
            df = pd.read_csv(log_files[stage])
            df = df[df['note'] != 'EXIT:plateau']
            df['global_tokens'] = df['tokens_seen'] + last_tokens
            df['stage_name'] = stage
            all_dfs.append(df)
            last_tokens += df['tokens_seen'].max()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 1. Plot Training Loss and Validation (TinyStories & WP) over tokens
    plt.figure(figsize=(12, 6))
    
    colors = {"Baseline": "#1f77b4", "Stage A": "#ff7f0e", "Stage B": "#2ca02c", "Stage C": "#d62728"}
    
    for stage, df in combined_df.groupby('stage_name', sort=False):
        plt.plot(df['global_tokens'] / 1e6, df['train_loss'], 
                 color=colors[stage], alpha=0.3, label=f'{stage} (Train)')
                 
        plt.plot(df['global_tokens'] / 1e6, df['val_s0'], 
                 color=colors[stage], linestyle='--', linewidth=2, label=f'{stage} (TinyStories Val)')
                 
        if 'tier_easy' in df.columns and stage in ["Stage B", "Stage C"]:
            wp_val = (df['tier_easy'] + df['tier_med'] + df['tier_hard']) / 3
            plt.plot(df['global_tokens'] / 1e6, wp_val, 
                     color=colors[stage], linestyle=':', linewidth=2, label=f'{stage} (WP Avg)')
    
    for stage in ["Stage A", "Stage B", "Stage C"]:
        stage_df = combined_df[combined_df['stage_name'] == stage]
        if not stage_df.empty:
            plt.axvline(x=stage_df['global_tokens'].min() / 1e6, color='k', linestyle='-', alpha=0.2)
                
    plt.title('Expansion Training: Loss vs. Tokens Seen', fontsize=14)
    plt.xlabel('Cumulative Tokens (Millions)', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.ylim(1.0, 6.0)
    plt.grid(True, alpha=0.3)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    plt.savefig('docs/figures/expansion_loss_trajectory.png', dpi=300)
    print("Saved docs/figures/expansion_loss_trajectory.png")
    
    # 2. Plot Trade-off (Forgetting vs. Learning)
    stages = []
    ts_loss = []
    wp_loss = []
    
    for stage, df in combined_df.groupby('stage_name', sort=False):
        last_row = df.iloc[-1]
        stages.append(stage)
        ts_loss.append(last_row['val_s0'])
        
        if stage == 'Baseline':
            wp_loss.append(8.0)
        elif stage == 'Stage A':
            wp_loss.append(1.59) 
        else:
            wp_loss.append((last_row['tier_easy'] + last_row['tier_med'] + last_row['tier_hard']) / 3)
            
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Expansion Stage')
    ax1.set_ylabel('TinyStories Validation Loss', color=color1)
    ax1.plot(stages, ts_loss, color=color1, marker='o', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('WritingPrompts Validation Loss', color=color2)
    ax2.plot(stages, wp_loss, color=color2, marker='s', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Performance Trade-off: Capability vs. Forgetting', fontsize=14)
    fig.tight_layout()
    plt.savefig('docs/figures/capability_tradeoff.png', dpi=300)
    print("Saved docs/figures/capability_tradeoff.png")

if __name__ == "__main__":
    main()
