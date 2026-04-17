"""
Quick evaluation of checkpoints using CSV data and model info.
Extracts perplexity, LM scores from existing training logs and checkpoint info.
"""

import os
import math
import json
from pathlib import Path


def compute_metrics_from_logs():
    """
    Extract evaluation metrics from CSV logs.
    Perplexity = exp(loss)
    """
    
    csv_files = {
        "stage0_best_1st_trial.pt": "/home/user20/NLP/slm/logs/stage0_20260405_050310.csv",
        "stage0_best_1stfull.pt": "/home/user20/NLP/slm/logs/stage0_20260405_060014.csv"
    }
    
    results = {}
    
    for model_name, csv_path in csv_files.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")
        
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Parse CSV
        headers = lines[0].strip().split(',')
        data_lines = lines[1:]
        
        # Extract last checkpoint line (with data, not NaN)
        last_valid_line = None
        for line in reversed(data_lines):
            parts = line.strip().split(',')
            if len(parts) > 4 and parts[3] and parts[3] != '':
                try:
                    float(parts[3])  # Try to convert train_loss
                    last_valid_line = parts
                    break
                except:
                    continue
        
        if not last_valid_line:
            print(f"WARNING: No valid data found in {csv_path}")
            continue
        
        # Map headers to values
        data = dict(zip(headers, last_valid_line))
        
        # Extract metrics
        step = int(data.get('step', 0))
        tokens = int(data.get('tokens_seen', 0))
        
        train_loss = float(data.get('train_loss', 0)) if data.get('train_loss') else None
        val_s0 = float(data.get('val_s0', 0)) if data.get('val_s0') else None
        val_s1 = float(data.get('val_s1', 0)) if data.get('val_s1') else None
        val_s2 = float(data.get('val_s2', 0)) if data.get('val_s2') else None
        lr = data.get('lr', '0')
        
        metrics = {
            "checkpoint": model_name,
            "training_info": {
                "step": step,
                "tokens_seen": tokens,
                "tokens_seen_formatted": f"{tokens/1e6:.1f}M",
                "learning_rate": lr
            },
            "loss_metrics": {},
            "perplexity_metrics": {},
            "language_modeling_scores": {},
        }
        
        # Loss Metrics
        if train_loss:
            metrics["loss_metrics"]["training_loss"] = train_loss
        if val_s0:
            metrics["loss_metrics"]["val_tinystories_loss"] = val_s0
        if val_s1:
            metrics["loss_metrics"]["val_simplewiki_loss"] = val_s1
        if val_s2:
            metrics["loss_metrics"]["val_fineweb_loss"] = val_s2
        
        # Perplexity Metrics (PPL = exp(loss))
        if train_loss:
            metrics["perplexity_metrics"]["training_perplexity"] = math.exp(train_loss)
        if val_s0:
            metrics["perplexity_metrics"]["val_tinystories_ppl"] = math.exp(val_s0)
        if val_s1:
            metrics["perplexity_metrics"]["val_simplewiki_ppl"] = math.exp(val_s1)
        if val_s2:
            metrics["perplexity_metrics"]["val_fineweb_ppl"] = math.exp(val_s2)
        
        # Language Modeling Score (based on loss trends)
        # A perfect LM would have PPL ~1, lower is better
        # Score = 1 - (PPL - baseline) / baseline (normalized 0-1, higher is better)
        if val_s0:
            ppl_s0 = math.exp(val_s0)
            # Baseline: random model would have PPL = vocab_size (50000)
            baseline_ppl = 50000
            lm_score = max(0, 1 - (ppl_s0 - 1) / (baseline_ppl - 1))
            metrics["language_modeling_scores"]["tinystories_lm_score"] = lm_score
            metrics["language_modeling_scores"]["tinystories_lm_score_percent"] = lm_score * 100
        
        if val_s1:
            ppl_s1 = math.exp(val_s1)
            baseline_ppl = 50000
            lm_score = max(0, 1 - (ppl_s1 - 1) / (baseline_ppl - 1))
            metrics["language_modeling_scores"]["simplewiki_lm_score"] = lm_score
            metrics["language_modeling_scores"]["simplewiki_lm_score_percent"] = lm_score * 100
        
        if val_s2:
            ppl_s2 = math.exp(val_s2)
            baseline_ppl = 50000
            lm_score = max(0, 1 - (ppl_s2 - 1) / (baseline_ppl - 1))
            metrics["language_modeling_scores"]["fineweb_lm_score"] = lm_score
            metrics["language_modeling_scores"]["fineweb_lm_score_percent"] = lm_score * 100
        
        results[model_name] = metrics
        
        # Print results
        print(f"\nTraining Info:")
        print(f"  Step: {step:,}")
        print(f"  Tokens: {tokens/1e6:.1f}M")
        
        print(f"\nLoss Metrics:")
        for key, val in metrics["loss_metrics"].items():
            print(f"  {key}: {val:.4f}")
        
        print(f"\nPerplexity (exp(loss)):")
        for key, val in metrics["perplexity_metrics"].items():
            print(f"  {key}: {val:.2f}")
        
        print(f"\nLanguage Modeling Scores (0-100%):")
        for key, val in metrics["language_modeling_scores"].items():
            if "percent" in key:
                print(f"  {key}: {val:.2f}%")
    
    return results


def main():
    """Main evaluation."""
    print("\n" + "="*70)
    print("CHECKPOINT EVALUATION: Perplexity, LM Score, Metrics")
    print("="*70)
    
    results = compute_metrics_from_logs()
    
    # Save results
    output_path = "/home/user20/NLP/slm/CHECKPOINT_METRICS.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: First Trial vs Full Training")
    print("="*70)
    
    if len(results) >= 2:
        models = list(results.keys())
        
        checkpoints_info = {
            "stage0_best_1st_trial.pt": {
                "name": "First Trial (Early Exit)",
                "tokens": "107.7M",
                "reason": "Loss spike detection"
            },
            "stage0_best_1stfull.pt": {
                "name": "Full Training",
                "tokens": "200M",
                "reason": "Token budget reached"
            }
        }
        
        for model in models:
            if model in checkpoints_info:
                info = checkpoints_info[model]
                print(f"\n{info['name']} ({info['tokens']})")
                print(f"  Termination: {info['reason']}")
                
                metrics = results[model]
                print(f"  Training Loss: {metrics['loss_metrics'].get('training_loss', 'N/A'):.4f}")
                
                if 'val_tinystories_ppl' in metrics['perplexity_metrics']:
                    print(f"  TinyStories PPL: {metrics['perplexity_metrics']['val_tinystories_ppl']:.2f}")
                    print(f"  TinyStories LM Score: {metrics['language_modeling_scores'].get('tinystories_lm_score_percent', 'N/A'):.2f}%")
                
                if 'val_simplewiki_ppl' in metrics['perplexity_metrics']:
                    print(f"  SimpleWiki PPL: {metrics['perplexity_metrics']['val_simplewiki_ppl']:.2f}")
                    print(f"  SimpleWiki LM Score: {metrics['language_modeling_scores'].get('simplewiki_lm_score_percent', 'N/A'):.2f}%")
        
        # Compute deltas
        print(f"\n" + "="*70)
        print("Improvement (Full vs First Trial):")
        print("="*70)
        
        m1 = results["stage0_best_1st_trial.pt"]
        m2 = results["stage0_best_1stfull.pt"]
        
        for dataset in ["tinystories", "simplewiki", "fineweb"]:
            ppl_key_1 = f"val_{dataset}_ppl"
            ppl_key_2 = f"val_{dataset}_ppl"
            
            if ppl_key_1 in m1.get("perplexity_metrics", {}) and ppl_key_2 in m2.get("perplexity_metrics", {}):
                ppl1 = m1["perplexity_metrics"][ppl_key_1]
                ppl2 = m2["perplexity_metrics"][ppl_key_2]
                delta = ((ppl1 - ppl2) / ppl1) * 100
                
                print(f"\n{dataset.upper()}:")
                print(f"  First Trial PPL: {ppl1:.2f}")
                print(f"  Full Training PPL: {ppl2:.2f}")
                print(f"  Improvement: {delta:.2f}%")


if __name__ == "__main__":
    main()
