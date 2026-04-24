"""
capability_logger.py - Tracks SLM multi-stage curriculum progress.

Evaluates the model across multiple domains (TinyStories, ROCStories,
Children Stories, SimpleWiki, WritingPrompts) and generates stylized text
to document capability acquisition.

Output is appended to docs/curriculum_capabilities.md
"""

import os
import argparse
import datetime
import torch
import math
from tokenizers import Tokenizer
import sys

sys.path.insert(0, '.')
from src.model import SLM
from src.dataset import load_all_val_sets
from train_curriculum import evaluate
from evaluate_curriculum import generate_sample, load_model_from_checkpoint

PROMPTS = {
    "TinyStories (Ages 3-5)": [
        "Once upon a time, there was a little",
        "The dog ran to the park because",
    ],
    "ROCStories (Commonsense Logic)": [
        "John went to the store to buy milk. He",
        "Sarah was excited for her birthday party.",
    ],
    "WritingPrompts (Adult Fiction)": [
        "The year is 2157. Humanity has",
        "Death appeared before him and said",
    ]
}

def compute_ppl(loss):
    return math.exp(loss) if loss < 20 else float('inf')

def run_capability_logging(checkpoint_path, tokenizer_path, stage_name=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    print(f"\n[logger] Loading checkpoint: {checkpoint_path}")
    model, cfg, info = load_model_from_checkpoint(checkpoint_path, device)
    
    # 1. Cross-Domain Validation
    print("\n[logger] Evaluating cross-domain perplexity...")
    val_loaders = load_all_val_sets(tokenizer, cache_dir="cache")
    results = {}
    
    for key, loader in val_loaders.items():
        loss = evaluate(model, loader, device, vocab_size=vocab_size, max_batches=50)
        results[key] = {
            "loss": loss,
            "ppl": compute_ppl(loss)
        }
        print(f"  {key:>8}: loss={loss:.3f}, ppl={compute_ppl(loss):.1f}")

    # 2. Stylistic Probing
    print("\n[logger] Running stylistic generation probes...")
    generations = {}
    for domain, prompt_list in PROMPTS.items():
        generations[domain] = []
        for prompt in prompt_list:
            text = generate_sample(model, tokenizer, prompt, device, max_tokens=100, temperature=0.7)
            text = text.replace('\n', ' ').strip()
            if len(text) > 250: text = text[:250] + "..."
            generations[domain].append((prompt, text))

    # 3. Format and Append to Markdown
    os.makedirs("docs", exist_ok=True)
    report_file = "docs/curriculum_capabilities.md"
    
    # Check if file exists to add header
    needs_header = not os.path.exists(report_file)
    
    with open(report_file, "a") as f:
        if needs_header:
            f.write("# SLM Curriculum Capability Progression\n\n")
            f.write("Tracking model capabilities across diverse domains as it progresses through the multi-stage curriculum.\n\n")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"## Stage: {stage_name or 'Manual Evaluation'} ({timestamp})\n")
        f.write(f"- **Checkpoint**: `{os.path.basename(checkpoint_path)}`\n")
        f.write(f"- **Parameters**: {model.num_params()/1e6:.1f}M\n")
        f.write(f"- **Config**: {cfg.n_layers}L, d_ff={cfg.d_ff}, ctx={cfg.ctx_len}\n\n")
        
        f.write("### Cross-Domain Metrics\n\n")
        f.write("| Domain | Val Key | Loss | Perplexity |\n")
        f.write("|--------|---------|------|------------|\n")
        # Format known domains nicely
        domain_mapping = {
            "s0": "TinyStories",
            "s1": "SimpleWiki",
            "s2": "FineWeb-Edu",
            "roc": "ROCStories",
            "simple": "SimpleStories",
            "child": "Children-Stories",
            "wp": "WritingPrompts"
        }
        for key, metrics in results.items():
            domain_name = domain_mapping.get(key, key)
            f.write(f"| {domain_name} | `{key}` | {metrics['loss']:.3f} | {metrics['ppl']:.1f} |\n")
            
        f.write("\n### Stylistic Probes\n\n")
        for domain, items in generations.items():
            f.write(f"**{domain}**\n")
            for prompt, text in items:
                f.write(f"> **Prompt**: `{prompt}`\n")
                f.write(f"> {text}\n>\n")
            f.write("\n")
            
        f.write("---\n\n")

    print(f"\n[logger] Report appended to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizers/tokenizer_corpus.json")
    parser.add_argument("--stage", type=str, default="Unknown Stage")
    args = parser.parse_args()
    
    run_capability_logging(args.checkpoint, args.tokenizer, args.stage)
