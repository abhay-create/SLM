#!/usr/bin/env python3
"""
generate_responses.py — Multi-prompt, multi-checkpoint generation script.

Load all *best.pt checkpoints from the checkpoints directory, generate text
for every user-specified prompt with configurable sampling settings, and store
the results as a structured JSON file and a readable Markdown report.

Usage examples:

  # Interactive — prompts entered at runtime:
  python scripts/generate_responses.py

  # Prompts from CLI:
  python scripts/generate_responses.py \
      --prompts "Once upon a time" "The scientist discovered" \
      --temperature 0.7 --top_k 40 --max_tokens 150

  # Prompts from a text file (one prompt per line):
  python scripts/generate_responses.py \
      --prompt_file prompts.txt \
      --temperature 0.5 0.8 1.0 \
      --top_k 30 50 \
      --max_tokens 200

  # Specify output directory and checkpoint directory:
  python scripts/generate_responses.py \
      --prompts "Hello world" \
      --checkpoint_dir checkpoints \
      --output_dir Logs/generations
"""

import os
import sys
import json
import glob
import time
import argparse
import datetime
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import SLM, SLMConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def discover_checkpoints(checkpoint_dir: str) -> list[dict]:
    """Find all *best*.pt checkpoints and extract stage metadata."""
    pattern = os.path.join(checkpoint_dir, "*best*.pt")
    paths = sorted(glob.glob(pattern))

    if not paths:
        # Fallback: try all .pt files that look like model checkpoints
        pattern = os.path.join(checkpoint_dir, "*.pt")
        paths = sorted(glob.glob(pattern))
        # Exclude SI omega files
        paths = [p for p in paths if "si_omega" not in os.path.basename(p)]

    checkpoints = []
    for path in paths:
        basename = os.path.basename(path)

        # Try to extract stage info from filename
        stage = "unknown"
        # Match patterns like stage_2_best.pt, stage_3_best.pt
        m = re.search(r"stage[_]?(\d+)", basename, re.IGNORECASE)
        if m:
            stage = f"stage_{m.group(1)}"
        # Match TinyStories patterns
        elif "TinyStories" in basename:
            if "Curriculum" in basename:
                stage = "stage_0_curriculum"
            elif "WithoutCurriculum" in basename:
                stage = "stage_0_baseline"
            else:
                stage = "stage_0"

        checkpoints.append({
            "path": path,
            "basename": basename,
            "stage": stage,
        })

    return checkpoints


def load_checkpoint_model(ckpt_path: str, device: str):
    """Load model from checkpoint, return (model, cfg, metadata)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "step": ckpt.get("step", "?"),
        "tokens_seen": ckpt.get("tokens_seen", "?"),
        "best_val_loss": ckpt.get("best_val_loss", "?"),
        "n_layers": cfg.n_layers,
        "d_ff": cfg.d_ff,
        "d_model": cfg.d_model,
        "ctx_len": cfg.ctx_len,
        "params_M": model.num_params() / 1e6,
    }
    return model, cfg, meta


@torch.no_grad()
def generate_text(
    model: SLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
    use_cache: bool = True,
) -> dict:
    """Generate text and return generation metadata."""
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.time()
    output = model.generate(
        idx,
        max_new=max_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        use_cache=use_cache,
    )
    elapsed = time.time() - t0

    generated_ids = output[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    # Extract only the generated portion (after the prompt tokens)
    generated_only = tokenizer.decode(generated_ids[len(ids):])

    return {
        "prompt": prompt,
        "generated_text": generated_only.strip(),
        "full_text": full_text.strip(),
        "num_prompt_tokens": len(ids),
        "num_generated_tokens": len(generated_ids) - len(ids),
        "generation_time_s": round(elapsed, 3),
        "tokens_per_second": round((len(generated_ids) - len(ids)) / max(elapsed, 1e-6), 1),
    }


def collect_prompts_interactive() -> list[str]:
    """Prompt user to enter multiple prompts interactively."""
    prompts = []
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Enter prompts (empty line to finish)       ║")
    print("╚══════════════════════════════════════════════╝\n")
    while True:
        try:
            p = input(f"  Prompt {len(prompts) + 1}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not p:
            break
        prompts.append(p)
    return prompts


def format_settings_label(temp: float, top_k: int, max_tokens: int) -> str:
    """Human-readable label for a settings combination."""
    return f"temp={temp}_topk={top_k}_len={max_tokens}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from all best.pt checkpoints for given prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_responses.py --prompts "Once upon a time" "The year is 2157"
  python scripts/generate_responses.py --prompt_file prompts.txt --temperature 0.5 0.8
  python scripts/generate_responses.py  # interactive prompt entry
        """,
    )

    # Prompt sources
    prompt_group = parser.add_argument_group("Prompts")
    prompt_group.add_argument(
        "--prompts", nargs="+", type=str, default=None,
        help="One or more prompts as CLI arguments",
    )
    prompt_group.add_argument(
        "--prompt_file", type=str, default=None,
        help="Path to a text file with one prompt per line",
    )

    # Generation settings (multiple values = grid search)
    settings_group = parser.add_argument_group("Generation Settings")
    settings_group.add_argument(
        "--temperature", nargs="+", type=float, default=[0.7],
        help="Temperature(s) for sampling (default: 0.7). Multiple values run a grid.",
    )
    settings_group.add_argument(
        "--top_k", nargs="+", type=int, default=[40],
        help="Top-k value(s) for sampling (default: 40, 0=disabled). Multiple values run a grid.",
    )
    settings_group.add_argument(
        "--max_tokens", nargs="+", type=int, default=[150],
        help="Max new tokens to generate (default: 150). Multiple values run a grid.",
    )
    settings_group.add_argument(
        "--no_cache", action="store_true",
        help="Disable KV cache during generation (slower but no length limits)",
    )

    # Paths
    paths_group = parser.add_argument_group("Paths")
    paths_group.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory containing checkpoint .pt files (default: checkpoints)",
    )
    paths_group.add_argument(
        "--tokenizer", type=str, default="tokenizers/tokenizer_corpus.json",
        help="Path to tokenizer JSON file",
    )
    paths_group.add_argument(
        "--output_dir", type=str, default="Logs/generations",
        help="Directory to write output files (default: Logs/generations)",
    )

    # Output control
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--tag", type=str, default=None,
        help="Optional tag to include in the output filename",
    )
    output_group.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-generation console output",
    )

    args = parser.parse_args()

    # ── Collect prompts ──────────────────────────────────────────────────────
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(line)
    if not prompts:
        prompts = collect_prompts_interactive()

    if not prompts:
        print("No prompts provided. Exiting.")
        return

    # ── Build settings grid ──────────────────────────────────────────────────
    settings_grid = []
    for temp in args.temperature:
        for top_k in args.top_k:
            for max_tok in args.max_tokens:
                settings_grid.append({
                    "temperature": temp,
                    "top_k": top_k,
                    "max_tokens": max_tok,
                })

    # ── Discover checkpoints ─────────────────────────────────────────────────
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"ERROR: No checkpoint files found in {args.checkpoint_dir}/")
        return

    # ── Setup ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    json_path = os.path.join(args.output_dir, f"generations_{timestamp}{tag}.json")
    md_path = os.path.join(args.output_dir, f"generations_{timestamp}{tag}.md")

    # ── Print run summary ────────────────────────────────────────────────────
    total_gens = len(prompts) * len(settings_grid) * len(checkpoints)
    print(f"\n{'═'*70}")
    print(f"  SLM Multi-Checkpoint Response Generation")
    print(f"{'═'*70}")
    print(f"  Device:       {device}")
    print(f"  Tokenizer:    {args.tokenizer}")
    print(f"  Checkpoints:  {len(checkpoints)} found")
    for ck in checkpoints:
        print(f"    • {ck['basename']} ({ck['stage']})")
    print(f"  Prompts:      {len(prompts)}")
    for i, p in enumerate(prompts):
        label = p if len(p) <= 60 else p[:57] + "..."
        print(f"    {i+1}. \"{label}\"")
    print(f"  Settings:     {len(settings_grid)} combination(s)")
    for s in settings_grid:
        print(f"    • temp={s['temperature']}, top_k={s['top_k']}, max_tokens={s['max_tokens']}")
    print(f"  Total gens:   {total_gens}")
    print(f"  Output JSON:  {json_path}")
    print(f"  Output MD:    {md_path}")
    print(f"{'═'*70}\n")

    # ── Generate ─────────────────────────────────────────────────────────────
    all_results = {
        "meta": {
            "timestamp": timestamp,
            "device": device,
            "tokenizer": args.tokenizer,
            "prompts": prompts,
            "settings_grid": settings_grid,
            "num_checkpoints": len(checkpoints),
            "total_generations": total_gens,
        },
        "results": [],
    }

    gen_count = 0

    for ck_info in checkpoints:
        print(f"\n{'─'*70}")
        print(f"  Loading: {ck_info['basename']} ({ck_info['stage']})")
        print(f"{'─'*70}")

        try:
            model, cfg, meta = load_checkpoint_model(ck_info["path"], device)
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}")
            continue

        print(f"  Model: {meta['params_M']:.1f}M params, "
              f"{meta['n_layers']}L, d_ff={meta['d_ff']}, ctx={meta['ctx_len']}")

        for settings in settings_grid:
            for prompt in prompts:
                gen_count += 1
                progress = f"[{gen_count}/{total_gens}]"

                try:
                    result = generate_text(
                        model, tokenizer, prompt, device,
                        max_tokens=settings["max_tokens"],
                        temperature=settings["temperature"],
                        top_k=settings["top_k"],
                        use_cache=not args.no_cache,
                    )
                except Exception as e:
                    result = {
                        "prompt": prompt,
                        "generated_text": f"ERROR: {e}",
                        "full_text": "",
                        "num_prompt_tokens": 0,
                        "num_generated_tokens": 0,
                        "generation_time_s": 0,
                        "tokens_per_second": 0,
                    }

                entry = {
                    "checkpoint": ck_info["basename"],
                    "stage": ck_info["stage"],
                    "model_params_M": meta["params_M"],
                    "model_config": {
                        "n_layers": meta["n_layers"],
                        "d_ff": meta["d_ff"],
                        "d_model": meta["d_model"],
                        "ctx_len": meta["ctx_len"],
                    },
                    "checkpoint_meta": {
                        "step": meta["step"],
                        "tokens_seen": meta["tokens_seen"],
                        "best_val_loss": meta["best_val_loss"],
                    },
                    "settings": settings,
                    **result,
                }
                all_results["results"].append(entry)

                if not args.quiet:
                    gen_preview = result["generated_text"]
                    if len(gen_preview) > 120:
                        gen_preview = gen_preview[:117] + "..."
                    print(
                        f"  {progress} "
                        f"temp={settings['temperature']}, "
                        f"top_k={settings['top_k']} | "
                        f"\"{prompt[:40]}...\" → "
                        f"\"{gen_preview}\" "
                        f"({result['generation_time_s']}s)"
                    )

        # Free GPU memory before loading next checkpoint
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save JSON ────────────────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[output] JSON saved → {json_path}")

    # ── Generate Markdown report ─────────────────────────────────────────────
    write_markdown_report(md_path, all_results, prompts, settings_grid, checkpoints)
    print(f"[output] Markdown report saved → {md_path}")

    print(f"\n{'═'*70}")
    print(f"  Generation complete: {gen_count} responses saved.")
    print(f"{'═'*70}\n")


def write_markdown_report(
    md_path: str,
    all_results: dict,
    prompts: list[str],
    settings_grid: list[dict],
    checkpoints: list[dict],
):
    """Write a human-readable Markdown report grouped by prompt."""
    with open(md_path, "w") as f:
        f.write("# SLM Multi-Checkpoint Generation Report\n\n")
        f.write(f"**Generated**: {all_results['meta']['timestamp']}\n\n")
        f.write(f"**Device**: {all_results['meta']['device']}\n\n")

        # Settings summary
        f.write("## Generation Settings\n\n")
        f.write("| Temperature | Top-K | Max Tokens |\n")
        f.write("|:-----------:|:-----:|:----------:|\n")
        for s in settings_grid:
            f.write(f"| {s['temperature']} | {s['top_k']} | {s['max_tokens']} |\n")

        # Checkpoints summary
        f.write("\n## Checkpoints\n\n")
        f.write("| # | Checkpoint | Stage | Params | Layers | d_ff | ctx_len | Val Loss |\n")
        f.write("|:-:|:-----------|:------|-------:|-------:|-----:|--------:|---------:|\n")
        for i, ck in enumerate(checkpoints):
            # Find a matching result to get meta
            matched = next(
                (r for r in all_results["results"] if r["checkpoint"] == ck["basename"]),
                None,
            )
            if matched:
                f.write(
                    f"| {i+1} | `{ck['basename']}` | {ck['stage']} | "
                    f"{matched['model_params_M']:.1f}M | "
                    f"{matched['model_config']['n_layers']} | "
                    f"{matched['model_config']['d_ff']} | "
                    f"{matched['model_config']['ctx_len']} | "
                    f"{matched['checkpoint_meta']['best_val_loss']:.4f} |\n"
                )
            else:
                f.write(f"| {i+1} | `{ck['basename']}` | {ck['stage']} | — | — | — | — | — |\n")

        # Results grouped by prompt
        f.write("\n---\n\n## Responses\n\n")

        for prompt_idx, prompt in enumerate(prompts):
            f.write(f"### Prompt {prompt_idx + 1}\n\n")
            f.write(f"> **{prompt}**\n\n")

            # Filter results for this prompt
            prompt_results = [
                r for r in all_results["results"]
                if r["prompt"] == prompt
            ]

            for settings in settings_grid:
                setting_label = (
                    f"temp={settings['temperature']}, "
                    f"top_k={settings['top_k']}, "
                    f"max_tokens={settings['max_tokens']}"
                )
                f.write(f"#### Settings: {setting_label}\n\n")

                # Table of responses across checkpoints
                f.write("| Stage | Checkpoint | Params | Generated Response | Tokens | Time |\n")
                f.write("|:------|:-----------|-------:|:-------------------|-------:|-----:|\n")

                for r in prompt_results:
                    if (
                        r["settings"]["temperature"] == settings["temperature"]
                        and r["settings"]["top_k"] == settings["top_k"]
                        and r["settings"]["max_tokens"] == settings["max_tokens"]
                    ):
                        # Truncate text for table
                        text = r["generated_text"].replace("\n", " ").replace("|", "\\|")
                        if len(text) > 200:
                            text = text[:197] + "..."

                        f.write(
                            f"| {r['stage']} | `{r['checkpoint']}` | "
                            f"{r['model_params_M']:.1f}M | "
                            f"{text} | "
                            f"{r['num_generated_tokens']} | "
                            f"{r['generation_time_s']}s |\n"
                        )

                f.write("\n")

            # Also write expanded view (easier to read for long text)
            f.write(f"<details>\n<summary>📖 Full responses for Prompt {prompt_idx + 1}</summary>\n\n")
            for r in prompt_results:
                setting_label = (
                    f"temp={r['settings']['temperature']}, "
                    f"top_k={r['settings']['top_k']}"
                )
                f.write(f"**{r['stage']}** (`{r['checkpoint']}`, {r['model_params_M']:.1f}M) — {setting_label}\n\n")
                f.write(f"```\n{r['full_text']}\n```\n\n")
            f.write("</details>\n\n---\n\n")

        f.write("*Report generated by `scripts/generate_responses.py`*\n")


if __name__ == "__main__":
    main()
