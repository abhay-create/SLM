"""
curriculum.py — Orchestrates all three training stages sequentially.

Calls train.py for each stage and passes the previous stage's best
checkpoint forward automatically.

Usage:
  python curriculum.py --tokenizer tokenizers/tokenizer_50k.json
  python curriculum.py --tokenizer tokenizers/tokenizer_50k.json --start_stage 1
  python curriculum.py --tokenizer tokenizers/tokenizer_50k.json --stages 0 1
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


STAGE_CONFIGS = {
    0: "configs/stage0.yaml",
    1: "configs/stage1.yaml",
    2: "configs/stage2.yaml",
}


def run_stage(
    stage          : int,
    tokenizer      : str,
    checkpoint_dir : str,
    log_dir        : str,
    cache_dir      : str,
    pos_type       : str,
    prev_ckpt      : str = None,
    resume         : bool = False,
):
    config = STAGE_CONFIGS[stage]
    cmd = [
        sys.executable, "train.py",
        "--stage",           str(stage),
        "--config",          config,
        "--tokenizer",       tokenizer,
        "--checkpoint_dir",  checkpoint_dir,
        "--log_dir",         log_dir,
        "--cache_dir",       cache_dir,
        "--pos_type",        pos_type,
    ]
    if prev_ckpt:
        cmd += ["--prev_checkpoint", prev_ckpt]
    if resume:
        cmd += ["--resume"]

    print(f"\n{'='*60}")
    print(f"  Starting Stage {stage}")
    print(f"  Config     : {config}")
    print(f"  Prev ckpt  : {prev_ckpt or 'None (fresh start)'}")
    print(f"  Resume     : {resume}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n[curriculum] Stage {stage} exited with code {result.returncode}")
        print(f"[curriculum] Check logs in {log_dir}/ for details")
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer",      type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir",        type=str, default="logs")
    p.add_argument("--cache_dir",      type=str, default="cache")
    p.add_argument("--pos_type",       type=str, default="learnable",
                   choices=["learnable", "rope"])
    p.add_argument("--start_stage",   type=int, default=0,
                   help="Which stage to start from (0/1/2)")
    p.add_argument("--stages",        type=int, nargs="+", default=None,
                   help="Explicit list of stages to run, e.g. --stages 1 2")
    p.add_argument("--resume_stage",  type=int, default=None,
                   help="Resume a specific stage from its last best checkpoint")
    args = p.parse_args()

    stages_to_run = args.stages or list(range(args.start_stage, 3))
    print(f"[curriculum] Running stages: {stages_to_run}")
    print(f"[curriculum] pos_type: {args.pos_type}")
    print(f"[curriculum] tokenizer: {args.tokenizer}")

    for stage in stages_to_run:
        # Find previous stage checkpoint (if applicable)
        prev_ckpt = None
        if stage > 0:
            prev_stage = stage - 1
            candidate  = os.path.join(args.checkpoint_dir, f"stage{prev_stage}_best.pt")
            if os.path.exists(candidate):
                prev_ckpt = candidate
            else:
                print(f"[curriculum] WARNING: No checkpoint found for stage {prev_stage} at {candidate}")
                print(f"[curriculum] Stage {stage} will initialise with random weights")

        resume = (args.resume_stage == stage)

        success = run_stage(
            stage          = stage,
            tokenizer      = args.tokenizer,
            checkpoint_dir = args.checkpoint_dir,
            log_dir        = args.log_dir,
            cache_dir      = args.cache_dir,
            pos_type       = args.pos_type,
            prev_ckpt      = prev_ckpt,
            resume         = resume,
        )

        if not success:
            print(f"[curriculum] Halting at stage {stage} due to error.")
            sys.exit(1)

    print(f"\n[curriculum] All stages complete.")
    print(f"[curriculum] Final checkpoint: checkpoints/stage2_best.pt")


if __name__ == "__main__":
    main()
