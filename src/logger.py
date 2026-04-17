"""
logger.py
Logs training metrics to both console and a per-stage CSV.

Columns:
  step | stage | tokens_seen | train_loss | val_s0 | val_s1 | val_s2 | lr | exit_reason
"""

import os
import csv
from datetime import datetime


COLUMNS = [
    "step", "stage", "tokens_seen", "train_loss",
    "val_s0", "val_s1", "val_s2", "lr", "curr_frac", "tier_easy", "tier_med", "tier_hard", "kv_div", "note"
]


class TrainingLogger:
    def __init__(self, stage: int, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path    = os.path.join(log_dir, f"stage{stage}_{timestamp}.csv")
        self.stage   = stage
        self._init_csv()

    def _init_csv(self):
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        print(f"[logger] Logging to {self.path}")

    def log(
        self,
        step        : int,
        tokens_seen : int,
        train_loss  : float,
        val_losses  : dict,   # {"s0": float, "s1": float, "s2": float}
        lr          : float,
        note        : str = "",
        curriculum  : dict = None,
    ):
        curriculum = curriculum or {}
        row = {
            "step"        : step,
            "stage"       : self.stage,
            "tokens_seen" : tokens_seen,
            "train_loss"  : f"{train_loss:.4f}",
            "val_s0"      : f"{val_losses.get('s0', float('nan')):.4f}",
            "val_s1"      : f"{val_losses.get('s1', float('nan')):.4f}",
            "val_s2"      : f"{val_losses.get('s2', float('nan')):.4f}",
            "lr"          : f"{lr:.2e}",
            "curr_frac"   : f"{curriculum.get('fraction', float('nan')):.3f}",
            "tier_easy"   : f"{curriculum.get('tier_easy', float('nan')):.4f}",
            "tier_med"    : f"{curriculum.get('tier_medium', float('nan')):.4f}",
            "tier_hard"   : f"{curriculum.get('tier_hard', float('nan')):.4f}",
            "kv_div"      : f"{curriculum.get('kv_div', float('nan')):.4f}",
            "note"        : note,
        }

        # Append to CSV
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

        # Console
        val_str = "  ".join(
            f"val_{k}={v:.4f}" for k, v in sorted(val_losses.items())
        )
        print(
            f"[stage{self.stage}] step={step:>6}  "
            f"tok={tokens_seen/1e6:>6.1f}M  "
            f"train={train_loss:.4f}  "
            f"{val_str}  "
            f"lr={lr:.1e}"
            + (f"  [{note}]" if note else "")
        )

    def log_exit(self, reason: str, step: int, tokens_seen: int):
        print(f"\n[logger] Stage {self.stage} exit at step {step} "
              f"({tokens_seen/1e6:.1f}M tokens): {reason}\n")
        # Write a final marker row
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow({c: "" for c in COLUMNS} | {
                "step": step,
                "stage": self.stage,
                "tokens_seen": tokens_seen,
                "note": f"EXIT:{reason}",
            })
