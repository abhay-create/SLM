"""
logger.py

TrainingLogger writes the metrics needed for experiment review to:

- a per-run CSV (`logs/stage...csv`)
- a small metadata sidecar (`logs/stage...meta.json`)
- the console, for live monitoring

The CSV schema is intentionally stable across curriculum and expansion stages.
Every stage gets the same validation-domain columns, anti-forgetting signals,
curriculum signals, and operational benchmark signals.
"""

from __future__ import annotations

import csv
import json
import math
import os
import socket
import time
from datetime import datetime
from typing import Any


DOMAIN_KEYS = ("s0", "s1", "s2", "roc", "simple", "child", "wp")

BASE_COLUMNS = [
    "step",
    "stage",
    "tokens_seen",
    "train_loss",
]

VAL_COLUMNS = [f"val_{key}" for key in DOMAIN_KEYS]
PPL_COLUMNS = [f"ppl_{key}" for key in DOMAIN_KEYS]

RUN_COLUMNS = [
    "val_key",
    "current_val",
    "best_val",
    "lr",
    "curr_frac",
    "curriculum_status",
    "tier_easy",
    "tier_med",
    "tier_hard",
    "kv_div",
    "replay_frac",
    "anchor_reg",
    "si_penalty",
    "ts_forgetting",
    "ts_forgetting_ema",
    "grad_norm",
    "deep_grad_norm",
]

OPERATIONAL_COLUMNS = [
    "elapsed_s",
    "step_time_s",
    "tokens_per_sec",
    "interval_tokens_per_sec",
    "gpu_mem_alloc_mb",
    "gpu_mem_reserved_mb",
    "gpu_mem_peak_mb",
]

COLUMNS = BASE_COLUMNS + VAL_COLUMNS + PPL_COLUMNS + RUN_COLUMNS + OPERATIONAL_COLUMNS + ["note"]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: Any, precision: int = 4) -> str:
    number = _safe_float(value)
    if math.isnan(number):
        return ""
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{precision}f}"


def _format_scientific(value: Any) -> str:
    number = _safe_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.2e}"


def _ppl(loss: Any) -> float:
    loss_value = _safe_float(loss)
    if math.isnan(loss_value):
        return float("nan")
    if loss_value >= 20:
        return float("inf")
    return math.exp(loss_value)


def _gpu_stats_mb() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_mem_alloc_mb": float("nan"),
                "gpu_mem_reserved_mb": float("nan"),
                "gpu_mem_peak_mb": float("nan"),
            }
        return {
            "gpu_mem_alloc_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "gpu_mem_peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
    except Exception:
        return {
            "gpu_mem_alloc_mb": float("nan"),
            "gpu_mem_reserved_mb": float("nan"),
            "gpu_mem_peak_mb": float("nan"),
        }


class TrainingLogger:
    def __init__(
        self,
        stage: int | str,
        log_dir: str = "logs",
        run_config: dict[str, Any] | None = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.host = socket.gethostname()
        self.pid = os.getpid()
        safe_stage = str(stage).replace(os.sep, "_").replace(" ", "_")
        fname = f"stage{safe_stage}_{self.host}_{self.pid}_{timestamp}.csv"

        self.path = os.path.join(log_dir, fname)
        self.meta_path = os.path.splitext(self.path)[0] + ".meta.json"
        self.stage = stage
        self.started_at = time.perf_counter()
        self.last_log_at = self.started_at
        self.last_log_step = 0
        self.last_tokens_seen = 0
        self.metadata = {
            "stage": str(stage),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": self.host,
            "pid": self.pid,
            "csv_path": self.path,
            "columns": COLUMNS,
            "domain_keys": DOMAIN_KEYS,
            "run_config": run_config or {},
        }

        self._init_csv()
        self._write_metadata()

    def _init_csv(self):
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        print(f"[logger] Logging to {self.path}")
        print(f"[logger] Metadata to {self.meta_path}")

    def _write_metadata(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)

    def _operational_row(self, step: int, tokens_seen: int) -> dict[str, str]:
        now = time.perf_counter()
        elapsed = max(now - self.started_at, 1e-9)
        interval_elapsed = max(now - self.last_log_at, 1e-9)
        step_delta = max(step - self.last_log_step, 1)
        token_delta = max(tokens_seen - self.last_tokens_seen, 0)
        gpu = _gpu_stats_mb()

        row = {
            "elapsed_s": _format_float(elapsed, 2),
            "step_time_s": _format_float(interval_elapsed / step_delta, 4),
            "tokens_per_sec": _format_float(tokens_seen / elapsed, 2),
            "interval_tokens_per_sec": _format_float(token_delta / interval_elapsed, 2),
            "gpu_mem_alloc_mb": _format_float(gpu["gpu_mem_alloc_mb"], 1),
            "gpu_mem_reserved_mb": _format_float(gpu["gpu_mem_reserved_mb"], 1),
            "gpu_mem_peak_mb": _format_float(gpu["gpu_mem_peak_mb"], 1),
        }

        self.last_log_at = now
        self.last_log_step = step
        self.last_tokens_seen = tokens_seen
        return row

    def log(
        self,
        step: int,
        tokens_seen: int,
        train_loss: float,
        val_losses: dict,
        lr: float,
        note: str = "",
        curriculum: dict | None = None,
    ):
        curriculum = curriculum or {}
        val_losses = dict(val_losses or {})
        val_key = str(curriculum.get("val_key", ""))
        current_val = curriculum.get("current_val")
        if current_val is None and val_key in val_losses:
            current_val = val_losses[val_key]

        row = {column: "" for column in COLUMNS}
        row.update(
            {
                "step": step,
                "stage": self.stage,
                "tokens_seen": tokens_seen,
                "train_loss": _format_float(train_loss, 4),
                "val_key": val_key,
                "current_val": _format_float(current_val, 4),
                "best_val": _format_float(curriculum.get("best_val"), 4),
                "lr": _format_scientific(lr),
                "curr_frac": _format_float(curriculum.get("fraction"), 3),
                "curriculum_status": curriculum.get("status", curriculum.get("curriculum_status", "")),
                "tier_easy": _format_float(curriculum.get("tier_easy"), 4),
                "tier_med": _format_float(curriculum.get("tier_medium"), 4),
                "tier_hard": _format_float(curriculum.get("tier_hard"), 4),
                "kv_div": _format_float(curriculum.get("kv_div"), 4),
                "replay_frac": _format_float(curriculum.get("replay_frac"), 3),
                "anchor_reg": _format_float(curriculum.get("anchor_reg"), 6),
                "si_penalty": _format_float(curriculum.get("si_penalty"), 6),
                "ts_forgetting": _format_float(curriculum.get("ts_forgetting"), 4),
                "ts_forgetting_ema": _format_float(
                    curriculum.get("ts_forgetting_ema", curriculum.get("forgetting_ema")),
                    4,
                ),
                "grad_norm": _format_float(curriculum.get("grad_norm"), 4),
                "deep_grad_norm": _format_float(curriculum.get("deep_grad_norm"), 4),
                "note": note,
            }
        )

        for key in DOMAIN_KEYS:
            loss = val_losses.get(key)
            row[f"val_{key}"] = _format_float(loss, 4)
            row[f"ppl_{key}"] = _format_float(_ppl(loss), 2)

        row.update(self._operational_row(step, tokens_seen))

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

        console_vals = []
        for key in DOMAIN_KEYS:
            value = _safe_float(val_losses.get(key))
            if not math.isnan(value):
                console_vals.append(f"val_{key}={value:.4f}")
        val_str = "  ".join(console_vals)
        print(
            f"[stage{self.stage}] step={step:>6}  "
            f"tok={tokens_seen/1e6:>6.1f}M  "
            f"train={train_loss:.4f}  "
            f"{val_str}  "
            f"lr={lr:.1e}"
            + (f"  [{note}]" if note else "")
        )

    def log_exit(self, reason: str, step: int, tokens_seen: int):
        print(
            f"\n[logger] Stage {self.stage} exit at step {step} "
            f"({tokens_seen/1e6:.1f}M tokens): {reason}\n"
        )
        row = {column: "" for column in COLUMNS}
        row.update(
            {
                "step": step,
                "stage": self.stage,
                "tokens_seen": tokens_seen,
                "note": f"EXIT:{reason}",
            }
        )
        row.update(self._operational_row(step, tokens_seen))

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

        self.metadata["exit"] = {
            "reason": reason,
            "step": step,
            "tokens_seen": tokens_seen,
            "elapsed_s": _safe_float(row["elapsed_s"]),
        }
        self._write_metadata()
