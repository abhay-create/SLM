---
id: logging-benchmarking-contract
title: Logging and Benchmarking Contract
type: workflow
status: active
priority: high
tags: [logging, benchmarking, metrics, training, pipeline, publication]
updated: 2026-04-25
summary: TrainingLogger now emits a stable all-stage CSV schema, metadata sidecars, and can be summarized with scripts/summarize_benchmarks.py.
---

# Logging and Benchmarking Contract

`src/logger.py` is the shared logging surface for both `train_curriculum.py`
and `train_expansion.py`.

It writes:

- `logs/stage*.csv` with validation loss/PPL for `s0`, `s1`, `s2`, `roc`,
  `simple`, `child`, and `wp`
- `logs/stage*.meta.json` with stage/run config, host, PID, schema, and domain keys
- console summaries for live monitoring

Important metrics:

- target-domain tracking: `val_key`, `current_val`, `best_val`
- forgetting/replay: `ts_forgetting`, `ts_forgetting_ema`, `replay_frac`
- stability: `kv_div`, `grad_norm`, `deep_grad_norm`
- tiers: `tier_easy`, `tier_med`, `tier_hard`
- operations: `elapsed_s`, `step_time_s`, `tokens_per_sec`,
  `interval_tokens_per_sec`, `gpu_mem_alloc_mb`, `gpu_mem_reserved_mb`,
  `gpu_mem_peak_mb`

Use this command after runs:

```bash
python scripts/summarize_benchmarks.py
```

It writes `docs/benchmark_summary.md`; add `--json-output` for machine-readable
results.

Why this card exists:

- Before this update, context and docs promised broad logging but the CSV only
  persisted `s0`, `s1`, and `s2`; later-stage domains were console-only.
- Expansion training passed the forgetting EMA as `forgetting_ema`, while the
  logger expected `ts_forgetting_ema`, so the CSV EMA column was empty.
- Replay metrics are only meaningful when a usable replay pool loads; the
  dataset now forces `replay_frac=0.0` if replay sources are missing or too
  short for the target sequence length.

When to read:

- changing logging columns
- publishing or comparing stage results
- debugging replay, forgetting, throughput, or GPU memory behavior
- updating `docs/benchmarking.md` or training report scripts

Source pointers:

- `src/logger.py`
- `train_curriculum.py`
- `train_expansion.py`
- `scripts/summarize_benchmarks.py`
- `docs/logging_and_benchmarking.md`
