# Logging and Benchmarking Contract

This project uses one logging contract for both `train_curriculum.py` and
`train_expansion.py`.

## Primary Artifacts

- `logs/stage*.csv`: per-evaluation training metrics.
- `logs/stage*.meta.json`: run metadata sidecar with stage, host, PID, source
  config, CSV schema, and run configuration.
- `docs/curriculum_capabilities.md`: end-of-stage capability report generated
  by `src/capability_logger.py`.
- `docs/benchmark_summary.md`: optional aggregate report generated from CSV logs.

## CSV Metric Groups

Validation metrics:

- Loss and perplexity for `s0`, `s1`, `s2`, `roc`, `simple`, `child`, and `wp`.
- `val_key`, `current_val`, and `best_val` identify the stage-driving metric.

Curriculum and anti-forgetting metrics:

- `curr_frac`, `curriculum_status`
- `tier_easy`, `tier_med`, `tier_hard`
- `kv_div`, `grad_norm`, `deep_grad_norm`
- `replay_frac`, `ts_forgetting`, `ts_forgetting_ema`
- `anchor_reg`, `si_penalty`

Operational metrics:

- `elapsed_s`, `step_time_s`
- `tokens_per_sec`, `interval_tokens_per_sec`
- `gpu_mem_alloc_mb`, `gpu_mem_reserved_mb`, `gpu_mem_peak_mb`

## Benchmark Summary

After a run, summarize all available logs:

```bash
python scripts/summarize_benchmarks.py
```

Optional JSON output:

```bash
python scripts/summarize_benchmarks.py --json-output docs/benchmark_summary.json
```

The summary table is designed for quick experiment comparison and publication
drafting. It reports rows, final token count, best validation value, forgetting,
replay usage, throughput, peak GPU memory, and per-domain best/latest losses.

## Minimum Publishability Checklist

Before publishing results for a stage, verify that the run has:

- A CSV log with non-empty target-domain validation loss.
- A metadata sidecar showing the exact stage config and source checkpoint.
- TinyStories forgetting (`ts_forgetting` and `ts_forgetting_ema`) if `s0` is
  present in validation.
- Replay fraction history if replay sources were configured and usable chunks loaded.
- End-of-stage capability logging in `docs/curriculum_capabilities.md`.
- A benchmark summary generated from the final CSV logs.

## Known Context-System Lesson

The LLM context system was sufficient to route the inspection to
`src/logger.py`, `train_curriculum.py`, `train_expansion.py`, and
`docs/benchmarking.md`. The missing durable memory was a retrievable logging
contract that lists the exact columns, artifacts, and verification command.
Future LLMs should read the `logging-benchmarking-contract` context card before
modifying metrics or publishing results.
