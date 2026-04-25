# Current State

Last updated: 2026-04-25

## Project Snapshot

This repository is a Small Language Model curriculum and expansion project. It
trains a decoder-only Transformer from a TinyStories baseline toward richer
narrative domains through staged curriculum training and model expansion.

The local workspace has a `.git` directory and currently appears to be a usable
Git checkout.

## Active User Goal

Test whether the file-based LLM context system gives future models enough
context to inspect and improve logging/benchmarking for the training pipeline.
The user asked for work on two fronts:

- as the LLM context environment developer, verify and improve the durable
  context handoff
- as a training-pipeline LLM, check and update logging/benchmarking needed for
  publishable results

## Context Adequacy Finding

The context system was sufficient to route inspection to the right files:

- `src/logger.py`
- `train_curriculum.py`
- `train_expansion.py`
- `docs/benchmarking.md`
- `docs/training_flow.md`

The missing durable memory was a dedicated logging/benchmarking contract that
lists exact emitted metrics, artifacts, and verification commands. That gap is
now covered by `llm_context/context_cards/logging-benchmarking-contract.md` and
`docs/logging_and_benchmarking.md`.

## Current Logging/Benchmarking State

`TrainingLogger` now emits a stable all-stage CSV schema with:

- validation loss and perplexity for `s0`, `s1`, `s2`, `roc`, `simple`, `child`,
  and `wp`
- target metric tracking through `val_key`, `current_val`, and `best_val`
- forgetting/replay signals including `ts_forgetting`, `ts_forgetting_ema`, and
  `replay_frac`
- curriculum, tier, gradient, throughput, wall-clock, and GPU-memory metrics
- a matching `.meta.json` sidecar per run with stage/run config and schema

`scripts/summarize_benchmarks.py` aggregates available CSVs into
`docs/benchmark_summary.md` and optional JSON.

## Current Correctness Audit State

A second audit found and patched these mismatches:

- cached generation prefill is now causal when building KV cache
- cached generation falls back to uncached sliding-window generation if the
  requested length exceeds `ctx_len`
- expansion utilities now copy untied `lm_head` parameters and preserve FFN
  biases during widening
- curriculum replay now truncates longer replay chunks, skips shorter chunks,
  and logs `replay_frac=0.0` when no usable replay pool loads
- docs/context now distinguish exact FFN widening from warm-start noisy depth
  cloning and approximate learned-position context interpolation
- `scripts/context_replay.py check` now validates `CONTEXT_INDEX.yaml`
  consistency, not only card front matter

Local verification passed for syntax, context checks, summarizer execution, and
a replay-loader smoke test. Full model behavior tests could not run locally
because `torch` is not installed in the available Python runtimes.

## Current Training Finding

`pipeline_output.log` shows Stage 2 failed before training began:

- failure happens at `model = SLM(old_cfg).to(device)` in `train_expansion.py`
- this is before dataset loading, optimizer creation, batches, or Stage 2
  forward/backward
- the likely immediate cause is unavailable GPU memory, not Stage 2 `batch_size`
  or `seq_len`

The user also reported SSH access to a shared machine with multiple Linux users.
In that setup, all users normally share the same physical GPU memory pool unless
isolation such as MIG, containers with device restrictions, or exclusive compute
mode is configured.

## Environment Notes

- Reported GPU: NVIDIA GeForce RTX 4060 Ti, 16 GB VRAM.
- `nvidia-smi` showed about 14.2 GB of 16.38 GB used.
- A `python3` process was using about 13.8 GB VRAM.
- `MIG M.` was `N/A`, so there is no MIG partitioning.
- `Compute M.` was `Default`, so the GPU is not exclusive to one process or user.

## Next Recommended Actions

1. Use the context prototype for future LLM sessions.
2. Retrieve `logging-benchmarking-contract` before changing metrics or preparing
   stage reports.
3. On the SSH machine, run `nvidia-smi` and `ps -o user,pid,cmd -p <PID>` before
   starting training.
4. Verify the Stage 0 checkpoint config on CPU before changing training
   hyperparameters.
5. After any real training run, run `python scripts/summarize_benchmarks.py`.
6. On the training environment with `torch` installed, run behavioral smoke
   tests for cached generation and expansion preservation before publishing
   model-level claims.
