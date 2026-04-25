---
id: correctness-audit-2026-04-25
title: Correctness Audit 2026-04-25
type: finding
status: active
priority: high
tags: [correctness, audit, generation, expansion, replay, context]
updated: 2026-04-25
summary: Audit found and patched causal cached-generation masking, expansion parameter-copy gaps, replay logging truthfulness, and context-index validation.
---

# Correctness Audit 2026-04-25

The user asked for a second pass focused on logical issues, false claims, wrong
implementations, and misleading implementations.

Patched findings:

- Cached generation prefill in `src/model.py` used `use_cache=True` with
  `is_causal=False`, allowing prompt tokens to attend bidirectionally. Prefill
  is now causal; one-token cached decode remains unmasked against past/current
  cache entries.
- `SLM.generate(..., use_cache=True)` could index past `ctx_len`; it now falls
  back to uncached sliding-window generation when prompt plus generated length
  exceeds the context window.
- Expansion utilities did not copy untied `lm_head` parameters and skipped FFN
  biases during FFN widening. `expand_model.py` now copies non-layer parameters
  consistently and preserves FFN biases.
- `CurriculumStageDataset` could log a nonzero `replay_frac` even when replay
  sources did not load. It now truncates longer replay chunks, skips shorter
  chunks, and forces `replay_frac=0.0` when no usable replay pool exists.
- Docs/context previously overstated depth cloning as function-preserving. They
  now distinguish exact FFN widening from warm-start noisy depth cloning and
  approximate learned-position context interpolation.
- `scripts/context_replay.py check` previously validated only card front matter.
  It now also validates `CONTEXT_INDEX.yaml` paths, ids, and unindexed cards.

Verification:

- `python -m py_compile ...` passed for changed Python files.
- `python scripts/context_replay.py check` passed.
- `python scripts/summarize_benchmarks.py` wrote `docs/benchmark_summary.md`.
- Replay loader smoke test passed using a dummy `torch` module and bundled
  Python with NumPy.

Blocked local verification:

- Behavioral model tests requiring real `torch` could not run in this local
  Python environment because `torch` is not installed. Run them on the training
  environment before publishing model-level results.

When to read:

- before changing generation, expansion, replay, or context-check behavior
- before making strong claims about function preservation
- before publishing benchmark or training results

Source pointers:

- `src/model.py`
- `expand_model.py`
- `src/curriculum_dataset.py`
- `scripts/context_replay.py`
- `README.md`
- `docs/benchmarking.md`
- `docs/training_flow.md`
