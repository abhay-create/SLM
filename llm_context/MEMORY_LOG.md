# Memory Log

Append durable session notes here. Keep entries short and factual.

## 2026-04-25

- User reported `pipeline_output.log` OOM during Stage 2. Analysis found the crash occurs while moving the source model to CUDA, before training settings such as `batch_size`, `seq_len`, optimizer state, or dataset loading matter.
- User clarified the training machine is accessed over SSH and has multiple users. Important operational finding: unless GPU isolation is configured, Linux users share the same physical GPU memory pool shown by `nvidia-smi`.
- Added the initial `llm_context/` replay-buffer prototype with system prompt, current state, index, memory log, decisions, findings, tasks, code map, tagged cards, templates, and retrieval/update helper script.

## 2026-04-25

- Actor: Codex
- Event: Tested the LLM context protocol against a logging/benchmarking task. Context retrieval was sufficient to identify the right files, but lacked a durable logging contract.
- Evidence: Added `llm_context/context_cards/logging-benchmarking-contract.md`, `docs/logging_and_benchmarking.md`, expanded `src/logger.py`, patched `train_curriculum.py` and `train_expansion.py`, and added `scripts/summarize_benchmarks.py`.
- Follow-up: After real training runs, run `python scripts/summarize_benchmarks.py` and include `docs/benchmark_summary.md` in result review.

## 2026-04-25

- Actor: Codex
- Event: Ran a correctness/claims audit across the context mechanism and training pipeline implementation.
- Evidence: Patched `src/model.py`, `expand_model.py`, `src/curriculum_dataset.py`, `scripts/context_replay.py`, docs, and context cards. Added `llm_context/context_cards/correctness-audit-2026-04-25.md`.
- Follow-up: Run full model behavioral checks on an environment with `torch` installed before publishing model-level generation or expansion-preservation claims.

## 2026-04-25

- Actor: Antigravity (Claude Opus 4.6)
- Event: Read LLM context system, fixed `expand_context_length()` crash in `expand_model.py` (`_copy_non_layer_parameters` was copying mismatched pos_emb shapes), created `run_pipeline_nohup.sh`, launched full expansion pipeline (stages 2→6) via nohup.
- Evidence: Pipeline running as PID 2566936. Stage 2 training confirmed active (~4.1 steps/s). GPU was free at launch (428 MiB / 16 GB used). Logs written to `Logs/pipeline_full_*.log` and `Logs/pipeline_stage_*_*.log`.
- Follow-up: Monitor pipeline completion. After all stages finish, run `python scripts/summarize_benchmarks.py` and review `docs/benchmark_summary.md`. Update context with training outcomes and any new findings.
