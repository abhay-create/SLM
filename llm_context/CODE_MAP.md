# Code Map

Use this file to avoid re-reading the entire repo. Verify exact source before editing.

## Main Entry Points

- `run_pipeline.sh`: runs expansion stages 2 through 6 sequentially.
- `train_expansion.py`: main orchestrator for staged expansion training.
- `train_curriculum.py`: curriculum training utilities, evaluation helpers, checkpoint helpers, LR schedule, early exit detectors.
- `expand_model.py`: model growth utilities; FFN widening is exact, depth cloning is warm-started, and learned-position context interpolation is approximate.

## Core Modules

- `src/model.py`: decoder-only SLM architecture, `SLMConfig`, generation, parameter count helpers.
- `src/dataset.py`: tokenization, chunking, validation set loaders, dataset iterators.
- `src/curriculum_dataset.py`: curriculum dataset, replay pool handling, competence scheduler.
- `src/logger.py`: CSV training logger.
- `src/capability_logger.py`: end-of-stage capability logging across validation domains and generation prompts.
- `src/writingprompts_dataset.py`: WritingPrompts iterator.
- `src/score_difficulty.py`: difficulty scoring using reference model features.

## Configuration

- `configs/stage0_full.yaml`: Stage 0 TinyStories curriculum config.
- `configs/expansion_stages.yaml`: source of truth for stages 2 through 6 expansion and training configs.

## Diagnostics

- `pipeline_output.log`: latest captured pipeline failure. Current known failure is CUDA OOM during source model `.to(device)` for Stage 2.
- `system_info.txt` and `system_report.txt`: machine and hardware notes captured from the training environment.
- `docs/training_flow.md`: detailed training flow and call chain.
- `docs/benchmarking.md`: evaluation and monitoring recommendations.
- `docs/logging_and_benchmarking.md`: exact logging schema, artifacts, and publication checklist.
- `scripts/summarize_benchmarks.py`: aggregates `logs/stage*.csv` files into `docs/benchmark_summary.md` and optional JSON.
- `scripts/generate_responses.py`: multi-prompt, multi-checkpoint generation script. Loads all `*best.pt` models, generates text with configurable settings (temperature, top_k, max_tokens), saves JSON + Markdown report to `Logs/generations/`.

## LLM Context System

- `LLM_CONTEXT.md`: root entry point.
- `llm_context/`: shared replay-buffer memory.
- `scripts/context_replay.py`: helper for retrieving context cards, validating card/index consistency, creating indexed cards, and appending memory log entries.
