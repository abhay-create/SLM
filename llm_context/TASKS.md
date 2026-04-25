# Tasks

## Active

- Use the new `llm_context/` system as the prototype memory layer for future LLM sessions.
- Verify the context retrieval flow with `python scripts/context_replay.py retrieve "<task keywords>"`.
- Use `llm_context/context_cards/logging-benchmarking-contract.md` before modifying metrics or preparing published training results.
- On the training machine, check whether the GPU is free before rerunning Stage 2.
- Verify `checkpoints/TinyStoriesWithCurriculum.pt` config on CPU before changing Stage 2 hyperparameters.
- After any real training run, run `python scripts/summarize_benchmarks.py` and review `docs/benchmark_summary.md`.
- On a Python environment with `torch` installed, run behavioral smoke tests for cached generation and expansion preservation.

## Backlog

- Add a pre-commit or manual checklist reminding LLMs to update `llm_context/`.
- Add cards for any future model-level findings, training outcomes, or pipeline fixes.
- Consider a dependency-free smoke fixture for `scripts/summarize_benchmarks.py` so the logging schema can be tested without launching training.
- Consider adding `scripts/context_replay.py check` to CI or a pre-commit hook; it now validates card front matter and `CONTEXT_INDEX.yaml` consistency.
