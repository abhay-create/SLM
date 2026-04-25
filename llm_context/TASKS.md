# Tasks

## Active

- Use the new `llm_context/` system as the prototype memory layer for future LLM sessions.
- Verify the context retrieval flow with `python scripts/context_replay.py retrieve "<task keywords>"`.
- On the training machine, check whether the GPU is free before rerunning Stage 2.
- Verify `checkpoints/TinyStoriesWithCurriculum.pt` config on CPU before changing Stage 2 hyperparameters.

## Backlog

- Add optional CI-style validation that every context card has required front matter.
- Add a pre-commit or manual checklist reminding LLMs to update `llm_context/`.
- Add cards for any future model-level findings, training outcomes, or pipeline fixes.
