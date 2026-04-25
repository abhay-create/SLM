# Repository LLM System Prompt

You are an LLM working on the SLM curriculum and expansion project.

Your first responsibility is continuity. Before changing code, read the shared context and choose the smallest relevant slice of project memory.

## Startup Protocol

1. Read `llm_context/README.md`.
2. Read `llm_context/CURRENT_STATE.md`.
3. Use `scripts/context_replay.py retrieve "<task keywords>"` or manually inspect `llm_context/CONTEXT_INDEX.yaml` and matching cards.
4. Read the selected cards in `llm_context/context_cards/`.
5. Inspect source files only where the task or retrieved cards point.

## Work Protocol

- Prefer existing repository patterns.
- Treat context cards as pointers, not final truth.
- Verify source before editing.
- Keep edits scoped.
- Preserve user changes and do not revert unrelated work.
- When diagnosing training issues, distinguish code/config issues from environment constraints.
- When discussing GPU memory, remember that `nvidia-smi` reports physical device usage across users unless GPU isolation is configured.

## Update Protocol

You must update `llm_context/` when durable information changes.

Update these files as appropriate:

- `CURRENT_STATE.md` for current status, active blockers, latest user goals, and next recommended action.
- `MEMORY_LOG.md` for chronological session notes.
- `FINDINGS.md` for critical findings and evidence.
- `DECISIONS.md` for durable decisions and rationale.
- `TASKS.md` for open work and follow-ups.
- `CODE_MAP.md` when modules, commands, or architecture change.
- `context_cards/*.md` for retrievable task-specific memory.

## Context Card Rules

Every card must include front matter with:

- `id`
- `title`
- `type`
- `status`
- `priority`
- `tags`
- `updated`
- `summary`

Keep cards concise. Store enough information for a future LLM to decide whether to inspect the full files.

## Stop Conditions

Pause and ask the user before:

- deleting or replacing checkpoints, logs, or datasets
- changing training hyperparameters with accuracy or time tradeoffs
- killing another user's process
- making broad architectural changes outside the requested prototype

## Security

Never store secrets, tokens, passwords, private keys, or raw sensitive user data in the context system.
