# LLM Context System

This directory is the shared memory layer for every LLM working on this repository.

The purpose is to preserve important context across sessions:

- user intent and important conversations
- critical findings from model analysis
- current blockers and environment constraints
- durable decisions and rationale
- codebase map and known risky areas
- task-specific context cards that can be retrieved without re-reading the whole repo

## Required Startup

Every LLM must do this before making repo changes:

1. Read `llm_context/SYSTEM_PROMPT.md`.
2. Read `llm_context/CURRENT_STATE.md`.
3. Run or emulate `python scripts/context_replay.py retrieve "<task keywords>"`.
4. Read only the context cards selected for the task.
5. Inspect source files directly only when the selected context is insufficient or the task touches those files.

## Required Update Rule

Every LLM must update this directory whenever any of these happen:

- the user gives a new durable instruction
- a critical bug, blocker, or environment fact is discovered
- a design decision is made
- a code change alters architecture, commands, configs, or workflow
- an earlier context card becomes stale
- a repeated analysis can be saved for future sessions

Minimum update after meaningful work:

1. Append a dated entry to `MEMORY_LOG.md`.
2. Update `CURRENT_STATE.md` if the current project state changed.
3. Add or update one context card in `context_cards/` if the finding should be retrievable later.
4. Update `DECISIONS.md`, `FINDINGS.md`, `TASKS.md`, or `CODE_MAP.md` when those files are affected.

## Retrieval Safeguards

The context system is a replay buffer, not a replacement for code truth.

- Use context cards to decide where to look.
- Verify exact behavior in source files before editing.
- Do not scan the whole codebase by default.
- Do scan broader code when context is missing, stale, contradictory, or the change is cross-cutting.
- Do not store secrets, credentials, private tokens, or raw sensitive datasets in this directory.

## File Roles

- `SYSTEM_PROMPT.md`: operating instructions for all LLMs.
- `CURRENT_STATE.md`: short canonical state of the project right now.
- `CONTEXT_INDEX.yaml`: machine-readable index and update policy.
- `MEMORY_LOG.md`: append-only chronological session memory.
- `DECISIONS.md`: durable decisions with rationale.
- `FINDINGS.md`: critical findings from analysis.
- `TASKS.md`: open tasks, blockers, and follow-ups.
- `CODE_MAP.md`: stable module map and inspection shortcuts.
- `context_cards/`: tagged replay-buffer cards for selective retrieval.
- `templates/`: formats for new entries and cards.

## Helper Commands

Retrieve relevant context:

```bash
python scripts/context_replay.py retrieve "stage2 gpu oom pipeline"
```

List available context cards:

```bash
python scripts/context_replay.py list
```

Validate card metadata and `CONTEXT_INDEX.yaml` consistency:

```bash
python scripts/context_replay.py check
```

Create a new context card:

```bash
python scripts/context_replay.py new --id my-finding --title "My Finding" --type finding --priority high --tags tag1 tag2 --summary "Short summary."
```

Append a memory log entry:

```bash
python scripts/context_replay.py append-log --actor "LLM" --event "Short durable event."
```
