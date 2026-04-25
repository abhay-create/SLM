---
id: llm-memory-contract
title: LLM Memory Contract
type: prompt
status: active
priority: critical
tags: [llm, memory, context, replay, prompt, update]
updated: 2026-04-25
summary: All LLMs must read startup files, retrieve relevant cards, and update memory after durable findings or changes.
---

# LLM Memory Contract

This project uses `llm_context/` as a file-based replay buffer.

Every future LLM must:

1. Read startup files.
2. Retrieve relevant cards by task keywords.
3. Use cards to choose source files to inspect.
4. Verify source before editing.
5. Update context after durable findings, decisions, user instructions, or workflow changes.

Required startup files:

- `llm_context/README.md`
- `llm_context/SYSTEM_PROMPT.md`
- `llm_context/CURRENT_STATE.md`

Recommended retrieval command:

```bash
python scripts/context_replay.py retrieve "<task keywords>"
```

Minimum update after meaningful work:

- append `llm_context/MEMORY_LOG.md`
- update `llm_context/CURRENT_STATE.md` if status changed
- add or update a context card when the information should be retrieved later

When to read:

- at the start of every LLM session
- before handing off work to another model
- before changing context-system files

Source pointers:

- `LLM_CONTEXT.md`
- `llm_context/README.md`
- `llm_context/SYSTEM_PROMPT.md`
- `scripts/context_replay.py`
