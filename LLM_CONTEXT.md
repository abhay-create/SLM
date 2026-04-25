# LLM Context Entry Point

Start here before any future LLM session works on this repository:

1. Read `llm_context/README.md`.
2. Read `llm_context/SYSTEM_PROMPT.md`.
3. Read `llm_context/CURRENT_STATE.md`.
4. Use `python scripts/context_replay.py retrieve "<task keywords>"` to select the most relevant context cards.
5. After making decisions, findings, code changes, or receiving important user instructions, update the context system.

This project uses `llm_context/` as a lightweight replay buffer for LLM collaboration. It is not only a summary file. It stores durable project state, decisions, critical findings, and tagged context cards so future models can avoid repeating the same analysis.
