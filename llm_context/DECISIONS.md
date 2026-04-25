# Decisions

## 2026-04-25 - Use a file-based LLM replay buffer

Decision: Implement the project memory system as Markdown files with YAML-style front matter plus a small standard-library helper script.

Rationale: This keeps the prototype portable across LLM tools, SSH sessions, and machines. It avoids requiring a vector database or external service while still allowing tagged retrieval and structured updates.

Consequences:

- Future LLMs can read `llm_context/CURRENT_STATE.md` for fast onboarding.
- Future LLMs can retrieve relevant cards from `llm_context/context_cards/`.
- The system depends on LLM discipline and review, so source files must still be verified before edits.

## 2026-04-25 - Context cards are pointers, not source of truth

Decision: Store summaries, findings, and file pointers in context cards, but require direct source inspection before code changes.

Rationale: Context can become stale. The replay buffer should reduce repeated analysis, not replace the codebase.
