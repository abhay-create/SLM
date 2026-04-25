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

## 2026-04-25 - Use one all-stage logging schema

Decision: Keep curriculum and expansion training on one stable
`TrainingLogger` CSV schema instead of adding stage-specific metrics files.

Rationale: Publishable comparisons need the same columns across stages even
when some values are empty. This makes benchmark aggregation simple and avoids
future LLMs having to infer which domains were logged by each stage.

Consequences:

- `src/logger.py` owns the canonical metric columns.
- Both `train_curriculum.py` and `train_expansion.py` pass run configuration to
  logger metadata sidecars.
- `scripts/summarize_benchmarks.py` can summarize all available stage CSVs with
  no external dependencies.

## 2026-04-25 - Validate the context index, not just card metadata

Decision: Extend `scripts/context_replay.py check` so it validates
`CONTEXT_INDEX.yaml` entries against the cards on disk and reports unindexed
cards.

Rationale: The context system depends on durable retrieval. A card with valid
front matter can still be effectively lost if the index drifts.

Consequences:

- New cards created through `scripts/context_replay.py new` are auto-added to
  the index.
- `check` now fails when an indexed path is missing, an indexed id does not
  match the card front matter, or a card exists without an index entry.
