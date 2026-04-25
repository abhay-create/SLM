# Memory Log

Append durable session notes here. Keep entries short and factual.

## 2026-04-25

- User reported `pipeline_output.log` OOM during Stage 2. Analysis found the crash occurs while moving the source model to CUDA, before training settings such as `batch_size`, `seq_len`, optimizer state, or dataset loading matter.
- User clarified the training machine is accessed over SSH and has multiple users. Important operational finding: unless GPU isolation is configured, Linux users share the same physical GPU memory pool shown by `nvidia-smi`.
- Added the initial `llm_context/` replay-buffer prototype with system prompt, current state, index, memory log, decisions, findings, tasks, code map, tagged cards, templates, and retrieval/update helper script.
