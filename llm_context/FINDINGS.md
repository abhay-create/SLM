# Findings

## 2026-04-25 - Stage 2 OOM occurs before training starts

Evidence: `pipeline_output.log` fails at `model = SLM(old_cfg).to(device)` in `train_expansion.py`.

Interpretation: The immediate OOM is not caused by Stage 2 batch size, sequence length, optimizer state, or activations because none of those have been created yet.

Likely cause: GPU VRAM was already heavily occupied, or the source checkpoint config is unexpectedly large.

Recommended checks:

```bash
nvidia-smi
ps -o user,pid,cmd -p <PID>
```

```bash
python - <<'PY'
import torch
ckpt = torch.load("checkpoints/TinyStoriesWithCurriculum.pt", map_location="cpu", weights_only=False)
print(ckpt["config"])
PY
```

## 2026-04-25 - Multiple SSH users usually share GPU memory

Evidence: User reported multiple users on the SSH machine. The reported `nvidia-smi` output showed one RTX 4060 Ti, `MIG M.` as `N/A`, and `Compute M.` as `Default`.

Interpretation: All users normally see and compete for the same physical GPU memory pool unless the machine is configured with isolation.

Operational impact: A heavy job from another user can cause this project's training run to fail with OOM even if the code and hyperparameters are reasonable.

## 2026-04-25 - Logging contract was under-specified for publishable results

Evidence: `docs/benchmarking.md` and `README.md` described all-domain and
operational logging, but `src/logger.py` only persisted `val_s0`, `val_s1`, and
`val_s2` to CSV. Later-stage domains such as `roc`, `simple`, `child`, and `wp`
were printed to console but not recorded as stable CSV columns. In
`train_expansion.py`, the forgetting EMA was passed as `forgetting_ema` while
the logger expected `ts_forgetting_ema`, leaving the CSV EMA column empty.

Interpretation: The prior context was enough to locate the right subsystem, but
future LLMs needed a durable logging/benchmarking contract to know what metrics
must exist for stage comparison and result publication.

Resolution: Added a stable all-stage logging schema, run metadata sidecars,
benchmark summary script, documentation, and a retrievable
`logging-benchmarking-contract` context card.

## 2026-04-25 - Correctness audit found implementation and claim mismatches

Evidence:

- `src/model.py` cached-generation prompt prefill used `is_causal=False`, so
  prompt tokens could attend bidirectionally while building KV cache.
- `expand_model.py` did not copy untied `lm_head` parameters during expansion
  and did not preserve FFN biases in `expand_ffn_width()`.
- `src/curriculum_dataset.py` kept a nonzero `replay_frac` even when replay
  sources were missing or unusable.
- `scripts/context_replay.py check` validated card front matter but not
  `CONTEXT_INDEX.yaml`, allowing unindexed cards or stale paths.
- Docs/context described noisy depth cloning as function-preserving even though
  appending active Transformer blocks changes outputs.

Resolution:

- Patched cached prefill masking and long cached-generation fallback.
- Patched expansion parameter copying and FFN bias preservation.
- Patched replay loading to truncate longer chunks, skip shorter chunks, and
  log `replay_frac=0.0` when no usable replay pool exists.
- Patched context checks to validate index paths, ids, and unindexed cards.
- Corrected docs/context to distinguish exact FFN widening from warm-start depth
  cloning and approximate context interpolation.

Verification note: Local syntax and context checks passed. Full model-behavior
tests require `torch`, which is not installed in this local Python environment.
