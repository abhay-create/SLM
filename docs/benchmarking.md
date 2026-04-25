Benchmarking & Monitoring Plan

Purpose
-------
Concrete, actionable tests and logging to detect capability gains and
catastrophic forgetting during curriculum and expansion training.

Core principles
---------------
- Measure per-domain validation loss and perplexity every `eval_interval`.
- Maintain an anchor baseline, normally TinyStories `s0`, saved with
  checkpoints; measure relative forgetting against that anchor.
- Track per-tier performance (`easy`, `medium`, `hard`) and deep-layer
  stability signals (`grad_norm`, `deep_grad_norm`, `kv_div`).
- Use adaptive replay when a usable replay pool is available.
- Persist metrics to CSV, metadata sidecars, and human-friendly reports.

Definitions
-----------
- Val loss: cross-entropy loss on held-out validation sets.
- Perplexity (PPL): `exp(val_loss)`, capped to `inf` for very large losses.
- Relative forgetting: `(val_s0_current - anchor_val) / anchor_val`.
- Tier losses: losses computed on easy/medium/hard chunks via
  `evaluate_by_tier()`.
- Backward Transfer (BWT): loss change on earlier domains after learning a new
  domain.
- Forward Transfer (FWT): improvement on future domains before direct training.
- Global grad norm: mean norm across parameter gradients after backward pass.
- KV divergence: `kv_divergence_metric()` instability signal.

When to evaluate
----------------
- Every `eval_interval` training steps, compute validation metrics for `s0`,
  `s1`, `s2`, `roc`, `simple`, `child`, and `wp`.
- On expansion, run the `capability_logger` after checkpointing the best model.
- For long runs, persist throughput and memory stats through `TrainingLogger`.

Alerting & automatic responses
------------------------------
- If `forgetting > 0.05` on anchor, increase `replay_frac` when a usable replay
  pool is loaded. Caps and scale are stage-configurable (`replay_cap`,
  `replay_scale`).
- If `kv_div > 0.4`, add scheduler patience and hold curriculum expansion for
  one to three eval intervals.
- If global grad norm spikes or deep-layer CV exceeds `0.15`, hold curriculum
  expansion and consider slowing learning rates.
- If plateau is detected, end the stage and checkpoint.

Adaptive replay policy
----------------------
- `CurriculumStageDataset.build(replay_sources=...)` loads cached replay chunks.
- Longer cached chunks can be truncated to the target sequence length.
- Shorter cached chunks are skipped so batch collation and loss masking remain
  valid.
- Replay is disabled and logged as `0.0` if no usable replay chunks load.
- Mapping used by training loops:

```python
new_replay = min(replay_cap, max(min_replay, forgetting_ema * replay_scale))
```

Expansion stages currently configure `replay_scale: 3.0` and caps around
`0.35-0.40`.

Suggested additional strategies
-------------------------------
- Elastic Weight Consolidation (EWC): penalize updates on parameters important
  to previous tasks.
- Memory-based replay / rehearsal: maintain exemplars of old tasks.
- Gradient Episodic Memory (GEM / A-GEM): constrain gradients to avoid
  increasing loss on replay samples.
- Knowledge distillation: keep a frozen teacher and add KD loss on an anchor set.
- Synaptic Intelligence (SI): online parameter-importance tracking.

Practical tests to run
----------------------
1. Cross-domain track:
   `python src/capability_logger.py --checkpoint <ckpt> --tokenizer tokenizers/tokenizer_corpus.json`
2. Per-tier consistency:
   `python evaluate_curriculum.py --checkpoint <ckpt> --tokenizer tokenizers/tokenizer_corpus.json`
3. Replay stress test: run short fine-tuning with replay enabled and vary
   `initial_replay_fraction`.
4. Ablation comparison: compare with and without replay/SI/distillation.
5. Statistical significance: run paired tests on per-sample losses between two
   checkpoints before making strong claims.

Logging & artifacts
-------------------
- Per-eval CSV: `logs/stage{stage}_{host}_{pid}_{timestamp}.csv` with stable
  columns for all validation domains, perplexities, replay/forgetting metrics,
  tier metrics, gradient stability, throughput, wall-clock time, and GPU memory.
- Metadata sidecar: matching `logs/stage{stage}_{host}_{pid}_{timestamp}.meta.json`
  containing the run config and CSV schema.
- Checkpoints embed `anchor_val`, `curriculum_state`, and where available
  `forgetting_ema`.
- Human-readable stage report: `docs/curriculum_capabilities.md` appended by
  the capability logger.
- Aggregate benchmark report:
  `python scripts/summarize_benchmarks.py` writes `docs/benchmark_summary.md`.
- Detailed logging contract: see `docs/logging_and_benchmarking.md`.

How to configure
----------------
Add to `configs/stage0_full.yaml` or `configs/expansion_stages.yaml` per stage:

```yaml
replay_sources:
  - tinystories
initial_replay_fraction: 0.15
forgetting_ema_alpha: 0.1
replay_cap: 0.40
replay_scale: 3.0
min_replay: 0.0
```

Next steps
----------
- Add optional EWC or distillation if replay/SI are insufficient.
- Add a small no-training smoke test for the logging schema and summarizer.
