Benchmarking & Monitoring Plan

Purpose
-------
Concrete, actionable tests and logging to detect capability gains and catastrophic forgetting during curriculum and expansion training.

Core principles
---------------
- Measure per-domain validation loss & perplexity frequently (every `eval_interval`).
- Maintain an "anchor" baseline (TinyStories `s0`) saved with checkpoints; measure relative forgetting vs anchor.
- Track per-tier performance (easy/medium/hard) and deep-layer stability signals (grad norms, KV divergence).
- Use adaptive replay (replay pool sampling) and simple rules to increase replay when forgetting is detected.
- Persist all metrics to CSV + keep human-friendly stage reports in `docs/curriculum_capabilities.md`.

Definitions (metrics)
---------------------
- Val loss: cross-entropy loss on held-out validation sets.
- Perplexity (PPL): `exp(val_loss)` (cap at a large value for overflow).
- Relative forgetting (anchor): `forgetting = (val_s0_current - anchor_val) / anchor_val`.
- Tier losses: losses computed on easy/medium/hard chunks via `evaluate_by_tier()`.
- Backward Transfer (BWT): average change on earlier domains after learning new domain. For LM use loss differences.
- Global grad norm: mean norm across all parameter gradients after backward pass.
- KV divergence: `kv_divergence_metric()` already implemented; used as an internal instability signal.

When to evaluate
-----------------
- Every `eval_interval` training steps compute per-domain validation metrics.
- On expansion, run the `capability_logger` after checkpointing the best model.
- For long runs, persist hourly throughput and memory stats.

Alerting & automatic responses
------------------------------
- If `forgetting > 0.05` (5%) on anchor → increase `replay_frac` (bounded, default ≤0.3) and add patience to scheduler.
- If `kv_div > 0.4` → add scheduler patience and hold curriculum expansion for 1-3 eval intervals.
- If global grad norm spikes or deep-layer CV > 0.15 → hold curriculum expansion and slow learning rates.
- If plateau detected (`PlateauDetector`) → end stage and checkpoint.

Adaptive replay policy (implemented)
-------------------------------------
- Keep a replay pool loaded from other cached train chunks (see `CurriculumStageDataset.build(replay_sources=...)`).
- Policy (simple mapping implemented):
  - `new_replay = min(0.3, max(0.0, forgetting * 2.0))`
  - This maps 5% forgetting → 0.10 replay fraction, 15% → 0.30 (cap).
- Use EMA smoothing for the observed forgetting signal before applying the mapping in future iterations (recommended).

Suggested additional strategies (literature)
-------------------------------------------
- Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017: penalize updates on parameters important to previous tasks.
- Memory-based replay (rehearsal) / iCaRL — Rebuffi et al., 2017: maintain exemplars of old tasks.
- Gradient Episodic Memory (GEM / A-GEM) — Lopez-Paz & Ranzato, Chaudhry et al.: constrain gradients to avoid increasing loss on replay samples.
- Knowledge distillation: keep a frozen teacher (old model) and add KD loss to preserve behaviour on anchor set.
- Synaptic Intelligence (SI) — Zenke et al., 2017: online parameter importance tracking.

Practical tests to run (automated)
---------------------------------
1. Cross-Domain Track: run `python src/capability_logger.py --checkpoint <ckpt> --tokenizer tokenizers/tokenizer_corpus.json`
   - Produces `docs/curriculum_capabilities.md` appendix with losses, PPL, sample generations.

2. Per-Tier Consistency: run `python evaluate_curriculum.py --checkpoint <ckpt> --tokenizer ...` to compute per-tier losses and compare vs baseline.

3. Replay Stress Test: run short fine-tuning on a new domain with replay enabled and vary `initial_replay_fraction` to measure tradeoff between adaptation speed and forgetting.

4. Ablation Comparison: compare with and without replay/EWC/distillation using `evaluate_curriculum.py --baseline <other.ckpt>`; compute % deltas per tier.

5. Statistical significance: use paired t-test / Wilcoxon on per-sample losses between two checkpoints to validate improvements.

Logging & artifacts
------------------
- Per-step CSV: `logs/stage{stage}_{host}_{pid}_{timestamp}.csv` including `replay_frac`, `ts_forgetting`, `grad_norm`, `kv_div`, tier metrics.
- Checkpoints embed `anchor_val` and `curriculum_state` for resuming reproducibly.
- Human-readable stage report: `docs/curriculum_capabilities.md` appended by the capability logger.
- Visualizations: produce PPL/ts_forgetting/replay_frac curves from CSV using `scripts/plot_stage2_metrics.py`.

How to configure (example YAML)
-------------------------------
Add to `configs/stage0_full.yaml` or `configs/expansion_stages.yaml` per-stage:

```
replay_sources:
  - tinystories
  - simple_stories_cache.pkl
initial_replay_fraction: 0.0
```

Next steps
----------
- If you want, I can: (a) add an automated plotting script for these new columns, (b) add an optional EWC regularizer implementation, or (c) wire metrics to TensorBoard/W&B.


