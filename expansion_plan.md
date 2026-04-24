# Model Expansion Instruction Guide

## Purpose

This document defines the step-by-step plan for expanding a **~46M parameter decoder-only Transformer** trained on **TinyStories** into a stronger **~100M parameter model** and then continuing training on a more complex dataset such as **WritingPrompts**.

The plan is designed for an LLM/agent that will modify the codebase, update training configuration, prepare curricula, and run experiments.

## Current Starting Point

* Decoder-only Transformer
* `d_model = 512`
* `n_layers = 6`
* `n_heads = 8`
* `head_dim = 64`
* `d_ff = 2048`
* `vocab_size = 40,000`
* `context_length = 256`
* `pos_type = "learnable"` (learnable positional embeddings)
* `RMSNorm`, `SwiGLU`
* Weight tying enabled
* Dropout = 0.0
* TinyStories pretraining already completed and stable
* Best checkpoint: `stagefull_dataset_curriculum_adaptive_best.pt` (val loss 1.537, step 46000)

## High-Level Goal

Grow the model in a controlled way so that:

1. The original TinyStories knowledge is preserved.
2. The expanded model learns harder generation tasks.
3. The curriculum becomes progressively more difficult.
4. The final system is measurable, reproducible, and comparable against a scratch-trained 100M baseline.

## Core Strategy

Use **depthwise expansion first**, then **FFN widening** to reach the target size.

The reason: pure depth cannot reach 100M at d_model=512 without excessive layers (~19 needed). Instead, we use a hybrid approach:

### Verified Parameter Counts

| Config               | Params  | Per Layer |
|---------------------|---------|-----------|
| Current (6L, ff=2048)  | 45.8M   | 4.20M     |
| Stage A (9L, ff=2048)  | 58.4M   | 4.20M     |
| Stage B (12L, ff=2048) | 71.0M   | 4.20M     |
| Stage C (12L, ff=3584) | 99.3M   | 6.55M     |

### Growth Direction

* **Stages A→B**: Increase depth from **6 → 9 → 12 layers** (reuses trained representations)
* **Stage C**: Widen FFN from **2048 → 3584** (adds capacity without changing attention)
* Keep `d_model = 512`, `n_heads = 8`, `head_dim = 64` constant throughout

This is the most defensible expansion path because it reuses trained representations, avoids destabilizing the embedding space, and reaches the target size without excessive depth.

## Step-by-Step Expansion Plan

### Step 1 — Freeze the base model and create a clean checkpoint

* Save the fully trained TinyStories checkpoint.
* Save optimizer state, scheduler state, tokenizer, and all training metadata.
* Create a reproducible evaluation script before any expansion.

### Step 2 — Define the target size and growth stages

Use staged growth instead of one large jump.

Stages:

* **Stage A:** 9 layers, ~58M (pure depth expansion)
* **Stage B:** 12 layers, ~71M (pure depth expansion)
* **Stage C:** 12 layers + d_ff=3584, ~99M (FFN widening)

Do not change too many dimensions at once.

### Step 3 — Expand depth using weight cloning / stacking

For each new layer:

* Copy weights from an existing trained layer.
* Initialize the new layer as a near-duplicate of its source.
* **Mandatory**: Add symmetry-breaking noise (σ=0.01) to all cloned weights. Without this, cloned layers produce identical gradients and provide no benefit.

Stacking rule:

* Duplicate layers from the upper portion of the network (layers closest to the output).
* Insert new layers at the top, extending the forward pass depth.

Example (Stage A):

* Original layers: `L0, L1, L2, L3, L4, L5`
* Cloned sources: `L3, L4, L5`
* Expanded model: `L0, L1, L2, L3, L4, L5, L3', L4', L5'` (9 layers)
* Each cloned layer L_i' = L_i + noise(σ=0.01)

Example (Stage B):

* From 9 layers: `L0..L5, L3', L4', L5'`
* Clone L6(=L3'), L7(=L4'), L8(=L5')
* Expanded: `L0..L5, L3', L4', L5', L3'', L4'', L5''` (12 layers)

### Step 4 — Expand FFN width (Stage C)

When depth expansion is complete (12 layers, ~71M), widen the FFN:

* Increase `d_ff` from 2048 to 3584 across all layers.
* **Function-preserving initialization**:
  * `w_gate`: (512, 2048) → (512, 3584) — Existing weights kept, new columns initialized with small random values (σ=0.02)
  * `w_up`: (512, 2048) → (512, 3584) — Same as w_gate
  * `w_down`: (2048, 512) → (3584, 512) — Existing weights kept, **new rows initialized to ZERO**

The zero-initialized w_down rows ensure that new FFN neurons are "dormant" at initialization — they don't affect the model's output. Gradients will teach them to contribute during training.

### Step 5 — Preserve function at growth time

After every expansion:

* Run a validation pass to confirm the new model still behaves like the old one.
* **Depth expansion**: Output cosine similarity > 0.95 (noise causes small differences)
* **FFN widening**: Output must be **identical** (zero-padded w_down guarantees this)
* The loss should not spike beyond 1.5× the pre-expansion value.
* Generation quality should remain roughly similar before further training.

If the output changes drastically immediately after expansion, the initialization is too aggressive.

### Step 6 — Handle optimizer state for new parameters

After expansion, the optimizer must be reset correctly:

* **New parameters**: Initialize with fresh Adam state (no momentum/variance).
* **Pretrained parameters**: Can optionally preserve optimizer state, but the safest approach is a full optimizer reset with warmup.
* **Differential learning rates**: Use lower LR for pretrained layers, higher LR for new layers.
  * Pretrained params: 50% of base LR
  * New params: 100% of base LR
* Apply a short warmup (500–1000 steps) after each expansion.

### Step 7 — Continue training on easy-to-hard curriculum

Do not immediately train on the hardest data.

Suggested sequence:

1. **Stage A**: Continue on TinyStories with full curriculum (model already learned easy data).
2. **Stage B**: Introduce WritingPrompts easy bucket alongside TinyStories replay.
3. **Stage C**: Add WritingPrompts medium and hard buckets gradually.

The curriculum should expand only when the model stabilizes.

### Step 8 — Build WritingPrompts difficulty buckets

Rank WritingPrompts examples by complexity before training.

Scoring dimensions (same composite as TinyStories scoring):

* **Perplexity** (GPT-2 reference): higher = harder
* **Type-Token Ratio**: higher lexical diversity = harder
* **Token length**: longer stories = harder
* **Coherence bonus**: strong discourse markers reduce effective difficulty

Bucket structure:

* **Easy**: short prompt, short story, low perplexity, one clear event
* **Medium**: moderate prompt length, multi-sentence story, some entity tracking
* **Hard**: long prompt, longer story, multiple events, strong narrative consistency requirement

### Step 9 — Use a dynamic data scheduler

Instead of mixing all data at once, increase complexity gradually.

Recommended scheduler:

* Start with 100% easy examples.
* When validation loss plateaus, add 10–20% medium examples.
* Then gradually increase hard examples.
* Keep a small percentage of easy examples throughout training to avoid forgetting.
* **Always include TinyStories replay** (10-15% of batches) to prevent catastrophic forgetting.

A good practical mix:

* Early phase: 80–90% easy, 10–20% medium
* Mid phase: 50% easy, 30% medium, 20% hard
* Late phase: 20% easy, 30% medium, 50% hard

### Step 10 — Expand context length gradually

Since the model uses **learnable positional embeddings** (not RoPE), context expansion requires extending the positional embedding matrix. This is done via linear interpolation: the existing 256 positions are interpolated to fill the new range, preserving relative position relationships.

Progression (starting from the trained context length):

* 256 tokens (current — no change)
* 384 tokens (Stage B)
* 512 tokens (Stage C)
* 768 tokens (optional, late Stage C)

At each stage:

* Interpolate positional embeddings to the new length.
* Train until validation stabilizes.
* Only then increase context length further.

### Step 11 — Keep the model stable during growth

Monitor these signals before allowing the next expansion stage:

* Validation loss trend
* Gradient norm stability
* Layer-wise update stability (CV < 0.15 for deep layer grads)
* Generation coherence on held-out prompts
* Forgetting on TinyStories-style samples (loss should not increase > 5%)
* KV divergence metric (< 0.4)

Treat instability as a signal to pause growth, not as a reason to force more complexity.

## Parameter Addition Scheduler

Use a staged scheduler instead of a single expansion event.

Recommended logic:

1. Train the base 46M model until it reaches a plateau.  ✓ (Done — val loss 1.537)
2. Expand to Stage A (9 layers, ~58M).
3. Reduce learning rate for a short stabilization period (500 steps warmup).
4. Resume training on TinyStories curriculum.
5. When stable, expand to Stage B (12 layers, ~71M).
6. Introduce WritingPrompts easy bucket.
7. When stable, expand to Stage C (12L + wider FFN, ~99M).
8. Full WritingPrompts curriculum.

Rule: Expand only when validation loss has not improved meaningfully for several checkpoints and the model is still stable.

## Vocabulary Strategy

**Decision: Keep existing 40K tokenizer.**

WritingPrompts uses common English vocabulary. The existing 40K BPE tokenizer is sufficient. Re-tokenizing a combined corpus would:

* Invalidate all 2.1M cached difficulty scores
* Require re-chunking and re-caching all training data
* Risk disrupting learned token representations

If vocabulary fragmentation becomes measurably problematic (monitored via tokens-per-word ratio on WritingPrompts), vocabulary expansion can be considered as a separate experiment.

## Architecture After Expansion

Final form for the expanded model:

* Decoder-only Transformer
* `d_model = 512`
* `n_layers = 12`
* `n_heads = 8`
* `d_ff = 3584`
* `pos_type = "learnable"`
* `vocab_size = 40,000`
* `ctx_len = 512` (or 768 if context expansion succeeds)
* Same RMSNorm / SwiGLU setup
* Same weight tying
* ~99.3M parameters

## Optional Future Experiments

After the Transformer expansion is working, you may test:

* Hybrid local/global attention
* Sliding-window attention for longer contexts
* Mamba-like or SSM-inspired blocks in some layers
* Migration from learnable to RoPE positional embeddings
* Per-layer embeddings or other efficiency-oriented additions

These should be separate experiments, not mixed into the first expansion run.

## Evaluation Plan

Track the following on three test sets (Easy / Medium / Hard) across both TinyStories and WritingPrompts:

Measure:

* Perplexity
* Next-token loss
* Prompt adherence
* Coherence
* Entity consistency
* Length control
* Repetition rate
* **Forgetting**: TinyStories val loss at each expansion stage

Expected behavior:

* Easy performance should remain stable or slightly improve.
* Medium performance should improve after expansion.
* Hard performance should improve the most if the curriculum is working.
* TinyStories performance should not degrade more than 5%.

## Required Ablations

To make the result credible, run at least these comparisons:

1. **Scratch 100M model** trained from zero
2. **Expanded model** grown from the 46M checkpoint
3. **Expanded model without curriculum**
4. **Curriculum model with growth scheduler**

This will show whether the growth strategy actually helps.

## Implementation Checklist

* [x] Save the trained 46M TinyStories checkpoint
* [ ] Freeze the current code and tokenizer version
* [ ] Implement depth expansion (layer cloning + symmetry noise)
* [ ] Implement FFN widening (function-preserving zero-pad)
* [ ] Implement context length extension (pos_emb interpolation)
* [ ] Add expansion stage config files
* [ ] Build WritingPrompts data iterator and difficulty scoring
* [ ] Split WritingPrompts into easy/medium/hard tiers
* [ ] Implement differential learning rates for expansion training
* [ ] Add stability checks for growth and curriculum progression
* [ ] Add evaluation scripts for expansion (forgetting, per-tier, cross-dataset)
* [ ] Run baseline scratch 100M comparison
* [ ] Document all hyperparameters and checkpoint versions

## Suggested Training Workflow

1. ✅ Train and validate the 46M model on TinyStories (val loss 1.537).
2. ✅ Freeze the checkpoint.
3. Expand depth to 9L (~58M) — Stage A.
4. Resume training on TinyStories curriculum.
5. Expand depth to 12L (~71M) — Stage B.
6. Introduce WritingPrompts easy bucket + TinyStories replay.
7. Expand FFN width to d_ff=3584 (~99M) — Stage C.
8. Gradual context expansion: 256 → 384 → 512.
9. Add WritingPrompts medium and hard buckets.
10. Compare against a scratch-trained 100M baseline.

## Final Recommendation

For the first expansion experiment, the safest and most defensible path is:

* **Depth-first layer stacking** with mandatory symmetry-breaking noise
* **FFN widening** with function-preserving zero-pad initialization
* **Differential learning rates** for old vs new parameters
* **Curriculum-based dataset growth** (TinyStories → WritingPrompts easy → medium → hard)
* **Gradual context expansion** via pos_emb interpolation (256 → 384 → 512)
* **Keep existing 40K tokenizer** (defer vocabulary expansion)

This gives you a strong chance of showing that the expanded model learns harder generation tasks better than the original TinyStories-only model, while preserving all knowledge from the base training.
