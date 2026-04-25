---
id: training-pipeline
title: Training Pipeline
type: workflow
status: active
priority: high
tags: [training, pipeline, stage, checkpoint, evaluation, replay]
updated: 2026-04-25
summary: Stage runs load config, tokenizer, source checkpoint, apply expansion, build optimizer and curriculum data, then train and evaluate.
---

# Training Pipeline

The expansion pipeline is launched by `run_pipeline.sh`, which runs stages 2 through 6 by invoking `train_expansion.py`.

High-level flow inside `train_expansion.py`:

1. Parse CLI args.
2. Load `configs/expansion_stages.yaml`.
3. Select `stage_<N>`.
4. Load tokenizer.
5. Load source checkpoint on CPU.
6. Instantiate `SLM(old_cfg)` and move it to device.
7. Load model weights.
8. Apply configured expansion operations.
9. Build differential-LR optimizer.
10. Build curriculum dataset and validation loaders.
11. Compute anchor validation for forgetting.
12. Train with mixed precision, periodic evaluation, replay adjustment, checkpointing, and capability logging.

Important current diagnosis:

- The captured Stage 2 failure happens at step 6, before optimizer, dataset, or training batches.

When to read:

- debugging pipeline failures
- changing stage orchestration
- changing checkpoint loading or evaluation behavior

Source pointers:

- `run_pipeline.sh`
- `train_expansion.py`
- `docs/training_flow.md`
- `src/curriculum_dataset.py`
- `src/capability_logger.py`
