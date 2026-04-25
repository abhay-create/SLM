---
id: stage-config-map
title: Stage Config Map
type: code-map
status: active
priority: medium
tags: [config, stages, yaml, hyperparameters, checkpoint]
updated: 2026-04-25
summary: Stage 2 through 6 settings live in configs/expansion_stages.yaml and should be checked before changing training behavior.
---

# Stage Config Map

`configs/expansion_stages.yaml` is the source of truth for Stage 2 through Stage 6.

Stage 2:

- source checkpoint: `checkpoints/TinyStoriesWithCurriculum.pt`
- target: 9 layers, `d_ff=2048`, context 384
- expansion: clone layers `[3, 4, 5]`, noise `0.01`, context 384
- dataset mix: ROCStories 0.6, SimpleStories 0.4
- batch size 24, seq len 384
- replay source: TinyStories

Stage 3:

- source checkpoint: `checkpoints/stage_2_best.pt`
- target remains 9 layers, context 384
- dataset mix includes SimpleStories, Children Stories, ROCStories, TinyStories

Stage 4:

- source checkpoint: `checkpoints/stage_3_best.pt`
- target: 12 layers, context 512
- expansion: clone layers `[6, 7, 8]`

Stage 5:

- source checkpoint: `checkpoints/stage_4_best.pt`
- target remains 12 layers, context 512
- WritingPrompts Easy domain alignment

Stage 6:

- source checkpoint: `checkpoints/stage_5_best.pt`
- target: 12 layers, `d_ff=3584`, context 768
- FFN and context expansion

When to read:

- modifying hyperparameters
- diagnosing training duration, memory, or source checkpoint issues
- explaining the roadmap

Source pointers:

- `configs/expansion_stages.yaml`
- `README.md`
