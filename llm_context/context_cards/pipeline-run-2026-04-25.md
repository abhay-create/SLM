---
id: pipeline-run-2026-04-25
title: Full Pipeline Run 2026-04-25
type: finding
status: active
priority: high
tags: [training, pipeline, nohup, stages, expansion, logs]
updated: 2026-04-25
summary: Full expansion pipeline (stages 2→6) launched via nohup after fixing expand_context_length pos_emb crash. Logs in Logs/ folder.
---

# Full Pipeline Run 2026-04-25

## What happened

User requested the complete expansion pipeline run via nohup with logs stored
in the `Logs/` folder.

## Bug fixed

`expand_context_length()` in `expand_model.py` crashed because
`_copy_non_layer_parameters()` tried to copy position embeddings of mismatched
shapes (256 → 384). Fixed by adding `skip_pos_emb=True` parameter.

## Pipeline details

- Launched: 2026-04-25 06:54 EDT via `run_pipeline_nohup.sh`
- PID: 2566936
- Stages: 2 → 3 → 4 → 5 → 6 (sequential)
- GPU: RTX 4060 Ti 16 GB, mostly free at launch

## Log locations

- Combined log: `Logs/pipeline_full_<timestamp>.log`
- Per-stage logs: `Logs/pipeline_stage_<N>_<timestamp>.log`
- CSV metrics: `Logs/stageexpansion_<N>_*.csv`
- Metadata: `Logs/stageexpansion_<N>_*.meta.json`
- nohup output: `Logs/nohup_output.log`

## Monitoring

```bash
tail -f Logs/nohup_output.log
nvidia-smi
ps aux | grep train_expansion
```

## When to read

- Checking pipeline status or log locations
- Debugging a pipeline failure mid-run
- Understanding the nohup log structure
