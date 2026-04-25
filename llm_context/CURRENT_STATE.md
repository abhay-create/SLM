# Current State

Last updated: 2026-04-25

## Project Snapshot

This repository is a Small Language Model curriculum and expansion project. It
trains a decoder-only Transformer from a TinyStories baseline toward richer
narrative domains through staged curriculum training and model expansion.

The local workspace is on the `suyash` branch, up to date with `origin/suyash`.

## Active User Goal

Run the complete expansion pipeline (stages 2→6) using nohup and store training
terminal output in the `Logs/` folder under appropriate names.

## Current Training Status

The full expansion pipeline was launched via nohup at 2026-04-25 06:54 EDT.

- Script: `run_pipeline_nohup.sh` (stages 2→6 sequentially)
- PID: 2566936 (background nohup process)
- Log files:
  - Combined: `Logs/pipeline_full_20260425_065439.log` (approximate timestamp)
  - Per-stage: `Logs/pipeline_stage_<N>_<timestamp>.log`
  - CSV metrics: `Logs/stageexpansion_<N>_*.csv` + `.meta.json` sidecars
  - nohup wrapper: `Logs/nohup_output.log`
- Stage 2 is actively training (step ~89/32,552, ~4.1 steps/s, ETA ~2h11m)
- Stages 3–6 will run sequentially after Stage 2 completes

### Bug fixed before launch

`expand_model.py` `expand_context_length()` crashed because
`_copy_non_layer_parameters()` tried to copy position embeddings of mismatched
sizes (old 256 → new 384). Fixed by adding `skip_pos_emb=True` parameter so the
interpolation code handles pos_emb separately.

### Available checkpoints

- `checkpoints/TinyStoriesWithCurriculum.pt` — Stage 0 (45.8M params)
- `checkpoints/stage_2_best.pt` — prior Stage 2 run
- `checkpoints/stage_3_best.pt` — prior Stage 3 run
- Stage 4/5/6 checkpoints do not yet exist

## Environment Notes

- GPU: NVIDIA GeForce RTX 4060 Ti, 16 GB VRAM
- At pipeline launch, GPU memory was mostly free (~428 MiB used by X.org)
- CUDA 13.1, Driver 590.48.01
- `torch.cuda.is_available()` confirmed True
- Machine is shared SSH (multiple Linux users share GPU unless isolated)

## Logging/Benchmarking State

`TrainingLogger` emits a stable all-stage CSV schema with validation loss/PPL
for `s0`, `s1`, `s2`, `roc`, `simple`, `child`, and `wp`, target metric
tracking, forgetting/replay signals, curriculum/tier/gradient/throughput/GPU
metrics, and `.meta.json` run metadata sidecars.

`scripts/summarize_benchmarks.py` aggregates CSVs into
`docs/benchmark_summary.md`. The pipeline script runs it automatically on
successful completion.

## Next Recommended Actions

1. Monitor training: `tail -f Logs/nohup_output.log`
2. Check GPU: `nvidia-smi`
3. After pipeline completes, run `python scripts/summarize_benchmarks.py`
4. Review per-stage logs in `Logs/pipeline_stage_<N>_*.log`
5. Verify stage checkpoints in `checkpoints/`
6. Update this context and add findings after training results are available
