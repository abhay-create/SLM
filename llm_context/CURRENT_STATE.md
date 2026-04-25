# Current State

Last updated: 2026-04-25

## Project Snapshot

This repository is a Small Language Model curriculum and expansion project. It trains a decoder-only Transformer from a TinyStories baseline toward richer narrative domains through staged curriculum training and function-preserving model expansion.

The local workspace appears to be a branch snapshot, not a full Git checkout. A `.git` directory was not visible during inspection, so branch metadata may need to be checked on the remote or in a different clone.

## Active User Goal

Build a durable LLM context and replay-buffer system so future LLMs can:

- access critical conversation history and findings
- retrieve relevant project context without re-reading the entire codebase
- record decisions, findings, and user instructions after each meaningful change
- share a common system prompt and update contract

## Current Training Finding

`pipeline_output.log` shows Stage 2 failed before training began:

- failure happens at `model = SLM(old_cfg).to(device)` in `train_expansion.py`
- this is before dataset loading, optimizer creation, batches, or Stage 2 forward/backward
- the likely immediate cause is unavailable GPU memory, not Stage 2 `batch_size` or `seq_len`

The user also reported SSH access to a shared machine with multiple Linux users. In that setup, all users normally share the same physical GPU memory pool unless isolation such as MIG, containers with device restrictions, or exclusive compute mode is configured.

## Environment Notes

- Reported GPU: NVIDIA GeForce RTX 4060 Ti, 16 GB VRAM.
- `nvidia-smi` showed about 14.2 GB of 16.38 GB used.
- A `python3` process was using about 13.8 GB VRAM.
- `MIG M.` was `N/A`, so there is no MIG partitioning.
- `Compute M.` was `Default`, so the GPU is not exclusive to one process or user.

## Next Recommended Actions

1. Use this context prototype for future LLM sessions.
2. On the SSH machine, run `nvidia-smi` and `ps -o user,pid,cmd -p <PID>` before starting training.
3. Verify the Stage 0 checkpoint config on CPU before changing training hyperparameters.
4. If the GPU is shared, coordinate a free VRAM window or configure job scheduling before rerunning Stage 2.
