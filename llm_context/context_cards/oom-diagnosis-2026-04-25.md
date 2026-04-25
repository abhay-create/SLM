---
id: oom-diagnosis-2026-04-25
title: Stage 2 CUDA OOM Diagnosis
type: finding
status: active
priority: critical
tags: [gpu, oom, ssh, nvidia-smi, stage2, pipeline, checkpoint]
updated: 2026-04-25
summary: Stage 2 OOM happens while moving the source model to CUDA, before training begins; shared GPU usage is the likely immediate cause.
---

# Stage 2 CUDA OOM Diagnosis

`pipeline_output.log` shows Stage 2 fails at:

```text
model = SLM(old_cfg).to(device)
torch.AcceleratorError: CUDA error: out of memory
```

This happens before:

- optimizer creation
- dataset loading
- dataloader creation
- forward pass
- backward pass
- Stage 2 activation memory

Therefore, do not first reduce `batch_size`, `seq_len`, or training tokens. Those settings may matter later, but they are not the reason for this specific crash.

The user reported a shared SSH machine with multiple users. The shown `nvidia-smi` output had:

- one RTX 4060 Ti with about 16 GB VRAM
- about 14.2 GB already used
- a `python3` process using about 13.8 GB
- `MIG M.` as `N/A`
- `Compute M.` as `Default`

Interpretation:

- all users normally share the same physical GPU unless isolation is configured
- another user's heavy process can cause this run to OOM
- `nvidia-smi` process rows show GPU users by PID, not necessarily by Linux username

Useful checks on the SSH machine:

```bash
nvidia-smi
ps -o user,pid,cmd -p <PID>
```

Useful checkpoint sanity check:

```bash
python - <<'PY'
import torch
ckpt = torch.load("checkpoints/TinyStoriesWithCurriculum.pt", map_location="cpu", weights_only=False)
print(ckpt["config"])
PY
```

When to read:

- any task involving OOM, GPU memory, Stage 2, SSH users, or `nvidia-smi`

Source pointers:

- `pipeline_output.log`
- `train_expansion.py`
- `configs/expansion_stages.yaml`
- `system_info.txt`
