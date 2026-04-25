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
