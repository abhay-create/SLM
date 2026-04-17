import time
import torch
import warnings
warnings.filterwarnings('ignore')
from src.model import SLM, SLMConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = SLMConfig(vocab_size=40000, pos_type='learnable', ctx_len=256)
model = SLM(cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randint(0, 40000, (32, 256)).to(device)
y = torch.randint(0, 40000, (32, 256)).to(device)

if device == "cuda":
    from torch.amp import autocast, GradScaler
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device, enabled=not use_bf16)

    for _ in range(2):
        optimizer.zero_grad()
        with autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        if use_bf16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.synchronize()
else:
    for _ in range(2):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

start = time.time()
n_iters = 10

if device == "cuda":
    for _ in range(n_iters):
        optimizer.zero_grad()
        with autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        if use_bf16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.synchronize()
else:
    for _ in range(n_iters):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

dt = time.time() - start
ms_per_step = (dt / n_iters) * 1000
tokens_per_sec = (n_iters * 32 * 256) / dt

print(f"Device: {device}")
print(f"GPU Benchmark: {ms_per_step:.1f} ms/step, {tokens_per_sec:,.0f} tokens/sec")
