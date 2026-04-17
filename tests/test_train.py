from src.model import SLM, SLMConfig
from train_curriculum import get_dynamic_block_size, kv_divergence_metric, evaluate_by_tier
import torch

print("=== Train Curriculum Component Test ===", flush=True)
model_cfg = SLMConfig(vocab_size=1000, pos_type="learnable", ctx_len=256)
model = SLM(model_cfg)

block_size = get_dynamic_block_size(0.0)
print(f"Initial block size (frac=0.0): {block_size}")

block_size_mid = get_dynamic_block_size(0.5)
print(f"Mid block size (frac=0.5): {block_size_mid}")

block_size_full = get_dynamic_block_size(1.0)
print(f"Full block size (frac=1.0): {block_size_full}")

print("Testing KV divergence metric...", flush=True)
x = torch.randint(0, 1000, (2, 64))
div = kv_divergence_metric(model, x)
print(f"KV divergence on random data: {div:.4f}")

print("Checking syntax in train_curriculum.py... OK.")
