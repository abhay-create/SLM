---
language:
  - en
license: mit
tags:
  - language-model
  - transformer
  - decoder-only
  - tinystories
  - fineweb
model-index:
  - name: SLM-Stage0
    results: []
  - name: SLM-Stage2
    results: []
---

# SLM (Small Language Model)

A lightweight decoder-only transformer trained on high-quality datasets in multiple stages.

## Model Details

- **Architecture**: Decoder-only Transformer
- **Parameters**: ~47-51M (depending on tokenizer and config)
- **Positional Embeddings**: RoPE (Rotary Position Embeddings)
- **Training**: Multi-stage curriculum learning

## Checkpoints

### Stage 0 - TinyStories
- **Checkpoint**: `stage0_best_1st_trial.pt`
- **Dataset**: TinyStories
- **Sequence Length**: 768
- **Training Tokens**: 200M
- **Context Length**: 512 tokens

### Stage 2 - FineWeb-Edu
- **Checkpoint**: `stage2_best.pt`
- **Dataset**: FineWeb-Edu + SimpleWiki (15% replay)
- **Sequence Length**: 512
- **Training Tokens**: 500M
- **Context Length**: 512 tokens

## Files

```
├── stage0_best_1st_trial.pt      # Stage 0 weights
├── stage2_best.pt                 # Stage 2 weights (recommended)
├── tokenizer_50k.json             # 50K vocab tokenizer
├── tokenizer_corpus.json          # Alternative tokenizer
├── config_stage0.yaml             # Stage 0 config
├── config_stage2.yaml             # Stage 2 config
├── model.py                       # Model architecture
├── tokenizer.py                   # Tokenizer wrapper
└── inference.py                   # Inference script
```

## Quick Start

### Installation

```bash
pip install torch transformers tokenizers pyyaml
```

### Loading the Model

```python
import torch
from model import SLM, SLMConfig
from tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer_50k.json")
vocab_size = tokenizer.get_vocab_size()

# Create config
config = SLMConfig(
    vocab_size=vocab_size,
    pos_type="rope",
    ctx_len=512,
)

# Load model
model = SLM(config)
checkpoint = torch.load("stage2_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Generation Example

```python
def generate(model, tokenizer, prompt, max_len=100, device="cuda"):
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, _ = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated = tokenizer.decode(input_ids[0].cpu().tolist())
    return generated

# Use it
prompt = "Once upon a time"
output = generate(model, tokenizer, prompt, max_len=50)
print(output)
```

## Training Details

### Stage 0 - Foundation
- TinyStories dataset for initial learning
- 200M tokens, batch_size=32, seq_len=768
- Learning rate: 3e-4 with warmup

### Stage 1 - (Optional)
- Additional curriculum stage

### Stage 2 - Scaling
- FineWeb-Edu: General knowledge from filtered web
- SimpleWiki: Replay buffer for knowledge retention
- 500M tokens, batch_size=32, seq_len=512
- Lower learning rate (5e-5) for fine-tuning

## Configurations

### Stage 0 Config
```yaml
stage: 0
dataset: tinystories
seq_len: 768
max_tokens: 200_000_000
batch_size: 32
learning_rate: 3e-4
lr_min: 3e-5
```

### Stage 2 Config
```yaml
stage: 2
dataset: fineweb_edu
seq_len: 512
max_tokens: 500_000_000
batch_size: 32
learning_rate: 5e-5
lr_min: 5e-6
replay_ratio: 0.15
replay_from: [tinystories]
```

## Model Architecture

- **Embedding Dim**: 512
- **Layers**: 6
- **Attention Heads**: 8
- **Head Dim**: 64
- **FFN Hidden**: 2048
- **Context Length**: 512
- **Weight Tying**: True (embeddings and output share weights)
- **Dropout**: 0.0
- **Bias**: False

## Tokenizer

Two tokenizers available:
- `tokenizer_50k.json`: 50K vocabulary, recommended for most tasks
- `tokenizer_corpus.json`: Alternative tokenizer

Both use the Hugging Face Tokenizers library (BPE).

## Evaluation Metrics

Training tracked three validation sets across stages:
- Dataset-specific validation loss
- Early stopping on plateau (patience=5-8)
- Loss spike detection for stability

## Citation

If you use this model, please cite:
```bibtex
@misc{slm2026,
  title={Small Language Model - Multi-Stage Training},
  year={2026},
  url={https://huggingface.co/your-username/slm}
}
```

## License

MIT License - See LICENSE file for details.

## Contact & Feedback

For issues, questions, or feedback, please open an issue on the model repository.
