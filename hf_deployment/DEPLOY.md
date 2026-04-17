# SLM Hugging Face Deployment

Quick setup to deploy the Small Language Model to Hugging Face Hub.

## Files Included

- **stage0_best_1st_trial.pt** — Stage 0 checkpoint (TinyStories)
- **stage2_best.pt** — Stage 2 checkpoint (FineWeb-Edu) ⭐ Recommended
- **tokenizer_50k.json** — 50K vocab tokenizer
- **tokenizer_corpus.json** — Alternative tokenizer
- **config_stage0.yaml** — Stage 0 configuration
- **config_stage2.yaml** — Stage 2 configuration
- **model.py** — Model architecture
- **tokenizer.py** — Tokenizer wrapper
- **inference.py** — Inference script
- **requirements.txt** — Dependencies
- **README.md** — Model card

## Setup (5 minutes)

### 1. Authenticate with Hugging Face

```bash
pip install huggingface-hub
huggingface-cli login
```

You'll be prompted for your HF token. Get it from: https://huggingface.co/settings/tokens

### 2. Copy Model Files

If deploying from another location, copy these files into this directory:

```bash
# Copy checkpoints
cp /home/user20/NLP/slm/checkpoints/stage0_best\ _1st_trial.pt ./
cp /home/user20/NLP/slm/checkpoints/stage2_best.pt ./

# Copy tokenizers
cp /home/user20/NLP/slm/tokenizers/tokenizer_*.json ./

# Copy configs
cp /home/user20/NLP/slm/configs/stage{0,2}.yaml ./

# Copy model files
cp /home/user20/NLP/slm/model.py ./
cp /home/user20/NLP/slm/tokenizer.py ./
```

### 3. Upload to Hugging Face

```bash
# Replace YOUR_USERNAME with your actual HF username
python upload_to_hf.py \
    --repo_id YOUR_USERNAME/slm \
    --create
```

This will:
- ✓ Create the repository (if --create flag used)
- ✓ Upload all model files
- ✓ Upload tokenizers
- ✓ Upload configs and scripts
- ✓ Add model card (README.md)

### 4. Done! 🎉

Your model is now on Hugging Face Hub at:
```
https://huggingface.co/YOUR_USERNAME/slm
```

## Usage After Upload

### From Hugging Face

```python
from huggingface_hub import hf_hub_download
import torch
from model import SLM

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="YOUR_USERNAME/slm",
    filename="stage2_best.pt"
)

# Load
ckpt = torch.load(checkpoint_path)
model = SLM(ckpt["config"])
model.load_state_dict(ckpt["model_state"])
```

### Inference Script

```bash
python inference.py \
    --checkpoint stage2_best.pt \
    --tokenizer tokenizer_50k.json \
    --prompt "Once upon a time" \
    --max_len 100
```

## Checkpoint Details

| Checkpoint | Dataset | Seq Len | Tokens | Context |
|-----------|---------|---------|--------|---------|
| stage0_best_1st_trial.pt | TinyStories | 768 | 200M | 512 |
| stage2_best.pt | FineWeb-Edu | 512 | 500M | 512 |

**Recommendation**: Use `stage2_best.pt` for best performance.

## Troubleshooting

### Authentication Error
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login
```

### File Not Found
- Check paths in `upload_to_hf.py` match your actual file locations
- Use absolute paths if in doubt

### Upload Failed
- Check internet connection
- Verify HF token has write access
- Try uploading a single file first to debug

## File Size Reference

- stage0_best_1st_trial.pt: ~195 MB
- stage2_best.pt: ~195 MB
- Tokenizers: ~200-300 KB each
- Total upload: ~400-500 MB

## Next Steps

After upload, you can:
1. **Add to Model Card**: Edit on huggingface.co directly
2. **Update README.md**: Customize model details
3. **Add Tags**: Add performance metrics, benchmarks
4. **Create Spaces**: Add interactive demo with Gradio/Streamlit
5. **Link Papers**: Add citations and related work

## Support

For issues:
- Check HF Hub documentation: https://huggingface.co/docs
- Verify model files are complete and valid
- Test locally first with `inference.py`

---

Happy deploying! 🚀
