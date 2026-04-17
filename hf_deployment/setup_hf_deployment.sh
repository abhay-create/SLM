#!/bin/bash
# setup_hf_deployment.sh — Quick setup script to prepare deployment folder
# Run from hf_deployment/ directory: bash setup_hf_deployment.sh

set -e

echo "=========================================="
echo "  SLM Hugging Face Deployment Setup"
echo "=========================================="
echo ""

# Verify we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ ERROR: Run this script from the hf_deployment/ directory"
    exit 1
fi

# Source directory
SOURCE_DIR="/home/user20/NLP/slm"

echo "📋 Copying files from $SOURCE_DIR..."
echo ""

# Copy model files
echo "Copying model files..."
cp "$SOURCE_DIR/model.py" model.py && echo "  ✓ model.py"
cp "$SOURCE_DIR/tokenizer.py" tokenizer.py && echo "  ✓ tokenizer.py"

# Copy tokenizers
echo ""
echo "Copying tokenizers..."
mkdir -p tokenizers
cp "$SOURCE_DIR/tokenizers/tokenizer_50k.json" tokenizer_50k.json && echo "  ✓ tokenizer_50k.json"
cp "$SOURCE_DIR/tokenizers/tokenizer_corpus.json" tokenizer_corpus.json && echo "  ✓ tokenizer_corpus.json"

# Copy configs
echo ""
echo "Copying configs..."
mkdir -p configs
cp "$SOURCE_DIR/configs/stage0.yaml" config_stage0.yaml && echo "  ✓ config_stage0.yaml"
cp "$SOURCE_DIR/configs/stage2.yaml" config_stage2.yaml && echo "  ✓ config_stage2.yaml"

# Copy checkpoints (with proper handling of space in filename)
echo ""
echo "Copying checkpoints..."
mkdir -p checkpoints
cp "$SOURCE_DIR/checkpoints/stage0_best _1st_trial.pt" "stage0_best_1st_trial.pt" 2>/dev/null && echo "  ✓ stage0_best_1st_trial.pt" || echo "  ⚠ stage0_best_1st_trial.pt (file not found or name issue)"
cp "$SOURCE_DIR/checkpoints/stage2_best.pt" stage2_best.pt 2>/dev/null && echo "  ✓ stage2_best.pt" || echo "  ⚠ stage2_best.pt (file not found)"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "  1. Authenticate with Hugging Face:"
echo "     huggingface-cli login"
echo ""
echo "  2. Upload to Hub:"
echo "     python upload_to_hf.py --repo_id YOUR_USERNAME/slm --create"
echo ""
echo "  3. View your model:"
echo "     https://huggingface.co/YOUR_USERNAME/slm"
echo ""
echo "=========================================="
