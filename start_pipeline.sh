#!/bin/bash
set -e

# Suppress HuggingFace warnings and optimize parallelization
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "=================================="
echo "    SLM TRAINING PIPELINE"
echo "=================================="
echo "$(date): Starting Data Scoring and Curriculum Division..."

# We use the base environment python which has all our dependencies installed
PYTHON_PATH=~/miniconda3/bin/python3

# Run the scoring script
# Note: we use our custom tokenizer to calculate accurate token lengths
$PYTHON_PATH src/score_difficulty.py \
    --output curriculum_scores.npy \
    --tokenizer_path tokenizers/tokenizer_corpus.json \
    --batch_size 32 \
    --max_stories 200000

echo "$(date): Scoring Complete."
echo "=================================="
echo "$(date): Starting Stage 0 Adaptive Training..."

# Start the training
$PYTHON_PATH train_curriculum.py \
    --config configs/stage0_curriculum.yaml \
    --tokenizer tokenizers/tokenizer_corpus.json

echo "$(date): Pipeline execution completely finished!"
