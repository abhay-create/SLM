#!/bin/bash
set -e

echo "Starting SLM Expansion Pipeline..."

for stage in 2 3 4 5 6; do
    echo "================================================="
    echo "Running Stage $stage"
    echo "================================================="
    python train_expansion.py \
        --stage $stage \
        --config configs/expansion_stages.yaml \
        --tokenizer tokenizers/tokenizer_corpus.json
done

echo "================================================="
echo "Pipeline completed successfully!"
echo "================================================="
