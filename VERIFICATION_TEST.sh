#!/bin/bash
# VERIFICATION TEST: What if we start Stage1 from the GOOD Stage0 checkpoint?

echo "================================================================"
echo "HYPOTHESIS TEST: Starting Stage1 from GOOD Stage0"
echo "================================================================"
echo ""
echo "Current setup:"
echo "  Stage0 best checkpoint:        stage0_best.pt (loss=5.97, BAD text)"
echo "  Stage0 trial checkpoint:       stage0_best _1st_trial.pt (loss=1.68, EXCELLENT text)"
echo "  Stage1 checkpoint:             stage1_best.pt (loss=3.56, poor text)"
echo ""
echo "Question: If we retrain Stage1 starting from the GOOD checkpoint,"
echo "          will it produce better text?"
echo ""
echo "Answer: We can test with partial training (just 1-2 steps)"
echo ""
echo "Test Training Command:"
echo "=================================================="
cat << 'CMD'

cd /home/user20/NLP/slm
conda activate nlp_env

# CUSTOM TEST: Train Stage1 for just 100 steps from GOOD Stage0
python train.py \
    --stage 1 \
    --config configs/stage1.yaml \
    --tokenizer tokenizers/tokenizer_corpus.json \
    --pos_type rope \
    --checkpoint_dir checkpoints/test_stage1_from_good_s0 \
    --log_dir logs/ \
    --cache_dir cache/ \
    --prev_checkpoint "checkpoints/stage0_best _1st_trial.pt" \
    [--max_tokens 2000000]

# This will create a new checkpoint after ~100 steps
# Then test its generation quality

CMD

echo ""
echo "If the new Stage1 (from good S0) generates MUCH better text even"
echo "after just 100 steps, then we've found the root cause:"
echo ""
echo "✗ Problem NOT: Undertraining duration"
echo "✓ Problem IS:  Wrong parent checkpoint"
echo ""
echo "Solution: Retrain Stage1 with:"
echo "  --prev_checkpoint 'checkpoints/stage0_best _1st_trial.pt'"
echo ""
