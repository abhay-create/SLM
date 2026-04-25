#!/bin/bash
# ─── SLM Expansion Pipeline (nohup-friendly) ─────────────────────────────────
# Runs stages 2→6 sequentially, logging each stage's terminal output
# to Logs/pipeline_stage_<N>_<timestamp>.log and a combined
# Logs/pipeline_full_<timestamp>.log
#
# Usage:
#   nohup bash run_pipeline_nohup.sh > /dev/null 2>&1 &
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="Logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMBINED_LOG="$LOG_DIR/pipeline_full_${TIMESTAMP}.log"

log_msg() {
    echo "$1" | tee -a "$COMBINED_LOG"
}

log_msg "========================================="
log_msg "SLM Expansion Pipeline — started $(date)"
log_msg "GPU status at start:"
nvidia-smi 2>&1 | tee -a "$COMBINED_LOG" || true
log_msg "========================================="

FAILED=0

for stage in 2 3 4 5 6; do
    STAGE_LOG="$LOG_DIR/pipeline_stage_${stage}_${TIMESTAMP}.log"

    log_msg ""
    log_msg "================================================="
    log_msg "Starting Stage $stage at $(date)"
    log_msg "================================================="

    # Run training; output goes to both per-stage log and combined log
    set +e
    python train_expansion.py \
        --stage "$stage" \
        --config configs/expansion_stages.yaml \
        --tokenizer tokenizers/tokenizer_corpus.json \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$STAGE_LOG" | tee -a "$COMBINED_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        log_msg ""
        log_msg "Stage $stage completed successfully at $(date)"
    else
        log_msg ""
        log_msg "Stage $stage FAILED (exit code $EXIT_CODE) at $(date)"
        FAILED=1
        break
    fi
done

log_msg ""
log_msg "========================================="
if [ $FAILED -eq 0 ]; then
    log_msg "Pipeline completed successfully at $(date)"
else
    log_msg "Pipeline FAILED at $(date)"
fi
log_msg "========================================="

# Run benchmark summary if pipeline succeeded
if [ $FAILED -eq 0 ]; then
    log_msg "Running benchmark summary..."
    python scripts/summarize_benchmarks.py 2>&1 | tee -a "$COMBINED_LOG" || true
fi
