#!/usr/bin/env bash
# ============================================================
# run.sh — Launch WritingPrompts SLM training with nohup
# Usage: bash run.sh [--mode curriculum|direct] [--resume PATH]
#
# Output goes to: train.log
# Monitor: tail -f train.log
# Stop:    kill $(cat train.pid)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="train.log"
PID_FILE="train.pid"

# Parse arguments
MODE="curriculum"
RESUME_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --resume) RESUME_ARG="--resume $2"; shift 2 ;;
        --help) echo "Usage: bash run.sh [--mode curriculum|direct] [--resume PATH]"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  WritingPrompts SLM Training Launcher"
echo "============================================================"
echo "  Mode:     $MODE"
echo "  Log:      $LOG_FILE"
echo "  PID file: $PID_FILE"
echo "  Resume:   ${RESUME_ARG:-none}"
echo "============================================================"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠ Training already running (PID $OLD_PID)"
        echo "  To stop: kill $OLD_PID"
        echo "  To force restart: rm $PID_FILE && bash run.sh"
        exit 1
    else
        echo "  Stale PID file found, cleaning up..."
        rm -f "$PID_FILE"
    fi
fi

# Step 1: Prepare data (if not already done)
echo ""
echo "Step 1: Preparing data..."
if python3 prepare_data.py 2>&1 | tee -a "$LOG_FILE"; then
    echo "  Data ready."
else
    echo "  Data prep failed, trying synthetic fallback..."
    python3 prepare_data.py --synthetic 2>&1 | tee -a "$LOG_FILE"
fi

# Step 2: Run smoke test first
echo ""
echo "Step 2: Running pre-training smoke test..."
if python3 train.py --test --mode "$MODE" 2>&1 | tee -a "$LOG_FILE"; then
    echo "  ✅ Smoke test passed!"
else
    echo "  ❌ Smoke test FAILED. Aborting."
    exit 1
fi

# Step 3: Launch training with nohup
echo ""
echo "Step 3: Launching training in background..."
echo "  Command: python3 train.py --mode $MODE $RESUME_ARG"
echo ""

# Back up old log if it exists
if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    BACKUP="train_backup_$(date +%Y%m%d_%H%M%S).log"
    cp "$LOG_FILE" "$BACKUP"
    echo "  Old log backed up to: $BACKUP"
fi

# Launch with nohup
nohup python3 -u train.py \
    --mode "$MODE" \
    $RESUME_ARG \
    >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

# Verify it started
sleep 3
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "  ✅ Training started successfully!"
    echo "  PID: $TRAIN_PID (saved to $PID_FILE)"
    echo ""
    echo "  Monitor progress:"
    echo "    tail -f $LOG_FILE"
    echo ""
    echo "  Stop training:"
    echo "    kill $TRAIN_PID"
    echo "    # or: kill \$(cat $PID_FILE)"
    echo ""
    echo "  Checkpoints saved to: checkpoints/$MODE/"
    echo "============================================================"
else
    echo "  ❌ Process died immediately! Check $LOG_FILE for errors."
    rm -f "$PID_FILE"
    exit 1
fi
