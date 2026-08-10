#!/bin/bash
# SAMI-Audio — Phase 3a Auto-Launcher
# Submits train_denoiser_2d.slurm in a loop until 80 epochs complete.
# Run in tmux:  tmux new -s phase3a "bash scripts/run_phase3a.sh"
set -euo pipefail

CKPT_DIR="checkpoints/nsynth/denoiser_2d"
EPOCHS=80
SLEEP_SEC=30

mkdir -p "$CKPT_DIR" logs

check_done() {
    local latest
    latest=$(ls -t "${CKPT_DIR}"/denoiser_epoch_*.pt 2>/dev/null | head -1 || true)
    if [ -z "$latest" ]; then
        echo "never_started"
        return
    fi
    local cur
    cur=$(basename "$latest" .pt | grep -oP '\d+$' || echo 0)
    cur=$((10#$cur))
    if [ "$cur" -ge "$EPOCHS" ]; then
        echo "done"
    else
        echo "$cur"
    fi
}

# Check if already complete
STATUS=$(check_done)
if [ "$STATUS" = "done" ]; then
    echo "Phase 3a already complete ($EPOCHS/$EPOCHS epochs)."
    exit 0
fi

echo "============================================================"
echo "  Phase 3a — Denoiser Pre-training ($EPOCHS epochs target)"
echo "  Checkpoint dir: $CKPT_DIR"
echo "============================================================"

ITER=0
while true; do
    ITER=$((ITER + 1))
    CUR=$(check_done)
    if [ "$CUR" = "done" ]; then
        echo "$(date): Phase 3a COMPLETE ($CUR/$EPOCHS epochs in $ITER jobs)"
        break
    fi

    echo ""
    echo "--- Job $ITER: $(date) ---"
    echo "Progress: epoch $CUR / $EPOCHS"

    JOBID=$(sbatch --parsable scripts/train_denoiser_2d.slurm)
    echo "Submitted job $JOBID"

    # Wait for job to finish
    while squeue -j "$JOBID" &>/dev/null && squeue -j "$JOBID" 2>/dev/null | grep -qP "^[[:space:]]*$JOBID"; do
        sleep "$SLEEP_SEC"
    done

    # Check exit status
    JOB_STATE=$(sacct -j "$JOBID" --format=State --noheader -n 2>/dev/null | head -1 | tr -d ' ')
    if [ "$JOB_STATE" != "COMPLETED" ] && [ "$JOB_STATE" != "RUNNING" ]; then
        echo "WARNING: Job $JOBID ended with state $JOB_STATE. Will retry..."
        sleep 10
    fi

    echo "Job $JOBID finished. Status: $JOB_STATE"
    ls -lt "${CKPT_DIR}"/denoiser_epoch_*.pt 2>/dev/null | head -3

    NEW_CUR=$(check_done)
    if [ "$NEW_CUR" != "done" ] && [ "$NEW_CUR" = "$CUR" ]; then
        echo "WARNING: No progress made. Epoch still at $CUR. Possible issue."
        echo "Waiting 60s before retry..."
        sleep 60
    fi
done

echo ""
echo "============================================================"
echo "  Phase 3a COMPLETE"
echo "  Final checkpoint: $CKPT_DIR/model_final.pt"
echo "============================================================"
