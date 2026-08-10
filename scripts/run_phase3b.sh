#!/bin/bash
# SAMI-Audio — Phase 3b Auto-Launcher (v2 robusto)
# Submits train_sami_encoder.slurm in a loop until EPOCHS complete.
# Logica: attende che NON ci siano job in coda (QOS libero) → submit →
# attende la fine del job → verifica progresso reale (checkpoint) → ripete.
# Run in tmux:  tmux new -s phase3b "bash scripts/run_phase3b.sh"
set -euo pipefail

CKPT_DIR="checkpoints/nsynth/sami_d32"
EPOCHS=50
SLEEP_SEC=30
BETA="${BETA:-1e-5}"

mkdir -p "$CKPT_DIR" logs

DENOISER_CKPT="checkpoints/nsynth/denoiser_2d/model_final.pt"
[ -f "$DENOISER_CKPT" ] || { echo "ERROR: Denoiser not found at $DENOISER_CKPT. Run Phase 3a first."; exit 1; }

check_done() {
    # Conta le epoche COMPLETATE (solo sami_epoch_*.pt — gli step di metà
    # epoca non contano come epoca finita)
    local latest cur
    latest=$(ls -t "${CKPT_DIR}"/sami_epoch_*.pt 2>/dev/null | head -1 || true)
    [ -z "$latest" ] && { echo "0"; return; }
    cur=$(basename "$latest" .pt | grep -oP '\d+$' || echo 0)
    echo $((10#$cur))
}

cleanup_steps() {
    # Tiene solo gli ultimi 3 checkpoint per-step (metà epoca): il resume
    # prende il più recente; i vecchi servono solo come fallback.
    ls -t "${CKPT_DIR}"/sami_step_*.pt 2>/dev/null | tail -n +4 | xargs -r rm -f
}

wait_qos_free() {
    # Attende che non ci siano job NOSTRI in coda/esecuzione (QOS libero).
    # Loop INFINITO: non muore mai per QOS, aspetta quanto serve.
    while true; do
        local mine
        mine=$(squeue -u "$USER" -h -t RUNNING,PENDING 2>/dev/null | wc -l)
        if [ "$mine" -eq 0 ]; then
            return 0
        fi
        sleep "$SLEEP_SEC"
    done
}

CUR=$(check_done)
if [ "$CUR" -ge "$EPOCHS" ]; then
    echo "Phase 3b already complete ($CUR/$EPOCHS epochs)."
    exit 0
fi

echo "============================================================"
echo "  Phase 3b — SAMI Encoder ($EPOCHS epochs, β=$BETA)"
echo "  Checkpoint dir: $CKPT_DIR"
echo "============================================================"

ITER=0
while true; do
    CUR=$(check_done)
    if [ "$CUR" -ge "$EPOCHS" ]; then
        echo "$(date): Phase 3b COMPLETE ($CUR/$EPOCHS epochs in $ITER jobs)"
        break
    fi

    ITER=$((ITER + 1))
    echo ""
    echo "--- Job $ITER: $(date) ---"
    echo "Progress: epoch $CUR / $EPOCHS, β=$BETA"

    # Attendi QOS libero, poi submitta (senza limite di tentativi)
    wait_qos_free
    echo "  QOS libero, submitto..."
    JOBID=$(BETA="$BETA" sbatch --parsable scripts/train_sami_encoder.slurm)
    echo "  Submitted job $JOBID"

    # Attendi la fine del job
    while squeue -j "$JOBID" &>/dev/null && squeue -j "$JOBID" 2>/dev/null | grep -qP "^[[:space:]]*$JOBID"; do
        sleep "$SLEEP_SEC"
    done

    JOB_STATE=$(sacct -j "$JOBID" --format=State --noheader -n 2>/dev/null | head -1 | tr -d ' ')
    echo "  Job $JOBID finished. Status: $JOB_STATE"
    cleanup_steps

    # Verifica progresso reale
    NEW_CUR=$(check_done)
    if [ "$NEW_CUR" -le "$CUR" ]; then
        echo "  WARNING: nessun progresso (epoca $CUR). Controllo il log del job..."
        tail -5 "logs/sami_enc_${JOBID}.out" 2>/dev/null || true
        sleep 30
    fi
done

echo ""
echo "============================================================"
echo "  Phase 3b COMPLETE"
echo "  Checkpoint: $CKPT_DIR/model_final.pt"
echo "============================================================"
