#!/bin/bash
#SBATCH --job-name=sami-enc
#SBATCH --output=logs/sami_enc_%j.out
#SBATCH --partition=students --gpus=1 --time=00:29:00 --mem=32G --cpus-per-task=8

# SAMI-Audio — Phase 3b: Training Finale SAMI Encoder (denoiser frozen)
# ================================================================
# Config definitiva dal report (Sami_train_definitivo.md §3.2) aggiornata
# con l'esito del mini-sweep β a D=32:
#   D=32, β=1e-5 (scelto dallo sweep: margine 0.576 a epoca 1),
#   free-bits 0.5, oversample_t Beta(4,1), AMP sul denoiser, batch 16,
#   50 epoche, early-stop sul margine R²−control, guardia L_z_raw.
# Probe ogni 2 epoche (denso nella fase critica 5-10 dove si vede se
# β=1e-5 regge o erode → piano B: β=5e-5).

set -euo pipefail
SING="singularity exec --nv pytorch_2.6.0_cuda12.6.sif"
CKPT_DIR="checkpoints/nsynth/sami_d32"
EPOCHS=50
BETA="${BETA:-1e-5}"

mkdir -p "$CKPT_DIR" logs

DENOISER_CKPT="checkpoints/nsynth/denoiser_2d/model_final.pt"
[ -f "$DENOISER_CKPT" ] || { echo "ERROR: Denoiser not found at $DENOISER_CKPT. Run Phase 3a first."; exit 1; }

# Auto-resume: prende il checkpoint PIÙ RECENTE (step di metà epoca o fine
# epoca). Senza checkpoint per-step, un job killato a metà epoca (epoca
# ~31 min > limite 29) ripartirebbe dall'inizio — loop infinito.
LATEST=$(ls -t "${CKPT_DIR}"/sami_epoch_*.pt "${CKPT_DIR}"/sami_step_*.pt 2>/dev/null | head -1 || true)
RESUME=""
[ -n "$LATEST" ] && RESUME="--resume $LATEST"
CUR_EP=$(ls -t "${CKPT_DIR}"/sami_epoch_*.pt 2>/dev/null | head -1 || true)
CUR=0
[ -n "$CUR_EP" ] && CUR=$(basename "$CUR_EP" .pt | grep -oP '\d+$' || echo 0)
# Force decimal: leading zeros would be interpreted as octal
CUR=$((10#$CUR))
[ "$CUR" -ge "$EPOCHS" ] && { echo "Phase 3b complete ($CUR/$EPOCHS epochs)"; exit 0; }

echo "=== Phase 3b: SAMI Encoder FINALE (β=$BETA, D=32, ep $CUR/$EPOCHS) ==="
$SING pip install --quiet --user wandb soundfile 2>/dev/null || true

$SING python train.py --mode sami-nsynth \
    --denoiser-checkpoint "$DENOISER_CKPT" \
    --use-wandb --epochs $EPOCHS --batch-size 16 --lr 3e-5 \
    --beta $BETA --T 1000 --latent-dim 32 \
    --oversample-t --amp --free-bits 0.5 \
    --ckpt-every-steps 1500 \
    --probe-every 2 --probe-size 2000 \
    --early-stop-patience 4 --collapse-thresh 0.05 \
    --checkpoint-dir "$CKPT_DIR" --checkpoint-every 1 \
    $RESUME
