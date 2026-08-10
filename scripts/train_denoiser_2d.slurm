#!/bin/bash
#SBATCH --job-name=sami-deno
#SBATCH --output=logs/sami_deno_%j.out
#SBATCH --partition=students --gpus=1 --time=00:29:00 --mem=32G --cpus-per-task=8

# SAMI-Audio — Phase 3a: Pre-training MelUNet denoiser on NSynth
# ================================================================
# Auto-resume: checkpoint ogni epoca (usando CachedMelDataset, ~2-3 min/epoca).

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SING="singularity exec --nv pytorch_2.6.0_cuda12.6.sif"
CKPT_DIR="checkpoints/nsynth/denoiser_2d"
EPOCHS=80

mkdir -p "$CKPT_DIR" logs

# Verifica dataset
$SING python -c "from data.nsynth import NSynthDataset; print(f'NSynth: {len(NSynthDataset())} samples')" || exit 1

# Auto-resume
LATEST=$(ls -t "${CKPT_DIR}"/denoiser_epoch_*.pt 2>/dev/null | head -1 || true)
RESUME=""
[ -n "$LATEST" ] && RESUME="--resume $LATEST"
CUR=$(basename "$LATEST" .pt 2>/dev/null | grep -oP '\d+$' || echo 0)
# Force decimal: leading zeros would be interpreted as octal
CUR=$((10#$CUR))
[ "$CUR" -ge "$EPOCHS" ] && { echo "Phase 3a complete ($CUR/$EPOCHS epochs)"; exit 0; }

echo "=== Phase 3a: Denoiser Pre-training (ep $CUR/$EPOCHS) ==="
$SING pip install --quiet --user wandb soundfile 2>/dev/null || true

$SING python train.py --mode sami-nsynth --pretrain-denoiser-2d \
    --use-wandb --epochs $EPOCHS --batch-size 32 --lr 1e-4 --T 1000 \
    --denoiser-channels 128 --checkpoint-dir "$CKPT_DIR" --checkpoint-every 1 \
    $RESUME
