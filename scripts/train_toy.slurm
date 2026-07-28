#!/bin/bash
#SBATCH --job-name=sami-toy
#SBATCH --output=logs/sami_toy_%j.out
#SBATCH --error=logs/sami_toy_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# SAMI-Audio — Phase 1: Toy Model Training (cluster GPU)
# ============================================================
# Esecuzione:
#   sbatch scripts/train_toy.slurm
#
# Monitoraggio:
#   squeue -u $USER
#   tail -f logs/sami_toy_<jobid>.out
# ============================================================

set -euo pipefail

echo "============================================================"
echo "  SAMI-Audio — Phase 1: Toy Model (SLURM)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Started:    $(date)"
echo "============================================================"

# ---- Setup -------------------------------------------------------------------
mkdir -p logs checkpoints/toy

# Carica moduli se necessario (adatta al tuo cluster)
# module load python/3.12
# module load cuda/12.1

# Verifica GPU
echo ""
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No nvidia-smi (using CPU or non-NVIDIA GPU)"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device count: {torch.cuda.device_count()}')" 2>/dev/null || echo "  PyTorch check failed"

# ---- Training ----------------------------------------------------------------
echo ""
echo "Starting training..."
echo "============================================================"

.venv/bin/python train.py \
    --use-wandb \
    --epochs 40 \
    --batch-size 256 \
    --n-per-class 1000 \
    --beta-start 1e-6 \
    --beta-end 1.0 \
    --beta-warmup 12 \
    --T 300 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints/toy \
    --checkpoint-every 5

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Training finished at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Checkpoint: checkpoints/toy/model_final.pt"
echo "============================================================"

# ---- Valutazione (opzionale, esegue Gate 1 se il training ha successo) --------
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Running evaluation (Gate 1)..."
    echo "============================================================"
    .venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
fi

exit ${EXIT_CODE}
