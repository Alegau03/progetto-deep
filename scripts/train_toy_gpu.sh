#!/bin/bash
# SAMI-Audio — Phase 1: Toy Model Training (sessione interattiva GPU)
# ============================================================
# Esecuzione:
#   bash scripts/train_toy_gpu.sh
# Oppure in sessione interattiva SLURM:
#   srun --gres=gpu:1 --time=01:00:00 --pty bash
#   bash scripts/train_toy_gpu.sh
# ============================================================

set -euo pipefail

echo "============================================================"
echo "  SAMI-Audio — Phase 1: Toy Model (Interactive GPU)"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

mkdir -p checkpoints/toy

python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')"

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
