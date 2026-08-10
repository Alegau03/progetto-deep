#!/bin/bash
# SAMI-Audio — Generazione figure finali per il report
# Esegue make_report_figures.py in plots/finals/ e verifica gli output.
# Run in tmux:  tmux new -s figures "bash scripts/run_report_figures.sh"
set -euo pipefail

OUT="plots/finals"
LOGFILE="logs/figures_run.log"
mkdir -p "$OUT" logs

echo "============================================================" | tee "$LOGFILE"
echo "  Generazione figure finali del report" | tee -a "$LOGFILE"
echo "  Output: $OUT" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# Usa il venv locale se disponibile, altrimenti python3 di sistema
PY=python3

echo "" | tee -a "$LOGFILE"
echo "--- $(date): eseguo make_report_figures.py ---" | tee -a "$LOGFILE"
$PY scripts/make_report_figures.py "$OUT" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Verifica output ---" | tee -a "$LOGFILE"
N=0
for png in fig1_denoiser_gate fig2_collapse_betas fig3_guidance_by_t \
          fig4_kl_perdim fig5_convergence fig6_transfer_demo fig7_demo_grid; do
    if [ -f "$OUT/$png.png" ]; then
        sz=$(stat -c%s "$OUT/$png.png" 2>/dev/null || echo "?")
        echo "  [OK] $png.png (${sz} bytes)" | tee -a "$LOGFILE"
        N=$((N+1))
    else
        echo "  [MISSING] $png.png" | tee -a "$LOGFILE"
    fi
done

echo "" | tee -a "$LOGFILE"
echo "=== $N/7 figure generate in $OUT ===" | tee -a "$LOGFILE"
echo "=== Completato $(date) ===" | tee -a "$LOGFILE"
