# SAMI-Audio

Disentanglement non supervisionato di pitch e timbro tramite Score-based Variational Autoencoder.

Deep Learning & Applied AI 2025/26 — Progetto.

---

## Roadmap

### ✅ Phase 0 — Setup & Dataset (completata)

| Task | File | Stato |
|------|------|-------|
| `pdm init`, `pyproject.toml` con dipendenze | `pyproject.toml`, `requirements.txt` | ✅ |
| `CONVENTIONS.md` (Nodo 1-5 fissati) | `CONVENTIONS.md` | ✅ |
| Script download NSynth subset | `data/download.sh` | ✅ |
| Dataset class + preprocessing mel | `data/nsynth.py` | ✅ |
| wandb login | — | ✅ |

**Gate 0:** `pdm run gate0` — verifica shape `(1, 128, 256)` in `[-1, 1]` e audio roundtrip via Griffin-Lim.

---

### 🔲 Phase 1 — Toy Model (giorni 2-5) ⬅ PROSSIMA FASE

Obiettivo: validare il meccanismo di guidance SAMI su sinusoidi sintetiche.

| Task | File |
|------|------|
| Half-UNet minimale → `(μ, σ²)`, latente 2D | `models/encoder.py` |
| Cosine schedule, `q_sample`, `ᾱ_t` monotona decrescente | `models/losses/diffusion.py` |
| Core SAMI: encode pulito + rumoroso, Mahalanobis, `g_t = autograd.grad(log_q, xt)`, loss `L_x + β·L_z` | `models/sami.py` |
| `L_KL` su `z ~ N(0,I)` | `models/losses/disentanglement.py` |
| Training loop toy (~5k step, minuti su GPU) | `train.py` |
| Scatter plot 2D colorato per frequenza/ampiezza, MIG | `evaluate.py` |

**Gate 1 (go/no-go):** Scatter 2D con cluster separati per frequenza su un asse e ampiezza sull'altro. MIG > 0.5.

---

### 🔲 Phase 2 — Baseline β-VAE (giorni 5-9)

Obiettivo: lower bound di disentanglement su NSynth.

| Task | File |
|------|------|
| β-VAE encoder/decoder CNN | `models/baselines.py` |
| Train/eval su NSynth con β ∈ {1, 2, 4, 8} | `train.py`, `evaluate.py` |
| Calcolo TAD, DCI, MIG, FactorVAE, FID, SI-SNR, log-mel L1 | `evaluate.py` |
| (Opzionale) FactorVAE baseline | `models/baselines.py` |

**Gate 2:** MIG ~0.10-0.20 con ricostruzioni sensate.

---

### 🔲 Phase 3 — SAMI-Audio Base (giorni 9-16)

Obiettivo: modello principale, verifica della claim centrale.

**3a — Denoiser incondizionato:**
| Task | File |
|------|------|
| U-Net standard, time embedding sinusoidale, no attention (base) | `models/unet.py` |
| Pre-training denoiser su mel NSynth | `train.py` |
| DDIM sampling → mel plausibili | `models/sami.py` |

**3b — Inference network + guidance:**
| Task | File |
|------|------|
| Freeze denoiser, train solo inference network (loss SAMI) | `train.py` |
| KL annealing esponenziale, binary search β massimo | `train.py` |
| Ablation D_latent ∈ {32, 64, 128} | config YAML |
| Logging wandb: `L_z`, sample audio, varianza posterior | — |

**Gate 3:** TAD/MIG > β-VAE a parità di FID/SI-SNR.

---

### 🔲 Phase 4 — Demo e Timbre Map (giorni 16-22)

| Task | File |
|------|------|
| Identificazione assi latenti (CREPE per pitch, classifier per timbro) | `evaluate.py` |
| Timbre transfer: 4 strum. × 4 strum. heatmap | `notebooks/audio_demo.ipynb` |
| UMAP timbre map, cluster per famiglia | `notebooks/timbre_map.ipynb` |
| Sintesi audio via HiFi-GAN | notebooks |

**Gate 4:** Heatmap transfer coerente (timbro cambia, pitch resta). Timbre map con cluster distinti.

---

### 🔲 Phase 5 — Analisi Scale, Ablation, Report (giorni 22-28)

| Task |
|------|
| Analisi scale di rumore caratteristiche per pitch/timbro/dinamica |
| Esperimento loss ausiliarie: SAMI + CLUB, SAMI + FactorVAE |
| Tabella comparativa finale: β-VAE vs FactorVAE vs SAMI |
| Draft report + AI Use Statement |

---

## Quick Start

```bash
# Install dependencies
pdm install

# Download NSynth subset (~21 GB download, ~4 GB after filtering)
bash data/download.sh

# Gate 0 — verify dataset
pdm run gate0
```

## Priorità

```
Toy model → baseline β-VAE → SAMI base run → eval + demo
```

---

## Convenzioni

Tutte le decisioni immutabili (sample rate, STFT params, β strategy, metrics, vocoder) sono documentate in [`CONVENTIONS.md`](CONVENTIONS.md). **Non modificarle** senza discutere l'impatto sull'intera pipeline.
