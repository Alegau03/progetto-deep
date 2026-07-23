# SAMI-Audio — Fixed Conventions

> **IMPORTANTE**: Questo documento fissa tutte le decisioni immutabili del progetto.
> Non modificare questi valori senza aver prima discusso l'impatto sull'intera pipeline.
> Ogni modifica richiede ri-addestramento e ri-valutazione.

---

## Nodo 1 — Audio Processing & Sample Rate

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| Sample rate | **16 kHz** | Nativo NSynth, nessun resampling necessario |
| `n_fft` | **1024** | Coerente con sample rate; divisibile per 8 (3 livelli U-Net) |
| `hop_length` | **256** | 1/4 della finestra FFT |
| `n_mels` | **128** | Divisibile per 8 per compatibilità U-Net |
| `f_min` | **0 Hz** | Copre tutta la gamma fino a 8 kHz |
| `f_max` | **8000 Hz** | Nyquist a 16 kHz |
| Shape mel | **(1, 128, 256)** | [canali, mel_bins, time_frames] |
| Value range | **[-1, 1]** | Log-mel normalizzato per-sample |
| Time crop/pad | **256 frames** | ~4 secondi a hop=256 |

**Regola**: Il diffusion model opera direttamente nello spazio mel (non in un latente compresso).
Tempo e frequenza devono essere divisibili per 8 per il denoiser a 3 livelli.
128 e 256 lo sono — non cambiare.

---

## Nodo 2 — KL Weight (β)

| Parametro | Valore |
|-----------|--------|
| Strategia | **Annealing esponenziale** da valore molto basso |
| Range β | **{1, 2, 4, 8}** per ablation study |
| Ricerca | **Binary search del β massimo** appena sotto la soglia di posterior collapse |
| Riferimento | SAMI paper usa KL weight 5e-6 → 1e-3, mai β=4 |

La tesi di SAMI non è "β non conta", ma: "il prior di diffusione elimina il trade-off
rate-distortion, quindi puoi regolarizzare forte senza rovinare la ricostruzione".

---

## Nodo 3 — Training Strategy & Curriculum

| Parametro | Valore |
|-----------|--------|
| Strategia principale | **Pre-addestrare denoiser incondizionato una volta, congelarlo, addestrare solo inference network** |
| Fase `z ~ N(0,I)` | **Eliminata** (insegnerebbe al denoiser a ignorare la guidance) |
| Sovracampionamento | **Livelli di rumore alti** quando il denoiser è congelato |
| Diffusione | T=1000, cosine schedule (Nichol & Dhariwal), `s=0.008` |
| Sampling inference | DDIM, 50-100 step |
| Ottimizzatore | Adam senza weight decay |
| Learning rate | 1e-4 (training congiunto); 3e-5 (solo inference su denoiser congelato) |
| Mixed precision | bf16 |
| Gradient clip | 1.0 |
| EMA | Sul denoiser (se addestrato) |

---

## Nodo 4 — Metrics

| Metrica | Tipo | Note |
|---------|------|------|
| **TAD** (Total AUROC Difference) | **Primaria** | Quanto ogni attributo è catturato da un singolo asse latente |
| DCI-Disentanglement | Secondaria | Complementare a TAD |
| Modularity | Secondaria | Complementare a TAD |
| MIG | Secondaria | Mutual Information Gap |
| FactorVAE metric | Secondaria | Classificatore a maggioranza, robusta e senza iperparametri |
| FID su mel | Qualità | Distanza tra distribuzione reale e generata |
| SI-SNR | Fedeltà | Signal-to-noise ratio |
| log-mel L1 | Fedeltà | Errore di ricostruzione nello spazio mel |
| Pitch accuracy (CREPE) | Demo | Accuratezza pitch per timbre transfer |
| Timbre Transfer Accuracy | Demo | Classificatore di strumento pre-addestrato |

**Importante**: Tutte le metriche di disentanglement sono implementate a mano.
NON usare `disentanglement-lib` (TensorFlow-based e obsoleto).
PESQ è escluso (tarato sul parlato, inappropriato per note strumentali).
Tutte le metriche quantitative sono calcolate nello spazio mel (indipendenti dal vocoder).

---

## Nodo 5 — Vocoder

| Componente | Scelta |
|------------|--------|
| Vocoder principale | **HiFi-GAN pre-addestrato** con mel config coerente (solo per demo) |
| Baseline | **Griffin-Lim** |
| Regola | Il vocoder impatta solo la demo, mai il percorso critico dei risultati |

---

## Architettura — Configurazioni di Partenza

### Inference Network (Half-UNet — solo Down + Mid)
- Base channels: 64
- Channel multiplier: `[1, 1, 1, 1]` (4 livelli)
- Niente bias (Mohan et al. 2022), nonlinearità ReLU/SiLU
- Due teste MLP: μ (D_latent) e σ² (Softplus + quadratura)
- **D_latent = 64** (punto di partenza; ablare: 32, 64, 128)
- **L'encoder è cieco al livello di rumore** — non riceve `t` in input

### Denoiser (U-Net standard)
- Base channels: 128
- Channel multiplier: `[1, 1, 2, 2]`
- Time embedding sinusoidale (stile DDPM)
- Senza self-attention nella versione base
- Con attention al bottleneck nella versione full

---

## Dataset — NSynth Subset

| Parametro | Valore |
|-----------|--------|
| Famiglie | **{guitar, keyboard, string, brass}** |
| Pitch MIDI | **[48, 84]** (C3–B5) |
| Dimensione | ~50k note, ~4 GB dopo filtraggio |
| Label | Pitch, instrument_family, qualities — **solo in evaluation, mai nel training** |

---

## Workflow di Progetto

```
Toy model → baseline β-VAE → un solo run SAMI base solido → eval + demo
```

Priorità se il tempo stringe: le prime 4 fasi fatte bene valgono più di 10 esperimenti tirati via.

---

*Documento redatto in accordo con le specifiche del progetto SAMI-Audio — DLAI 2025/26.*
