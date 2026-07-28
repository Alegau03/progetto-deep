# SAMI-Audio — Roadmap Dettagliata

> **Legenda:** ✅ Completato | 🔲 Da fare | 🔄 In corso | ⏸️ In pausa

---

## Phase 0 — Setup & Dataset ✅

**Obiettivo:** Ambiente pronto, dati scaricati, convenzioni fissate.
**Stato:** COMPLETATA (2025-07-23)

### Task

- [x] `pdm init` + `pyproject.toml` con tutte le dipendenze
- [x] `requirements.txt` (pip fallback)
- [x] `CONVENTIONS.md` — decisioni immutabili (Nodo 1-5)
- [x] `.gitignore` configurato
- [x] `data/download.sh` — script download NSynth subset (21 GB → ~4 GB dopo filtraggio)
  - Famiglie: guitar, keyboard, string, brass
  - Pitch MIDI: [48, 84] (C3-B5)
  - Filtraggio automatico post-estrazione
- [x] `data/nsynth.py` — Dataset class completa
  - `MelConfig`: config immutabile STFT/mel
  - `NSynthDataset(Dataset)`: load, filter, mel spectrogram, normalize, crop/pad
  - `mel_to_audio()`: Griffin-Lim reconstruction
  - `__main__`: Gate 0 sanity check
- [x] `data/__init__.py`, `models/__init__.py`, `models/losses/__init__.py`
- [x] `pdm install` — virtualenv creato, dipendenze installate
- [x] `wandb login` — API key configurata
- [x] `torchcrepe` al posto di `crepe` (build rotto)
- [x] Struttura directory: `experiments/`, `notebooks/`, `checkpoints/`
- [x] `docs/AGENTS.md`, `docs/roadmap.md`, `docs/CHANGELOG.md`
- [x] `README.md` aggiornato con roadmap completa

### Gate 0
```
pdm run gate0
```
**Criterio:** `dataset[i]` restituisce `(1, 128, 256)` in `[-1, 1]`. Audio ricostruito via Griffin-Lim suona come l'originale.

### Note
- Drop-in per `crepe`: usiamo `torchcrepe` (PyTorch port). L'API è diversa — da adattare in Phase 4.
- NSynth scaricato e filtrato: 61,531 campioni in `data/nsynth-train/`.
- Gate 0 PASSED ✓

---

## Phase 1 — Toy Model ✅ (CODICE PRONTO — eseguire su cluster GPU)

**Obiettivo:** Validare il meccanismo di guidance SAMI su sinusoidi sintetiche.
**Stato:** CODICE COMPLETATO (2025-07-28), pronto per esecuzione su cluster GPU.
**Esecuzione:** `sbatch scripts/train_toy.slurm`
**Dataset:** Sinusoidi generate proceduralmente — 10 frequenze × 5 ampiezze = 50 classi, ~1000 campioni/classe.
**Fattori ground-truth:** frequenza (= pitch), ampiezza (= dinamica).
**Latente:** 2D (per visualizzazione diretta in scatter plot).

### Task

- [x] `models/encoder.py` — Half-UNet minimale (1D)
  - Solo blocchi Down + Mid (niente Up)
  - Base channels 32, 3 livelli di downsampling (stride 2)
  - Due teste MLP: μ (dim=D_latent) e logvar → Softplus + ε → σ²
  - `D_latent = 2` per toy model
  - **Test:** forward pass, shape corrette, σ² > 0

- [x] `models/losses/diffusion.py` — Diffusion utilities
  - Cosine noise schedule (Nichol & Dhariwal), T=1000, s=0.008
  - `q_sample(x0, t, noise)`: forward diffusion step
  - `alphas`, `alphas_cumprod` (ᾱ_t), `betas`
  - `gamma(t)`: √(1 - ᾱ_t) per guidance
  - **Test:** ᾱ_t monotona decrescente, `q_sample` produce rumore crescente con t

- [x] `models/sami.py` — Core SAMI module (architettura-agnostico)
  - `encode(x)`: chiama encoder, produce z via reparameterization
  - `_log_q_and_grad(z, xt)`: Mahalanobis + autograd.grad con create_graph=True
  - `forward(x0)`: encode pulito → z, encode rumoroso → g_t, L_x + β·L_z
  - `sample(z, shape, n_steps)`: DDIM reverse process con guidance
  - **Test isolato:** gradienti propagati, no NaN, loss > 0

- [x] `models/losses/disentanglement.py` — Disentanglement losses
  - `kl_divergence(mu, sigma2)`: KL(N(μ,σ²) || N(0,I))
  - `mahalanobis_log_prob(z, mu, sigma2)`: log q(z|x) scalare per campione
  - **Test:** KL = 0 quando μ=0, σ=1; KL > 0 altrimenti

- [x] `train.py` — Training loop toy
  - `generate_toy_dataset()`: 10 freq × 5 amp × 1000 = 50k sinusoidi
  - Loop congiunto encoder + denoiser, ~50 epoche
  - Logging: loss components, scheduler cosine, checkpoint, wandb opzionale

- [x] `models/unet.py` — Denoiser 1D per toy
  - U-Net 1D con 3 livelli, ResBlock con time conditioning
  - Sinusoidal time embedding, skip connections
  - ~50-100k parametri

- [x] `evaluate.py` — Metriche toy
  - Scatter plot 2D del latente colorato per frequenza e ampiezza
  - MIG (Mutual Information Gap) implementato manualmente
  - Gate 1 go/no-go automatico

- [x] `scripts/train_toy.slurm` — SLURM batch script
  - GPU request, log, checkpoint, valutazione automatica post-training
  - Parametri ottimizzati per GPU: batch=256, n_per_class=1000, T=300

- [x] `scripts/train_toy_gpu.sh` — Script interattivo GPU
  - Per sessioni `srun` o esecuzione diretta su nodo GPU

- [x] KL annealing esponenziale (fix posterior collapse)
  - β: 1e-6 → 1.0 su 12 epoche (curva esponenziale)
  - Previene L_z=0 e garantisce guidance attiva

### Gate 1 (go/no-go dell'intero progetto)
Scatter 2D con:
- Cluster che si separano per **frequenza** lungo un asse
- Cluster che si separano per **ampiezza** lungo l'altro
- MIG > 0.5

**Esecuzione Gate 1:**
```bash
# Sul cluster (automatico dopo sbatch se training ok)
# Oppure locale dopo aver scaricato il checkpoint:
.venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
```

**Se fallisce:** Il bug è nel codice (loss/guidance/sampling), non nell'audio. Si scopre in minuti.
**Se passa:** Pronti per Phase 2 (β-VAE baseline su NSynth).

---

## Phase 2 — Baseline β-VAE su NSynth 🔲

**Obiettivo:** Lower bound del disentanglement su dati reali.
**Durata:** giorni 5-9
**Dataset:** NSynth subset (4 famiglie, pitch 48-84).

### Task

- [ ] `models/baselines.py` — β-VAE
  - Encoder CNN: mel (1,128,256) → latente D_latent
  - Decoder CNN: latente → mel (1,128,256)
  - Loss: reconstruction (MSE/L1) + β · KL

- [ ] `models/baselines.py` — FactorVAE (opzionale)
  - Come β-VAE + Total Correlation penalty via discriminator

- [ ] `train.py` — Adattamento per NSynth
  - DataLoader con NSynthDataset
  - Training loop con Accelerate
  - Hydra config YAML in `experiments/`
  - β ∈ {1, 2, 4, 8}

- [ ] `evaluate.py` — Metriche complete
  - TAD (Total AUROC Difference) — primaria
  - DCI-Disentanglement, Modularity
  - MIG (Mutual Information Gap)
  - FactorVAE metric (classificatore a maggioranza)
  - FID su mel
  - SI-SNR, log-mel L1
  - Pitch accuracy (torchcrepe)
  - Timbre transfer accuracy

### Gate 2
MIG ~0.10-0.20 con ricostruzioni sensate. Pipeline di misura affidabile e riutilizzabile.

---

## Phase 3 — SAMI-Audio Base 🔲

**Obiettivo:** Modello principale. Verifica della claim centrale.
**Durata:** giorni 9-16

### Sotto-fase 3a: Denoiser incondizionato

- [ ] `models/unet.py` — U-Net completa
  - Base channels 128, channel multiplier `[1,1,2,2]`
  - Time embedding sinusoidale (stile DDPM)
  - Senza self-attention (base); con attention al bottleneck (full)
  - Input/output: mel (1, 128, 256)

- [ ] Pre-training denoiser su mel NSynth
  - Loss: semplice diffusion loss (senza encoder)
  - Salva checkpoint

- [ ] Verifica: DDIM sampling produce mel plausibili da rumore puro

### Sotto-fase 3b: Inference network + guidance

- [ ] Freeze denoiser pre-addestrato
- [ ] Addestra solo inference network (Half-UNet) con loss SAMI
- [ ] Sovracampiona livelli di rumore alti
- [ ] KL annealing esponenziale
- [ ] Binary search del β massimo sotto posterior collapse
- [ ] Logging wandb: L_z, sample audio ogni 1000 step
- [ ] Ablation: D_latent ∈ {32, 64, 128}

### Gate 3
TAD/MIG > β-VAE a parità di FID/SI-SNR. Claim principale del progetto.

---

## Phase 4 — Demo e Timbre Map 🔲

**Obiettivo:** Risultati visuali comprensibili per la presentazione.
**Durata:** giorni 16-22

### Task

- [ ] Identificazione assi latenti
  - Correlazione asse latente con pitch (via torchcrepe)
  - Correlazione asse latente con instrument_family (classifier)
  - Individuazione assi "pitch" e "timbro"

- [ ] `notebooks/audio_demo.ipynb` — Timbre transfer
  - 4 strumenti di riferimento (chitarra, piano, violino, tromba)
  - Estrai latenti su stessa nota
  - Scambia assi di timbro
  - Genera 16 combinazioni (4×4)
  - Misura Timbre Transfer Accuracy e Pitch Accuracy
  - Visualizza come heatmap

- [ ] `notebooks/timbre_map.ipynb` — Timbre map
  - Proietta assi di timbro del test set con UMAP
  - Colora per famiglia strumentale
  - Colora per brightness, percussivity
  - Verifica cluster distinti

- [ ] Sintesi audio via HiFi-GAN pre-addestrato
  - Mel config coerente con la nostra

### Gate 4
Heatmap di transfer con diagonale coerente (timbro cambia, pitch resta). Timbre map con cluster per famiglia.

---

## Phase 5 — Analisi Scale, Ablation, Report 🔲

**Obiettivo:** Contributo scientifico + report finale.
**Durata:** giorni 22-28

### Task

- [ ] Analisi scale di rumore caratteristiche
  - Per pitch, timbro, dinamica: a quale t ogni fattore smette di essere informativo
  - Differenza magnitudine score prior vs posterior lungo il denoising
  - Risponde: le note strumentali sono multi-scala o a scala singola?

- [ ] Esperimento loss ausiliarie (se c'è tempo)
  - SAMI + L_MI (CLUB): l'info mutua aggiunge qualcosa?
  - SAMI + L_TC (FactorVAE): la total correlation aggiunge qualcosa?

- [ ] Tabella comparativa finale
  - β-VAE vs FactorVAE vs SAMI-Audio
  - Tutte le metriche: TAD, DCI, MIG, FactorVAE, FID, SI-SNR

- [ ] Draft report
  - Struttura: intro, metodo, esperimenti, risultati, discussione
  - AI Use Statement (obbligatorio)

---

## Phase 6 (Opzionale) — Clip Polifoniche 🔲

**Obiettivo:** SAMI come source separation latente su clip polifoniche URMP.

**⚠️ Tagliabile.** Non sacrificare il report per aggiungerla.

---

## Priorità Finale

Se il tempo stringe, questo è l'ordine di importanza:

```
1. Toy model             (Phase 1) — validazione core
2. β-VAE baseline        (Phase 2) — termine di paragone
3. SAMI base run         (Phase 3) — risultato principale
4. Eval + demo           (Phase 4) — presentazione
5. Scale analysis        (Phase 5) — contributo scientifico
```

Tutto il resto (ablation multiple, CLUB/TC, polifonia) è incrementale.

---

*Ultimo aggiornamento: 2025-07-28 — Phase 1 pronta per cluster GPU, posterior collapse fixato.*
