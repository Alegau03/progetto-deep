# CHANGELOG — SAMI-Audio

Log cronologico delle modifiche al progetto. Ogni entry include data, fase, descrizione e file coinvolti.

---

## 2025-07-28 — Phase 1: Toy Model (completamento e fix)

### Update: posterior collapse fix + cluster GPU setup

**Problema riscontrato:**
- Training con β=1.0 fisso → L_z collassa a 0 all'epoca 2
- Causa: termine KL domina prima che l'encoder impari a codificare informazione
- Effetto: g_t = 0, nessuna guidance, modello degenera in denoiser incondizionato

**Soluzione applicata:**
- KL annealing esponenziale: β: 1e-6 → 1.0 su 12 epoche
- Nuovi parametri CLI: `--beta-start`, `--beta-end`, `--beta-warmup`
- Logging: β corrente stampato a ogni epoca e loggato su wandb

**Ottimizzazioni per cluster GPU:**
- Default adattati: `--epochs 30`, `--batch-size 128`, `--n-per-class 500`, `--T 200`
- Su GPU consigliati: `--epochs 40`, `--batch-size 256`, `--n-per-class 1000`, `--T 300`

**Nuovi file:**
| File | Descrizione |
|------|-------------|
| `scripts/train_toy.slurm` | SLURM batch script: GPU job + training + valutazione automatica |
| `scripts/train_toy_gpu.sh` | Script per sessione interattiva GPU |
| `docs/phases/Phase1.md` | Documentazione completa della fase (10 KB) |

**Documentazione aggiornata:**
| File | Modifica |
|------|----------|
| `docs/AGENTS.md` | Aggiunta sezione cluster GPU, SLURM, trasferimento risultati |
| `docs/roadmap.md` | Phase 1: aggiunti SLURM script, KL annealing, note esecuzione Gate 1 |
| `docs/Phase1.md` | Sezione 0 (setup cluster), sezione 6 (output e workflow), sezione 3 (parametri GPU) |
| `README.md` | (da aggiornare dopo Gate 1) |

### Comandi
```bash
# Cluster GPU
sbatch scripts/train_toy.slurm             # batch job
bash scripts/train_toy_gpu.sh              # interattivo

# Locale
.venv/bin/python train.py --use-wandb       # default ottimizzati

# Valutazione
.venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
```

### Stato Phase 1
- ✅ Codice completato
- ✅ Posterior collapse fixato (KL annealing)
- ✅ SLURM script pronto
- ✅ Documentazione completa
- ⬜ Training su cluster GPU (da eseguire)
- ⬜ Gate 1 (da validare dopo training)

### Branch: `fase-1`

### File creati

| File | Descrizione |
|------|-------------|
| `models/losses/diffusion.py` | `DiffusionSchedule`: cosine schedule (Nichol & Dhariwal), `q_sample`, `gamma()` |
| `models/encoder.py` | `ToyEncoder`: Half-UNet 1D (3 Down + Mid) → (μ, σ²). D_latent=2 per toy. |
| `models/losses/disentanglement.py` | `kl_divergence()` e `mahalanobis_log_prob()` |
| `models/unet.py` | `ToyUNet`: 1D U-Net denoiser con sinusoidal time embedding, skip connections, ResBlock |
| `models/sami.py` | `SAMI`: core module con forward training loop (autodiff guidance) e DDIM sampling |
| `train.py` | `generate_toy_dataset()` (50k sinusoidi), `train_toy()`, CLI argparse + wandb |
| `evaluate.py` | `compute_mig()`, `plot_latent_space()`, Gate 1 check, CLI |

### Test di validazione
- Tutti gli import OK
- Forward pass encoder: σ² > 0 ✓
- Diffusion: q_sample e gamma ✓
- KL(0,I) = 0, KL(non-zero) > 0 ✓
- UNet: output shape = input shape ✓
- SAMI forward: loss calcolata, gradienti propagati, no NaN ✓
- Dataset: 50k segnali (1, 256), 10 freq × 5 amp, range ~[-1, 1] ✓

### Comandi
```bash
# Training (2-5 min su GPU, ~10 min CPU)
.venv/bin/python train.py --use-wandb

# Valutazione + Gate 1
.venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
```

### Decisioni tecniche
- Architettura 1D per il toy: più veloce, stessa validazione del meccanismo SAMI
- DDPM training + DDIM sampling (50 step, come paper originale)
- z detached nel calcolo di g_t (corretto per training asimmetrico)
- Global normalization sui segnali (preserva ampiezza come fattore)
- MIG implementato con discretizzazione a 20 bin (stile FactorVAE paper)

### Riferimenti
- Modulo `sami.py` è architettura-agnostico: stessa classe per toy e NSynth
- Tutte le metriche di disentanglement sono implementate a mano (no disentanglement-lib)

**Struttura directory:**
- `data/`, `models/`, `models/losses/`, `experiments/`, `notebooks/`, `checkpoints/`
- `docs/AGENTS.md`, `docs/roadmap.md`, `docs/CHANGELOG.md`

**File creati:**

| File | Descrizione |
|------|-------------|
| `.gitignore` | Ignora venv, data, checkpoints, wandb, notebook checkpoints |
| `pyproject.toml` | PDM project config: `sami-audio` v0.1.0, Python >=3.10, tutte le dipendenze |
| `requirements.txt` | Pip fallback con stesse dipendenze |
| `CONVENTIONS.md` | 5 decisioni immutabili (Nodo 1: audio params, Nodo 2: β strategy, Nodo 3: training, Nodo 4: metrics, Nodo 5: vocoder) |
| `data/download.sh` | Script bash: download NSynth train (21 GB), estrai, filtra 4 famiglie + pitch [48,84], elimina wav non necessari |
| `data/nsynth.py` | Dataset class completa: MelConfig (immutable), NSynthDataset (filter, mel, normalize), mel_to_audio (Griffin-Lim), Gate 0 test in `__main__` |
| `data/__init__.py` | Package init, esporta NSynthDataset, MelConfig, mel_to_audio |
| `models/__init__.py` | Package init con docstring dei moduli futuri |
| `models/losses/__init__.py` | Package init per diffusion e disentanglement losses |
| `README.md` | Roadmap completa con stato di ogni fase, quick start, priorità |

**Dipendenze installate (PDM venv):**
- `torch==2.13.0`, `torchaudio==2.11.0`, `accelerate==1.14.0`
- `einops==0.8.2`, `hydra-core==1.3.4`
- `librosa==0.11.0`, `soundfile==0.14.0`
- `torchmetrics==1.9.0`, `torchcrepe==0.0.24`
- `umap-learn==0.5.12`, `matplotlib==3.10.9`, `scikit-learn==1.7.2`
- `wandb==0.28.1`, `tqdm==4.69.0`, `numpy==2.2.6`, `pandas==2.3.3`

**Decisioni tecniche:**
1. `torchcrepe` al posto di `crepe` — il pacchetto `crepe` originale ha il build rotto (`pkg_resources` non trovato). `torchcrepe` è il porting PyTorch mantenuto.
2. Wandb API key configurata via `wandb login`.

**Da fare (manuale):**
- [x] Eseguire `bash data/download.sh` per scaricare NSynth
- [x] Eseguire `pdm run gate0` per validare il dataset

**Risultato Gate 0:** PASSED ✓
- 61,531 campioni (guitar: 15,643, keyboard: 24,261, string: 12,587, brass: 9,040)
- Shape: `(1, 128, 256)`, valori in `[-1, 1]`
- Griffin-Lim reconstruction: `data/gate0_test.wav`

**Fix applicati:**
- `torchaudio.load` → `soundfile.read`: torchaudio 2.11.0 richiede `torchcodec` obbligatoriamente. Sostituito con soundfile (già installato).
- `torchaudio.save` → `soundfile.write`: stessa ragione.
- Gate 0 check: da tutti i 61k campioni a 200 random (più veloce).
- `download.sh`: `EXTRACT_DIR` esportato nell'environment per lo script Python inline.

**File non modificati:**
- `docs/Proggetto_Deep.pdf` (specifica originale)

---

## Template per entry future

```markdown
## YYYY-MM-DD — Phase X: Nome Fase

### Cosa fatto
- Breve descrizione

### File modificati
| File | Tipo | Descrizione |
|------|------|-------------|
| `path/file.py` | Modifica/Creato | Cosa contiene |

### Decisioni prese
- Decisione 1 e motivazione
- Decisione 2 e motivazione

### Gate status
- [x] Gate X: PASSED/FAILED — note

### Blocker / Da fare
- [ ] Cosa rimane da fare
```
