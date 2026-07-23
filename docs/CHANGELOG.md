# CHANGELOG — SAMI-Audio

Log cronologico delle modifiche al progetto. Ogni entry include data, fase, descrizione e file coinvolti.

---

## 2025-07-23 — Phase 0: Setup & Dataset

### Creato progetto da zero

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
- [ ] Eseguire `bash data/download.sh` per scaricare NSynth
- [ ] Eseguire `pdm run gate0` per validare il dataset

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
