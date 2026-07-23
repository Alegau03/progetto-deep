# AGENTS.md — SAMI-Audio

Istruzioni operative per agenti AI che lavorano su questo progetto.

---

## Regole Generali

1. **Leggi `CONVENTIONS.md` prima di modificare qualsiasi file.** Contiene decisioni immutabili (Nodo 1-5).
2. **Non modificare mai `CONVENTIONS.md` senza discuterne le conseguenze.** Ogni modifica richiede ri-addestramento.
3. **Ogni file Python deve essere commentato** con docstring di modulo, classe e metodo in stile Google.
4. **Ogni fase ha un gate go/no-go.** Non passare alla fase successiva finché il gate non è superato.
5. **Testa in isolamento prima di scalare.** Toy model (Phase 1) prima di NSynth (Phase 2+).
6. **Aggiorna `CHANGELOG.md` a ogni modifica significativa.** Data, cosa fatto, file coinvolti.
7. **Aggiorna `roadmap.md` al completamento di ogni task.** Spunta i task completati.

## Stack Tecnologico

| Componente | Tool |
|------------|------|
| Package manager | **PDM** (`pdm install`, `pdm add`, `pdm run`) |
| Virtual env | `.venv/` (creato automaticamente da PDM) |
| Python | 3.12.7 |
| ML framework | PyTorch 2.13.0 + torchaudio 2.11.0 |
| Accelerazione | `accelerate` (HuggingFace) |
| Configurazione | `hydra-core` (YAML-based, `experiments/`) |
| Logging | `wandb` (Weights & Biases) |
| Pitch detection | `torchcrepe` (PyTorch port, NON `crepe` originale — build rotto) |
| Visualizzazione | `matplotlib`, `umap-learn` |
| Metriche disentanglement | **Implementate a mano** (NO `disentanglement-lib`) |
| Vocoder demo | HiFi-GAN pre-addestrato; baseline Griffin-Lim |

## Struttura del Codice

```
sami_audio/
├── CONVENTIONS.md          # Decisioni immutabili — LEGGERE PRIMA DI TUTTO
├── data/
│   ├── download.sh         # Download + filtra NSynth
│   └── nsynth.py           # Dataset class, MelConfig, mel_to_audio
├── models/
│   ├── encoder.py          # Half-UNet → (μ, σ²)
│   ├── unet.py             # Denoiser incondizionato
│   ├── sami.py             # Core SAMI: guidance + loss + sampling
│   ├── baselines.py        # β-VAE, FactorVAE
│   └── losses/
│       ├── diffusion.py    # Cosine schedule, q_sample, ᾱ_t
│       └── disentanglement.py  # KL, CLUB, FactorVAE TC
├── train.py                # Training loop (Accelerate + Hydra)
├── evaluate.py             # TAD, DCI, MIG, FactorVAE, FID, SI-SNR
├── experiments/            # Hydra YAML config per esperimento
├── notebooks/
│   ├── audio_demo.ipynb    # Timbre transfer demo
│   └── timbre_map.ipynb    # UMAP visualization
├── docs/
│   ├── AGENTS.md           # Questo file
│   ├── roadmap.md          # Roadmap dettagliata con task
│   ├── CHANGELOG.md        # Log modifiche
│   └── Proggetto_Deep.pdf  # Specifica originale
└── pyproject.toml          # Dipendenze e script PDM
```

## Workflow di Sviluppo

### Prima di ogni modifica
```bash
pdm install          # assicurati che le dipendenze siano aggiornate
```

### Per eseguire script
```bash
pdm run gate0        # Gate 0: verifica dataset
pdm run download     # Scarica NSynth (solo se necessario)
.venv/bin/python script.py   # Esegui qualsiasi script Python nel venv
```

### Per aggiungere dipendenze
```bash
pdm add <package>
# Aggiorna anche requirements.txt se necessario
```

## Pattern Critici

### Guidance via autodiff (il punto più delicato)

```python
# Questo è il cuore di SAMI. Deve funzionare in un grafo computazionale
# dove xt richiede gradienti e il gradiente di log_q viene usato per
# deviare la stima del rumore.

log_q = mahalanobis_log_prob(z, mu_t, sigma_t)  # scalare
g_t = torch.autograd.grad(log_q.sum(), xt, create_graph=True)[0]
# g_t ha la stessa shape di xt

# Loss:
eps_hat = eps_theta - gamma_t * g_t
L_x = ((eps - eps_hat) ** 2).mean()
L_z = kl_divergence(mu0, sigma0)
loss = L_x + beta * L_z
loss.backward()  # propaga attraverso g_t grazie a create_graph=True
```

### Mel spectrogram shape invariants

```
Input waveform:  (1, num_samples) @ 16kHz
Mel spectrogram: (1, 128, num_frames) dopo log + clamp
Dopo crop/pad:   (1, 128, 256) in [-1, 1]
```

## Note sul Dataset

- NSynth subset: 4 famiglie (guitar, keyboard, string, brass), pitch MIDI [48, 84]
- Label (pitch, instrument_family, qualities) esposte ma **mai usate nel training**
- Il dataset originale è ~21 GB; dopo filtraggio ~4 GB
- Se `data/nsynth-train/` non esiste, eseguire `bash data/download.sh`

## Convenzioni di Codice

- **Niente commenti superflui** nei file .py. I commenti spiegano il *perché*, non il *cosa*.
- Docstring obbligatorie per moduli, classi e metodi pubblici.
- Type hints su tutte le firme di funzioni/metodi pubbliche.
- Naming: `snake_case` per variabili/funzioni, `PascalCase` per classi, `UPPER_CASE` per costanti.
- Lunghezza righe: 100 caratteri massimo.
