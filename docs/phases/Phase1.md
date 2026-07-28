# Phase 1 — Toy Model: Validazione del Meccanismo SAMI

> **Branch:** `fase-1`
> **Durata:** ~5 minuti su GPU, ~30 secondi su GPU con i parametri ottimizzati
> **Dataset:** Sinusoidi sintetiche (non NSynth), generato in memoria
> **Stato:** ✅ Codice completato — pronto per l'esecuzione su cluster GPU
> **Esecuzione:** Cluster GPU (SLURM o session interattiva)

---

## 0. Setup Rapido per il Cluster

### 0.1 Trasferimento codice al cluster

```bash
# Dal tuo Mac, copia il progetto sul cluster
rsync -avz --exclude '.venv' --exclude 'data/nsynth-*' --exclude 'wandb' \
  ~/Documents/GitHub/progetto-deep/ \
  user@cluster:/path/to/sami-audio/

# Sul cluster, reinstalla le dipendenze
ssh user@cluster
cd /path/to/sami-audio
pdm install
wandb login  # usa la tua API key
```

### 0.2 Esecuzione

```bash
# Opzione A: SLURM batch job (raccomandato)
sbatch scripts/train_toy.slurm

# Opzione B: Sessione interattiva con GPU
srun --gres=gpu:1 --time=01:00:00 --pty bash
.venv/bin/python train.py --use-wandb

# Opzione C: Locale con GPU
.venv/bin/python train.py --use-wandb
```

### 0.3 Monitoraggio

- **Wandb:** https://wandb.ai/alessandro-gautieri-sapienza-universit-di-roma/sami-audio-toy
- **SLURM:** `squeue -u $USER`, `tail -f slurm-<jobid>.out`
- **Checkpoint:** `checkpoints/toy/model_final.pt` quando il job completa

### 0.4 Dopo il training

```bash
# Scarica i risultati sul Mac
rsync -avz user@cluster:/path/to/sami-audio/checkpoints/ ./checkpoints/
rsync -avz user@cluster:/path/to/sami-audio/plots/ ./plots/

# Valuta il modello
.venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
```

Validare che il meccanismo centrale di SAMI — la **guidance via autodiff** — funzioni correttamente, su un problema minuscolo dove conosciamo la verità.

**Perché prima di NSynth?** Se c'è un bug nel codice (loss, guidance, sampling), lo scopriamo in **minuti** anziché dopo ore di training su dati reali. Questo è il gate go/no-go dell'intero progetto.

---

## 2. Cosa Abbiamo Costruito

### 2.1 Dataset Sintetico

Generiamo proceduralmente 50 classi di sinusoidi:

```
y(t) = A · sin(2π · f · t / L + φ) + N(0, 0.02)
```

| Parametro | Valori |
|-----------|--------|
| Frequenze | 10 valori lineari tra 1.0 e 20.0 (cicli su L=256) |
| Ampiezze | 5 valori lineari tra 0.2 e 1.0 |
| Campioni per classe | 500 (25.000 totali) |
| Lunghezza segnale | 256 punti (1D) |
| Rumore | Gaussiano, σ=0.02 |
| Fase | Random per ogni campione |

**Fattori ground-truth noti:**
- **Frequenza** = simulazione del pitch
- **Ampiezza** = simulazione della dinamica

Il dataset sta interamente in memoria (nessun file su disco).

### 2.2 Architettura

Tutti i modelli sono **1D** (segnali di 256 punti). Il passaggio a 2D per i mel-spettrogrammi NSynth in Phase 3 è meccanico — il meccanismo SAMI è agnostico alla dimensionalità.

#### Inference Network (Half-UNet 1D)
```
Input: segnale 1D (B, 1, 256)
├── Down1: Conv1d(1→32, stride=2) + GroupNorm + SiLU  → (B, 32, 128)
├── Down2: Conv1d(32→64, stride=2) + GroupNorm + SiLU → (B, 64, 64)
├── Down3: Conv1d(64→128, stride=2) + GroupNorm + SiLU → (B, 128, 32)
├── Mid:   Conv1d(128→128) + GroupNorm + SiLU        → (B, 128, 32)
├── Pool:  AdaptiveAvgPool1d(1)                       → (B, 128)
├── μ head:  Linear(128, 2)                           → (B, 2)
└── σ² head: Linear(128, 2) → Softplus + ε            → (B, 2)
```
D_latent = 2 per visualizzazione diretta in scatter plot.

#### Denoiser (U-Net 1D)
```
Input: segnale rumoroso (B, 1, 256) + time t
├── TimeEmbedding: sinusoidale → MLP(128→512→128)
├── Down1: 2× ResBlock1D(1→32, stride=2) + skip     → (B, 32, 128)
├── Down2: 2× ResBlock1D(32→64, stride=2) + skip    → (B, 64, 64)
├── Down3: 2× ResBlock1D(64→128, stride=2) + skip   → (B, 128, 32)
├── Mid: 2× ResBlock1D(128→128)
├── Up3: upsample + cat skip + 2× ResBlock           → (B, 64, 64)
├── Up2: upsample + cat skip + 2× ResBlock           → (B, 32, 128)
├── Up1: upsample + cat skip + 2× ResBlock           → (B, 32, 256)
└── Out: GroupNorm + SiLU + Conv1d(32→1)             → (B, 1, 256)
```
~576.000 parametri totali (encoder + denoiser).

#### Diffusion Schedule
```
Cosine schedule (Nichol & Dhariwal 2021):
- T = 200 timesteps (ridotto per velocità)
- s = 0.008 (offset di stabilità numerica)
- DDPM training, DDIM sampling (50 step)
```

### 2.3 Il Meccanismo SAMI nel Dettaglio

Questo è il training loop che gira ad ogni batch:

```python
# 1. Forward diffusion
t ~ Uniform(0, T)
ε ~ N(0, I)
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε

# 2. Encoder su x₀ pulito → latente z
μ₀, σ²₀ = encoder(x₀)
z = μ₀ + √σ²₀ · ε_z              # reparameterization
z = z.detach()                    # distaccato per training asimmetrico

# 3. Encoder su x_t rumoroso → guida
x_t.requires_grad_(True)
μ_t, σ²_t = encoder(x_t)

log_q = mahalanobis_log_prob(z, μ_t, σ²_t).sum()
g_t = torch.autograd.grad(log_q, x_t, create_graph=True)[0]

# 4. Denoiser + guidance
ε̂ = denoiser(x_t, t)
ε_guided = ε̂ - γ_t · g_t         # deviazione via guidance

# 5. Loss
L_x = MSE(ε_guided, ε)           # score matching
L_z = KL(N(μ₀,σ²₀) || N(0,I))    # regolarizzazione
loss = L_x + β · L_z
```

**Il punto critico:** `create_graph=True` in `autograd.grad` permette di backpropagare attraverso il gradiente stesso. Se questo rompe, il modello degenera in un semplice denoiser senza disentanglement.

---

## 3. Codice

### 3.1 File creati

| File | Contenuto | Righe |
|------|-----------|-------|
| `models/losses/diffusion.py` | `DiffusionSchedule`: cosine schedule, `q_sample`, `gamma()` | 50 |
| `models/encoder.py` | `ToyEncoder`: Half-UNet 1D → (μ, σ²), D_latent=2 | 60 |
| `models/losses/disentanglement.py` | `kl_divergence()`, `mahalanobis_log_prob()` | 45 |
| `models/unet.py` | `ToyUNet`: 1D U-Net denoiser con ResBlock e skip | 170 |
| `models/sami.py` | `SAMI`: core con forward + DDIM sampling (arch-agnostico) | 170 |
| `train.py` | Dataset sinusoidi + training loop + CLI | 315 |
| `evaluate.py` | MIG + scatter plot + Gate 1 check | 220 |

### 3.2 Esecuzione

**Su cluster GPU (SLURM):**
```bash
# Job batch (raccomandato)
sbatch scripts/train_toy.slurm

# Sessione interattiva
srun --gres=gpu:1 --time=01:00:00 --pty bash
.venv/bin/python train.py --use-wandb --epochs 30 --batch-size 256
```

**Locale (qualsiasi GPU o CPU):**
```bash
.venv/bin/python train.py --use-wandb
```

**Valutazione (dopo training):**
```bash
.venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
```

**Con parametri personalizzati:**
```bash
# Dataset più grande se hai VRAM
.venv/bin/python train.py --n-per-class 1000 --batch-size 256 --epochs 40

# Più timestep di diffusione (più accurato, più lento)
.venv/bin/python train.py --T 500 --epochs 50

# Debug rapido (pochi dati, poche epoche)
.venv/bin/python train.py --n-per-class 100 --epochs 5 --beta-warmup 3
```

### 3.3 Parametri da CLI

| Parametro | Default | GPU consigliato | Descrizione |
|-----------|---------|-----------------|-------------|
| `--epochs` | 30 | 30-50 | Epoche totali |
| `--batch-size` | 128 | 256-512 | Dimensione batch (raddoppiabile su GPU) |
| `--lr` | 1e-4 | 1e-4 | Learning rate Adam |
| `--beta-start` | 1e-6 | 1e-6 | β iniziale (KL quasi zero) |
| `--beta-end` | 1.0 | 1.0 | β target dopo warmup |
| `--beta-warmup` | 10 | 10-15 | Epoche di annealing esponenziale |
| `--n-per-class` | 500 | 1000-2000 | Campioni per classe (×50 classi) |
| `--T` | 200 | 200-500 | Timestep di diffusione |
| `--latent-dim` | 2 | 2 | Dimensione latente (fisso per toy) |
| `--use-wandb` | false | true | Abilita logging wandb |

**Nota GPU:** Con 48 GB (L40S), batch 512 e n-per-class 2000 entrano comodamente. Il toy model usa <1 GB VRAM.

---

## 4. Problema Riscontrato: Posterior Collapse

### Sintomo

Al primo tentativo di training (β=1.0 fisso):

```
Epoch  1 | loss=0.1200  L_x=0.1195  L_z=0.0005
Epoch  2 | loss=0.0532  L_x=0.0532  L_z=0.0000   ← collapse!
Epoch  3 | loss=0.0464  L_x=0.0464  L_z=0.0000
...
Epoch 13 | loss=0.0339  L_x=0.0339  L_z=0.0000
```

### Diagnosi

Con β alto dall'inizio, il termine KL domina e spinge l'encoder verso `μ≈0, σ²≈1` per ogni input. Quando questo succede:

1. `KL(N(0,1) || N(0,1)) = 0` → **L_z scompare**
2. `log q(z | x_t)` diventa costante → **g_t = 0**
3. La guidance è inattiva → il modello degenera in un **denoiser incondizionato**
4. **Nessun disentanglement possibile**

### Soluzione: KL Annealing Esponenziale

β parte da `1e-6` (KL trascurabile) e cresce esponenzialmente:

| Epoch | β | Significato |
|-------|---|-------------|
| 1 | 3.98e-06 | L'encoder impara a ricostruire senza pressione KL |
| 3 | 6.31e-05 | Inizia una leggera regolarizzazione |
| 5 | 1.00e-03 | L_z inizia a contare |
| 7 | 1.58e-02 | Struttura latente emerge |
| 10+ | 1.00e+00 | KL pieno, disentanglement atteso |

L'idea: il modello impara prima a codificare informazione utile nel latente (β basso → L_z non punisce), poi viene gradualmente forzato a fattorizzarla (β alto → KL forza indipendenza).

---

## 5. Gate 1 — Criterio di Successo

### Cosa valutiamo

Dopo il training, eseguiamo `evaluate.py` che:

1. **Encoda l'intero dataset** nel latente 2D (μ dell'encoder per ogni campione)
2. **Genera scatter plot** `plots/toy_latent_space.png`:
   - Pannello sinistro: punti colorati per **frequenza**
   - Pannello destro: punti colorati per **ampiezza**
3. **Calcola MIG** (Mutual Information Gap) per entrambi i fattori

### Criterio

```
GATE 1 PASSED se:
  - Scatter plot: cluster separati per frequenza su un asse e ampiezza sull'altro
  - MIG_freq > 0.5
  - MIG_amp > 0.5
```

### Cosa significa MIG

Il MIG misura quanto **esclusivamente** ogni fattore è codificato da un singolo asse latente:

```
Per ogni fattore (es. frequenza):
  1. Calcola I(z_j; frequenza) per ogni dimensione j del latente
  2. Ordina le dimensioni per I decrescente
  3. MIG = [I(z_best; freq) - I(z_second; freq)] / H(freq)
```

MIG = 0 → nessun asse codifica il fattore meglio degli altri.
MIG = 1 → un asse codifica perfettamente il fattore e tutti gli altri lo ignorano.

---

## 6. Output e Flusso di Lavoro

### 6.1 Sul cluster (dopo `sbatch`)

Il job SLURM produce:
- `slurm-<jobid>.out` — log completo del training
- `checkpoints/toy/toy_epoch_0005.pt` — checkpoint ogni 5 epoche
- `checkpoints/toy/model_final.pt` — checkpoint finale
- Log wandb in tempo reale (streaming)

### 6.2 Dopo il training (sul Mac)

```bash
# 1. Scarica i risultati dal cluster
rsync -avz user@cluster:/path/to/sami-audio/checkpoints/toy/ ./checkpoints/toy/

# 2. Valuta il modello + Gate 1
.venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt

# 3. Output generati
#    plots/toy_latent_space.png  — scatter plot 2D
#    Terminal: GATE 1: PASSED/FAILED con MIG scores
```

### 6.3 Wandb dashboard

Accessibile da qualsiasi browser dopo il login:
https://wandb.ai/alessandro-gautieri-sapienza-universit-di-roma/sami-audio-toy

**Cosa guardare:**
| Metrica | Comportamento atteso |
|---------|---------------------|
| `train/loss` | Decresce rapidamente, si stabilizza ~0.02-0.05 |
| `train/L_x` | Decresce (score matching migliora) |
| `train/L_z` | Cresce da 0 a ~1-5 con l'annealing (NON deve restare 0) |
| `train/beta` | Curva esponenziale da 1e-6 a 1.0 |
| `train/lr` | Cosine decay da 1e-4 a ~0 |

**⚠️ Alert:** Se `L_z` resta a 0.0000 dopo l'epoca 3 → posterior collapse. Ferma e riduci `--beta-start`.

---

## 7. Cosa Impariamo

| Se... | Allora... |
|-------|-----------|
| MIG > 0.5 e scatter bello | Il meccanismo SAMI funziona. Pronti per NSynth (Phase 2). |
| L_z resta zero nonostante annealing | β_start troppo basso o troppo alto. Tuning iperparametri. |
| L_x non scende | Learning rate o architettura da rivedere. |
| MIG < 0.3 e scatter uniforme | Bug nel codice di guidance (molto probabile: `create_graph=True` o `detach()`). |

---

## 8. Stato Finale e Transizione a Phase 2

### Cosa è pronto

- [x] Dataset sintetico generato in memoria (25k-100k campioni)
- [x] Architettura 1D completa (encoder Half-UNet, denoiser U-Net)
- [x] Diffusion schedule cosine con T configurabile
- [x] Core SAMI con guidance via autodiff (`create_graph=True`)
- [x] KL annealing esponenziale per prevenire posterior collapse
- [x] Training loop con logging wandb e checkpoint automatici
- [x] Valutazione con MIG e scatter plot 2D
- [x] SLURM script per cluster GPU
- [x] Gate 1 go/no-go automatico

### Cosa rimane da fare

- [ ] Eseguire `sbatch scripts/train_toy.slurm` sul cluster
- [ ] Verificare su wandb che L_z > 0 (no posterior collapse)
- [ ] Scaricare checkpoint e valutare con `evaluate.py`
- [ ] Confermare Gate 1 PASSED (MIG > 0.5, scatter bello)

### Con Gate 1 superato — Phase 2

| Modulo | Stato | Azione per Phase 2 |
|--------|-------|--------------------|
| `diffusion.py` | ✅ Riutilizzabile | Nessuna modifica (arch-agnostico) |
| `disentanglement.py` | ✅ Riutilizzabile | Nessuna modifica |
| `sami.py` | ✅ Riutilizzabile | Nessuna modifica (stessa classe) |
| `encoder.py` | 🔄 Da riscrivere | Nuovo encoder 2D per mel (1,128,256) |
| `unet.py` | 🔄 Da riscrivere | Nuovo denoiser 2D per mel |
| `train.py` | 🔄 Da adattare | NSynth dataset + Hydra config |
| `evaluate.py` | 🔄 Da estendere | TAD, DCI, FID, SI-SNR metriche |

---

*Documento creato il 2025-07-28. Ultimo aggiornamento: aggiunta sezione cluster GPU, SLURM script, flusso di lavoro end-to-end.*
