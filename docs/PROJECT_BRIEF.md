# SAMI-Audio — Brief di Progetto

> **Da leggere all'inizio di ogni sessione.**
> Questo documento spiega COSA stiamo costruendo, PERCHÉ, e COME funziona il meccanismo centrale.
> Per i dettagli operativi vedi `roadmap.md`. Per le decisioni fisse vedi `CONVENTIONS.md`.

---

## 1. L'Idea in Una Frase

Costruiamo un modello che, dato lo spettrogramma di una nota musicale, impara **da solo** (senza supervisione) a separare l'altezza (pitch) dal timbro (colore sonoro dello strumento) in uno spazio latente strutturato.

Se funziona, possiamo prendere il "pitch" di un violino e il "timbro" di un pianoforte e generare un suono nuovo: un pianoforte che suona quella nota.

---

## 2. Il Problema che Risolviamo

- **I VAE classici** producono spazi latenti non strutturati: le dimensioni di `z` non corrispondono a fattori interpretabili.
- **Il β-VAE** forza il disentanglement aumentando il peso del termine KL, ma paga in qualità di ricostruzione (tradeoff rate-distortion).
- **I modelli audio esistenti** che separano pitch e timbro (es. GANSynth, pGESAM) richiedono **supervisione** sul pitch — glielo devi dire tu qual è il pitch durante il training.

Noi vogliamo un modello **completamente non supervisionato** che faccia emergere pitch e timbro come assi distinti nello spazio latente.

---

## 3. SAMI: Il Metodo

### 3.1 Cos'è SAMI

**SAMI** (Score-based Autoencoder for Multiscale Inference) è un paper del 2025 di Lyo, Simoncelli & Savin (NYU, arXiv:2512.17127) che propone un nuovo tipo di autoencoder generativo.

L'intuizione: sostituire il decoder deterministico del VAE con un **diffusion model** (DDPM), e usare il processo di denoising come "guida" che struttura lo spazio latente.

### 3.2 Il Meccanismo Centrale (LA COSA DA NON SBAGLIARE)

SAMI **NON** inietta `z` dentro la U-Net (come fanno DiffAE, DisDiff, ecc.). Invece:

1. **Abbiamo due reti separate:**
   - **Denoiser incondizionato** `ε_θ(x_t, t)` — una U-Net standard che **non vede mai `z`**
   - **Inference network** `f_φ` — una "Half-UNet" (solo Down + Mid, senza Up) che codifica sia l'input pulito `x₀` sia quello rumoroso `x_t`, producendo `(μ, σ²)` del posterior

2. **La guidance score è il gradiente** (via autodiff) della log-verosimiglianza del latente:
   ```
   g_t = ∇_{x_t} log q_φ(z | x_t)
   ```
   dove `log q_φ(z|x_t)` usa la distanza di Mahalanobis rispetto a `(μ_φ(x_t), Σ_φ(x_t))`.

3. **La stima del rumore viene deviata:**
   ```
   ε̂(x_t, t, z) = ε_θ(x_t, t) − γ_t · g_t
   ```
   con `γ_t = √(1 − ᾱ_t)`.

4. **Loss di training (β-SAMI):**
   ```
   L = E[ λ_t · ‖ ε − (ε_θ(x_t, t) − γ_t · g_t) ‖² ] + β · KL( q_φ(z|x₀) ‖ N(0, I) )
   ```

### 3.3 Perché Questo Meccanismo Funziona

- Il denoiser impara a ricostruire **tutti** i dettagli dell'immagine/suono.
- L'encoder impara un latente `z` che, quando usato per deviare il denoiser via `g_t`, aiuta la ricostruzione.
- La teoria di SAMI (Teoremi in Appendice B del paper) dimostra che questo processo **da solo** spinge il latente a fattorizzarsi in fattori generativi distinti, **senza regolarizzatori espliciti**.
- Il disentanglement emerge se i fattori vivono a **livelli di rumore caratteristici distinti** (es. struttura grossolana vs dettaglio fine).

### 3.4 Pseudocodice del Training

```python
for batch:
    x0 = dati                                   # mel pulito
    t ~ Uniform(0, T)                           # timestep
    eps ~ N(0, I)                               # rumore gaussiano
    xt = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * eps    # forward diffusion

    # Encoder su x0 pulito — produce il latente z
    mu0, sigma0 = f_phi(x0)
    z = mu0 + sigma0 * eps_z                    # reparameterization

    # Encoder su xt rumoroso — STESSA RETE
    mu_t, sigma_t = f_phi(xt)

    # Log-verosimiglianza con Mahalanobis
    log_q = -0.5 * (mahalanobis(z, mu_t, sigma_t) + log|sigma_t|)

    # Guidance score via autodiff (IL PUNTO CRITICO)
    g_t = torch.autograd.grad(log_q.sum(), xt, create_graph=True)[0]

    # Loss: denoising deviato + regolarizzazione KL
    eps_pred = eps_theta(xt, t) - sqrt(1 - ᾱ_t) * g_t
    L_x = || eps - eps_pred ||^2
    L_z = KL( N(mu0, sigma0) || N(0, I) )
    loss = L_x + beta * L_z

    loss.backward()  # propaga attraverso g_t grazie a create_graph=True
```

### 3.5 Nota Critica: Il Latente NON È Partizionato

SAMI non divide `z` in `[z_pitch | z_timbre | z_dynamics]`. Usa un **singolo vettore latente** (noi: 32-128 dim) e lascia che il disentanglement emerga naturalmente sugli assi. Dopo il training, identifichiamo a posteriori quali assi codificano pitch (sonda con CREPE) e timbro (sonda con `instrument_family`).

---

## 4. Dominio: Dall'Immagine all'Audio

SAMI è nato per le immagini. Noi lo portiamo sull'**audio strumentale**.

### 4.1 Dataset: NSynth

- **NSynth** (Engel et al., 2017): dataset di note musicali sintetizzate da vari strumenti.
- **Subset usato:** 4 famiglie strumentali (guitar, keyboard, string, brass), pitch MIDI [48, 84] (C3-B5).
- **Dimensione:** ~50.000 note, ~4 GB dopo filtraggio.
- **Label** (pitch, instrument_family, qualities) disponibili ma usate **solo in evaluation, mai nel training**.

### 4.2 Preprocessing Audio

Ogni nota `.wav` (16 kHz mono) viene convertita in mel-spettrogramma:

```
.wav (16 kHz) → STFT (n_fft=1024, hop=256) → Mel (n_mels=128, f_min=0, f_max=8000)
→ log compression → normalize to [-1, 1] → crop/pad to 256 frames
```

**Output shape:** `(1, 128, 256)` = [canali, mel_bins, time_frames]

### 4.3 Perché lo Spettrogramma e Non il Raw Audio

- Il diffusion model opera nello spazio dati (il mel), non in un latente compresso.
- 128 e 256 sono divisibili per 8 → compatibili con U-Net a 3 livelli di downsampling.
- Le metriche quantitative sono in spazio mel → indipendenti dal vocoder.

---

## 5. Architettura

### 5.1 Inference Network (Half-UNet)

```
Input: mel (1, 128, 256)
Blocchi Down (×4): Conv2d → SiLU → GroupNorm
Blocco Mid: Conv2d → SiLU → GroupNorm
Testa μ: MLP → vettore D_latent
Testa σ²: MLP → Softplus → quadratura → vettore D_latent

Base channels: 64
Channel multiplier: [1, 1, 1, 1]
D_latent: 64 (default; ablare 32, 128)
```

**Importante:** L'encoder è **cieco al livello di rumore** — non riceve `t` in input. Questo è fedele al design di SAMI.

### 5.2 Denoiser (U-Net)

```
Input: mel (1, 128, 256) + time embedding
Blocchi Down (×3): ResBlock + eventuale self-attention
Blocco Mid: ResBlock + self-attention
Blocchi Up (×3): ResBlock + skip connections
Output: mel (1, 128, 256)

Base channels: 128
Channel multiplier: [1, 1, 2, 2]
Time embedding: sinusoidale (stile DDPM)
Self-attention: solo al bottleneck nella versione base
```

### 5.3 Diffusion Process

- **T = 1000** timesteps
- **Cosine schedule** (Nichol & Dhariwal), `s = 0.008`
- **Inference:** DDIM, 50-100 step (sampling veloce)

---

## 6. Training Strategy

### Strategia a Basso Costo Computazionale

Invece di addestrare encoder + denoiser insieme (costoso), il paper SAMI valida questa strategia:

1. **Pre-addestra il denoiser incondizionato una volta** — impara a generare mel-spettrogrammi.
2. **Congela il denoiser.**
3. **Addestra solo l'inference network** (Half-UNet, molto più piccola) con la loss SAMI.
4. **Sovracampiona i livelli di rumore alti** — alle alte `t` la guidance domina.

Questo riduce sostanzialmente lo sforzo computazionale (validato dal paper su CelebA-HQ).

### Iperparametri Chiave

| Parametro | Valore |
|-----------|--------|
| Ottimizzatore | Adam senza weight decay |
| Learning rate | 1e-4 (congiunto) / 3e-5 (solo inference) |
| Mixed precision | bf16 |
| Gradient clip | 1.0 |
| EMA | Sul denoiser (se addestrato) |
| Batch size | Il più grande che entra in GPU (128-256) |
| β strategy | Annealing esponenziale + binary search sotto collapse |
| GPU target | L40S (48 GB) |

---

## 7. Metriche di Valutazione

### Disentanglement (primarie)
| Metrica | Cosa misura |
|---------|-------------|
| **TAD** (Total AUROC Difference) | Quanto ogni attributo è catturato da un singolo asse latente |
| DCI-Disentanglement | Complementare a TAD |
| MIG (Mutual Information Gap) | Gap tra i due assi con più alta informazione mutua per ogni fattore |
| FactorVAE metric | Classificatore a maggioranza, robusta e senza iperparametri |

### Qualità di Generazione
| Metrica | Cosa misura |
|---------|-------------|
| FID su mel | Distanza tra distribuzione reale e generata |
| SI-SNR | Signal-to-noise ratio |
| log-mel L1 | Errore di ricostruzione |

### Demo (qualitative)
| Metrica | Cosa misura |
|---------|-------------|
| Pitch accuracy (CREPE) | Il pitch si conserva nel timbre transfer? |
| Timbre Transfer Accuracy | Il timbro cambia come richiesto? |

---

## 8. Fasi del Progetto

```
Phase 0: Setup & Dataset                               ✅ COMPLETATA
Phase 1: Toy Model (sinusoidi, latente 2D)              🔲 PROSSIMA
Phase 2: β-VAE Baseline su NSynth                       🔲
Phase 3: SAMI-Audio Base (denoiser + inference)         🔲
Phase 4: Demo (timbre map, timbre transfer)             🔲
Phase 5: Analisi scale + Report                         🔲
Phase 6: Polifonia (opzionale, tagliabile)              🔲
```

**Priorità se il tempo stringe:** Toy model → β-VAE baseline → un solo run SAMI solido → eval + demo.

---

## 9. Domanda di Ricerca

> La formulazione score-based di SAMI, applicata a spettrogrammi di note strumentali, produce un latent space più disentangled (pitch vs timbro) di un β-VAE, senza usare label di pitch al training?

E, domanda secondaria ma scientificamente interessante:

> Le note strumentali sono multi-scala (fondamentale = struttura grossolana, armoniche = dettaglio fine) come le immagini naturali, o a scala singola come dSprites?

In entrambi i casi il risultato — positivo o negativo, purché analizzato — è un contributo.

---

## 10. Contributi Attesi

1. **Primo porting documentato** di SAMI dal dominio immagini all'audio.
2. **Completamente non supervisionato**: nessuna label di pitch al training.
3. **Analisi comparativa quantitativa**: SAMI vs β-VAE vs FactorVAE su metriche standard.
4. **Analisi delle scale di rumore**: test empirico dell'assunzione teorica di SAMI sull'audio.
5. **Demo interattiva**: timbre map 2D e generazione di timbri ibridi.

---

## 11. Riferimenti Rapidi

| Cosa | Dove |
|------|------|
| Specifica originale | `docs/Proggetto_Deep.pdf` |
| Decisioni fisse | `CONVENTIONS.md` |
| Roadmap operativa | `docs/roadmap.md` |
| Istruzioni per AI | `docs/AGENTS.md` |
| Log modifiche | `docs/CHANGELOG.md` |
| Questo documento | `docs/PROJECT_BRIEF.md` |

---

*SAMI-Audio — DLAI 2025/26. Progetto di Alessandro Gautieri.*
