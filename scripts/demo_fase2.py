#!/usr/bin/env python3
"""
Demo Fase 2 — ricalcolo onesto con metriche corrette.

Fix rispetto a Fase 1a/1b:
  1. Timbro su OUTPUT con MAGGIORANZA (≥3/5 vicini = guitar), non any()
  2. Denominatore "X/Y semi validi" (A, T, B tutti misurabili con CREPE conf>0.5)
  3. Sanity check incluso per ogni s/α (A vs B distinti per seme)
  4. WAV salvati per A, B, T (debug e verifica manuale)

USO: python demo_fase2.py <s1,s2,...> <alpha1,alpha2,...> [n_reps]
Es:  python demo_fase2.py 3,5,7 0.5
     python demo_fase2.py 5 0.3,0.7,1.0
"""
import os, sys, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nsynth import CachedMelDataset, mel_to_audio
from models.encoder import MelEncoder
from models.unet import MelUNet
from models.sami import SAMI
from models.losses.diffusion import DiffusionSchedule
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

CKPT = "checkpoints/nsynth/sami_d32/model_final.pt"
DENOISER = "checkpoints/nsynth/denoiser_2d/model_final.pt"
OUT = "plots/demo"
N_REF = 8000
N_STEPS = 200
PITCH_A, PITCH_B = 60, 67
FAM_A, FAM_B = "guitar", "brass"
CONF_THRESH = 0.5
GUITAR_LABEL = 1  # LabelEncoder su [brass, guitar, keyboard, string] → 0,1,2,3

SCALES = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [3.0, 5.0, 7.0]
ALPHAS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5]
N_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SEED_BASE = 1000


def crepe_f0(wav, sr=16000):
    """CREPE f0: (f0 mediana, confidenza media) dei frame con conf > soglia."""
    import torchcrepe
    audio = wav.reshape(1, -1)
    out = torchcrepe.predict(
        audio, sr, hop_length=256, fmin=50, fmax=2000,
        model="full", batch_size=64, device="cuda", return_periodicity=True)
    if isinstance(out, tuple):
        f0, conf = out[0], out[1]
    else:
        f0, conf = out, torch.ones_like(out)
    f0_np = f0[0].cpu().numpy()
    conf_np = conf[0].cpu().numpy()
    mask = ~np.isnan(f0_np) & (conf_np > CONF_THRESH)
    if mask.sum() < 5:
        return float("nan"), 0.0
    return float(np.median(f0_np[mask])), float(np.mean(conf_np[mask]))


def generate(sami, mu_v, seed, guidance_scale, n_steps=N_STEPS):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    with torch.no_grad():
        xgen = sami.sample_seeded(
            torch.from_numpy(mu_v).unsqueeze(0).to(next(sami.encoder.parameters()).device),
            (1, 128, 256), n_steps=n_steps, generator=g,
            guidance_scale=guidance_scale)
    return xgen


def classify_output_timbre(encoder, knn, mu_mean, mu_std, xgen, family_ref):
    """Ri-encoda il mel GENERATO e classifica il suo μ (timbro sull'output).
    Ritorna il voto di MAGGIORANZA dei 5 vicini (conteggi per classe)."""
    with torch.no_grad():
        mu_gen, _ = encoder(xgen)
    mu_v = (mu_gen.cpu().numpy()[0] - mu_mean) / mu_std
    _, inds = knn.kneighbors(mu_v.reshape(1, -1))
    votes = [family_ref[i] for i in inds[0]]
    counts = {c: votes.count(c) for c in set(votes)}
    majority = max(counts, key=counts.get)
    return majority, votes, counts


def torchaudio_save(wav, path):
    import torchaudio
    torchaudio.save(path, wav.cpu().reshape(1, -1), 16000)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  A={FAM_A}/{PITCH_A}  B={FAM_B}/{PITCH_B}  "
          f"s∈{SCALES}  α∈{ALPHAS}  semi={N_REPS}  timbro=OUTPUT majority")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    encoder = MelEncoder(in_channels=1, latent_dim=32, base_channels=64,
                         input_size=(128, 256)).to(device)
    sd = {k[len("encoder."):]: v for k, v in ckpt["model_state_dict"].items()
          if k.startswith("encoder.")}
    encoder.load_state_dict(sd, strict=True)
    encoder.eval()

    denoiser = MelUNet(in_channels=1, base_channels=128,
                       channel_mult=(1, 1, 2, 2), time_dim=128).to(device)
    d_ckpt = torch.load(DENOISER, map_location=device, weights_only=False)
    denoiser.load_state_dict(d_ckpt["model_state_dict"])
    for p in denoiser.parameters():
        p.requires_grad = False
    denoiser.eval()

    diffusion = DiffusionSchedule(T=1000, s=0.008).to(device)
    sami = SAMI(encoder, denoiser, diffusion, beta=1e-5,
                frozen_denoiser=True, oversample_t=True, free_bits=0.5).to(device)

    ds = CachedMelDataset()
    metas = ds.metas

    rng = np.random.default_rng(42)
    idx_ref = rng.choice(len(ds), N_REF, replace=False)
    mu_ref = []
    with torch.no_grad():
        for b in range(0, N_REF, 256):
            x = torch.from_numpy(np.asarray(ds.arr[idx_ref[b:b + 256]]).copy()).to(device)
            mu, _ = encoder(x)
            mu_ref.append(mu.cpu().numpy())
    mu_ref = np.concatenate(mu_ref)
    pitch_ref = np.array([metas[i]["pitch"] for i in idx_ref], dtype=float)
    family_ref = LabelEncoder().fit_transform([metas[i]["instrument_family"] for i in idx_ref])

    mu_mean = mu_ref.mean(0)
    mu_std = mu_ref.std(0) + 1e-8
    mu_s = (mu_ref - mu_mean) / mu_std

    lo, hi = np.percentile(pitch_ref, [33, 67])
    w_pitch = mu_s[pitch_ref >= hi].mean(0) - mu_s[pitch_ref <= lo].mean(0)
    w_pitch = w_pitch / (np.linalg.norm(w_pitch) + 1e-12)

    knn = NearestNeighbors(n_neighbors=5).fit(mu_s)
    label_names = {0: "brass", 1: "guitar", 2: "keyboard", 3: "string"}

    def find(family, pitch):
        for i, m in enumerate(metas):
            if m["instrument_family"] == family and m["pitch"] == pitch:
                return i
    iA = find(FAM_A, PITCH_A)
    iB = find(FAM_B, PITCH_B)
    print(f"[INFO] A: idx={iA} {metas[iA]['instrument_family']}/{metas[iA]['pitch']}  "
          f"B: idx={iB} {metas[iB]['instrument_family']}/{metas[iB]['pitch']}")

    xA = torch.from_numpy(np.asarray(ds.arr[iA]).copy()).unsqueeze(0).to(device)
    xB = torch.from_numpy(np.asarray(ds.arr[iB]).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        muA, _ = encoder(xA)
        muB, _ = encoder(xB)
    muA_s = (muA.cpu().numpy()[0] - mu_mean) / mu_std
    muB_s = (muB.cpu().numpy()[0] - mu_mean) / mu_std
    muA_v = muA.cpu().numpy()[0]
    muB_v = muB.cpu().numpy()[0]

    comp_A = np.dot(muA_s, w_pitch) * w_pitch
    comp_B = np.dot(muB_s, w_pitch) * w_pitch

    os.makedirs(OUT, exist_ok=True)

    for s in SCALES:
        for alpha in ALPHAS:
            mu_new_s = muA_s - comp_A + alpha * comp_B
            mu_new_v = mu_new_s * mu_std + mu_mean
            tag = f"s{s}_a{alpha}"
            print(f"\n{'='*70}\n=== {tag} ===")
            print(f"    {'seed':>5s}  {'f0_A':>8s}  {'f0_T':>8s}  {'f0_B':>8s}  "
                  f"{'TimbroT':>9s}  {'voti':>25s}  {'A≠B':>5s}")
            rows = []
            for rep in range(N_REPS):
                seed = SEED_BASE + rep
                xgenA = generate(sami, muA_v, seed, s)
                xgenT = generate(sami, mu_new_v, seed, s)
                xgenB = generate(sami, muB_v, seed, s)
                wavA = mel_to_audio(torch.from_numpy(xgenA[0, 0].cpu().numpy()).unsqueeze(0))
                wavT = mel_to_audio(torch.from_numpy(xgenT[0, 0].cpu().numpy()).unsqueeze(0))
                wavB = mel_to_audio(torch.from_numpy(xgenB[0, 0].cpu().numpy()).unsqueeze(0))
                fA, _ = crepe_f0(wavA)
                fT, _ = crepe_f0(wavT)
                fB, _ = crepe_f0(wavB)
                maj, votes, counts = classify_output_timbre(
                    encoder, knn, mu_mean, mu_std, xgenT, family_ref)
                rows.append((seed, fA, fT, fB, maj, votes))
                torchaudio_save(wavA, os.path.join(OUT, f"{tag}_seed{seed}_A.wav"))
                torchaudio_save(wavT, os.path.join(OUT, f"{tag}_seed{seed}_T.wav"))
                torchaudio_save(wavB, os.path.join(OUT, f"{tag}_seed{seed}_B.wav"))
                ab_ok = (not np.isnan(fA)) and (not np.isnan(fB)) and (abs(fA - fB) > 20)
                votes_str = str([label_names.get(v, str(v)) for v in votes])
                print(f"    {seed:5d}  {fA:8.1f}  {fT:8.1f}  {fB:8.1f}  "
                      f"{label_names.get(maj, str(maj)):>9s}  {votes_str:>25s}  {'Y' if ab_ok else 'N':>5s}")

            # Conteggio onesto: X su Y semi VALIDI
            valid = [r for r in rows if not (np.isnan(r[1]) or np.isnan(r[2]) or np.isnan(r[3]))]
            ok = 0
            for (seed, fA, fT, fB, maj, votes) in valid:
                pitch_moved = abs(fT - fB) < abs(fT - fA)
                timbre_ok = (maj == GUITAR_LABEL)
                if pitch_moved and timbre_ok:
                    ok += 1
            print(f"  → SUCCESSO: {ok}/{len(valid)} semi validi  "
                  f"(di {N_REPS} totali; {N_REPS - len(valid)} con B/A non misurabili)")

    print(f"\nWAV in {OUT}/")


if __name__ == "__main__":
    main()
