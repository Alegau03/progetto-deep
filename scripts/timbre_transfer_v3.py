#!/usr/bin/env python3
"""
Timbre transfer v3 — confronto appaiato con seme condiviso.

Correzioni rispetto a v2:
  1. SEME x_T condiviso tra A/B/transfer (stesso rumore iniziale → la
     differenza di pitch è attribuibile a z, non al rumore). Il confronto
     appaiato, di nuovo.
  2. DDIM 200 step invece di 50 (generazione più vincolata dalla guidance).
  3. N ripetizioni con semi diversi; mediana dei soli run ad alta confidenza
     CREPE (>0.5).
  4. α ∈ {0.3, 0.5, 0.7, 1.0} — la calibrazione è verso il basso (α=2 era
     overshoot: il transfer atterrava sotto B).

Report: triplette appaiate (A, transfer, B) per seme, non solo mediane.
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

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/nsynth/sami_t3/sami_epoch_0008.pt"
DENOISER = "checkpoints/nsynth/denoiser_2d/model_final.pt"
OUT = "plots/timbre_transfer_v3"
N_REF = 8000
N_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
N_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ALPHAS = [float(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.3, 0.5, 0.7, 1.0]
CONF_THRESH = 0.5


def crepe_f0(wav, sr=16000):
    """CREPE f0: ritorna (f0 mediana, confidenza media) dei frame > soglia."""
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


def generate(sami, mu_v, seed, n_steps=N_STEPS):
    """Genera con x_T determinato dal seed (stesso rumore iniziale per A/B/T)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    with torch.no_grad():
        xgen = sami.sample_seeded(
            torch.from_numpy(mu_v).unsqueeze(0).to(next(sami.encoder.parameters()).device),
            (1, 128, 256), n_steps=n_steps, generator=g)
    mel_t = torch.from_numpy(xgen[0, 0].cpu().numpy()).unsqueeze(0)
    return mel_to_audio(mel_t)


def torchaudio_save(wav, path):
    import torchaudio
    torchaudio.save(path, wav.cpu().reshape(1, -1), 16000)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  CKPT: {CKPT}  N_STEPS={N_STEPS}  N_REPS={N_REPS}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    encoder = MelEncoder(in_channels=1, latent_dim=128, base_channels=64,
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
    sami = SAMI(encoder, denoiser, diffusion, beta=2e-5,
                frozen_denoiser=True, oversample_t=True, free_bits=0.5).to(device)

    ds = CachedMelDataset()
    metas = ds.metas

    # Riferimento + standardizzazione + direzione di intervento (terzili)
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

    lo_thr, hi_thr = np.percentile(pitch_ref, [33, 67])
    w_pitch = mu_s[pitch_ref >= hi_thr].mean(0) - mu_s[pitch_ref <= lo_thr].mean(0)
    w_pitch = w_pitch / (np.linalg.norm(w_pitch) + 1e-12)
    print(f"[INFO] Direzione di intervento (terzili {lo_thr:.0f}/{hi_thr:.0f})")

    knn = NearestNeighbors(n_neighbors=5).fit(mu_s)

    def classify_timbre(mu_vec):
        mu_v = (mu_vec - mu_mean) / mu_std
        _, inds = knn.kneighbors(mu_v.reshape(1, -1))
        return [family_ref[i] for i in inds[0]]

    def find(family, pitch):
        for i, m in enumerate(metas):
            if m["instrument_family"] == family and m["pitch"] == pitch:
                return i
    iA = find("guitar", 60)
    iB = find("brass", 72)
    print(f"[INFO] A: {metas[iA]['instrument_family']}/{metas[iA]['pitch']}  "
          f"B: {metas[iB]['instrument_family']}/{metas[iB]['pitch']}")

    xA = torch.from_numpy(np.asarray(ds.arr[iA]).copy()).unsqueeze(0).to(device)
    xB = torch.from_numpy(np.asarray(ds.arr[iB]).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        muA, _ = encoder(xA)
        muB, _ = encoder(xB)
    muA_s = (muA.cpu().numpy()[0] - mu_mean) / mu_std
    muB_s = (muB.cpu().numpy()[0] - mu_mean) / mu_std
    muA_v = muA.cpu().numpy()[0]
    muB_v = muB.cpu().numpy()[0]

    os.makedirs(OUT, exist_ok=True)

    for alpha in ALPHAS:
        comp_A = np.dot(muA_s, w_pitch) * w_pitch
        comp_B = np.dot(muB_s, w_pitch) * w_pitch
        mu_new_s = muA_s - comp_A + alpha * comp_B
        mu_new_v = mu_new_s * mu_std + mu_mean
        print(f"\n=== α={alpha}  (Δ pitch = {alpha*np.dot(muB_s, w_pitch) - np.dot(muA_s, w_pitch):.3f}) ===")
        print(f"    {'seed':>5s}  {'f0_A':>8s}  {'f0_T':>8s}  {'f0_B':>8s}  {'conf_A':>6s}  {'conf_T':>6s}  {'conf_B':>6s}  timbro_T")
        rows = []
        for rep in range(N_REPS):
            seed = 1000 + rep
            wavA = generate(sami, muA_v, seed)
            wavT = generate(sami, mu_new_v, seed)
            wavB = generate(sami, muB_v, seed)
            fA, cA = crepe_f0(wavA)
            fT, cT = crepe_f0(wavT)
            fB, cB = crepe_f0(wavB)
            timbre = classify_timbre(mu_new_v)
            rows.append((fA, fT, fB, cA, cT, cB))
            print(f"    {seed:5d}  {fA:8.1f}  {fT:8.1f}  {fB:8.1f}  {cA:6.2f}  {cT:6.2f}  {cB:6.2f}  {timbre}")
            torchaudio_save(wavT, os.path.join(OUT, f"alpha{alpha}_seed{seed}_T.wav"))

        # Mediana dei soli run ad alta confidenza (tutte e tre le condizioni)
        ok = [r for r in rows if r[3] > CONF_THRESH and r[4] > CONF_THRESH and r[5] > CONF_THRESH]
        if ok:
            med = lambda i: float(np.median([r[i] for r in ok]))
            print(f"    MEDIANA (n={len(ok)}): f0_A={med(0):.1f}  f0_T={med(1):.1f}  f0_B={med(2):.1f}")
        else:
            print(f"    Nessun run ad alta confidenza su tutte le condizioni")

    print(f"\nWAV in {OUT}/")


if __name__ == "__main__":
    main()
