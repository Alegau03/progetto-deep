"""
SAMI-Audio — Evaluate (Phase 1: Toy Model)
============================================
Evaluates the trained SAMI toy model on the synthetic sinusoid dataset.

Computes:
- Scatter plot of the 2D latent space colored by frequency and amplitude
- MIG (Mutual Information Gap) for both factors
- Gate 1 go/no-go decision

Usage:
    .venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder import ToyEncoder
from models.unet import ToyUNet
from models.losses.diffusion import DiffusionSchedule
from models.sami import SAMI
from train import generate_toy_dataset, SinusoidDataset


# ---------------------------------------------------------------------------
# MIG computation
# ---------------------------------------------------------------------------

def compute_mig(
    z: np.ndarray, labels: np.ndarray, n_bins: int = 20
) -> float:
    """
    Compute Mutual Information Gap (MIG) for a single factor.

    Implementation follows Chen et al. (2018) and the FactorVAE paper:
    1. For each latent dimension, estimate I(z_j; v) via discretization.
    2. MIG = (I_max - I_second) / H(v), where H(v) = log(n_classes).

    Parameters
    ----------
    z : (N, D) latent codes.
    labels : (N,) ground-truth factor labels (discrete).
    n_bins : int
        Number of bins for discretizing continuous z_j.

    Returns
    -------
    mig : float in [0, 1].
    """
    N, D = z.shape
    mi_scores = np.zeros(D)

    for j in range(D):
        z_j = z[:, j]
        if z_j.std() == 0:
            mi_scores[j] = 0.0
            continue
        bins = np.percentile(z_j, np.linspace(0, 100, n_bins + 1)[1:-1])
        z_disc = np.digitize(z_j, bins)
        mi_scores[j] = mutual_info_regression(
            z_disc.reshape(-1, 1), labels, discrete_features=True
        )[0]

    sorted_idx = np.argsort(mi_scores)[::-1]
    n_classes = len(np.unique(labels))
    h_v = np.log(n_classes) if n_classes > 1 else 1.0  # type: ignore[assignment]

    mig = (mi_scores[sorted_idx[0]] - mi_scores[sorted_idx[1]]) / h_v
    return max(0.0, float(mig))


# ---------------------------------------------------------------------------
# Latent space visualization
# ---------------------------------------------------------------------------

def plot_latent_space(
    z: np.ndarray,
    freq_labels: np.ndarray,
    amp_labels: np.ndarray,
    save_dir: str = "plots",
) -> None:
    """
    Create scatter plots of the 2D latent space colored by ground-truth factors.

    Generates:
    - plots/toy_latent_freq.png  : colored by frequency class
    - plots/toy_latent_amp.png   : colored by amplitude class
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scatter_freq = axes[0].scatter(z[:, 0], z[:, 1], c=freq_labels, cmap="viridis", s=1, alpha=0.5)
    axes[0].set_xlabel("Latent dim 0")
    axes[0].set_ylabel("Latent dim 1")
    axes[0].set_title("Colored by Frequency")
    plt.colorbar(scatter_freq, ax=axes[0], label="Frequency class")

    scatter_amp = axes[1].scatter(z[:, 0], z[:, 1], c=amp_labels, cmap="plasma", s=1, alpha=0.5)
    axes[1].set_xlabel("Latent dim 0")
    axes[1].set_ylabel("Latent dim 1")
    axes[1].set_title("Colored by Amplitude")
    plt.colorbar(scatter_amp, ax=axes[1], label="Amplitude class")

    plt.tight_layout()
    path = os.path.join(save_dir, "toy_latent_space.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO]  Latent space plot saved: {path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_toy(checkpoint_path: str, output_dir: str = "plots") -> dict:
    """
    Load trained model, encode the full toy dataset, compute MIG and plot.

    Returns a dict with evaluation results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    # ---- Load checkpoint ----------------------------------------------------
    print(f"[INFO]  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    encoder = ToyEncoder(
        in_channels=1,
        latent_dim=config["latent_dim"],
        base_channels=config["base_channels"],
        signal_length=config["signal_length"],
    )
    denoiser = ToyUNet(
        in_channels=1,
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
    )
    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
    sami = SAMI(encoder, denoiser, diffusion, beta=config["beta"])

    sami.load_state_dict(ckpt["model_state_dict"])
    sami.to(device)
    sami.eval()
    epoch = ckpt.get("epoch", "unknown")
    print(f"[INFO]  Model loaded (epoch {epoch}), params: {sum(p.numel() for p in sami.parameters()):,}")

    # ---- Data ---------------------------------------------------------------
    signals, freq_labels, amp_labels = generate_toy_dataset(
        n_freqs=config["n_freqs"],
        n_amps=config["n_amps"],
        n_per_class=config["n_per_class"],
        signal_length=config["signal_length"],
    )
    dataset = SinusoidDataset(signals, freq_labels, amp_labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # ---- Encode all samples ------------------------------------------------
    all_z: list[np.ndarray] = []
    all_freq: list[int] = []
    all_amp: list[int] = []

    print("[INFO]  Encoding dataset...")
    with torch.no_grad():
        for x, f, a in tqdm(loader, desc="Encoding"):
            x = x.to(device)
            mu, _ = sami.encoder(x)
            all_z.append(mu.cpu().numpy())
            all_freq.extend(f.tolist())
            all_amp.extend(a.tolist())

    z = np.concatenate(all_z, axis=0)
    freq_labels_arr = np.array(all_freq)
    amp_labels_arr = np.array(all_amp)
    print(f"[INFO]  Encoded {z.shape[0]} samples → latent shape {z.shape}")

    # ---- MIG ------------------------------------------------------------------
    print("[INFO]  Computing MIG...")
    mig_freq = compute_mig(z, freq_labels_arr)
    mig_amp = compute_mig(z, amp_labels_arr)
    print(f"  MIG (frequency)   = {mig_freq:.4f}")
    print(f"  MIG (amplitude)   = {mig_amp:.4f}")

    # ---- Plot -----------------------------------------------------------------
    if config["latent_dim"] == 2:
        plot_latent_space(z, freq_labels_arr, amp_labels_arr, output_dir)
    else:
        print(f"[WARN]  Latent dim = {config['latent_dim']}, skipping 2D scatter (only works for D=2)")

    # ---- Gate 1 --------------------------------------------------------------
    results = {"mig_freq": mig_freq, "mig_amp": mig_amp}

    print()
    print("=" * 60)
    if mig_freq > 0.5 and mig_amp > 0.5:
        print("  GATE 1: PASSED ✓")
        print(f"  MIG freq = {mig_freq:.3f} > 0.5, MIG amp = {mig_amp:.3f} > 0.5")
        print("  Disentanglement funziona. Pronto per Phase 2 (β-VAE su NSynth).")
    else:
        print("  GATE 1: FAILED ✗")
        print(f"  MIG freq = {mig_freq:.3f}, MIG amp = {mig_amp:.3f}")
        print("  Controlla il meccanismo di guidance (sami.py: _log_q_and_grad)")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAMI-Audio — Toy Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    evaluate_toy(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
