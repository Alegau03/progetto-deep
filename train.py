"""
SAMI-Audio — Train (Phase 1: Toy Model)
=========================================
Generates synthetic sinusoid dataset and trains the SAMI toy model
to validate the guidance mechanism in isolation.

Usage:
    .venv/bin/python train.py                     # CPU or auto-detect GPU
    .venv/bin/python train.py --use-wandb         # with Weights & Biases logging
    .venv/bin/python train.py --epochs 100 --batch-size 256
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.encoder import ToyEncoder
from models.unet import ToyUNet
from models.losses.diffusion import DiffusionSchedule
from models.sami import SAMI


# ---------------------------------------------------------------------------
# Toy dataset: synthetic sinusoids
# ---------------------------------------------------------------------------

def generate_toy_dataset(
    n_freqs: int = 10,
    n_amps: int = 5,
    n_per_class: int = 1000,
    signal_length: int = 256,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Generate synthetic 1D sinusoids with known frequency and amplitude labels.

    Each sample: y(t) = A · sin(2π · f · t / L) + N(0, noise_std)

    Returns
    -------
    signals : (N, 1, L) tensor in approximate range [-1, 1].
    freq_labels : (N,) integer frequency class index [0, n_freqs).
    amp_labels : (N,) integer amplitude class index [0, n_amps).
    """
    rng = np.random.default_rng(seed)
    L = signal_length
    t = np.arange(L)

    frequencies = np.linspace(1.0, 20.0, n_freqs)
    amplitudes = np.linspace(0.2, 1.0, n_amps)

    signals_list: list[np.ndarray] = []
    freq_label_list: list[int] = []
    amp_label_list: list[int] = []

    for fi, f in enumerate(frequencies):
        for ai, A in enumerate(amplitudes):
            for _ in range(n_per_class):
                phase = rng.uniform(0, 2 * np.pi)
                y = A * np.sin(2.0 * np.pi * f * t / L + phase)
                y += rng.normal(0, noise_std, size=L)
                signals_list.append(y)
                freq_label_list.append(fi)
                amp_label_list.append(ai)

    signals = np.stack(signals_list, axis=0)
    signals = signals[:, np.newaxis, :].astype(np.float32)

    return torch.from_numpy(signals), np.array(freq_label_list), np.array(amp_label_list)


class SinusoidDataset(Dataset):
    """Simple Dataset wrapper for toy sinusoid data."""

    def __init__(
        self,
        signals: torch.Tensor,
        freq_labels: np.ndarray,
        amp_labels: np.ndarray,
    ) -> None:
        self.signals = signals
        self.freq_labels = freq_labels
        self.amp_labels = amp_labels

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return self.signals[idx], self.freq_labels[idx], self.amp_labels[idx]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_toy(config: dict) -> tuple[SAMI, dict]:
    """
    Train the SAMI toy model on synthetic sinusoids.

    Returns the trained model and a dict of training metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    # ---- Data ---------------------------------------------------------------
    print("[INFO]  Generating toy dataset...")
    signals, freq_labels, amp_labels = generate_toy_dataset(
        n_freqs=config["n_freqs"],
        n_amps=config["n_amps"],
        n_per_class=config["n_per_class"],
        signal_length=config["signal_length"],
    )
    dataset = SinusoidDataset(signals, freq_labels, amp_labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # ---- Models -------------------------------------------------------------
    signal_length = config["signal_length"]
    latent_dim = config["latent_dim"]

    encoder = ToyEncoder(
        in_channels=1,
        latent_dim=latent_dim,
        base_channels=config["base_channels"],
        signal_length=signal_length,
    )
    denoiser = ToyUNet(
        in_channels=1,
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
    )
    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
    sami = SAMI(encoder, denoiser, diffusion, beta=config["beta_start"])

    encoder.to(device)
    denoiser.to(device)
    diffusion.to(device)
    sami.to(device)

    total_params = sum(p.numel() for p in sami.parameters())
    print(f"[INFO]  Total parameters: {total_params:,}")

    # ---- Optimizer ----------------------------------------------------------
    optimizer = torch.optim.Adam(sami.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(loader)
    )

    # ---- W&B ------------------------------------------------------------------
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="sami-audio-toy", config=config)
        print("[INFO]  wandb logging enabled")

    # ---- KL annealing schedule -------------------------------------------------
    beta_start = config["beta_start"]
    beta_end = config["beta_end"]
    beta_warmup = config["beta_warmup"]
    print(f"[INFO]  KL annealing: {beta_start:.1e} → {beta_end:.1e} over {beta_warmup} epochs (exponential)")

    # ---- Training loop ------------------------------------------------------
    sami.train()
    stats: dict[str, list[float]] = {"loss": [], "L_x": [], "L_z": []}
    global_step = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        # Exponential KL annealing
        if epoch <= beta_warmup:
            beta = beta_start * (beta_end / beta_start) ** (epoch / beta_warmup)
        else:
            beta = beta_end
        sami.beta = beta

        epoch_loss = 0.0
        epoch_lx = 0.0
        epoch_lz = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{config['epochs']}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)

            optimizer.zero_grad()
            loss, L_x, L_z, z = sami(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sami.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_lx += L_x.item()
            epoch_lz += L_z.item()
            global_step += 1

            stats["loss"].append(loss.item())
            stats["L_x"].append(L_x.item())
            stats["L_z"].append(L_z.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}", Lx=f"{L_x.item():.4f}", Lz=f"{L_z.item():.4f}")

            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/L_x": L_x.item(),
                    "train/L_z": L_z.item(),
                    "train/beta": beta,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = epoch_loss / len(loader)
        avg_lx = epoch_lx / len(loader)
        avg_lz = epoch_lz / len(loader)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}  L_x={avg_lx:.4f}  L_z={avg_lz:.4f}  β={beta:.2e}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint
        if epoch % config["checkpoint_every"] == 0:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"toy_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": sami.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "stats": stats,
            }, ckpt_path)
            print(f"  [CKPT] Saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": sami.state_dict(),
        "config": config,
        "stats": stats,
    }, final_path)
    print(f"[INFO]  Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()

    return sami, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAMI-Audio — Toy Model Training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta-start", type=float, default=1e-6, help="Initial KL weight")
    parser.add_argument("--beta-end", type=float, default=1.0, help="Target KL weight after warmup")
    parser.add_argument("--beta-warmup", type=int, default=10, help="Epochs of exponential KL annealing")
    parser.add_argument("--n-per-class", type=int, default=500, help="Samples per class (default 500)")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimension (keep 2 for toy)")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels")
    parser.add_argument("--time-dim", type=int, default=128, help="Time embedding dim")
    parser.add_argument("--T", type=int, default=200, help="Diffusion steps")
    parser.add_argument("--s", type=float, default=0.008, help="Cosine schedule offset")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/toy", help="Checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    config = {
        "n_freqs": 10,
        "n_amps": 5,
        "n_per_class": args.n_per_class,
        "signal_length": 256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_warmup": args.beta_warmup,
        "latent_dim": args.latent_dim,
        "base_channels": args.base_channels,
        "time_dim": args.time_dim,
        "T": args.T,
        "s": args.s,
        "grad_clip": args.grad_clip,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_every": args.checkpoint_every,
        "use_wandb": args.use_wandb,
    }

    print("=" * 60)
    print("  SAMI-Audio — Phase 1: Toy Model Training")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:<20} = {v}")

    train_toy(config)


if __name__ == "__main__":
    main()
