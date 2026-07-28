"""
SAMI-Audio — Diffusion Utilities
=================================
Cosine noise schedule and forward diffusion (q_sample) for DDPM.

Implements the cosine schedule from Nichol & Dhariwal (2021) with
s=0.008 offset for numerical stability near t=0.

Reference:
    Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion
    Probabilistic Models. arXiv:2102.09672.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    """
    Cosine noise schedule for DDPM training.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps. Default 1000.
    s : float
        Offset for cosine schedule. Default 0.008.
    beta_clip : float
        Maximum beta value to clip. Default 0.999.
    """

    def __init__(self, T: int = 1000, s: float = 0.008, beta_clip: float = 0.999) -> None:
        super().__init__()
        self.T = T
        self.s = s

        betas = self._cosine_beta_schedule(T, s, beta_clip)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float, clip: float) -> torch.Tensor:
        """Compute cosine beta schedule (Nichol & Dhariwal 2021)."""
        steps = torch.arange(T + 1, dtype=torch.float64)
        t_s = (steps / T + s) / (1.0 + s) * (torch.pi / 2.0)
        alpha_bar = torch.cos(t_s) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, max=clip).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε.

        Parameters
        ----------
        x0 : (B, C, L) clean data.
        t : (B,) or (B, 1) integer timesteps in [0, T-1].
        eps : (B, C, L) Gaussian noise.

        Returns
        -------
        xt : (B, C, L) noised data.
        """
        t_idx = t.long().clamp(0, self.T - 1)
        sqrt_alpha_cumprod = self.alphas_cumprod[t_idx].sqrt()
        sqrt_one_minus = (1.0 - self.alphas_cumprod[t_idx]).sqrt()

        while sqrt_alpha_cumprod.dim() < x0.dim():
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha_cumprod * x0 + sqrt_one_minus * eps

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Guidance weight: γ_t = √(1 - ᾱ_t).

        Parameters
        ----------
        t : (B,) integer timesteps.

        Returns
        -------
        gamma_t : (B,) per-sample guidance weight.
        """
        t_idx = t.long().clamp(0, self.T - 1)
        return (1.0 - self.alphas_cumprod[t_idx]).sqrt()

    def extra_repr(self) -> str:
        return f"T={self.T}, s={self.s}"
