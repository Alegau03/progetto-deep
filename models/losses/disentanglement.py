"""
SAMI-Audio — Disentanglement Losses
====================================
Loss functions for disentangled representation learning.

- KL divergence: regularization toward unit Gaussian prior.
- Mahalanobis log-probability: used for the guidance score.
"""

from __future__ import annotations

import math

import torch


def kl_divergence(mu: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    KL divergence D_KL( N(μ, σ²) ‖ N(0, I) ), averaged over batch.

    Closed form for diagonal Gaussian:
        KL = 0.5 * Σ ( μ² + σ² - log(σ²) - 1 )

    Parameters
    ----------
    mu : (B, D) posterior mean.
    sigma2 : (B, D) posterior variance (> 0).

    Returns
    -------
    kl : scalar, mean over batch.
    """
    return 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1.0).sum(dim=1).mean()


def mahalanobis_log_prob(
    z: torch.Tensor, mu: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """
    Log-probability of z under N(mu, diag(sigma2)).

    log q(z) = -0.5 * Σ [ (z - μ)² / σ² + log(2π · σ²) ]

    Parameters
    ----------
    z : (B, D) latent sample.
    mu : (B, D) posterior mean.
    sigma2 : (B, D) posterior variance (> 0).

    Returns
    -------
    log_prob : (B,) per-sample log-probability.
    """
    D = z.shape[-1]
    return -0.5 * (
        ((z - mu).pow(2) / sigma2).sum(dim=1)
        + sigma2.log().sum(dim=1)
        + D * math.log(2.0 * math.pi)
    )
