"""
SAMI-Audio — Models Module
==========================
Neural network architectures for SAMI-Audio.

Modules:
- encoder: Half-UNet inference network → (μ, σ²)
- unet: Denoiser (U-Net with sinusoidal time embedding)
- sami: Core SAMI model (guidance via autodiff + loss + sampling)
- baselines: β-VAE and FactorVAE baseline models
- losses/diffusion: Cosine noise schedule and diffusion utilities
- losses/disentanglement: KL divergence and disentanglement losses
"""
