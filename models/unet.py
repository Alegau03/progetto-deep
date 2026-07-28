"""
SAMI-Audio — U-Net Denoiser
=============================
1D U-Net denoiser for the toy sinusoid model.

Architecture: 3 down blocks, mid, 3 up blocks with skip connections.
Sinusoidal time embeddings are injected into every ResBlock.

For Phase 1 (toy): 1D convolutions on 1D signals.
For Phase 3 (NSynth): will be replaced with 2D U-Net.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (DDPM-style) followed by a small MLP."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        freqs = freqs.to(t.device)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# ResBlock with time conditioning
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """
    1D residual block with GroupNorm, SiLU, and optional time conditioning.

    Time embedding is projected and added as a bias after the first normalization.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 128, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# 1D U-Net
# ---------------------------------------------------------------------------

class ToyUNet(nn.Module):
    """
    1D U-Net denoiser for toy sinusoid generation.

    Parameters
    ----------
    in_channels : int
        Input channels. Default 1.
    base_channels : int
        Base channel count. Default 32.
    channel_mult : tuple
        Channel multipliers per level. Default (1, 2, 2) for 3 levels.
    time_dim : int
        Time embedding dimension. Default 128.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mult: tuple = (1, 2, 2),
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        chs = [base_channels * m for m in channel_mult]
        n_levels = len(chs)

        self.input_conv = nn.Conv1d(in_channels, chs[0], kernel_size=3, padding=1, bias=False)

        self.down_blocks = nn.ModuleList()
        self.down_chs = [chs[0]]
        for i in range(n_levels):
            in_ch = chs[i - 1] if i > 0 else chs[0]
            out_ch = chs[i]
            self.down_blocks.append(
                nn.ModuleList([
                    ResBlock1D(in_ch, out_ch, time_dim, stride=2),
                    ResBlock1D(out_ch, out_ch, time_dim, stride=1),
                ])
            )
            self.down_chs.append(out_ch)

        self.mid_block1 = ResBlock1D(chs[-1], chs[-1], time_dim, stride=1)
        self.mid_block2 = ResBlock1D(chs[-1], chs[-1], time_dim, stride=1)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.down_chs[i]
            out_ch = chs[max(i - 1, 0)] if i > 0 else chs[0]
            self.up_blocks.append(
                nn.ModuleList([
                    ResBlock1D(chs[i] + skip_ch, out_ch, time_dim, stride=1),
                    ResBlock1D(out_ch, out_ch, time_dim, stride=1),
                ])
            )

        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv1d(chs[0], in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise component ε given noisy input x_t and timestep t.

        Parameters
        ----------
        x : (B, C, L) noisy data.
        t : (B,) integer timesteps.

        Returns
        -------
        eps : (B, C, L) predicted noise.
        """
        t_emb = self.time_embed(t)

        h = self.input_conv(x)
        skips = [h]

        for block1, block2 in self.down_blocks:
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for (block1, block2), skip in zip(self.up_blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
