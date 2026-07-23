"""
SAMI-Audio — Data Module
========================
Dataset loading and preprocessing for NSynth instrument sounds.

Provides:
- NSynthDataset: PyTorch Dataset for mel-spectrogram extraction
- mel_to_audio: Griffin-Lim reconstruction for sanity checks
- MelConfig: Immutable configuration for audio preprocessing parameters
"""

from data.nsynth import NSynthDataset, MelConfig, mel_to_audio

__all__ = ["NSynthDataset", "MelConfig", "mel_to_audio"]
