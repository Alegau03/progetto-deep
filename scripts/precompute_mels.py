"""
SAMI-Audio — Mel Cache Precomputation
======================================
Pre-computes all NSynth mel spectrograms into a memory-mapped .npy file.
Eliminates the I/O bottleneck (~14 min/epoch) and makes 29-min SLURM jobs
viable for training on RTX 6000.

Run AFTER compute_norm_stats.py (needs data/norm_stats.json):
    python scripts/precompute_mels.py

Output:
    data/mel_cache.npy   — (N, 1, 128, 256) float32 memmap
    data/mel_meta.pkl    — per-sample metadata dicts
"""

import pickle
import numpy as np
from data.nsynth import NSynthDataset


def precompute(root: str = "data/nsynth-train",
               cache: str = "data/mel_cache.npy",
               meta_out: str = "data/mel_meta.pkl"):
    """Pre-calcola tutti i mel-spettrogrammi in un memmap.

    Elimina il collo di bottiglia I/O del training (il calcolo mel da .wav
    costava ~14 min/epoca): il CachedMelDataset legge direttamente dal
    memmap (~2-3 min/epoca). Va eseguito dopo compute_norm_stats.py.

    Parameters
    ----------
    root : str
        Cartella NSynth di input.
    cache : str
        Percorso del memmap di output (N, 1, 128, 256) float32.
    meta_out : str
        Percorso del pickle con i metadati (pitch, famiglia, ...).
    """
    dataset = NSynthDataset(root=root, normalize=True)
    N = len(dataset)
    print(f"Precomputing {N} mel spectrograms...")

    arr = np.lib.format.open_memmap(
        cache, mode="w+", dtype=np.float32, shape=(N, 1, 128, 256))
    metas = []

    for i in range(N):
        mel, meta = dataset[i]
        arr[i] = mel.numpy()
        metas.append(meta)
        if i % 5000 == 0:
            print(f"  {i}/{N}")

    arr.flush()
    pickle.dump(metas, open(meta_out, "wb"))
    print(f"Cache saved: {cache} ({N} samples)")


if __name__ == "__main__":
    precompute()
