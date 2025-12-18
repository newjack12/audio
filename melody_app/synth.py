from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .audio import read_wav_mono_16bit, write_wav_mono_16bit
from .midi_io import NoteEvent


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _midi_to_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


# ------------------------------------------------------------
# Simple harmonic synth (preview only)
# ------------------------------------------------------------
def synthesize_harmonic(
    notes: List[NoteEvent],
    sr: int,
    total_len: float,
    *,
    n_harmonics: int = 6,
) -> np.ndarray:
    """
    Lightweight harmonic synthesizer for preview.

    Notes
    -----
    - This is NOT for final audio, only for "did we get the melody right?"
    - Intentionally simple and dependency-free.
    """
    n = int(np.ceil(total_len * sr))
    y = np.zeros(n, dtype=np.float32)

    attack = int(0.008 * sr)
    release = int(0.03 * sr)

    for note in notes:
        f0 = _midi_to_freq(note.pitch)
        s = int(max(0, np.floor(note.start * sr)))
        e = int(min(n, np.ceil(note.end * sr)))
        if e <= s + 2:
            continue

        t = (np.arange(e - s, dtype=np.float32) / sr)
        amp = min(1.0, max(0.05, note.velocity / 127.0)) * 0.22

        sig = np.zeros_like(t, dtype=np.float32)
        for k in range(1, n_harmonics + 1):
            sig += (1.0 / k) * np.sin(2.0 * np.pi * (f0 * k) * t)

        # normalize per-note (avoid clipping)
        peak = max(1e-6, float(np.max(np.abs(sig))))
        sig *= (amp / peak)

        # simple ADSR-like envelope
        env = np.ones_like(sig, dtype=np.float32)
        if attack > 0 and len(env) > attack:
            env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release > 0 and len(env) > release:
            env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)

        y[s:e] += sig * env

    y = np.clip(y, -1.0, 1.0)
    return y


# ------------------------------------------------------------
# Preview rendering (INDUSTRIAL-SAFE)
# ------------------------------------------------------------
def render_previews(
    preprocessed_wav: Path,
    melody_notes: List[NoteEvent],
    out_dir: Path,
) -> None:
    """
    Render preview WAVs:
      - melody only
      - original + melody mix

    Industrial note:
    - Always align buffer lengths before mixing.
    - Off-by-one samples are NORMAL after resampling / loudnorm.
    """
    sr, orig = read_wav_mono_16bit(preprocessed_wav)
    total_len = len(orig) / sr

    mel = synthesize_harmonic(
        melody_notes,
        sr=sr,
        total_len=total_len,
    )

    # ---------- CRITICAL FIX: length alignment ----------
    n = min(len(orig), len(mel))
    if len(orig) != len(mel):
        # Optional debug log (kept silent by default)
        # print(f"[INFO] preview length aligned: orig={len(orig)}, mel={len(mel)} -> {n}")
        orig = orig[:n]
        mel = mel[:n]
    # ---------------------------------------------------

    out_mel = out_dir / "03_preview_melody.wav"
    out_mix = out_dir / "03_preview_mix.wav"

    write_wav_mono_16bit(out_mel, sr, mel)

    mix = (orig * 0.55) + (mel * 1.00)
    mix = np.clip(mix, -1.0, 1.0)
    write_wav_mono_16bit(out_mix, sr, mix)
