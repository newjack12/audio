from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from ..audio import read_wav_mono_16bit
from ..utils.optional_import import require_module


@dataclass(frozen=True)
class PyinConfig:
    fmin: float = 50.0
    fmax: float = 1200.0
    frame_length: int = 2048
    hop_length: int = 256
    voiced_prob_threshold: float = 0.5


def extract_f0_pyin(
    wav_path: Path,
    cfg: PyinConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract monophonic f0 using probabilistic YIN (pYIN) via librosa."""
    require_module("librosa")
    import librosa  # type: ignore

    sr, audio = read_wav_mono_16bit(wav_path)
    audio = audio.astype(np.float32)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=float(cfg.fmin),
        fmax=float(cfg.fmax),
        sr=int(sr),
        frame_length=int(cfg.frame_length),
        hop_length=int(cfg.hop_length),
    )

    f0 = np.asarray(f0, dtype=np.float32)
    voiced_probs = np.asarray(voiced_probs, dtype=np.float32)

    # librosa returns nan for unvoiced
    voiced = (
        np.asarray(voiced_flag, dtype=bool)
        & np.isfinite(f0)
        & (voiced_probs >= float(cfg.voiced_prob_threshold))
    )
    f0_clean = f0.copy()
    f0_clean[~voiced] = 0.0

    times = librosa.frames_to_time(
        np.arange(len(f0_clean)),
        sr=int(sr),
        hop_length=int(cfg.hop_length),
    ).astype(np.float32)

    return times, f0_clean, voiced, voiced_probs
