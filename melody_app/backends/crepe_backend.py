from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from ..audio import read_wav_mono_16bit
from ..utils.optional_import import require_module


@dataclass(frozen=True)
class CrepeConfig:
    """Configuration for CREPE pitch tracking."""

    step_size_ms: int = 10
    model_capacity: str = "full"  # tiny|small|medium|large|full
    viterbi: bool = True
    confidence_threshold: float = 0.5

    fmin: float = 50.0
    fmax: float = 1200.0


def extract_f0_crepe(
    wav_path: Path,
    cfg: CrepeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract monophonic f0 using CREPE.

    Returns
    -------
    times_s, f0_hz, voiced, confidence
    """
    require_module("crepe")
    import crepe  # type: ignore

    sr, audio = read_wav_mono_16bit(wav_path)

    if sr != 16000:
        raise ValueError(
            f"CREPE 需要 16000Hz 音频。当前 sr={sr}。\n"
            "请在 pipeline 里把预处理 sr 设置为 16000。"
        )

    audio = audio.astype(np.float32)

    time, frequency, confidence, _activation = crepe.predict(
        audio,
        sr,
        viterbi=bool(cfg.viterbi),
        step_size=int(cfg.step_size_ms),
        model_capacity=str(cfg.model_capacity),
    )

    time = np.asarray(time, dtype=np.float32)
    frequency = np.asarray(frequency, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)

    voiced = confidence >= float(cfg.confidence_threshold)

    if cfg.fmin:
        voiced &= frequency >= float(cfg.fmin)
    if cfg.fmax:
        voiced &= frequency <= float(cfg.fmax)

    f0 = frequency.copy()
    f0[~voiced] = 0.0

    return time, f0, voiced.astype(bool), confidence
