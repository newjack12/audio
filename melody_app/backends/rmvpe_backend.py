from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

import torch
from rmvpe import RMVPE


@dataclass
class RmvpeConfig:
    hop_length: int = 160        # 10ms @ 16k
    f0_min: float = 50.0
    f0_max: float = 1100.0
    device: str = "cpu"          # "cuda" if available


_model: RMVPE | None = None


def _get_model(cfg: RmvpeConfig) -> RMVPE:
    global _model
    if _model is None:
        _model = RMVPE(
            model_path=None,   # use built-in pretrained
            device=cfg.device,
            is_half=False,
        )
    return _model


def extract_f0_rmvpe(
    wav_path: Path,
    cfg: RmvpeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times (s),
      f0_hz,
      voiced (bool),
      confidence (0~1)
    """
    wav, sr = sf.read(str(wav_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        raise ValueError("RMVPE expects 16kHz audio. Please preprocess to 16k.")

    model = _get_model(cfg)

    with torch.no_grad():
        f0, voiced, confidence = model.infer_from_audio(
            wav.astype(np.float32),
            sr=sr,
            hop_length=cfg.hop_length,
            f0_min=cfg.f0_min,
            f0_max=cfg.f0_max,
        )

    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.asarray(voiced, dtype=np.bool_)
    confidence = np.asarray(confidence, dtype=np.float32)

    times = np.arange(len(f0), dtype=np.float32) * (cfg.hop_length / sr)

    return times, f0, voiced, confidence
