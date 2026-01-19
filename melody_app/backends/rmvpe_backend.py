from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

from .rmvpe.rmvpe import RMVPE


@dataclass
class RmvpeConfig:
    # 注意：RMVPE 通常固定 16k / hop=160（10ms）
    hop_length: int = 160
    f0_min: float = 50.0
    f0_max: float = 1100.0
    device: str = "auto"     # "auto" | "cpu" | "cuda"
    is_half: bool = False    # cuda 上可 True；不稳定就 False
    threshold: float = 0.03  # 越大越“保守”，越不容易误判为有声


_model: RMVPE | None = None


def _resolve_device(want: str) -> str:
    want = (want or "auto").lower()
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def _get_model(cfg: RmvpeConfig) -> RMVPE:
    global _model
    if _model is None:
        device = _resolve_device(cfg.device)
        # 默认从同目录加载 rmvpe.pt
        model_path = Path(__file__).parent / "rmvpe" / "rmvpe.pt"
        _model = RMVPE(
            model_path=model_path,
            device=device,
            is_half=bool(cfg.is_half),
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
    wav, sr = sf.read(str(wav_path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        raise ValueError("RMVPE backend 期望输入 wav 为 16kHz。请在 pipeline 里先 preprocess 到 16k。")

    model = _get_model(cfg)
    f0, conf = model.infer_from_audio(
        wav.astype(np.float32),
        sr=sr,
        thred=float(cfg.threshold),
    )

    # 限制频段（超出范围视为无声）
    f0 = np.asarray(f0, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    bad = (f0 < float(cfg.f0_min)) | (f0 > float(cfg.f0_max))
    if np.any(bad):
        f0[bad] = 0.0
        conf[bad] = 0.0

    voiced = (f0 > 0.0)
    times = (np.arange(len(f0), dtype=np.float32) * (float(cfg.hop_length) / float(sr))).astype(np.float32)
    return times, f0, voiced.astype(np.bool_), conf
