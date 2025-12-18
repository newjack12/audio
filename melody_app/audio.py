from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ------------------------------------------------------------
# ffmpeg discovery
# ------------------------------------------------------------
def _find_ffmpeg() -> str:
    """Find ffmpeg executable.

    Priority:
      1) ffmpeg in PATH
      2) ffmpeg inside conda env prefix
    """
    p = shutil.which("ffmpeg")
    if p:
        return p

    prefix = Path(sys.prefix)
    candidates = [
        prefix / "Library" / "bin" / "ffmpeg.exe",
        prefix / "Scripts" / "ffmpeg.exe",
        prefix / "bin" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "找不到 ffmpeg。请确认已安装 ffmpeg，并且运行时使用的是正确的 conda 环境。\n"
        "推荐：conda install -c conda-forge ffmpeg"
    )


def _ffmpeg_supports_soxr(ffmpeg: str) -> bool:
    """Check whether ffmpeg supports soxr resampler."""
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "aresample" in r.stdout and "soxr" in r.stdout
    except Exception:
        return False


# ------------------------------------------------------------
# Preprocess spec
# ------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessSpec:
    """Audio preprocessing spec (industrial-grade defaults)."""

    sr: int = 44100
    highpass_hz: int = 80
    lowpass_hz: Optional[int] = None

    # Light denoise (optional, good for phone recordings)
    denoise: bool = False
    denoise_strength: float = 12.0  # afftdn: ~0–30

    # Loudness normalization (very important for model stability)
    loudnorm: bool = True
    loudnorm_i: float = -16.0
    loudnorm_tp: float = -1.5
    loudnorm_lra: float = 11.0

    # Optional silence trimming
    trim_silence: bool = False
    trim_db: float = -50.0

    # Preferred resampler (will auto-fallback if unavailable)
    resampler: Optional[str] = "soxr"


# ------------------------------------------------------------
# Preprocess pipeline
# ------------------------------------------------------------
def preprocess_to_wav(
    input_audio: Path,
    output_wav: Path,
    spec: PreprocessSpec,
) -> None:
    """Preprocess any audio into mono 16-bit PCM WAV via ffmpeg."""
    ffmpeg = _find_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    chain = []

    # --------------------------------------------------------
    # Resampler (industrial-safe)
    # --------------------------------------------------------
    if spec.resampler == "soxr":
        if _ffmpeg_supports_soxr(ffmpeg):
            chain.append("aresample=resampler=soxr")
        else:
            # Fallback silently (this is the KEY fix)
            print("[WARN] ffmpeg 不支持 soxr，已回退到默认重采样器")

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------
    if spec.highpass_hz and spec.highpass_hz > 0:
        chain.append(f"highpass=f={spec.highpass_hz}")

    if spec.lowpass_hz and spec.lowpass_hz > 0:
        chain.append(f"lowpass=f={spec.lowpass_hz}")

    if spec.denoise:
        strength = float(max(0.0, min(30.0, spec.denoise_strength)))
        chain.append(f"afftdn=nf={strength}")

    if spec.loudnorm:
        chain.append(
            f"loudnorm=I={spec.loudnorm_i}:"
            f"TP={spec.loudnorm_tp}:"
            f"LRA={spec.loudnorm_lra}"
        )

    if spec.trim_silence:
        db = float(spec.trim_db)
        chain.append(
            f"silenceremove=start_periods=1:start_threshold={db}dB:"
            f"stop_periods=1:stop_threshold={db}dB"
        )

    # --------------------------------------------------------
    # ffmpeg command
    # --------------------------------------------------------
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",  # mono
        "-ar",
        str(spec.sr),
        "-vn",
        "-acodec",
        "pcm_s16le",
    ]

    if chain:
        cmd += ["-af", ",".join(chain)]

    cmd += [str(output_wav)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "ffmpeg 预处理失败。\n"
            f"命令: {' '.join(cmd)}\n"
            f"stdout:\n{r.stdout}\n"
            f"stderr:\n{r.stderr}\n"
        )

    if not output_wav.exists() or output_wav.stat().st_size < 1000:
        raise RuntimeError("预处理输出 wav 生成失败或文件过小，请检查输入音频。")


# ------------------------------------------------------------
# WAV IO helpers
# ------------------------------------------------------------
def read_wav_mono_16bit(path: Path) -> Tuple[int, np.ndarray]:
    """Read 16-bit PCM WAV (mono or stereo) into float32 [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(
            f"当前只支持 16-bit PCM wav。实际 sampwidth={sampwidth}。"
            "请用 preprocess_to_wav 先转换。"
        )

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return sr, audio


def write_wav_mono_16bit(path: Path, sr: int, audio: np.ndarray) -> None:
    """Write float32 [-1, 1] to 16-bit PCM WAV (mono)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
