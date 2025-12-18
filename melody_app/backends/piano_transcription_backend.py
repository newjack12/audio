from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..audio import read_wav_mono_16bit
from ..utils.optional_import import require_module


@dataclass(frozen=True)
class PianoTranscriptionConfig:
    device: str = "cpu"  # 'cpu' | 'cuda'
    checkpoint_path: Optional[str] = None


def run_piano_transcription_inference(
    wav_path: Path,
    out_midi: Path,
    cfg: PianoTranscriptionConfig,
) -> Path:
    """Run qiuqiangkong/piano_transcription_inference (PyTorch) to transcribe piano to MIDI."""
    require_module("piano_transcription_inference")

    from piano_transcription_inference import PianoTranscription, sample_rate  # type: ignore

    sr, audio = read_wav_mono_16bit(wav_path)
    if sr != int(sample_rate):
        raise ValueError(
            f"piano_transcription_inference 需要 {sample_rate}Hz 音频。当前 sr={sr}。\n"
            "请在 pipeline 里把预处理 sr 设置为 sample_rate。"
        )

    audio = audio.astype(np.float32)

    out_midi.parent.mkdir(parents=True, exist_ok=True)

    transcriptor = PianoTranscription(device=str(cfg.device), checkpoint_path=cfg.checkpoint_path)
    transcriptor.transcribe(audio, str(out_midi))

    if not out_midi.exists():
        raise RuntimeError(f"piano_transcription_inference 推理结束，但未找到输出：{out_midi}")
    return out_midi
