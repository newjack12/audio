from __future__ import annotations

from pathlib import Path

from ..utils.optional_import import require_module


def run_basic_pitch(
    wav_path: Path,
    out_dir: Path,
    *,
    save_model_outputs: bool = False,
    save_notes: bool = False,
) -> Path:
    """Run Spotify Basic Pitch (ICASSP 2022) to get a polyphonic MIDI."""
    require_module("basic_pitch")

    from basic_pitch.inference import predict_and_save  # type: ignore
    from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore

    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        [str(wav_path)],
        str(out_dir),
        sonify_midi=False,
        save_midi=True,
        save_model_outputs=bool(save_model_outputs),
        save_notes=bool(save_notes),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    expected_mid = out_dir / f"{wav_path.stem}_basic_pitch.mid"
    if not expected_mid.exists():
        raise RuntimeError(f"Basic Pitch 推理完成，但未找到 MIDI 文件：{expected_mid}")

    return expected_mid
