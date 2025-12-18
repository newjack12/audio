from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ..utils.optional_import import require_module


def transcribe_chords_omnizart(
    wav_path: Path,
    out_dir: Path,
    *,
    model_path: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Chord transcription using Omnizart (Harmony Transformer).

    Omnizart's chord module outputs BOTH:
      - a MIDI file (for quick listening)
      - a CSV file (chord label + start/end)

    Returns (midi_path, csv_path). Either can be None if not found.
    """
    require_module("omnizart")
    from omnizart.chord import transcribe  # type: ignore

    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = {p.name for p in out_dir.glob("*")}
    _ = transcribe(str(wav_path), model_path=model_path, output=str(out_dir))
    after = {p.name for p in out_dir.glob("*")}
    new_files = sorted(after - before)

    midi_path = None
    csv_path = None
    for fn in new_files:
        p = out_dir / fn
        if p.suffix.lower() == ".mid":
            midi_path = p
        elif p.suffix.lower() == ".csv":
            csv_path = p

    # fallback to conventional names
    if midi_path is None:
        p = out_dir / "chord.mid"
        if p.exists():
            midi_path = p
    if csv_path is None:
        p = out_dir / "chord.csv"
        if p.exists():
            csv_path = p

    return midi_path, csv_path
