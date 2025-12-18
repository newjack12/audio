from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    import pretty_midi
except Exception as e:  # pragma: no cover
    raise ImportError(
        "缺少 pretty_midi。请安装：\n"
        "  python -m pip install pretty_midi\n"
        f"原始错误: {e}"
    )


@dataclass(frozen=True)
class NoteEvent:
    pitch: int        # MIDI note number
    start: float      # seconds
    end: float        # seconds
    velocity: int     # 1..127

    def duration(self) -> float:
        return float(self.end - self.start)


def load_midi_notes(midi_path: Path, instrument_name: Optional[str] = None) -> List[NoteEvent]:
    """Load notes from a MIDI file.

    Parameters
    ----------
    instrument_name:
        If provided, only load instruments whose name contains this substring (case-insensitive).
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[NoteEvent] = []

    for inst in pm.instruments:
        if instrument_name:
            if instrument_name.lower() not in (inst.name or "").lower():
                continue
        for n in inst.notes:
            notes.append(
                NoteEvent(
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                    velocity=int(n.velocity) if n.velocity else 80,
                )
            )

    notes.sort(key=lambda x: (x.start, x.pitch))
    return notes


def save_midi_notes(
    notes: List[NoteEvent],
    out_midi: Path,
    program: int = 0,
    name: str = "melody",
    is_drum: bool = False,
) -> None:
    """Save a single-track MIDI."""
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name=name, is_drum=is_drum)

    for n in notes:
        vel = int(max(1, min(127, n.velocity)))
        inst.notes.append(
            pretty_midi.Note(
                velocity=vel,
                pitch=int(n.pitch),
                start=float(n.start),
                end=float(n.end),
            )
        )

    pm.instruments.append(inst)
    pm.write(str(out_midi))


def save_multitrack_midi(
    tracks: Mapping[str, List[NoteEvent]],
    out_midi: Path,
    program_map: Optional[Mapping[str, int]] = None,
) -> None:
    """Save multiple named tracks to a MIDI file."""
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()

    for name, notes in tracks.items():
        program = int(program_map.get(name, 0) if program_map else 0)
        inst = pretty_midi.Instrument(program=program, name=str(name))
        for n in notes:
            vel = int(max(1, min(127, n.velocity)))
            inst.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                )
            )
        pm.instruments.append(inst)

    pm.write(str(out_midi))


def save_notes_json(
    notes: List[NoteEvent],
    out_json: Path,
    meta: Dict[str, Any],
    stats: Dict[str, Any],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "stats": stats,
        "notes": [
            {
                "pitch": n.pitch,
                "name": pretty_midi.note_number_to_name(n.pitch),
                "start": n.start,
                "end": n.end,
                "duration": n.end - n.start,
                "velocity": n.velocity,
            }
            for n in notes
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jianpu_degree(pitch: int, tonic_pc: int, mode: str) -> str:
    """Very light Jianpu representation (for quick human inspection)."""
    pc = pitch % 12
    rel = (pc - tonic_pc) % 12

    if mode == "minor":
        scale = [0, 2, 3, 5, 7, 8, 10]  # natural minor
    else:
        scale = [0, 2, 4, 5, 7, 9, 11]  # major

    if rel not in scale:
        return pretty_midi.note_number_to_name(pitch)

    degree = scale.index(rel) + 1

    octave = pitch // 12 - 1
    diff = octave - 4
    if diff > 0:
        return f"{degree}" + ("'" * diff)
    if diff < 0:
        return f"{degree}" + ("," * (-diff))
    return f"{degree}"


def save_notes_text(
    notes: List[NoteEvent],
    out_note_names: Path,
    out_jianpu: Path,
    key_name: str,
    tonic_pc: int,
    mode: str,
    confidence: float,
) -> None:
    out_note_names.parent.mkdir(parents=True, exist_ok=True)

    lines_names = []
    lines_jianpu = []
    for n in notes:
        dur_ms = int(round((n.end - n.start) * 1000))
        nm = pretty_midi.note_number_to_name(n.pitch)
        lines_names.append(f"{nm}\t{dur_ms}ms")
        lines_jianpu.append(f"{_jianpu_degree(n.pitch, tonic_pc, mode)}\t{dur_ms}ms")

    header = [
        f"# Key: {key_name} (confidence={confidence:.3f})",
        "# Format: NOTE(or degree) <TAB> duration_ms",
        "",
    ]
    out_note_names.write_text("\n".join(header + lines_names), encoding="utf-8")
    out_jianpu.write_text("\n".join(header + lines_jianpu), encoding="utf-8")
