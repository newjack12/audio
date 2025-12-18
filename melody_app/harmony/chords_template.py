from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..midi_io import NoteEvent


@dataclass(frozen=True)
class ChordEvent:
    label: str
    start: float
    end: float
    pitches: Tuple[int, ...]


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pc_histogram(notes: List[NoteEvent], t0: float, t1: float) -> np.ndarray:
    """Pitch-class histogram weighted by overlap duration & velocity."""
    h = np.zeros(12, dtype=np.float32)
    for n in notes:
        if n.end <= t0 or n.start >= t1:
            continue
        overlap = max(0.0, min(n.end, t1) - max(n.start, t0))
        if overlap <= 0:
            continue
        low_bias = 1.0 + 0.15 * (1.0 - (n.pitch / 127.0))
        w = overlap * (max(1, n.velocity) / 127.0) * low_bias
        h[n.pitch % 12] += float(w)

    s = float(np.sum(h))
    if s > 0:
        h /= s
    return h


def _chord_templates() -> Dict[str, np.ndarray]:
    """Return chord templates for 24 major/minor triads."""
    templates: Dict[str, np.ndarray] = {}
    maj = np.zeros(12, dtype=np.float32)
    maj[[0, 4, 7]] = 1.0
    minr = np.zeros(12, dtype=np.float32)
    minr[[0, 3, 7]] = 1.0

    for root in range(12):
        name = _NOTE_NAMES[root]
        templates[f"{name}:maj"] = np.roll(maj, root)
        templates[f"{name}:min"] = np.roll(minr, root)

    for k, v in list(templates.items()):
        vn = v / max(1e-6, np.linalg.norm(v))
        templates[k] = vn.astype(np.float32)

    return templates


_TEMPLATES = _chord_templates()


def estimate_chords(
    notes: List[NoteEvent],
    *,
    hop: float = 0.5,
    min_duration: float = 0.5,
    no_chord_threshold: float = 0.25,
    output_octave: int = 4,
) -> List[ChordEvent]:
    """Estimate a chord progression from note events (template matching baseline)."""
    if not notes:
        return []

    t0 = min(n.start for n in notes)
    t1 = max(n.end for n in notes)
    if t1 <= t0:
        return []

    n_frames = int(np.ceil((t1 - t0) / hop))
    raw: List[Tuple[str, float, float]] = []

    for i in range(n_frames):
        a = t0 + i * hop
        b = min(t1, a + hop)
        h = _pc_histogram(notes, a, b)
        if float(np.sum(h)) <= 0:
            raw.append(("N", a, b))
            continue

        h_norm = h / max(1e-6, float(np.linalg.norm(h)))

        best_label = "N"
        best_score = -1e9
        for lab, tpl in _TEMPLATES.items():
            score = float(np.dot(h_norm, tpl))
            if score > best_score:
                best_score = score
                best_label = lab

        if best_score < float(no_chord_threshold):
            best_label = "N"

        raw.append((best_label, a, b))

    merged: List[Tuple[str, float, float]] = []
    for lab, a, b in raw:
        if not merged:
            merged.append((lab, a, b))
            continue
        lab0, a0, b0 = merged[-1]
        if lab == lab0:
            merged[-1] = (lab0, a0, b)
        else:
            merged.append((lab, a, b))

    merged2 = [(lab, a, b) for (lab, a, b) in merged if (b - a) >= min_duration]

    events: List[ChordEvent] = []
    base_c = 12 * (output_octave + 1)  # C4=60 when output_octave=4

    for lab, a, b in merged2:
        if lab == "N":
            events.append(ChordEvent(label=lab, start=float(a), end=float(b), pitches=()))
            continue

        root_name, qual = lab.split(":")
        root_pc = _NOTE_NAMES.index(root_name)
        root = base_c + root_pc
        if qual == "maj":
            pitches = (root, root + 4, root + 7)
        else:
            pitches = (root, root + 3, root + 7)

        events.append(
            ChordEvent(
                label=lab,
                start=float(a),
                end=float(b),
                pitches=tuple(int(p) for p in pitches),
            )
        )

    return events


def chords_to_notes(events: List[ChordEvent], velocity: int = 70) -> List[NoteEvent]:
    notes: List[NoteEvent] = []
    v = int(max(1, min(127, velocity)))
    for ev in events:
        for p in ev.pitches:
            notes.append(NoteEvent(pitch=int(p), start=float(ev.start), end=float(ev.end), velocity=v))
    notes.sort(key=lambda x: (x.start, x.pitch))
    return notes
