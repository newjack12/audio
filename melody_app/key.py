from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .midi_io import NoteEvent


@dataclass(frozen=True)
class KeyInfo:
    tonic_pc: int     # 0..11 (C=0)
    mode: str         # "major" | "minor"
    confidence: float # 0..1
    name: str         # e.g. "Cmaj"


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (widely used)
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def parse_key_string(s: str) -> KeyInfo:
    """Parse keys like: Cmaj, Amin, D#maj, F#min."""
    ss = s.strip()
    if len(ss) < 2:
        raise ValueError(f"非法 key: {s}")

    mode = "major"
    if ss.lower().endswith("min"):
        mode = "minor"
        root = ss[:-3]
    elif ss.lower().endswith("maj"):
        mode = "major"
        root = ss[:-3]
    else:
        root = ss

    root = root.strip()
    if root not in _NOTE_NAMES:
        raise ValueError(f"非法 key root: {root}，请用如 Cmaj / F#min 的格式")

    tonic_pc = _NOTE_NAMES.index(root)
    name = f"{root}{'min' if mode == 'minor' else 'maj'}"
    return KeyInfo(tonic_pc=tonic_pc, mode=mode, confidence=1.0, name=name)


def estimate_key_from_notes(notes: List[NoteEvent]) -> KeyInfo:
    """Estimate key from notes using duration*velocity weighted pitch-class histogram."""
    if not notes:
        return KeyInfo(tonic_pc=0, mode="major", confidence=0.0, name="Cmaj")

    hist = np.zeros(12, dtype=np.float64)
    for n in notes:
        dur = max(0.0, n.end - n.start)
        w = dur * max(1, n.velocity)
        hist[n.pitch % 12] += w

    total = float(hist.sum())
    if total <= 0:
        return KeyInfo(tonic_pc=0, mode="major", confidence=0.0, name="Cmaj")
    hist /= total

    maj = _MAJOR_PROFILE / _MAJOR_PROFILE.sum()
    minr = _MINOR_PROFILE / _MINOR_PROFILE.sum()

    best_mode = "major"
    best_tonic = 0
    best_score = -1e9

    for tonic in range(12):
        score_maj = float(np.dot(hist, np.roll(maj, tonic)))
        score_min = float(np.dot(hist, np.roll(minr, tonic)))
        if score_maj > best_score:
            best_score = score_maj
            best_mode = "major"
            best_tonic = tonic
        if score_min > best_score:
            best_score = score_min
            best_mode = "minor"
            best_tonic = tonic

    # confidence: top1-top2 gap
    scores = []
    for tonic in range(12):
        scores.append(float(np.dot(hist, np.roll(maj, tonic))))
        scores.append(float(np.dot(hist, np.roll(minr, tonic))))
    scores = np.array(scores, dtype=np.float64)
    scores.sort()
    top1 = float(scores[-1])
    top2 = float(scores[-2]) if len(scores) > 1 else float(scores[-1])
    conf = float(max(0.0, min(1.0, (top1 - top2) / (abs(top1) + 1e-9))))

    root = _NOTE_NAMES[best_tonic]
    name = f"{root}{'min' if best_mode == 'minor' else 'maj'}"
    return KeyInfo(tonic_pc=best_tonic, mode=best_mode, confidence=conf, name=name)


def scale_pcs(key: KeyInfo) -> List[int]:
    if key.mode == "minor":
        base = [0, 2, 3, 5, 7, 8, 10]
    else:
        base = [0, 2, 4, 5, 7, 9, 11]
    return [int((key.tonic_pc + x) % 12) for x in base]


def snap_pitch_to_scale(pitch: int, scale: List[int], max_adjust: int = 1) -> int:
    pc = pitch % 12
    if pc in scale:
        return pitch

    best = pitch
    best_dist = 999
    for d in range(-max_adjust, max_adjust + 1):
        cand = pitch + d
        if (cand % 12) in scale:
            dist = abs(d)
            if dist < best_dist:
                best_dist = dist
                best = cand
    return int(best)
