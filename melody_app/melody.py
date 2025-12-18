from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .key import KeyInfo, scale_pcs, snap_pitch_to_scale
from .midi_io import NoteEvent


@dataclass(frozen=True)
class MelodyDPConfig:
    """Dynamic programming config for predominant melody selection (polyphonic -> monophonic)."""

    hop: float = 0.02

    # local / transition scores
    salience_weight: float = 2.0
    change_penalty: float = 0.15
    jump_cost_per_octave: float = 0.6
    silence_penalty: float = 0.05
    onset_penalty: float = 0.02
    offset_penalty: float = 0.02

    # Bias
    prefer_high_pitch: bool = True
    pitch_bias: float = 0.04  # small bias toward higher pitch if prefer_high_pitch

    # Candidate constraints
    min_midi: int = 36
    max_midi: int = 96


@dataclass(frozen=True)
class NoteCleanupConfig:
    min_note_len: float = 0.08
    merge_gap: float = 0.05


def polyphony_ratio(notes: List[NoteEvent]) -> float:
    """Rough polyphony estimate: sample time points and check overlap >= 2."""
    if not notes:
        return 0.0
    t0 = min(n.start for n in notes)
    t1 = max(n.end for n in notes)
    if t1 <= t0:
        return 0.0

    ts = np.linspace(t0, t1, 200, dtype=np.float64)
    poly = 0
    for t in ts:
        active = 0
        for n in notes:
            if n.start <= t < n.end:
                active += 1
                if active >= 2:
                    poly += 1
                    break
    return float(poly / len(ts))


def remove_short_notes(notes: List[NoteEvent], min_len: float) -> List[NoteEvent]:
    return [n for n in notes if (n.end - n.start) >= min_len]


def merge_same_pitch(notes: List[NoteEvent], gap: float) -> List[NoteEvent]:
    if not notes:
        return []
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    merged = [notes[0]]
    for n in notes[1:]:
        last = merged[-1]
        if n.pitch == last.pitch and (n.start - last.end) <= gap:
            merged[-1] = NoteEvent(
                pitch=last.pitch,
                start=last.start,
                end=max(last.end, n.end),
                velocity=int(max(last.velocity, n.velocity)),
            )
        else:
            merged.append(n)
    return merged


def _notes_to_frame_candidates(
    notes: List[NoteEvent],
    start_t: float,
    end_t: float,
    hop: float,
    min_midi: int,
    max_midi: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Build per-frame candidate pitches and saliences (velocity)."""
    dur = max(0.0, end_t - start_t)
    n_frames = int(np.ceil(dur / hop)) + 1

    cand: List[Dict[int, int]] = [dict() for _ in range(n_frames)]
    for n in notes:
        if n.pitch < min_midi or n.pitch > max_midi:
            continue
        s = int(np.floor((n.start - start_t) / hop))
        e = int(np.ceil((n.end - start_t) / hop))
        s = max(0, min(n_frames - 1, s))
        e = max(0, min(n_frames, e))
        for i in range(s, e):
            d = cand[i]
            v = int(n.velocity)
            prev = d.get(n.pitch)
            if prev is None or v > prev:
                d[n.pitch] = v

    pitches_per_frame: List[List[int]] = []
    salience_per_frame: List[List[int]] = []
    for i in range(n_frames):
        d = cand[i]
        ps = [-1] + sorted(d.keys())
        vs = [0] + [int(d[p]) for p in ps[1:]]
        pitches_per_frame.append(ps)
        salience_per_frame.append(vs)

    return pitches_per_frame, salience_per_frame


def _transition_score(prev: int, cur: int, cfg: MelodyDPConfig) -> float:
    if prev == -1 and cur == -1:
        return 0.0
    if prev == -1 and cur != -1:
        return -cfg.onset_penalty
    if prev != -1 and cur == -1:
        return -cfg.offset_penalty

    if prev != cur:
        jump = abs(float(cur) - float(prev)) / 12.0
        return -(cfg.change_penalty + cfg.jump_cost_per_octave * jump)
    return 0.0


def to_monophonic_dp(
    notes: List[NoteEvent],
    cfg: MelodyDPConfig,
) -> Tuple[List[NoteEvent], Dict[str, float]]:
    """Extract a single predominant melody line from polyphonic notes using DP/Viterbi."""
    if not notes:
        return [], {"polyphony_ratio": 0.0}

    start_t = min(n.start for n in notes)
    end_t = max(n.end for n in notes)
    if end_t <= start_t:
        return [], {"polyphony_ratio": 0.0}

    pitches_pf, sal_pf = _notes_to_frame_candidates(
        notes,
        start_t=start_t,
        end_t=end_t,
        hop=cfg.hop,
        min_midi=cfg.min_midi,
        max_midi=cfg.max_midi,
    )

    prev_scores: Optional[np.ndarray] = None
    back_ptrs: List[np.ndarray] = []

    for i, (cands, sals) in enumerate(zip(pitches_pf, sal_pf)):
        cands_np = np.array(cands, dtype=np.int16)
        sals_np = np.array(sals, dtype=np.float32) / 127.0

        local = cfg.salience_weight * sals_np
        if cfg.prefer_high_pitch and cfg.pitch_bias > 0:
            pitch_norm = np.clip(cands_np.astype(np.float32), 0, 127) / 127.0
            local += cfg.pitch_bias * pitch_norm

        local = local.copy()
        local[cands_np == -1] -= cfg.silence_penalty

        if i == 0:
            prev_scores = local.astype(np.float32)
            back_ptrs.append(np.zeros(len(cands_np), dtype=np.int32))
            continue

        assert prev_scores is not None
        prev_cands = np.array(pitches_pf[i - 1], dtype=np.int16)

        scores = np.full(len(cands_np), -1e9, dtype=np.float32)
        bp = np.zeros(len(cands_np), dtype=np.int32)

        for k, cur in enumerate(cands_np):
            best = -1e9
            best_j = 0
            for j, prev in enumerate(prev_cands):
                val = float(prev_scores[j]) + _transition_score(int(prev), int(cur), cfg)
                if val > best:
                    best = val
                    best_j = j
            scores[k] = best + float(local[k])
            bp[k] = best_j

        prev_scores = scores
        back_ptrs.append(bp)

    assert prev_scores is not None
    last_k = int(np.argmax(prev_scores))
    seq: List[int] = []
    k = last_k
    for i in range(len(pitches_pf) - 1, -1, -1):
        seq.append(int(pitches_pf[i][k]))
        k = int(back_ptrs[i][k])
    seq.reverse()

    out: List[NoteEvent] = []
    cur_pitch = -1
    cur_start: Optional[float] = None
    cur_vels: List[int] = []

    for i, p in enumerate(seq):
        t = float(start_t + i * cfg.hop)
        if int(p) == int(cur_pitch):
            if cur_pitch != -1:
                idx = pitches_pf[i].index(int(p))
                cur_vels.append(int(sal_pf[i][idx]))
            continue

        if cur_pitch != -1 and cur_start is not None:
            v = int(np.median(cur_vels)) if cur_vels else 80
            out.append(NoteEvent(pitch=int(cur_pitch), start=float(cur_start), end=float(t), velocity=v))

        if p != -1:
            cur_pitch = int(p)
            cur_start = t
            idx = pitches_pf[i].index(int(p))
            cur_vels = [int(sal_pf[i][idx])]
        else:
            cur_pitch = -1
            cur_start = None
            cur_vels = []

    if cur_pitch != -1 and cur_start is not None:
        v = int(np.median(cur_vels)) if cur_vels else 80
        out.append(NoteEvent(pitch=int(cur_pitch), start=float(cur_start), end=float(end_t), velocity=v))

    out = [n for n in out if n.duration() >= cfg.hop * 1.5]
    stats = {"polyphony_ratio": float(polyphony_ratio(notes))}
    return out, stats


def snap_notes_to_key(
    notes: List[NoteEvent],
    key: KeyInfo,
    max_snap: int = 1,
    gap: float = 0.05,
) -> List[NoteEvent]:
    if not notes:
        return []
    pcs = scale_pcs(key)
    snapped = [
        NoteEvent(
            pitch=snap_pitch_to_scale(n.pitch, pcs, max_adjust=max_snap),
            start=n.start,
            end=n.end,
            velocity=n.velocity,
        )
        for n in notes
    ]
    return merge_same_pitch(snapped, gap=gap)


def f0_to_notes(
    times: np.ndarray,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    *,
    velocity_from: str = "confidence",
    conf: Optional[np.ndarray] = None,
    min_note_len: float = 0.08,
    merge_gap: float = 0.05,
) -> List[NoteEvent]:
    """Convert a monophonic f0 track to MIDI notes."""
    assert times.ndim == 1 and f0_hz.ndim == 1 and voiced.ndim == 1
    assert len(times) == len(f0_hz) == len(voiced)

    if len(times) >= 2:
        hop = float(np.median(np.diff(times)))
    else:
        hop = 0.01

    midi = np.full(len(times), -1, dtype=np.int32)
    for i, (hz, v) in enumerate(zip(f0_hz, voiced)):
        if (not bool(v)) or float(hz) <= 0:
            continue
        m = 69.0 + 12.0 * np.log2(float(hz) / 440.0)
        midi[i] = int(np.round(m))

    if velocity_from == "confidence" and conf is not None:
        vel_arr = np.clip(conf, 0.0, 1.0) * 127.0
    else:
        vel_arr = np.full(len(times), 80.0, dtype=np.float32)

    out: List[NoteEvent] = []
    cur = -1
    cur_start: Optional[float] = None
    cur_vels: List[float] = []

    for i, p in enumerate(midi):
        t = float(times[i])
        if int(p) == int(cur):
            if cur != -1:
                cur_vels.append(float(vel_arr[i]))
            continue

        if cur != -1 and cur_start is not None:
            v = int(np.median(cur_vels)) if cur_vels else 80
            out.append(NoteEvent(pitch=int(cur), start=float(cur_start), end=float(t), velocity=v))

        if p != -1:
            cur = int(p)
            cur_start = t
            cur_vels = [float(vel_arr[i])]
        else:
            cur = -1
            cur_start = None
            cur_vels = []

    if cur != -1 and cur_start is not None:
        v = int(np.median(cur_vels)) if cur_vels else 80
        out.append(NoteEvent(pitch=int(cur), start=float(cur_start), end=float(times[-1] + hop), velocity=v))

    out = remove_short_notes(out, min_note_len)
    out = merge_same_pitch(out, gap=merge_gap)
    return out


@dataclass(frozen=True)
class MelodyPostProcessConfig:
    mode: str = "voice"  # voice|string|piano|poly
    cleanup: NoteCleanupConfig = NoteCleanupConfig()
    dp: MelodyDPConfig = MelodyDPConfig()
    snap_scale: bool = True
    max_snap: int = 1
    key: Optional[KeyInfo] = None


def simplify_polyphonic_notes(
    raw_notes: List[NoteEvent],
    cfg: MelodyPostProcessConfig,
) -> Tuple[List[NoteEvent], Dict[str, float]]:
    """Full melody extraction from polyphonic notes (BasicPitch / piano transcription outputs)."""
    if not raw_notes:
        return [], {"raw_notes": 0.0, "melody_notes": 0.0, "polyphony_ratio": 0.0}

    dp = cfg.dp
    if cfg.mode in ("voice", "humming"):
        dp = MelodyDPConfig(**{**dp.__dict__, "min_midi": 45, "max_midi": 93, "prefer_high_pitch": False})
    elif cfg.mode in ("string", "instrument"):
        dp = MelodyDPConfig(**{**dp.__dict__, "min_midi": 40, "max_midi": 100, "prefer_high_pitch": True})
    elif cfg.mode == "piano":
        dp = MelodyDPConfig(**{**dp.__dict__, "min_midi": 36, "max_midi": 108, "prefer_high_pitch": True})

    mono, stats = to_monophonic_dp(raw_notes, cfg=dp)

    mono = remove_short_notes(mono, cfg.cleanup.min_note_len)
    mono = merge_same_pitch(mono, gap=cfg.cleanup.merge_gap)

    if cfg.snap_scale and cfg.key is not None:
        mono = snap_notes_to_key(mono, cfg.key, max_snap=cfg.max_snap, gap=cfg.cleanup.merge_gap)

    stats = {
        **stats,
        "raw_notes": float(len(raw_notes)),
        "melody_notes": float(len(mono)),
    }
    return mono, stats
