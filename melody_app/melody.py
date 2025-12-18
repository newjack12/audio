from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .key import KeyInfo, scale_pcs, snap_pitch_to_scale
from .midi_io import NoteEvent


# ============================================================
# Configs
# ============================================================

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


@dataclass(frozen=True)
class MelodyPostProcessConfig:
    """
    Post-process config for polyphonic melody extraction.
    This is kept for pipeline compatibility.
    """
    mode: str = "voice"  # voice|string|piano|poly
    cleanup: NoteCleanupConfig = NoteCleanupConfig()
    dp: MelodyDPConfig = MelodyDPConfig()
    snap_scale: bool = True
    max_snap: int = 1
    key: Optional[KeyInfo] = None


# ============================================================
# Utilities
# ============================================================

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
    return [n for n in notes if (n.end - n.start) >= float(min_len)]


# ============================================================
# Merge logic (shared, PARAMETERIZED)
# ============================================================

def merge_same_pitch(
    notes: List[NoteEvent],
    gap: float,
    *,
    tol_semitones: int = 0,
    agg: str = "mode",  # "median" | "mode"
) -> List[NoteEvent]:
    """
    Merge notes that are (almost) same pitch with small gaps.

    tol_semitones:
      - 1  : humming / F0 route
      - 0  : polyphonic / note route (STRICT)

    agg:
      - "median" : humming/F0 route
      - "mode"   : polyphonic/note route
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    merged: List[NoteEvent] = []
    group_pitches: List[int] = []
    group_vels: List[int] = []

    def _agg_pitch(pitches: List[int]) -> int:
        if not pitches:
            return 60
        if agg == "mode":
            mn = int(min(pitches))
            bc = np.bincount(np.array(pitches, dtype=np.int32) - mn)
            return int(np.argmax(bc) + mn)
        # default median
        return int(np.round(float(np.median(np.array(pitches, dtype=np.float32)))))

    cur = notes[0]
    group_pitches = [int(cur.pitch)]
    group_vels = [int(cur.velocity)]

    for n in notes[1:]:
        close_enough = abs(int(n.pitch) - int(cur.pitch)) <= int(tol_semitones)
        gap_ok = (float(n.start) - float(cur.end)) <= float(gap)

        if close_enough and gap_ok:
            group_pitches.append(int(n.pitch))
            group_vels.append(int(n.velocity))
            cur = NoteEvent(
                pitch=int(_agg_pitch(group_pitches)),
                start=float(cur.start),
                end=float(max(cur.end, n.end)),
                velocity=int(max(group_vels)),
            )
        else:
            merged.append(cur)
            cur = n
            group_pitches = [int(cur.pitch)]
            group_vels = [int(cur.velocity)]

    merged.append(cur)
    return merged


# ============================================================
# Key snapping (pipeline needs this symbol)
# ============================================================

def snap_notes_to_key(
    notes: List[NoteEvent],
    key: KeyInfo,
    max_snap: int = 1,
    gap: float = 0.05,
    *,
    # keep defaults safe for general usage
    tol_semitones_merge: int = 0,
    agg: str = "mode",
) -> List[NoteEvent]:
    """
    Snap notes to a musical key/scale, then merge (optional).
    pipeline.py imports this function, so it must exist.
    """
    if not notes:
        return []
    pcs = scale_pcs(key)
    snapped = [
        NoteEvent(
            pitch=int(snap_pitch_to_scale(int(n.pitch), pcs, max_adjust=int(max_snap))),
            start=float(n.start),
            end=float(n.end),
            velocity=int(n.velocity),
        )
        for n in notes
    ]
    return merge_same_pitch(snapped, gap=float(gap), tol_semitones=int(tol_semitones_merge), agg=str(agg))


# ============================================================
# DP for polyphonic -> monophonic (note route)
# ============================================================

def _notes_to_frame_candidates(
    notes: List[NoteEvent],
    start_t: float,
    end_t: float,
    hop: float,
    min_midi: int,
    max_midi: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Build per-frame candidate pitches and saliences (velocity)."""
    dur = max(0.0, float(end_t) - float(start_t))
    n_frames = int(np.ceil(dur / float(hop))) + 1

    cand: List[Dict[int, int]] = [dict() for _ in range(n_frames)]
    for n in notes:
        if int(n.pitch) < int(min_midi) or int(n.pitch) > int(max_midi):
            continue
        s = int(np.floor((float(n.start) - float(start_t)) / float(hop)))
        e = int(np.ceil((float(n.end) - float(start_t)) / float(hop)))
        s = max(0, min(n_frames - 1, s))
        e = max(0, min(n_frames, e))
        for i in range(s, e):
            v = int(n.velocity)
            prev = cand[i].get(int(n.pitch))
            if prev is None or v > prev:
                cand[i][int(n.pitch)] = v

    pitches_pf: List[List[int]] = []
    sal_pf: List[List[int]] = []
    for d in cand:
        ps = [-1] + sorted(d.keys())
        vs = [0] + [int(d[p]) for p in ps[1:]]
        pitches_pf.append(ps)
        sal_pf.append(vs)

    return pitches_pf, sal_pf


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

    start_t = float(min(n.start for n in notes))
    end_t = float(max(n.end for n in notes))
    if end_t <= start_t:
        return [], {"polyphony_ratio": 0.0}

    pitches_pf, sal_pf = _notes_to_frame_candidates(
        notes,
        start_t=start_t,
        end_t=end_t,
        hop=float(cfg.hop),
        min_midi=int(cfg.min_midi),
        max_midi=int(cfg.max_midi),
    )

    prev_scores: Optional[np.ndarray] = None
    back_ptrs: List[np.ndarray] = []

    for i, (cands, sals) in enumerate(zip(pitches_pf, sal_pf)):
        cands_np = np.array(cands, dtype=np.int16)
        sals_np = np.array(sals, dtype=np.float32) / 127.0

        local = float(cfg.salience_weight) * sals_np
        if cfg.prefer_high_pitch and cfg.pitch_bias > 0:
            pitch_norm = np.clip(cands_np.astype(np.float32), 0, 127) / 127.0
            local += float(cfg.pitch_bias) * pitch_norm

        local = local.copy()
        local[cands_np == -1] -= float(cfg.silence_penalty)

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
        t = float(start_t + i * float(cfg.hop))
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

    out = [n for n in out if n.duration() >= float(cfg.hop) * 1.5]
    return out, {"polyphony_ratio": float(polyphony_ratio(notes))}


# ============================================================
# F0 -> Notes (方案 A：F0 路线专用优化)
# ============================================================

def f0_to_notes(
    times: np.ndarray,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    *,
    velocity_from: str = "confidence",
    conf: Optional[np.ndarray] = None,
    min_note_len: float = 0.08,
    merge_gap: float = 0.05,
    # --- 方案 A ---
    voiced_on: float = 0.60,
    voiced_off: float = 0.40,
    change_stable_frames: int = 3,          # 2~3 recommended
    pitch_tolerance_semitones: int = 1,     # treat +/-1 semitone as same pitch while smoothing
) -> List[NoteEvent]:
    """
    Convert a monophonic f0 track to MIDI notes.

    方案 A（F0 路线专用）：
      A) voiced hysteresis: enter if prob>0.6, exit if prob<0.4, else keep state
      B) pitch change debounce: new pitch must stay stable N frames before switching
      C) merge note segments allowing +/-1 semitone, aggregate pitch by median
    """
    assert times.ndim == 1 and f0_hz.ndim == 1 and voiced.ndim == 1
    assert len(times) == len(f0_hz) == len(voiced)

    hop = float(np.median(np.diff(times))) if len(times) >= 2 else 0.01

    # confidence / voiced_prob
    if velocity_from == "confidence" and conf is not None:
        voiced_prob = np.clip(conf.astype(np.float32), 0.0, 1.0)
        vel_arr = voiced_prob * 127.0
    else:
        voiced_prob = voiced.astype(np.float32)
        vel_arr = np.full(len(times), 80.0, dtype=np.float32)

    # A) voiced hysteresis
    voiced_h = np.zeros(len(times), dtype=np.bool_)
    state = False
    for i in range(len(times)):
        p = float(voiced_prob[i])
        if not state:
            if p >= float(voiced_on):
                state = True
        else:
            if p <= float(voiced_off):
                state = False
        voiced_h[i] = state

    # hz -> midi per frame
    midi_raw = np.full(len(times), -1, dtype=np.int32)
    for i, (hz, v) in enumerate(zip(f0_hz, voiced_h)):
        if (not bool(v)) or float(hz) <= 0.0:
            continue
        m = 69.0 + 12.0 * np.log2(float(hz) / 440.0)
        midi_raw[i] = int(np.round(m))

    # B) pitch stability gate
    stable_n = int(max(1, change_stable_frames))
    tol = int(max(0, pitch_tolerance_semitones))

    midi_smooth = np.full_like(midi_raw, -1)

    cur = -1
    pending = -1
    pending_count = 0
    pending_start_i: Optional[int] = None

    for i, p in enumerate(midi_raw):
        if p == -1:
            cur = -1
            pending = -1
            pending_count = 0
            pending_start_i = None
            midi_smooth[i] = -1
            continue

        if cur == -1:
            cur = int(p)
            midi_smooth[i] = cur
            continue

        # treat +/- tol as same pitch
        if abs(int(p) - int(cur)) <= tol:
            midi_smooth[i] = cur
            pending = -1
            pending_count = 0
            pending_start_i = None
            continue

        # new pending
        if pending == -1 or int(p) != int(pending):
            pending = int(p)
            pending_count = 1
            pending_start_i = i
            midi_smooth[i] = cur
            continue

        # pending continues
        pending_count += 1
        midi_smooth[i] = cur

        if pending_count >= stable_n and pending_start_i is not None:
            cur = int(pending)
            for j in range(pending_start_i, i + 1):
                midi_smooth[j] = cur
            pending = -1
            pending_count = 0
            pending_start_i = None

    # build notes
    out: List[NoteEvent] = []
    cur_pitch = -1
    cur_start: Optional[float] = None
    cur_vels: List[float] = []

    for i, p in enumerate(midi_smooth):
        t = float(times[i])

        if int(p) == int(cur_pitch):
            if cur_pitch != -1:
                cur_vels.append(float(vel_arr[i]))
            continue

        if cur_pitch != -1 and cur_start is not None:
            v = int(np.median(cur_vels)) if cur_vels else 80
            out.append(NoteEvent(pitch=int(cur_pitch), start=float(cur_start), end=float(t), velocity=v))

        if p != -1:
            cur_pitch = int(p)
            cur_start = t
            cur_vels = [float(vel_arr[i])]
        else:
            cur_pitch = -1
            cur_start = None
            cur_vels = []

    if cur_pitch != -1 and cur_start is not None:
        v = int(np.median(cur_vels)) if cur_vels else 80
        out.append(NoteEvent(pitch=int(cur_pitch), start=float(cur_start), end=float(times[-1] + hop), velocity=v))

    # cleanup + C) tolerant merge (F0 route only)
    out = remove_short_notes(out, float(min_note_len))
    out = merge_same_pitch(out, gap=float(merge_gap), tol_semitones=1, agg="median")
    return out


# ============================================================
# Polyphonic -> Melody (note route, STRICT)
# ============================================================

def simplify_polyphonic_notes(
    raw_notes: List[NoteEvent],
    cfg: MelodyPostProcessConfig,
) -> Tuple[List[NoteEvent], Dict[str, float]]:
    """Full melody extraction from polyphonic notes (BasicPitch / piano transcription outputs)."""
    if not raw_notes:
        return [], {"raw_notes": 0.0, "melody_notes": 0.0, "polyphony_ratio": 0.0}

    dp = cfg.dp
    # mode-specific DP bounds (keeps it robust)
    if cfg.mode in ("voice", "humming"):
        dp = MelodyDPConfig(**{**dp.__dict__, "min_midi": 45, "max_midi": 93, "prefer_high_pitch": False})
    elif cfg.mode in ("string", "instrument"):
        dp = MelodyDPConfig(**{**dp.__dict__, "min_midi": 40, "max_midi": 100, "prefer_high_pitch": True})
    elif cfg.mode == "piano":
        dp = MelodyDPConfig(**{**dp.__dict__, "min_midi": 36, "max_midi": 108, "prefer_high_pitch": True})

    mono, stats = to_monophonic_dp(raw_notes, cfg=dp)

    mono = remove_short_notes(mono, float(cfg.cleanup.min_note_len))

    # STRICT merge for note route: do NOT use humming tolerance here
    mono = merge_same_pitch(mono, gap=float(cfg.cleanup.merge_gap), tol_semitones=0, agg="mode")

    if cfg.snap_scale and cfg.key is not None:
        # snap strictly, then strict merge again (optional but safe)
        mono = snap_notes_to_key(
            mono,
            key=cfg.key,
            max_snap=int(cfg.max_snap),
            gap=float(cfg.cleanup.merge_gap),
            tol_semitones_merge=0,
            agg="mode",
        )

    stats = {
        **stats,
        "raw_notes": float(len(raw_notes)),
        "melody_notes": float(len(mono)),
    }
    return mono, stats
