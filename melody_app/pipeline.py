from __future__ import annotations
from .backends.rmvpe_backend import RmvpeConfig, extract_f0_rmvpe

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .audio import PreprocessSpec, preprocess_to_wav
from .backends.basic_pitch_backend import run_basic_pitch
from .backends.crepe_backend import CrepeConfig, extract_f0_crepe
from .backends.piano_transcription_backend import PianoTranscriptionConfig, run_piano_transcription_inference
from .backends.pyin_backend import PyinConfig, extract_f0_pyin
from .key import KeyInfo, estimate_key_from_notes, parse_key_string
from .melody import (
    MelodyDPConfig,
    MelodyPostProcessConfig,
    NoteCleanupConfig,
    f0_to_notes,
    simplify_polyphonic_notes,
    snap_notes_to_key,
)
from .midi_io import load_midi_notes, save_midi_notes, save_notes_json, save_notes_text
from .synth import render_previews
from .utils.optional_import import optional_import


def _write_pitch_csv(out_csv: Path, times: np.ndarray, f0: np.ndarray, voiced: np.ndarray, conf: np.ndarray) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "f0_hz", "voiced", "confidence"])
        for t, hz, v, c in zip(times, f0, voiced, conf):
            w.writerow([float(t), float(hz), int(bool(v)), float(c)])


def run_pipeline(
    *,
    input_audio: Path,
    out_dir: Path,
    mode: str = "voice",  # voice|string|piano|poly
    engine: str = "auto",  # auto|crepe|pyin|basic_pitch|piano_transcription
    chords: str = "template",  # none|template|omnizart
    make_preview: bool = True,
    # cleanup / key
    min_note_len: Optional[float] = None,
    merge_gap: float = 0.05,
    frame_hop: float = 0.02,
    snap_scale: bool = True,
    max_snap: int = 1,
    key_override: Optional[str] = None,
    # device for torch backends
    device: str = "cpu",
) -> None:
    """Run a full transcription pipeline and write outputs to out_dir."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_audio = Path(input_audio)
    if not input_audio.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_audio}")

    # --- 0) Always create a 44.1k reference wav for preview / BasicPitch ---
    pre44 = out_dir / "00_preprocessed_44100.wav"
    preprocess_to_wav(
        input_audio=input_audio,
        output_wav=pre44,
        spec=PreprocessSpec(
            sr=44100,
            highpass_hz=80,
            lowpass_hz=None if mode in ("piano", "poly") else 10000,
            denoise=False,
            loudnorm=True,
        ),
    )

    # Decide engine if auto
    if engine == "auto":
        if mode in ("voice", "humming", "string", "instrument"):
            if optional_import("crepe") is not None:
                engine = "crepe"
            elif optional_import("librosa") is not None:
                engine = "pyin"
            else:
                engine = "basic_pitch"
        elif mode == "piano":
            if optional_import("piano_transcription_inference") is not None:
                engine = "piano_transcription"
            else:
                engine = "basic_pitch"
        else:
            engine = "basic_pitch"

    stats: Dict[str, float] = {}
    key_info: Optional[KeyInfo] = None

    raw_mid: Optional[Path] = None
    raw_notes = []

    # --- 1) Transcribe or extract pitch ---
    if engine in ("crepe", "pyin", "rmvpe"):
        # Monophonic pitch tracking -> notes

        if engine == "crepe":
            pre = out_dir / "00_preprocessed_16000.wav"
            preprocess_to_wav(
                input_audio=input_audio,
                output_wav=pre,
                spec=PreprocessSpec(
                    sr=16000,
                    highpass_hz=80,
                    lowpass_hz=8000 if mode in ("voice", "humming") else None,
                    denoise=False,
                    loudnorm=True,
                ),
            )
            times, f0, voiced, conf = extract_f0_crepe(pre, cfg=CrepeConfig())

        elif engine == "pyin":
            pre = out_dir / "00_preprocessed_22050.wav"
            preprocess_to_wav(
                input_audio=input_audio,
                output_wav=pre,
                spec=PreprocessSpec(
                    sr=22050,
                    highpass_hz=80,
                    lowpass_hz=8000 if mode in ("voice", "humming") else None,
                    denoise=False,
                    loudnorm=True,
                ),
            )
            times, f0, voiced, conf = extract_f0_pyin(pre, cfg=PyinConfig())

        elif engine == "rmvpe":
            # RMVPE prefers 16k / 40k, but 16k is enough and faster
            pre = out_dir / "00_preprocessed_16000.wav"
            preprocess_to_wav(
                input_audio=input_audio,
                output_wav=pre,
                spec=PreprocessSpec(
                    sr=16000,
                    highpass_hz=80,
                    lowpass_hz=8000 if mode in ("voice", "humming") else None,
                    denoise=False,
                    loudnorm=True,
                ),
            )
            times, f0, voiced, conf = extract_f0_rmvpe(
                pre,
                cfg=RmvpeConfig(device=device),
            )

        _write_pitch_csv(out_dir / "01_pitch_track.csv", times, f0, voiced, conf)

        if min_note_len is None:
            min_note_len = 0.08 if mode in ("voice", "humming") else 0.05

        melody_notes = f0_to_notes(
            times,
            f0,
            voiced,
            conf=conf,
            min_note_len=float(min_note_len),
            merge_gap=float(merge_gap),
        )

        # key
        if key_override:
            key_info = parse_key_string(key_override)
        else:
            key_info = estimate_key_from_notes(melody_notes)

        if snap_scale and key_info is not None:
            melody_notes = snap_notes_to_key(
                melody_notes,
                key=key_info,
                max_snap=int(max_snap),
                gap=float(merge_gap),
            )

        stats.update(
            {
                "raw_notes": float(len(melody_notes)),
                "melody_notes": float(len(melody_notes)),
                "polyphony_ratio": 0.0,
            }
        )

    else:
        # Polyphonic transcription -> extract melody
        if engine == "piano_transcription":
            if optional_import("piano_transcription_inference") is None:
                raise ImportError(
                    "未安装 piano_transcription_inference。请按 README 安装，或者使用 --engine basic_pitch。"
                )
            from piano_transcription_inference import sample_rate  # type: ignore

            sr = int(sample_rate)
            pre = out_dir / f"00_preprocessed_{sr}.wav"
            preprocess_to_wav(
                input_audio=input_audio,
                output_wav=pre,
                spec=PreprocessSpec(
                    sr=sr,
                    highpass_hz=30,
                    lowpass_hz=None,
                    denoise=False,
                    loudnorm=True,
                ),
            )
            raw_mid = out_dir / "01_raw_transcription.mid"
            run_piano_transcription_inference(
                pre,
                out_midi=raw_mid,
                cfg=PianoTranscriptionConfig(device=str(device)),
            )
        else:
            # BasicPitch
            pre = pre44
            raw_mid = out_dir / "01_raw_transcription.mid"
            tmp_mid = run_basic_pitch(pre, out_dir=out_dir)
            if tmp_mid.resolve() != raw_mid.resolve():
                raw_mid.write_bytes(tmp_mid.read_bytes())

        raw_notes = load_midi_notes(raw_mid)

        if min_note_len is None:
            min_note_len = 0.08 if mode in ("voice", "humming") else 0.05

        # key
        if key_override:
            key_info = parse_key_string(key_override)
        else:
            key_info = estimate_key_from_notes(raw_notes)

        melody_notes, mstats = simplify_polyphonic_notes(
            raw_notes,
            cfg=MelodyPostProcessConfig(
                mode=mode,
                cleanup=NoteCleanupConfig(min_note_len=float(min_note_len), merge_gap=float(merge_gap)),
                dp=MelodyDPConfig(hop=float(frame_hop)),
                snap_scale=bool(snap_scale),
                max_snap=int(max_snap),
                key=key_info,
            ),
        )
        stats.update(mstats)

    # --- 2) Save outputs ---
    out_mel_mid = out_dir / "02_melody.mid"
    out_mel_json = out_dir / "02_melody.json"
    out_note_names = out_dir / "02_melody_note_names.txt"
    out_jianpu = out_dir / "02_melody_jianpu.txt"

    save_midi_notes(melody_notes, out_mel_mid, program=0, name="melody")

    meta = {
        "mode": mode,
        "engine": engine,
        "input": str(input_audio),
    }
    save_notes_json(melody_notes, out_mel_json, meta=meta, stats=stats)

    if key_info is None:
        key_info = estimate_key_from_notes(melody_notes)

    save_notes_text(
        melody_notes,
        out_note_names,
        out_jianpu,
        key_name=key_info.name,
        tonic_pc=key_info.tonic_pc,
        mode=key_info.mode,
        confidence=key_info.confidence,
    )

    # --- 3) Chords / harmony (optional) ---
    if chords and chords != "none":
        try:
            if chords == "omnizart":
                from .harmony.omnizart_chords import transcribe_chords_omnizart

                midi_path, csv_path = transcribe_chords_omnizart(pre44, out_dir / "04_chords_omnizart")
                if midi_path and midi_path.exists():
                    (out_dir / "04_chords.mid").write_bytes(midi_path.read_bytes())
                if csv_path and csv_path.exists():
                    (out_dir / "04_chords.csv").write_bytes(csv_path.read_bytes())
            else:
                from .harmony.chords_template import chords_to_notes, estimate_chords

                chord_events = estimate_chords(raw_notes if raw_notes else melody_notes)
                chord_notes = chords_to_notes(chord_events)
                save_midi_notes(chord_notes, out_dir / "04_chords.mid", program=0, name="chords")
                csv_out = out_dir / "04_chords.csv"
                csv_out.parent.mkdir(parents=True, exist_ok=True)
                with csv_out.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["start_s", "end_s", "label"])
                    for ev in chord_events:
                        w.writerow([float(ev.start), float(ev.end), ev.label])
        except Exception as e:
            (out_dir / "04_chords_error.txt").write_text(str(e), encoding="utf-8")

    # --- 4) Preview ---
    if make_preview:
        render_previews(pre44, melody_notes, out_dir=out_dir)

    # --- 5) Summary ---
    print("\n===== 处理完成 =====")
    print(f"输入: {input_audio}")
    print(f"输出目录: {out_dir}")
    print(f"旋律 MIDI: {out_mel_mid}")
    print(f"旋律 JSON: {out_mel_json}")
    if make_preview:
        print(f"试听(旋律): {out_dir / '03_preview_melody.wav'}")
        print(f"试听(混音): {out_dir / '03_preview_mix.wav'}")
    if chords and chords != "none":
        print(f"和声/和弦: {out_dir / '04_chords.mid'} (以及 04_chords.csv)")

    print("\n===== 调性（用于意图还原）=====" )
    print(f"{key_info.name} (confidence={key_info.confidence:.3f})")

    print("\n===== 统计 =====")
    for k, v in stats.items():
        print(f"{k}: {v}")
