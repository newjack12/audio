from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="melody_app",
        description=(
            "Industrial-grade: 音频(人声/哼唱/弦乐器/钢琴) -> 旋律 MIDI + (可选)和弦 + 试听 wav\n"
            "支持多个后端: CREPE/pYIN/BasicPitch/piano_transcription_inference"
        ),
    )

    p.add_argument("input", type=str, help="输入音频路径（m4a/mp3/wav 等）")
    p.add_argument("--out", type=str, default="out_melody", help="输出目录")

    p.add_argument(
        "--mode",
        type=str,
        default="voice",
        choices=["voice", "string", "piano", "poly", "auto"],
        help="voice: 人声/哼唱；string: 单旋律弦乐器；piano: 钢琴；poly: 其他复调；auto: 自动(保守)",
    )

    p.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["auto", "crepe", "pyin", "basic_pitch", "piano_transcription"],
        help="推理后端。auto 会按 mode 选择最优可用后端。",
    )

    p.add_argument(
        "--chords",
        type=str,
        default="template",
        choices=["none", "template", "omnizart"],
        help="和弦/和声输出：template=轻量模板法；omnizart=Harmony Transformer(质量高)；none=关闭",
    )

    p.add_argument("--device", type=str, default="cpu", help="Torch 后端设备: cpu 或 cuda")

    # Melody cleanup
    p.add_argument("--min_note_len", type=float, default=None, help="最短音符时长（秒）。默认按 mode 自动选")
    p.add_argument("--merge_gap", type=float, default=0.05, help="同音合并：间隔小于该值(秒)则合并")
    p.add_argument("--frame_hop", type=float, default=0.02, help="DP 压单旋律时的帧步长(秒)")

    # Key snapping
    p.add_argument("--snap_scale", action="store_true", help="开启：估计调性并把音符吸附到调内音级")
    p.add_argument("--max_snap", type=int, default=1, help="音级吸附最大半音调整量")
    p.add_argument("--key", type=str, default="", help="手动指定调性，如 Cmaj/Amin/D#maj (空则自动)")

    p.add_argument("--no_preview", action="store_true", help="不生成 preview wav（默认生成）")

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    run_pipeline(
        input_audio=input_path,
        out_dir=out_dir,
        mode=str(args.mode),
        engine=str(args.engine),
        chords=str(args.chords),
        make_preview=(not args.no_preview),
        min_note_len=args.min_note_len,
        merge_gap=float(args.merge_gap),
        frame_hop=float(args.frame_hop),
        snap_scale=bool(args.snap_scale),
        max_snap=int(args.max_snap),
        key_override=(args.key.strip() or None),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
