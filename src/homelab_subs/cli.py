# src/homelab_subs/cli.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from .core.audio import FFmpeg, FFmpegError
from .core.srt import write_srt_file
from .core.transcription import Transcriber, TranscriberConfig, TranscriptionTask


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subsvc",
        description="Generate subtitles from video files using ffmpeg and faster-whisper.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- generate ----
    gen = subparsers.add_parser(
        "generate",
        help="Generate subtitles for a single video file.",
    )
    gen.add_argument(
        "video",
        type=Path,
        help="Path to the input video file.",
    )
    gen.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output .srt path. "
        "Defaults to <video_dir>/<video_name>.<lang>.srt",
    )
    gen.add_argument(
        "--lang",
        default="en",
        help="Language code for transcription (default: en). Use None/empty for auto-detect.",
    )
    gen.add_argument(
        "--model",
        default="small",
        help='Whisper model name (e.g. "tiny", "base", "small", "medium", "large-v2"). '
        "Default: small",
    )
    gen.add_argument(
        "--device",
        default="cpu",
        help='Device to use (e.g. "cpu" or "cuda"). Default: cpu',
    )
    gen.add_argument(
        "--compute-type",
        dest="compute_type",
        default="int8",
        help='Compute type for faster-whisper (e.g. "int8", "int8_float16", "float16"). '
        "Default: int8",
    )
    gen.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help='Task: "transcribe" keeps language, "translate" outputs English. Default: transcribe',
    )
    gen.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (higher = better, slower). Default: 5",
    )
    gen.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering (voice activity detection). Enabled by default.",
    )

    # ---- batch ----
    batch = subparsers.add_parser(
        "batch",
        help="Run multiple subtitle generation jobs from a YAML config file.",
    )
    batch.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file describing jobs.",
    )

    return parser


def _default_output_path(video_path: Path, lang: str) -> Path:
    """
    Derive a default output SRT path based on video name and language.
    e.g. /path/Movie.mkv -> /path/Movie.en.srt
    """
    video_path = Path(video_path)
    suffix_lang = (lang or "auto").lower()
    return video_path.with_suffix(f".{suffix_lang}.srt")


def _run_generate(
    video_path: Path,
    output_path: Optional[Path],
    lang: Optional[str],
    model_name: str,
    device: str,
    compute_type: str,
    task: TranscriptionTask,
    beam_size: int,
    vad_filter: bool,
) -> Path:
    """
    End-to-end generation: video -> audio -> transcription -> SRT.
    """
    ff = FFmpeg()

    if output_path is None:
        output_path = _default_output_path(video_path, lang or "auto")

    # Extract audio
    audio_path = ff.extract_audio_to_wav(video_path)

    # Configure transcriber
    config = TranscriberConfig(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    transcriber = Transcriber(config=config)

    # If lang is an empty string, treat as auto-detect
    language_param: Optional[str] = lang if lang else None

    segments = transcriber.transcribe_file(
        audio_path,
        language=language_param,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    # Write SRT
    return write_srt_file(segments, output_path)


def _run_batch(config_path: Path) -> None:
    """
    Run multiple jobs defined in a YAML config.

    Example YAML:

    jobs:
      - file: /media/movies/Movie.2023.mkv
        lang: en
        model: small
        output: /subs/Movie.2023.en.srt
      - file: /media/shows/Show.S01E01.mkv
        lang: en
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data: dict[str, Any] = yaml.safe_load(config_path.read_text())
    jobs = data.get("jobs") or []

    if not isinstance(jobs, list):
        raise ValueError("Config file must contain a 'jobs' list.")

    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each job in 'jobs' must be a mapping/dict.")

        file_path = job.get("file")
        if not file_path:
            raise ValueError("Job is missing required 'file' field.")

        video = Path(file_path)
        lang = job.get("lang", "en")
        model = job.get("model", "small")
        device = job.get("device", "cpu")
        compute_type = job.get("compute_type", "int8")
        task = job.get("task", "transcribe")
        beam_size = int(job.get("beam_size", 5))
        vad_filter = bool(job.get("vad_filter", True))

        output = job.get("output")
        output_path = Path(output) if output else None

        print(f"[subsvc] Processing job for video: {video}")
        srt_path = _run_generate(
            video_path=video,
            output_path=output_path,
            lang=lang,
            model_name=model,
            device=device,
            compute_type=compute_type,
            task=task,  # type: ignore[arg-type]
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        print(f"[subsvc] -> Created subtitles: {srt_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            srt_path = _run_generate(
                video_path=args.video,
                output_path=args.output,
                lang=args.lang,
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                task=args.task,
                beam_size=args.beam_size,
                vad_filter=not args.no_vad,
            )
            print(f"[subsvc] Subtitles written to: {srt_path}")
            return 0

        if args.command == "batch":
            _run_batch(args.config)
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 1

    except (FFmpegError, FileNotFoundError, ValueError) as exc:
        print(f"[subsvc] Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[subsvc] Interrupted by user.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
