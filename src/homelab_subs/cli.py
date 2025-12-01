# src/homelab_subs/cli.py

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from .core.audio import FFmpeg, FFmpegError
from .core.srt import write_srt_file
from .core.transcription import Transcriber, TranscriberConfig, TranscriptionTask
from .logging_config import setup_logging, get_logger, log_stage, log_file_info

logger = get_logger(__name__)


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
    gen.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    gen.add_argument(
        "--log-file",
        dest="log_file",
        type=Path,
        help="Optional log file path for JSON-formatted logs",
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
    batch.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    batch.add_argument(
        "--log-file",
        dest="log_file",
        type=Path,
        help="Optional log file path for JSON-formatted logs",
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
    job_id: Optional[str] = None,
) -> Path:
    """
    End-to-end generation: video -> audio -> transcription -> SRT.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())[:8]

    context = {
        "job_id": job_id,
        "video_file": str(video_path.name),
    }

    logger.info(f"Starting subtitle generation for {video_path.name}", extra=context)
    log_file_info(logger, video_path, context)

    ff = FFmpeg()

    if output_path is None:
        output_path = _default_output_path(video_path, lang or "auto")

    # Extract audio
    with log_stage(logger, "audio_extraction", **context):
        audio_path = ff.extract_audio_to_wav(video_path)
        log_file_info(logger, audio_path, context)

    # Configure transcriber
    config = TranscriberConfig(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    transcriber = Transcriber(config=config)

    # If lang is an empty string, treat as auto-detect
    language_param: Optional[str] = lang if lang else None

    logger.info(
        f"Transcribing with model={model_name}, device={device}, lang={language_param or 'auto'}",
        extra={**context, "model": model_name, "device": device, "language": language_param},
    )

    with log_stage(logger, "transcription", **context):
        segments = transcriber.transcribe_file(
            audio_path,
            language=language_param,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

    # Write SRT
    with log_stage(logger, "srt_generation", **context):
        result_path = write_srt_file(segments, output_path)
        log_file_info(logger, result_path, context)

    logger.info(
        f"Successfully generated subtitles: {result_path.name}",
        extra={**context, "output_file": str(result_path)},
    )

    return result_path


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

    logger.info(f"Loading batch configuration from {config_path}")

    data: dict[str, Any] = yaml.safe_load(config_path.read_text())
    jobs = data.get("jobs") or []

    if not isinstance(jobs, list):
        raise ValueError("Config file must contain a 'jobs' list.")

    logger.info(f"Found {len(jobs)} job(s) to process")

    successful = 0
    failed = 0

    for idx, job in enumerate(jobs, start=1):
        job_id = f"batch_{idx}"

        if not isinstance(job, dict):
            logger.error(f"Job {idx} is not a valid dictionary, skipping")
            failed += 1
            continue

        file_path = job.get("file")
        if not file_path:
            logger.error(f"Job {idx} is missing required 'file' field, skipping")
            failed += 1
            continue

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

        logger.info(f"Processing job {idx}/{len(jobs)}: {video.name}", extra={"job_id": job_id})

        try:
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
                job_id=job_id,
            )
            logger.info(f"Job {idx} completed successfully: {srt_path}", extra={"job_id": job_id})
            successful += 1
        except Exception as exc:
            logger.error(f"Job {idx} failed: {exc}", extra={"job_id": job_id}, exc_info=True)
            failed += 1

    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_file = getattr(args, "log_file", None)
    log_level = getattr(args, "log_level", "INFO")
    setup_logging(level=log_level, log_file=log_file)

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
            logger.info(f"✓ Subtitles written to: {srt_path}")
            return 0

        if args.command == "batch":
            _run_batch(args.config)
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 1

    except (FFmpegError, FileNotFoundError, ValueError) as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
