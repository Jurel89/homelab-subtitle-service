# src/homelab_subs/cli.py

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm
import yaml

from .core.audio import FFmpegError
from .core.transcription import TranscriptionTask
from .logging_config import setup_logging, get_logger, log_file_info
from .services.job_service import JobService

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
    gen.add_argument(
        "--no-monitoring",
        dest="enable_monitoring",
        action="store_false",
        help="Disable performance monitoring (CPU, memory, GPU tracking)",
    )
    gen.add_argument(
        "--no-db-logging",
        dest="enable_db_logging",
        action="store_false",
        help="Disable database logging (job history and metrics storage)",
    )
    gen.add_argument(
        "--db-path",
        dest="db_path",
        type=Path,
        help="Custom database path (default: ~/.homelab-subs/logs.db)",
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

    # ---- history ----
    history = subparsers.add_parser(
        "history",
        help="View job history and statistics from database logs.",
    )
    history.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent jobs to display (default: 20)",
    )
    history.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed"],
        help="Filter jobs by status",
    )
    history.add_argument(
        "--db-path",
        dest="db_path",
        type=Path,
        help="Custom database path (default: ~/.homelab-subs/logs.db)",
    )
    history.add_argument(
        "--stats",
        action="store_true",
        help="Show overall statistics instead of job list",
    )
    history.add_argument(
        "--job-id",
        dest="job_id",
        help="Show detailed metrics for a specific job ID",
    )

    # ---- translate ----
    translate = subparsers.add_parser(
        "translate",
        help="Translate an existing SRT subtitle file to another language.",
    )
    translate.add_argument(
        "input",
        type=Path,
        help="Path to the input SRT file to translate.",
    )
    translate.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output SRT path. "
        "Defaults to <input_dir>/<input_name>.<target_lang>.srt",
    )
    translate.add_argument(
        "--source-lang",
        dest="source_lang",
        default="en",
        help="Source language code (ISO 639-1, e.g., 'en'). Default: en",
    )
    translate.add_argument(
        "--target-lang",
        dest="target_lang",
        required=True,
        help="Target language code (ISO 639-1, e.g., 'es', 'fr', 'de').",
    )
    translate.add_argument(
        "--backend",
        choices=["helsinki", "nllb"],
        default="nllb",
        help='Translation backend: "helsinki" (MarianMT) or "nllb" (NLLB-200). Default: nllb',
    )
    translate.add_argument(
        "--model",
        default=None,
        help="Specific model name (optional). For NLLB: 'facebook/nllb-200-distilled-600M' "
        "(default) or 'facebook/nllb-200-3.3B' (best quality).",
    )
    translate.add_argument(
        "--device",
        default="cpu",
        help='Device to use (e.g. "cpu" or "cuda"). Default: cpu',
    )
    translate.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="Batch size for translation (default: 8).",
    )
    translate.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    translate.add_argument(
        "--log-file",
        dest="log_file",
        type=Path,
        help="Optional log file path for JSON-formatted logs",
    )

    # ---- languages ----
    languages = subparsers.add_parser(
        "languages",
        help="List supported languages for translation.",
    )
    languages.add_argument(
        "--backend",
        choices=["helsinki", "nllb"],
        default="nllb",
        help='Translation backend to query: "helsinki" or "nllb". Default: nllb',
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
    enable_monitoring: bool = True,
    enable_db_logging: bool = True,
    db_path: Optional[Path] = None,
) -> Path:
    """
    End-to-end generation driven by the JobService orchestrator.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())[:8]

    context = {
        "job_id": job_id,
        "video_file": str(video_path.name),
    }

    logger.info(f"Starting subtitle generation for {video_path.name}", extra=context)
    log_file_info(logger, video_path, context)

    if output_path is None:
        output_path = _default_output_path(video_path, lang or "auto")

    service = JobService(
        enable_monitoring=enable_monitoring,
        enable_db_logging=enable_db_logging,
        db_path=db_path,
    )

    if enable_monitoring and not service.monitoring_available:
        logger.warning(
            "Monitoring requested but dependencies not installed. "
            "Install with: pip install psutil nvidia-ml-py"
        )

    if enable_db_logging and not service.db_logging_available:
        logger.warning(
            "Database logging requested but dependencies unavailable. Logs will not be recorded."
        )

    language_param: Optional[str] = lang if lang else None

    pbar = tqdm(total=100, unit="%", desc="Transcribing", leave=True)

    def progress_cb(pct: float, count: int) -> None:
        pbar.n = int(pct)
        pbar.refresh()
        logger.info(
            f"CLI progress: {pct:.1f}%",
            extra={**context, "progress": pct, "segment_count": count},
        )

    try:
        result_path = service.generate_subtitles(
            video_path=video_path,
            output_path=output_path,
            lang=language_param,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            job_id=job_id,
            progress_callback=progress_cb,
        )
        pbar.n = 100
        pbar.refresh()
        logger.info(
            f"Successfully generated subtitles: {result_path.name}",
            extra={**context, "output_file": str(result_path)},
        )
        return result_path
    finally:
        pbar.close()


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


def _run_history(
    limit: int = 20,
    status: Optional[str] = None,
    db_path: Optional[Path] = None,
    show_stats: bool = False,
    job_id: Optional[str] = None,
) -> None:
    """
    Display job history and statistics from database logs.
    """
    service = JobService(
        enable_monitoring=False,
        enable_db_logging=True,
        db_path=db_path,
    )

    try:
        if job_id:
            details = service.get_job_details(job_id)
            if details is None:
                logger.error(f"Job not found: {job_id}")
                return

            job = details["job"]
            metrics = details["metrics"]

            print(f"\n{'='*80}")
            print(f"Job Details: {job_id}")
            print(f"{'='*80}")
            print(f"Video:      {job.video_path}")
            print(f"Output:     {job.output_path}")
            print(f"Status:     {job.status}")
            print(f"Language:   {job.language}")
            print(f"Model:      {job.model}")
            print(f"Task:       {job.task}")
            print(f"Started:    {job.started_at}")
            print(f"Completed:  {job.completed_at or 'N/A'}")
            print(f"Duration:   {job.duration_seconds:.2f}s" if job.duration_seconds else "Duration:   N/A")

            if job.error_message:
                print(f"Error:      {job.error_message}")

            print(f"\n{'='*80}")
            print("Performance Summary")
            print(f"{'='*80}")
            print(f"CPU Avg:    {job.cpu_avg:.1f}%" if job.cpu_avg else "CPU Avg:    N/A")
            print(f"CPU Max:    {job.cpu_max:.1f}%" if job.cpu_max else "CPU Max:    N/A")
            print(f"Memory Avg: {job.memory_avg_mb:.1f} MB" if job.memory_avg_mb else "Memory Avg: N/A")
            print(f"Memory Max: {job.memory_max_mb:.1f} MB" if job.memory_max_mb else "Memory Max: N/A")
            print(f"GPU Avg:    {job.gpu_avg:.1f}%" if job.gpu_avg else "GPU Avg:    N/A")
            print(f"GPU Max:    {job.gpu_max:.1f}%" if job.gpu_max else "GPU Max:    N/A")

            if metrics:
                print(f"\n{'='*80}")
                print(f"Metrics Timeline ({len(metrics)} samples)")
                print(f"{'='*80}")
                print(f"{'Time':>8} | {'CPU%':>6} | {'Memory%':>8} | {'GPU%':>6} | {'GPU Mem MB':>10}")
                print(f"{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")

                for i, m in enumerate(metrics):
                    if i >= 10:
                        print(f"... ({len(metrics) - 10} more samples)")
                        break
                    time_str = m.timestamp.strftime("%H:%M:%S")
                    gpu_str = f"{m.gpu_utilization:>6.1f}" if m.gpu_utilization else "   N/A"
                    gpu_mem_str = f"{m.gpu_memory_used_mb:>10.1f}" if m.gpu_memory_used_mb else "       N/A"
                    print(
                        f"{time_str} | {m.cpu_percent:>6.1f} | {m.memory_percent:>8.1f} | {gpu_str} | {gpu_mem_str}"
                    )

            print()
            return

        if show_stats:
            stats = service.get_statistics()

            print(f"\n{'='*80}")
            print("Overall Statistics")
            print(f"{'='*80}")
            print(f"Total Jobs: {stats['total_jobs']}")
            print("\nStatus Breakdown:")
            for status_name, count in stats["status_counts"].items():
                print(f"  {status_name:12} : {count:>5}")

            if stats["avg_duration_seconds"]:
                print("\nDuration:")
                print(f"  Average: {stats['avg_duration_seconds']:.2f}s")
                print(f"  Min:     {stats['min_duration_seconds']:.2f}s")
                print(f"  Max:     {stats['max_duration_seconds']:.2f}s")

            if stats["avg_cpu_percent"]:
                print("\nPerformance Averages:")
                print(f"  CPU:    {stats['avg_cpu_percent']:.1f}%")
                print(f"  Memory: {stats['avg_memory_mb']:.1f} MB")
                if stats["avg_gpu_percent"]:
                    print(f"  GPU:    {stats['avg_gpu_percent']:.1f}%")

            print()
            return

        jobs = service.get_recent_jobs(limit=limit, status=status)

        if not jobs:
            print(f"\nNo jobs found{' with status: ' + status if status else ''}.")
            return

        print(f"\n{'='*120}")
        print(f"Recent Jobs ({len(jobs)}{' with status: ' + status if status else ''})")
        print(f"{'='*120}")
        print(f"{'Job ID':>10} | {'Status':>10} | {'Video':45} | {'Duration':>10} | {'CPU Avg':>8} | {'Mem Avg':>9}")
        print(f"{'-'*10}-+-{'-'*10}-+-{'-'*45}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}")

        for job in jobs:
            video_name = Path(job.video_path).name
            if len(video_name) > 45:
                video_name = video_name[:42] + "..."

            duration_str = f"{job.duration_seconds:.2f}s" if job.duration_seconds else "N/A"
            cpu_str = f"{job.cpu_avg:.1f}%" if job.cpu_avg else "N/A"
            mem_str = f"{job.memory_avg_mb:.0f} MB" if job.memory_avg_mb else "N/A"

            print(
                f"{job.job_id:>10} | {job.status:>10} | {video_name:45} | {duration_str:>10} | {cpu_str:>8} | {mem_str:>9}"
            )

        print("\nUse 'subsvc history --job-id <id>' to see detailed metrics for a specific job.")
        print()

    except RuntimeError as exc:
        logger.error(str(exc))


def _run_translate(
    input_path: Path,
    output_path: Optional[Path],
    source_lang: str,
    target_lang: str,
    backend: str,
    model_name: Optional[str],
    device: str,
    batch_size: int,
) -> Path:
    """
    Translate an SRT subtitle file to another language.
    """
    try:
        from .core.translation import Translator, TranslatorConfig, TranslationBackend
    except ImportError as e:
        raise ImportError(
            "Translation requires additional dependencies. "
            "Install with: pip install homelab-subtitle-service[translation]"
        ) from e

    context = {
        "input_file": str(input_path.name),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "backend": backend,
    }

    logger.info(
        f"Starting translation: {input_path.name} ({source_lang} -> {target_lang})",
        extra=context,
    )

    if output_path is None:
        # Default output: <input_dir>/<input_name>.<target_lang>.srt
        output_path = input_path.with_suffix(f".{target_lang}.srt")

    config = TranslatorConfig(
        backend=backend,  # type: ignore[arg-type]
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    translator = Translator(config=config)

    # Check if language pair is supported
    if not translator.is_language_pair_supported(source_lang, target_lang):
        raise ValueError(
            f"Language pair '{source_lang}' -> '{target_lang}' not supported "
            f"with backend '{backend}'. Use 'subsvc languages --backend {backend}' "
            "to see supported languages."
        )

    pbar = tqdm(total=100, unit="%", desc="Translating", leave=True)

    def progress_cb(pct: float, count: int) -> None:
        pbar.n = int(pct)
        pbar.refresh()

    try:
        result_path = translator.translate_srt_file(
            input_path=input_path,
            output_path=output_path,
            source_lang=source_lang,
            target_lang=target_lang,
            progress_callback=progress_cb,
        )
        pbar.n = 100
        pbar.refresh()
        logger.info(
            f"Successfully translated subtitles: {result_path.name}",
            extra={**context, "output_file": str(result_path)},
        )
        return result_path
    finally:
        pbar.close()


def _run_list_languages(backend: str) -> None:
    """
    List supported languages for a translation backend.
    """
    try:
        from .core.translation import list_supported_languages, TranslationBackend
    except ImportError as e:
        raise ImportError(
            "Translation requires additional dependencies. "
            "Install with: pip install homelab-subtitle-service[translation]"
        ) from e

    languages = list_supported_languages(backend)  # type: ignore[arg-type]

    print(f"\nSupported languages for '{backend}' backend:")
    print(f"{'='*50}")
    print(f"{'Code':<8} | {'Language':<30}")
    print(f"{'-'*8}-+-{'-'*30}")

    for code, name in sorted(languages.items()):
        print(f"{code:<8} | {name:<30}")

    print(f"\nTotal: {len(languages)} languages")
    print()


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
                enable_monitoring=args.enable_monitoring,
                enable_db_logging=args.enable_db_logging,
                db_path=args.db_path,
            )
            logger.info(f"✓ Subtitles written to: {srt_path}")
            return 0

        if args.command == "batch":
            _run_batch(args.config)
            return 0

        if args.command == "history":
            _run_history(
                limit=args.limit,
                status=args.status,
                db_path=args.db_path,
                show_stats=args.stats,
                job_id=args.job_id,
            )
            return 0

        if args.command == "translate":
            srt_path = _run_translate(
                input_path=args.input,
                output_path=args.output,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                backend=args.backend,
                model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
            )
            logger.info(f"✓ Translated subtitles written to: {srt_path}")
            return 0

        if args.command == "languages":
            _run_list_languages(backend=args.backend)
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
