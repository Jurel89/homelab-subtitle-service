"""
RQ Worker task implementation.

This module contains the process_job function that is executed by RQ workers
to process subtitle generation jobs through the pipeline stages.
"""

import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from redis import Redis
from rq import Worker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from homelab_subs.server.settings import get_settings
from homelab_subs.server.models import Job, JobStatus, JobStage, JobType
from homelab_subs.core.audio import FFmpeg
from homelab_subs.core.transcription import Transcriber, TranscriberConfig
from homelab_subs.core.translation import Translator, TranslatorConfig
from homelab_subs.core.srt import segments_to_srt, write_srt_file
from homelab_subs.core.sync import SubtitleSyncer, parse_srt_content
from homelab_subs.core.comparison import SubtitleComparator

logger = logging.getLogger(__name__)


class JobCancelledException(Exception):
    """Raised when a job is cancelled during processing."""

    pass


class JobContext:
    """
    Context manager for job processing.

    Provides database session, job model, and helper methods for
    updating job progress and checking cancellation status.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.sync_database_url,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        self.Session = sessionmaker(bind=self.engine)
        self.session: Optional[Any] = None
        self.job: Optional[Job] = None
        self._temp_files: list[Path] = []

    def __enter__(self) -> "JobContext":
        self.session = self.Session()
        self.job = self.session.query(Job).filter(Job.id == self.job_id).first()
        if not self.job:
            raise ValueError(f"Job not found: {self.job_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()

        return False  # Don't suppress exceptions

    def check_cancelled(self) -> bool:
        """Check if job has been cancelled. Refreshes from database."""
        self.session.refresh(self.job)
        return self.job.status == JobStatus.CANCELED

    def raise_if_cancelled(self):
        """Raise JobCancelledException if job was cancelled."""
        if self.check_cancelled():
            raise JobCancelledException(f"Job {self.job_id} was cancelled")

    def update_stage(self, stage: JobStage, progress: int = 0):
        """Update job stage and progress."""
        self.job.current_stage = stage
        self.job.progress = progress
        self.session.commit()
        logger.info(f"Job {self.job_id}: stage={stage.value}, progress={progress}%")

    def update_progress(self, progress: int):
        """Update job progress percentage (0-100)."""
        self.job.progress = min(100, max(0, progress))
        self.session.commit()

    def add_log(self, message: str, level: str = "info"):
        """Append a log message to the job."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"

        if self.job.logs is None:
            self.job.logs = log_entry
        else:
            self.job.logs += f"\n{log_entry}"
        self.session.commit()

    def complete(self, output_path: Optional[str] = None):
        """Mark job as completed."""
        self.job.status = JobStatus.DONE
        self.job.current_stage = JobStage.COMPLETED
        self.job.progress = 100
        if output_path:
            self.job.output_path = output_path
        self.session.commit()
        logger.info(f"Job {self.job_id} completed successfully")

    def fail(self, error_message: str):
        """Mark job as failed with error message."""
        self.job.status = JobStatus.FAILED
        self.job.error_message = error_message
        self.add_log(error_message, level="error")
        self.session.commit()
        logger.error(f"Job {self.job_id} failed: {error_message}")

    def create_temp_file(self, suffix: str = "") -> Path:
        """Create a temporary file that will be cleaned up on exit."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        temp_path = Path(path)
        self._temp_files.append(temp_path)
        return temp_path


def process_job(job_id: str) -> dict:
    """
    Main job processing function executed by RQ workers.

    This function:
    1. Loads job from database
    2. Runs the appropriate pipeline based on job type
    3. Updates progress at each stage
    4. Handles cancellation gracefully
    5. Records success or failure

    Args:
        job_id: UUID of the job to process

    Returns:
        dict with status and output_path on success

    Raises:
        JobCancelledException: If job was cancelled
        Exception: On any processing error
    """
    logger.info(f"Starting job processing: {job_id}")

    with JobContext(job_id) as ctx:
        try:
            # Check if already cancelled before starting
            ctx.raise_if_cancelled()

            # Update status to running
            ctx.job.status = JobStatus.RUNNING
            ctx.session.commit()
            ctx.add_log("Job started processing")

            # Route to appropriate processor based on job type
            job_type = ctx.job.type

            if job_type == JobType.TRANSCRIBE:
                result = _process_transcription(ctx)
            elif job_type == JobType.TRANSLATE:
                result = _process_translation(ctx)
            elif job_type == JobType.SYNC_SUBTITLE:
                result = _process_sync(ctx)
            elif job_type == JobType.COMPARE:
                result = _process_comparison(ctx)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            ctx.complete(result.get("output_path"))
            ctx.add_log(f"Job completed successfully: {result}")

            return {"status": "completed", **result}

        except JobCancelledException:
            ctx.add_log("Job was cancelled by user")
            return {"status": "cancelled", "job_id": job_id}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            ctx.fail(error_msg)
            raise


def _process_transcription(ctx: JobContext) -> dict:
    """
    Process a transcription job.

    Stages: EXTRACTING_AUDIO -> TRANSCRIBING -> GENERATING_SRT
    """
    input_path = Path(ctx.job.source_path)
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_suffix(".srt")
    )

    # Get options from job
    options = ctx.job.options or {}
    model_name = ctx.job.model_name or "small"
    compute_type = ctx.job.compute_type or "int8"
    device = ctx.job.device or "cpu"

    # Stage 1: Extract audio (if video file)
    ctx.update_stage(JobStage.EXTRACTING_AUDIO, 0)
    ctx.add_log(f"Extracting audio from {input_path}")
    ctx.raise_if_cancelled()

    ffmpeg = FFmpeg()

    # Check if input is video (needs audio extraction) or already audio
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    if input_path.suffix.lower() in video_extensions:
        audio_path = ffmpeg.extract_audio_to_wav(input_path)
        ctx.add_log("Audio extracted to temporary file")
    else:
        # Assume it's already an audio file
        audio_path = input_path

    ctx.update_progress(20)
    ctx.raise_if_cancelled()

    # Stage 2: Transcribe audio
    ctx.update_stage(JobStage.TRANSCRIBING, 20)
    ctx.add_log(f"Transcribing with model={model_name}, compute_type={compute_type}")

    config = TranscriberConfig(
        model_name=model_name,
        compute_type=compute_type,
        device=device,
    )
    transcriber = Transcriber(config=config)

    # Create progress callback
    def transcription_progress(pct: float, segment_count: int):
        # Map transcription progress to 20-80% of job progress
        job_pct = 20 + int(pct * 0.6)
        ctx.update_progress(job_pct)

    # Transcribe
    source_language = ctx.job.language
    task = options.get("task", "transcribe")
    segments = transcriber.transcribe_file(
        audio_path=audio_path,
        language=source_language,
        task=task,
        progress_callback=transcription_progress,
    )

    ctx.update_progress(80)
    ctx.raise_if_cancelled()

    # Stage 3: Generate SRT
    ctx.update_stage(JobStage.GENERATING_SRT, 80)
    ctx.add_log(f"Generating SRT file at {output_path}")

    write_srt_file(segments, output_path)

    ctx.update_progress(100)
    ctx.add_log(f"SRT generated with {len(segments)} segments")

    return {
        "output_path": str(output_path),
        "segments_count": len(segments),
    }


def _process_translation(ctx: JobContext) -> dict:
    """
    Process a translation job.

    Takes an existing SRT file and translates it.
    Stages: TRANSLATING -> GENERATING_SRT
    """
    input_path = Path(ctx.job.source_path)
    options = ctx.job.options or {}
    source_lang = ctx.job.language or "en"
    target_lang = ctx.job.target_language

    if not target_lang:
        raise ValueError("Target language is required for translation jobs")

    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_suffix(f".{target_lang}.srt")
    )

    # Stage 1: Translate SRT file
    ctx.update_stage(JobStage.TRANSLATING, 0)
    ctx.add_log(f"Translating {input_path} from {source_lang} to {target_lang}")
    ctx.raise_if_cancelled()

    translator = Translator(config=TranslatorConfig(
        backend=options.get("backend", "nllb"),
        device=ctx.job.device or "cpu",
    ))

    def translation_progress(pct: float, items_done: int):
        # Map translation progress to 0-90% of job progress
        job_pct = int(pct * 0.9)
        ctx.update_progress(job_pct)

    translator.translate_srt_file(
        input_path=input_path,
        output_path=output_path,
        source_lang=source_lang,
        target_lang=target_lang,
        progress_callback=translation_progress,
    )

    ctx.update_progress(100)
    ctx.raise_if_cancelled()

    ctx.add_log(f"Translation complete: {output_path}")

    return {
        "output_path": str(output_path),
    }


def _process_sync(ctx: JobContext) -> dict:
    """
    Process a sync job.

    Synchronizes subtitles with video audio.
    Stages: SYNCING
    """
    video_path = Path(ctx.job.source_path)
    subtitle_path = Path(ctx.job.subtitle_path) if ctx.job.subtitle_path else None
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else subtitle_path.with_stem(f"{subtitle_path.stem}.synced")
        if subtitle_path
        else video_path.with_suffix(".synced.srt")
    )

    if not subtitle_path:
        raise ValueError("Subtitle path is required for sync jobs")

    # Stage 1: Synchronize subtitles with video audio
    ctx.update_stage(JobStage.SYNCING, 0)
    ctx.add_log(f"Synchronizing {subtitle_path} with {video_path}")
    ctx.raise_if_cancelled()

    syncer = SubtitleSyncer()

    def sync_progress(pct: float, segment_count: int):
        ctx.update_progress(int(pct * 0.9))

    result = syncer.sync_subtitles(
        video_path=video_path,
        srt_path=subtitle_path,
        output_path=output_path,
        progress_callback=sync_progress,
    )

    ctx.update_progress(100)
    ctx.add_log(
        f"Sync complete: {result.matched_count} matched, "
        f"{result.interpolated_count} interpolated, "
        f"{result.unchanged_count} unchanged"
    )

    return {
        "output_path": str(output_path),
        "matched_count": result.matched_count,
        "interpolated_count": result.interpolated_count,
        "unchanged_count": result.unchanged_count,
    }


def _process_comparison(ctx: JobContext) -> dict:
    """
    Process a comparison job.

    Compares two SRT files and generates accuracy metrics.
    Stages: COMPARING
    """
    import json
    from dataclasses import asdict

    reference_path = Path(ctx.job.source_path)  # Reference (human) subtitles
    hypothesis_path = Path(ctx.job.subtitle_path)  # Hypothesis (machine) subtitles
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else reference_path.with_stem(f"{reference_path.stem}_comparison").with_suffix(".json")
    )

    if not ctx.job.subtitle_path:
        raise ValueError(
            "Subtitle path (hypothesis SRT) is required for comparison jobs"
        )

    ctx.update_stage(JobStage.COMPARING, 0)
    ctx.add_log(f"Comparing {reference_path} with {hypothesis_path}")
    ctx.raise_if_cancelled()

    # Create comparator
    comparator = SubtitleComparator()

    ctx.update_progress(20)

    # Run comparison
    result = comparator.compare_files(
        reference_path=reference_path,
        hypothesis_path=hypothesis_path,
    )

    ctx.update_progress(80)
    ctx.raise_if_cancelled()

    # Save results
    ctx.add_log(f"Writing comparison results to {output_path}")

    # Serialize result using dataclasses.asdict, converting Path objects to strings
    result_dict = asdict(result)
    result_dict.pop("segment_comparisons", None)  # Remove verbose per-segment data
    if result_dict.get("reference_path"):
        result_dict["reference_path"] = str(result_dict["reference_path"])
    if result_dict.get("hypothesis_path"):
        result_dict["hypothesis_path"] = str(result_dict["hypothesis_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    ctx.update_progress(100)
    ctx.add_log(
        f"Comparison complete - WER: {result.text_metrics.word_error_rate:.2%}, "
        f"CER: {result.text_metrics.character_error_rate:.2%}"
    )

    return {
        "output_path": str(output_path),
        "wer": result.text_metrics.word_error_rate,
        "cer": result.text_metrics.character_error_rate,
    }


def run_worker(
    queues: Optional[list[str]] = None,
    burst: bool = False,
    name: Optional[str] = None,
):
    """
    Run an RQ worker.

    This function is called from the CLI or as an entrypoint to start
    a worker process that listens for jobs.

    Args:
        queues: List of queue names to listen on (default: ['default'])
        burst: If True, worker quits after processing all jobs
        name: Worker name for identification
    """
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Connect to Redis
    redis_conn = Redis.from_url(settings.redis_url)

    # Determine queues
    if queues is None:
        queues = ["high", "default", "low"]

    logger.info(f"Starting worker on queues: {queues}")

    # Create and run worker
    from rq import Queue

    queue_instances = [Queue(name=q, connection=redis_conn) for q in queues]

    worker = Worker(
        queues=queue_instances,
        connection=redis_conn,
        name=name,
    )

    worker.work(burst=burst)


if __name__ == "__main__":
    # Allow running worker directly: python -m homelab_subs.server.worker
    run_worker()
