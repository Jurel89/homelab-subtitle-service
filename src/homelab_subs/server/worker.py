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
from homelab_subs.core.audio import AudioExtractor
from homelab_subs.core.transcription import WhisperTranscriber
from homelab_subs.core.translation import SubtitleTranslator
from homelab_subs.core.srt import SRTParser, SRTGenerator
from homelab_subs.core.sync import SubtitleSynchronizer
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
        return self.job.status == JobStatus.CANCELLED

    def raise_if_cancelled(self):
        """Raise JobCancelledException if job was cancelled."""
        if self.check_cancelled():
            raise JobCancelledException(f"Job {self.job_id} was cancelled")

    def update_stage(self, stage: JobStage, progress: int = 0):
        """Update job stage and progress."""
        self.job.stage = stage
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
        self.job.status = JobStatus.COMPLETED
        self.job.stage = JobStage.COMPLETED
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
            elif job_type == JobType.SYNC:
                result = _process_sync(ctx)
            elif job_type == JobType.COMPARE:
                result = _process_comparison(ctx)
            elif job_type == JobType.FULL_PIPELINE:
                result = _process_full_pipeline(ctx)
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
    input_path = Path(ctx.job.input_path)
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_suffix(".srt")
    )

    # Get options from job
    options = ctx.job.options or {}
    model_size = ctx.job.model_size or "base"
    compute_type = ctx.job.compute_type or "float16"

    # Stage 1: Extract audio (if video file)
    ctx.update_stage(JobStage.EXTRACTING_AUDIO, 0)
    ctx.add_log(f"Extracting audio from {input_path}")
    ctx.raise_if_cancelled()

    audio_extractor = AudioExtractor()
    audio_path = ctx.create_temp_file(suffix=".wav")

    # Check if input is video (needs audio extraction) or already audio
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    if input_path.suffix.lower() in video_extensions:
        audio_extractor.extract(str(input_path), str(audio_path))
        ctx.add_log("Audio extracted to temporary file")
    else:
        # Assume it's already an audio file
        audio_path = input_path

    ctx.update_progress(20)
    ctx.raise_if_cancelled()

    # Stage 2: Transcribe audio
    ctx.update_stage(JobStage.TRANSCRIBING, 20)
    ctx.add_log(f"Transcribing with model={model_size}, compute_type={compute_type}")

    transcriber = WhisperTranscriber(
        model_size=model_size,
        compute_type=compute_type,
        device=options.get("device", "auto"),
    )

    # Create progress callback
    def transcription_progress(current: int, total: int):
        # Map transcription progress to 20-80% of job progress
        if total > 0:
            pct = 20 + int((current / total) * 60)
            ctx.update_progress(pct)

    # Transcribe
    source_language = ctx.job.source_language
    segments = transcriber.transcribe(
        str(audio_path),
        language=source_language,
        task="transcribe",
    )

    ctx.update_progress(80)
    ctx.raise_if_cancelled()

    # Stage 3: Generate SRT
    ctx.update_stage(JobStage.GENERATING_SRT, 80)
    ctx.add_log(f"Generating SRT file at {output_path}")

    generator = SRTGenerator()
    generator.generate(segments, str(output_path))

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
    input_path = Path(ctx.job.input_path)
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_stem(f"{input_path.stem}_{ctx.job.target_language}")
    )

    options = ctx.job.options or {}
    source_lang = ctx.job.source_language or "auto"
    target_lang = ctx.job.target_language

    if not target_lang:
        raise ValueError("Target language is required for translation jobs")

    # Stage 1: Parse source SRT
    ctx.update_stage(JobStage.TRANSLATING, 0)
    ctx.add_log(f"Parsing source SRT: {input_path}")
    ctx.raise_if_cancelled()

    parser = SRTParser()
    cues = parser.parse(str(input_path))
    ctx.add_log(f"Parsed {len(cues)} subtitle cues")
    ctx.update_progress(10)

    # Stage 2: Translate
    ctx.add_log(f"Translating from {source_lang} to {target_lang}")

    translator = SubtitleTranslator(
        model_name=options.get("translation_model", "Helsinki-NLP/opus-mt-en-es"),
        source_lang=source_lang,
        target_lang=target_lang,
    )

    translated_cues = translator.translate_cues(cues)
    ctx.update_progress(80)
    ctx.raise_if_cancelled()

    # Stage 3: Generate translated SRT
    ctx.update_stage(JobStage.GENERATING_SRT, 80)
    ctx.add_log(f"Writing translated SRT to {output_path}")

    generator = SRTGenerator()
    generator.generate_from_cues(translated_cues, str(output_path))

    ctx.update_progress(100)
    ctx.add_log(f"Translation complete: {len(translated_cues)} cues")

    return {
        "output_path": str(output_path),
        "cues_count": len(translated_cues),
    }


def _process_sync(ctx: JobContext) -> dict:
    """
    Process a sync job.

    Synchronizes subtitles with audio.
    Stages: EXTRACTING_AUDIO -> SYNCING -> GENERATING_SRT
    """
    input_path = Path(ctx.job.input_path)
    reference_path = Path(ctx.job.reference_path) if ctx.job.reference_path else None
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_stem(f"{input_path.stem}_synced")
    )

    if not reference_path:
        raise ValueError("Reference audio/video path is required for sync jobs")

    options = ctx.job.options or {}

    # Stage 1: Extract audio if needed
    ctx.update_stage(JobStage.EXTRACTING_AUDIO, 0)
    ctx.add_log(f"Preparing audio from {reference_path}")
    ctx.raise_if_cancelled()

    audio_extractor = AudioExtractor()
    audio_path = ctx.create_temp_file(suffix=".wav")

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    if reference_path.suffix.lower() in video_extensions:
        audio_extractor.extract(str(reference_path), str(audio_path))
    else:
        audio_path = reference_path

    ctx.update_progress(20)
    ctx.raise_if_cancelled()

    # Stage 2: Parse source subtitles
    ctx.update_stage(JobStage.SYNCING, 20)
    ctx.add_log(f"Parsing subtitles from {input_path}")

    parser = SRTParser()
    cues = parser.parse(str(input_path))
    ctx.add_log(f"Parsed {len(cues)} subtitle cues")
    ctx.update_progress(30)

    # Stage 3: Synchronize
    ctx.add_log("Synchronizing subtitles with audio")

    synchronizer = SubtitleSynchronizer()
    synced_cues = synchronizer.sync(
        cues=cues,
        audio_path=str(audio_path),
        model_size=options.get("sync_model", "tiny"),
    )

    ctx.update_progress(80)
    ctx.raise_if_cancelled()

    # Stage 4: Generate synced SRT
    ctx.update_stage(JobStage.GENERATING_SRT, 80)
    ctx.add_log(f"Writing synced SRT to {output_path}")

    generator = SRTGenerator()
    generator.generate_from_cues(synced_cues, str(output_path))

    ctx.update_progress(100)
    ctx.add_log(f"Sync complete: {len(synced_cues)} cues")

    return {
        "output_path": str(output_path),
        "cues_count": len(synced_cues),
    }


def _process_comparison(ctx: JobContext) -> dict:
    """
    Process a comparison job.

    Compares two SRT files and generates accuracy metrics.
    Stages: COMPARING
    """
    input_path = Path(ctx.job.input_path)  # Reference (human) subtitles
    reference_path = Path(ctx.job.reference_path)  # Hypothesis (machine) subtitles
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_stem(f"{input_path.stem}_comparison").with_suffix(".json")
    )

    if not reference_path:
        raise ValueError(
            "Reference path (hypothesis SRT) is required for comparison jobs"
        )

    # Note: options available via ctx.job.options if needed for future extensions

    ctx.update_stage(JobStage.COMPARING, 0)
    ctx.add_log(f"Comparing {input_path} with {reference_path}")
    ctx.raise_if_cancelled()

    # Create comparator
    comparator = SubtitleComparator()

    ctx.update_progress(20)

    # Run comparison
    result = comparator.compare_files(
        reference_path=str(input_path),
        hypothesis_path=str(reference_path),
    )

    ctx.update_progress(80)
    ctx.raise_if_cancelled()

    # Save results
    ctx.add_log(f"Writing comparison results to {output_path}")

    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    ctx.update_progress(100)
    ctx.add_log(
        f"Comparison complete - WER: {result.text_metrics.wer:.2%}, CER: {result.text_metrics.cer:.2%}"
    )

    return {
        "output_path": str(output_path),
        "wer": result.text_metrics.wer,
        "cer": result.text_metrics.cer,
    }


def _process_full_pipeline(ctx: JobContext) -> dict:
    """
    Process a full pipeline job.

    Complete workflow: video -> audio -> transcription -> (optional translation) -> SRT
    Stages: EXTRACTING_AUDIO -> TRANSCRIBING -> TRANSLATING (optional) -> GENERATING_SRT
    """
    input_path = Path(ctx.job.input_path)
    output_path = (
        Path(ctx.job.output_path)
        if ctx.job.output_path
        else input_path.with_suffix(".srt")
    )

    options = ctx.job.options or {}
    model_size = ctx.job.model_size or "base"
    compute_type = ctx.job.compute_type or "float16"
    source_lang = ctx.job.source_language
    target_lang = ctx.job.target_language

    # Stage 1: Extract audio
    ctx.update_stage(JobStage.EXTRACTING_AUDIO, 0)
    ctx.add_log(f"Extracting audio from {input_path}")
    ctx.raise_if_cancelled()

    audio_extractor = AudioExtractor()
    audio_path = ctx.create_temp_file(suffix=".wav")

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    if input_path.suffix.lower() in video_extensions:
        audio_extractor.extract(str(input_path), str(audio_path))
        ctx.add_log("Audio extracted successfully")
    else:
        audio_path = input_path

    ctx.update_progress(10)
    ctx.raise_if_cancelled()

    # Stage 2: Transcribe
    ctx.update_stage(JobStage.TRANSCRIBING, 10)
    ctx.add_log(f"Transcribing with model={model_size}, compute_type={compute_type}")

    transcriber = WhisperTranscriber(
        model_size=model_size,
        compute_type=compute_type,
        device=options.get("device", "auto"),
    )

    segments = transcriber.transcribe(
        str(audio_path),
        language=source_lang,
        task="transcribe",
    )

    ctx.update_progress(50)
    ctx.add_log(f"Transcription complete: {len(segments)} segments")
    ctx.raise_if_cancelled()

    # Stage 3: Translate (if target language specified)
    cues = segments  # Will be replaced if translation happens

    if target_lang and target_lang != source_lang:
        ctx.update_stage(JobStage.TRANSLATING, 50)
        ctx.add_log(f"Translating from {source_lang or 'auto'} to {target_lang}")

        translator = SubtitleTranslator(
            source_lang=source_lang or "auto",
            target_lang=target_lang,
        )

        cues = translator.translate_cues(segments)
        ctx.update_progress(80)
        ctx.add_log(f"Translation complete: {len(cues)} cues")

        # Update output path to include target language
        if not ctx.job.output_path:
            output_path = input_path.with_stem(
                f"{input_path.stem}.{target_lang}"
            ).with_suffix(".srt")
    else:
        ctx.update_progress(80)

    ctx.raise_if_cancelled()

    # Stage 4: Generate SRT
    ctx.update_stage(JobStage.GENERATING_SRT, 80)
    ctx.add_log(f"Generating SRT file at {output_path}")

    generator = SRTGenerator()
    generator.generate_from_cues(cues, str(output_path))

    ctx.update_progress(100)
    ctx.add_log("SRT generated successfully")

    return {
        "output_path": str(output_path),
        "segments_count": len(cues),
        "translated": bool(target_lang and target_lang != source_lang),
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
