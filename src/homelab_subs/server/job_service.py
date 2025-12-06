# src/homelab_subs/server/job_service.py

"""
High-level job service for the server mode.

This service orchestrates job creation, queuing, and management
using PostgreSQL for persistence and RQ for job execution.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from .models import Job, JobStatus, JobType
from .repository import JobRepository
from .queue import QueueClient
from .settings import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class ServerJobService:
    """
    High-level service for managing subtitle jobs in server mode.

    This service combines the JobRepository (persistence) and
    QueueClient (execution) to provide a complete job management API.

    Parameters
    ----------
    repository : JobRepository, optional
        Database repository. If not provided, creates one from settings.
    queue_client : QueueClient, optional
        Queue client. If not provided, creates one from settings.
    settings : Settings, optional
        Application settings. If not provided, uses get_settings().

    Examples
    --------
    >>> service = ServerJobService()
    >>> job = service.create_job(
    ...     job_type=JobType.TRANSCRIBE,
    ...     source_path="/media/movie.mkv",
    ...     language="en",
    ... )
    >>> print(f"Job {job.id} created with status {job.status}")
    """

    def __init__(
        self,
        repository: Optional[JobRepository] = None,
        queue_client: Optional[QueueClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repository = repository or JobRepository(settings=self._settings)
        self._queue_client = queue_client or QueueClient(settings=self._settings)

        logger.info("ServerJobService initialized")

    @property
    def repository(self) -> JobRepository:
        """Get the job repository."""
        return self._repository

    @property
    def queue_client(self) -> QueueClient:
        """Get the queue client."""
        return self._queue_client

    # =========================================================================
    # Job Creation
    # =========================================================================

    def create_job(
        self,
        *,
        job_type: JobType | str,
        source_path: str | Path,
        language: str = "en",
        target_language: Optional[str] = None,
        subtitle_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        model_name: Optional[str] = None,
        backend: str = "faster-whisper",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        priority: str = "default",
        options: Optional[dict[str, Any]] = None,
        enqueue: bool = True,
    ) -> Job:
        """
        Create a new job and optionally enqueue it for processing.

        Parameters
        ----------
        job_type : JobType or str
            Type of job (transcribe, translate, sync_subtitle, compare).
        source_path : str or Path
            Path to source video file.
        language : str
            Source audio language code (e.g., 'en', 'es').
        target_language : str, optional
            Translation target language.
        subtitle_path : str or Path, optional
            Path to existing subtitle (for sync/compare).
        output_path : str or Path, optional
            Path for output SRT file.
        model_name : str, optional
            Whisper model name. Uses default from settings if not provided.
        backend : str
            Processing backend.
        device : str, optional
            Device to use. Uses default from settings if not provided.
        compute_type : str, optional
            Compute precision. Uses default from settings if not provided.
        priority : str
            Queue priority ('high', 'default', 'gpu').
        options : dict, optional
            Additional job options.
        enqueue : bool
            Whether to immediately enqueue the job.

        Returns
        -------
        Job
            The created job entity.

        Raises
        ------
        FileNotFoundError
            If source file doesn't exist and validation is enabled.
        ValueError
            If job_type is invalid.
        """
        # Convert string to enum
        if isinstance(job_type, str):
            job_type = JobType(job_type)

        # Use defaults from settings
        model_name = model_name or self._settings.default_model
        device = device or self._settings.default_device
        compute_type = compute_type or self._settings.default_compute_type

        # Convert paths to strings
        source_path = str(source_path)
        if subtitle_path:
            subtitle_path = str(subtitle_path)
        if output_path:
            output_path = str(output_path)

        # Map priority string to integer
        priority_map = {"high": 10, "default": 0, "low": -10, "gpu": 5}
        priority_int = priority_map.get(priority, 0)

        # Create job in database
        job = self._repository.create_job(
            job_type=job_type,
            source_path=source_path,
            language=language,
            target_language=target_language,
            subtitle_path=subtitle_path,
            output_path=output_path,
            model_name=model_name,
            backend=backend,
            device=device,
            compute_type=compute_type,
            priority=priority_int,
            options=options,
        )

        logger.info(f"Created job {job.id}: {job_type.value} for {source_path}")

        # Enqueue for processing
        if enqueue:
            self.enqueue_job(job.id, priority=priority)

        return job

    def enqueue_job(
        self,
        job_id: uuid.UUID | str,
        priority: str = "default",
        timeout: int = 3600,
    ) -> str:
        """
        Enqueue a job for processing.

        Parameters
        ----------
        job_id : UUID or str
            The job ID to enqueue.
        priority : str
            Queue priority.
        timeout : int
            Job timeout in seconds.

        Returns
        -------
        str
            The RQ job ID.
        """
        rq_job_id = self._queue_client.enqueue(
            job_id,
            priority=priority,
            timeout=timeout,
        )
        return rq_job_id

    # =========================================================================
    # Job Queries
    # =========================================================================

    def get_job(self, job_id: uuid.UUID | str) -> Optional[Job]:
        """
        Get a job by ID.

        Parameters
        ----------
        job_id : UUID or str
            The job ID.

        Returns
        -------
        Job or None
            The job if found.
        """
        return self._repository.get_job(job_id)

    def list_jobs(
        self,
        *,
        status: Optional[JobStatus | str] = None,
        job_type: Optional[JobType | str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Job]:
        """
        List jobs with optional filtering.

        Parameters
        ----------
        status : JobStatus or str, optional
            Filter by status.
        job_type : JobType or str, optional
            Filter by type.
        limit : int
            Maximum number of jobs.
        offset : int
            Pagination offset.

        Returns
        -------
        Sequence[Job]
            List of matching jobs.
        """
        return self._repository.list_jobs(
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )

    def get_job_details(self, job_id: uuid.UUID | str) -> Optional[dict[str, Any]]:
        """
        Get detailed job information including events and metrics.

        Parameters
        ----------
        job_id : UUID or str
            The job ID.

        Returns
        -------
        dict or None
            Job details with events and metrics.
        """
        job = self._repository.get_job(job_id)
        if job is None:
            return None

        events = self._repository.get_job_events(job_id)
        metrics = self._repository.get_job_metrics(job_id)

        return {
            "job": job.to_dict(),
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "old_status": e.old_status,
                    "new_status": e.new_status,
                    "old_stage": e.old_stage,
                    "new_stage": e.new_stage,
                    "message": e.message,
                }
                for e in events
            ],
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_mb": m.memory_used_mb,
                    "gpu_utilization": m.gpu_utilization,
                    "gpu_memory_used_mb": m.gpu_memory_used_mb,
                }
                for m in metrics
            ],
        }

    # =========================================================================
    # Job Control
    # =========================================================================

    def cancel_job(self, job_id: uuid.UUID | str) -> bool:
        """
        Cancel a job.

        For queued jobs, removes from queue.
        For running jobs, sets cancel flag (worker must check).

        Parameters
        ----------
        job_id : UUID or str
            The job ID to cancel.

        Returns
        -------
        bool
            True if job was canceled or cancel was requested.
        """
        if isinstance(job_id, str):
            job_id_str = job_id
        else:
            job_id_str = str(job_id)

        job = self._repository.get_job(job_id)
        if job is None:
            return False

        # Can only cancel pending or running jobs
        if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            logger.warning(f"Cannot cancel job {job_id} with status {job.status}")
            return False

        # Try to cancel from queue if still pending
        if job.status == JobStatus.PENDING:
            rq_job_id = f"subsvc:{job_id_str}"
            self._queue_client.cancel(rq_job_id)

        # Update status in database
        self._repository.update_status(job_id, JobStatus.CANCELED)
        logger.info(f"Canceled job {job_id}")

        return True

    def retry_job(
        self, job_id: uuid.UUID | str, priority: str = "default"
    ) -> Optional[Job]:
        """
        Retry a failed job.

        Creates a new job with the same parameters.

        Parameters
        ----------
        job_id : UUID or str
            The job ID to retry.
        priority : str
            Queue priority for the retry.

        Returns
        -------
        Job or None
            The new job, or None if original not found.
        """
        original = self._repository.get_job(job_id)
        if original is None:
            return None

        # Can only retry failed jobs
        if original.status != JobStatus.FAILED:
            logger.warning(f"Cannot retry job {job_id} with status {original.status}")
            return None

        # Create new job with same parameters
        new_job = self.create_job(
            job_type=original.type,
            source_path=original.source_path,
            language=original.language,
            target_language=original.target_language,
            subtitle_path=original.subtitle_path,
            output_path=original.output_path,
            model_name=original.model_name,
            backend=original.backend,
            device=original.device,
            compute_type=original.compute_type,
            priority=priority,
            options={
                **(original.options or {}),
                "retry_of": str(original.id),
            },
            enqueue=True,
        )

        logger.info(f"Created retry job {new_job.id} for failed job {job_id}")
        return new_job

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get overall service statistics.

        Returns
        -------
        dict
            Statistics from database and queues.
        """
        db_stats = self._repository.get_statistics()
        queue_stats = self._queue_client.get_queue_stats()

        return {
            "database": db_stats,
            "queues": queue_stats,
        }

    def health_check(self) -> dict[str, Any]:
        """
        Check health of all service components.

        Returns
        -------
        dict
            Health status of database and Redis.
        """
        redis_ok = self._queue_client.ping()

        # Simple database check
        try:
            self._repository.get_statistics()
            db_ok = True
        except Exception:
            db_ok = False

        return {
            "healthy": redis_ok and db_ok,
            "redis": "ok" if redis_ok else "error",
            "database": "ok" if db_ok else "error",
        }
