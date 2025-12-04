# src/homelab_subs/server/queue.py

"""
Redis Queue (RQ) client for job queue management.

Provides a clean abstraction over RQ for enqueueing jobs,
checking status, and managing queues.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

try:
    from redis import Redis
    from rq import Queue, Retry
    from rq.job import Job as RQJob

    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    Redis = None  # type: ignore[misc, assignment]
    Queue = None  # type: ignore[misc, assignment]
    RQJob = None  # type: ignore[misc, assignment]

from .settings import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class QueueClient:
    """
    Client for managing job queues via Redis and RQ.

    This class provides:
    - Job enqueueing with priority-based queue selection
    - Job status checking
    - Job cancellation
    - Queue statistics

    Parameters
    ----------
    redis_url : str, optional
        Redis connection URL. If not provided, uses settings.
    settings : Settings, optional
        Application settings. If not provided, uses get_settings().

    Examples
    --------
    >>> client = QueueClient()
    >>> rq_job_id = client.enqueue("job-uuid-here", priority="high")
    >>> status = client.get_status(rq_job_id)
    >>> print(status)  # 'queued', 'started', 'finished', 'failed'
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not RQ_AVAILABLE:
            raise RuntimeError(
                "RQ and Redis are required for QueueClient. "
                "Install with: pip install homelab-subtitle-service[server]"
            )

        self._settings = settings or get_settings()
        self._redis_url = redis_url or self._settings.redis_url

        # Initialize Redis connection
        self._redis = Redis.from_url(self._redis_url)

        # Initialize queues
        self._queues: dict[str, Queue] = {
            "default": Queue(self._settings.queue_default, connection=self._redis),
            "high": Queue(self._settings.queue_high, connection=self._redis),
            "gpu": Queue(self._settings.queue_gpu, connection=self._redis),
        }

        logger.info(
            f"QueueClient initialized with Redis: {self._redis_url.split('@')[-1]}"
        )

    @property
    def redis(self) -> "Redis":
        """Get the Redis connection."""
        return self._redis

    def get_queue(self, name: str = "default") -> "Queue":
        """
        Get a queue by name.

        Parameters
        ----------
        name : str
            Queue name ('default', 'high', 'gpu').

        Returns
        -------
        Queue
            The RQ queue instance.
        """
        if name not in self._queues:
            # Create queue on demand
            self._queues[name] = Queue(name, connection=self._redis)
        return self._queues[name]

    def enqueue(
        self,
        job_id: uuid.UUID | str,
        *,
        func: Optional[Callable[..., Any]] = None,
        priority: str = "default",
        timeout: int = 3600,  # 1 hour default
        retry_count: int = 0,
        retry_interval: int = 60,
        job_ttl: Optional[int] = None,
        result_ttl: int = 86400,  # 24 hours
        **kwargs: Any,
    ) -> str:
        """
        Enqueue a job for processing.

        Parameters
        ----------
        job_id : UUID or str
            The database job ID to process.
        func : Callable, optional
            The function to execute. If not provided, uses default worker task.
        priority : str
            Queue priority ('high', 'default', 'gpu').
        timeout : int
            Job timeout in seconds.
        retry_count : int
            Number of times to retry on failure.
        retry_interval : int
            Seconds between retries.
        job_ttl : int, optional
            Time-to-live for queued job (None = forever).
        result_ttl : int
            Time-to-live for job result.
        **kwargs
            Additional arguments to pass to the worker function.

        Returns
        -------
        str
            The RQ job ID.
        """
        if isinstance(job_id, uuid.UUID):
            job_id = str(job_id)

        queue = self.get_queue(priority)

        # Import worker function if not provided
        if func is None:
            from .worker import process_job

            func = process_job

        # Build retry policy
        retry = None
        if retry_count > 0:
            retry = Retry(max=retry_count, interval=retry_interval)

        # Enqueue the job
        rq_job = queue.enqueue(
            func,
            job_id=f"subsvc:{job_id}",  # Use predictable RQ job ID
            **kwargs,
            job_timeout=timeout,
            result_ttl=result_ttl,
            ttl=job_ttl,
            retry=retry,
        )

        logger.info(
            f"Enqueued job {job_id} to queue '{priority}' as RQ job {rq_job.id}"
        )
        return rq_job.id

    def get_rq_job(self, rq_job_id: str) -> Optional["RQJob"]:
        """
        Get an RQ job by ID.

        Parameters
        ----------
        rq_job_id : str
            The RQ job ID.

        Returns
        -------
        RQJob or None
            The RQ job if found.
        """
        try:
            return RQJob.fetch(rq_job_id, connection=self._redis)
        except Exception:
            return None

    def get_status(self, rq_job_id: str) -> Optional[str]:
        """
        Get the status of an RQ job.

        Parameters
        ----------
        rq_job_id : str
            The RQ job ID.

        Returns
        -------
        str or None
            Status string ('queued', 'started', 'finished', 'failed', etc.)
            or None if job not found.
        """
        job = self.get_rq_job(rq_job_id)
        if job is None:
            return None
        return job.get_status()

    def cancel(self, rq_job_id: str) -> bool:
        """
        Cancel a queued job.

        Note: This only works for jobs that haven't started yet.
        For running jobs, use the cancel flag in the database.

        Parameters
        ----------
        rq_job_id : str
            The RQ job ID to cancel.

        Returns
        -------
        bool
            True if job was canceled, False otherwise.
        """
        job = self.get_rq_job(rq_job_id)
        if job is None:
            return False

        try:
            job.cancel()
            logger.info(f"Canceled RQ job {rq_job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel RQ job {rq_job_id}: {e}")
            return False

    def get_result(self, rq_job_id: str) -> Any:
        """
        Get the result of a completed job.

        Parameters
        ----------
        rq_job_id : str
            The RQ job ID.

        Returns
        -------
        Any
            The job result, or None if not available.
        """
        job = self.get_rq_job(rq_job_id)
        if job is None:
            return None
        return job.result

    def get_exception(self, rq_job_id: str) -> Optional[str]:
        """
        Get the exception info from a failed job.

        Parameters
        ----------
        rq_job_id : str
            The RQ job ID.

        Returns
        -------
        str or None
            The exception traceback, or None if not failed.
        """
        job = self.get_rq_job(rq_job_id)
        if job is None:
            return None
        return job.exc_info

    # =========================================================================
    # Queue Statistics
    # =========================================================================

    def get_queue_stats(self) -> dict[str, dict[str, int]]:
        """
        Get statistics for all queues.

        Returns
        -------
        dict
            Dictionary mapping queue names to their stats.
        """
        stats = {}
        for name, queue in self._queues.items():
            stats[name] = {
                "queued": len(queue),
                "started": queue.started_job_registry.count,
                "finished": queue.finished_job_registry.count,
                "failed": queue.failed_job_registry.count,
                "deferred": queue.deferred_job_registry.count,
                "scheduled": queue.scheduled_job_registry.count,
            }
        return stats

    def get_failed_jobs(
        self, queue_name: str = "default", limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get list of failed jobs from a queue.

        Parameters
        ----------
        queue_name : str
            Queue name to check.
        limit : int
            Maximum number of failed jobs to return.

        Returns
        -------
        list[dict]
            List of failed job info.
        """
        queue = self.get_queue(queue_name)
        failed_registry = queue.failed_job_registry

        failed_jobs = []
        for job_id in failed_registry.get_job_ids()[:limit]:
            job = self.get_rq_job(job_id)
            if job:
                failed_jobs.append(
                    {
                        "id": job.id,
                        "func_name": job.func_name,
                        "args": job.args,
                        "kwargs": job.kwargs,
                        "exc_info": job.exc_info,
                        "enqueued_at": job.enqueued_at.isoformat()
                        if job.enqueued_at
                        else None,
                        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                    }
                )

        return failed_jobs

    def retry_failed(self, rq_job_id: str, queue_name: str = "default") -> bool:
        """
        Retry a failed job.

        Parameters
        ----------
        rq_job_id : str
            The RQ job ID to retry.
        queue_name : str
            Queue to requeue to.

        Returns
        -------
        bool
            True if job was requeued, False otherwise.
        """
        job = self.get_rq_job(rq_job_id)
        if job is None:
            return False

        try:
            job.requeue()
            logger.info(f"Requeued failed job {rq_job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to requeue job {rq_job_id}: {e}")
            return False

    def clear_queue(self, queue_name: str = "default") -> int:
        """
        Clear all jobs from a queue.

        Parameters
        ----------
        queue_name : str
            Queue to clear.

        Returns
        -------
        int
            Number of jobs cleared.
        """
        queue = self.get_queue(queue_name)
        count = len(queue)
        queue.empty()
        logger.warning(f"Cleared {count} jobs from queue '{queue_name}'")
        return count

    def ping(self) -> bool:
        """
        Check if Redis is reachable.

        Returns
        -------
        bool
            True if Redis responded to ping.
        """
        try:
            return self._redis.ping()
        except Exception:
            return False
