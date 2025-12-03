"""
Database logging system for persistent storage of job logs and metrics.

This module provides:
- SQLite database for storing job execution logs
- Performance metrics storage
- Query interface for Web UI integration
- Automatic database schema management
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class JobLog:
    """
    Represents a single job execution log entry.
    """

    job_id: str
    video_path: str
    output_path: Optional[str]
    status: str  # pending, running, completed, failed
    language: Optional[str]
    model: str
    task: str  # transcribe or translate
    started_at: datetime
    device: Optional[str] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Performance metrics summary
    cpu_avg: Optional[float] = None
    cpu_max: Optional[float] = None
    memory_avg_mb: Optional[float] = None
    memory_max_mb: Optional[float] = None
    gpu_avg: Optional[float] = None
    gpu_max: Optional[float] = None

    # Row ID (set by database)
    id: Optional[int] = None


@dataclass
class MetricLog:
    """
    Represents a single performance metrics snapshot.
    """
    job_id: str
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_read_mb: float
    disk_write_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None
    
    # Row ID (set by database)
    id: Optional[int] = None


class DatabaseLogger:
    """
    Manages persistent storage of job logs and performance metrics.
    """
    
    def __init__(self, db_path: Path | str = None):
        """
        Initialize database logger.
        
        Parameters
        ----------
        db_path : Path | str, optional
            Path to SQLite database file.
            If None, uses ~/.homelab-subs/logs.db
        """
        if db_path is None:
            db_dir = Path.home() / ".homelab-subs"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "logs.db"
        
        self.db_path = Path(db_path)
        self._ensure_schema()
        logger.info(f"Database logger initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    video_path TEXT NOT NULL,
                    output_path TEXT,
                    status TEXT NOT NULL,
                    language TEXT,
                    model TEXT NOT NULL,
                    task TEXT NOT NULL,
                    device TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    duration_seconds REAL,
                    cpu_avg REAL,
                    cpu_max REAL,
                    memory_avg_mb REAL,
                    memory_max_mb REAL,
                    gpu_avg REAL,
                    gpu_max REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_mb REAL NOT NULL,
                    disk_read_mb REAL NOT NULL,
                    disk_write_mb REAL NOT NULL,
                    gpu_utilization REAL,
                    gpu_memory_used_mb REAL,
                    gpu_memory_percent REAL,
                    gpu_temperature REAL,
                    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs (job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_started_at ON jobs (started_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_job_id ON metrics (job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)")
            
            # Ensure newer columns exist for legacy databases
            cursor.execute("PRAGMA table_info(jobs)")
            columns = {row[1] for row in cursor.fetchall()}
            if "device" not in columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN device TEXT")

            conn.commit()

    def create_job(self, job: JobLog) -> int:
        """
        Create a new job log entry.
        
        Parameters
        ----------
        job : JobLog
            Job information
            
        Returns
        -------
        int
            Database row ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO jobs (
                    job_id, video_path, output_path, status, language,
                    model, task, device, started_at, completed_at,
                    error_message, duration_seconds, cpu_avg, cpu_max,
                    memory_avg_mb, memory_max_mb, gpu_avg, gpu_max
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    str(job.video_path),
                    str(job.output_path) if job.output_path else None,
                    job.status,
                    job.language,
                    job.model,
                    job.task,
                    job.device,
                    job.started_at.isoformat(),
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.error_message,
                    job.duration_seconds,
                    job.cpu_avg,
                    job.cpu_max,
                    job.memory_avg_mb,
                    job.memory_max_mb,
                    job.gpu_avg,
                    job.gpu_max,
                ),
            )
            return cursor.lastrowid

    def update_job(
        self,
        job_id: JobLog | str,
        *,
        status: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        output_path: Optional[str | Path] = None,
        language: Optional[str] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        device: Optional[str] = None,
        performance_summary: Optional[dict] = None,
    ):
        """
        Update an existing job log entry.
        
        Parameters
        ----------
        job_id : str
            Job identifier
        status : str, optional
            New status
        completed_at : datetime, optional
            Completion timestamp
        error_message : str, optional
            Error message if failed
        duration_seconds : float, optional
            Job duration
        performance_summary : dict, optional
            Performance metrics summary
        """
        job_obj: Optional[JobLog] = None
        if isinstance(job_id, JobLog):
            job_obj = job_id
            job_id = job_obj.job_id

            status = status if status is not None else job_obj.status
            completed_at = completed_at if completed_at is not None else job_obj.completed_at
            error_message = error_message if error_message is not None else job_obj.error_message
            duration_seconds = (
                duration_seconds if duration_seconds is not None else job_obj.duration_seconds
            )
            output_path = output_path if output_path is not None else job_obj.output_path
            language = language if language is not None else job_obj.language
            model = model if model is not None else job_obj.model
            task = task if task is not None else job_obj.task
            device = device if device is not None else job_obj.device

            if performance_summary is None:
                performance_summary = {}
                if job_obj.cpu_avg is not None:
                    performance_summary["cpu_avg"] = job_obj.cpu_avg
                if job_obj.cpu_max is not None:
                    performance_summary["cpu_max"] = job_obj.cpu_max
                if job_obj.memory_avg_mb is not None:
                    performance_summary["memory_avg_mb"] = job_obj.memory_avg_mb
                if job_obj.memory_max_mb is not None:
                    performance_summary["memory_max_mb"] = job_obj.memory_max_mb
                if job_obj.gpu_avg is not None:
                    performance_summary["gpu_avg"] = job_obj.gpu_avg
                if job_obj.gpu_max is not None:
                    performance_summary["gpu_max"] = job_obj.gpu_max

        updates = []
        values = []

        if status is not None:
            updates.append("status = ?")
            values.append(status)

        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at.isoformat())

        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)

        if duration_seconds is not None:
            updates.append("duration_seconds = ?")
            values.append(duration_seconds)

        if output_path is not None:
            updates.append("output_path = ?")
            values.append(str(output_path))

        if language is not None:
            updates.append("language = ?")
            values.append(language)

        if model is not None:
            updates.append("model = ?")
            values.append(model)

        if task is not None:
            updates.append("task = ?")
            values.append(task)

        if device is not None:
            updates.append("device = ?")
            values.append(device)

        if performance_summary:
            if "cpu_avg" in performance_summary:
                updates.append("cpu_avg = ?")
                values.append(performance_summary["cpu_avg"])
            if "cpu_max" in performance_summary:
                updates.append("cpu_max = ?")
                values.append(performance_summary["cpu_max"])
            memory_avg_value = performance_summary.get("memory_avg_mb")
            if memory_avg_value is None:
                memory_avg_value = performance_summary.get("memory_avg")
            if memory_avg_value is not None:
                updates.append("memory_avg_mb = ?")
                values.append(memory_avg_value)

            memory_max_value = performance_summary.get("memory_max_mb")
            if memory_max_value is None:
                memory_max_value = performance_summary.get("memory_max")
            if memory_max_value is not None:
                updates.append("memory_max_mb = ?")
                values.append(memory_max_value)
            if "gpu_avg" in performance_summary:
                updates.append("gpu_avg = ?")
                values.append(performance_summary["gpu_avg"])
            if "gpu_max" in performance_summary:
                updates.append("gpu_max = ?")
                values.append(performance_summary["gpu_max"])

        if not updates:
            return

        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)

    def add_metric(self, metric: MetricLog):
        """
        Add a performance metrics snapshot.
        
        Parameters
        ----------
        metric : MetricLog
            Metrics snapshot
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (
                    job_id, timestamp, cpu_percent, memory_percent,
                    memory_used_mb, disk_read_mb, disk_write_mb,
                    gpu_utilization, gpu_memory_used_mb, gpu_memory_percent,
                    gpu_temperature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.job_id,
                metric.timestamp.isoformat(),
                metric.cpu_percent,
                metric.memory_percent,
                metric.memory_used_mb,
                metric.disk_read_mb,
                metric.disk_write_mb,
                metric.gpu_utilization,
                metric.gpu_memory_used_mb,
                metric.gpu_memory_percent,
                metric.gpu_temperature,
            ))

    def get_job(self, job_id: str) -> Optional[JobLog]:
        """
        Retrieve a job by ID.
        
        Parameters
        ----------
        job_id : str
            Job identifier
            
        Returns
        -------
        JobLog or None
            Job information if found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_job(row)

    def get_recent_jobs(self, limit: int = 50, status: Optional[str] = None) -> list[JobLog]:
        """
        Get recent jobs, optionally filtered by status.
        
        Parameters
        ----------
        limit : int
            Maximum number of jobs to return
        status : str, optional
            Filter by status (pending, running, completed, failed)
            
        Returns
        -------
        list[JobLog]
            Recent jobs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?",
                    (limit,)
                )
            
            return [self._row_to_job(row) for row in cursor.fetchall()]

    def get_job_metrics(self, job_id: str) -> list[MetricLog]:
        """
        Get all performance metrics for a job.
        
        Parameters
        ----------
        job_id : str
            Job identifier
            
        Returns
        -------
        list[MetricLog]
            Performance metrics snapshots
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM metrics WHERE job_id = ? ORDER BY timestamp",
                (job_id,)
            )
            return [self._row_to_metric(row) for row in cursor.fetchall()]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get overall statistics about jobs.
        
        Returns
        -------
        dict
            Statistics summary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total jobs by status
            cursor.execute("SELECT status, COUNT(*) as count FROM jobs GROUP BY status")
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Average durations
            cursor.execute("""
                SELECT 
                    AVG(duration_seconds) as avg_duration,
                    MIN(duration_seconds) as min_duration,
                    MAX(duration_seconds) as max_duration
                FROM jobs
                WHERE duration_seconds IS NOT NULL
            """)
            duration_stats = cursor.fetchone()
            
            # Average performance
            cursor.execute("""
                SELECT 
                    AVG(cpu_avg) as avg_cpu,
                    AVG(memory_avg_mb) as avg_memory,
                    AVG(gpu_avg) as avg_gpu
                FROM jobs
                WHERE cpu_avg IS NOT NULL
            """)
            perf_stats = cursor.fetchone()
            
            return {
                "total_jobs": sum(status_counts.values()),
                "status_counts": status_counts,
                "avg_duration_seconds": duration_stats["avg_duration"],
                "min_duration_seconds": duration_stats["min_duration"],
                "max_duration_seconds": duration_stats["max_duration"],
                "avg_cpu_percent": perf_stats["avg_cpu"],
                "avg_memory_mb": perf_stats["avg_memory"],
                "avg_gpu_percent": perf_stats["avg_gpu"],
            }

    def _row_to_job(self, row: sqlite3.Row) -> JobLog:
        """Convert database row to JobLog object."""
        keys = set(row.keys())
        return JobLog(
            id=row["id"],
            job_id=row["job_id"],
            video_path=row["video_path"],
            output_path=row["output_path"],
            status=row["status"],
            language=row["language"],
            model=row["model"],
            task=row["task"],
            device=row["device"] if "device" in keys else None,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            error_message=row["error_message"],
            duration_seconds=row["duration_seconds"],
            cpu_avg=row["cpu_avg"],
            cpu_max=row["cpu_max"],
            memory_avg_mb=row["memory_avg_mb"],
            memory_max_mb=row["memory_max_mb"],
            gpu_avg=row["gpu_avg"],
            gpu_max=row["gpu_max"],
        )

    def _row_to_metric(self, row: sqlite3.Row) -> MetricLog:
        """Convert database row to MetricLog object."""
        return MetricLog(
            id=row["id"],
            job_id=row["job_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            cpu_percent=row["cpu_percent"],
            memory_percent=row["memory_percent"],
            memory_used_mb=row["memory_used_mb"],
            disk_read_mb=row["disk_read_mb"],
            disk_write_mb=row["disk_write_mb"],
            gpu_utilization=row["gpu_utilization"],
            gpu_memory_used_mb=row["gpu_memory_used_mb"],
            gpu_memory_percent=row["gpu_memory_percent"],
            gpu_temperature=row["gpu_temperature"],
        )
