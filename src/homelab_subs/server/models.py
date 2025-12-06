# src/homelab_subs/server/models.py

"""
Database models for job management.

Defines the Job model and related enums using SQLAlchemy 2.0 style
with dataclass-like syntax for clean type hints.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

try:
    from sqlalchemy import (
        DateTime,
        Enum,
        Float,
        Integer,
        String,
        Text,
        func,
    )
    from sqlalchemy.dialects.postgresql import JSONB, UUID
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    DeclarativeBase = object  # type: ignore[misc, assignment]
    Mapped = Any  # type: ignore[misc, assignment]

    def _noop_mapped_column(*args, **kwargs):  # noqa: ARG001
        """Fallback when SQLAlchemy is not available."""
        return None

    mapped_column = _noop_mapped_column  # type: ignore[misc, assignment]


class JobStatus(str, enum.Enum):
    """
    Status of a job in the queue system.

    Lifecycle: pending -> running -> done/failed/canceled
    """

    PENDING = "pending"  # Job created, waiting in queue
    RUNNING = "running"  # Worker picked up the job
    DONE = "done"  # Job completed successfully
    FAILED = "failed"  # Job failed with an error
    CANCELED = "canceled"  # Job was canceled by user


class JobType(str, enum.Enum):
    """
    Type of subtitle job to perform.
    """

    TRANSCRIBE = "transcribe"  # Generate subtitles from audio
    TRANSLATE = "translate"  # Transcribe + translate
    SYNC_SUBTITLE = "sync_subtitle"  # Sync existing SRT with audio
    COMPARE = "compare"  # Compare human vs machine SRT


class JobStage(str, enum.Enum):
    """
    Current processing stage within a running job.
    """

    QUEUED = "queued"  # Waiting in queue
    INITIALIZING = "initializing"  # Setting up job
    EXTRACTING_AUDIO = "extracting_audio"  # ffmpeg audio extraction
    TRANSCRIBING = "transcribing"  # Whisper transcription
    TRANSLATING = "translating"  # Text translation
    GENERATING_SRT = "generating_srt"  # SRT file generation
    SYNCING = "syncing"  # Subtitle synchronization
    COMPARING = "comparing"  # Subtitle comparison
    FINALIZING = "finalizing"  # Cleanup and final steps
    COMPLETED = "completed"  # Done


if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        """Base class for all database models."""

        pass

    class User(Base):
        """
        User model for authentication.

        Stores user credentials and metadata. Password is hashed
        using Argon2 via passlib.

        Attributes
        ----------
        id : UUID
            Unique user identifier.
        username : str
            Unique username for login.
        password_hash : str
            Argon2 hashed password.
        is_active : bool
            Whether user can log in.
        is_admin : bool
            Whether user has admin privileges.
        created_at : datetime
            When the user was created.
        last_login : datetime, optional
            Last successful login time.
        """

        __tablename__ = "users"

        id: Mapped[uuid.UUID] = mapped_column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
        )
        username: Mapped[str] = mapped_column(
            String(100),
            unique=True,
            nullable=False,
            index=True,
        )
        password_hash: Mapped[str] = mapped_column(
            String(255),
            nullable=False,
        )
        is_active: Mapped[bool] = mapped_column(
            default=True,
            nullable=False,
        )
        is_admin: Mapped[bool] = mapped_column(
            default=False,
            nullable=False,
        )
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        )
        last_login: Mapped[Optional[datetime]] = mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )

        def __repr__(self) -> str:
            return f"User(id={self.id!r}, username={self.username!r})"

        def to_dict(self) -> dict[str, Any]:
            """Convert user to dictionary (excluding password)."""
            return {
                "id": str(self.id),
                "username": self.username,
                "is_active": self.is_active,
                "is_admin": self.is_admin,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_login": self.last_login.isoformat() if self.last_login else None,
            }

    class GlobalSettings(Base):
        """
        Global application settings.

        Singleton table (always id=1) storing application-wide configuration
        that can be modified through the UI.

        Attributes
        ----------
        id : int
            Always 1 (singleton pattern).
        media_folders : list[str]
            Allowed directories for file browsing.
        default_model : str
            Default Whisper model for new jobs.
        default_device : str
            Default device (cpu/cuda) for processing.
        default_compute_type : str
            Default compute precision.
        default_language : str
            Default source language.
        default_translation_backend : str
            Default translation backend (nllb/helsinki).
        worker_count : int
            Number of workers to spawn.
        log_retention_days : int
            Days to keep logs before cleanup.
        job_retention_days : int
            Days to keep completed jobs before cleanup.
        prefer_gpu : bool
            Default preference for GPU processing.
        updated_at : datetime
            Last settings update time.
        """

        __tablename__ = "global_settings"

        id: Mapped[int] = mapped_column(
            Integer,
            primary_key=True,
            default=1,
        )
        media_folders: Mapped[Optional[list]] = mapped_column(
            JSONB,
            nullable=True,
            default=list,
        )
        default_model: Mapped[str] = mapped_column(
            String(50),
            default="small",
            nullable=False,
        )
        default_device: Mapped[str] = mapped_column(
            String(20),
            default="cpu",
            nullable=False,
        )
        default_compute_type: Mapped[str] = mapped_column(
            String(20),
            default="int8",
            nullable=False,
        )
        default_language: Mapped[str] = mapped_column(
            String(10),
            default="en",
            nullable=False,
        )
        default_translation_backend: Mapped[str] = mapped_column(
            String(50),
            default="nllb",
            nullable=False,
        )
        worker_count: Mapped[int] = mapped_column(
            Integer,
            default=1,
            nullable=False,
        )
        log_retention_days: Mapped[int] = mapped_column(
            Integer,
            default=30,
            nullable=False,
        )
        job_retention_days: Mapped[int] = mapped_column(
            Integer,
            default=90,
            nullable=False,
        )
        prefer_gpu: Mapped[bool] = mapped_column(
            default=False,
            nullable=False,
        )
        updated_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        )

        def __repr__(self) -> str:
            return f"GlobalSettings(id={self.id}, updated_at={self.updated_at})"

        def to_dict(self) -> dict[str, Any]:
            """Convert settings to dictionary."""
            return {
                "media_folders": self.media_folders or [],
                "default_model": self.default_model,
                "default_device": self.default_device,
                "default_compute_type": self.default_compute_type,
                "default_language": self.default_language,
                "default_translation_backend": self.default_translation_backend,
                "worker_count": self.worker_count,
                "log_retention_days": self.log_retention_days,
                "job_retention_days": self.job_retention_days,
                "prefer_gpu": self.prefer_gpu,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            }

    class Job(Base):
        """
        Represents a subtitle processing job.

        This is the central model for tracking all job metadata,
        status, progress, and results.

        Attributes
        ----------
        id : UUID
            Unique job identifier.
        created_at : datetime
            When the job was created.
        started_at : datetime, optional
            When the worker started processing.
        finished_at : datetime, optional
            When the job completed (success or failure).
        status : JobStatus
            Current job status.
        type : JobType
            Type of job to perform.
        source_path : str
            Path to the source video file.
        subtitle_path : str, optional
            Path to existing subtitle (for sync/compare).
        output_path : str, optional
            Path where output SRT will be written.
        language : str
            Source audio language code (e.g., 'en', 'es').
        target_language : str, optional
            Translation target language code.
        model_name : str
            Whisper model name (tiny, small, medium, large-v2).
        backend : str
            Processing backend (faster-whisper, whisperx, etc.).
        device : str
            Device to use (cpu, cuda).
        compute_type : str
            Compute precision (int8, float16, etc.).
        progress : float
            Job progress percentage (0-100).
        current_stage : JobStage
            Current processing stage.
        error_message : str, optional
            Error message if job failed.
        options : dict
            Additional job options (JSON).
        priority : int
            Job priority (higher = more urgent).

        Performance Metrics (populated after completion):
        cpu_avg, cpu_max : float
            CPU usage statistics.
        memory_avg_mb, memory_max_mb : float
            Memory usage statistics.
        gpu_avg, gpu_max : float
            GPU usage statistics (if applicable).
        """

        __tablename__ = "jobs"

        # Primary key
        id: Mapped[uuid.UUID] = mapped_column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
        )

        # Timestamps
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        )
        started_at: Mapped[Optional[datetime]] = mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
        finished_at: Mapped[Optional[datetime]] = mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )

        # Status
        status: Mapped[JobStatus] = mapped_column(
            Enum(JobStatus, native_enum=False),
            default=JobStatus.PENDING,
            nullable=False,
            index=True,
        )

        # Job type
        type: Mapped[JobType] = mapped_column(
            Enum(JobType, native_enum=False),
            nullable=False,
            index=True,
        )

        # File paths
        source_path: Mapped[str] = mapped_column(Text, nullable=False)
        subtitle_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

        # Language settings
        language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
        target_language: Mapped[Optional[str]] = mapped_column(
            String(10), nullable=True
        )

        # Model/backend settings
        model_name: Mapped[str] = mapped_column(
            String(50), default="small", nullable=False
        )
        backend: Mapped[str] = mapped_column(
            String(50), default="faster-whisper", nullable=False
        )
        device: Mapped[str] = mapped_column(String(20), default="cpu", nullable=False)
        compute_type: Mapped[str] = mapped_column(
            String(20), default="int8", nullable=False
        )

        # Progress tracking
        progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
        current_stage: Mapped[JobStage] = mapped_column(
            Enum(JobStage, native_enum=False),
            default=JobStage.QUEUED,
            nullable=False,
        )

        # Error handling
        error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

        # Additional options (JSON)
        options: Mapped[Optional[dict]] = mapped_column(
            JSONB, nullable=True, default=dict
        )

        # Priority (for queue ordering)
        priority: Mapped[int] = mapped_column(
            Integer, default=0, nullable=False, index=True
        )

        # Performance metrics (populated after completion)
        cpu_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        cpu_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        memory_avg_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        memory_max_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        gpu_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        gpu_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

        # Duration (computed on completion)
        duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

        # Output statistics
        segment_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        output_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

        def __repr__(self) -> str:
            return (
                f"Job(id={self.id!r}, type={self.type.value!r}, "
                f"status={self.status.value!r}, progress={self.progress:.1f}%)"
            )

        def to_dict(self) -> dict[str, Any]:
            """Convert job to dictionary for API responses."""
            return {
                "id": str(self.id),
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat()
                if self.finished_at
                else None,
                "status": self.status.value,
                "type": self.type.value,
                "source_path": self.source_path,
                "subtitle_path": self.subtitle_path,
                "output_path": self.output_path,
                "language": self.language,
                "target_language": self.target_language,
                "model_name": self.model_name,
                "backend": self.backend,
                "device": self.device,
                "compute_type": self.compute_type,
                "progress": self.progress,
                "current_stage": self.current_stage.value,
                "error_message": self.error_message,
                "options": self.options,
                "priority": self.priority,
                "cpu_avg": self.cpu_avg,
                "cpu_max": self.cpu_max,
                "memory_avg_mb": self.memory_avg_mb,
                "memory_max_mb": self.memory_max_mb,
                "gpu_avg": self.gpu_avg,
                "gpu_max": self.gpu_max,
                "duration_seconds": self.duration_seconds,
                "segment_count": self.segment_count,
                "output_size_bytes": self.output_size_bytes,
            }

    class JobEvent(Base):
        """
        Event log for job state changes.

        Tracks the history of status/stage transitions for debugging
        and auditing purposes.
        """

        __tablename__ = "job_events"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        job_id: Mapped[uuid.UUID] = mapped_column(
            UUID(as_uuid=True),
            nullable=False,
            index=True,
        )
        timestamp: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        )
        event_type: Mapped[str] = mapped_column(String(50), nullable=False)
        old_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
        new_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
        old_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
        new_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
        message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

        def __repr__(self) -> str:
            return (
                f"JobEvent(id={self.id}, job_id={self.job_id}, event={self.event_type})"
            )

    class JobMetric(Base):
        """
        Performance metric snapshot during job execution.

        Records CPU, memory, GPU usage at regular intervals
        for monitoring and analysis.
        """

        __tablename__ = "job_metrics"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        job_id: Mapped[uuid.UUID] = mapped_column(
            UUID(as_uuid=True),
            nullable=False,
            index=True,
        )
        timestamp: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        )
        cpu_percent: Mapped[float] = mapped_column(Float, nullable=False)
        memory_percent: Mapped[float] = mapped_column(Float, nullable=False)
        memory_used_mb: Mapped[float] = mapped_column(Float, nullable=False)
        disk_read_mb: Mapped[float] = mapped_column(Float, nullable=True)
        disk_write_mb: Mapped[float] = mapped_column(Float, nullable=True)
        gpu_utilization: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        gpu_memory_used_mb: Mapped[Optional[float]] = mapped_column(
            Float, nullable=True
        )
        gpu_temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

        def __repr__(self) -> str:
            return f"JobMetric(id={self.id}, job_id={self.job_id}, cpu={self.cpu_percent:.1f}%)"

else:
    # Placeholder classes when SQLAlchemy is not available
    class Base:  # type: ignore[no-redef]
        """Placeholder base class."""

        pass

    class User:  # type: ignore[no-redef]
        """Placeholder User class."""

        pass

    class GlobalSettings:  # type: ignore[no-redef]
        """Placeholder GlobalSettings class."""

        pass

    class Job:  # type: ignore[no-redef]
        """Placeholder Job class."""

        pass

    class JobEvent:  # type: ignore[no-redef]
        """Placeholder JobEvent class."""

        pass

    class JobMetric:  # type: ignore[no-redef]
        """Placeholder JobMetric class."""

        pass
