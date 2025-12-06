# src/homelab_subs/server/repository.py

"""
Database repository for job management.

Provides a clean abstraction over SQLAlchemy for CRUD operations
on jobs and related entities.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional, Sequence

try:
    from sqlalchemy import create_engine, select, desc, and_
    from sqlalchemy.orm import Session, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .models import (
    Base,
    Job,
    JobEvent,
    JobMetric,
    JobStatus,
    JobType,
    JobStage,
    User,
    GlobalSettings,
    SQLALCHEMY_AVAILABLE as MODELS_AVAILABLE,
)
from .settings import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class JobRepository:
    """
    Repository for managing Job entities in PostgreSQL.

    This class provides all database operations for jobs including:
    - Creating new jobs
    - Updating job status and progress
    - Querying jobs with filters
    - Recording events and metrics

    Parameters
    ----------
    database_url : str, optional
        PostgreSQL connection URL. If not provided, uses settings.
    settings : Settings, optional
        Application settings. If not provided, uses get_settings().

    Examples
    --------
    >>> repo = JobRepository()
    >>> job = repo.create_job(
    ...     job_type=JobType.TRANSCRIBE,
    ...     source_path="/media/movie.mkv",
    ...     language="en",
    ... )
    >>> print(job.id)

    >>> repo.update_status(job.id, JobStatus.RUNNING)
    >>> repo.update_progress(job.id, 50.0, JobStage.TRANSCRIBING)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not SQLALCHEMY_AVAILABLE or not MODELS_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy is required for JobRepository. "
                "Install with: pip install homelab-subtitle-service[server]"
            )

        self._settings = settings or get_settings()
        self._database_url = database_url or self._settings.database_url

        self._engine = create_engine(
            self._database_url,
            echo=self._settings.log_level == "DEBUG",
            pool_pre_ping=True,  # Check connection health
        )
        self._session_factory = sessionmaker(bind=self._engine)

        logger.info(
            f"JobRepository initialized with database: {self._database_url.split('@')[-1]}"
        )

    def create_tables(self) -> None:
        """Create all database tables if they don't exist."""
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created/verified")

    def drop_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        Base.metadata.drop_all(self._engine)
        logger.warning("Database tables dropped")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Handles commit/rollback automatically.

        Yields
        ------
        Session
            SQLAlchemy session for database operations.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # =========================================================================
    # Job CRUD Operations
    # =========================================================================

    def create_job(
        self,
        *,
        job_type: JobType,
        source_path: str,
        language: str = "en",
        target_language: Optional[str] = None,
        subtitle_path: Optional[str] = None,
        output_path: Optional[str] = None,
        model_name: str = "small",
        backend: str = "faster-whisper",
        device: str = "cpu",
        compute_type: str = "int8",
        priority: int = 0,
        options: Optional[dict[str, Any]] = None,
    ) -> Job:
        """
        Create a new job in the database.

        Parameters
        ----------
        job_type : JobType
            Type of job (transcribe, translate, sync, compare).
        source_path : str
            Path to source video file.
        language : str
            Source audio language code.
        target_language : str, optional
            Translation target language.
        subtitle_path : str, optional
            Path to existing subtitle for sync/compare.
        output_path : str, optional
            Path for output SRT file.
        model_name : str
            Whisper model name.
        backend : str
            Processing backend.
        device : str
            Device to use (cpu, cuda).
        compute_type : str
            Compute precision.
        priority : int
            Job priority (higher = more urgent).
        options : dict, optional
            Additional job options.

        Returns
        -------
        Job
            The created job entity.
        """
        job = Job(
            type=job_type,
            source_path=source_path,
            language=language,
            target_language=target_language,
            subtitle_path=subtitle_path,
            output_path=output_path,
            model_name=model_name,
            backend=backend,
            device=device,
            compute_type=compute_type,
            priority=priority,
            options=options or {},
            status=JobStatus.PENDING,
            current_stage=JobStage.QUEUED,
            progress=0.0,
        )

        with self.session() as session:
            session.add(job)
            session.flush()  # Get the ID
            job_id = job.id

            # Record creation event
            event = JobEvent(
                job_id=job_id,
                event_type="created",
                new_status=JobStatus.PENDING.value,
                new_stage=JobStage.QUEUED.value,
                message=f"Job created: {job_type.value} for {source_path}",
            )
            session.add(event)

        logger.info(f"Created job {job_id}: {job_type.value} for {source_path}")
        return self.get_job(job_id)  # Return fresh copy

    def get_job(self, job_id: uuid.UUID | str) -> Optional[Job]:
        """
        Get a job by ID.

        Parameters
        ----------
        job_id : UUID or str
            The job ID to look up.

        Returns
        -------
        Job or None
            The job if found, None otherwise.
        """
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            job = session.get(Job, job_id)
            if job:
                session.expunge(job)  # Detach from session
            return job

    def list_jobs(
        self,
        *,
        status: Optional[JobStatus | str] = None,
        job_type: Optional[JobType | str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> Sequence[Job]:
        """
        List jobs with optional filtering.

        Parameters
        ----------
        status : JobStatus or str, optional
            Filter by job status.
        job_type : JobType or str, optional
            Filter by job type.
        limit : int
            Maximum number of jobs to return.
        offset : int
            Number of jobs to skip (for pagination).
        order_by : str
            Column to order by.
        order_desc : bool
            Whether to order descending (newest first).

        Returns
        -------
        Sequence[Job]
            List of matching jobs.
        """
        with self.session() as session:
            query = select(Job)

            # Apply filters
            conditions = []
            if status is not None:
                if isinstance(status, str):
                    status = JobStatus(status)
                conditions.append(Job.status == status)

            if job_type is not None:
                if isinstance(job_type, str):
                    job_type = JobType(job_type)
                conditions.append(Job.type == job_type)

            if conditions:
                query = query.where(and_(*conditions))

            # Apply ordering
            order_column = getattr(Job, order_by, Job.created_at)
            if order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(order_column)

            # Apply pagination
            query = query.limit(limit).offset(offset)

            jobs = list(session.scalars(query).all())
            for job in jobs:
                session.expunge(job)

            return jobs

    def update_status(
        self,
        job_id: uuid.UUID | str,
        status: JobStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update job status.

        Parameters
        ----------
        job_id : UUID or str
            The job ID to update.
        status : JobStatus
            New status.
        error_message : str, optional
            Error message if status is FAILED.

        Returns
        -------
        bool
            True if job was updated, False if not found.
        """
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            job = session.get(Job, job_id)
            if not job:
                return False

            old_status = job.status
            job.status = status

            # Update timestamps
            now = datetime.utcnow()
            if status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = now
            elif status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED):
                job.finished_at = now
                if job.started_at:
                    job.duration_seconds = (now - job.started_at).total_seconds()

            if error_message:
                job.error_message = error_message

            # Record event
            event = JobEvent(
                job_id=job_id,
                event_type="status_change",
                old_status=old_status.value,
                new_status=status.value,
                message=error_message,
            )
            session.add(event)

        logger.info(f"Job {job_id} status: {old_status.value} -> {status.value}")
        return True

    def update_progress(
        self,
        job_id: uuid.UUID | str,
        progress: float,
        stage: Optional[JobStage] = None,
    ) -> bool:
        """
        Update job progress.

        Parameters
        ----------
        job_id : UUID or str
            The job ID to update.
        progress : float
            Progress percentage (0-100).
        stage : JobStage, optional
            New processing stage.

        Returns
        -------
        bool
            True if job was updated, False if not found.
        """
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            job = session.get(Job, job_id)
            if not job:
                return False

            old_stage = job.current_stage
            job.progress = min(100.0, max(0.0, progress))

            if stage and stage != old_stage:
                job.current_stage = stage

                # Record stage change event
                event = JobEvent(
                    job_id=job_id,
                    event_type="stage_change",
                    old_stage=old_stage.value,
                    new_stage=stage.value,
                )
                session.add(event)
                logger.debug(f"Job {job_id} stage: {old_stage.value} -> {stage.value}")

        return True

    def update_metrics(
        self,
        job_id: uuid.UUID | str,
        *,
        cpu_avg: Optional[float] = None,
        cpu_max: Optional[float] = None,
        memory_avg_mb: Optional[float] = None,
        memory_max_mb: Optional[float] = None,
        gpu_avg: Optional[float] = None,
        gpu_max: Optional[float] = None,
        segment_count: Optional[int] = None,
        output_size_bytes: Optional[int] = None,
    ) -> bool:
        """
        Update job performance metrics.

        Called when job completes to store aggregate metrics.
        """
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            job = session.get(Job, job_id)
            if not job:
                return False

            if cpu_avg is not None:
                job.cpu_avg = cpu_avg
            if cpu_max is not None:
                job.cpu_max = cpu_max
            if memory_avg_mb is not None:
                job.memory_avg_mb = memory_avg_mb
            if memory_max_mb is not None:
                job.memory_max_mb = memory_max_mb
            if gpu_avg is not None:
                job.gpu_avg = gpu_avg
            if gpu_max is not None:
                job.gpu_max = gpu_max
            if segment_count is not None:
                job.segment_count = segment_count
            if output_size_bytes is not None:
                job.output_size_bytes = output_size_bytes

        return True

    def set_output_path(self, job_id: uuid.UUID | str, output_path: str) -> bool:
        """Set the output path for a job."""
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            job = session.get(Job, job_id)
            if not job:
                return False
            job.output_path = output_path

        return True

    # =========================================================================
    # Metric Recording
    # =========================================================================

    def record_metric(
        self,
        job_id: uuid.UUID | str,
        *,
        cpu_percent: float,
        memory_percent: float,
        memory_used_mb: float,
        disk_read_mb: float = 0.0,
        disk_write_mb: float = 0.0,
        gpu_utilization: Optional[float] = None,
        gpu_memory_used_mb: Optional[float] = None,
        gpu_temperature: Optional[float] = None,
    ) -> None:
        """
        Record a performance metric snapshot.

        Called periodically during job execution.
        """
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        metric = JobMetric(
            job_id=job_id,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_temperature=gpu_temperature,
        )

        with self.session() as session:
            session.add(metric)

    def get_job_metrics(
        self,
        job_id: uuid.UUID | str,
        limit: int = 1000,
    ) -> Sequence[JobMetric]:
        """Get performance metrics for a job."""
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            query = (
                select(JobMetric)
                .where(JobMetric.job_id == job_id)
                .order_by(JobMetric.timestamp)
                .limit(limit)
            )
            metrics = list(session.scalars(query).all())
            for m in metrics:
                session.expunge(m)
            return metrics

    # =========================================================================
    # Event Logging
    # =========================================================================

    def record_event(
        self,
        job_id: uuid.UUID | str,
        event_type: str,
        message: Optional[str] = None,
    ) -> None:
        """Record a custom event for a job."""
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        event = JobEvent(
            job_id=job_id,
            event_type=event_type,
            message=message,
        )

        with self.session() as session:
            session.add(event)

    def get_job_events(
        self,
        job_id: uuid.UUID | str,
        limit: int = 100,
    ) -> Sequence[JobEvent]:
        """Get events for a job."""
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            query = (
                select(JobEvent)
                .where(JobEvent.job_id == job_id)
                .order_by(JobEvent.timestamp)
                .limit(limit)
            )
            events = list(session.scalars(query).all())
            for e in events:
                session.expunge(e)
            return events

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get aggregate statistics across all jobs.

        Returns
        -------
        dict
            Statistics including counts by status, average duration, etc.
        """
        with self.session() as session:
            # Count by status
            status_counts = {}
            for status in JobStatus:
                count = session.scalar(
                    select(func.count(Job.id)).where(Job.status == status)
                )
                status_counts[status.value] = count or 0

            # Average duration for completed jobs
            avg_duration = session.scalar(
                select(func.avg(Job.duration_seconds))
                .where(Job.status == JobStatus.DONE)
                .where(Job.duration_seconds.isnot(None))
            )

            # Total jobs
            total = sum(status_counts.values())

            return {
                "total_jobs": total,
                "status_counts": status_counts,
                "avg_duration_seconds": float(avg_duration) if avg_duration else None,
                "pending_count": status_counts.get(JobStatus.PENDING.value, 0),
                "running_count": status_counts.get(JobStatus.RUNNING.value, 0),
            }


# Import func for statistics (SQLAlchemy)
if SQLALCHEMY_AVAILABLE:
    from sqlalchemy import func


class UserRepository:
    """
    Repository for managing User entities in PostgreSQL.

    This class provides all database operations for users including:
    - Creating new users (with password hashing)
    - User authentication
    - Updating user information

    Parameters
    ----------
    database_url : str, optional
        PostgreSQL connection URL. If not provided, uses settings.
    settings : Settings, optional
        Application settings.

    Examples
    --------
    >>> repo = UserRepository()
    >>> user = repo.create_user("admin", "SecurePass123")
    >>> authenticated = repo.authenticate("admin", "SecurePass123")
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not SQLALCHEMY_AVAILABLE or not MODELS_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy is required for UserRepository. "
                "Install with: pip install homelab-subtitle-service[server]"
            )

        self._settings = settings or get_settings()
        self._database_url = database_url or self._settings.database_url

        self._engine = create_engine(
            self._database_url,
            echo=self._settings.log_level == "DEBUG",
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)

        logger.info("UserRepository initialized")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_user(
        self,
        username: str,
        password: str,
        is_admin: bool = False,
    ) -> User:
        """
        Create a new user with hashed password.

        Parameters
        ----------
        username : str
            Unique username.
        password : str
            Plain text password (will be hashed).
        is_admin : bool
            Whether user has admin privileges.

        Returns
        -------
        User
            The created user entity.

        Raises
        ------
        ValueError
            If username already exists.
        """
        from .auth import hash_password, validate_password_strength

        # Validate password strength
        validate_password_strength(password)

        # Hash the password
        password_hash = hash_password(password)

        user = User(
            username=username,
            password_hash=password_hash,
            is_admin=is_admin,
            is_active=True,
        )

        with self.session() as session:
            # Check if username exists
            existing = session.scalar(select(User).where(User.username == username))
            if existing:
                raise ValueError(f"Username '{username}' already exists")

            session.add(user)
            session.flush()
            user_id = user.id

        logger.info(f"Created user '{username}' (admin={is_admin})")
        return self.get_user_by_id(user_id)

    def get_user_by_id(self, user_id: uuid.UUID | str) -> Optional[User]:
        """Get a user by their ID."""
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)

        with self.session() as session:
            user = session.get(User, user_id)
            if user:
                session.expunge(user)
            return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by their username."""
        with self.session() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user:
                session.expunge(user)
            return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.

        Parameters
        ----------
        username : str
            Username to authenticate.
        password : str
            Plain text password.

        Returns
        -------
        User or None
            The authenticated user if credentials are valid, None otherwise.
        """
        from .auth import verify_password

        user = self.get_user_by_username(username)

        if user is None:
            logger.warning(f"Authentication failed: user '{username}' not found")
            return None

        if not user.is_active:
            logger.warning(f"Authentication failed: user '{username}' is inactive")
            return None

        if not verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: invalid password for '{username}'")
            return None

        # Update last login
        self.update_last_login(user.id)

        logger.info(f"User '{username}' authenticated successfully")
        return user

    def update_last_login(self, user_id: uuid.UUID | str) -> None:
        """Update user's last login timestamp."""
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)

        with self.session() as session:
            user = session.get(User, user_id)
            if user:
                user.last_login = datetime.utcnow()

    def update_password(
        self,
        user_id: uuid.UUID | str,
        new_password: str,
    ) -> bool:
        """
        Update a user's password.

        Parameters
        ----------
        user_id : UUID or str
            User ID.
        new_password : str
            New plain text password (will be hashed).

        Returns
        -------
        bool
            True if password was updated, False if user not found.
        """
        from .auth import hash_password, validate_password_strength

        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)

        # Validate password strength
        validate_password_strength(new_password)

        password_hash = hash_password(new_password)

        with self.session() as session:
            user = session.get(User, user_id)
            if not user:
                return False

            user.password_hash = password_hash

        logger.info(f"Password updated for user {user_id}")
        return True

    def set_user_active(self, user_id: uuid.UUID | str, is_active: bool) -> bool:
        """Enable or disable a user account."""
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)

        with self.session() as session:
            user = session.get(User, user_id)
            if not user:
                return False

            user.is_active = is_active

        logger.info(f"User {user_id} active status set to {is_active}")
        return True

    def count_users(self) -> int:
        """Get total number of users."""
        with self.session() as session:
            if SQLALCHEMY_AVAILABLE:
                count = session.scalar(select(func.count(User.id)))
                return count or 0
            return 0

    def has_any_users(self) -> bool:
        """Check if any users exist (for first-time setup)."""
        return self.count_users() > 0


class SettingsRepository:
    """
    Repository for managing GlobalSettings in PostgreSQL.

    GlobalSettings uses a singleton pattern - there's only one row
    with id=1 that contains all server-wide settings.

    Parameters
    ----------
    database_url : str, optional
        PostgreSQL connection URL.
    settings : Settings, optional
        Application settings.

    Examples
    --------
    >>> repo = SettingsRepository()
    >>> settings = repo.get_settings()
    >>> repo.update_settings(worker_count=4, prefer_gpu=True)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not SQLALCHEMY_AVAILABLE or not MODELS_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy is required for SettingsRepository. "
                "Install with: pip install homelab-subtitle-service[server]"
            )

        self._settings = settings or get_settings()
        self._database_url = database_url or self._settings.database_url

        self._engine = create_engine(
            self._database_url,
            echo=self._settings.log_level == "DEBUG",
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)

        logger.info("SettingsRepository initialized")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_settings(self) -> GlobalSettings:
        """
        Get the global settings.

        Creates default settings if none exist.

        Returns
        -------
        GlobalSettings
            The server-wide settings.
        """
        with self.session() as session:
            settings = session.get(GlobalSettings, 1)

            if settings is None:
                # Create default settings
                settings = GlobalSettings()
                session.add(settings)
                session.flush()
                logger.info("Created default global settings")

            session.expunge(settings)
            return settings

    def update_settings(self, **kwargs: Any) -> GlobalSettings:
        """
        Update global settings.

        Parameters
        ----------
        **kwargs
            Settings fields to update. Valid fields:
            - media_folders: list[str]
            - default_model: str
            - default_device: str
            - default_compute_type: str
            - default_language: str
            - default_translation_backend: str
            - worker_count: int
            - log_retention_days: int
            - job_retention_days: int
            - prefer_gpu: bool

        Returns
        -------
        GlobalSettings
            The updated settings.

        Raises
        ------
        ValueError
            If an invalid setting field is provided.
        """
        valid_fields = {
            "media_folders",
            "default_model",
            "default_device",
            "default_compute_type",
            "default_language",
            "default_translation_backend",
            "worker_count",
            "log_retention_days",
            "job_retention_days",
            "prefer_gpu",
        }

        invalid_fields = set(kwargs.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid settings fields: {invalid_fields}")

        with self.session() as session:
            settings = session.get(GlobalSettings, 1)

            if settings is None:
                settings = GlobalSettings()
                session.add(settings)

            for field, value in kwargs.items():
                setattr(settings, field, value)

            settings.updated_at = datetime.utcnow()
            session.flush()

        logger.info(f"Updated global settings: {list(kwargs.keys())}")
        return self.get_settings()

    def add_media_folder(self, folder_path: str) -> GlobalSettings:
        """
        Add a media folder to the allowed list.

        Parameters
        ----------
        folder_path : str
            Absolute path to media folder.

        Returns
        -------
        GlobalSettings
            The updated settings.
        """
        settings = self.get_settings()
        folders = list(settings.media_folders)

        if folder_path not in folders:
            folders.append(folder_path)
            return self.update_settings(media_folders=folders)

        return settings

    def remove_media_folder(self, folder_path: str) -> GlobalSettings:
        """
        Remove a media folder from the allowed list.

        Parameters
        ----------
        folder_path : str
            Path to remove.

        Returns
        -------
        GlobalSettings
            The updated settings.
        """
        settings = self.get_settings()
        folders = list(settings.media_folders)

        if folder_path in folders:
            folders.remove(folder_path)
            return self.update_settings(media_folders=folders)

        return settings
