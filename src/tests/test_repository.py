# src/tests/test_repository.py

"""
Tests for the JobRepository database layer.

Uses SQLite in-memory for fast testing without PostgreSQL.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest


# Check if SQLAlchemy is available
try:
    from sqlalchemy import create_engine  # noqa: F401
    from sqlalchemy.orm import sessionmaker  # noqa: F401

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Check if models are available
try:
    from homelab_subs.server.models import (
        Base,  # noqa: F401
        Job,
        JobStatus,
        JobType,
        JobStage,
        JobEvent,
        SQLALCHEMY_AVAILABLE as MODELS_AVAILABLE,
    )
except ImportError:
    MODELS_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (SQLALCHEMY_AVAILABLE and MODELS_AVAILABLE),
    reason="SQLAlchemy or models not available",
)


class TestJobStatusEnum:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """JobStatus should have expected values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.DONE.value == "done"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELED.value == "canceled"

    def test_job_status_from_string(self):
        """JobStatus should be creatable from string."""
        status = JobStatus("pending")
        assert status == JobStatus.PENDING


class TestJobTypeEnum:
    """Tests for JobType enum."""

    def test_job_type_values(self):
        """JobType should have expected values."""
        assert JobType.TRANSCRIBE.value == "transcribe"
        assert JobType.TRANSLATE.value == "translate"
        assert JobType.SYNC_SUBTITLE.value == "sync_subtitle"
        assert JobType.COMPARE.value == "compare"


class TestJobStageEnum:
    """Tests for JobStage enum."""

    def test_job_stage_values(self):
        """JobStage should have expected lifecycle stages."""
        stages = [
            JobStage.QUEUED,
            JobStage.INITIALIZING,
            JobStage.EXTRACTING_AUDIO,
            JobStage.TRANSCRIBING,
            JobStage.TRANSLATING,
            JobStage.GENERATING_SRT,
            JobStage.SYNCING,
            JobStage.COMPARING,
            JobStage.FINALIZING,
            JobStage.COMPLETED,
        ]
        for stage in stages:
            assert stage.value is not None


class TestJobModel:
    """Tests for Job SQLAlchemy model."""

    def test_job_creation_defaults(self):
        """Job should have sensible defaults."""
        job = Job(
            type=JobType.TRANSCRIBE,
            source_path="/media/video.mkv",
            language="en",
        )

        assert job.status == JobStatus.PENDING
        assert job.current_stage == JobStage.QUEUED
        assert job.progress == 0.0
        assert job.model_name == "small"
        assert job.device == "cpu"

    def test_job_to_dict(self):
        """Job.to_dict() should return dictionary representation."""
        job = Job(
            type=JobType.TRANSCRIBE,
            source_path="/media/video.mkv",
            language="en",
        )
        # Set ID manually for test
        job.id = uuid.uuid4()

        job_dict = job.to_dict()
        assert "id" in job_dict
        assert job_dict["type"] == "transcribe"
        assert job_dict["source_path"] == "/media/video.mkv"
        assert job_dict["language"] == "en"
        assert job_dict["status"] == "pending"


class TestJobRepositoryInit:
    """Tests for JobRepository initialization."""

    def test_repository_requires_sqlalchemy(self):
        """JobRepository should require SQLAlchemy."""
        # This is implicitly tested by the module import checks
        from homelab_subs.server.repository import SQLALCHEMY_AVAILABLE

        assert SQLALCHEMY_AVAILABLE is True


@pytest.fixture
def mock_settings():
    """Create mock settings for repository."""
    settings = MagicMock()
    settings.database_url = "sqlite:///:memory:"
    settings.log_level = "INFO"
    return settings


@pytest.fixture
def in_memory_repository(mock_settings):
    """Create a repository with in-memory SQLite for testing."""
    from homelab_subs.server.repository import JobRepository

    # Create repository with SQLite
    with patch(
        "homelab_subs.server.repository.get_settings", return_value=mock_settings
    ):
        repo = JobRepository(database_url="sqlite:///:memory:", settings=mock_settings)
        repo.create_tables()
        yield repo


class TestJobRepositoryWithMock:
    """Tests for JobRepository with mocked database."""

    def test_repository_initialization(self, mock_settings):
        """Repository should initialize with settings."""
        from homelab_subs.server.repository import JobRepository

        with patch(
            "homelab_subs.server.repository.get_settings", return_value=mock_settings
        ):
            repo = JobRepository(
                database_url="sqlite:///:memory:", settings=mock_settings
            )
            assert repo is not None

    def test_create_tables(self, in_memory_repository):
        """create_tables should not raise."""
        # Tables already created in fixture
        in_memory_repository.create_tables()
        # Should not raise

    def test_create_job(self, in_memory_repository):
        """create_job should create a new job."""
        job = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test.mkv",
            language="en",
            model_name="tiny",
        )

        assert job is not None
        assert job.id is not None
        assert job.type == JobType.TRANSCRIBE
        assert job.source_path == "/media/test.mkv"
        assert job.status == JobStatus.PENDING

    def test_get_job(self, in_memory_repository):
        """get_job should retrieve a job by ID."""
        created_job = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test.mkv",
            language="en",
        )

        retrieved_job = in_memory_repository.get_job(created_job.id)
        assert retrieved_job is not None
        assert retrieved_job.id == created_job.id
        assert retrieved_job.source_path == created_job.source_path

    def test_get_job_not_found(self, in_memory_repository):
        """get_job should return None for non-existent job."""
        result = in_memory_repository.get_job(uuid.uuid4())
        assert result is None

    def test_get_job_with_string_id(self, in_memory_repository):
        """get_job should accept string UUID."""
        created_job = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test.mkv",
            language="en",
        )

        retrieved_job = in_memory_repository.get_job(str(created_job.id))
        assert retrieved_job is not None

    def test_list_jobs_empty(self, in_memory_repository):
        """list_jobs should return empty list when no jobs."""
        jobs = in_memory_repository.list_jobs()
        assert jobs == []

    def test_list_jobs_returns_jobs(self, in_memory_repository):
        """list_jobs should return created jobs."""
        in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test1.mkv",
            language="en",
        )
        in_memory_repository.create_job(
            job_type=JobType.TRANSLATE,
            source_path="/media/test2.mkv",
            language="en",
            target_language="es",
        )

        jobs = in_memory_repository.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_filter_by_status(self, in_memory_repository):
        """list_jobs should filter by status."""
        job1 = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test1.mkv",
            language="en",
        )
        in_memory_repository.update_status(job1.id, JobStatus.RUNNING)

        in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test2.mkv",
            language="en",
        )

        pending_jobs = in_memory_repository.list_jobs(status=JobStatus.PENDING)
        assert len(pending_jobs) == 1

        running_jobs = in_memory_repository.list_jobs(status=JobStatus.RUNNING)
        assert len(running_jobs) == 1

    def test_list_jobs_filter_by_type(self, in_memory_repository):
        """list_jobs should filter by job type."""
        in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test1.mkv",
            language="en",
        )
        in_memory_repository.create_job(
            job_type=JobType.TRANSLATE,
            source_path="/media/test2.mkv",
            language="en",
            target_language="es",
        )

        transcribe_jobs = in_memory_repository.list_jobs(job_type=JobType.TRANSCRIBE)
        assert len(transcribe_jobs) == 1
        assert transcribe_jobs[0].type == JobType.TRANSCRIBE

    def test_update_status(self, in_memory_repository):
        """update_status should change job status."""
        job = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test.mkv",
            language="en",
        )

        result = in_memory_repository.update_status(job.id, JobStatus.RUNNING)
        assert result is True

        updated_job = in_memory_repository.get_job(job.id)
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None

    def test_update_status_not_found(self, in_memory_repository):
        """update_status should return False for non-existent job."""
        result = in_memory_repository.update_status(uuid.uuid4(), JobStatus.RUNNING)
        assert result is False

    def test_update_status_with_error(self, in_memory_repository):
        """update_status should store error message."""
        job = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test.mkv",
            language="en",
        )

        result = in_memory_repository.update_status(
            job.id, JobStatus.FAILED, error_message="Something went wrong"
        )
        assert result is True

        updated_job = in_memory_repository.get_job(job.id)
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error_message == "Something went wrong"

    def test_update_progress(self, in_memory_repository):
        """update_progress should change job progress."""
        job = in_memory_repository.create_job(
            job_type=JobType.TRANSCRIBE,
            source_path="/media/test.mkv",
            language="en",
        )

        result = in_memory_repository.update_progress(
            job.id, 50.0, stage=JobStage.TRANSCRIBING
        )
        assert result is True

        updated_job = in_memory_repository.get_job(job.id)
        assert updated_job.progress == 50.0
        assert updated_job.current_stage == JobStage.TRANSCRIBING

    def test_update_progress_not_found(self, in_memory_repository):
        """update_progress should return False for non-existent job."""
        result = in_memory_repository.update_progress(uuid.uuid4(), 50.0)
        assert result is False


class TestJobEvent:
    """Tests for JobEvent model."""

    def test_job_event_creation(self):
        """JobEvent should record job lifecycle events."""
        event = JobEvent(
            job_id=uuid.uuid4(),
            event_type="status_change",
            old_status="pending",
            new_status="running",
            message="Job started",
        )

        assert event.event_type == "status_change"
        assert event.old_status == "pending"
        assert event.new_status == "running"
