# src/tests/test_worker.py

"""
Tests for the RQ worker and job processing.

Uses mocked dependencies for testing without actual job execution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# Check if worker dependencies are available
try:
    from redis import Redis  # noqa: F401
    from rq import Worker  # noqa: F401
    from sqlalchemy import create_engine  # noqa: F401

    WORKER_DEPS_AVAILABLE = True
except ImportError:
    WORKER_DEPS_AVAILABLE = False


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not WORKER_DEPS_AVAILABLE,
    reason="Worker dependencies (redis, rq, sqlalchemy) not installed",
)


class TestJobCancelledException:
    """Tests for JobCancelledException."""

    def test_exception_creation(self):
        """JobCancelledException should be raised with message."""
        from homelab_subs.server.worker import JobCancelledException

        with pytest.raises(JobCancelledException) as excinfo:
            raise JobCancelledException("Job 123 was cancelled")

        assert "Job 123 was cancelled" in str(excinfo.value)


class TestJobContext:
    """Tests for JobContext context manager."""

    @patch("homelab_subs.server.worker.get_settings")
    @patch("homelab_subs.server.worker.create_engine")
    @patch("homelab_subs.server.worker.sessionmaker")
    def test_job_context_initialization(
        self, mock_sessionmaker, mock_create_engine, mock_settings
    ):
        """JobContext should initialize with job_id."""
        from homelab_subs.server.worker import JobContext

        settings = MagicMock()
        settings.sync_database_url = "postgresql://localhost/test"
        mock_settings.return_value = settings

        ctx = JobContext("test-job-id")
        assert ctx.job_id == "test-job-id"

    @patch("homelab_subs.server.worker.get_settings")
    @patch("homelab_subs.server.worker.create_engine")
    @patch("homelab_subs.server.worker.sessionmaker")
    def test_job_context_enter_loads_job(
        self, mock_sessionmaker, mock_create_engine, mock_settings
    ):
        """JobContext __enter__ should load job from database."""
        from homelab_subs.server.worker import JobContext
        from homelab_subs.server.models import JobStatus

        settings = MagicMock()
        settings.sync_database_url = "postgresql://localhost/test"
        mock_settings.return_value = settings

        # Mock session and query
        mock_session = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "test-job-id"
        mock_job.status = JobStatus.PENDING

        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_session.query.return_value = mock_query

        mock_session_class = MagicMock(return_value=mock_session)
        mock_sessionmaker.return_value = mock_session_class

        ctx = JobContext("test-job-id")
        result = ctx.__enter__()

        assert result == ctx
        assert ctx.job == mock_job

    @patch("homelab_subs.server.worker.get_settings")
    @patch("homelab_subs.server.worker.create_engine")
    @patch("homelab_subs.server.worker.sessionmaker")
    def test_job_context_enter_raises_on_missing_job(
        self, mock_sessionmaker, mock_create_engine, mock_settings
    ):
        """JobContext __enter__ should raise if job not found."""
        from homelab_subs.server.worker import JobContext

        settings = MagicMock()
        settings.sync_database_url = "postgresql://localhost/test"
        mock_settings.return_value = settings

        # Mock session returning None for job
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        mock_session_class = MagicMock(return_value=mock_session)
        mock_sessionmaker.return_value = mock_session_class

        ctx = JobContext("nonexistent-job-id")

        with pytest.raises(ValueError) as excinfo:
            ctx.__enter__()

        assert "Job not found" in str(excinfo.value)


class TestJobContextMethods:
    """Tests for JobContext helper methods."""

    @pytest.fixture
    def mock_context(self):
        """Create a mocked JobContext."""
        from homelab_subs.server.worker import JobContext
        from homelab_subs.server.models import JobStatus, JobStage

        with patch("homelab_subs.server.worker.get_settings") as mock_settings:
            with patch("homelab_subs.server.worker.create_engine"):
                with patch("homelab_subs.server.worker.sessionmaker"):
                    settings = MagicMock()
                    settings.sync_database_url = "postgresql://localhost/test"
                    mock_settings.return_value = settings

                    ctx = JobContext("test-job-id")
                    ctx.session = MagicMock()
                    ctx.job = MagicMock()
                    ctx.job.status = JobStatus.RUNNING
                    ctx.job.stage = JobStage.INITIALIZING
                    ctx.job.progress = 0
                    ctx.job.logs = None

                    yield ctx

    def test_check_cancelled_false(self, mock_context):
        """check_cancelled should return False for running job."""
        from homelab_subs.server.models import JobStatus

        mock_context.job.status = JobStatus.RUNNING
        result = mock_context.check_cancelled()
        assert result is False

    def test_check_cancelled_true(self, mock_context):
        """check_cancelled should return True for cancelled job."""
        from homelab_subs.server.models import JobStatus

        mock_context.job.status = JobStatus.CANCELLED
        result = mock_context.check_cancelled()
        assert result is True

    def test_raise_if_cancelled_raises(self, mock_context):
        """raise_if_cancelled should raise when job cancelled."""
        from homelab_subs.server.worker import JobCancelledException
        from homelab_subs.server.models import JobStatus

        mock_context.job.status = JobStatus.CANCELLED

        with pytest.raises(JobCancelledException):
            mock_context.raise_if_cancelled()

    def test_raise_if_cancelled_not_cancelled(self, mock_context):
        """raise_if_cancelled should not raise for running job."""
        from homelab_subs.server.models import JobStatus

        mock_context.job.status = JobStatus.RUNNING
        # Should not raise
        mock_context.raise_if_cancelled()

    def test_update_stage(self, mock_context):
        """update_stage should update job stage and progress."""
        from homelab_subs.server.models import JobStage

        mock_context.update_stage(JobStage.TRANSCRIBING, progress=25)

        assert mock_context.job.stage == JobStage.TRANSCRIBING
        assert mock_context.job.progress == 25
        mock_context.session.commit.assert_called()

    def test_update_progress(self, mock_context):
        """update_progress should update job progress."""
        mock_context.update_progress(50)

        assert mock_context.job.progress == 50
        mock_context.session.commit.assert_called()

    def test_update_progress_clamps_values(self, mock_context):
        """update_progress should clamp to 0-100."""
        mock_context.update_progress(150)
        assert mock_context.job.progress == 100

        mock_context.update_progress(-10)
        assert mock_context.job.progress == 0

    def test_add_log(self, mock_context):
        """add_log should append log message."""
        mock_context.add_log("Processing started", level="info")

        assert mock_context.job.logs is not None
        assert "Processing started" in mock_context.job.logs
        assert "[INFO]" in mock_context.job.logs

    def test_add_log_appends(self, mock_context):
        """add_log should append to existing logs."""
        mock_context.job.logs = "Previous log"
        mock_context.add_log("New log")

        assert "Previous log" in mock_context.job.logs
        assert "New log" in mock_context.job.logs

    def test_complete(self, mock_context):
        """complete should mark job as completed."""
        from homelab_subs.server.models import JobStatus, JobStage

        mock_context.complete(output_path="/output/test.srt")

        assert mock_context.job.status == JobStatus.COMPLETED
        assert mock_context.job.stage == JobStage.COMPLETED
        assert mock_context.job.progress == 100
        assert mock_context.job.output_path == "/output/test.srt"

    def test_fail(self, mock_context):
        """fail should mark job as failed with error."""
        from homelab_subs.server.models import JobStatus

        mock_context.fail("Something went wrong")

        assert mock_context.job.status == JobStatus.FAILED
        assert mock_context.job.error_message == "Something went wrong"


class TestProcessJob:
    """Tests for the main process_job function."""

    @patch("homelab_subs.server.worker.JobContext")
    def test_process_job_not_found(self, mock_context_class):
        """process_job should handle job not found."""
        from homelab_subs.server.worker import process_job

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(side_effect=ValueError("Job not found"))
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context_class.return_value = mock_context

        result = process_job("nonexistent-job")

        assert result["status"] == "failed"
        assert "Job not found" in result["error"]


class TestWorkerHelpers:
    """Tests for worker helper functions."""

    def test_temp_file_creation(self):
        """create_temp_file should create temp file."""
        from homelab_subs.server.worker import JobContext

        with patch("homelab_subs.server.worker.get_settings") as mock_settings:
            with patch("homelab_subs.server.worker.create_engine"):
                with patch("homelab_subs.server.worker.sessionmaker"):
                    settings = MagicMock()
                    settings.sync_database_url = "postgresql://localhost/test"
                    mock_settings.return_value = settings

                    ctx = JobContext("test-job-id")
                    temp_file = ctx.create_temp_file(suffix=".wav")

                    assert temp_file.exists()
                    assert str(temp_file).endswith(".wav")

                    # Cleanup
                    temp_file.unlink()
