# src/tests/test_server.py

"""
Tests for the server module components.

These tests cover:
- Settings configuration
- Job models and enums
- JobRepository (mocked database)
- QueueClient (mocked Redis)
- ServerJobService (integration of repository + queue)
- FastAPI API endpoints

Note: These tests require the 'server' optional dependencies.
      Install with: pip install homelab-subs[server]
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# Check if server dependencies are available
try:
    from homelab_subs.server.models import Job, JobStatus, JobType, JobStage
    from homelab_subs.server.settings import Settings, get_settings

    SERVER_DEPS_AVAILABLE = True
except ImportError:
    SERVER_DEPS_AVAILABLE = False
    Job = None
    JobStatus = None
    JobType = None
    JobStage = None
    Settings = None
    get_settings = None


pytestmark = pytest.mark.skipif(
    not SERVER_DEPS_AVAILABLE,
    reason="Server dependencies not installed. Install with: pip install homelab-subs[server]",
)


# =============================================================================
# JobStatus Enum Tests
# =============================================================================


class TestJobStatusEnum:
    """Tests for JobStatus enumeration."""

    def test_has_pending_status(self):
        """Should have PENDING status."""
        assert hasattr(JobStatus, "PENDING")
        assert JobStatus.PENDING.value == "pending"

    def test_has_running_status(self):
        """Should have RUNNING status."""
        assert hasattr(JobStatus, "RUNNING")
        assert JobStatus.RUNNING.value == "running"

    def test_has_completed_status(self):
        """Should have COMPLETED status."""
        assert hasattr(JobStatus, "COMPLETED")
        assert JobStatus.COMPLETED.value == "completed"

    def test_has_failed_status(self):
        """Should have FAILED status."""
        assert hasattr(JobStatus, "FAILED")
        assert JobStatus.FAILED.value == "failed"

    def test_has_cancelled_status(self):
        """Should have CANCELLED status."""
        assert hasattr(JobStatus, "CANCELLED")
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_all_statuses_are_strings(self):
        """All status values should be strings."""
        for status in JobStatus:
            assert isinstance(status.value, str)

    def test_status_values_are_lowercase(self):
        """All status values should be lowercase."""
        for status in JobStatus:
            assert status.value == status.value.lower()


# =============================================================================
# JobType Enum Tests
# =============================================================================


class TestJobTypeEnum:
    """Tests for JobType enumeration."""

    def test_has_transcribe_type(self):
        """Should have TRANSCRIBE type."""
        assert hasattr(JobType, "TRANSCRIBE")
        assert JobType.TRANSCRIBE.value == "transcribe"

    def test_has_translate_type(self):
        """Should have TRANSLATE type."""
        assert hasattr(JobType, "TRANSLATE")
        assert JobType.TRANSLATE.value == "translate"

    def test_has_sync_type(self):
        """Should have SYNC type."""
        assert hasattr(JobType, "SYNC")
        assert JobType.SYNC.value == "sync"

    def test_has_compare_type(self):
        """Should have COMPARE type."""
        assert hasattr(JobType, "COMPARE")
        assert JobType.COMPARE.value == "compare"

    def test_has_full_pipeline_type(self):
        """Should have FULL_PIPELINE type."""
        assert hasattr(JobType, "FULL_PIPELINE")
        assert JobType.FULL_PIPELINE.value == "full_pipeline"

    def test_all_types_are_strings(self):
        """All type values should be strings."""
        for job_type in JobType:
            assert isinstance(job_type.value, str)


# =============================================================================
# JobStage Enum Tests
# =============================================================================


class TestJobStageEnum:
    """Tests for JobStage enumeration."""

    def test_has_queued_stage(self):
        """Should have QUEUED stage."""
        assert hasattr(JobStage, "QUEUED")
        assert JobStage.QUEUED.value == "queued"

    def test_has_extracting_audio_stage(self):
        """Should have EXTRACTING_AUDIO stage."""
        assert hasattr(JobStage, "EXTRACTING_AUDIO")
        assert JobStage.EXTRACTING_AUDIO.value == "extracting_audio"

    def test_has_transcribing_stage(self):
        """Should have TRANSCRIBING stage."""
        assert hasattr(JobStage, "TRANSCRIBING")
        assert JobStage.TRANSCRIBING.value == "transcribing"

    def test_has_translating_stage(self):
        """Should have TRANSLATING stage."""
        assert hasattr(JobStage, "TRANSLATING")
        assert JobStage.TRANSLATING.value == "translating"

    def test_has_syncing_stage(self):
        """Should have SYNCING stage."""
        assert hasattr(JobStage, "SYNCING")
        assert JobStage.SYNCING.value == "syncing"

    def test_has_comparing_stage(self):
        """Should have COMPARING stage."""
        assert hasattr(JobStage, "COMPARING")
        assert JobStage.COMPARING.value == "comparing"

    def test_has_generating_srt_stage(self):
        """Should have GENERATING_SRT stage."""
        assert hasattr(JobStage, "GENERATING_SRT")
        assert JobStage.GENERATING_SRT.value == "generating_srt"

    def test_has_completed_stage(self):
        """Should have COMPLETED stage."""
        assert hasattr(JobStage, "COMPLETED")
        assert JobStage.COMPLETED.value == "completed"

    def test_has_failed_stage(self):
        """Should have FAILED stage."""
        assert hasattr(JobStage, "FAILED")
        assert JobStage.FAILED.value == "failed"


# =============================================================================
# Job Model Tests (without database)
# =============================================================================


class TestJobModel:
    """Tests for Job SQLAlchemy model."""

    def test_job_has_required_attributes(self):
        """Job model should have all required attributes."""
        required_attrs = [
            "id",
            "type",
            "status",
            "stage",
            "progress",
            "input_path",
            "output_path",
            "reference_path",
            "source_language",
            "target_language",
            "model_size",
            "compute_type",
            "options",
            "logs",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
        ]
        for attr in required_attrs:
            assert hasattr(Job, attr), f"Job model missing attribute: {attr}"

    def test_job_tablename(self):
        """Job model should have correct table name."""
        assert Job.__tablename__ == "jobs"

    def test_job_default_status_is_pending(self):
        """Default status should be PENDING."""
        # Check the column default
        status_column = Job.__table__.columns["status"]
        assert status_column.default.arg == JobStatus.PENDING

    def test_job_default_stage_is_queued(self):
        """Default stage should be QUEUED."""
        stage_column = Job.__table__.columns["stage"]
        assert stage_column.default.arg == JobStage.QUEUED

    def test_job_default_progress_is_zero(self):
        """Default progress should be 0."""
        progress_column = Job.__table__.columns["progress"]
        assert progress_column.default.arg == 0


# =============================================================================
# Settings Tests
# =============================================================================


class TestSettings:
    """Tests for application settings."""

    def test_settings_import(self):
        """Settings should be importable."""
        from homelab_subs.server.settings import Settings, get_settings

        assert Settings is not None
        assert get_settings is not None

    def test_settings_has_database_url(self):
        """Settings should have database_url."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        assert hasattr(settings, "database_url")
        assert isinstance(settings.database_url, str)

    def test_settings_has_redis_url(self):
        """Settings should have redis_url."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        assert hasattr(settings, "redis_url")
        assert isinstance(settings.redis_url, str)

    def test_settings_has_queue_names(self):
        """Settings should have queue names."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        assert hasattr(settings, "queue_default")
        assert hasattr(settings, "queue_high")
        assert hasattr(settings, "queue_gpu")

    def test_settings_cors_origins_list(self):
        """Settings should parse CORS origins into list."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        origins = settings.cors_origins_list
        assert isinstance(origins, list)
        assert len(origins) > 0

    def test_settings_all_queues(self):
        """Settings should provide all queue names."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        queues = settings.all_queues
        assert isinstance(queues, list)
        assert len(queues) == 3

    def test_settings_sync_database_url(self):
        """Settings should convert async URL to sync."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        sync_url = settings.sync_database_url
        assert not sync_url.startswith("postgresql+asyncpg://")

    def test_settings_async_database_url(self):
        """Settings should convert sync URL to async."""
        from homelab_subs.server.settings import Settings

        settings = Settings()
        # Override database_url to test conversion
        settings.database_url = "postgresql://localhost:5432/test"
        async_url = settings.async_database_url
        assert async_url.startswith("postgresql+asyncpg://")

    def test_get_settings_cached(self):
        """get_settings should return cached singleton."""
        from homelab_subs.server.settings import get_settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


# =============================================================================
# Queue Client Tests (Mocked)
# =============================================================================


class TestQueueClientMocked:
    """Tests for QueueClient with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mocked Redis connection."""
        with patch("homelab_subs.server.queue.Redis") as mock:
            mock_conn = MagicMock()
            mock.from_url.return_value = mock_conn
            yield mock_conn

    @pytest.fixture
    def mock_queue(self):
        """Create a mocked RQ Queue."""
        with patch("homelab_subs.server.queue.Queue") as mock:
            mock_q = MagicMock()
            mock.return_value = mock_q
            yield mock_q

    def test_queue_client_import(self):
        """QueueClient should be importable."""
        from homelab_subs.server.queue import QueueClient

        assert QueueClient is not None


# =============================================================================
# Repository Tests (Mocked)
# =============================================================================


class TestJobRepositoryMocked:
    """Tests for JobRepository with mocked database."""

    def test_repository_import(self):
        """JobRepository should be importable."""
        from homelab_subs.server.repository import JobRepository

        assert JobRepository is not None


# =============================================================================
# Job Service Tests (Mocked)
# =============================================================================


class TestServerJobServiceMocked:
    """Tests for ServerJobService with mocked dependencies."""

    def test_service_import(self):
        """ServerJobService should be importable."""
        from homelab_subs.server.job_service import ServerJobService

        assert ServerJobService is not None


# =============================================================================
# API Schema Tests
# =============================================================================


class TestAPISchemas:
    """Tests for API request/response schemas."""

    def test_job_create_request_schema(self):
        """JobCreateRequest should validate correctly."""
        from homelab_subs.server.api import JobCreateRequest

        # Valid request
        request = JobCreateRequest(
            type="transcribe",
            input_path="/path/to/video.mp4",
        )
        assert request.input_path == "/path/to/video.mp4"
        assert request.model_size == "base"  # default
        assert request.priority == "default"  # default

    def test_job_create_request_with_all_fields(self):
        """JobCreateRequest should accept all fields."""
        from homelab_subs.server.api import JobCreateRequest

        request = JobCreateRequest(
            type="translate",
            input_path="/path/to/subtitles.srt",
            output_path="/path/to/output.srt",
            source_language="en",
            target_language="es",
            model_size="large",
            compute_type="float16",
            priority="high",
            options={"custom_key": "custom_value"},
        )
        assert request.type == "translate"
        assert request.target_language == "es"
        assert request.options["custom_key"] == "custom_value"

    def test_job_response_schema(self):
        """JobResponse should have all required fields."""
        from homelab_subs.server.api import JobResponse

        # Verify field names
        field_names = set(JobResponse.model_fields.keys())
        required_fields = {
            "id",
            "type",
            "status",
            "stage",
            "progress",
            "input_path",
            "output_path",
            "reference_path",
            "source_language",
            "target_language",
            "model_size",
            "compute_type",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
        }
        assert required_fields.issubset(field_names)

    def test_job_statistics_response_schema(self):
        """JobStatisticsResponse should have all status counts."""
        from homelab_subs.server.api import JobStatisticsResponse

        stats = JobStatisticsResponse(
            total_jobs=100,
            pending=10,
            running=5,
            completed=80,
            failed=3,
            cancelled=2,
        )
        assert stats.total_jobs == 100
        assert (
            stats.pending
            + stats.running
            + stats.completed
            + stats.failed
            + stats.cancelled
            == 100
        )

    def test_health_response_schema(self):
        """HealthResponse should have service status fields."""
        from homelab_subs.server.api import HealthResponse

        health = HealthResponse(
            status="healthy",
            database="healthy",
            redis="healthy",
            version="0.3.0",
        )
        assert health.status == "healthy"


# =============================================================================
# Worker Tests (Unit)
# =============================================================================


class TestWorkerFunctions:
    """Tests for worker module functions."""

    def test_worker_import(self):
        """Worker module should be importable."""
        from homelab_subs.server.worker import process_job, run_worker

        assert process_job is not None
        assert run_worker is not None

    def test_job_cancelled_exception(self):
        """JobCancelledException should be defined."""
        from homelab_subs.server.worker import JobCancelledException

        exc = JobCancelledException("Test cancellation")
        assert str(exc) == "Test cancellation"

    def test_job_context_class(self):
        """JobContext class should be defined."""
        from homelab_subs.server.worker import JobContext

        assert JobContext is not None


# =============================================================================
# Integration Tests (Require dependencies)
# =============================================================================


@pytest.mark.skip(
    reason="Requires server dependencies: pip install homelab-subs[server]"
)
class TestServerIntegration:
    """Integration tests requiring server dependencies."""

    def test_fastapi_app_creation(self):
        """FastAPI app should be creatable."""
        from homelab_subs.server.api import create_app

        app = create_app()
        assert app is not None

    def test_health_endpoint(self):
        """Health endpoint should respond."""
        from fastapi.testclient import TestClient
        from homelab_subs.server.api import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
