# src/tests/test_api.py

"""
Tests for the FastAPI REST API.

Uses FastAPI TestClient with mocked services.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed",
)


class TestAPISchemas:
    """Tests for Pydantic schemas."""

    def test_job_create_request_defaults(self):
        """JobCreateRequest should have default values."""
        from homelab_subs.server.api import JobCreateRequest
        from homelab_subs.server.models import JobType

        request = JobCreateRequest(input_path="/media/test.mkv")
        assert request.type == JobType.TRANSCRIBE
        assert request.model_size == "base"
        assert request.compute_type == "float16"
        assert request.priority == "default"

    def test_job_create_request_custom_values(self):
        """JobCreateRequest should accept custom values."""
        from homelab_subs.server.api import JobCreateRequest
        from homelab_subs.server.models import JobType

        request = JobCreateRequest(
            type=JobType.TRANSLATE,
            input_path="/media/test.mkv",
            output_path="/output/test.srt",
            source_language="en",
            target_language="es",
            model_size="large",
            priority="high",
        )
        assert request.type == JobType.TRANSLATE
        assert request.target_language == "es"
        assert request.model_size == "large"
        assert request.priority == "high"

    def test_login_request_validation(self):
        """LoginRequest should validate username and password."""
        from homelab_subs.server.api import LoginRequest

        request = LoginRequest(username="testuser", password="password123")
        assert request.username == "testuser"
        assert request.password == "password123"

    def test_register_request_validation(self):
        """RegisterRequest should validate fields."""
        from homelab_subs.server.api import RegisterRequest

        request = RegisterRequest(username="newuser", password="securepass123")
        assert request.username == "newuser"

    def test_settings_update_request(self):
        """SettingsUpdateRequest should be optional fields."""
        from homelab_subs.server.api import SettingsUpdateRequest

        request = SettingsUpdateRequest(default_model="large", worker_count=4)
        assert request.default_model == "large"
        assert request.worker_count == 4
        assert request.media_folders is None

    def test_file_item_schema(self):
        """FileItem should represent file/directory."""
        from homelab_subs.server.api import FileItem

        item = FileItem(
            name="video.mkv",
            path="/media/video.mkv",
            is_directory=False,
            size=1024,
            extension=".mkv",
        )
        assert item.name == "video.mkv"
        assert item.is_directory is False

    def test_health_response_schema(self):
        """HealthResponse should contain system status."""
        from homelab_subs.server.api import HealthResponse

        response = HealthResponse(
            status="healthy",
            database="healthy",
            redis="healthy",
            version="0.3.0",
        )
        assert response.status == "healthy"

    def test_error_response_schema(self):
        """ErrorResponse should contain error details."""
        from homelab_subs.server.api import ErrorResponse

        response = ErrorResponse(detail="Not found", error_code="JOB_NOT_FOUND")
        assert response.detail == "Not found"
        assert response.error_code == "JOB_NOT_FOUND"


class TestAppCreation:
    """Tests for FastAPI app creation."""

    def test_create_app_returns_fastapi_instance(self):
        """create_app should return a FastAPI instance."""
        from homelab_subs.server.api import create_app

        with patch("homelab_subs.server.api.get_settings") as mock_settings:
            settings = MagicMock()
            settings.cors_origins_list = ["http://localhost:3000"]
            mock_settings.return_value = settings

            app = create_app()
            assert isinstance(app, FastAPI)

    def test_app_has_correct_title(self):
        """App should have correct title."""
        from homelab_subs.server.api import app

        assert app.title == "Homelab Subtitle Service"

    def test_app_has_version(self):
        """App should have version."""
        from homelab_subs.server.api import app

        assert app.version == "0.3.0"


class TestDependencies:
    """Tests for FastAPI dependencies."""

    def test_get_settings_dep(self):
        """get_settings_dep should return settings."""
        from homelab_subs.server.api import get_settings_dep

        with patch("homelab_subs.server.api.get_settings") as mock_settings:
            settings = MagicMock()
            mock_settings.return_value = settings

            result = get_settings_dep()
            assert result == settings


@pytest.fixture
def mock_job_service():
    """Create a mock job service."""
    service = MagicMock()
    service.repository = MagicMock()
    service.queue_client = MagicMock()
    return service


@pytest.fixture
def test_client(mock_job_service):
    """Create a test client with mocked dependencies."""
    from homelab_subs.server.api import app, get_job_service

    # Override the dependency
    async def mock_get_job_service():
        return mock_job_service

    app.dependency_overrides[get_job_service] = mock_get_job_service

    with TestClient(app) as client:
        yield client

    # Clean up
    app.dependency_overrides.clear()


class TestHealthEndpoints:
    """Tests for health and status endpoints."""

    def test_health_endpoint_format(self):
        """Health endpoint should return correct format."""
        from homelab_subs.server.api import HealthResponse

        # Just verify the schema structure
        response = HealthResponse(
            status="healthy",
            database="healthy",
            redis="healthy",
            version="0.3.0",
        )
        assert response.status in ["healthy", "degraded", "unhealthy"]


class TestJobResponseHelpers:
    """Tests for job response helper functions."""

    def test_job_response_schema(self):
        """JobResponse schema should have all required fields."""
        from homelab_subs.server.api import JobResponse

        job_data = {
            "id": uuid.uuid4(),
            "type": "transcribe",
            "status": "pending",
            "stage": "queued",
            "progress": 0,
            "input_path": "/media/test.mkv",
            "output_path": None,
            "reference_path": None,
            "source_language": "en",
            "target_language": None,
            "model_size": "base",
            "compute_type": "float16",
            "error_message": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
        }

        response = JobResponse(**job_data)
        assert response.type == "transcribe"
        assert response.status == "pending"


class TestTokenResponse:
    """Tests for token response schema."""

    def test_token_response_schema(self):
        """TokenResponse should have token fields."""
        from homelab_subs.server.api import TokenResponse

        response = TokenResponse(
            access_token="access.token.here",
            refresh_token="refresh.token.here",
            token_type="bearer",
            expires_in=1800,
        )

        assert response.access_token == "access.token.here"
        assert response.token_type == "bearer"
        assert response.expires_in == 1800


class TestUserResponse:
    """Tests for user response schema."""

    def test_user_response_schema(self):
        """UserResponse should have user fields."""
        from homelab_subs.server.api import UserResponse

        response = UserResponse(
            id=uuid.uuid4(),
            username="testuser",
            is_admin=False,
            is_active=True,
            created_at=datetime.now(),
            last_login=None,
        )

        assert response.username == "testuser"
        assert response.is_admin is False


class TestSettingsSchemas:
    """Tests for settings schemas."""

    def test_settings_response_schema(self):
        """SettingsResponse should have all settings fields."""
        from homelab_subs.server.api import SettingsResponse

        response = SettingsResponse(
            media_folders=["/media"],
            default_model="small",
            default_device="cpu",
            default_compute_type="int8",
            default_language="en",
            default_translation_backend="nllb",
            worker_count=2,
            log_retention_days=30,
            job_retention_days=7,
            prefer_gpu=False,
            updated_at=datetime.now(),
        )

        assert response.default_model == "small"
        assert response.worker_count == 2


class TestFileBrowserSchemas:
    """Tests for file browser schemas."""

    def test_file_browser_response_schema(self):
        """FileBrowserResponse should list files."""
        from homelab_subs.server.api import FileBrowserResponse, FileItem

        items = [
            FileItem(
                name="video.mkv",
                path="/media/video.mkv",
                is_directory=False,
                size=1024,
            ),
            FileItem(
                name="subfolder",
                path="/media/subfolder",
                is_directory=True,
            ),
        ]

        response = FileBrowserResponse(
            current_path="/media",
            parent_path="/",
            items=items,
            total_items=2,
        )

        assert response.current_path == "/media"
        assert len(response.items) == 2
        assert response.items[0].is_directory is False
        assert response.items[1].is_directory is True


class TestJobStatisticsSchema:
    """Tests for job statistics schema."""

    def test_job_statistics_response_schema(self):
        """JobStatisticsResponse should have job counts."""
        from homelab_subs.server.api import JobStatisticsResponse

        response = JobStatisticsResponse(
            total_jobs=100,
            pending=10,
            running=5,
            completed=80,
            failed=3,
            cancelled=2,
        )

        assert response.total_jobs == 100
        assert response.pending + response.running + response.completed + response.failed + response.cancelled == 100


class TestQueueStatusSchema:
    """Tests for queue status schema."""

    def test_queue_status_response_schema(self):
        """QueueStatusResponse should have queue info."""
        from homelab_subs.server.api import QueueStatusResponse

        response = QueueStatusResponse(
            queues={"default": 5, "high": 2, "gpu": 0},
            workers=["worker-1", "worker-2"],
            total_jobs=7,
            failed_jobs=0,
        )

        assert response.total_jobs == 7
        assert len(response.workers) == 2


class TestSetupStatusSchema:
    """Tests for setup status schema."""

    def test_setup_status_response(self):
        """SetupStatusResponse should indicate setup state."""
        from homelab_subs.server.api import SetupStatusResponse

        response = SetupStatusResponse(
            setup_required=True,
            message="No admin user exists. Please register.",
        )

        assert response.setup_required is True


class TestChangePasswordSchema:
    """Tests for change password schema."""

    def test_change_password_request(self):
        """ChangePasswordRequest should have password fields."""
        from homelab_subs.server.api import ChangePasswordRequest

        request = ChangePasswordRequest(
            current_password="oldpassword123",
            new_password="newpassword456",
        )

        assert request.current_password == "oldpassword123"
        assert request.new_password == "newpassword456"


class TestRefreshRequestSchema:
    """Tests for token refresh schema."""

    def test_refresh_request(self):
        """RefreshRequest should have refresh token."""
        from homelab_subs.server.api import RefreshRequest

        request = RefreshRequest(refresh_token="refresh.token.here")
        assert request.refresh_token == "refresh.token.here"


class TestJobLogsSchema:
    """Tests for job logs schema."""

    def test_job_logs_response(self):
        """JobLogsResponse should have logs."""
        from homelab_subs.server.api import JobLogsResponse

        response = JobLogsResponse(
            job_id=uuid.uuid4(),
            logs="Processing started...\nExtracted audio...",
            status="running",
            stage="transcribing",
        )

        assert response.logs is not None
        assert "Processing" in response.logs
