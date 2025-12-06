# src/tests/test_queue.py

"""
Tests for the QueueClient (Redis/RQ).

Uses mocked Redis/RQ for testing without real Redis.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# Check if RQ is available
try:
    from redis import Redis  # noqa: F401
    from rq import Queue  # noqa: F401
    from rq.job import Job as RQJob  # noqa: F401

    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not RQ_AVAILABLE,
    reason="RQ and Redis not installed",
)


class TestQueueClientInit:
    """Tests for QueueClient initialization."""

    def test_rq_available_flag(self):
        """RQ_AVAILABLE should be True when deps installed."""
        from homelab_subs.server.queue import RQ_AVAILABLE

        assert RQ_AVAILABLE is True

    @patch("homelab_subs.server.queue.Redis")
    @patch("homelab_subs.server.queue.Queue")
    @patch("homelab_subs.server.queue.get_settings")
    def test_queue_client_initialization(self, mock_settings, mock_queue, mock_redis):
        """QueueClient should initialize with settings."""
        from homelab_subs.server.queue import QueueClient

        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379/0"
        settings.queue_default = "default"
        settings.queue_high = "high"
        settings.queue_gpu = "gpu"
        mock_settings.return_value = settings

        mock_redis_instance = MagicMock()
        mock_redis.from_url.return_value = mock_redis_instance

        client = QueueClient(redis_url="redis://localhost:6379/0", settings=settings)
        assert client is not None
        mock_redis.from_url.assert_called_once_with("redis://localhost:6379/0")


@pytest.fixture
def mock_settings():
    """Create mock settings for queue client."""
    settings = MagicMock()
    settings.redis_url = "redis://localhost:6379/0"
    settings.queue_default = "default"
    settings.queue_high = "high"
    settings.queue_gpu = "gpu"
    return settings


@pytest.fixture
def mock_queue_client(mock_settings):
    """Create a QueueClient with mocked Redis."""
    from homelab_subs.server.queue import QueueClient

    with patch("homelab_subs.server.queue.Redis") as mock_redis:
        with patch("homelab_subs.server.queue.Queue"):
            with patch(
                "homelab_subs.server.queue.get_settings", return_value=mock_settings
            ):
                mock_redis_instance = MagicMock()
                mock_redis.from_url.return_value = mock_redis_instance

                client = QueueClient(
                    redis_url="redis://localhost:6379/0", settings=mock_settings
                )
                client._mock_redis = mock_redis_instance
                yield client


class TestQueueClientOperations:
    """Tests for QueueClient operations with mocked Redis."""

    def test_get_queue_default(self, mock_queue_client):
        """get_queue should return default queue."""
        queue = mock_queue_client.get_queue("default")
        assert queue is not None

    def test_get_queue_creates_on_demand(self, mock_queue_client):
        """get_queue should create unknown queues on demand."""
        with patch("homelab_subs.server.queue.Queue"):
            _ = mock_queue_client.get_queue("custom")
            # Should have created a new queue
            assert "custom" in mock_queue_client._queues

    def test_redis_property(self, mock_queue_client):
        """redis property should return Redis connection."""
        redis = mock_queue_client.redis
        assert redis is not None

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_rq_job_found(self, mock_rq_job, mock_queue_client):
        """get_rq_job should return job when found."""
        mock_job = MagicMock()
        mock_rq_job.fetch.return_value = mock_job

        result = mock_queue_client.get_rq_job("test-job-id")
        assert result == mock_job
        mock_rq_job.fetch.assert_called_once()

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_rq_job_not_found(self, mock_rq_job, mock_queue_client):
        """get_rq_job should return None when not found."""
        mock_rq_job.fetch.side_effect = Exception("Not found")

        result = mock_queue_client.get_rq_job("nonexistent-job")
        assert result is None

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_status(self, mock_rq_job, mock_queue_client):
        """get_status should return job status string."""
        mock_job = MagicMock()
        mock_job.get_status.return_value = "queued"
        mock_rq_job.fetch.return_value = mock_job

        status = mock_queue_client.get_status("test-job-id")
        assert status == "queued"

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_status_not_found(self, mock_rq_job, mock_queue_client):
        """get_status should return None when job not found."""
        mock_rq_job.fetch.side_effect = Exception("Not found")

        status = mock_queue_client.get_status("nonexistent-job")
        assert status is None

    @patch("homelab_subs.server.queue.RQJob")
    def test_cancel_success(self, mock_rq_job, mock_queue_client):
        """cancel should cancel a queued job."""
        mock_job = MagicMock()
        mock_rq_job.fetch.return_value = mock_job

        result = mock_queue_client.cancel("test-job-id")
        assert result is True
        mock_job.cancel.assert_called_once()

    @patch("homelab_subs.server.queue.RQJob")
    def test_cancel_not_found(self, mock_rq_job, mock_queue_client):
        """cancel should return False when job not found."""
        mock_rq_job.fetch.side_effect = Exception("Not found")

        result = mock_queue_client.cancel("nonexistent-job")
        assert result is False

    @patch("homelab_subs.server.queue.RQJob")
    def test_cancel_failure(self, mock_rq_job, mock_queue_client):
        """cancel should return False when cancellation fails."""
        mock_job = MagicMock()
        mock_job.cancel.side_effect = Exception("Cannot cancel running job")
        mock_rq_job.fetch.return_value = mock_job

        result = mock_queue_client.cancel("test-job-id")
        assert result is False

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_result(self, mock_rq_job, mock_queue_client):
        """get_result should return job result."""
        mock_job = MagicMock()
        mock_job.result = {"success": True}
        mock_rq_job.fetch.return_value = mock_job

        result = mock_queue_client.get_result("test-job-id")
        assert result == {"success": True}

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_result_not_found(self, mock_rq_job, mock_queue_client):
        """get_result should return None when job not found."""
        mock_rq_job.fetch.side_effect = Exception("Not found")

        result = mock_queue_client.get_result("nonexistent-job")
        assert result is None

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_exception(self, mock_rq_job, mock_queue_client):
        """get_exception should return exception info."""
        mock_job = MagicMock()
        mock_job.exc_info = "Traceback..."
        mock_rq_job.fetch.return_value = mock_job

        result = mock_queue_client.get_exception("test-job-id")
        assert result == "Traceback..."

    @patch("homelab_subs.server.queue.RQJob")
    def test_get_exception_not_found(self, mock_rq_job, mock_queue_client):
        """get_exception should return None when job not found."""
        mock_rq_job.fetch.side_effect = Exception("Not found")

        result = mock_queue_client.get_exception("nonexistent-job")
        assert result is None


class TestQueueStats:
    """Tests for queue statistics."""

    def test_get_queue_stats(self, mock_queue_client):
        """get_queue_stats should return stats for all queues."""
        # Mock queue lengths
        for queue in mock_queue_client._queues.values():
            queue.__len__ = MagicMock(return_value=5)
            queue.count = 5

        stats = mock_queue_client.get_queue_stats()
        assert isinstance(stats, dict)
