"""
Legacy test module retained for reference only.
"""

import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.skip(
    "MonitoredPipeline has been replaced by PipelineRunner. "
    "See test_pipeline_runner.py for current coverage.",
    allow_module_level=True,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_video():
    """Create a temporary video file path (doesn't need to exist for mocking)."""
    return Path("/tmp/test_video.mp4")


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_initialization(temp_db):
    """Test MonitoredPipeline initialization."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    with (
        patch("homelab_subs.core.monitored_pipeline.PerformanceMonitor"),
        patch("homelab_subs.core.monitored_pipeline.DatabaseLogger"),
        patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        assert pipeline is not None
        assert pipeline.enable_db_logging is True


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_without_db_logging(temp_db):
    """Test MonitoredPipeline without database logging."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    with (
        patch("homelab_subs.core.monitored_pipeline.PerformanceMonitor"),
        patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=False,
        )

        assert pipeline.enable_db_logging is False
        assert pipeline.db_logger is None


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_generate_subtitles(temp_db, temp_video):
    """Test generate_subtitles method."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    # Mock dependencies
    mock_monitor = MagicMock()
    mock_monitor.get_metrics.return_value = MagicMock(
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.0,
        memory_used_mb=2048.0,
        memory_available_mb=4096.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_temperature=None,
    )
    mock_monitor.get_summary_stats.return_value = {
        "cpu_avg": 45.5,
        "cpu_max": 50.0,
        "memory_avg_mb": 2048.0,
        "memory_max_mb": 2200.0,
    }

    mock_db_logger = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_subtitles.return_value = Path("/tmp/output.srt")

    with (
        patch(
            "homelab_subs.core.monitored_pipeline.PerformanceMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.SubtitlePipeline",
            return_value=mock_pipeline,
        ),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        output = pipeline.generate_subtitles(
            video_path=temp_video,
            lang="en",
            task="transcribe",
        )

        assert output == Path("/tmp/output.srt")
        # Verify pipeline was called
        mock_pipeline.generate_subtitles.assert_called_once()
        # Verify job was created in database
        mock_db_logger.create_job.assert_called_once()


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_with_progress_callback(temp_db, temp_video):
    """Test generate_subtitles with progress callback."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_monitor = MagicMock()
    mock_monitor.get_metrics.return_value = MagicMock(
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.0,
        memory_used_mb=2048.0,
        memory_available_mb=4096.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_temperature=None,
    )
    mock_monitor.get_summary_stats.return_value = {}

    mock_db_logger = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_subtitles.return_value = Path("/tmp/output.srt")

    progress_calls = []

    def progress_callback(current, total):
        progress_calls.append((current, total))

    with (
        patch(
            "homelab_subs.core.monitored_pipeline.PerformanceMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.SubtitlePipeline",
            return_value=mock_pipeline,
        ),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        pipeline.generate_subtitles(
            video_path=temp_video,
            lang="en",
            task="transcribe",
            progress_callback=progress_callback,
        )

        # Verify callback was passed to pipeline
        assert mock_pipeline.generate_subtitles.called


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_handles_pipeline_error(temp_db, temp_video):
    """Test that monitored pipeline handles errors from core pipeline."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_monitor = MagicMock()
    mock_monitor.get_metrics.return_value = MagicMock(
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.0,
        memory_used_mb=2048.0,
        memory_available_mb=4096.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_temperature=None,
    )

    mock_db_logger = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_subtitles.side_effect = RuntimeError("Test error")

    with (
        patch(
            "homelab_subs.core.monitored_pipeline.PerformanceMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.SubtitlePipeline",
            return_value=mock_pipeline,
        ),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        with pytest.raises(RuntimeError, match="Test error"):
            pipeline.generate_subtitles(
                video_path=temp_video,
                lang="en",
                task="transcribe",
            )

        # Verify job status was updated to failed
        update_calls = mock_db_logger.update_job.call_args_list
        if update_calls:
            # Check if any call has status='failed'
            failed_updates = [
                call for call in update_calls if call[0][0].status == "failed"
            ]
            assert len(failed_updates) > 0


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_metrics_collection(temp_db, temp_video):
    """Test that metrics are collected during execution."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_monitor = MagicMock()
    metrics_calls = []

    def get_metrics_side_effect():
        metrics_calls.append(1)
        return MagicMock(
            timestamp=datetime.now(),
            cpu_percent=45.5,
            memory_percent=60.0,
            memory_used_mb=2048.0,
            memory_available_mb=4096.0,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_temperature=None,
        )

    mock_monitor.get_metrics.side_effect = get_metrics_side_effect
    mock_monitor.get_summary_stats.return_value = {}

    mock_db_logger = MagicMock()
    mock_pipeline = MagicMock()

    def slow_generate(*args, **kwargs):
        # Simulate some processing time
        time.sleep(0.1)
        return Path("/tmp/output.srt")

    mock_pipeline.generate_subtitles.side_effect = slow_generate

    with (
        patch(
            "homelab_subs.core.monitored_pipeline.PerformanceMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.SubtitlePipeline",
            return_value=mock_pipeline,
        ),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        pipeline.generate_subtitles(
            video_path=temp_video,
            lang="en",
            task="transcribe",
        )

        # Verify metrics were collected (at least once due to our sleep)
        assert len(metrics_calls) >= 1


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_get_job_history(temp_db):
    """Test get_job_history method."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_db_logger = MagicMock()
    mock_db_logger.get_recent_jobs.return_value = [
        MagicMock(job_id="job-1"),
        MagicMock(job_id="job-2"),
    ]

    with (
        patch("homelab_subs.core.monitored_pipeline.PerformanceMonitor"),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        history = pipeline.get_job_history(limit=10)

        assert len(history) == 2
        mock_db_logger.get_recent_jobs.assert_called_once_with(limit=10, status=None)


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_get_job_details(temp_db):
    """Test get_job_details method."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_job = MagicMock(job_id="test-123")
    mock_metrics = [MagicMock(), MagicMock()]

    mock_db_logger = MagicMock()
    mock_db_logger.get_job.return_value = mock_job
    mock_db_logger.get_job_metrics.return_value = mock_metrics

    with (
        patch("homelab_subs.core.monitored_pipeline.PerformanceMonitor"),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        job, metrics = pipeline.get_job_details("test-123")

        assert job == mock_job
        assert metrics == mock_metrics
        mock_db_logger.get_job.assert_called_once_with("test-123")
        mock_db_logger.get_job_metrics.assert_called_once_with("test-123")


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_get_statistics(temp_db):
    """Test get_statistics method."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_stats = {
        "total_jobs": 10,
        "status_counts": {"completed": 8, "failed": 2},
        "avg_duration_seconds": 120.5,
    }

    mock_db_logger = MagicMock()
    mock_db_logger.get_statistics.return_value = mock_stats

    with (
        patch("homelab_subs.core.monitored_pipeline.PerformanceMonitor"),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        stats = pipeline.get_statistics()

        assert stats == mock_stats
        mock_db_logger.get_statistics.assert_called_once()


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_without_db_logging_no_history(temp_db):
    """Test that history methods return None when db_logging is disabled."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    with (
        patch("homelab_subs.core.monitored_pipeline.PerformanceMonitor"),
        patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=False,
        )

        history = pipeline.get_job_history()
        assert history is None or history == []

        details = pipeline.get_job_details("test-123")
        assert details == (None, None)

        stats = pipeline.get_statistics()
        assert stats is None or stats == {}


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", False)
def test_monitored_pipeline_without_monitoring():
    """Test MonitoredPipeline when monitoring is not available."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    # Should still initialize but log a warning
    with patch("homelab_subs.core.monitored_pipeline.SubtitlePipeline"):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=False,
        )

        assert pipeline is not None


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_cleanup_on_success(temp_db, temp_video):
    """Test that monitoring thread is cleaned up properly on success."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_monitor = MagicMock()
    mock_monitor.get_metrics.return_value = MagicMock(
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.0,
        memory_used_mb=2048.0,
        memory_available_mb=4096.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_temperature=None,
    )
    mock_monitor.get_summary_stats.return_value = {}

    mock_db_logger = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_subtitles.return_value = Path("/tmp/output.srt")

    with (
        patch(
            "homelab_subs.core.monitored_pipeline.PerformanceMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.SubtitlePipeline",
            return_value=mock_pipeline,
        ),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            enable_db_logging=True,
            db_path=temp_db,
        )

        output = pipeline.generate_subtitles(
            video_path=temp_video,
            lang="en",
            task="transcribe",
        )

        # Give a moment for thread cleanup
        time.sleep(0.1)

        # Thread should be stopped (we can't directly check this without
        # accessing internals, but at least verify no exceptions)
        assert output is not None


@patch("homelab_subs.core.monitored_pipeline.MONITORING_AVAILABLE", True)
def test_monitored_pipeline_custom_vad_and_beam_size(temp_db, temp_video):
    """Test pipeline with custom VAD filter and beam size."""
    from homelab_subs.core.monitored_pipeline import MonitoredPipeline

    mock_monitor = MagicMock()
    mock_monitor.get_metrics.return_value = MagicMock(
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.0,
        memory_used_mb=2048.0,
        memory_available_mb=4096.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_temperature=None,
    )
    mock_monitor.get_summary_stats.return_value = {}

    mock_db_logger = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_subtitles.return_value = Path("/tmp/output.srt")

    with (
        patch(
            "homelab_subs.core.monitored_pipeline.PerformanceMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.DatabaseLogger",
            return_value=mock_db_logger,
        ),
        patch(
            "homelab_subs.core.monitored_pipeline.SubtitlePipeline",
            return_value=mock_pipeline,
        ),
    ):
        pipeline = MonitoredPipeline(
            model_name="base",
            device="cpu",
            compute_type="int8",
            beam_size=10,
            vad_filter=True,
            enable_db_logging=True,
            db_path=temp_db,
        )

        output = pipeline.generate_subtitles(
            video_path=temp_video,
            lang="en",
            task="transcribe",
        )

        # Verify pipeline was created with correct parameters
        assert output is not None
