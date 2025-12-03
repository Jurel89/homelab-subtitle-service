"""
Tests for database logging module.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


def test_database_logger_initialization(temp_db):
    """Test DatabaseLogger initialization creates tables."""
    from homelab_subs.core.db_logger import DatabaseLogger

    DatabaseLogger(temp_db)

    # Verify database file exists
    assert temp_db.exists()

    # Verify tables were created
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Check jobs table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
    )
    assert cursor.fetchone() is not None

    # Check metrics table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
    )
    assert cursor.fetchone() is not None

    conn.close()


def test_job_log_dataclass():
    """Test JobLog dataclass."""
    from homelab_subs.core.db_logger import JobLog

    job = JobLog(
        job_id="test-123",
        video_path="/path/to/video.mp4",
        output_path="/path/to/output.srt",
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="completed",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_seconds=120.5,
        cpu_avg=45.5,
        cpu_max=60.0,
        memory_avg_mb=2048.0,
        memory_max_mb=2500.0,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )

    assert job.job_id == "test-123"
    assert job.status == "completed"
    assert job.duration_seconds == 120.5
    assert job.gpu_avg is None


def test_metric_log_dataclass():
    """Test MetricLog dataclass."""
    from homelab_subs.core.db_logger import MetricLog

    metric = MetricLog(
        job_id="test-123",
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.2,
        memory_used_mb=2048.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=75.0,
        gpu_memory_used_mb=4096.0,
        gpu_temperature=65.0,
    )

    assert metric.job_id == "test-123"
    assert metric.cpu_percent == 45.5
    assert metric.gpu_utilization == 75.0


def test_create_job(temp_db):
    """Test creating a job record."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    job = JobLog(
        job_id="test-123",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="pending",
        started_at=datetime.now(),
        completed_at=None,
        duration_seconds=None,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )

    db_logger.create_job(job)

    # Verify job was created
    retrieved_job = db_logger.get_job("test-123")
    assert retrieved_job is not None
    assert retrieved_job.job_id == "test-123"
    assert retrieved_job.video_path == "/path/to/video.mp4"
    assert retrieved_job.status == "pending"


def test_update_job(temp_db):
    """Test updating a job record."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    # Create initial job
    job = JobLog(
        job_id="test-123",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="pending",
        started_at=datetime.now(),
        completed_at=None,
        duration_seconds=None,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )
    db_logger.create_job(job)

    # Update job
    job.status = "completed"
    job.completed_at = datetime.now()
    job.duration_seconds = 120.5
    job.cpu_avg = 45.5
    job.output_path = "/path/to/output.srt"

    db_logger.update_job(job)

    # Verify updates
    retrieved_job = db_logger.get_job("test-123")
    assert retrieved_job.status == "completed"
    assert retrieved_job.duration_seconds == 120.5
    assert retrieved_job.cpu_avg == 45.5
    assert retrieved_job.output_path == "/path/to/output.srt"


def test_add_metric(temp_db):
    """Test adding a metric record."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog, MetricLog

    db_logger = DatabaseLogger(temp_db)

    # Create a job first
    job = JobLog(
        job_id="test-123",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="running",
        started_at=datetime.now(),
        completed_at=None,
        duration_seconds=None,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )
    db_logger.create_job(job)

    # Add metric
    metric = MetricLog(
        job_id="test-123",
        timestamp=datetime.now(),
        cpu_percent=45.5,
        memory_percent=60.2,
        memory_used_mb=2048.0,
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        gpu_temperature=None,
    )

    db_logger.add_metric(metric)

    # Verify metric was added
    metrics = db_logger.get_job_metrics("test-123")
    assert len(metrics) == 1
    assert metrics[0].cpu_percent == 45.5
    assert metrics[0].memory_used_mb == 2048.0


def test_get_job_metrics(temp_db):
    """Test retrieving job metrics."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog, MetricLog

    db_logger = DatabaseLogger(temp_db)

    # Create job
    job = JobLog(
        job_id="test-123",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="running",
        started_at=datetime.now(),
        completed_at=None,
        duration_seconds=None,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )
    db_logger.create_job(job)

    # Add multiple metrics
    for i in range(5):
        metric = MetricLog(
            job_id="test-123",
            timestamp=datetime.now() + timedelta(seconds=i * 2),
            cpu_percent=40.0 + i * 5,
            memory_percent=60.0,
            memory_used_mb=2048.0,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_temperature=None,
        )
        db_logger.add_metric(metric)

    # Retrieve metrics
    metrics = db_logger.get_job_metrics("test-123")
    assert len(metrics) == 5
    assert metrics[0].cpu_percent == 40.0
    assert metrics[4].cpu_percent == 60.0


def test_get_recent_jobs(temp_db):
    """Test retrieving recent jobs."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    # Create multiple jobs
    for i in range(10):
        job = JobLog(
            job_id=f"test-{i}",
            video_path=f"/path/to/video{i}.mp4",
            output_path=None,
            language="en",
            model="base",
            task="transcribe",
            device="cpu",
            status="completed" if i % 2 == 0 else "failed",
            started_at=datetime.now() + timedelta(minutes=i),
            completed_at=None,
            duration_seconds=None,
            cpu_avg=None,
            cpu_max=None,
            memory_avg_mb=None,
            memory_max_mb=None,
            gpu_avg=None,
            gpu_max=None,
            error_message=None,
        )
        db_logger.create_job(job)

    # Get recent jobs
    recent_jobs = db_logger.get_recent_jobs(limit=5)
    assert len(recent_jobs) == 5

    # Get all jobs
    all_jobs = db_logger.get_recent_jobs(limit=20)
    assert len(all_jobs) == 10


def test_get_recent_jobs_with_status_filter(temp_db):
    """Test retrieving recent jobs filtered by status."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    # Create jobs with different statuses
    for i in range(10):
        job = JobLog(
            job_id=f"test-{i}",
            video_path=f"/path/to/video{i}.mp4",
            output_path=None,
            language="en",
            model="base",
            task="transcribe",
            device="cpu",
            status="completed" if i < 6 else "failed",
            started_at=datetime.now() + timedelta(minutes=i),
            completed_at=None,
            duration_seconds=None,
            cpu_avg=None,
            cpu_max=None,
            memory_avg_mb=None,
            memory_max_mb=None,
            gpu_avg=None,
            gpu_max=None,
            error_message=None,
        )
        db_logger.create_job(job)

    # Filter by completed
    completed_jobs = db_logger.get_recent_jobs(status="completed")
    assert len(completed_jobs) == 6
    assert all(job.status == "completed" for job in completed_jobs)

    # Filter by failed
    failed_jobs = db_logger.get_recent_jobs(status="failed")
    assert len(failed_jobs) == 4
    assert all(job.status == "failed" for job in failed_jobs)


def test_get_statistics(temp_db):
    """Test getting overall statistics."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    # Create jobs with various statuses and metrics
    for i in range(10):
        job = JobLog(
            job_id=f"test-{i}",
            video_path=f"/path/to/video{i}.mp4",
            output_path=None,
            language="en",
            model="base",
            task="transcribe",
            device="cpu",
            status=["completed", "failed", "running"][i % 3],
            started_at=datetime.now(),
            completed_at=datetime.now() + timedelta(seconds=120),
            duration_seconds=120.0 + i * 10,
            cpu_avg=40.0 + i * 2,
            cpu_max=50.0 + i * 2,
            memory_avg_mb=2000.0 + i * 100,
            memory_max_mb=2500.0 + i * 100,
            gpu_avg=70.0 if i < 5 else None,
            gpu_max=80.0 if i < 5 else None,
            error_message=None,
        )
        db_logger.create_job(job)

    # Get statistics
    stats = db_logger.get_statistics()

    assert stats["total_jobs"] == 10
    assert "status_counts" in stats
    assert stats["status_counts"]["completed"] > 0
    assert stats["status_counts"]["failed"] > 0
    assert "avg_duration_seconds" in stats
    assert "avg_cpu_percent" in stats
    assert "avg_memory_mb" in stats


def test_get_statistics_empty_db(temp_db):
    """Test statistics with empty database."""
    from homelab_subs.core.db_logger import DatabaseLogger

    db_logger = DatabaseLogger(temp_db)

    stats = db_logger.get_statistics()

    assert stats["total_jobs"] == 0
    assert stats["status_counts"] == {}
    assert stats["avg_duration_seconds"] is None


def test_get_nonexistent_job(temp_db):
    """Test retrieving a job that doesn't exist."""
    from homelab_subs.core.db_logger import DatabaseLogger

    db_logger = DatabaseLogger(temp_db)

    job = db_logger.get_job("nonexistent-id")
    assert job is None


def test_get_metrics_for_nonexistent_job(temp_db):
    """Test retrieving metrics for non-existent job."""
    from homelab_subs.core.db_logger import DatabaseLogger

    db_logger = DatabaseLogger(temp_db)

    metrics = db_logger.get_job_metrics("nonexistent-id")
    assert len(metrics) == 0


def test_database_logger_default_path():
    """Test DatabaseLogger with default path."""
    from homelab_subs.core.db_logger import DatabaseLogger

    # Create with default path
    db_logger = DatabaseLogger()

    # Should create database in home directory
    assert db_logger.db_path.parent.exists()

    # Cleanup
    if db_logger.db_path.exists():
        db_logger.db_path.unlink()
        # Try to remove the directory if empty
        try:
            db_logger.db_path.parent.rmdir()
        except OSError:
            pass  # Directory not empty or other error


def test_job_with_error_message(temp_db):
    """Test creating and retrieving job with error message."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    job = JobLog(
        job_id="test-error",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="failed",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_seconds=10.0,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message="Test error message",
    )

    db_logger.create_job(job)

    retrieved_job = db_logger.get_job("test-error")
    assert retrieved_job.status == "failed"
    assert retrieved_job.error_message == "Test error message"


def test_concurrent_metric_additions(temp_db):
    """Test adding multiple metrics rapidly."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog, MetricLog

    db_logger = DatabaseLogger(temp_db)

    # Create job
    job = JobLog(
        job_id="test-concurrent",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="running",
        started_at=datetime.now(),
        completed_at=None,
        duration_seconds=None,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )
    db_logger.create_job(job)

    # Add many metrics quickly
    for i in range(100):
        metric = MetricLog(
            job_id="test-concurrent",
            timestamp=datetime.now() + timedelta(microseconds=i),
            cpu_percent=40.0 + (i % 50),
            memory_percent=60.0,
            memory_used_mb=2048.0,
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_temperature=None,
        )
        db_logger.add_metric(metric)

    # Verify all metrics were added
    metrics = db_logger.get_job_metrics("test-concurrent")
    assert len(metrics) == 100


def test_update_nonexistent_job(temp_db):
    """Test updating a job that doesn't exist."""
    from homelab_subs.core.db_logger import DatabaseLogger, JobLog

    db_logger = DatabaseLogger(temp_db)

    job = JobLog(
        job_id="nonexistent",
        video_path="/path/to/video.mp4",
        output_path=None,
        language="en",
        model="base",
        task="transcribe",
        device="cpu",
        status="completed",
        started_at=datetime.now(),
        completed_at=None,
        duration_seconds=None,
        cpu_avg=None,
        cpu_max=None,
        memory_avg_mb=None,
        memory_max_mb=None,
        gpu_avg=None,
        gpu_max=None,
        error_message=None,
    )

    # Update should not raise an error (may be a no-op)
    db_logger.update_job(job)

    # Job should still not exist
    retrieved = db_logger.get_job("nonexistent")
    assert retrieved is None
