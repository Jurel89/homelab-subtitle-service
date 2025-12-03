from pathlib import Path
from unittest.mock import patch

import pytest

from homelab_subs.services.job_service import JobService


def test_job_service_generate_subtitles(tmp_path):
    video = tmp_path / "movie.mp4"
    output = tmp_path / "movie.srt"

    with patch("homelab_subs.services.job_service.MONITORING_AVAILABLE", True), patch(
        "homelab_subs.services.job_service.DB_LOGGING_AVAILABLE", True
    ), patch(
        "homelab_subs.services.job_service.PipelineRunner"
    ) as MockRunner:
        runner_instance = MockRunner.return_value
        runner_instance.run.return_value = output

        service = JobService(enable_monitoring=True, enable_db_logging=False)
        result = service.generate_subtitles(video_path=video, output_path=output, job_id="job42")

    assert result == output
    runner_instance.run.assert_called_once()


def test_job_service_history_helpers(tmp_path):
    db_path = tmp_path / "logs.db"

    with patch("homelab_subs.services.job_service.DB_LOGGING_AVAILABLE", True), patch(
        "homelab_subs.services.job_service.DatabaseLogger"
    ) as MockLogger:
        db_instance = MockLogger.return_value
        db_instance.get_recent_jobs.return_value = [object()]
        db_instance.get_statistics.return_value = {"total_jobs": 1, "status_counts": {}, "avg_duration_seconds": 0}
        db_instance.get_job.return_value = object()
        db_instance.get_job_metrics.return_value = []

        service = JobService(enable_monitoring=False, enable_db_logging=True, db_path=db_path)

        assert service.get_recent_jobs(limit=1)
        assert service.get_statistics()
        assert service.get_job_details("job42")


def test_job_service_requires_db_logger():
    service = JobService(enable_monitoring=False, enable_db_logging=False)

    with pytest.raises(RuntimeError):
        service.get_recent_jobs()
