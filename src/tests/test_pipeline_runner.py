from pathlib import Path
from unittest.mock import patch

import pytest

from homelab_subs.core.pipeline import PipelineRunner


@patch(
    "homelab_subs.core.pipeline._execute_pipeline", return_value=Path("/tmp/out.srt")
)
@patch("homelab_subs.core.pipeline.DatabaseLogger")
@patch("homelab_subs.core.pipeline.PerformanceMonitor")
def test_pipeline_runner_records_success(
    mock_monitor, mock_db_logger, mock_execute, tmp_path
):
    video = tmp_path / "video.mp4"
    output = tmp_path / "subs.srt"
    video.touch()

    with (
        patch("homelab_subs.core.pipeline.MONITORING_AVAILABLE", True),
        patch("homelab_subs.core.pipeline.DB_LOGGING_AVAILABLE", True),
        patch.object(PipelineRunner, "_start_metrics_collection"),
        patch.object(PipelineRunner, "_stop_metrics_collection"),
        patch.object(
            PipelineRunner, "_get_performance_summary", return_value={"cpu_avg": 42.0}
        ) as summary_mock,
    ):
        runner = PipelineRunner(
            enable_monitoring=True, enable_db_logging=True, db_path=tmp_path / "logs.db"
        )
        result = runner.run(
            video_path=video,
            output_path=output,
            job_id="job123",
            lang="en",
            model_name="tiny",
            device="cpu",
        )

    assert result == Path("/tmp/out.srt")
    mock_execute.assert_called_once()
    mock_db_logger.return_value.create_job.assert_called_once()
    mock_db_logger.return_value.update_job.assert_called_once()
    summary_mock.assert_called_once()


def test_pipeline_runner_updates_failure(tmp_path):
    video = tmp_path / "video.mp4"
    output = tmp_path / "subs.srt"
    video.touch()

    with (
        patch(
            "homelab_subs.core.pipeline._execute_pipeline",
            side_effect=RuntimeError("boom"),
        ),
        patch("homelab_subs.core.pipeline.DatabaseLogger") as mock_db_logger,
        patch("homelab_subs.core.pipeline.DB_LOGGING_AVAILABLE", True),
    ):
        runner = PipelineRunner(
            enable_monitoring=False,
            enable_db_logging=True,
            db_path=tmp_path / "logs.db",
        )

        with pytest.raises(RuntimeError):
            runner.run(
                video_path=video,
                output_path=output,
                job_id="job999",
            )

    mock_db_logger.return_value.update_job.assert_called_once()
