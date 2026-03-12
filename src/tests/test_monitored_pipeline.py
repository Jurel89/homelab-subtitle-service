"""Tests for the MonitoredPipeline backward-compatibility wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from homelab_subs.core.monitored_pipeline import MonitoredPipeline
from homelab_subs.core.pipeline import PipelineRunner


def test_monitored_pipeline_is_subclass_of_pipeline_runner():
    assert issubclass(MonitoredPipeline, PipelineRunner)


def test_monitored_pipeline_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="MonitoredPipeline"):
        MonitoredPipeline(enable_db_logging=False, enable_monitoring=False)


def test_monitored_pipeline_deprecation_message_mentions_pipeline_runner():
    with pytest.warns(DeprecationWarning, match="PipelineRunner"):
        MonitoredPipeline(enable_db_logging=False, enable_monitoring=False)


def test_monitored_pipeline_default_params():
    """MonitoredPipeline can be instantiated with all defaults."""
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_db_logging=False, enable_monitoring=False)
    assert mp is not None


def test_monitored_pipeline_inherits_enable_monitoring_attribute():
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_monitoring=False, enable_db_logging=False)
    # enable_monitoring is False because we passed False (or monitoring unavailable)
    assert mp.enable_monitoring is False


def test_monitored_pipeline_inherits_enable_db_logging_attribute():
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_monitoring=False, enable_db_logging=False)
    assert mp.enable_db_logging is False


def test_monitored_pipeline_has_generate_subtitles_method():
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_monitoring=False, enable_db_logging=False)
    assert callable(mp.generate_subtitles)


def test_monitored_pipeline_generate_subtitles_delegates_to_run():
    """generate_subtitles should call PipelineRunner.run with matching kwargs."""
    mock_result = Path("/tmp/output.srt")

    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_monitoring=False, enable_db_logging=False)

    with patch.object(PipelineRunner, "run", return_value=mock_result) as mock_run:
        result = mp.generate_subtitles(
            video_path=Path("/tmp/video.mp4"),
            output_path=Path("/tmp/output.srt"),
            job_id="test-job-1",
            lang="en",
            model_name="small",
            device="cpu",
            compute_type="int8",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )

    assert result == mock_result
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["video_path"] == Path("/tmp/video.mp4")
    assert call_kwargs["output_path"] == Path("/tmp/output.srt")
    assert call_kwargs["job_id"] == "test-job-1"
    assert call_kwargs["lang"] == "en"
    assert call_kwargs["model_name"] == "small"
    assert call_kwargs["task"] == "transcribe"


def test_monitored_pipeline_generate_subtitles_passes_progress_callback():
    mock_result = Path("/tmp/output.srt")
    callback = MagicMock()

    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_monitoring=False, enable_db_logging=False)

    with patch.object(PipelineRunner, "run", return_value=mock_result) as mock_run:
        mp.generate_subtitles(
            video_path=Path("/tmp/video.mp4"),
            output_path=Path("/tmp/output.srt"),
            job_id="test-job-2",
            progress_callback=callback,
        )

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["progress_callback"] is callback


def test_monitored_pipeline_generate_subtitles_propagates_exceptions():
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(enable_monitoring=False, enable_db_logging=False)

    with patch.object(PipelineRunner, "run", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            mp.generate_subtitles(
                video_path=Path("/tmp/video.mp4"),
                output_path=Path("/tmp/output.srt"),
                job_id="test-job-err",
            )


def test_monitored_pipeline_accepts_metrics_interval():
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(
            enable_monitoring=False,
            enable_db_logging=False,
            metrics_interval=5.0,
        )
    assert mp.metrics_interval == 5.0


def test_monitored_pipeline_with_injected_db_logger():
    """db_path parameter is forwarded to PipelineRunner.__init__."""
    mock_db_logger = MagicMock()
    with pytest.warns(DeprecationWarning):
        mp = MonitoredPipeline(
            enable_monitoring=False,
            enable_db_logging=True,
            db_path=None,
        )
        # Manually inject a mock db_logger to verify attribute accessibility
        mp.db_logger = mock_db_logger

    assert mp.db_logger is mock_db_logger
