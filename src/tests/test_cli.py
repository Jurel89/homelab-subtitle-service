# tests/test_cli.py

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import MagicMock

from homelab_subs import cli
from homelab_subs.cli import _run_generate


def test_cli_help(capsys):
    # Call main() with --help and expect exit code 0 and some usage text.
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "usage" in captured.out.lower() or "subsvc" in captured.out.lower()


def test_cli_generate_calls_run_generate(monkeypatch, tmp_path, capsys):
    # Arrange: create a fake video path (we won't actually use ffmpeg here)
    video = tmp_path / "test.mkv"
    video.touch()

    called = {}

    def fake_run_generate(**kwargs):
        # Record the arguments so we can assert on them
        called.update(kwargs)
        return Path("/fake/output.srt")

    # Monkeypatch the internal helper
    monkeypatch.setattr(cli, "_run_generate", fake_run_generate)

    # Act
    exit_code = cli.main(
        [
            "generate",
            str(video),
            "--lang",
            "en",
            "--model",
            "small",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
        ]
    )

    # Assert
    assert exit_code == 0
    assert called["video_path"] == video
    assert called["lang"] == "en"
    assert called["model_name"] == "small"
    assert called["device"] == "cpu"
    assert called["compute_type"] == "int8"

    captured = capsys.readouterr()
    assert "Subtitles written to" in captured.out


def test_cli_batch_calls_run_batch(monkeypatch, tmp_path, capsys):
    # Arrange: fake config file
    cfg = tmp_path / "jobs.yaml"
    cfg.write_text("jobs: []\n")

    called = {}

    def fake_run_batch(config_path):
        called["config_path"] = config_path

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)

    # Act
    exit_code = cli.main(["batch", str(cfg)])

    # Assert
    assert exit_code == 0
    assert called["config_path"] == cfg

    captured = capsys.readouterr()
    # batch prints per-job logs, but with empty jobs, probably nothing special
    # Just verify it didn't blow up
    assert captured.err == ""


def test_run_generate_success(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    output_path = tmp_path / "output.srt"

    mock_service_cls = MagicMock()
    mock_service_instance = mock_service_cls.return_value
    mock_service_instance.generate_subtitles.return_value = output_path
    mock_service_instance.monitoring_available = True
    mock_service_instance.db_logging_available = True
    monkeypatch.setattr("homelab_subs.cli.JobService", mock_service_cls)

    class DummyPBar:
        def __init__(self):
            self.n = 0

        def refresh(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("homelab_subs.cli.tqdm", lambda *args, **kwargs: DummyPBar())

    result = _run_generate(
        video_path=video_path,
        output_path=output_path,
        lang="en",
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        job_id="test_job",
    )

    assert result == output_path
    mock_service_instance.generate_subtitles.assert_called_once()
    kwargs = mock_service_instance.generate_subtitles.call_args.kwargs
    assert kwargs["video_path"] == video_path
    assert kwargs["output_path"] == output_path
    assert callable(kwargs["progress_callback"])
