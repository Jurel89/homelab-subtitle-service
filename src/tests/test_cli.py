# tests/test_cli.py

from __future__ import annotations

from pathlib import Path

import pytest

from homelab_subs import cli


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

    def fake_run_generate(
        video_path,
        output_path,
        lang,
        model_name,
        device,
        compute_type,
        task,
        beam_size,
        vad_filter,
    ):
        # Record the arguments so we can assert on them
        called["video_path"] = video_path
        called["output_path"] = output_path
        called["lang"] = lang
        called["model_name"] = model_name
        called["device"] = device
        called["compute_type"] = compute_type
        called["task"] = task
        called["beam_size"] = beam_size
        called["vad_filter"] = vad_filter
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
