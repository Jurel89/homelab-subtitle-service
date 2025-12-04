import subprocess
from pathlib import Path

import pytest

from homelab_subs.core.audio import FFmpeg, FFmpegError


def _create_test_video(tmp_path: Path) -> Path:
    """
    Use ffmpeg to create a tiny 1-second test video with audio.
    """
    output = tmp_path / "test.mp4"

    # This uses ffmpeg itself; if ffmpeg is missing, the test will fail,
    # which is acceptable because the whole project depends on ffmpeg.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:duration=1",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output


@pytest.mark.integration
def test_probe_and_extract(tmp_path):
    ff = FFmpeg()

    video = _create_test_video(tmp_path)
    info = ff.probe_video(video)

    assert info.has_audio is True
    assert info.has_video is True
    assert info.duration is not None
    assert info.duration > 0

    audio_path = ff.extract_audio_to_wav(video)
    assert audio_path.is_file()


def test_ffmpeg_missing(monkeypatch):
    """Test that FFmpegError is raised when binaries are missing."""
    # Mock shutil.which to return None (not found)
    monkeypatch.setattr("shutil.which", lambda x: None)

    ff = FFmpeg()

    # ensure_available should raise FFmpegError
    with pytest.raises(FFmpegError) as excinfo:
        ff.ensure_available()

    assert "not found in PATH" in str(excinfo.value)
