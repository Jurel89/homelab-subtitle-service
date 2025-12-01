import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from homelab_subs.core.pipeline import generate_subtitles_for_video
from homelab_subs.core.transcription import Segment

@pytest.fixture
def mock_ffmpeg():
    with patch("homelab_subs.core.pipeline.FFmpeg") as MockFFmpeg:
        instance = MockFFmpeg.return_value
        instance.extract_audio_to_wav.return_value = Path("/tmp/fake_audio.wav")
        yield instance

@pytest.fixture
def mock_transcriber():
    with patch("homelab_subs.core.pipeline.Transcriber") as MockTranscriber:
        instance = MockTranscriber.return_value
        instance.transcribe_file.return_value = [
            Segment(index=1, start=0.0, end=1.0, text="Hello"),
            Segment(index=2, start=1.0, end=2.0, text="World"),
        ]
        yield instance

@pytest.fixture
def mock_write_srt():
    with patch("homelab_subs.core.pipeline.write_srt_file") as mock_write:
        mock_write.return_value = Path("/tmp/output.srt")
        yield mock_write

def test_generate_subtitles_for_video(mock_ffmpeg, mock_transcriber, mock_write_srt, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    output_path = tmp_path / "output.srt"

    result = generate_subtitles_for_video(
        video_path=video_path,
        output_path=output_path,
        lang="en",
        model_name="tiny",
        device="cpu"
    )

    # Verify FFmpeg was called
    mock_ffmpeg.extract_audio_to_wav.assert_called_once_with(video_path)

    # Verify Transcriber was initialized and called
    # Note: We can't easily check init args of Transcriber class here because we mocked the class instance, 
    # but we can check the method call.
    mock_transcriber.transcribe_file.assert_called_once()
    call_args = mock_transcriber.transcribe_file.call_args
    assert call_args[0][0] == Path("/tmp/fake_audio.wav")
    assert call_args[1]["language"] == "en"

    # Verify write_srt_file was called
    mock_write_srt.assert_called_once()
    assert mock_write_srt.call_args[0][1] == output_path

    assert result == Path("/tmp/output.srt")
