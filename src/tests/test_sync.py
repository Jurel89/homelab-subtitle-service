# src/tests/test_sync.py

"""
Tests for the subtitle synchronization module.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from homelab_subs.core.sync import (
    SubtitleCue,
    SyncConfig,
    SyncResult,
    SubtitleSyncer,
    parse_srt_content,
    parse_srt_file,
    write_srt_from_cues,
    _timestamp_to_seconds,
    _seconds_to_timestamp,
    _normalize_text,
    _text_similarity,
)


class TestTimestampConversion:
    """Tests for timestamp conversion functions."""

    def test_timestamp_to_seconds_basic(self):
        """Test basic timestamp parsing."""
        assert _timestamp_to_seconds("00:00:00,000") == 0.0
        assert _timestamp_to_seconds("00:00:01,000") == 1.0
        assert _timestamp_to_seconds("00:01:00,000") == 60.0
        assert _timestamp_to_seconds("01:00:00,000") == 3600.0

    def test_timestamp_to_seconds_with_milliseconds(self):
        """Test timestamp parsing with milliseconds."""
        assert _timestamp_to_seconds("00:00:00,500") == 0.5
        assert _timestamp_to_seconds("00:00:01,250") == 1.25
        assert _timestamp_to_seconds("00:00:10,123") == 10.123

    def test_timestamp_to_seconds_with_dot(self):
        """Test timestamp parsing with dot separator."""
        assert _timestamp_to_seconds("00:00:01.500") == 1.5

    def test_timestamp_to_seconds_complex(self):
        """Test complex timestamp."""
        # 1 hour, 23 minutes, 45 seconds, 678 ms
        expected = 3600 + 23 * 60 + 45 + 0.678
        assert _timestamp_to_seconds("01:23:45,678") == expected

    def test_timestamp_to_seconds_invalid(self):
        """Test invalid timestamp raises error."""
        with pytest.raises(ValueError):
            _timestamp_to_seconds("invalid")
        with pytest.raises(ValueError):
            _timestamp_to_seconds("1:2:3,4")

    def test_seconds_to_timestamp_basic(self):
        """Test basic seconds to timestamp conversion."""
        assert _seconds_to_timestamp(0.0) == "00:00:00,000"
        assert _seconds_to_timestamp(1.0) == "00:00:01,000"
        assert _seconds_to_timestamp(60.0) == "00:01:00,000"
        assert _seconds_to_timestamp(3600.0) == "01:00:00,000"

    def test_seconds_to_timestamp_with_milliseconds(self):
        """Test seconds to timestamp with milliseconds."""
        assert _seconds_to_timestamp(1.5) == "00:00:01,500"
        assert _seconds_to_timestamp(10.123) == "00:00:10,123"

    def test_seconds_to_timestamp_negative(self):
        """Test negative seconds clamps to zero."""
        assert _seconds_to_timestamp(-5.0) == "00:00:00,000"

    def test_roundtrip(self):
        """Test timestamp roundtrip conversion."""
        timestamps = [
            "00:00:00,000",
            "00:01:23,456",
            "01:23:45,678",
            "12:34:56,789",
        ]
        for ts in timestamps:
            seconds = _timestamp_to_seconds(ts)
            result = _seconds_to_timestamp(seconds)
            assert result == ts


class TestTextNormalization:
    """Tests for text normalization and similarity."""

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        assert _normalize_text("Hello World") == "hello world"
        assert _normalize_text("  Multiple   Spaces  ") == "multiple spaces"

    def test_normalize_text_punctuation(self):
        """Test punctuation removal."""
        assert _normalize_text("Hello, World!") == "hello world"
        assert _normalize_text("What's up?") == "what's up"  # Keep apostrophe

    def test_normalize_text_html_tags(self):
        """Test HTML tag removal."""
        assert _normalize_text("<i>Italic</i>") == "italic"
        assert _normalize_text("<b>Bold</b> text") == "bold text"

    def test_normalize_text_ass_tags(self):
        """Test ASS-style tag removal."""
        assert _normalize_text("{\\an8}Subtitle") == "subtitle"

    def test_text_similarity_identical(self):
        """Test similarity of identical texts."""
        assert _text_similarity("Hello World", "Hello World") == 1.0

    def test_text_similarity_case_insensitive(self):
        """Test case-insensitive similarity."""
        assert _text_similarity("Hello World", "hello world") == 1.0

    def test_text_similarity_different(self):
        """Test similarity of different texts."""
        sim = _text_similarity("Hello", "Goodbye")
        assert 0.0 <= sim < 1.0

    def test_text_similarity_similar(self):
        """Test similarity of similar texts."""
        sim = _text_similarity("Hello World", "Hello Worlds")
        assert sim > 0.8

    def test_text_similarity_empty(self):
        """Test similarity with empty text."""
        assert _text_similarity("", "Hello") == 0.0
        assert _text_similarity("Hello", "") == 0.0


class TestSubtitleCue:
    """Tests for SubtitleCue dataclass."""

    def test_create_cue(self):
        """Test creating a subtitle cue."""
        cue = SubtitleCue(
            index=1,
            start_seconds=0.0,
            end_seconds=2.5,
            text="Hello World",
        )
        assert cue.index == 1
        assert cue.start_seconds == 0.0
        assert cue.end_seconds == 2.5
        assert cue.text == "Hello World"

    def test_cue_duration(self):
        """Test cue duration property."""
        cue = SubtitleCue(index=1, start_seconds=1.0, end_seconds=3.5, text="Test")
        assert cue.duration == 2.5


class TestSRTParsing:
    """Tests for SRT parsing functions."""

    def test_parse_srt_content_basic(self):
        """Test parsing basic SRT content."""
        content = """1
00:00:01,000 --> 00:00:03,000
Hello World

2
00:00:04,000 --> 00:00:06,000
Second subtitle
"""
        cues = parse_srt_content(content)
        assert len(cues) == 2
        assert cues[0].index == 1
        assert cues[0].start_seconds == 1.0
        assert cues[0].end_seconds == 3.0
        assert cues[0].text == "Hello World"
        assert cues[1].index == 2
        assert cues[1].text == "Second subtitle"

    def test_parse_srt_content_multiline(self):
        """Test parsing multiline subtitles."""
        content = """1
00:00:01,000 --> 00:00:03,000
Line one
Line two
"""
        cues = parse_srt_content(content)
        assert len(cues) == 1
        assert "Line one" in cues[0].text
        assert "Line two" in cues[0].text

    def test_parse_srt_content_windows_lineendings(self):
        """Test parsing with Windows line endings."""
        content = "1\r\n00:00:01,000 --> 00:00:03,000\r\nHello\r\n\r\n"
        cues = parse_srt_content(content)
        assert len(cues) == 1
        assert cues[0].text == "Hello"

    def test_parse_srt_content_dot_separator(self):
        """Test parsing with dot as millisecond separator."""
        content = """1
00:00:01.500 --> 00:00:03.500
Test
"""
        cues = parse_srt_content(content)
        assert len(cues) == 1
        assert cues[0].start_seconds == 1.5

    def test_parse_srt_file_not_found(self):
        """Test parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_srt_file(Path("/nonexistent/file.srt"))

    def test_parse_srt_file(self, tmp_path):
        """Test parsing SRT file from disk."""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("""1
00:00:01,000 --> 00:00:03,000
Test subtitle
""")
        cues = parse_srt_file(srt_file)
        assert len(cues) == 1
        assert cues[0].text == "Test subtitle"


class TestSRTWriting:
    """Tests for SRT writing functions."""

    def test_write_srt_from_cues(self, tmp_path):
        """Test writing cues to SRT file."""
        cues = [
            SubtitleCue(index=1, start_seconds=1.0, end_seconds=3.0, text="First"),
            SubtitleCue(index=2, start_seconds=4.0, end_seconds=6.0, text="Second"),
        ]
        output_path = tmp_path / "output.srt"
        
        result = write_srt_from_cues(cues, output_path)
        
        assert result == output_path
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "00:00:01,000 --> 00:00:03,000" in content
        assert "First" in content
        assert "Second" in content

    def test_write_srt_creates_parent_dirs(self, tmp_path):
        """Test that writing creates parent directories."""
        output_path = tmp_path / "deep" / "nested" / "output.srt"
        cues = [SubtitleCue(index=1, start_seconds=0, end_seconds=1, text="Test")]
        
        write_srt_from_cues(cues, output_path)
        
        assert output_path.exists()


class TestSyncConfig:
    """Tests for SyncConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SyncConfig()
        assert config.model_name == "small"
        assert config.device == "cpu"
        assert config.compute_type == "int8"
        assert config.language is None
        assert config.min_similarity == 0.6
        assert config.max_offset_seconds == 30.0
        assert config.interpolate_unmatched is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SyncConfig(
            model_name="medium",
            device="cuda",
            min_similarity=0.8,
        )
        assert config.model_name == "medium"
        assert config.device == "cuda"
        assert config.min_similarity == 0.8


class TestSubtitleSyncer:
    """Tests for SubtitleSyncer class."""

    def test_syncer_initialization(self):
        """Test syncer initialization."""
        syncer = SubtitleSyncer()
        assert syncer.config is not None
        assert syncer.config.model_name == "small"

    def test_syncer_custom_config(self):
        """Test syncer with custom config."""
        config = SyncConfig(model_name="tiny")
        syncer = SubtitleSyncer(config)
        assert syncer.config.model_name == "tiny"

    @patch("homelab_subs.core.sync.SubtitleSyncer._transcribe_video")
    def test_sync_subtitles_file_not_found(self, mock_transcribe, tmp_path):
        """Test sync with non-existent files."""
        syncer = SubtitleSyncer()
        
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            syncer.sync_subtitles(
                video_path=Path("/nonexistent/video.mp4"),
                srt_path=tmp_path / "test.srt",
            )

    @patch("homelab_subs.core.sync.SubtitleSyncer._transcribe_video")
    def test_sync_subtitles_srt_not_found(self, mock_transcribe, tmp_path):
        """Test sync with non-existent SRT file."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        
        syncer = SubtitleSyncer()
        
        with pytest.raises(FileNotFoundError, match="SRT file not found"):
            syncer.sync_subtitles(
                video_path=video_file,
                srt_path=Path("/nonexistent/subtitle.srt"),
            )


class TestSyncResultDataclass:
    """Tests for SyncResult dataclass."""

    def test_sync_result_creation(self):
        """Test creating sync result."""
        cues = [SubtitleCue(index=1, start_seconds=0, end_seconds=1, text="Test")]
        result = SyncResult(
            synced_cues=cues,
            matched_count=5,
            interpolated_count=2,
            unchanged_count=1,
            avg_offset_seconds=1.5,
            max_offset_seconds=3.0,
        )
        assert len(result.synced_cues) == 1
        assert result.matched_count == 5
        assert result.interpolated_count == 2
        assert result.unchanged_count == 1
        assert result.avg_offset_seconds == 1.5
        assert result.max_offset_seconds == 3.0


class TestIntegration:
    """Integration tests for subtitle syncing."""

    @patch("homelab_subs.core.sync.SubtitleSyncer._transcribe_video")
    def test_full_sync_workflow(self, mock_transcribe, tmp_path):
        """Test complete sync workflow with mocked transcription."""
        from homelab_subs.core.transcription import Segment
        
        # Create mock transcription segments
        mock_segments = [
            Segment(index=0, start=1.0, end=3.0, text="Hello World"),
            Segment(index=1, start=5.0, end=7.0, text="How are you"),
            Segment(index=2, start=10.0, end=12.0, text="Goodbye"),
        ]
        mock_transcribe.return_value = mock_segments
        
        # Create input SRT with offset timing
        srt_content = """1
00:00:02,000 --> 00:00:04,000
Hello World

2
00:00:06,000 --> 00:00:08,000
How are you

3
00:00:11,000 --> 00:00:13,000
Goodbye
"""
        srt_file = tmp_path / "input.srt"
        srt_file.write_text(srt_content)
        
        # Create dummy video file
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        
        # Run sync
        syncer = SubtitleSyncer(SyncConfig(min_similarity=0.8))
        result = syncer.sync_subtitles(
            video_path=video_file,
            srt_path=srt_file,
        )
        
        # Verify results
        assert result.matched_count >= 2  # Should match at least some
        assert len(result.synced_cues) == 3
        
        # Check output file was created
        output_file = srt_file.with_suffix(".synced.srt")
        assert output_file.exists()

    @patch("homelab_subs.core.sync.SubtitleSyncer._transcribe_video")
    def test_sync_with_custom_output_path(self, mock_transcribe, tmp_path):
        """Test sync with custom output path."""
        from homelab_subs.core.transcription import Segment
        
        mock_transcribe.return_value = [
            Segment(index=0, start=1.0, end=3.0, text="Test"),
        ]
        
        srt_file = tmp_path / "input.srt"
        srt_file.write_text("""1
00:00:01,000 --> 00:00:03,000
Test
""")
        
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        
        custom_output = tmp_path / "custom_output.srt"
        
        syncer = SubtitleSyncer()
        syncer.sync_subtitles(
            video_path=video_file,
            srt_path=srt_file,
            output_path=custom_output,
        )
        
        assert custom_output.exists()
