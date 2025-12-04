# src/tests/test_comparison.py

"""
Unit tests for the subtitle comparison module.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from textwrap import dedent

from homelab_subs.core.comparison import (
    SubtitleComparator,
    ComparisonResult,
    TextMetrics,
    TimingMetrics,
    SegmentMetrics,
    SegmentComparison,
    compare_subtitle_files,
    format_comparison_report,
    _normalize_text,
    _get_words,
    _levenshtein_distance,
    calculate_wer,
    calculate_cer,
)
from homelab_subs.core.sync import SubtitleCue


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def reference_srt_content() -> str:
    """Reference (human) SRT content for testing."""
    return dedent("""\
        1
        00:00:01,000 --> 00:00:04,000
        Hello, how are you today?

        2
        00:00:05,000 --> 00:00:08,000
        I am doing very well, thank you.

        3
        00:00:10,000 --> 00:00:13,000
        What are you working on?
    """)


@pytest.fixture
def hypothesis_srt_content() -> str:
    """Hypothesis (machine) SRT content with some errors."""
    return dedent("""\
        1
        00:00:01,200 --> 00:00:04,100
        Hello, how are you today?

        2
        00:00:05,100 --> 00:00:08,200
        I'm doing very well, thanks.

        3
        00:00:10,100 --> 00:00:13,100
        What are you working on?
    """)


@pytest.fixture
def reference_srt_file(tmp_path: Path, reference_srt_content: str) -> Path:
    """Create a temporary reference SRT file."""
    file_path = tmp_path / "reference.srt"
    file_path.write_text(reference_srt_content)
    return file_path


@pytest.fixture
def hypothesis_srt_file(tmp_path: Path, hypothesis_srt_content: str) -> Path:
    """Create a temporary hypothesis SRT file."""
    file_path = tmp_path / "hypothesis.srt"
    file_path.write_text(hypothesis_srt_content)
    return file_path


# =============================================================================
# Test Text Normalization and Tokenization
# =============================================================================

class TestNormalization:
    """Tests for text normalization functions."""

    def test_normalize_lowercase(self):
        """Test that text is lowercased."""
        assert _normalize_text("Hello World") == "hello world"

    def test_normalize_removes_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        assert _normalize_text("hello   world") == "hello world"
        assert _normalize_text("  hello world  ") == "hello world"

    def test_normalize_removes_punctuation(self):
        """Test that punctuation is normalized."""
        result = _normalize_text("Hello, world!")
        # Punctuation should be removed or normalized
        assert "," not in result
        assert "!" not in result

    def test_normalize_newlines(self):
        """Test that newlines are handled."""
        assert _normalize_text("hello\nworld") == "hello world"


class TestTokenization:
    """Tests for tokenization function."""

    def test_tokenize_simple(self):
        """Test basic tokenization."""
        tokens = _get_words("hello world")
        assert tokens == ["hello", "world"]

    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation."""
        tokens = _get_words("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        assert _get_words("") == []

    def test_tokenize_whitespace(self):
        """Test tokenization of whitespace-only string."""
        assert _get_words("   ") == []


# =============================================================================
# Test WER and CER Calculation
# =============================================================================

class TestWERCalculation:
    """Tests for Word Error Rate calculation."""

    def test_wer_perfect_match(self):
        """Test WER when texts match perfectly."""
        wer, s, i, d = calculate_wer("hello world", "hello world")
        assert wer == 0.0
        assert s == 0
        assert i == 0
        assert d == 0

    def test_wer_complete_mismatch(self):
        """Test WER when texts are completely different."""
        wer, s, i, d = calculate_wer("hello world", "goodbye moon")
        assert wer == 1.0  # 100% error rate

    def test_wer_substitution(self):
        """Test WER with word substitution."""
        wer, s, i, d = calculate_wer("hello world", "hello there")
        assert s == 1  # 1 substitution
        assert i == 0
        assert d == 0
        assert wer == 0.5  # 1 error in 2 words

    def test_wer_insertion(self):
        """Test WER with word insertion."""
        wer, s, i, d = calculate_wer("hello world", "hello big world")
        assert i == 1  # 1 insertion

    def test_wer_deletion(self):
        """Test WER with word deletion."""
        wer, s, i, d = calculate_wer("hello big world", "hello world")
        assert d == 1  # 1 deletion

    def test_wer_empty_reference(self):
        """Test WER with empty reference."""
        wer, s, i, d = calculate_wer("", "hello world")
        assert wer == 1.0  # All insertions

    def test_wer_empty_hypothesis(self):
        """Test WER with empty hypothesis."""
        wer, s, i, d = calculate_wer("hello world", "")
        assert wer == 1.0  # All deletions


class TestCERCalculation:
    """Tests for Character Error Rate calculation."""

    def test_cer_perfect_match(self):
        """Test CER when texts match perfectly."""
        assert calculate_cer("hello", "hello") == 0.0

    def test_cer_complete_mismatch(self):
        """Test CER when texts are completely different."""
        cer = calculate_cer("hello", "world")
        assert cer > 0.0

    def test_cer_one_char_diff(self):
        """Test CER with one character difference."""
        cer = calculate_cer("hello", "hallo")
        assert cer == pytest.approx(0.2, abs=0.01)  # 1/5

    def test_cer_empty_reference(self):
        """Test CER with empty reference."""
        assert calculate_cer("", "hello") == 1.0

    def test_cer_empty_hypothesis(self):
        """Test CER with empty hypothesis."""
        assert calculate_cer("hello", "") == 1.0


class TestLevenshteinDistance:
    """Tests for Levenshtein distance calculation."""

    def test_levenshtein_identical(self):
        """Test distance between identical sequences."""
        dist, _, _, _ = _levenshtein_distance(["a", "b", "c"], ["a", "b", "c"])
        assert dist == 0

    def test_levenshtein_one_insertion(self):
        """Test distance with one insertion."""
        dist, _, _, _ = _levenshtein_distance(["a", "b"], ["a", "x", "b"])
        assert dist == 1

    def test_levenshtein_one_deletion(self):
        """Test distance with one deletion."""
        dist, _, _, _ = _levenshtein_distance(["a", "x", "b"], ["a", "b"])
        assert dist == 1

    def test_levenshtein_one_substitution(self):
        """Test distance with one substitution."""
        dist, _, _, _ = _levenshtein_distance(["a", "b", "c"], ["a", "x", "c"])
        assert dist == 1

    def test_levenshtein_empty_source(self):
        """Test distance from empty sequence."""
        dist, _, _, _ = _levenshtein_distance([], ["a", "b"])
        assert dist == 2

    def test_levenshtein_empty_target(self):
        """Test distance to empty sequence."""
        dist, _, _, _ = _levenshtein_distance(["a", "b"], [])
        assert dist == 2


# =============================================================================
# Test SubtitleCue Comparison
# =============================================================================

class TestSubtitleCueCreation:
    """Tests for creating SubtitleCue objects."""

    def test_subtitle_cue_creation(self):
        """Test basic SubtitleCue creation."""
        cue = SubtitleCue(
            index=1,
            start_seconds=1.0,
            end_seconds=4.0,
            text="Hello, world!"
        )
        assert cue.index == 1
        assert cue.start_seconds == 1.0
        assert cue.end_seconds == 4.0
        assert cue.text == "Hello, world!"

    def test_subtitle_cue_duration(self):
        """Test SubtitleCue duration calculation."""
        cue = SubtitleCue(index=1, start_seconds=1.0, end_seconds=4.5, text="Test")
        assert cue.duration == 3.5


# =============================================================================
# Test SubtitleComparator
# =============================================================================

class TestSubtitleComparator:
    """Tests for SubtitleComparator class."""

    def test_comparator_initialization(self):
        """Test comparator initialization with default settings."""
        comparator = SubtitleComparator()
        assert comparator.timing_threshold_ms == 500.0

    def test_comparator_custom_threshold(self):
        """Test comparator initialization with custom threshold."""
        comparator = SubtitleComparator(timing_threshold_ms=250.0)
        assert comparator.timing_threshold_ms == 250.0

    def test_compare_files(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test comparing two SRT files."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)

        assert isinstance(result, ComparisonResult)
        assert isinstance(result.text_metrics, TextMetrics)
        assert isinstance(result.timing_metrics, TimingMetrics)
        assert isinstance(result.segment_metrics, SegmentMetrics)
        assert result.reference_path == reference_srt_file
        assert result.hypothesis_path == hypothesis_srt_file

    def test_compare_files_overall_score(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test that overall score is reasonable."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)

        # Score should be between 0 and 100
        assert 0 <= result.overall_score <= 100
        # With similar subtitles, score should be relatively high
        assert result.overall_score > 50

    def test_compare_files_segment_count(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test segment counting."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)

        assert result.segment_metrics.reference_count == 3
        assert result.segment_metrics.hypothesis_count == 3

    def test_compare_content_identical(self):
        """Test comparing identical content."""
        content = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hello, world!
        """)
        comparator = SubtitleComparator()
        result = comparator.compare_content(content, content)

        assert result.text_metrics.word_error_rate == 0.0
        assert result.text_metrics.avg_similarity == 1.0
        assert result.overall_score > 95

    def test_compare_content_different(self):
        """Test comparing completely different content."""
        ref = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hello, world!
        """)
        hyp = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Goodbye, universe!
        """)
        comparator = SubtitleComparator()
        result = comparator.compare_content(ref, hyp)

        assert result.text_metrics.word_error_rate > 0.5
        assert result.text_metrics.avg_similarity < 0.5


class TestComparisonMetrics:
    """Tests for comparison metrics details."""

    def test_text_metrics_structure(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test TextMetrics structure."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)
        tm = result.text_metrics

        assert hasattr(tm, "word_error_rate")
        assert hasattr(tm, "character_error_rate")
        assert hasattr(tm, "avg_similarity")
        assert hasattr(tm, "exact_match_count")
        assert hasattr(tm, "total_words_reference")
        assert hasattr(tm, "total_words_hypothesis")
        assert hasattr(tm, "insertions")
        assert hasattr(tm, "deletions")
        assert hasattr(tm, "substitutions")

    def test_timing_metrics_structure(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test TimingMetrics structure."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)
        tim = result.timing_metrics

        assert hasattr(tim, "avg_start_offset_ms")
        assert hasattr(tim, "avg_end_offset_ms")
        assert hasattr(tim, "max_start_offset_ms")
        assert hasattr(tim, "max_end_offset_ms")
        assert hasattr(tim, "timing_accuracy_pct")
        assert hasattr(tim, "overlap_ratio")

    def test_timing_offset_calculation(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test that timing offsets are calculated correctly."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)

        # Hypothesis has 100-200ms offset from reference
        assert result.timing_metrics.avg_start_offset_ms > 0
        assert result.timing_metrics.avg_start_offset_ms < 300

    def test_segment_metrics_structure(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test SegmentMetrics structure."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)
        sm = result.segment_metrics

        assert hasattr(sm, "reference_count")
        assert hasattr(sm, "hypothesis_count")
        assert hasattr(sm, "matched_count")
        assert hasattr(sm, "unmatched_reference")
        assert hasattr(sm, "unmatched_hypothesis")
        assert hasattr(sm, "segmentation_similarity")

    def test_segment_comparisons_list(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test that segment comparisons are returned."""
        comparator = SubtitleComparator()
        result = comparator.compare_files(reference_srt_file, hypothesis_srt_file)

        assert len(result.segment_comparisons) > 0
        for comp in result.segment_comparisons:
            assert isinstance(comp, SegmentComparison)
            assert isinstance(comp.reference, SubtitleCue)
            assert isinstance(comp.hypothesis, SubtitleCue)
            assert 0 <= comp.text_similarity <= 1


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_compare_subtitle_files(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test the compare_subtitle_files convenience function."""
        result = compare_subtitle_files(reference_srt_file, hypothesis_srt_file)
        assert isinstance(result, ComparisonResult)

    def test_compare_subtitle_files_custom_threshold(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test compare_subtitle_files with custom timing threshold."""
        result = compare_subtitle_files(
            reference_srt_file,
            hypothesis_srt_file,
            timing_threshold_ms=250.0,
        )
        assert isinstance(result, ComparisonResult)


class TestReportFormatting:
    """Tests for report formatting functions."""

    def test_format_comparison_report(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test basic report formatting."""
        result = compare_subtitle_files(reference_srt_file, hypothesis_srt_file)
        report = format_comparison_report(result)

        assert isinstance(report, str)
        assert "SUBTITLE COMPARISON REPORT" in report
        assert "TEXT ACCURACY" in report
        assert "TIMING ACCURACY" in report
        assert "SEGMENT ALIGNMENT" in report
        assert "Word Error Rate" in report

    def test_format_comparison_report_detailed(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test detailed report formatting."""
        result = compare_subtitle_files(reference_srt_file, hypothesis_srt_file)
        report = format_comparison_report(result, detailed=True)

        assert "SEGMENT DETAILS" in report
        assert "Segment 1:" in report

    def test_format_comparison_report_includes_filenames(
        self,
        reference_srt_file: Path,
        hypothesis_srt_file: Path,
    ):
        """Test that report includes file names."""
        result = compare_subtitle_files(reference_srt_file, hypothesis_srt_file)
        report = format_comparison_report(result)

        assert "reference.srt" in report
        assert "hypothesis.srt" in report


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_compare_empty_files(self, tmp_path: Path):
        """Test comparing empty SRT files."""
        ref_path = tmp_path / "empty_ref.srt"
        hyp_path = tmp_path / "empty_hyp.srt"
        ref_path.write_text("")
        hyp_path.write_text("")

        comparator = SubtitleComparator()
        result = comparator.compare_files(ref_path, hyp_path)

        assert result.segment_metrics.reference_count == 0
        assert result.segment_metrics.hypothesis_count == 0

    def test_compare_single_segment(self, tmp_path: Path):
        """Test comparing single-segment SRT files."""
        content = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hello, world!
        """)
        ref_path = tmp_path / "single_ref.srt"
        hyp_path = tmp_path / "single_hyp.srt"
        ref_path.write_text(content)
        hyp_path.write_text(content)

        result = compare_subtitle_files(ref_path, hyp_path)
        assert result.segment_metrics.reference_count == 1
        assert result.text_metrics.word_error_rate == 0.0

    def test_compare_different_segment_counts(self, tmp_path: Path):
        """Test comparing files with different segment counts."""
        ref = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            First line.

            2
            00:00:05,000 --> 00:00:08,000
            Second line.
        """)
        hyp = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            First line.
        """)
        ref_path = tmp_path / "ref.srt"
        hyp_path = tmp_path / "hyp.srt"
        ref_path.write_text(ref)
        hyp_path.write_text(hyp)

        result = compare_subtitle_files(ref_path, hyp_path)
        assert result.segment_metrics.reference_count == 2
        assert result.segment_metrics.hypothesis_count == 1
        assert result.segment_metrics.unmatched_reference >= 1

    def test_compare_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised for missing files."""
        ref_path = tmp_path / "existing.srt"
        ref_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

        comparator = SubtitleComparator()
        with pytest.raises(FileNotFoundError):
            comparator.compare_files(ref_path, tmp_path / "nonexistent.srt")


# =============================================================================
# Test Multi-line Subtitles
# =============================================================================

class TestMultilineSubtitles:
    """Tests for handling multi-line subtitles."""

    def test_multiline_text_comparison(self, tmp_path: Path):
        """Test comparing subtitles with multi-line text."""
        ref = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            This is line one
            This is line two
        """)
        hyp = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            This is line one
            This is line two
        """)
        ref_path = tmp_path / "multiline_ref.srt"
        hyp_path = tmp_path / "multiline_hyp.srt"
        ref_path.write_text(ref)
        hyp_path.write_text(hyp)

        result = compare_subtitle_files(ref_path, hyp_path)
        assert result.text_metrics.word_error_rate == 0.0
        assert result.text_metrics.avg_similarity == 1.0


# =============================================================================
# Test Timing Accuracy Threshold
# =============================================================================

class TestTimingThreshold:
    """Tests for timing threshold functionality."""

    def test_timing_within_threshold(self, tmp_path: Path):
        """Test subtitles within timing threshold."""
        ref = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hello, world!
        """)
        hyp = dedent("""\
            1
            00:00:01,100 --> 00:00:04,100
            Hello, world!
        """)
        ref_path = tmp_path / "ref.srt"
        hyp_path = tmp_path / "hyp.srt"
        ref_path.write_text(ref)
        hyp_path.write_text(hyp)

        # With 500ms threshold, 100ms offset should be within threshold
        comparator = SubtitleComparator(timing_threshold_ms=500.0)
        result = comparator.compare_files(ref_path, hyp_path)
        assert result.timing_metrics.timing_accuracy_pct == 100.0

    def test_timing_outside_threshold(self, tmp_path: Path):
        """Test subtitles outside timing threshold."""
        ref = dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hello, world!
        """)
        hyp = dedent("""\
            1
            00:00:02,000 --> 00:00:05,000
            Hello, world!
        """)
        ref_path = tmp_path / "ref.srt"
        hyp_path = tmp_path / "hyp.srt"
        ref_path.write_text(ref)
        hyp_path.write_text(hyp)

        # With 500ms threshold, 1000ms offset should be outside threshold
        comparator = SubtitleComparator(timing_threshold_ms=500.0)
        result = comparator.compare_files(ref_path, hyp_path)
        assert result.timing_metrics.timing_accuracy_pct < 100.0
