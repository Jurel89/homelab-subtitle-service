# src/homelab_subs/core/comparison.py

"""
Subtitle comparison and accuracy measurement module.

Provides tools to compare human-generated (reference) subtitles with
machine-generated subtitles to measure transcription quality.

Key Metrics:
- Word Error Rate (WER): Standard ASR metric measuring insertions, deletions, substitutions
- Character Error Rate (CER): Character-level error rate
- Text Similarity: Fuzzy matching score between subtitle texts
- Timing Accuracy: How well the timing aligns between reference and hypothesis
- Segment Alignment: How well subtitle segments correspond to each other

This is useful for:
- Comparing different Whisper models (tiny vs small vs medium vs large)
- Evaluating different compute types (int8 vs float16)
- Understanding quality vs speed tradeoffs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .sync import SubtitleCue, parse_srt_file, parse_srt_content
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TimingMetrics:
    """
    Metrics related to subtitle timing accuracy.
    
    Attributes
    ----------
    avg_start_offset_ms : float
        Average offset between reference and hypothesis start times (ms).
    avg_end_offset_ms : float
        Average offset between reference and hypothesis end times (ms).
    max_start_offset_ms : float
        Maximum start time offset observed (ms).
    max_end_offset_ms : float
        Maximum end time offset observed (ms).
    timing_accuracy_pct : float
        Percentage of subtitles with timing within acceptable threshold.
    overlap_ratio : float
        Average overlap ratio between matched subtitle time ranges (0-1).
    """
    avg_start_offset_ms: float
    avg_end_offset_ms: float
    max_start_offset_ms: float
    max_end_offset_ms: float
    timing_accuracy_pct: float
    overlap_ratio: float


@dataclass
class TextMetrics:
    """
    Metrics related to text transcription accuracy.
    
    Attributes
    ----------
    word_error_rate : float
        Word Error Rate (WER) - lower is better. 0 = perfect, 1 = 100% errors.
    character_error_rate : float
        Character Error Rate (CER) - lower is better.
    avg_similarity : float
        Average text similarity score (0-1) - higher is better.
    exact_match_count : int
        Number of subtitles that matched exactly (after normalization).
    total_words_reference : int
        Total word count in reference subtitles.
    total_words_hypothesis : int
        Total word count in hypothesis subtitles.
    insertions : int
        Number of inserted words (in hypothesis but not in reference).
    deletions : int
        Number of deleted words (in reference but not in hypothesis).
    substitutions : int
        Number of substituted words (different between ref and hyp).
    """
    word_error_rate: float
    character_error_rate: float
    avg_similarity: float
    exact_match_count: int
    total_words_reference: int
    total_words_hypothesis: int
    insertions: int
    deletions: int
    substitutions: int


@dataclass
class SegmentMetrics:
    """
    Metrics related to segment alignment and coverage.
    
    Attributes
    ----------
    reference_count : int
        Number of segments in reference subtitles.
    hypothesis_count : int
        Number of segments in hypothesis subtitles.
    matched_count : int
        Number of segments that were successfully matched.
    unmatched_reference : int
        Reference segments with no corresponding hypothesis.
    unmatched_hypothesis : int
        Hypothesis segments with no corresponding reference.
    segmentation_similarity : float
        How similar the segmentation is between reference and hypothesis (0-1).
    """
    reference_count: int
    hypothesis_count: int
    matched_count: int
    unmatched_reference: int
    unmatched_hypothesis: int
    segmentation_similarity: float


@dataclass
class SegmentComparison:
    """
    Detailed comparison of a single matched segment pair.
    
    Attributes
    ----------
    reference : SubtitleCue
        The reference (human) subtitle.
    hypothesis : SubtitleCue
        The hypothesis (machine) subtitle.
    text_similarity : float
        Text similarity score (0-1).
    start_offset_ms : float
        Start time offset in milliseconds (hypothesis - reference).
    end_offset_ms : float
        End time offset in milliseconds.
    word_error_rate : float
        WER for this specific segment.
    """
    reference: SubtitleCue
    hypothesis: SubtitleCue
    text_similarity: float
    start_offset_ms: float
    end_offset_ms: float
    word_error_rate: float


@dataclass
class ComparisonResult:
    """
    Complete comparison results between reference and hypothesis subtitles.
    
    Attributes
    ----------
    text_metrics : TextMetrics
        Text accuracy metrics (WER, CER, similarity).
    timing_metrics : TimingMetrics
        Timing accuracy metrics.
    segment_metrics : SegmentMetrics
        Segment alignment metrics.
    segment_comparisons : list[SegmentComparison]
        Detailed per-segment comparison data.
    overall_score : float
        Combined quality score (0-100) for easy comparison.
    reference_path : Optional[Path]
        Path to reference SRT file (if from file).
    hypothesis_path : Optional[Path]
        Path to hypothesis SRT file (if from file).
    """
    text_metrics: TextMetrics
    timing_metrics: TimingMetrics
    segment_metrics: SegmentMetrics
    segment_comparisons: list[SegmentComparison] = field(default_factory=list)
    overall_score: float = 0.0
    reference_path: Optional[Path] = None
    hypothesis_path: Optional[Path] = None


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing formatting and extra whitespace.
    
    Parameters
    ----------
    text : str
        Original text.
    
    Returns
    -------
    str
        Normalized lowercase text.
    """
    # Remove common subtitle formatting tags
    text = re.sub(r"<[^>]+>", "", text)  # HTML-like tags
    text = re.sub(r"\{[^}]+\}", "", text)  # ASS-style tags
    text = re.sub(r"\[[^\]]+\]", "", text)  # Square bracket annotations
    
    # Remove speaker labels like "JOHN:" at start of lines
    text = re.sub(r"^[A-Z]+:\s*", "", text, flags=re.MULTILINE)
    
    # Normalize punctuation
    text = re.sub(r"[""''「」『』]", '"', text)  # Normalize quotes
    text = re.sub(r"[–—]", "-", text)  # Normalize dashes
    text = re.sub(r"…", "...", text)  # Normalize ellipsis
    
    # Remove punctuation except apostrophes (important for contractions)
    text = re.sub(r"[^\w\s']", " ", text)
    
    # Normalize whitespace and lowercase
    text = " ".join(text.lower().split())
    
    return text


def _get_words(text: str) -> list[str]:
    """
    Split normalized text into words.
    
    Parameters
    ----------
    text : str
        Text to split.
    
    Returns
    -------
    list[str]
        List of words.
    """
    normalized = _normalize_text(text)
    return normalized.split() if normalized else []


def _levenshtein_distance(s1: list[str], s2: list[str]) -> tuple[int, int, int, int]:
    """
    Calculate Levenshtein distance with edit operation counts.
    
    Parameters
    ----------
    s1 : list[str]
        Reference sequence (words or characters).
    s2 : list[str]
        Hypothesis sequence.
    
    Returns
    -------
    tuple[int, int, int, int]
        (distance, substitutions, insertions, deletions)
    """
    m, n = len(s1), len(s2)
    
    # DP table: (distance, subs, ins, dels)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = (i, 0, 0, i)  # All deletions
    for j in range(n + 1):
        dp[0][j] = (j, 0, j, 0)  # All insertions
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                
                # Insertion (word in hypothesis not in reference)
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])
                
                # Deletion (word in reference not in hypothesis)
                del_ = dp[i - 1][j]
                del_cost = (del_[0] + 1, del_[1], del_[2], del_[3] + 1)
                
                # Choose minimum
                dp[i][j] = min(sub_cost, ins_cost, del_cost, key=lambda x: x[0])
    
    return dp[m][n]


def calculate_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int]:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference word count
    
    Parameters
    ----------
    reference : str
        Reference (human) text.
    hypothesis : str
        Hypothesis (machine) text.
    
    Returns
    -------
    tuple[float, int, int, int]
        (wer, substitutions, insertions, deletions)
    """
    ref_words = _get_words(reference)
    hyp_words = _get_words(hypothesis)
    
    if not ref_words:
        return (1.0 if hyp_words else 0.0, 0, len(hyp_words), 0)
    
    distance, subs, ins, dels = _levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    
    return (min(wer, 1.0), subs, ins, dels)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Parameters
    ----------
    reference : str
        Reference text.
    hypothesis : str
        Hypothesis text.
    
    Returns
    -------
    float
        Character error rate (0-1).
    """
    ref_chars = list(_normalize_text(reference).replace(" ", ""))
    hyp_chars = list(_normalize_text(hypothesis).replace(" ", ""))
    
    if not ref_chars:
        return 1.0 if hyp_chars else 0.0
    
    distance, _, _, _ = _levenshtein_distance(ref_chars, hyp_chars)
    return min(distance / len(ref_chars), 1.0)


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity ratio.
    
    Parameters
    ----------
    text1 : str
        First text.
    text2 : str
        Second text.
    
    Returns
    -------
    float
        Similarity ratio (0-1).
    """
    norm1 = _normalize_text(text1)
    norm2 = _normalize_text(text2)
    
    if not norm1 and not norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0
    
    return SequenceMatcher(None, norm1, norm2).ratio()


def calculate_overlap_ratio(
    ref_start: float, ref_end: float,
    hyp_start: float, hyp_end: float,
) -> float:
    """
    Calculate the overlap ratio between two time ranges.
    
    Parameters
    ----------
    ref_start, ref_end : float
        Reference time range (seconds).
    hyp_start, hyp_end : float
        Hypothesis time range (seconds).
    
    Returns
    -------
    float
        Overlap ratio (0-1). 1 = perfect overlap, 0 = no overlap.
    """
    overlap_start = max(ref_start, hyp_start)
    overlap_end = min(ref_end, hyp_end)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    union_start = min(ref_start, hyp_start)
    union_end = max(ref_end, hyp_end)
    union_duration = union_end - union_start
    
    if union_duration <= 0:
        return 0.0
    
    return overlap_duration / union_duration


class SubtitleComparator:
    """
    Compare reference and hypothesis subtitles to measure accuracy.
    
    Example
    -------
    >>> comparator = SubtitleComparator()
    >>> result = comparator.compare_files(
    ...     reference_path=Path("human.srt"),
    ...     hypothesis_path=Path("whisper.srt"),
    ... )
    >>> print(f"WER: {result.text_metrics.word_error_rate:.2%}")
    >>> print(f"Overall Score: {result.overall_score:.1f}/100")
    """
    
    def __init__(
        self,
        timing_threshold_ms: float = 500.0,
        min_overlap_for_match: float = 0.3,
    ) -> None:
        """
        Initialize the comparator.
        
        Parameters
        ----------
        timing_threshold_ms : float
            Maximum timing offset (ms) to consider "accurate" timing.
        min_overlap_for_match : float
            Minimum time overlap ratio to consider segments matched.
        """
        self.timing_threshold_ms = timing_threshold_ms
        self.min_overlap_for_match = min_overlap_for_match
    
    def compare_files(
        self,
        reference_path: Path,
        hypothesis_path: Path,
        encoding: str = "utf-8",
    ) -> ComparisonResult:
        """
        Compare two SRT files.
        
        Parameters
        ----------
        reference_path : Path
            Path to reference (human) SRT file.
        hypothesis_path : Path
            Path to hypothesis (machine) SRT file.
        encoding : str
            File encoding.
        
        Returns
        -------
        ComparisonResult
            Comparison results with all metrics.
        """
        reference_cues = parse_srt_file(reference_path, encoding)
        hypothesis_cues = parse_srt_file(hypothesis_path, encoding)
        
        result = self.compare_cues(reference_cues, hypothesis_cues)
        result.reference_path = reference_path
        result.hypothesis_path = hypothesis_path
        
        return result
    
    def compare_content(
        self,
        reference_content: str,
        hypothesis_content: str,
    ) -> ComparisonResult:
        """
        Compare two SRT content strings.
        
        Parameters
        ----------
        reference_content : str
            Reference SRT content.
        hypothesis_content : str
            Hypothesis SRT content.
        
        Returns
        -------
        ComparisonResult
            Comparison results.
        """
        reference_cues = parse_srt_content(reference_content)
        hypothesis_cues = parse_srt_content(hypothesis_content)
        
        return self.compare_cues(reference_cues, hypothesis_cues)
    
    def compare_cues(
        self,
        reference_cues: list[SubtitleCue],
        hypothesis_cues: list[SubtitleCue],
    ) -> ComparisonResult:
        """
        Compare two lists of subtitle cues.
        
        Parameters
        ----------
        reference_cues : list[SubtitleCue]
            Reference subtitle cues.
        hypothesis_cues : list[SubtitleCue]
            Hypothesis subtitle cues.
        
        Returns
        -------
        ComparisonResult
            Complete comparison results.
        """
        # Match segments based on time overlap
        matches = self._match_segments(reference_cues, hypothesis_cues)
        
        # Calculate segment comparisons
        segment_comparisons = self._compute_segment_comparisons(matches)
        
        # Calculate aggregate metrics
        text_metrics = self._compute_text_metrics(reference_cues, hypothesis_cues, matches)
        timing_metrics = self._compute_timing_metrics(segment_comparisons)
        segment_metrics = self._compute_segment_metrics(
            reference_cues, hypothesis_cues, matches
        )
        
        # Calculate overall score
        overall_score = self._compute_overall_score(
            text_metrics, timing_metrics, segment_metrics
        )
        
        return ComparisonResult(
            text_metrics=text_metrics,
            timing_metrics=timing_metrics,
            segment_metrics=segment_metrics,
            segment_comparisons=segment_comparisons,
            overall_score=overall_score,
        )
    
    def _match_segments(
        self,
        reference_cues: list[SubtitleCue],
        hypothesis_cues: list[SubtitleCue],
    ) -> list[tuple[SubtitleCue, SubtitleCue]]:
        """
        Match reference and hypothesis segments based on time overlap.
        
        Uses a greedy matching algorithm that pairs segments with
        highest overlap ratios.
        """
        matches: list[tuple[SubtitleCue, SubtitleCue]] = []
        used_hypothesis: set[int] = set()
        
        for ref_cue in reference_cues:
            best_match: Optional[SubtitleCue] = None
            best_overlap: float = 0.0
            best_idx: int = -1
            
            for idx, hyp_cue in enumerate(hypothesis_cues):
                if idx in used_hypothesis:
                    continue
                
                overlap = calculate_overlap_ratio(
                    ref_cue.start_seconds, ref_cue.end_seconds,
                    hyp_cue.start_seconds, hyp_cue.end_seconds,
                )
                
                if overlap > best_overlap and overlap >= self.min_overlap_for_match:
                    best_overlap = overlap
                    best_match = hyp_cue
                    best_idx = idx
            
            if best_match is not None:
                matches.append((ref_cue, best_match))
                used_hypothesis.add(best_idx)
        
        return matches
    
    def _compute_segment_comparisons(
        self,
        matches: list[tuple[SubtitleCue, SubtitleCue]],
    ) -> list[SegmentComparison]:
        """Compute detailed comparison for each matched segment pair."""
        comparisons: list[SegmentComparison] = []
        
        for ref_cue, hyp_cue in matches:
            similarity = calculate_similarity(ref_cue.text, hyp_cue.text)
            wer, _, _, _ = calculate_wer(ref_cue.text, hyp_cue.text)
            
            start_offset_ms = (hyp_cue.start_seconds - ref_cue.start_seconds) * 1000
            end_offset_ms = (hyp_cue.end_seconds - ref_cue.end_seconds) * 1000
            
            comparisons.append(SegmentComparison(
                reference=ref_cue,
                hypothesis=hyp_cue,
                text_similarity=similarity,
                start_offset_ms=start_offset_ms,
                end_offset_ms=end_offset_ms,
                word_error_rate=wer,
            ))
        
        return comparisons
    
    def _compute_text_metrics(
        self,
        reference_cues: list[SubtitleCue],
        hypothesis_cues: list[SubtitleCue],
        matches: list[tuple[SubtitleCue, SubtitleCue]],
    ) -> TextMetrics:
        """Compute aggregate text accuracy metrics."""
        # Combine all text for overall WER/CER
        ref_text = " ".join(cue.text for cue in reference_cues)
        hyp_text = " ".join(cue.text for cue in hypothesis_cues)
        
        overall_wer, subs, ins, dels = calculate_wer(ref_text, hyp_text)
        overall_cer = calculate_cer(ref_text, hyp_text)
        
        # Calculate per-segment metrics
        similarities: list[float] = []
        exact_matches = 0
        
        for ref_cue, hyp_cue in matches:
            sim = calculate_similarity(ref_cue.text, hyp_cue.text)
            similarities.append(sim)
            
            if _normalize_text(ref_cue.text) == _normalize_text(hyp_cue.text):
                exact_matches += 1
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return TextMetrics(
            word_error_rate=overall_wer,
            character_error_rate=overall_cer,
            avg_similarity=avg_similarity,
            exact_match_count=exact_matches,
            total_words_reference=len(_get_words(ref_text)),
            total_words_hypothesis=len(_get_words(hyp_text)),
            insertions=ins,
            deletions=dels,
            substitutions=subs,
        )
    
    def _compute_timing_metrics(
        self,
        comparisons: list[SegmentComparison],
    ) -> TimingMetrics:
        """Compute timing accuracy metrics from segment comparisons."""
        if not comparisons:
            return TimingMetrics(
                avg_start_offset_ms=0.0,
                avg_end_offset_ms=0.0,
                max_start_offset_ms=0.0,
                max_end_offset_ms=0.0,
                timing_accuracy_pct=0.0,
                overlap_ratio=0.0,
            )
        
        start_offsets = [abs(c.start_offset_ms) for c in comparisons]
        end_offsets = [abs(c.end_offset_ms) for c in comparisons]
        
        # Count segments within timing threshold
        accurate_count = sum(
            1 for c in comparisons
            if abs(c.start_offset_ms) <= self.timing_threshold_ms
            and abs(c.end_offset_ms) <= self.timing_threshold_ms
        )
        
        # Calculate overlap ratios
        overlaps: list[float] = []
        for c in comparisons:
            overlap = calculate_overlap_ratio(
                c.reference.start_seconds, c.reference.end_seconds,
                c.hypothesis.start_seconds, c.hypothesis.end_seconds,
            )
            overlaps.append(overlap)
        
        return TimingMetrics(
            avg_start_offset_ms=sum(start_offsets) / len(start_offsets),
            avg_end_offset_ms=sum(end_offsets) / len(end_offsets),
            max_start_offset_ms=max(start_offsets),
            max_end_offset_ms=max(end_offsets),
            timing_accuracy_pct=accurate_count / len(comparisons) * 100,
            overlap_ratio=sum(overlaps) / len(overlaps),
        )
    
    def _compute_segment_metrics(
        self,
        reference_cues: list[SubtitleCue],
        hypothesis_cues: list[SubtitleCue],
        matches: list[tuple[SubtitleCue, SubtitleCue]],
    ) -> SegmentMetrics:
        """Compute segment alignment metrics."""
        ref_count = len(reference_cues)
        hyp_count = len(hypothesis_cues)
        matched_count = len(matches)
        
        # Segmentation similarity: how similar the segment counts are
        if ref_count == 0 and hyp_count == 0:
            seg_sim = 1.0
        elif ref_count == 0 or hyp_count == 0:
            seg_sim = 0.0
        else:
            seg_sim = min(ref_count, hyp_count) / max(ref_count, hyp_count)
        
        return SegmentMetrics(
            reference_count=ref_count,
            hypothesis_count=hyp_count,
            matched_count=matched_count,
            unmatched_reference=ref_count - matched_count,
            unmatched_hypothesis=hyp_count - matched_count,
            segmentation_similarity=seg_sim,
        )
    
    def _compute_overall_score(
        self,
        text_metrics: TextMetrics,
        timing_metrics: TimingMetrics,
        segment_metrics: SegmentMetrics,
    ) -> float:
        """
        Compute a single overall quality score (0-100).
        
        Weights:
        - Text accuracy (WER inverse): 50%
        - Timing accuracy: 25%
        - Segment alignment: 25%
        """
        # Text score: 100 - (WER * 100), clamped to 0-100
        text_score = max(0, 100 - (text_metrics.word_error_rate * 100))
        
        # Timing score: based on overlap ratio and timing accuracy
        timing_score = (
            timing_metrics.overlap_ratio * 50 +
            timing_metrics.timing_accuracy_pct * 0.5
        )
        
        # Segment score: based on match ratio and segmentation similarity
        if segment_metrics.reference_count > 0:
            match_ratio = segment_metrics.matched_count / segment_metrics.reference_count
        else:
            match_ratio = 1.0 if segment_metrics.hypothesis_count == 0 else 0.0
        
        segment_score = (
            match_ratio * 50 +
            segment_metrics.segmentation_similarity * 50
        )
        
        # Weighted combination
        overall = (
            text_score * 0.50 +
            timing_score * 0.25 +
            segment_score * 0.25
        )
        
        return min(100.0, max(0.0, overall))


def compare_subtitle_files(
    reference_path: Path,
    hypothesis_path: Path,
    timing_threshold_ms: float = 500.0,
) -> ComparisonResult:
    """
    Convenience function to compare two SRT files.
    
    Parameters
    ----------
    reference_path : Path
        Path to reference (human) SRT file.
    hypothesis_path : Path
        Path to hypothesis (machine) SRT file.
    timing_threshold_ms : float
        Threshold for timing accuracy measurement.
    
    Returns
    -------
    ComparisonResult
        Comparison results with all metrics.
    """
    comparator = SubtitleComparator(timing_threshold_ms=timing_threshold_ms)
    return comparator.compare_files(reference_path, hypothesis_path)


def format_comparison_report(result: ComparisonResult, detailed: bool = False) -> str:
    """
    Format comparison results as a human-readable report.
    
    Parameters
    ----------
    result : ComparisonResult
        Comparison results to format.
    detailed : bool
        If True, include per-segment details.
    
    Returns
    -------
    str
        Formatted report string.
    """
    lines: list[str] = []
    
    lines.append("=" * 70)
    lines.append("SUBTITLE COMPARISON REPORT")
    lines.append("=" * 70)
    
    if result.reference_path:
        lines.append(f"Reference:  {result.reference_path.name}")
    if result.hypothesis_path:
        lines.append(f"Hypothesis: {result.hypothesis_path.name}")
    
    lines.append("")
    lines.append(f"{'OVERALL SCORE:':<25} {result.overall_score:.1f}/100")
    lines.append("")
    
    # Text Metrics
    lines.append("-" * 70)
    lines.append("TEXT ACCURACY")
    lines.append("-" * 70)
    tm = result.text_metrics
    lines.append(f"{'Word Error Rate (WER):':<30} {tm.word_error_rate:.2%}")
    lines.append(f"{'Character Error Rate (CER):':<30} {tm.character_error_rate:.2%}")
    lines.append(f"{'Average Similarity:':<30} {tm.avg_similarity:.2%}")
    lines.append(f"{'Exact Matches:':<30} {tm.exact_match_count}")
    lines.append(f"{'Reference Words:':<30} {tm.total_words_reference}")
    lines.append(f"{'Hypothesis Words:':<30} {tm.total_words_hypothesis}")
    lines.append(f"{'Substitutions:':<30} {tm.substitutions}")
    lines.append(f"{'Insertions:':<30} {tm.insertions}")
    lines.append(f"{'Deletions:':<30} {tm.deletions}")
    lines.append("")
    
    # Timing Metrics
    lines.append("-" * 70)
    lines.append("TIMING ACCURACY")
    lines.append("-" * 70)
    tim = result.timing_metrics
    lines.append(f"{'Avg Start Offset:':<30} {tim.avg_start_offset_ms:.0f} ms")
    lines.append(f"{'Avg End Offset:':<30} {tim.avg_end_offset_ms:.0f} ms")
    lines.append(f"{'Max Start Offset:':<30} {tim.max_start_offset_ms:.0f} ms")
    lines.append(f"{'Max End Offset:':<30} {tim.max_end_offset_ms:.0f} ms")
    lines.append(f"{'Timing Accuracy:':<30} {tim.timing_accuracy_pct:.1f}%")
    lines.append(f"{'Average Overlap:':<30} {tim.overlap_ratio:.2%}")
    lines.append("")
    
    # Segment Metrics
    lines.append("-" * 70)
    lines.append("SEGMENT ALIGNMENT")
    lines.append("-" * 70)
    sm = result.segment_metrics
    lines.append(f"{'Reference Segments:':<30} {sm.reference_count}")
    lines.append(f"{'Hypothesis Segments:':<30} {sm.hypothesis_count}")
    lines.append(f"{'Matched Segments:':<30} {sm.matched_count}")
    lines.append(f"{'Unmatched (Reference):':<30} {sm.unmatched_reference}")
    lines.append(f"{'Unmatched (Hypothesis):':<30} {sm.unmatched_hypothesis}")
    lines.append(f"{'Segmentation Similarity:':<30} {sm.segmentation_similarity:.2%}")
    lines.append("")
    
    # Detailed per-segment comparison
    if detailed and result.segment_comparisons:
        lines.append("-" * 70)
        lines.append("SEGMENT DETAILS (showing first 10)")
        lines.append("-" * 70)
        
        for i, comp in enumerate(result.segment_comparisons[:10]):
            lines.append(f"\nSegment {i + 1}:")
            lines.append(f"  Reference: \"{comp.reference.text[:50]}{'...' if len(comp.reference.text) > 50 else ''}\"")
            lines.append(f"  Hypothesis: \"{comp.hypothesis.text[:50]}{'...' if len(comp.hypothesis.text) > 50 else ''}\"")
            lines.append(f"  Similarity: {comp.text_similarity:.2%}, WER: {comp.word_error_rate:.2%}")
            lines.append(f"  Time offset: start={comp.start_offset_ms:+.0f}ms, end={comp.end_offset_ms:+.0f}ms")
        
        if len(result.segment_comparisons) > 10:
            lines.append(f"\n... and {len(result.segment_comparisons) - 10} more segments")
        
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
