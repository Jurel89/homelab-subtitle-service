# src/homelab_subs/core/sync.py

"""
Subtitle synchronization module.

Synchronizes existing subtitle files with video audio by:
1. Transcribing the video audio with Whisper to get accurate timing
2. Aligning the existing subtitle text with the transcription
3. Adjusting timestamps to match the actual audio

This is useful when you have high-quality human-generated subtitles
that have timing offsets (common when subtitles are from a different
video release).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

from .audio import FFmpeg
from .transcription import Segment, Transcriber, TranscriberConfig
from ..logging_config import get_logger, log_stage

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[float, int], None]]


@dataclass
class SubtitleCue:
    """
    Represents a single subtitle cue with timing information.
    """
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    
    @property
    def duration(self) -> float:
        """Duration of the cue in seconds."""
        return self.end_seconds - self.start_seconds


@dataclass
class SyncConfig:
    """
    Configuration for subtitle synchronization.
    
    Attributes
    ----------
    model_name : str
        Whisper model to use for transcription reference.
    device : str
        Device to use for transcription ("cpu" or "cuda").
    compute_type : str
        Compute type for faster-whisper.
    language : Optional[str]
        Language code for transcription. None for auto-detect.
    min_similarity : float
        Minimum text similarity (0-1) required to match subtitles.
        Higher values require closer text matches.
    max_offset_seconds : float
        Maximum time offset allowed between original and synced timing.
        Subtitles with larger offsets will use interpolation.
    interpolate_unmatched : bool
        If True, interpolate timing for unmatched subtitles based on
        nearby matched subtitles. If False, keep original timing.
    """
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    min_similarity: float = 0.6
    max_offset_seconds: float = 30.0
    interpolate_unmatched: bool = True


@dataclass
class SyncResult:
    """
    Result of a synchronization operation.
    
    Attributes
    ----------
    synced_cues : list[SubtitleCue]
        The synchronized subtitle cues with adjusted timing.
    matched_count : int
        Number of subtitles that were matched with transcription.
    interpolated_count : int
        Number of subtitles with interpolated timing.
    unchanged_count : int
        Number of subtitles that kept original timing.
    avg_offset_seconds : float
        Average timing offset applied.
    max_offset_seconds : float
        Maximum timing offset applied.
    """
    synced_cues: list[SubtitleCue]
    matched_count: int
    interpolated_count: int
    unchanged_count: int
    avg_offset_seconds: float
    max_offset_seconds: float


def _timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert SRT timestamp (HH:MM:SS,mmm) to seconds.
    
    Parameters
    ----------
    timestamp : str
        Timestamp in format "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
    
    Returns
    -------
    float
        Time in seconds.
    """
    # Handle both comma and dot as millisecond separator
    timestamp = timestamp.replace(",", ".")
    
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, ms = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + ms / 1000


def _seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Parameters
    ----------
    seconds : float
        Time in seconds.
    
    Returns
    -------
    str
        Timestamp in SRT format.
    """
    if seconds < 0:
        seconds = 0.0
    
    total_ms = int(round(seconds * 1000))
    hours = total_ms // (3600 * 1000)
    total_ms %= 3600 * 1000
    minutes = total_ms // (60 * 1000)
    total_ms %= 60 * 1000
    secs = total_ms // 1000
    ms = total_ms % 1000
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation and extra whitespace.
    
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
    
    # Remove punctuation except apostrophes (important for contractions)
    text = re.sub(r"[^\w\s']", " ", text)
    
    # Normalize whitespace and lowercase
    text = " ".join(text.lower().split())
    
    return text


def _text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts.
    
    Parameters
    ----------
    text1 : str
        First text.
    text2 : str
        Second text.
    
    Returns
    -------
    float
        Similarity ratio between 0 and 1.
    """
    norm1 = _normalize_text(text1)
    norm2 = _normalize_text(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    return SequenceMatcher(None, norm1, norm2).ratio()


def parse_srt_file(srt_path: Path, encoding: str = "utf-8") -> list[SubtitleCue]:
    """
    Parse an SRT file into a list of SubtitleCue objects.
    
    Parameters
    ----------
    srt_path : Path
        Path to the SRT file.
    encoding : str
        File encoding (default: utf-8).
    
    Returns
    -------
    list[SubtitleCue]
        List of parsed subtitle cues.
    
    Raises
    ------
    FileNotFoundError
        If the SRT file doesn't exist.
    """
    srt_path = Path(srt_path)
    
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    
    content = srt_path.read_text(encoding=encoding)
    return parse_srt_content(content)


def parse_srt_content(content: str) -> list[SubtitleCue]:
    """
    Parse SRT content string into a list of SubtitleCue objects.
    
    Parameters
    ----------
    content : str
        SRT file content.
    
    Returns
    -------
    list[SubtitleCue]
        List of parsed subtitle cues.
    """
    cues: list[SubtitleCue] = []
    
    # Normalize line endings
    content = content.replace("\r\n", "\n")
    
    # Split into blocks (separated by blank lines)
    blocks = re.split(r"\n\n+", content.strip())
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        # Parse index
        try:
            index = int(lines[0].strip())
        except ValueError:
            logger.warning(f"Skipping invalid subtitle index: {lines[0]}")
            continue
        
        # Parse timestamp line
        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            logger.warning(f"Skipping invalid timestamp: {lines[1]}")
            continue
        
        try:
            start_seconds = _timestamp_to_seconds(timestamp_match.group(1))
            end_seconds = _timestamp_to_seconds(timestamp_match.group(2))
        except ValueError as e:
            logger.warning(f"Skipping cue with invalid timestamp: {e}")
            continue
        
        # Remaining lines are the subtitle text
        text = "\n".join(lines[2:]).strip()
        
        cues.append(SubtitleCue(
            index=index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            text=text,
        ))
    
    return cues


def write_srt_from_cues(
    cues: list[SubtitleCue],
    output_path: Path,
    encoding: str = "utf-8",
) -> Path:
    """
    Write subtitle cues to an SRT file.
    
    Parameters
    ----------
    cues : list[SubtitleCue]
        Subtitle cues to write.
    output_path : Path
        Output file path.
    encoding : str
        File encoding (default: utf-8).
    
    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    blocks: list[str] = []
    
    for i, cue in enumerate(cues, start=1):
        start_ts = _seconds_to_timestamp(cue.start_seconds)
        end_ts = _seconds_to_timestamp(cue.end_seconds)
        
        block = f"{i}\n{start_ts} --> {end_ts}\n{cue.text}\n"
        blocks.append(block)
    
    content = "\n".join(blocks)
    output_path.write_text(content, encoding=encoding)
    
    logger.info(f"Wrote {len(cues)} cues to {output_path}")
    return output_path


class SubtitleSyncer:
    """
    Synchronizes subtitle timing with video audio.
    
    Uses Whisper transcription as a timing reference to align
    existing subtitle text with the actual audio.
    
    Example
    -------
    >>> syncer = SubtitleSyncer(SyncConfig(model_name="small"))
    >>> result = syncer.sync_subtitles(
    ...     video_path=Path("movie.mkv"),
    ...     srt_path=Path("movie.srt"),
    ...     output_path=Path("movie.synced.srt"),
    ... )
    >>> print(f"Matched {result.matched_count} subtitles")
    """
    
    def __init__(self, config: Optional[SyncConfig] = None) -> None:
        self.config = config or SyncConfig()
        self._ffmpeg = FFmpeg()
        self._transcriber: Optional[Transcriber] = None
    
    def _ensure_transcriber(self) -> Transcriber:
        """Lazily initialize the transcriber."""
        if self._transcriber is None:
            transcriber_config = TranscriberConfig(
                model_name=self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._transcriber = Transcriber(config=transcriber_config)
        return self._transcriber
    
    def _transcribe_video(
        self,
        video_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> list[Segment]:
        """
        Transcribe video audio to get timing reference.
        
        Parameters
        ----------
        video_path : Path
            Path to the video file.
        progress_callback : ProgressCallback
            Optional progress callback.
        
        Returns
        -------
        list[Segment]
            Transcription segments with accurate timing.
        """
        context = {"video_file": str(video_path.name)}
        
        with log_stage(logger, "audio_extraction", **context):
            audio_path = self._ffmpeg.extract_audio_to_wav(video_path)
        
        transcriber = self._ensure_transcriber()
        
        with log_stage(logger, "transcription_for_sync", **context):
            segments = transcriber.transcribe_file(
                audio_path,
                language=self.config.language,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                progress_callback=progress_callback,
            )
        
        return segments
    
    def _find_best_match(
        self,
        cue: SubtitleCue,
        segments: list[Segment],
        time_window: float = 60.0,
    ) -> Optional[tuple[Segment, float]]:
        """
        Find the best matching transcription segment for a subtitle cue.
        
        Parameters
        ----------
        cue : SubtitleCue
            The subtitle cue to match.
        segments : list[Segment]
            Available transcription segments.
        time_window : float
            Time window (seconds) around cue timing to search.
        
        Returns
        -------
        Optional[tuple[Segment, float]]
            Best matching segment and similarity score, or None if no match.
        """
        best_match: Optional[Segment] = None
        best_score: float = 0.0
        
        cue_mid = (cue.start_seconds + cue.end_seconds) / 2
        
        for segment in segments:
            seg_mid = (segment.start + segment.end) / 2
            
            # Skip segments too far from the cue's expected time
            if abs(seg_mid - cue_mid) > time_window:
                continue
            
            similarity = _text_similarity(cue.text, segment.text)
            
            # Prefer matches closer in time when similarity is similar
            time_penalty = abs(seg_mid - cue_mid) / time_window * 0.1
            adjusted_score = similarity - time_penalty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_match = segment
        
        if best_match and best_score >= self.config.min_similarity - 0.1:
            actual_similarity = _text_similarity(cue.text, best_match.text)
            if actual_similarity >= self.config.min_similarity:
                return best_match, actual_similarity
        
        return None
    
    def _interpolate_timing(
        self,
        cue: SubtitleCue,
        matched_cues: list[tuple[SubtitleCue, SubtitleCue]],
    ) -> SubtitleCue:
        """
        Interpolate timing for an unmatched cue based on nearby matches.
        
        Parameters
        ----------
        cue : SubtitleCue
            The unmatched cue.
        matched_cues : list[tuple[SubtitleCue, SubtitleCue]]
            List of (original, synced) cue pairs for matched subtitles.
        
        Returns
        -------
        SubtitleCue
            Cue with interpolated timing.
        """
        if not matched_cues:
            return cue
        
        # Find the closest matched cue before and after
        before: Optional[tuple[SubtitleCue, SubtitleCue]] = None
        after: Optional[tuple[SubtitleCue, SubtitleCue]] = None
        
        for orig, synced in matched_cues:
            if orig.start_seconds <= cue.start_seconds:
                if before is None or orig.start_seconds > before[0].start_seconds:
                    before = (orig, synced)
            if orig.start_seconds >= cue.start_seconds:
                if after is None or orig.start_seconds < after[0].start_seconds:
                    after = (orig, synced)
        
        # Calculate offset based on nearby matches
        if before and after and before != after:
            # Interpolate between before and after
            orig_before, sync_before = before
            orig_after, sync_after = after
            
            # Linear interpolation
            total_orig_span = orig_after.start_seconds - orig_before.start_seconds
            if total_orig_span > 0:
                position = (cue.start_seconds - orig_before.start_seconds) / total_orig_span
                
                offset_before = sync_before.start_seconds - orig_before.start_seconds
                offset_after = sync_after.start_seconds - orig_after.start_seconds
                
                offset = offset_before + (offset_after - offset_before) * position
            else:
                offset = sync_before.start_seconds - orig_before.start_seconds
        elif before:
            offset = before[1].start_seconds - before[0].start_seconds
        elif after:
            offset = after[1].start_seconds - after[0].start_seconds
        else:
            return cue
        
        # Apply offset to cue
        new_start = max(0, cue.start_seconds + offset)
        new_end = max(new_start + 0.1, cue.end_seconds + offset)
        
        return SubtitleCue(
            index=cue.index,
            start_seconds=new_start,
            end_seconds=new_end,
            text=cue.text,
        )
    
    def sync_subtitles(
        self,
        video_path: Path,
        srt_path: Path,
        output_path: Optional[Path] = None,
        progress_callback: ProgressCallback = None,
    ) -> SyncResult:
        """
        Synchronize subtitle timing with video audio.
        
        Parameters
        ----------
        video_path : Path
            Path to the video file.
        srt_path : Path
            Path to the existing SRT subtitle file.
        output_path : Optional[Path]
            Output path for synced SRT. If None, uses <original>.synced.srt.
        progress_callback : ProgressCallback
            Optional progress callback for transcription phase.
        
        Returns
        -------
        SyncResult
            Synchronization results including the synced cues.
        
        Raises
        ------
        FileNotFoundError
            If video or SRT file doesn't exist.
        """
        video_path = Path(video_path)
        srt_path = Path(srt_path)
        
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not srt_path.is_file():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
        if output_path is None:
            output_path = srt_path.with_suffix(".synced.srt")
        output_path = Path(output_path)
        
        context = {
            "video_file": str(video_path.name),
            "srt_file": str(srt_path.name),
        }
        
        logger.info(
            f"Starting subtitle sync: {srt_path.name} -> {video_path.name}",
            extra=context,
        )
        
        # Parse existing subtitles
        with log_stage(logger, "parse_subtitles", **context):
            original_cues = parse_srt_file(srt_path)
        
        logger.info(f"Parsed {len(original_cues)} subtitle cues", extra=context)
        
        # Transcribe video for timing reference
        segments = self._transcribe_video(video_path, progress_callback)
        
        logger.info(
            f"Got {len(segments)} transcription segments for matching",
            extra=context,
        )
        
        # Match and sync subtitles
        with log_stage(logger, "align_subtitles", **context):
            synced_cues: list[SubtitleCue] = []
            matched_pairs: list[tuple[SubtitleCue, SubtitleCue]] = []
            offsets: list[float] = []
            
            matched_count = 0
            interpolated_count = 0
            unchanged_count = 0
            
            # First pass: find direct matches
            unmatched_cues: list[SubtitleCue] = []
            
            for cue in original_cues:
                match_result = self._find_best_match(cue, segments)
                
                if match_result:
                    segment, similarity = match_result
                    
                    # Calculate new timing based on segment
                    duration = cue.duration
                    new_start = segment.start
                    new_end = segment.end
                    
                    # If segment is much shorter, extend to preserve original duration
                    if (new_end - new_start) < duration * 0.5:
                        new_end = new_start + duration
                    
                    synced_cue = SubtitleCue(
                        index=cue.index,
                        start_seconds=new_start,
                        end_seconds=new_end,
                        text=cue.text,
                    )
                    
                    offset = new_start - cue.start_seconds
                    offsets.append(abs(offset))
                    
                    matched_pairs.append((cue, synced_cue))
                    synced_cues.append(synced_cue)
                    matched_count += 1
                    
                    logger.debug(
                        f"Matched cue {cue.index}: offset={offset:.2f}s, "
                        f"similarity={similarity:.2f}"
                    )
                else:
                    unmatched_cues.append(cue)
            
            # Second pass: interpolate unmatched cues
            for cue in unmatched_cues:
                if self.config.interpolate_unmatched and matched_pairs:
                    synced_cue = self._interpolate_timing(cue, matched_pairs)
                    offset = synced_cue.start_seconds - cue.start_seconds
                    
                    if abs(offset) > 0.1:  # Non-trivial adjustment
                        offsets.append(abs(offset))
                        interpolated_count += 1
                    else:
                        unchanged_count += 1
                    
                    synced_cues.append(synced_cue)
                else:
                    synced_cues.append(cue)
                    unchanged_count += 1
            
            # Sort by start time
            synced_cues.sort(key=lambda c: c.start_seconds)
        
        # Calculate statistics
        avg_offset = sum(offsets) / len(offsets) if offsets else 0.0
        max_offset = max(offsets) if offsets else 0.0
        
        logger.info(
            f"Sync complete: {matched_count} matched, {interpolated_count} interpolated, "
            f"{unchanged_count} unchanged. Avg offset: {avg_offset:.2f}s",
            extra=context,
        )
        
        # Write output file
        with log_stage(logger, "write_synced_srt", **context):
            write_srt_from_cues(synced_cues, output_path)
        
        return SyncResult(
            synced_cues=synced_cues,
            matched_count=matched_count,
            interpolated_count=interpolated_count,
            unchanged_count=unchanged_count,
            avg_offset_seconds=avg_offset,
            max_offset_seconds=max_offset,
        )


def sync_subtitle_file(
    video_path: Path,
    srt_path: Path,
    output_path: Optional[Path] = None,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    min_similarity: float = 0.6,
    progress_callback: ProgressCallback = None,
) -> SyncResult:
    """
    Convenience function to sync a subtitle file with video audio.
    
    Parameters
    ----------
    video_path : Path
        Path to the video file.
    srt_path : Path
        Path to the existing SRT subtitle file.
    output_path : Optional[Path]
        Output path for synced SRT. If None, uses <original>.synced.srt.
    model_name : str
        Whisper model for transcription reference.
    device : str
        Device for transcription ("cpu" or "cuda").
    compute_type : str
        Compute type for faster-whisper.
    language : Optional[str]
        Language code for transcription.
    min_similarity : float
        Minimum text similarity for matching (0-1).
    progress_callback : ProgressCallback
        Optional progress callback.
    
    Returns
    -------
    SyncResult
        Synchronization results.
    """
    config = SyncConfig(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        min_similarity=min_similarity,
    )
    
    syncer = SubtitleSyncer(config)
    return syncer.sync_subtitles(
        video_path=video_path,
        srt_path=srt_path,
        output_path=output_path,
        progress_callback=progress_callback,
    )
