# src/homelab_subs/core/text_utils.py

"""
Shared text utilities for subtitle processing.

Canonical implementations of text normalization, timestamp formatting,
and SRT parsing used across the subtitle pipeline (sync, comparison,
SRT generation, translation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def normalize_text(text: str) -> str:
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
    text = re.sub(r"[\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f]", '"', text)  # Normalize quotes
    text = re.sub(r"[\u2013\u2014]", "-", text)  # Normalize dashes
    text = re.sub(r"\u2026", "...", text)  # Normalize ellipsis

    # Remove punctuation except apostrophes (important for contractions)
    text = re.sub(r"[^\w\s']", " ", text)

    # Normalize whitespace and lowercase
    text = " ".join(text.lower().split())

    return text


def format_timestamp(seconds: float) -> str:
    """
    Convert a time in seconds to SRT timestamp format: HH:MM:SS,mmm

    Parameters
    ----------
    seconds : float
        Time in seconds (can be fractional).

    Returns
    -------
    str
        Timestamp string in SRT format.
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


def parse_srt_content(content: str) -> list[SubtitleCue]:
    """
    Parse SRT content string into a list of SubtitleCue objects.

    This is the canonical SRT block-splitting implementation used across
    the subtitle pipeline.

    Parameters
    ----------
    content : str
        SRT file content.

    Returns
    -------
    list[SubtitleCue]
        List of parsed subtitle cues.
    """
    import logging

    _logger = logging.getLogger(__name__)

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
            _logger.warning(f"Skipping invalid subtitle index: {lines[0]}")
            continue

        # Parse timestamp line (accept both comma and dot as millisecond separator)
        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            _logger.warning(f"Skipping invalid timestamp: {lines[1]}")
            continue

        try:
            start_seconds = _timestamp_to_seconds(timestamp_match.group(1))
            end_seconds = _timestamp_to_seconds(timestamp_match.group(2))
        except ValueError as e:
            _logger.warning(f"Skipping cue with invalid timestamp: {e}")
            continue

        # Remaining lines are the subtitle text
        text = "\n".join(lines[2:]).strip()

        cues.append(
            SubtitleCue(
                index=index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
            )
        )

    return cues
