# src/homelab_subs/core/text_utils.py

"""
Shared text utilities for subtitle processing.

Canonical implementations of text normalization and timestamp formatting
used across the subtitle pipeline (sync, comparison, SRT generation).
"""

from __future__ import annotations

import re


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
