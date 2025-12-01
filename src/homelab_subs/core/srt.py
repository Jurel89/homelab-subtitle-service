from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .transcription import Segment


def _format_timestamp(seconds: float) -> str:
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


def segments_to_srt(segments: Iterable[Segment]) -> str:
    """
    Convert an iterable of Segment objects into SRT file content.

    Parameters
    ----------
    segments : Iterable[Segment]
        Segments in chronological order.

    Returns
    -------
    str
        Full SRT content as a single string, including trailing newline.
    """
    lines: list[str] = []

    for idx, seg in enumerate(segments, start=1):
        start_ts = _format_timestamp(seg.start)
        end_ts = _format_timestamp(seg.end)
        text = seg.text.strip()

        if not text:
            # Skip empty segments (can happen with VAD / no_speech filters)
            continue

        # SRT block:
        # <index>
        # HH:MM:SS,mmm --> HH:MM:SS,mmm
        # text
        #
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")  # blank line between cues

    # Ensure final newline
    return "\n".join(lines) + "\n"


def write_srt_file(
    segments: Iterable[Segment],
    output_path: Path,
    encoding: str = "utf-8",
) -> Path:
    """
    Write an SRT file from a sequence of segments.

    Parameters
    ----------
    segments : Iterable[Segment]
        Transcription segments to convert to subtitles.
    output_path : Path
        Destination path for the .srt file. Parent directories will be created.
    encoding : str
        Text encoding for the SRT file (default: "utf-8").

    Returns
    -------
    Path
        The output_path, for convenience.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    srt_content = segments_to_srt(segments)
    output_path.write_text(srt_content, encoding=encoding)

    return output_path
