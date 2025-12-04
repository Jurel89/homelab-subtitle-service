# src/homelab_subs/core/__init__.py

"""
Core functionality for subtitle generation, translation, synchronization, and comparison.
"""

from .audio import FFmpeg, FFmpegError
from .srt import segments_to_srt, write_srt_file
from .transcription import Segment, Transcriber, TranscriberConfig
from .sync import (
    SubtitleCue,
    SubtitleSyncer,
    SyncConfig,
    SyncResult,
    parse_srt_file,
    parse_srt_content,
    write_srt_from_cues,
    sync_subtitle_file,
)
from .comparison import (
    SubtitleComparator,
    ComparisonResult,
    TextMetrics,
    TimingMetrics,
    SegmentMetrics,
    SegmentComparison,
    compare_subtitle_files,
    format_comparison_report,
)

# Translation is optional (requires extra dependencies)
try:
    from .translation import (
        Translator,
        TranslatorConfig,
        SubtitleEntry,
        list_supported_languages,
        get_available_backends,
        TranslationBackend,
        NLLB_LANGUAGE_CODES,
        HELSINKI_LANGUAGE_PAIRS,
    )
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    # Define None placeholders when translation is unavailable
    Translator = None  # type: ignore[misc, assignment]
    TranslatorConfig = None  # type: ignore[misc, assignment]
    SubtitleEntry = None  # type: ignore[misc, assignment]
    list_supported_languages = None  # type: ignore[misc, assignment]
    get_available_backends = None  # type: ignore[misc, assignment]
    TranslationBackend = None  # type: ignore[misc, assignment]
    NLLB_LANGUAGE_CODES = None  # type: ignore[misc, assignment]
    HELSINKI_LANGUAGE_PAIRS = None  # type: ignore[misc, assignment]

__all__ = [
    # Audio
    "FFmpeg",
    "FFmpegError",
    # SRT
    "segments_to_srt",
    "write_srt_file",
    # Transcription
    "Segment",
    "Transcriber",
    "TranscriberConfig",
    # Sync
    "SubtitleCue",
    "SubtitleSyncer",
    "SyncConfig",
    "SyncResult",
    "parse_srt_file",
    "parse_srt_content",
    "write_srt_from_cues",
    "sync_subtitle_file",
    # Comparison
    "SubtitleComparator",
    "ComparisonResult",
    "TextMetrics",
    "TimingMetrics",
    "SegmentMetrics",
    "SegmentComparison",
    "compare_subtitle_files",
    "format_comparison_report",
    # Translation (optional)
    "TRANSLATION_AVAILABLE",
    "Translator",
    "TranslatorConfig",
    "SubtitleEntry",
    "list_supported_languages",
    "get_available_backends",
    "TranslationBackend",
    "NLLB_LANGUAGE_CODES",
    "HELSINKI_LANGUAGE_PAIRS",
]
