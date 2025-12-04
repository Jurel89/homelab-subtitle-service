# src/homelab_subs/core/__init__.py

"""
Core functionality for subtitle generation and translation.
"""

from .audio import FFmpeg, FFmpegError
from .srt import segments_to_srt, write_srt_file
from .transcription import Segment, Transcriber, TranscriberConfig

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
    # Translation (optional)
    "TRANSLATION_AVAILABLE",
]

# Add translation exports if available
if TRANSLATION_AVAILABLE:
    __all__.extend([
        "Translator",
        "TranslatorConfig",
        "SubtitleEntry",
        "list_supported_languages",
        "get_available_backends",
        "TranslationBackend",
        "NLLB_LANGUAGE_CODES",
        "HELSINKI_LANGUAGE_PAIRS",
    ])
