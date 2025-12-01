from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from faster_whisper import WhisperModel

from ..logging_config import get_logger

logger = get_logger(__name__)


TranscriptionTask = Literal["transcribe", "translate"]


@dataclass
class Segment:
    """
    One segment of transcribed audio.

    Times are in seconds, relative to the start of the audio file.
    """
    index: int
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass
class TranscriberConfig:
    """
    Configuration for the Whisper transcriber.

    Attributes
    ----------
    model_name:
        Name of the Whisper model, e.g. "tiny", "base", "small", "medium", "large-v2".
    device:
        Device to run on. Typically "cpu" or "cuda".
    compute_type:
        Quantization/precision type, e.g. "int8", "int8_float16", "float16", "float32".
        For CPU, "int8" or "int8_float16" is a good default.
    download_root:
        Optional directory to cache/download models. If None, the default location is used.
    """

    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    download_root: Optional[str] = None


class Transcriber:
    """
    High-level wrapper around faster-whisper's WhisperModel.

    This class is responsible for:
    - Loading the Whisper model lazily
    - Running transcription on audio files
    - Returning a list of Segment objects
    """

    def __init__(self, config: Optional[TranscriberConfig] = None) -> None:
        self.config = config or TranscriberConfig()
        self._model: Optional[WhisperModel] = None

    # ---------- internal helpers ----------

    def _ensure_model_loaded(self) -> WhisperModel:
        """
        Lazily load the Whisper model on first use.
        """
        if self._model is None:
            logger.info(
                f"Loading Whisper model: {self.config.model_name} "
                f"(device={self.config.device}, compute_type={self.config.compute_type})"
            )
            kwargs = {
                "device": self.config.device,
                "compute_type": self.config.compute_type,
            }
            if self.config.download_root is not None:
                kwargs["download_root"] = self.config.download_root

            self._model = WhisperModel(self.config.model_name, **kwargs)
            logger.info("Whisper model loaded successfully")

        return self._model

    # ---------- public API ----------

    def transcribe_file(
        self,
        audio_path: Path,
        language: Optional[str] = "en",
        task: TranscriptionTask = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> list[Segment]:
        """
        Transcribe an audio file into a list of segments.

        Parameters
        ----------
        audio_path : Path
            Path to the audio file (e.g. WAV extracted with ffmpeg).
        language : Optional[str]
            Language code for transcription (e.g. "en").
            If None, the model will try to detect the language automatically.
        task : {"transcribe", "translate"}
            "transcribe" keeps language as-is, "translate" outputs English.
        beam_size : int
            Beam size used during decoding (higher = better quality, slower).
        vad_filter : bool
            Whether to apply voice-activity-detection filtering to remove silence.

        Returns
        -------
        list[Segment]
            List of transcribed segments in chronological order.

        Raises
        ------
        FileNotFoundError
            If the audio file does not exist.
        RuntimeError
            If the underlying model call fails for some reason.
        """
        audio_path = Path(audio_path)

        if not audio_path.is_file():
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(
            f"Starting transcription of {audio_path.name} "
            f"(lang={language or 'auto'}, task={task}, beam_size={beam_size}, vad={vad_filter})"
        )

        model = self._ensure_model_loaded()

        # faster-whisper returns (segments_generator, info)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        logger.debug(
            f"Transcription info: language={info.language}, "
            f"language_probability={info.language_probability:.4f}, "
            f"duration={info.duration:.2f}s"
        )

        segments = list(self._segments_from_iterable(segments_iter))
        logger.info(f"Transcription complete: {len(segments)} segments generated")

        return segments

    # ---------- helpers for shaping output ----------

    def _segments_from_iterable(self, segments_iter: Iterable) -> Iterable[Segment]:
        """
        Convert faster-whisper segments into our Segment dataclass objects.
        """
        segment_count = 0
        for idx, seg in enumerate(segments_iter, start=1):
            # `seg` is a faster_whisper.transcribe.Segment object
            # with attributes: start, end, text, avg_logprob, no_speech_prob, etc.
            segment_count += 1
            if segment_count % 50 == 0:
                logger.debug(f"Processed {segment_count} segments...")
            
            yield Segment(
                index=idx,
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                avg_logprob=getattr(seg, "avg_logprob", None),
                no_speech_prob=getattr(seg, "no_speech_prob", None),
            )


# A convenient default instance that most of the codebase can use
default_transcriber = Transcriber()
