# src/homelab_subs/core/pipeline.py

from __future__ import annotations

from pathlib import Path

from .audio import FFmpeg
from .transcription import Transcriber, TranscriberConfig
from .srt import write_srt_file
from ..logging_config import get_logger, log_stage

logger = get_logger(__name__)


def generate_subtitles_for_video(
    video_path: Path,
    output_path: Path,
    lang: str = "en",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    task: str = "transcribe",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Path:
    """
    High-level pipeline: video -> audio -> transcription -> SRT.
    """
    logger.info(f"Starting pipeline for {video_path.name}")

    context = {"video_file": str(video_path.name)}

    ff = FFmpeg()

    with log_stage(logger, "audio_extraction", **context):
        audio_path = ff.extract_audio_to_wav(video_path)

    config = TranscriberConfig(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    transcriber = Transcriber(config=config)

    with log_stage(logger, "transcription", **context):
        segments = transcriber.transcribe_file(
            audio_path,
            language=lang,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

    with log_stage(logger, "srt_generation", **context):
        result = write_srt_file(segments, output_path)

    logger.info(f"Pipeline complete: {result}")
    return result
