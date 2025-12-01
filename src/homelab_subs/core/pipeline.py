# src/homelab_subs/core/pipeline.py

from __future__ import annotations

from pathlib import Path

from .audio import FFmpeg
from .transcription import Transcriber, TranscriberConfig
from .srt import write_srt_file


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
    ff = FFmpeg()
    audio_path = ff.extract_audio_to_wav(video_path)

    config = TranscriberConfig(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    transcriber = Transcriber(config=config)

    segments = transcriber.transcribe_file(
        audio_path,
        language=lang,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    return write_srt_file(segments, output_path)
