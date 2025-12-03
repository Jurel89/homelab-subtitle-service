"""Compatibility wrapper that now delegates to :class:`PipelineRunner`."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Callable

from .pipeline import PipelineRunner


class MonitoredPipeline(PipelineRunner):
    """Deprecated wrapper kept for backward compatibility."""

    def __init__(
        self,
        enable_monitoring: bool = True,
        enable_db_logging: bool = True,
        db_path: Optional[Path] = None,
        metrics_interval: float = 2.0,
    ) -> None:
        warnings.warn(
            "MonitoredPipeline is deprecated. Use PipelineRunner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            enable_monitoring=enable_monitoring,
            enable_db_logging=enable_db_logging,
            db_path=db_path,
            metrics_interval=metrics_interval,
        )

    def generate_subtitles(
        self,
        *,
        video_path: Path,
        output_path: Path,
        job_id: str,
        lang: Optional[str] = "en",
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_callback: Optional[Callable[[float, int], None]] = None,
    ) -> Path:
        """Proxy to :meth:`PipelineRunner.run` for legacy callers."""
        return super().run(
            video_path=video_path,
            output_path=output_path,
            job_id=job_id,
            lang=lang,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            progress_callback=progress_callback,
        )
