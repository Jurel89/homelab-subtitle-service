# src/homelab_subs/core/pipeline.py

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from .audio import FFmpeg
from .srt import write_srt_file
from .transcription import Transcriber, TranscriberConfig
from .translation import Translator, TranslatorConfig, TranslationBackend
from ..logging_config import get_logger, log_stage

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[float, int], None]]

try:  # pragma: no cover - optional dependency
    from .performance import PerformanceMonitor as _PerformanceMonitor, SystemMetrics as _SystemMetrics
except ImportError:  # pragma: no cover - handled gracefully at runtime
    _PerformanceMonitor = None
    _SystemMetrics = None

try:  # pragma: no cover - optional dependency
    from .db_logger import DatabaseLogger as _DatabaseLogger, JobLog as _JobLog, MetricLog as _MetricLog
except ImportError:  # pragma: no cover - handled gracefully at runtime
    _DatabaseLogger = None
    _JobLog = None
    _MetricLog = None

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .performance import PerformanceMonitor as PerformanceMonitorType, SystemMetrics as SystemMetricsType
    from .db_logger import DatabaseLogger as DatabaseLoggerType, JobLog as JobLogType, MetricLog as MetricLogType
else:  # pragma: no cover - runtime fallbacks
    PerformanceMonitorType = object
    SystemMetricsType = object
    DatabaseLoggerType = object
    JobLogType = object
    MetricLogType = object

PerformanceMonitor = _PerformanceMonitor
SystemMetrics = _SystemMetrics
DatabaseLogger = _DatabaseLogger
JobLog = _JobLog
MetricLog = _MetricLog

MONITORING_AVAILABLE = PerformanceMonitor is not None
DB_LOGGING_AVAILABLE = DatabaseLogger is not None


def _normalize_language(lang: Optional[str]) -> Optional[str]:
    if lang is None:
        return None
    lang = lang.strip()
    return lang or None


def _execute_pipeline(
    video_path: Path,
    output_path: Path,
    lang: Optional[str] = "en",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    task: str = "transcribe",
    beam_size: int = 5,
    vad_filter: bool = True,
    progress_callback: ProgressCallback = None,
    target_lang: Optional[str] = None,
    translation_backend: TranslationBackend = "nllb",
    translation_model: Optional[str] = None,
) -> Path:
    """Core subtitle generation steps shared by all runners.
    
    If target_lang is provided and differs from the source language (lang),
    automatic translation will be performed after transcription.
    """
    context = {"video_file": str(video_path.name)}
    logger.info("Starting pipeline for %s", video_path.name, extra=context)

    ff = FFmpeg()

    with log_stage(logger, "audio_extraction", **context):
        audio_path = ff.extract_audio_to_wav(video_path)

    config = TranscriberConfig(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    transcriber = Transcriber(config=config)

    language_param = _normalize_language(lang)

    with log_stage(logger, "transcription", **context):
        segments = transcriber.transcribe_file(
            audio_path,
            language=language_param,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            progress_callback=progress_callback,
        )

    with log_stage(logger, "srt_generation", **context):
        result = write_srt_file(segments, output_path)

    # Automatic translation if target language differs from source
    source_lang = language_param or "en"  # Default to English if auto-detected
    if target_lang and target_lang != source_lang:
        logger.info(
            "Translating subtitles from '%s' to '%s'",
            source_lang,
            target_lang,
            extra=context,
        )
        
        # Generate translated output path (e.g., video.en.srt -> video.es.srt)
        translated_output = output_path.with_suffix(f".{target_lang}.srt")
        
        translator_config = TranslatorConfig(
            backend=translation_backend,
            model_name=translation_model,
            device=device,
        )
        translator = Translator(config=translator_config)
        
        with log_stage(logger, "translation", **context):
            translator.translate_srt_file(
                input_path=result,
                output_path=translated_output,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        
        logger.info("Translation complete: %s", translated_output, extra=context)
        result = translated_output

    logger.info("Pipeline complete: %s", result, extra=context)
    return result


def generate_subtitles_for_video(
    video_path: Path,
    output_path: Path,
    lang: Optional[str] = "en",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    task: str = "transcribe",
    beam_size: int = 5,
    vad_filter: bool = True,
    progress_callback: ProgressCallback = None,
    target_lang: Optional[str] = None,
    translation_backend: TranslationBackend = "nllb",
    translation_model: Optional[str] = None,
) -> Path:
    """Public API for the minimal pipeline (no monitoring).
    
    Parameters
    ----------
    target_lang : Optional[str]
        If provided and different from lang, subtitles will be automatically
        translated to this language after transcription.
    translation_backend : TranslationBackend
        Translation backend to use: "helsinki" or "nllb". Default: "nllb".
    translation_model : Optional[str]
        Specific translation model name. If None, uses default for backend.
    """
    return _execute_pipeline(
        video_path=video_path,
        output_path=output_path,
        lang=lang,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
        progress_callback=progress_callback,
        target_lang=target_lang,
        translation_backend=translation_backend,
        translation_model=translation_model,
    )


class PipelineRunner:
    """Optional monitoring wrapper around the core pipeline."""

    def __init__(
        self,
        *,
        enable_monitoring: bool = True,
        enable_db_logging: bool = True,
        db_logger: Optional["DatabaseLoggerType"] = None,
        db_path: Optional[Path] = None,
        metrics_interval: float = 2.0,
        pipeline_fn: Optional[Callable[..., Path]] = None,
    ) -> None:
        self.enable_monitoring = bool(enable_monitoring and MONITORING_AVAILABLE)
        self.enable_db_logging = bool(enable_db_logging and DB_LOGGING_AVAILABLE)
        self.metrics_interval = metrics_interval
        self._pipeline_fn = pipeline_fn or _execute_pipeline

        self.monitor: Optional["PerformanceMonitorType"] = None
        if self.enable_monitoring:
            try:
                self.monitor = PerformanceMonitor()  # type: ignore[operator]
                logger.info("Performance monitoring enabled")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialize performance monitor: %s", exc)
                self.enable_monitoring = False

        self.db_logger = db_logger
        if self.enable_db_logging and self.db_logger is None:
            try:
                self.db_logger = DatabaseLogger(db_path)  # type: ignore[operator]
                logger.info("Database logging enabled at %s", self.db_logger.db_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialize database logger: %s", exc)
                self.enable_db_logging = False

        self._metrics_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._collected_metrics: list["SystemMetricsType"] = []

    def run(
        self,
        *,
        video_path: Path,
        output_path: Path,
        job_id: Optional[str] = None,
        lang: Optional[str] = "en",
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_callback: ProgressCallback = None,
        target_lang: Optional[str] = None,
        translation_backend: TranslationBackend = "nllb",
        translation_model: Optional[str] = None,
    ) -> Path:
        job_id = job_id or uuid.uuid4().hex[:8]
        start_time = time.time()
        started_at = datetime.now()

        context = {
            "job_id": job_id,
            "video_file": str(video_path.name),
        }

        logger.info("Starting monitored pipeline for job %s", job_id, extra=context)

        if self.enable_db_logging and self.db_logger:
            job_log = JobLog(
                job_id=job_id,
                video_path=str(video_path),
                output_path=str(output_path),
                status="running",
                language=_normalize_language(lang),
                model=model_name,
                task=task,
                device=device,
                started_at=started_at,
            )
            self.db_logger.create_job(job_log)

        if self.enable_monitoring and self.monitor:
            self._start_metrics_collection(job_id)

        try:
            result = self._pipeline_fn(
                video_path=video_path,
                output_path=output_path,
                lang=lang,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                progress_callback=progress_callback,
                target_lang=target_lang,
                translation_backend=translation_backend,
                translation_model=translation_model,
            )

            duration = time.time() - start_time

            if self.enable_db_logging and self.db_logger:
                summary = self._get_performance_summary()
                self.db_logger.update_job(
                    job_id=job_id,
                    status="completed",
                    completed_at=datetime.now(),
                    duration_seconds=duration,
                    performance_summary=summary,
                )

            logger.info(
                "Pipeline complete for job %s", job_id, extra={**context, "duration": duration}
            )
            return result

        except Exception as exc:
            duration = time.time() - start_time
            if self.enable_db_logging and self.db_logger:
                self.db_logger.update_job(
                    job_id=job_id,
                    status="failed",
                    completed_at=datetime.now(),
                    duration_seconds=duration,
                    error_message=str(exc),
                )

            logger.error(
                "Pipeline failed for job %s: %s", job_id, exc, extra=context, exc_info=True
            )
            raise

        finally:
            if self.enable_monitoring:
                self._stop_metrics_collection()

    def _start_metrics_collection(self, job_id: str) -> None:
        if not self.monitor:
            return

        self._stop_monitoring.clear()
        self._collected_metrics = []

        def collect_metrics() -> None:
            while not self._stop_monitoring.is_set():
                try:
                    metrics = self.monitor.get_metrics()
                    self._collected_metrics.append(metrics)

                    if self.enable_db_logging and self.db_logger:
                        metric_log = MetricLog(
                            job_id=job_id,
                            timestamp=datetime.fromtimestamp(metrics.timestamp),
                            cpu_percent=metrics.cpu_percent,
                            memory_percent=metrics.memory_percent,
                            memory_used_mb=metrics.memory_used_mb,
                            disk_read_mb=metrics.disk_read_mb,
                            disk_write_mb=metrics.disk_write_mb,
                            gpu_utilization=metrics.gpu_utilization,
                            gpu_memory_used_mb=metrics.gpu_memory_used_mb,
                            gpu_memory_percent=metrics.gpu_memory_percent,
                            gpu_temperature=metrics.gpu_temperature,
                        )
                        self.db_logger.add_metric(metric_log)

                    if len(self._collected_metrics) % 15 == 0:
                        logger.debug(
                            "Metrics snapshot: CPU=%.1f%% Memory=%.1f%%",
                            metrics.cpu_percent,
                            metrics.memory_percent,
                        )

                except Exception as exc:
                    logger.warning("Failed to collect metrics: %s", exc)

                time.sleep(self.metrics_interval)

        self._metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        self._metrics_thread.start()

    def _stop_metrics_collection(self) -> None:
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._stop_monitoring.set()
            self._metrics_thread.join(timeout=5)

    def _get_performance_summary(self) -> dict:
        if not self.monitor or not self._collected_metrics:
            return {}
        return self.monitor.get_summary_stats(self._collected_metrics)
