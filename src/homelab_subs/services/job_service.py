from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal, Optional, Callable, TYPE_CHECKING

from ..logging_config import get_logger
from ..core.pipeline import PipelineRunner, MONITORING_AVAILABLE, DB_LOGGING_AVAILABLE

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[float, int], None]]
TranslationBackend = Literal["helsinki", "nllb"]

try:  # pragma: no cover - optional dependency
    from ..core.db_logger import DatabaseLogger as _DatabaseLogger
except ImportError:  # pragma: no cover - handled gracefully at runtime
    _DatabaseLogger = None

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ..core.db_logger import DatabaseLogger as DatabaseLoggerType
else:
    DatabaseLoggerType = object

DatabaseLogger = _DatabaseLogger


class JobService:
    """High level orchestration for subtitle jobs."""

    def __init__(
        self,
        *,
        enable_monitoring: bool = True,
        enable_db_logging: bool = True,
        db_path: Optional[Path] = None,
        metrics_interval: float = 2.0,
    ) -> None:
        self.enable_monitoring = enable_monitoring and MONITORING_AVAILABLE
        self.enable_db_logging = enable_db_logging and DB_LOGGING_AVAILABLE
        self.db_path = db_path
        self.metrics_interval = metrics_interval

        self._db_logger: Optional["DatabaseLoggerType"] = None
        if self.enable_db_logging:
            if DatabaseLogger is None:
                logger.warning(
                    "Database logging requested but DatabaseLogger dependencies are unavailable"
                )
                self.enable_db_logging = False
            else:
                self._db_logger = DatabaseLogger(db_path)

    def generate_subtitles(
        self,
        *,
        video_path: Path,
        output_path: Path,
        lang: Optional[str] = "en",
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        job_id: Optional[str] = None,
        progress_callback: ProgressCallback = None,
        target_lang: Optional[str] = None,
        translation_backend: TranslationBackend = "nllb",
        translation_model: Optional[str] = None,
    ) -> Path:
        """Generate subtitles for a video file.

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
        job_id = job_id or uuid.uuid4().hex[:8]
        runner = PipelineRunner(
            enable_monitoring=self.enable_monitoring,
            enable_db_logging=self.enable_db_logging,
            db_logger=self._db_logger,
            db_path=self.db_path,
            metrics_interval=self.metrics_interval,
        )
        return runner.run(
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
            target_lang=target_lang,
            translation_backend=translation_backend,
            translation_model=translation_model,
        )

    # Database convenience helpers -------------------------------------------------
    def get_recent_jobs(self, limit: int = 20, status: Optional[str] = None):
        logger.debug("Fetching recent jobs: limit=%s status=%s", limit, status)
        db_logger = self._require_db_logger()
        return db_logger.get_recent_jobs(limit=limit, status=status)

    def get_job_details(self, job_id: str):
        db_logger = self._require_db_logger()
        job = db_logger.get_job(job_id)
        if job is None:
            return None
        metrics = db_logger.get_job_metrics(job_id)
        return {"job": job, "metrics": metrics}

    def get_statistics(self):
        db_logger = self._require_db_logger()
        return db_logger.get_statistics()

    def _require_db_logger(self) -> "DatabaseLoggerType":
        if not self.enable_db_logging or self._db_logger is None:
            raise RuntimeError(
                "Database logging is disabled or unavailable. Install monitoring dependencies."
            )
        return self._db_logger

    @property
    def monitoring_available(self) -> bool:
        return MONITORING_AVAILABLE

    @property
    def db_logging_available(self) -> bool:
        return DB_LOGGING_AVAILABLE
