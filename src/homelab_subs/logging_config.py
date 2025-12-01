# src/homelab_subs/logging_config.py

"""
Centralized logging configuration for homelab-subtitle-service.

This module provides structured logging with:
- Console and file output
- JSON formatting for web UI integration
- Contextual information (job IDs, stages, timing)
- Performance metrics
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs for easy parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Standard LogRecord attributes to ignore
        STANDARD_ATTRS = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread", "threadName"
        }

        # Add extra fields dynamically
        for key, value in record.__dict__.items():
            if key not in STANDARD_ATTRS and key not in log_data:
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Formatter for human-readable console output with colors (optional).
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Build the base message
        levelname = record.levelname
        if self.use_colors:
            color = self.COLORS.get(levelname, "")
            levelname = f"{color}{levelname}{self.RESET}"

        parts = [
            f"[{self.formatTime(record, self.datefmt)}]",
            f"[{levelname}]",
            f"[{record.name}]",
        ]

        # Add contextual information if present
        if hasattr(record, "job_id"):
            parts.append(f"[Job:{record.job_id}]")
        if hasattr(record, "stage"):
            parts.append(f"[{record.stage}]")

        parts.append(record.getMessage())

        # Add duration if present
        if hasattr(record, "duration"):
            parts.append(f"(took {record.duration:.2f}s)")

        msg = " ".join(parts)

        # Add exception info if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    use_colors: bool = True,
) -> None:
    """
    Configure the logging system for the application.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Optional[Path]
        If provided, logs will also be written to this file in JSON format.
    json_format : bool
        If True, console output will use JSON format. Otherwise, human-readable.
    use_colors : bool
        If True and console is a TTY, use colors in human-readable format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level.upper())

    if json_format:
        console_formatter = StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    else:
        console_formatter = HumanReadableFormatter(
            use_colors=use_colors,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (always JSON format for easier parsing)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_formatter = StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters
    ----------
    name : str
        Name of the module (typically __name__).

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding structured context to log records.

    Example:
        with LogContext(job_id="job_123", stage="audio_extraction"):
            logger.info("Extracting audio...")
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self.logger_class = logging.getLoggerClass()

    def __enter__(self):
        # Store the old factory
        self.old_factory = logging.getLogRecordFactory()

        # Create a new factory that adds our context
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the old factory
        logging.setLogRecordFactory(self.old_factory)


@contextmanager
def log_stage(logger: logging.Logger, stage: str, **context):
    """
    Context manager for logging a stage with timing information.

    Parameters
    ----------
    logger : logging.Logger
        The logger to use.
    stage : str
        Name of the stage being executed.
    **context
        Additional context to include in log records.

    Example:
        with log_stage(logger, "audio_extraction", video_file="movie.mp4"):
            # Do work...
            pass
    """
    context["stage"] = stage
    start_time = time.time()

    logger.info(f"Starting {stage}", extra=context)

    try:
        with LogContext(**context):
            yield
        duration = time.time() - start_time
        logger.info(
            f"Completed {stage}",
            extra={**context, "duration": duration},
        )
    except Exception as exc:
        duration = time.time() - start_time
        logger.error(
            f"Failed {stage}: {exc}",
            extra={**context, "duration": duration},
            exc_info=True,
        )
        raise


def log_file_info(logger: logging.Logger, file_path: Path, context: dict[str, Any]):
    """
    Log information about a file.

    Parameters
    ----------
    logger : logging.Logger
        The logger to use.
    file_path : Path
        Path to the file.
    context : dict
        Additional context to include in the log.
    """
    if file_path.exists():
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.debug(
            f"File: {file_path.name} ({size_mb:.2f} MB)",
            extra={**context, "file_size": file_size},
        )
