"""
Tests for the logging configuration module.

This tests the logging infrastructure without needing actual video files.
"""

import json
import time

from homelab_subs.logging_config import (
    setup_logging,
    get_logger,
    log_stage,
    log_file_info,
    LogContext,
    StructuredFormatter,
    HumanReadableFormatter,
)


def test_basic_logging(tmp_path):
    """Test basic logging functionality."""
    log_file = tmp_path / "test.log"

    # Setup logging with both console and file output
    setup_logging(level="DEBUG", log_file=log_file, json_format=False)

    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Verify log file was created and contains JSON
    assert log_file.exists(), "Log file was not created"

    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0, "Log file is empty"

        # Verify first line is valid JSON
        first_log = json.loads(lines[0])
        assert "timestamp" in first_log
        assert "level" in first_log
        assert "message" in first_log


def test_log_context():
    """Test LogContext functionality."""
    setup_logging(level="INFO", json_format=False, use_colors=False)
    logger = get_logger("test_context")

    # Test with context
    with LogContext(job_id="test_123", video_file="test.mp4"):
        logger.info("Processing video")

    # Context should not affect logs outside the context manager
    logger.info("Outside context")


def test_stage_timing(tmp_path):
    """Test stage timing functionality."""
    log_file = tmp_path / "timing.log"
    setup_logging(level="INFO", log_file=log_file, json_format=False)
    logger = get_logger("timing_test")

    with log_stage(logger, "test_stage", job_id="timing_123"):
        time.sleep(0.01)  # Short sleep for testing

    # Verify log file contains stage and duration
    with open(log_file) as f:
        logs = [json.loads(line) for line in f]

    # Should have at least 2 logs: start and complete
    stage_logs = [log for log in logs if log.get("stage") == "test_stage"]
    assert len(stage_logs) >= 2

    # Complete log should have duration
    complete_logs = [log for log in stage_logs if "duration" in log]
    assert len(complete_logs) > 0
    assert complete_logs[0]["duration"] > 0


def test_json_format(tmp_path):
    """Test JSON log format."""
    log_file = tmp_path / "json.log"
    setup_logging(level="INFO", log_file=log_file, json_format=True)
    logger = get_logger("json_test")

    logger.info("Test message", extra={"job_id": "json_123", "custom_field": "value"})

    with open(log_file) as f:
        log = json.loads(f.readline())

    assert log["level"] == "INFO"
    assert log["message"] == "Test message"
    assert log["job_id"] == "json_123"
    assert log["custom_field"] == "value"


def test_structured_formatter():
    """Test StructuredFormatter directly."""
    import logging

    formatter = StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.job_id = "test_123"
    record.stage = "test_stage"

    output = formatter.format(record)
    log = json.loads(output)

    assert log["level"] == "INFO"
    assert log["message"] == "Test message"
    assert log["job_id"] == "test_123"
    assert log["stage"] == "test_stage"


def test_human_readable_formatter():
    """Test HumanReadableFormatter directly."""
    import logging

    formatter = HumanReadableFormatter(use_colors=False, datefmt="%Y-%m-%d %H:%M:%S")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    assert "INFO" in output
    assert "Test message" in output


def test_log_file_info(tmp_path):
    """Test log_file_info helper function."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    log_file = tmp_path / "info.log"
    setup_logging(level="DEBUG", log_file=log_file, json_format=False)
    logger = get_logger("info_test")

    log_file_info(logger, test_file, {"job_id": "info_123"})

    # Verify log was created
    with open(log_file) as f:
        logs = [json.loads(line) for line in f]

    assert len(logs) > 0
    # Should contain file size info
    file_logs = [log for log in logs if "file_size" in log]
    assert len(file_logs) > 0
