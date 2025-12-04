# src/homelab_subs/server/__init__.py

"""
Server module for the homelab subtitle service.

This module provides:
- FastAPI web application for REST API
- RQ-based job queue for background processing
- PostgreSQL-backed job persistence
- Web UI for monitoring and management
"""

from .models import Job, JobStatus, JobType, JobStage
from .settings import Settings, get_settings

__all__ = [
    "Job",
    "JobStatus",
    "JobType",
    "JobStage",
    "Settings",
    "get_settings",
]
