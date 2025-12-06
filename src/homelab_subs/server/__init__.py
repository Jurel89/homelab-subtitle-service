# src/homelab_subs/server/__init__.py

"""
Server module for the homelab subtitle service.

This module provides:
- FastAPI web application for REST API
- RQ-based job queue for background processing
- PostgreSQL-backed job persistence
- Authentication and authorization
- Web UI for monitoring and management
"""

from .models import Job, JobStatus, JobType, JobStage, User, GlobalSettings
from .settings import Settings, get_settings

__all__ = [
    "Job",
    "JobStatus",
    "JobType",
    "JobStage",
    "User",
    "GlobalSettings",
    "Settings",
    "get_settings",
]
