# src/homelab_subs/server/settings.py

"""
Application settings and configuration.

Uses pydantic-settings to load configuration from environment variables
with sensible defaults for local development.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback for when pydantic-settings is not installed
    from pydantic import BaseModel as BaseSettings  # type: ignore[assignment]

    SettingsConfigDict = None  # type: ignore[misc, assignment]


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Environment Variables
    ---------------------
    DATABASE_URL : str
        PostgreSQL connection URL.
        Default: postgresql://localhost:5432/homelab_subs
    REDIS_URL : str
        Redis connection URL for job queue.
        Default: redis://localhost:6379/0
    QUEUE_DEFAULT : str
        Default queue name for RQ workers.
        Default: default
    QUEUE_HIGH : str
        High priority queue name.
        Default: high
    QUEUE_GPU : str
        GPU-specific queue name.
        Default: gpu
    WORKER_CONCURRENCY : int
        Max concurrent jobs per worker.
        Default: 1
    LOG_LEVEL : str
        Logging level.
        Default: INFO
    CORS_ORIGINS : str
        Comma-separated list of allowed CORS origins.
        Default: http://localhost:3000,http://localhost:8000
    API_PREFIX : str
        API route prefix.
        Default: /api/v1
    MEDIA_ROOT : str
        Root directory for media files.
        Default: /media
    OUTPUT_ROOT : str
        Root directory for output files.
        Default: /output
    """

    # Database
    database_url: str = "postgresql://localhost:5432/homelab_subs"

    # Redis / Queue
    redis_url: str = "redis://localhost:6379/0"
    queue_default: str = "default"
    queue_high: str = "high"
    queue_gpu: str = "gpu"

    # Worker settings
    worker_concurrency: int = 1

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API settings
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    api_prefix: str = "/api/v1"

    # File paths
    media_root: str = "/media"
    output_root: str = "/output"

    # Job defaults
    default_model: str = "small"
    default_device: str = "cpu"
    default_compute_type: str = "int8"
    default_translation_backend: str = "nllb"

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
        )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def sync_database_url(self) -> str:
        """
        Get synchronous database URL.

        Converts postgresql+asyncpg:// to postgresql:// for sync usage.
        """
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url.replace("postgresql+asyncpg://", "postgresql://")
        return self.database_url

    @property
    def async_database_url(self) -> str:
        """
        Get async database URL.

        Converts postgresql:// to postgresql+asyncpg:// for async usage.
        """
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        return self.database_url

    @property
    def all_queues(self) -> list[str]:
        """All queue names for workers to listen on."""
        return [self.queue_high, self.queue_default, self.queue_gpu]


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns
    -------
    Settings
        Application configuration singleton.
    """
    return Settings()
