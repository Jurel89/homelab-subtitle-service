# src/tests/test_settings_repository.py

"""
Tests for the SettingsRepository database layer.

Uses SQLite in-memory for fast testing without PostgreSQL.
GlobalSettings uses a singleton pattern (id=1).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# Check if SQLAlchemy is available
try:
    from sqlalchemy import create_engine  # noqa: F401
    from sqlalchemy.orm import sessionmaker  # noqa: F401

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Check if models are available
try:
    from homelab_subs.server.models import (
        Base,  # noqa: F401
        GlobalSettings,
        SQLALCHEMY_AVAILABLE as MODELS_AVAILABLE,
    )
except ImportError:
    MODELS_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (SQLALCHEMY_AVAILABLE and MODELS_AVAILABLE),
    reason="SQLAlchemy or models not available",
)


@pytest.fixture
def mock_settings():
    """Create mock settings for repository."""
    settings = MagicMock()
    settings.database_url = "sqlite:///:memory:"
    settings.log_level = "INFO"
    return settings


@pytest.fixture
def in_memory_settings_repository(mock_settings):
    """Create a SettingsRepository with in-memory SQLite for testing."""
    from homelab_subs.server.repository import SettingsRepository

    with patch(
        "homelab_subs.server.repository.get_settings", return_value=mock_settings
    ):
        repo = SettingsRepository(
            database_url="sqlite:///:memory:", settings=mock_settings
        )
        # Create all tables
        from homelab_subs.server.models import Base

        Base.metadata.create_all(repo._engine)
        yield repo


class TestSettingsRepositoryInit:
    """Tests for SettingsRepository initialization."""

    def test_repository_initializes(self, mock_settings):
        """SettingsRepository should initialize without error."""
        from homelab_subs.server.repository import SettingsRepository

        with patch(
            "homelab_subs.server.repository.get_settings", return_value=mock_settings
        ):
            repo = SettingsRepository(
                database_url="sqlite:///:memory:", settings=mock_settings
            )
            assert repo is not None


class TestGetSettings:
    """Tests for SettingsRepository.get_settings."""

    def test_get_settings_creates_default_when_none_exist(
        self, in_memory_settings_repository
    ):
        """get_settings should create and return default settings on first call."""
        settings = in_memory_settings_repository.get_settings()

        assert settings is not None
        assert settings.id == 1

    def test_get_settings_returns_global_settings_instance(
        self, in_memory_settings_repository
    ):
        """get_settings should return a GlobalSettings object."""
        settings = in_memory_settings_repository.get_settings()

        assert isinstance(settings, GlobalSettings)

    def test_get_settings_default_model(self, in_memory_settings_repository):
        """Default settings should have 'small' as default_model."""
        settings = in_memory_settings_repository.get_settings()

        assert settings.default_model == "small"

    def test_get_settings_default_device(self, in_memory_settings_repository):
        """Default settings should use 'cpu' as default_device."""
        settings = in_memory_settings_repository.get_settings()

        assert settings.default_device == "cpu"

    def test_get_settings_default_language(self, in_memory_settings_repository):
        """Default settings should use 'en' as default_language."""
        settings = in_memory_settings_repository.get_settings()

        assert settings.default_language == "en"

    def test_get_settings_prefer_gpu_false_by_default(
        self, in_memory_settings_repository
    ):
        """Default settings should have prefer_gpu as False."""
        settings = in_memory_settings_repository.get_settings()

        assert settings.prefer_gpu is False

    def test_get_settings_idempotent(self, in_memory_settings_repository):
        """Calling get_settings twice should return the same singleton row."""
        settings1 = in_memory_settings_repository.get_settings()
        settings2 = in_memory_settings_repository.get_settings()

        assert settings1.id == settings2.id == 1


class TestUpdateSettings:
    """Tests for SettingsRepository.update_settings."""

    def test_update_settings_creates_record_if_not_exists(
        self, in_memory_settings_repository
    ):
        """update_settings should create settings if none exist."""
        settings = in_memory_settings_repository.update_settings(default_model="large-v2")

        assert settings is not None
        assert settings.id == 1
        assert settings.default_model == "large-v2"

    def test_update_settings_persists_changes(self, in_memory_settings_repository):
        """update_settings should persist changes that are visible on next get."""
        in_memory_settings_repository.update_settings(worker_count=4)

        fetched = in_memory_settings_repository.get_settings()
        assert fetched.worker_count == 4

    def test_update_settings_multiple_fields(self, in_memory_settings_repository):
        """update_settings should accept and apply multiple fields at once."""
        settings = in_memory_settings_repository.update_settings(
            default_model="medium",
            default_device="cuda",
            prefer_gpu=True,
            worker_count=2,
        )

        assert settings.default_model == "medium"
        assert settings.default_device == "cuda"
        assert settings.prefer_gpu is True
        assert settings.worker_count == 2

    def test_update_settings_partial_update_preserves_other_fields(
        self, in_memory_settings_repository
    ):
        """Updating one field should not reset other fields to defaults."""
        in_memory_settings_repository.update_settings(worker_count=8)
        in_memory_settings_repository.update_settings(default_model="large-v2")

        settings = in_memory_settings_repository.get_settings()
        assert settings.worker_count == 8
        assert settings.default_model == "large-v2"

    def test_update_settings_invalid_field_raises(self, in_memory_settings_repository):
        """update_settings should raise ValueError for unknown field names."""
        with pytest.raises(ValueError, match="Invalid settings fields"):
            in_memory_settings_repository.update_settings(
                nonexistent_field="some_value"
            )

    def test_update_settings_log_retention_days(self, in_memory_settings_repository):
        """update_settings should update log_retention_days."""
        settings = in_memory_settings_repository.update_settings(log_retention_days=60)
        assert settings.log_retention_days == 60

    def test_update_settings_job_retention_days(self, in_memory_settings_repository):
        """update_settings should update job_retention_days."""
        settings = in_memory_settings_repository.update_settings(job_retention_days=180)
        assert settings.job_retention_days == 180

    def test_update_settings_default_compute_type(self, in_memory_settings_repository):
        """update_settings should update default_compute_type."""
        settings = in_memory_settings_repository.update_settings(
            default_compute_type="float16"
        )
        assert settings.default_compute_type == "float16"

    def test_update_settings_default_translation_backend(
        self, in_memory_settings_repository
    ):
        """update_settings should update default_translation_backend."""
        settings = in_memory_settings_repository.update_settings(
            default_translation_backend="helsinki"
        )
        assert settings.default_translation_backend == "helsinki"


class TestAddMediaFolder:
    """Tests for SettingsRepository.add_media_folder."""

    def test_add_media_folder_adds_path(self, in_memory_settings_repository):
        """add_media_folder should add the path to media_folders list."""
        settings = in_memory_settings_repository.add_media_folder("/media/movies")

        assert "/media/movies" in (settings.media_folders or [])

    def test_add_media_folder_multiple_paths(self, in_memory_settings_repository):
        """add_media_folder should support adding multiple distinct paths."""
        in_memory_settings_repository.add_media_folder("/media/movies")
        settings = in_memory_settings_repository.add_media_folder("/media/tv")

        folders = settings.media_folders or []
        assert "/media/movies" in folders
        assert "/media/tv" in folders

    def test_add_media_folder_is_idempotent(self, in_memory_settings_repository):
        """Adding the same folder twice should not create duplicates."""
        in_memory_settings_repository.add_media_folder("/media/movies")
        settings = in_memory_settings_repository.add_media_folder("/media/movies")

        folders = settings.media_folders or []
        assert folders.count("/media/movies") == 1

    def test_add_media_folder_persists(self, in_memory_settings_repository):
        """Added folders should be visible on subsequent get_settings calls."""
        in_memory_settings_repository.add_media_folder("/media/movies")

        fetched = in_memory_settings_repository.get_settings()
        assert "/media/movies" in (fetched.media_folders or [])


class TestRemoveMediaFolder:
    """Tests for SettingsRepository.remove_media_folder."""

    def test_remove_media_folder_removes_path(self, in_memory_settings_repository):
        """remove_media_folder should remove the specified path from media_folders."""
        in_memory_settings_repository.add_media_folder("/media/movies")

        settings = in_memory_settings_repository.remove_media_folder("/media/movies")

        assert "/media/movies" not in (settings.media_folders or [])

    def test_remove_media_folder_only_removes_target(
        self, in_memory_settings_repository
    ):
        """remove_media_folder should not affect other folders in the list."""
        in_memory_settings_repository.add_media_folder("/media/movies")
        in_memory_settings_repository.add_media_folder("/media/tv")

        settings = in_memory_settings_repository.remove_media_folder("/media/movies")

        folders = settings.media_folders or []
        assert "/media/tv" in folders
        assert "/media/movies" not in folders

    def test_remove_nonexistent_folder_is_noop(self, in_memory_settings_repository):
        """remove_media_folder on a path that doesn't exist should not raise."""
        # Should not raise even if the folder was never added
        settings = in_memory_settings_repository.remove_media_folder(
            "/media/doesnotexist"
        )
        assert settings is not None

    def test_remove_media_folder_persists(self, in_memory_settings_repository):
        """Removed folders should not appear on subsequent get_settings calls."""
        in_memory_settings_repository.add_media_folder("/media/movies")
        in_memory_settings_repository.remove_media_folder("/media/movies")

        fetched = in_memory_settings_repository.get_settings()
        assert "/media/movies" not in (fetched.media_folders or [])


class TestGlobalSettingsToDict:
    """Tests for GlobalSettings.to_dict method."""

    def test_to_dict_returns_dict(self, in_memory_settings_repository):
        """GlobalSettings.to_dict() should return a dictionary."""
        settings = in_memory_settings_repository.get_settings()
        d = settings.to_dict()

        assert isinstance(d, dict)

    def test_to_dict_contains_expected_keys(self, in_memory_settings_repository):
        """GlobalSettings.to_dict() should include all settings fields."""
        settings = in_memory_settings_repository.get_settings()
        d = settings.to_dict()

        expected_keys = {
            "media_folders",
            "default_model",
            "default_device",
            "default_compute_type",
            "default_language",
            "default_translation_backend",
            "worker_count",
            "log_retention_days",
            "job_retention_days",
            "prefer_gpu",
            "updated_at",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_media_folders_is_list(self, in_memory_settings_repository):
        """GlobalSettings.to_dict() should return media_folders as a list."""
        settings = in_memory_settings_repository.get_settings()
        d = settings.to_dict()

        assert isinstance(d["media_folders"], list)
