# src/tests/test_user_repository.py

"""
Tests for the UserRepository database layer.

Uses SQLite in-memory for fast testing without PostgreSQL.
"""

from __future__ import annotations

import uuid
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
        User,  # noqa: F401
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
def in_memory_user_repository(mock_settings):
    """Create a UserRepository with in-memory SQLite for testing."""
    from homelab_subs.server.repository import UserRepository

    repo = UserRepository(database_url="sqlite:///:memory:", settings=mock_settings)
    # Create all tables
    from homelab_subs.server.models import Base

    Base.metadata.create_all(repo._engine)
    yield repo


# Strong password that passes all validation requirements
VALID_PASSWORD = "SecurePass123!"
ANOTHER_VALID_PASSWORD = "NewSecurePass456@"


class TestUserRepositoryInit:
    """Tests for UserRepository initialization."""

    def test_repository_initializes(self, mock_settings):
        """UserRepository should initialize without error."""
        from homelab_subs.server.repository import UserRepository

        with patch(
            "homelab_subs.server.repository.get_settings", return_value=mock_settings
        ):
            repo = UserRepository(
                database_url="sqlite:///:memory:", settings=mock_settings
            )
            assert repo is not None

    def test_repository_available_flag(self):
        """SQLALCHEMY_AVAILABLE should be True when deps are installed."""
        from homelab_subs.server.repository import SQLALCHEMY_AVAILABLE

        assert SQLALCHEMY_AVAILABLE is True


class TestHasAnyUsers:
    """Tests for UserRepository.has_any_users."""

    def test_has_any_users_returns_false_when_empty(self, in_memory_user_repository):
        """has_any_users should return False when no users exist."""
        assert in_memory_user_repository.has_any_users() is False

    def test_has_any_users_returns_true_after_creation(self, in_memory_user_repository):
        """has_any_users should return True after a user is created."""
        in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        assert in_memory_user_repository.has_any_users() is True

    def test_count_users_starts_at_zero(self, in_memory_user_repository):
        """count_users should return 0 when no users exist."""
        assert in_memory_user_repository.count_users() == 0

    def test_count_users_increments_on_creation(self, in_memory_user_repository):
        """count_users should reflect the number of created users."""
        in_memory_user_repository.create_user("user1", VALID_PASSWORD)
        assert in_memory_user_repository.count_users() == 1

        in_memory_user_repository.create_user("user2", VALID_PASSWORD)
        assert in_memory_user_repository.count_users() == 2


class TestCreateUser:
    """Tests for UserRepository.create_user."""

    def test_create_user_returns_user(self, in_memory_user_repository):
        """create_user should return the created User entity."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        assert user is not None
        assert user.id is not None
        assert user.username == "alice"

    def test_create_user_is_active_by_default(self, in_memory_user_repository):
        """Newly created user should be active."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        assert user.is_active is True

    def test_create_user_is_not_admin_by_default(self, in_memory_user_repository):
        """Newly created user should not be admin by default."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        assert user.is_admin is False

    def test_create_admin_user(self, in_memory_user_repository):
        """create_user with is_admin=True should create an admin user."""
        user = in_memory_user_repository.create_user(
            "admin", VALID_PASSWORD, is_admin=True
        )
        assert user.is_admin is True

    def test_create_user_hashes_password(self, in_memory_user_repository):
        """The stored password_hash should not equal the plain password."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        assert user.password_hash != VALID_PASSWORD
        assert len(user.password_hash) > 0

    def test_create_duplicate_user_raises(self, in_memory_user_repository):
        """Creating a user with an existing username should raise ValueError."""
        in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        with pytest.raises(ValueError, match="already exists"):
            in_memory_user_repository.create_user("alice", VALID_PASSWORD)

    def test_create_user_with_weak_password_raises(self, in_memory_user_repository):
        """create_user should reject passwords that fail strength validation."""
        with pytest.raises(Exception):
            in_memory_user_repository.create_user("alice", "weak")


class TestAuthenticate:
    """Tests for UserRepository.authenticate."""

    def test_authenticate_correct_password_returns_user(
        self, in_memory_user_repository
    ):
        """authenticate should return the User when credentials are correct."""
        in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        user = in_memory_user_repository.authenticate("alice", VALID_PASSWORD)

        assert user is not None
        assert user.username == "alice"

    def test_authenticate_wrong_password_returns_none(self, in_memory_user_repository):
        """authenticate should return None when password is wrong."""
        in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        result = in_memory_user_repository.authenticate("alice", "WrongPassword1!")

        assert result is None

    def test_authenticate_unknown_user_returns_none(self, in_memory_user_repository):
        """authenticate should return None for a non-existent username."""
        result = in_memory_user_repository.authenticate("nobody", VALID_PASSWORD)

        assert result is None

    def test_authenticate_inactive_user_returns_none(self, in_memory_user_repository):
        """authenticate should return None when the user account is inactive."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        in_memory_user_repository.set_user_active(user.id, is_active=False)

        result = in_memory_user_repository.authenticate("alice", VALID_PASSWORD)

        assert result is None

    def test_authenticate_updates_last_login(self, in_memory_user_repository):
        """A successful authenticate should update last_login."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        assert user.last_login is None

        in_memory_user_repository.authenticate("alice", VALID_PASSWORD)

        updated_user = in_memory_user_repository.get_user_by_username("alice")
        assert updated_user.last_login is not None


class TestGetUser:
    """Tests for UserRepository.get_user_by_id and get_user_by_username."""

    def test_get_user_by_id(self, in_memory_user_repository):
        """get_user_by_id should return the user with matching ID."""
        created = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        found = in_memory_user_repository.get_user_by_id(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.username == "alice"

    def test_get_user_by_id_string(self, in_memory_user_repository):
        """get_user_by_id should accept a string UUID."""
        created = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        found = in_memory_user_repository.get_user_by_id(str(created.id))

        assert found is not None
        assert found.username == "alice"

    def test_get_user_by_id_not_found(self, in_memory_user_repository):
        """get_user_by_id should return None for a non-existent ID."""
        result = in_memory_user_repository.get_user_by_id(uuid.uuid4())
        assert result is None

    def test_get_user_by_username(self, in_memory_user_repository):
        """get_user_by_username should return the user with matching username."""
        in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        found = in_memory_user_repository.get_user_by_username("alice")

        assert found is not None
        assert found.username == "alice"

    def test_get_user_by_username_not_found(self, in_memory_user_repository):
        """get_user_by_username should return None when username doesn't exist."""
        result = in_memory_user_repository.get_user_by_username("nobody")
        assert result is None


class TestUpdatePassword:
    """Tests for UserRepository.update_password."""

    def test_update_password_allows_reauthentication(self, in_memory_user_repository):
        """After update_password, user should authenticate with the new password."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        result = in_memory_user_repository.update_password(
            user.id, ANOTHER_VALID_PASSWORD
        )
        assert result is True

        authenticated = in_memory_user_repository.authenticate(
            "alice", ANOTHER_VALID_PASSWORD
        )
        assert authenticated is not None
        assert authenticated.username == "alice"

    def test_update_password_invalidates_old_password(self, in_memory_user_repository):
        """After update_password, the old password should no longer authenticate."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        in_memory_user_repository.update_password(user.id, ANOTHER_VALID_PASSWORD)

        old_auth = in_memory_user_repository.authenticate("alice", VALID_PASSWORD)
        assert old_auth is None

    def test_update_password_not_found_returns_false(self, in_memory_user_repository):
        """update_password should return False for a non-existent user ID."""
        result = in_memory_user_repository.update_password(
            uuid.uuid4(), ANOTHER_VALID_PASSWORD
        )
        assert result is False

    def test_update_password_accepts_string_id(self, in_memory_user_repository):
        """update_password should accept a string UUID."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        result = in_memory_user_repository.update_password(
            str(user.id), ANOTHER_VALID_PASSWORD
        )
        assert result is True


class TestSetUserActive:
    """Tests for UserRepository.set_user_active."""

    def test_deactivate_user(self, in_memory_user_repository):
        """set_user_active(False) should deactivate the user."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        result = in_memory_user_repository.set_user_active(user.id, is_active=False)
        assert result is True

        found = in_memory_user_repository.get_user_by_id(user.id)
        assert found.is_active is False

    def test_reactivate_user(self, in_memory_user_repository):
        """set_user_active(True) should reactivate a deactivated user."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        in_memory_user_repository.set_user_active(user.id, is_active=False)

        in_memory_user_repository.set_user_active(user.id, is_active=True)

        found = in_memory_user_repository.get_user_by_id(user.id)
        assert found.is_active is True

    def test_set_user_active_not_found_returns_false(self, in_memory_user_repository):
        """set_user_active should return False for non-existent user."""
        result = in_memory_user_repository.set_user_active(
            uuid.uuid4(), is_active=False
        )
        assert result is False


class TestIncrementTokenVersion:
    """Tests for UserRepository.increment_token_version."""

    def test_initial_token_version_is_zero(self, in_memory_user_repository):
        """Newly created user should have token_version of 0."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        assert user.token_version == 0

    def test_increment_token_version(self, in_memory_user_repository):
        """increment_token_version should increase the version by 1."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        in_memory_user_repository.increment_token_version(user.id)

        updated = in_memory_user_repository.get_user_by_id(user.id)
        assert updated.token_version == 1

    def test_multiple_increments(self, in_memory_user_repository):
        """Calling increment_token_version multiple times should accumulate."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)

        in_memory_user_repository.increment_token_version(user.id)
        in_memory_user_repository.increment_token_version(user.id)

        updated = in_memory_user_repository.get_user_by_id(user.id)
        assert updated.token_version == 2


class TestUserToDict:
    """Tests for User.to_dict method."""

    def test_to_dict_excludes_password_hash(self, in_memory_user_repository):
        """User.to_dict() should not include the password_hash field."""
        user = in_memory_user_repository.create_user("alice", VALID_PASSWORD)
        d = user.to_dict()

        assert "password_hash" not in d
        assert "id" in d
        assert d["username"] == "alice"
        assert d["is_active"] is True
        assert d["is_admin"] is False
