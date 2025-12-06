# src/tests/test_auth_service.py

"""
Tests for the authentication service.

Tests password hashing, JWT token creation/validation,
and authentication flows.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest


# Check if auth dependencies are available
try:
    from passlib.context import CryptContext
    from jose import jwt

    AUTH_DEPS_AVAILABLE = True
except ImportError:
    AUTH_DEPS_AVAILABLE = False


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not AUTH_DEPS_AVAILABLE,
    reason="Auth dependencies (passlib, python-jose) not installed",
)


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password_returns_hash(self):
        """hash_password should return a hashed string."""
        from homelab_subs.server.auth import hash_password

        password = "SecurePassword123!"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        # Argon2 hashes start with $argon2
        assert hashed.startswith("$argon2") or hashed.startswith("$2")

    def test_hash_password_produces_different_hashes(self):
        """Same password should produce different hashes (salted)."""
        from homelab_subs.server.auth import hash_password

        password = "SamePassword123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    def test_verify_password_correct(self):
        """verify_password should return True for correct password."""
        from homelab_subs.server.auth import hash_password, verify_password

        password = "CorrectPassword!"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """verify_password should return False for incorrect password."""
        from homelab_subs.server.auth import hash_password, verify_password

        password = "CorrectPassword!"
        hashed = hash_password(password)

        assert verify_password("WrongPassword!", hashed) is False

    def test_verify_password_empty_fails(self):
        """verify_password should return False for empty password."""
        from homelab_subs.server.auth import hash_password, verify_password

        password = "RealPassword!"
        hashed = hash_password(password)

        assert verify_password("", hashed) is False


class TestJWTTokenCreation:
    """Tests for JWT token creation."""

    def test_create_access_token_returns_string(self):
        """create_access_token should return a JWT string."""
        from homelab_subs.server.auth import create_access_token

        user_id = uuid4()
        token = create_access_token(user_id, "testuser")

        assert isinstance(token, str)
        assert len(token) > 0
        # JWT has 3 parts separated by dots
        assert len(token.split(".")) == 3

    def test_create_access_token_contains_claims(self):
        """Access token should contain user claims."""
        from homelab_subs.server.auth import create_access_token, JWT_SECRET_KEY, JWT_ALGORITHM

        user_id = uuid4()
        username = "testuser"
        token = create_access_token(user_id, username, is_admin=True)

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == str(user_id)
        assert payload["username"] == username
        assert payload["is_admin"] is True
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_create_access_token_custom_expiry(self):
        """Access token should accept custom expiry."""
        from homelab_subs.server.auth import create_access_token, JWT_SECRET_KEY, JWT_ALGORITHM

        user_id = uuid4()
        expires_delta = timedelta(hours=2)
        token = create_access_token(user_id, "testuser", expires_delta=expires_delta)

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected_min = datetime.now(timezone.utc) + timedelta(hours=1, minutes=55)
        expected_max = datetime.now(timezone.utc) + timedelta(hours=2, minutes=5)

        assert expected_min < exp_time < expected_max

    def test_create_refresh_token_returns_string(self):
        """create_refresh_token should return a JWT string."""
        from homelab_subs.server.auth import create_refresh_token

        user_id = uuid4()
        token = create_refresh_token(user_id)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    def test_create_refresh_token_contains_claims(self):
        """Refresh token should contain minimal claims."""
        from homelab_subs.server.auth import create_refresh_token, JWT_SECRET_KEY, JWT_ALGORITHM

        user_id = uuid4()
        token = create_refresh_token(user_id)

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "refresh"
        assert "exp" in payload
        # Refresh tokens should NOT contain username or is_admin
        assert "username" not in payload
        assert "is_admin" not in payload

    def test_create_token_pair_returns_both_tokens(self):
        """create_token_pair should return access and refresh tokens."""
        from homelab_subs.server.auth import create_token_pair

        user_id = uuid4()
        token_pair = create_token_pair(user_id, "testuser", is_admin=False)

        assert token_pair.access_token is not None
        assert token_pair.refresh_token is not None
        assert token_pair.token_type == "bearer"
        assert token_pair.expires_in > 0


class TestJWTTokenValidation:
    """Tests for JWT token validation."""

    def test_decode_token_valid(self):
        """decode_token should decode a valid token."""
        from homelab_subs.server.auth import create_access_token, decode_token

        user_id = uuid4()
        token = create_access_token(user_id, "testuser")

        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == str(user_id)

    def test_decode_token_invalid(self):
        """decode_token should return None for invalid token."""
        from homelab_subs.server.auth import decode_token

        result = decode_token("invalid.token.here")
        assert result is None

    def test_decode_token_expired(self):
        """decode_token should return None for expired token."""
        from homelab_subs.server.auth import create_access_token, decode_token

        user_id = uuid4()
        # Create token that expired 1 hour ago
        token = create_access_token(
            user_id, "testuser", expires_delta=timedelta(hours=-1)
        )

        result = decode_token(token)
        assert result is None

    def test_validate_access_token_valid(self):
        """validate_access_token should return TokenData for valid token."""
        from homelab_subs.server.auth import (
            create_access_token,
            validate_access_token,
        )

        user_id = uuid4()
        username = "testuser"
        token = create_access_token(user_id, username, is_admin=True)

        token_data = validate_access_token(token)
        assert token_data is not None
        assert token_data.user_id == str(user_id)
        assert token_data.username == username
        assert token_data.is_admin is True

    def test_validate_access_token_refresh_token_fails(self):
        """validate_access_token should reject refresh tokens."""
        from homelab_subs.server.auth import (
            create_refresh_token,
            validate_access_token,
        )

        user_id = uuid4()
        refresh_token = create_refresh_token(user_id)

        result = validate_access_token(refresh_token)
        assert result is None

    def test_validate_refresh_token_valid(self):
        """validate_refresh_token should return user_id for valid token."""
        from homelab_subs.server.auth import (
            create_refresh_token,
            validate_refresh_token,
        )

        user_id = uuid4()
        token = create_refresh_token(user_id)

        result_id = validate_refresh_token(token)
        assert result_id == str(user_id)

    def test_validate_refresh_token_access_token_fails(self):
        """validate_refresh_token should reject access tokens."""
        from homelab_subs.server.auth import (
            create_access_token,
            validate_refresh_token,
        )

        user_id = uuid4()
        access_token = create_access_token(user_id, "testuser")

        result = validate_refresh_token(access_token)
        assert result is None


class TestTokenData:
    """Tests for TokenData model."""

    def test_token_data_creation(self):
        """TokenData should be created from valid data."""
        from homelab_subs.server.auth import TokenData

        token_data = TokenData(
            user_id="test-uuid",
            username="testuser",
            is_admin=True,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert token_data.user_id == "test-uuid"
        assert token_data.username == "testuser"
        assert token_data.is_admin is True


class TestTokenPair:
    """Tests for TokenPair model."""

    def test_token_pair_creation(self):
        """TokenPair should be created with all fields."""
        from homelab_subs.server.auth import TokenPair

        token_pair = TokenPair(
            access_token="access.token.here",
            refresh_token="refresh.token.here",
            token_type="bearer",
            expires_in=1800,
        )

        assert token_pair.access_token == "access.token.here"
        assert token_pair.refresh_token == "refresh.token.here"
        assert token_pair.token_type == "bearer"
        assert token_pair.expires_in == 1800


class TestAuthDependencyCheck:
    """Tests for AUTH_DEPS_AVAILABLE flag."""

    def test_auth_deps_available_flag(self):
        """AUTH_DEPS_AVAILABLE should be True when deps installed."""
        from homelab_subs.server.auth import AUTH_DEPS_AVAILABLE

        assert AUTH_DEPS_AVAILABLE is True
