# src/tests/test_auth.py

"""
Tests for the authentication module.

Tests password hashing, JWT token handling, and authentication flows.
These tests can run without server dependencies (PostgreSQL, Redis).
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

# Check if auth dependencies are available
try:
    import passlib  # noqa: F401
    import jose  # noqa: F401

    AUTH_DEPS_AVAILABLE = True
except ImportError:
    AUTH_DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AUTH_DEPS_AVAILABLE,
    reason="Auth dependencies not installed (passlib, python-jose)",
)


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password_creates_hash(self):
        """Test that hash_password creates a valid hash."""
        from homelab_subs.server.auth import hash_password

        password = "SecurePassword123"
        hashed = hash_password(password)

        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 50  # Argon2 hashes are long

    def test_hash_password_different_for_same_password(self):
        """Test that hashing the same password twice gives different hashes (salt)."""
        from homelab_subs.server.auth import hash_password

        password = "SecurePassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2  # Salt should make them different

    def test_verify_password_correct(self):
        """Test that verify_password works with correct password."""
        from homelab_subs.server.auth import hash_password, verify_password

        password = "SecurePassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test that verify_password fails with wrong password."""
        from homelab_subs.server.auth import hash_password, verify_password

        password = "SecurePassword123"
        hashed = hash_password(password)

        assert verify_password("WrongPassword123", hashed) is False

    def test_verify_password_case_sensitive(self):
        """Test that password verification is case-sensitive."""
        from homelab_subs.server.auth import hash_password, verify_password

        password = "SecurePassword123"
        hashed = hash_password(password)

        assert verify_password("securepassword123", hashed) is False
        assert verify_password("SECUREPASSWORD123", hashed) is False


class TestPasswordValidation:
    """Tests for password strength validation."""

    def test_validate_password_strength_valid(self):
        """Test that valid passwords pass validation."""
        from homelab_subs.server.auth import validate_password_strength

        # These should not raise
        validate_password_strength("SecurePassword123")
        validate_password_strength("MyP@ssw0rd!")
        validate_password_strength("Test1234")

    def test_validate_password_too_short(self):
        """Test that short passwords are rejected."""
        from homelab_subs.server.auth import (
            validate_password_strength,
            PasswordValidationError,
        )

        with pytest.raises(PasswordValidationError) as exc:
            validate_password_strength("Short1")

        assert "at least 8 characters" in str(exc.value)

    def test_validate_password_no_uppercase(self):
        """Test that passwords without uppercase are rejected."""
        from homelab_subs.server.auth import (
            validate_password_strength,
            PasswordValidationError,
        )

        with pytest.raises(PasswordValidationError) as exc:
            validate_password_strength("lowercase123")

        assert "uppercase" in str(exc.value)

    def test_validate_password_no_lowercase(self):
        """Test that passwords without lowercase are rejected."""
        from homelab_subs.server.auth import (
            validate_password_strength,
            PasswordValidationError,
        )

        with pytest.raises(PasswordValidationError) as exc:
            validate_password_strength("UPPERCASE123")

        assert "lowercase" in str(exc.value)

    def test_validate_password_no_digit(self):
        """Test that passwords without digits are rejected."""
        from homelab_subs.server.auth import (
            validate_password_strength,
            PasswordValidationError,
        )

        with pytest.raises(PasswordValidationError) as exc:
            validate_password_strength("NoDigitsHere")

        assert "digit" in str(exc.value)

    def test_validate_password_multiple_errors(self):
        """Test that multiple validation errors are reported."""
        from homelab_subs.server.auth import (
            validate_password_strength,
            PasswordValidationError,
        )

        with pytest.raises(PasswordValidationError) as exc:
            validate_password_strength("bad")

        # Should have multiple error messages
        error_msg = str(exc.value)
        assert "8 characters" in error_msg
        assert "uppercase" in error_msg


class TestJWTTokens:
    """Tests for JWT token creation and validation."""

    def test_create_access_token(self):
        """Test creating an access token."""
        from homelab_subs.server.auth import create_access_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
            is_admin=False,
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_admin(self):
        """Test creating an access token for admin."""
        from homelab_subs.server.auth import create_access_token, decode_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="admin",
            is_admin=True,
        )

        payload = decode_token(token)
        assert payload["is_admin"] is True
        assert payload["type"] == "access"

    def test_create_refresh_token(self):
        """Test creating a refresh token."""
        from homelab_subs.server.auth import create_refresh_token

        user_id = uuid4()
        token = create_refresh_token(user_id=user_id)

        assert token is not None
        assert isinstance(token, str)

    def test_create_token_pair(self):
        """Test creating a token pair."""
        from homelab_subs.server.auth import create_token_pair

        user_id = uuid4()
        tokens = create_token_pair(
            user_id=user_id,
            username="testuser",
            is_admin=False,
        )

        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "bearer"
        assert tokens.expires_in > 0

    def test_decode_valid_token(self):
        """Test decoding a valid token."""
        from homelab_subs.server.auth import create_access_token, decode_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
            is_admin=False,
        )

        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["username"] == "testuser"
        assert payload["is_admin"] is False

    def test_decode_invalid_token(self):
        """Test decoding an invalid token."""
        from homelab_subs.server.auth import decode_token

        payload = decode_token("invalid.token.here")

        assert payload is None

    def test_decode_tampered_token(self):
        """Test decoding a tampered token fails."""
        from homelab_subs.server.auth import create_access_token, decode_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
            is_admin=False,
        )

        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1][::-1]  # Reverse the payload
        tampered_token = ".".join(parts)

        payload = decode_token(tampered_token)

        assert payload is None

    def test_validate_access_token(self):
        """Test validating an access token."""
        from homelab_subs.server.auth import create_access_token, validate_access_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
            is_admin=True,
        )

        token_data = validate_access_token(token)

        assert token_data is not None
        assert token_data.user_id == str(user_id)
        assert token_data.username == "testuser"
        assert token_data.is_admin is True

    def test_validate_access_token_rejects_refresh(self):
        """Test that access token validation rejects refresh tokens."""
        from homelab_subs.server.auth import (
            create_refresh_token,
            validate_access_token,
        )

        user_id = uuid4()
        refresh_token = create_refresh_token(user_id=user_id)

        token_data = validate_access_token(refresh_token)

        assert token_data is None

    def test_validate_refresh_token(self):
        """Test validating a refresh token."""
        from homelab_subs.server.auth import (
            create_refresh_token,
            validate_refresh_token,
        )

        user_id = uuid4()
        token = create_refresh_token(user_id=user_id)

        extracted_id = validate_refresh_token(token)

        assert extracted_id == str(user_id)

    def test_validate_refresh_token_rejects_access(self):
        """Test that refresh token validation rejects access tokens."""
        from homelab_subs.server.auth import (
            create_access_token,
            validate_refresh_token,
        )

        user_id = uuid4()
        access_token = create_access_token(
            user_id=user_id,
            username="testuser",
        )

        extracted_id = validate_refresh_token(access_token)

        assert extracted_id is None

    def test_token_custom_expiration(self):
        """Test creating tokens with custom expiration."""
        from homelab_subs.server.auth import create_access_token, decode_token

        user_id = uuid4()
        expires = timedelta(hours=2)

        token = create_access_token(
            user_id=user_id,
            username="testuser",
            expires_delta=expires,
        )

        payload = decode_token(token)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)

        # Should expire in about 2 hours (with some tolerance)
        delta = exp - now
        assert timedelta(hours=1, minutes=50) < delta < timedelta(hours=2, minutes=10)


class TestTokenDataModel:
    """Tests for the TokenData pydantic model."""

    def test_token_data_from_valid_token(self):
        """Test creating TokenData from a valid token."""
        from homelab_subs.server.auth import create_access_token, validate_access_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
            is_admin=True,
        )

        token_data = validate_access_token(token)

        assert token_data.user_id == str(user_id)
        assert token_data.username == "testuser"
        assert token_data.is_admin is True
        assert isinstance(token_data.exp, datetime)


class TestTokenPairModel:
    """Tests for the TokenPair model."""

    def test_token_pair_structure(self):
        """Test TokenPair has correct structure."""
        from homelab_subs.server.auth import create_token_pair

        user_id = uuid4()
        tokens = create_token_pair(
            user_id=user_id,
            username="testuser",
            is_admin=False,
        )

        assert hasattr(tokens, "access_token")
        assert hasattr(tokens, "refresh_token")
        assert hasattr(tokens, "token_type")
        assert hasattr(tokens, "expires_in")

    def test_token_pair_default_type(self):
        """Test TokenPair has correct default token type."""
        from homelab_subs.server.auth import create_token_pair

        tokens = create_token_pair(
            user_id=uuid4(),
            username="testuser",
        )

        assert tokens.token_type == "bearer"


class TestFastAPIDependencies:
    """Tests for FastAPI authentication dependencies."""

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """Test get_current_user with valid token."""
        from homelab_subs.server.auth import (
            get_current_user,
            create_access_token,
        )
        from fastapi.security import HTTPAuthorizationCredentials

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
            is_admin=False,
        )

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token,
        )

        token_data = await get_current_user(credentials)

        assert token_data.user_id == str(user_id)
        assert token_data.username == "testuser"

    @pytest.mark.asyncio
    async def test_get_current_user_no_credentials(self):
        """Test get_current_user with no credentials."""
        from homelab_subs.server.auth import get_current_user
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await get_current_user(None)

        assert exc.value.status_code == 401
        assert "Not authenticated" in exc.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test get_current_user with invalid token."""
        from homelab_subs.server.auth import get_current_user
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid.token.here",
        )

        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials)

        assert exc.value.status_code == 401
        assert "Invalid or expired" in exc.value.detail

    @pytest.mark.asyncio
    async def test_get_current_admin_user_is_admin(self):
        """Test get_current_admin_user with admin user."""
        from homelab_subs.server.auth import (
            get_current_admin_user,
            create_access_token,
            validate_access_token,
        )

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="admin",
            is_admin=True,
        )

        token_data = validate_access_token(token)
        result = await get_current_admin_user(token_data)

        assert result.is_admin is True

    @pytest.mark.asyncio
    async def test_get_current_admin_user_not_admin(self):
        """Test get_current_admin_user with non-admin user."""
        from homelab_subs.server.auth import (
            get_current_admin_user,
            create_access_token,
            validate_access_token,
        )
        from fastapi import HTTPException

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="regularuser",
            is_admin=False,
        )

        token_data = validate_access_token(token)

        with pytest.raises(HTTPException) as exc:
            await get_current_admin_user(token_data)

        assert exc.value.status_code == 403
        assert "Admin privileges required" in exc.value.detail

    @pytest.mark.asyncio
    async def test_get_optional_user_with_token(self):
        """Test get_optional_user with valid token."""
        from homelab_subs.server.auth import (
            get_optional_user,
            create_access_token,
        )
        from fastapi.security import HTTPAuthorizationCredentials

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            username="testuser",
        )

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token,
        )

        token_data = await get_optional_user(credentials)

        assert token_data is not None
        assert token_data.username == "testuser"

    @pytest.mark.asyncio
    async def test_get_optional_user_without_token(self):
        """Test get_optional_user without token."""
        from homelab_subs.server.auth import get_optional_user

        token_data = await get_optional_user(None)

        assert token_data is None
