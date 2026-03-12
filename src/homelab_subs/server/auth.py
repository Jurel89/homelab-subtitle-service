# src/homelab_subs/server/auth.py

"""
Authentication and authorization service.

Provides:
- Password hashing with Argon2 (via passlib)
- JWT token creation and validation
- User authentication flows
- FastAPI dependencies for protected routes
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

try:
    from passlib.context import CryptContext
    import jwt
    from jwt.exceptions import PyJWTError as JWTError
    from pydantic import BaseModel

    AUTH_DEPS_AVAILABLE = True
except ImportError:
    AUTH_DEPS_AVAILABLE = False
    CryptContext = None  # type: ignore[misc, assignment]
    JWTError = Exception  # type: ignore[misc, assignment]
    jwt = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# JWT Settings - MUST be set via environment variable in production
_DEFAULT_SECRET = "your-secret-key-change-in-production"  # nosec B105 - Fallback only
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


def _validate_jwt_secret() -> str:
    key = os.environ.get("JWT_SECRET_KEY", _DEFAULT_SECRET)
    if key == _DEFAULT_SECRET:
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured. "
            "Set the JWT_SECRET_KEY environment variable to a random string of at least 32 characters. "
            'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if len(key) < 32:
        raise RuntimeError(
            f"JWT_SECRET_KEY is too short ({len(key)} chars). Minimum 32 characters required."
        )
    return key


_jwt_secret_key: Optional[str] = None


def get_jwt_secret() -> str:
    global _jwt_secret_key
    if _jwt_secret_key is None:
        _jwt_secret_key = _validate_jwt_secret()
    return _jwt_secret_key


# =============================================================================
# Password Hashing
# =============================================================================

if AUTH_DEPS_AVAILABLE and CryptContext is not None:
    # Argon2 is the recommended algorithm for password hashing
    pwd_context = CryptContext(
        schemes=["argon2", "bcrypt"],
        deprecated="auto",
        argon2__memory_cost=65536,  # 64 MB
        argon2__time_cost=3,
        argon2__parallelism=4,
    )
else:
    pwd_context = None  # type: ignore[assignment]


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.

    Parameters
    ----------
    password : str
        Plain text password.

    Returns
    -------
    str
        Argon2 password hash.

    Raises
    ------
    RuntimeError
        If auth dependencies are not installed.
    """
    if pwd_context is None:
        raise RuntimeError(
            "Authentication dependencies not installed. "
            "Install with: pip install homelab-subs[server]"
        )
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Parameters
    ----------
    plain_password : str
        Plain text password to verify.
    hashed_password : str
        Stored password hash.

    Returns
    -------
    bool
        True if password matches, False otherwise.
    """
    if pwd_context is None:
        raise RuntimeError(
            "Authentication dependencies not installed. "
            "Install with: pip install homelab-subs[server]"
        )
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# JWT Token Management
# =============================================================================


class TokenData(BaseModel if AUTH_DEPS_AVAILABLE else object):  # type: ignore[misc]
    """Data extracted from a JWT token."""

    user_id: str
    username: str
    is_admin: bool = False
    exp: datetime


class TokenPair(BaseModel if AUTH_DEPS_AVAILABLE else object):  # type: ignore[misc]
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


def create_access_token(
    user_id: UUID,
    username: str,
    is_admin: bool = False,
    expires_delta: Optional[timedelta] = None,
    token_version: int = 0,
) -> str:
    """
    Create a JWT access token.

    Parameters
    ----------
    user_id : UUID
        User's unique identifier.
    username : str
        User's username.
    is_admin : bool
        Whether user has admin privileges.
    expires_delta : timedelta, optional
        Custom expiration time. Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.
    token_version : int
        User's current token version for revocation support.

    Returns
    -------
    str
        Encoded JWT access token.
    """
    if jwt is None:
        raise RuntimeError(
            "Authentication dependencies not installed. "
            "Install with: pip install homelab-subs[server]"
        )

    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": str(user_id),
        "username": username,
        "is_admin": is_admin,
        "exp": expire,
        "type": "access",
        "token_version": token_version,
    }

    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(
    user_id: UUID,
    expires_delta: Optional[timedelta] = None,
    token_version: int = 0,
) -> str:
    """
    Create a JWT refresh token.

    Refresh tokens are longer-lived and used to obtain new access tokens.

    Parameters
    ----------
    user_id : UUID
        User's unique identifier.
    expires_delta : timedelta, optional
        Custom expiration time. Defaults to REFRESH_TOKEN_EXPIRE_DAYS.
    token_version : int
        User's current token version for revocation support.

    Returns
    -------
    str
        Encoded JWT refresh token.
    """
    if jwt is None:
        raise RuntimeError(
            "Authentication dependencies not installed. "
            "Install with: pip install homelab-subs[server]"
        )

    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh",
        "token_version": token_version,
    }

    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_token_pair(
    user_id: UUID,
    username: str,
    is_admin: bool = False,
    token_version: int = 0,
) -> TokenPair:
    """
    Create both access and refresh tokens.

    Parameters
    ----------
    user_id : UUID
        User's unique identifier.
    username : str
        User's username.
    is_admin : bool
        Whether user has admin privileges.
    token_version : int
        User's current token version for revocation support.

    Returns
    -------
    TokenPair
        Access and refresh tokens with metadata.
    """
    access_token = create_access_token(user_id, username, is_admin, token_version=token_version)
    refresh_token = create_refresh_token(user_id, token_version=token_version)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",  # nosec B106 - OAuth2 standard token type, not a password
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.

    Parameters
    ----------
    token : str
        JWT token to decode.

    Returns
    -------
    dict or None
        Token payload if valid, None if invalid or expired.
    """
    if jwt is None:
        raise RuntimeError(
            "Authentication dependencies not installed. "
            "Install with: pip install homelab-subs[server]"
        )

    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token validation failed: {e}")
        return None


def validate_access_token(token: str) -> Optional[TokenData]:
    """
    Validate an access token and extract user data.

    Parameters
    ----------
    token : str
        JWT access token.

    Returns
    -------
    TokenData or None
        User data if token is valid, None otherwise.
    """
    payload = decode_token(token)

    if payload is None:
        return None

    # Verify it's an access token
    if payload.get("type") != "access":
        logger.warning("Token is not an access token")
        return None

    try:
        return TokenData(
            user_id=payload["sub"],
            username=payload["username"],
            is_admin=payload.get("is_admin", False),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to extract token data: {e}")
        return None


def validate_refresh_token(token: str) -> Optional[str]:
    """
    Validate a refresh token and extract user ID.

    Parameters
    ----------
    token : str
        JWT refresh token.

    Returns
    -------
    str or None
        User ID if token is valid, None otherwise.
    """
    payload = decode_token(token)

    if payload is None:
        return None

    # Verify it's a refresh token
    if payload.get("type") != "refresh":
        logger.warning("Token is not a refresh token")
        return None

    return payload.get("sub")


# =============================================================================
# Password Validation
# =============================================================================


class PasswordValidationError(Exception):
    """Raised when password doesn't meet requirements."""

    pass


def validate_password_strength(password: str) -> list[str]:
    """
    Validate password meets security requirements.

    Requirements:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Parameters
    ----------
    password : str
        Password to validate.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list means password is valid.
    """
    errors = []

    if len(password) < 12:
        errors.append("Password must be at least 12 characters long")

    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")

    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")

    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit")

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;'/`~]", password):
        errors.append("Password must contain at least one special character")

    return errors


# =============================================================================
# FastAPI Dependencies
# =============================================================================

# These will be imported and used in the API module
# They require FastAPI to be available

try:
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    # Security scheme for JWT Bearer tokens
    security = HTTPBearer(auto_error=False)

    async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> TokenData:
        """
        FastAPI dependency to get current authenticated user.

        Use this dependency on routes that require authentication.
        Verifies token version against the database to support token revocation.

        Returns
        -------
        TokenData
            Current user's data from the JWT token.

        Raises
        ------
        HTTPException
            401 if not authenticated or token is invalid/revoked.
        """
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = validate_access_token(credentials.credentials)

        if token_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify token version against database
        payload = decode_token(credentials.credentials)
        if payload is not None:
            from homelab_subs.server.repository import UserRepository
            from homelab_subs.server.settings import get_settings

            user_repo = UserRepository(settings=get_settings())
            user = user_repo.get_user_by_id(token_data.user_id)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if payload.get("token_version") != user.token_version:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return token_data

    async def get_current_admin_user(
        current_user: TokenData = Depends(get_current_user),
    ) -> TokenData:
        """
        FastAPI dependency to require admin privileges.

        Use this dependency on routes that require admin access.

        Returns
        -------
        TokenData
            Current admin user's data.

        Raises
        ------
        HTTPException
            403 if user is not an admin.
        """
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )
        return current_user

    async def get_optional_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> Optional[TokenData]:
        """
        FastAPI dependency for optionally authenticated routes.

        Use this for routes that work differently for authenticated
        vs anonymous users.

        Returns
        -------
        TokenData or None
            Current user's data if authenticated, None otherwise.
        """
        if credentials is None:
            return None

        return validate_access_token(credentials.credentials)

except ImportError:
    # FastAPI not available - dependencies will fail at runtime
    security = None  # type: ignore[assignment]
    get_current_user = None  # type: ignore[assignment]
    get_current_admin_user = None  # type: ignore[assignment]
    get_optional_user = None  # type: ignore[assignment]
