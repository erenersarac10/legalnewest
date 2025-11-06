"""
Authentication Dependencies for Turkish Legal AI Platform.

Provides FastAPI dependency injection for authentication and authorization.

Features:
- JWT token validation
- User extraction from token
- Permission checking
- Optional vs required authentication
- API key support

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.database import get_db
from backend.core import (
    UnauthorizedException,
    get_logger,
)
from backend.core.database.models import User

logger = get_logger(__name__)


async def get_current_user_id(request: Request) -> UUID:
    """
    Extract current user ID from request state.

    User ID is set by AuthMiddleware after JWT validation.

    Args:
        request: FastAPI request

    Returns:
        User UUID

    Raises:
        HTTPException: If user is not authenticated

    Example:
        @app.get("/profile")
        async def get_profile(user_id: UUID = Depends(get_current_user_id)):
            return {"user_id": str(user_id)}
    """
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UUID(user_id) if isinstance(user_id, str) else user_id


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> User:
    """
    Get current authenticated user from database.

    Fetches full user object based on user_id from JWT token.

    Args:
        db: Database session
        user_id: User ID from token

    Returns:
        User model instance

    Raises:
        HTTPException: If user not found or inactive

    Example:
        @app.get("/me")
        async def get_me(current_user: User = Depends(get_current_user)):
            return {
                "id": str(current_user.id),
                "email": current_user.email,
                "name": current_user.full_name,
            }
    """
    result = await db.execute(
        select(User)
        .where(User.id == user_id)
        .where(User.is_deleted == False)  # noqa: E712
    )

    user = result.scalar_one_or_none()

    if not user:
        logger.warning("User not found", user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.is_active:
        logger.warning("Inactive user attempted access", user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user (alias for get_current_user).

    Args:
        current_user: Current user

    Returns:
        User model instance

    Example:
        @app.post("/contracts")
        async def create_contract(
            user: User = Depends(get_current_active_user)
        ):
            # User is guaranteed to be active
            pass
    """
    return current_user


async def get_optional_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.

    Useful for endpoints that work both authenticated and anonymous.

    Args:
        request: FastAPI request
        db: Database session

    Returns:
        User instance or None

    Example:
        @app.get("/public-data")
        async def get_data(user: Optional[User] = Depends(get_optional_current_user)):
            if user:
                # Return personalized data
                return {"data": "personalized", "user_id": str(user.id)}
            else:
                # Return public data
                return {"data": "public"}
    """
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        return None

    try:
        result = await db.execute(
            select(User)
            .where(User.id == UUID(user_id) if isinstance(user_id, str) else user_id)
            .where(User.is_deleted == False)  # noqa: E712
            .where(User.is_active == True)  # noqa: E712
        )

        return result.scalar_one_or_none()

    except Exception as e:
        logger.warning("Failed to fetch optional user", error=str(e))
        return None


def require_permission(permission: str):
    """
    Dependency factory for permission checking.

    Args:
        permission: Required permission name

    Returns:
        Dependency function

    Example:
        @app.delete("/users/{user_id}")
        async def delete_user(
            user_id: UUID,
            _: None = Depends(require_permission("users:delete")),
        ):
            # User has users:delete permission
            pass
    """

    async def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user),
    ) -> None:
        """Check if user has required permission."""
        # Get permissions from request state (set by AuthMiddleware)
        permissions = getattr(request.state, "permissions", [])

        if permission not in permissions:
            logger.warning(
                "Permission denied",
                user_id=str(current_user.id),
                required_permission=permission,
                user_permissions=permissions,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required",
            )

    return permission_checker


def require_role(role: str):
    """
    Dependency factory for role checking.

    Args:
        role: Required role name

    Returns:
        Dependency function

    Example:
        @app.post("/admin/settings")
        async def update_settings(
            _: None = Depends(require_role("admin")),
        ):
            # User has admin role
            pass
    """

    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> None:
        """Check if user has required role."""
        # Check user's role
        # (Assumes User model has role or roles relationship)
        if not hasattr(current_user, "role") or current_user.role != role:
            logger.warning(
                "Role denied",
                user_id=str(current_user.id),
                required_role=role,
                user_role=getattr(current_user, "role", None),
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role denied: {role} required",
            )

    return role_checker


__all__ = [
    "get_current_user_id",
    "get_current_user",
    "get_current_active_user",
    "get_optional_current_user",
    "require_permission",
    "require_role",
]
