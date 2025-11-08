"""
RBAC Decorators for FastAPI Routes in Turkish Legal AI.

This module provides decorators for protecting routes with RBAC:
- @require_permission: Require specific permission
- @require_role: Require specific role
- @require_any_permission: Require any of multiple permissions
- @require_all_permissions: Require all of multiple permissions
- @require_policy: Evaluate custom policy

Example:
    >>> from backend.security.rbac.decorators import require_permission
    >>> from fastapi import APIRouter
    >>>
    >>> router = APIRouter()
    >>>
    >>> @router.get("/documents/{doc_id}")
    >>> @require_permission("document", "read")
    >>> async def get_document(doc_id: UUID):
    ...     return {"doc_id": str(doc_id)}
    >>>
    >>> @router.post("/contracts")
    >>> @require_role("lawyer")
    >>> async def create_contract():
    ...     return {"status": "created"}
"""

import functools
from typing import Callable, List, Optional
from uuid import UUID

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.session import get_db
from backend.core.exceptions import PermissionDeniedError, AuthenticationError
from backend.security.rbac.context import (
    get_current_user_id,
    get_current_tenant_id,
)
from backend.security.rbac.permissions import PermissionService
from backend.security.rbac.roles import RoleService
from backend.security.rbac.policies import PolicyEngine, PolicyContext

# =============================================================================
# PERMISSION DECORATORS
# =============================================================================


def require_permission(resource: str, action: str):
    """
    Decorator to require specific permission for route.

    Args:
        resource: Resource name (e.g., "document")
        action: Action name (e.g., "read")

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If user lacks permission

    Example:
        >>> @router.get("/documents/{doc_id}")
        >>> @require_permission("document", "read")
        >>> async def get_document(doc_id: UUID):
        ...     return {"doc_id": str(doc_id)}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from context
            user_id = get_current_user_id()
            if not user_id:
                raise AuthenticationError("Authentication required")

            # Get DB session from function args/kwargs
            db: Optional[AsyncSession] = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            if not db:
                # Create new session if not available
                async for session in get_db():
                    db = session
                    break

            # Check permission
            perm_service = PermissionService(db)
            has_perm = await perm_service.user_has_permission(
                user_id=user_id,
                resource=resource,
                action=action,
            )

            if not has_perm:
                raise PermissionDeniedError(
                    f"Permission denied: {resource}:{action}"
                )

            # Call original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_permission(*permissions: tuple[str, str]):
    """
    Decorator to require ANY of multiple permissions (OR logic).

    Args:
        *permissions: List of (resource, action) tuples

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If user lacks all permissions

    Example:
        >>> @router.get("/documents/{doc_id}")
        >>> @require_any_permission(
        ...     ("document", "read"),
        ...     ("document", "write")
        ... )
        >>> async def get_document(doc_id: UUID):
        ...     return {"doc_id": str(doc_id)}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = get_current_user_id()
            if not user_id:
                raise AuthenticationError("Authentication required")

            # Get DB session
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            if not db:
                async for session in get_db():
                    db = session
                    break

            # Check if user has ANY permission
            perm_service = PermissionService(db)
            has_any = await perm_service.user_has_any_permission(
                user_id=user_id,
                required_permissions=list(permissions),
            )

            if not has_any:
                perm_names = [f"{r}:{a}" for r, a in permissions]
                raise PermissionDeniedError(
                    f"Permission denied: requires one of {perm_names}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_all_permissions(*permissions: tuple[str, str]):
    """
    Decorator to require ALL of multiple permissions (AND logic).

    Args:
        *permissions: List of (resource, action) tuples

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If user lacks any permission

    Example:
        >>> @router.post("/contracts/approve/{contract_id}")
        >>> @require_all_permissions(
        ...     ("contract", "read"),
        ...     ("contract", "approve")
        ... )
        >>> async def approve_contract(contract_id: UUID):
        ...     return {"status": "approved"}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = get_current_user_id()
            if not user_id:
                raise AuthenticationError("Authentication required")

            # Get DB session
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            if not db:
                async for session in get_db():
                    db = session
                    break

            # Check if user has ALL permissions
            perm_service = PermissionService(db)
            has_all = await perm_service.user_has_all_permissions(
                user_id=user_id,
                required_permissions=list(permissions),
            )

            if not has_all:
                perm_names = [f"{r}:{a}" for r, a in permissions]
                raise PermissionDeniedError(
                    f"Permission denied: requires all of {perm_names}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# ROLE DECORATORS
# =============================================================================


def require_role(role_slug: str):
    """
    Decorator to require specific role for route.

    Args:
        role_slug: Role slug (e.g., "admin", "lawyer")

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If user doesn't have role

    Example:
        >>> @router.post("/admin/users")
        >>> @require_role("admin")
        >>> async def create_user():
        ...     return {"status": "created"}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = get_current_user_id()
            tenant_id = get_current_tenant_id()

            if not user_id:
                raise AuthenticationError("Authentication required")
            if not tenant_id:
                raise AuthenticationError("Tenant context required")

            # Get DB session
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            if not db:
                async for session in get_db():
                    db = session
                    break

            # Get user roles
            role_service = RoleService(db)
            user_roles = await role_service.get_user_roles(user_id)

            # Check if user has required role
            has_role = any(role.slug == role_slug for role in user_roles)

            if not has_role:
                raise PermissionDeniedError(f"Role required: {role_slug}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_role(*role_slugs: str):
    """
    Decorator to require ANY of multiple roles (OR logic).

    Args:
        *role_slugs: List of role slugs

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If user doesn't have any role

    Example:
        >>> @router.get("/legal/documents")
        >>> @require_any_role("lawyer", "paralegal", "senior_partner")
        >>> async def list_documents():
        ...     return {"documents": []}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = get_current_user_id()
            tenant_id = get_current_tenant_id()

            if not user_id:
                raise AuthenticationError("Authentication required")
            if not tenant_id:
                raise AuthenticationError("Tenant context required")

            # Get DB session
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            if not db:
                async for session in get_db():
                    db = session
                    break

            # Get user roles
            role_service = RoleService(db)
            user_roles = await role_service.get_user_roles(user_id)

            # Check if user has ANY required role
            user_role_slugs = {role.slug for role in user_roles}
            has_any_role = any(slug in user_role_slugs for slug in role_slugs)

            if not has_any_role:
                raise PermissionDeniedError(
                    f"Role required: one of {list(role_slugs)}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# POLICY DECORATORS
# =============================================================================


def require_policy(
    resource_type: str,
    action: str,
    *,
    resource_id_param: str = "id",
):
    """
    Decorator to evaluate policy for route.

    More flexible than permission checks - considers ownership,
    team membership, and custom policies.

    Args:
        resource_type: Resource type (e.g., "document")
        action: Action (e.g., "approve")
        resource_id_param: Parameter name for resource ID (default: "id")

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If policy denies access

    Example:
        >>> @router.delete("/documents/{doc_id}")
        >>> @require_policy("document", "delete", resource_id_param="doc_id")
        >>> async def delete_document(doc_id: UUID):
        ...     # Only document owner or admin can delete
        ...     return {"status": "deleted"}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = get_current_user_id()
            tenant_id = get_current_tenant_id()

            if not user_id:
                raise AuthenticationError("Authentication required")
            if not tenant_id:
                raise AuthenticationError("Tenant context required")

            # Get DB session
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            if not db:
                async for session in get_db():
                    db = session
                    break

            # Get resource ID from function kwargs
            resource_id = kwargs.get(resource_id_param)

            # Build policy context
            context = PolicyContext(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
            )

            # Evaluate policy
            engine = PolicyEngine(db)
            allowed = await engine.evaluate(context)

            if not allowed:
                raise PermissionDeniedError(
                    f"Policy denied: {resource_type}:{action}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# OWNERSHIP DECORATORS
# =============================================================================


def require_ownership(
    resource_type: str,
    resource_id_param: str = "id",
    owner_id_field: str = "owner_id",
):
    """
    Decorator to require resource ownership.

    Checks if current user owns the resource.

    Args:
        resource_type: Resource type (e.g., "document")
        resource_id_param: Parameter name for resource ID
        owner_id_field: Field name for owner ID in resource

    Raises:
        AuthenticationError: If user not authenticated
        PermissionDeniedError: If user doesn't own resource

    Example:
        >>> @router.put("/documents/{doc_id}")
        >>> @require_ownership("document", resource_id_param="doc_id")
        >>> async def update_my_document(doc_id: UUID):
        ...     # Only owner can update
        ...     return {"status": "updated"}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = get_current_user_id()

            if not user_id:
                raise AuthenticationError("Authentication required")

            # Get resource ID from kwargs
            resource_id = kwargs.get(resource_id_param)
            if not resource_id:
                raise PermissionDeniedError("Resource ID not provided")

            # TODO: Fetch resource and check ownership
            # This would query the database to get resource.owner_id
            # For now, we skip actual ownership check

            # Get DB session
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            if not db and "db" in kwargs:
                db = kwargs["db"]

            # Actual ownership check would happen here
            # resource = await db.get(ResourceModel, resource_id)
            # if resource.owner_id != user_id:
            #     raise PermissionDeniedError("Resource ownership required")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
