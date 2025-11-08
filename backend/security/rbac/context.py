"""
RBAC Request Context Management for Turkish Legal AI.

This module provides request context utilities for RBAC:
- Extract user/tenant from request
- Build PolicyContext from HTTP request
- Context variables for async request handling
- Integration with FastAPI dependency injection

Example:
    >>> from backend.security.rbac.context import get_current_user, build_policy_context
    >>> from fastapi import Depends
    >>>
    >>> @app.get("/documents/{doc_id}")
    >>> async def get_document(
    ...     doc_id: UUID,
    ...     user: User = Depends(get_current_user),
    ...     db: AsyncSession = Depends(get_db)
    ... ):
    ...     # Build policy context
    ...     ctx = await build_policy_context(
    ...         request=request,
    ...         user=user,
    ...         resource_type="document",
    ...         resource_id=doc_id,
    ...         action="read"
    ...     )
    ...
    ...     # Evaluate policy
    ...     engine = PolicyEngine(db)
    ...     if not await engine.evaluate(ctx):
    ...         raise PermissionDeniedError()
    ...
    ...     return await get_document_by_id(doc_id)
"""

from contextvars import ContextVar
from typing import Optional
from uuid import UUID

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AuthenticationError
from backend.security.rbac.policies import PolicyContext

# =============================================================================
# CONTEXT VARIABLES (Thread-safe request context)
# =============================================================================

# Current user ID for the request
current_user_id: ContextVar[Optional[UUID]] = ContextVar(
    "current_user_id", default=None
)

# Current tenant ID for the request
current_tenant_id: ContextVar[Optional[UUID]] = ContextVar(
    "current_tenant_id", default=None
)

# Current session ID
current_session_id: ContextVar[Optional[str]] = ContextVar(
    "current_session_id", default=None
)


# =============================================================================
# CONTEXT SETTERS/GETTERS
# =============================================================================


def set_current_user(user_id: UUID) -> None:
    """
    Set current user ID in request context.

    Args:
        user_id: User ID

    Example:
        >>> set_current_user(user.id)
    """
    current_user_id.set(user_id)


def get_current_user_id() -> Optional[UUID]:
    """
    Get current user ID from request context.

    Returns:
        User ID or None

    Example:
        >>> user_id = get_current_user_id()
    """
    return current_user_id.get()


def set_current_tenant(tenant_id: UUID) -> None:
    """
    Set current tenant ID in request context.

    Args:
        tenant_id: Tenant ID

    Example:
        >>> set_current_tenant(tenant.id)
    """
    current_tenant_id.set(tenant_id)


def get_current_tenant_id() -> Optional[UUID]:
    """
    Get current tenant ID from request context.

    Returns:
        Tenant ID or None

    Example:
        >>> tenant_id = get_current_tenant_id()
    """
    return current_tenant_id.get()


def set_current_session(session_id: str) -> None:
    """
    Set current session ID in request context.

    Args:
        session_id: Session ID

    Example:
        >>> set_current_session(session.id)
    """
    current_session_id.set(session_id)


def get_current_session_id() -> Optional[str]:
    """
    Get current session ID from request context.

    Returns:
        Session ID or None

    Example:
        >>> session_id = get_current_session_id()
    """
    return current_session_id.get()


def clear_context() -> None:
    """
    Clear all context variables.

    Should be called after request completes.

    Example:
        >>> clear_context()
    """
    current_user_id.set(None)
    current_tenant_id.set(None)
    current_session_id.set(None)


# =============================================================================
# POLICY CONTEXT BUILDERS
# =============================================================================


async def build_policy_context(
    request: Request,
    *,
    user_id: Optional[UUID] = None,
    tenant_id: Optional[UUID] = None,
    resource_type: str,
    resource_id: Optional[UUID] = None,
    action: str,
    resource_owner_id: Optional[UUID] = None,
    resource_team_id: Optional[UUID] = None,
    db: Optional[AsyncSession] = None,
) -> PolicyContext:
    """
    Build PolicyContext from HTTP request.

    Extracts context information from:
    - Request headers (user-agent)
    - Request client (IP address)
    - Context variables (user_id, tenant_id)
    - Function parameters

    Args:
        request: FastAPI Request object
        user_id: Override user ID (defaults to context)
        tenant_id: Override tenant ID (defaults to context)
        resource_type: Resource type
        resource_id: Resource ID
        action: Action being performed
        resource_owner_id: Resource owner ID
        resource_team_id: Resource team ID
        db: Database session (for fetching additional context)

    Returns:
        PolicyContext instance

    Raises:
        AuthenticationError: If user_id or tenant_id not available

    Example:
        >>> ctx = await build_policy_context(
        ...     request=request,
        ...     resource_type="document",
        ...     resource_id=doc_id,
        ...     action="read"
        ... )
    """
    # Get user_id and tenant_id from context or parameters
    user_id = user_id or get_current_user_id()
    tenant_id = tenant_id or get_current_tenant_id()

    if not user_id:
        raise AuthenticationError("User not authenticated")

    if not tenant_id:
        raise AuthenticationError("Tenant context not available")

    # Extract request metadata
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    session_id = get_current_session_id()

    # Build context
    return PolicyContext(
        user_id=user_id,
        tenant_id=tenant_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        ip_address=ip_address,
        user_agent=user_agent,
        session_id=session_id,
        resource_owner_id=resource_owner_id,
        resource_team_id=resource_team_id,
    )


async def build_policy_context_simple(
    *,
    user_id: UUID,
    tenant_id: UUID,
    resource_type: str,
    action: str,
    resource_id: Optional[UUID] = None,
) -> PolicyContext:
    """
    Build simple PolicyContext without HTTP request.

    Useful for background tasks, CLI commands, etc.

    Args:
        user_id: User ID
        tenant_id: Tenant ID
        resource_type: Resource type
        action: Action
        resource_id: Optional resource ID

    Returns:
        PolicyContext instance

    Example:
        >>> # In a Celery task
        >>> ctx = await build_policy_context_simple(
        ...     user_id=user_id,
        ...     tenant_id=tenant_id,
        ...     resource_type="document",
        ...     action="process"
        ... )
    """
    return PolicyContext(
        user_id=user_id,
        tenant_id=tenant_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
    )


# =============================================================================
# FASTAPI DEPENDENCIES
# =============================================================================


async def require_authentication(request: Request) -> UUID:
    """
    FastAPI dependency to require authentication.

    Returns user ID from context or raises AuthenticationError.

    Args:
        request: FastAPI Request

    Returns:
        User ID

    Raises:
        AuthenticationError: If not authenticated

    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(
        ...     user_id: UUID = Depends(require_authentication)
        ... ):
        ...     return {"user_id": str(user_id)}
    """
    user_id = get_current_user_id()

    if not user_id:
        raise AuthenticationError("Authentication required")

    return user_id


async def require_tenant(request: Request) -> UUID:
    """
    FastAPI dependency to require tenant context.

    Returns tenant ID from context or raises AuthenticationError.

    Args:
        request: FastAPI Request

    Returns:
        Tenant ID

    Raises:
        AuthenticationError: If tenant context not available

    Example:
        >>> @app.get("/tenant-resource")
        >>> async def tenant_route(
        ...     tenant_id: UUID = Depends(require_tenant)
        ... ):
        ...     return {"tenant_id": str(tenant_id)}
    """
    tenant_id = get_current_tenant_id()

    if not tenant_id:
        raise AuthenticationError("Tenant context required")

    return tenant_id
