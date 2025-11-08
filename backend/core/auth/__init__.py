"""
Authentication & Authorization Module - Harvey/Legora %100.

Production-ready RBAC system for Turkish Legal AI:
- Multi-tenant security
- Role-based access control
- JWT authentication
- Permission management
- Audit integration

Usage:
    >>> from backend.core.auth import RBACService, require_permission
    >>>
    >>> # Create service
    >>> rbac = RBACService(db_session, audit_service)
    >>>
    >>> # Check permission
    >>> allowed = await rbac.check_permission(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     permission="documents:read"
    ... )
    >>>
    >>> # Use decorator
    >>> @require_permission("documents:write")
    >>> async def create_document(...):
    ...     pass
"""

from backend.core.auth.models import (
    User,
    Role,
    Permission,
    UserRole,
    RolePermission,
    Tenant,
    TenantMembership,
    Session,
    UserStatusEnum,
    TenantStatusEnum,
    PermissionScopeEnum,
)
from backend.core.auth.service import RBACService
from backend.core.auth.middleware import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_current_tenant_id,
    get_current_user_and_tenant,
    require_permission,
    require_role,
    require_superadmin,
    AuthenticationMiddleware,
    TenantIsolationMiddleware,
)


__all__ = [
    # Models
    "User",
    "Role",
    "Permission",
    "UserRole",
    "RolePermission",
    "Tenant",
    "TenantMembership",
    "Session",
    "UserStatusEnum",
    "TenantStatusEnum",
    "PermissionScopeEnum",
    # Service
    "RBACService",
    # JWT
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    # Dependencies
    "get_current_user",
    "get_current_tenant_id",
    "get_current_user_and_tenant",
    # Decorators
    "require_permission",
    "require_role",
    "require_superadmin",
    # Middleware
    "AuthenticationMiddleware",
    "TenantIsolationMiddleware",
]
