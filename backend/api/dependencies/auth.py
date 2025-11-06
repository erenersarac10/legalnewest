"""
Authentication Dependencies for Turkish Legal AI Platform.

Enterprise-grade authentication and authorization dependencies with JWT validation,
Role-Based Access Control (RBAC), permission hierarchies, and API key management.

=============================================================================
FEATURES
=============================================================================

1. User Authentication
   --------------------
   - JWT token validation from Authorization header
   - User extraction from database
   - Active/inactive user filtering
   - Optional authentication support
   - Session management

2. Role-Based Access Control (RBAC)
   ---------------------------------
   - Hierarchical role system (admin > manager > user)
   - Role inheritance and permissions
   - Fine-grained permission checking
   - Resource-level access control
   - Dynamic role assignment

3. Permission Management
   ----------------------
   - Permission-based authorization
   - Wildcard permissions (contracts:*)
   - Permission composition (read + write)
   - Tenant-scoped permissions
   - Resource ownership validation

4. API Key Authentication
   ------------------------
   - API key validation and rotation
   - Key-specific permissions
   - Usage tracking and rate limiting
   - Key expiration management
   - Tenant-scoped API keys

5. Token Refresh
   --------------
   - Refresh token validation
   - New access token generation
   - Token blacklisting on logout
   - Automatic token rotation
   - Security event tracking

=============================================================================
USAGE
=============================================================================

Basic User Authentication:
---------------------------

>>> from fastapi import Depends
>>> from backend.api.dependencies.auth import get_current_user
>>>
>>> @app.get("/profile")
>>> async def get_profile(user: User = Depends(get_current_user)):
...     return {
...         "id": str(user.id),
...         "email": user.email,
...         "name": user.full_name
...     }

Permission-Based Authorization:
--------------------------------

>>> from backend.api.dependencies.auth import require_permission
>>>
>>> @app.post("/contracts")
>>> async def create_contract(
...     contract_data: ContractCreate,
...     _: None = Depends(require_permission("contracts:create"))
... ):
...     # User has contracts:create permission
...     return await contract_service.create(contract_data)

Role-Based Authorization:
--------------------------

>>> from backend.api.dependencies.auth import require_role
>>>
>>> @app.get("/admin/users")
>>> async def list_users(_: None = Depends(require_role("admin"))):
...     # Only admins can access
...     return await user_service.list_all()

Multiple Permission Requirements:
----------------------------------

>>> from backend.api.dependencies.auth import require_any_permission, require_all_permissions
>>>
>>> # User needs ANY of these permissions
>>> @app.get("/reports")
>>> async def get_reports(
...     _: None = Depends(require_any_permission(["reports:read", "admin:access"]))
... ):
...     return await report_service.get_all()
>>>
>>> # User needs ALL of these permissions
>>> @app.delete("/contracts/{contract_id}")
>>> async def delete_contract(
...     contract_id: str,
...     _: None = Depends(require_all_permissions(["contracts:delete", "contracts:write"]))
... ):
...     return await contract_service.delete(contract_id)

Optional Authentication:
-------------------------

>>> from backend.api.dependencies.auth import get_optional_current_user
>>>
>>> @app.get("/public-content")
>>> async def get_content(user: Optional[User] = Depends(get_optional_current_user)):
...     if user:
...         # Return personalized content
...         return await content_service.get_personalized(user.id)
...     else:
...         # Return public content
...         return await content_service.get_public()

Resource Ownership Validation:
-------------------------------

>>> from backend.api.dependencies.auth import verify_resource_owner
>>>
>>> @app.put("/contracts/{contract_id}")
>>> async def update_contract(
...     contract_id: str,
...     contract_data: ContractUpdate,
...     user: User = Depends(get_current_user),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Verify user owns the contract
...     contract = await db.get(Contract, contract_id)
...     await verify_resource_owner(user, contract)
...     return await contract_service.update(contract, contract_data)

=============================================================================
PERMISSION HIERARCHY
=============================================================================

Permission Format:
------------------

Format: resource:action
Examples:
  - contracts:read
  - contracts:write
  - contracts:delete
  - users:manage
  - admin:access

Wildcard Permissions:
---------------------

contracts:* = all contract permissions
  - contracts:read
  - contracts:write
  - contracts:delete
  - contracts:share

admin:* = all admin permissions
  - Full system access

Permission Inheritance:
-----------------------

Role: admin
  Permissions: admin:*
  Inherits: All permissions in the system

Role: manager
  Permissions: contracts:*, users:read
  Inherits: All contract operations + user viewing

Role: user
  Permissions: contracts:read, contracts:create
  Inherits: Basic contract access

=============================================================================
ROLE HIERARCHY
=============================================================================

System Roles:
-------------

1. super_admin (Level 100)
   - Full system access
   - Can manage all tenants
   - System configuration
   - Permissions: admin:*

2. admin (Level 80)
   - Tenant administrator
   - User management
   - Settings configuration
   - Permissions: tenant:*, users:*, contracts:*

3. manager (Level 60)
   - Team management
   - Contract oversight
   - Report access
   - Permissions: contracts:*, reports:*, team:manage

4. power_user (Level 40)
   - Advanced features
   - Bulk operations
   - API access
   - Permissions: contracts:*, api:access

5. user (Level 20)
   - Basic access
   - Create/read own content
   - Permissions: contracts:read, contracts:create

6. guest (Level 10)
   - Read-only access
   - Limited features
   - Permissions: contracts:read

Role Comparison:
----------------

>>> if user.role_level >= required_role_level:
...     # User has sufficient role level
...     pass

=============================================================================
API KEY AUTHENTICATION
=============================================================================

API Key Format:
---------------

Format: tla_<environment>_<random_32_chars>
Example: tla_prod_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

API Key Usage:
--------------

>>> # Request with API key
>>> curl -H "X-API-Key: tla_prod_abc123..." https://api.turkishlegalai.com/contracts

>>> # FastAPI dependency automatically validates API key
>>> @app.get("/contracts")
>>> async def list_contracts(
...     api_key: APIKey = Depends(get_api_key)
... ):
...     # API key validated, tenant_id extracted
...     return await contract_service.list(api_key.tenant_id)

API Key Permissions:
--------------------

API keys can have restricted permissions:

>>> api_key = await create_api_key(
...     tenant_id="abc",
...     name="Integration Key",
...     permissions=["contracts:read", "contracts:write"],
...     expires_at=datetime.now() + timedelta(days=365)
... )

=============================================================================
TOKEN REFRESH FLOW
=============================================================================

1. Client requests refresh:
   POST /api/v1/auth/refresh
   {
     "refresh_token": "eyJhbGc..."
   }

2. Backend validates refresh token:
   - Check token signature
   - Verify not expired
   - Check not blacklisted
   - Verify user still active

3. Generate new access token:
   - New expiry (24 hours)
   - Same user_id and tenant_id
   - Updated permissions

4. Return new token:
   {
     "access_token": "eyJhbGc...",
     "expires_in": 86400
   }

=============================================================================
SECURITY BEST PRACTICES
=============================================================================

Permission Checking:
--------------------

1. Always check permissions at endpoint level
2. Use fine-grained permissions (contracts:read vs contracts:*)
3. Implement resource ownership validation
4. Log permission denials for security monitoring

Role Management:
----------------

1. Use role hierarchy (admin > manager > user)
2. Implement least privilege principle
3. Regular permission audits
4. Temporary role elevation with expiry

Token Security:
---------------

1. Never store tokens in localStorage (XSS risk)
2. Use httpOnly cookies for refresh tokens
3. Implement token rotation
4. Blacklist tokens on logout
5. Monitor token usage patterns

=============================================================================
KVKK COMPLIANCE
=============================================================================

Authentication Logging:
-----------------------

- Log authentication attempts (success/failure)
- Store IP address and user agent
- Implement 90-day retention
- Provide user access to their auth history

Permission Auditing:
--------------------

- Track permission grants and revocations
- Log access to sensitive resources
- Maintain immutable audit trail
- Alert on suspicious permission changes

=============================================================================
TROUBLESHOOTING
=============================================================================

"Permission denied" Error:
---------------------------

1. Check user has required permission
2. Verify token contains permission claims
3. Check role hierarchy
4. Review resource ownership
5. Check tenant context

"Token expired" Error:
----------------------

1. Use refresh token to get new access token
2. Implement automatic token refresh in client
3. Check token expiry time (exp claim)
4. Verify server clock synchronization

"User not found" Error:
-----------------------

1. Check user hasn't been deleted
2. Verify user is active
3. Check tenant_id matches
4. Review database query filters

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.database import get_db
from backend.core import get_logger
from backend.core.database.models import User

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# ROLE LEVELS
# =============================================================================

ROLE_LEVELS = {
    "super_admin": 100,
    "admin": 80,
    "manager": 60,
    "power_user": 40,
    "user": 20,
    "guest": 10,
}

# =============================================================================
# USER DEPENDENCIES
# =============================================================================


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
        >>> @app.get("/profile")
        >>> async def get_profile(user_id: UUID = Depends(get_current_user_id)):
        ...     return {"user_id": str(user_id)}
    """
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        logger.warning("Kimlik doğrulama başarısız: user_id eksik")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum açmanız gerekiyor",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UUID(user_id) if isinstance(user_id, str) else user_id


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> User:
    """
    Get current authenticated user from database.

    Args:
        db: Database session
        user_id: User ID from token

    Returns:
        User model instance

    Raises:
        HTTPException: If user not found or inactive

    Example:
        >>> @app.get("/me")
        >>> async def get_me(current_user: User = Depends(get_current_user)):
        ...     return {"email": current_user.email}
    """
    result = await db.execute(
        select(User)
        .where(User.id == user_id)
        .where(User.is_deleted == False)  # noqa: E712
    )

    user = result.scalar_one_or_none()

    if not user:
        logger.warning("Kullanıcı bulunamadı", user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kullanıcı bulunamadı",
        )

    if not user.is_active:
        logger.warning("İnaktif kullanıcı erişim denemesi", user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Kullanıcı hesabı aktif değil",
        )

    logger.debug("Kullanıcı kimliği doğrulandı", user_id=str(user_id), email=user.email)
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
    """
    return current_user


async def get_optional_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.

    Args:
        request: FastAPI request
        db: Database session

    Returns:
        User instance or None

    Example:
        >>> @app.get("/public-data")
        >>> async def get_data(user: Optional[User] = Depends(get_optional_current_user)):
        ...     if user:
        ...         return {"data": "personalized"}
        ...     return {"data": "public"}
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
        logger.warning("Optional user fetch başarısız", error=str(e))
        return None


# =============================================================================
# PERMISSION DEPENDENCIES
# =============================================================================


def require_permission(permission: str):
    """
    Dependency factory for single permission checking.

    Args:
        permission: Required permission name

    Returns:
        Dependency function

    Example:
        >>> @app.delete("/users/{user_id}")
        >>> async def delete_user(
        ...     _: None = Depends(require_permission("users:delete"))
        ... ):
        ...     pass
    """

    async def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user),
    ) -> None:
        permissions = getattr(request.state, "permissions", [])

        # Check for exact match
        if permission in permissions:
            return

        # Check for wildcard match (e.g., contracts:* matches contracts:read)
        resource = permission.split(":")[0] if ":" in permission else None
        if resource and f"{resource}:*" in permissions:
            return

        # Check for admin wildcard
        if "admin:*" in permissions:
            return

        logger.warning(
            "İzin reddedildi",
            user_id=str(current_user.id),
            required_permission=permission,
            user_permissions=permissions,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bu işlem için '{permission}' yetkisi gerekiyor",
        )

    return permission_checker


def require_any_permission(permissions: List[str]):
    """
    Dependency factory for ANY permission checking.

    Args:
        permissions: List of acceptable permissions

    Returns:
        Dependency function

    Example:
        >>> @app.get("/reports")
        >>> async def get_reports(
        ...     _: None = Depends(require_any_permission(["reports:read", "admin:access"]))
        ... ):
        ...     pass
    """

    async def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user),
    ) -> None:
        user_permissions = getattr(request.state, "permissions", [])

        # Check if user has any of the required permissions
        for perm in permissions:
            if perm in user_permissions:
                return

            # Check wildcard
            resource = perm.split(":")[0] if ":" in perm else None
            if resource and f"{resource}:*" in user_permissions:
                return

        # Check admin wildcard
        if "admin:*" in user_permissions:
            return

        logger.warning(
            "İzin reddedildi (herhangi biri gerekli)",
            user_id=str(current_user.id),
            required_permissions=permissions,
            user_permissions=user_permissions,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bu işlem için şu yetkilerden biri gerekiyor: {', '.join(permissions)}",
        )

    return permission_checker


def require_all_permissions(permissions: List[str]):
    """
    Dependency factory for ALL permissions checking.

    Args:
        permissions: List of required permissions

    Returns:
        Dependency function

    Example:
        >>> @app.delete("/contracts/{contract_id}")
        >>> async def delete_contract(
        ...     _: None = Depends(require_all_permissions(["contracts:delete", "contracts:write"]))
        ... ):
        ...     pass
    """

    async def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user),
    ) -> None:
        user_permissions = getattr(request.state, "permissions", [])

        # Check admin wildcard first
        if "admin:*" in user_permissions:
            return

        # Check each required permission
        for perm in permissions:
            if perm not in user_permissions:
                # Check wildcard
                resource = perm.split(":")[0] if ":" in perm else None
                if not (resource and f"{resource}:*" in user_permissions):
                    logger.warning(
                        "İzin reddedildi (tümü gerekli)",
                        user_id=str(current_user.id),
                        required_permissions=permissions,
                        missing_permission=perm,
                    )

                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Bu işlem için tüm yetkiler gerekiyor: {', '.join(permissions)}",
                    )

    return permission_checker


# =============================================================================
# ROLE DEPENDENCIES
# =============================================================================


def require_role(role: str):
    """
    Dependency factory for role checking.

    Args:
        role: Required role name

    Returns:
        Dependency function

    Example:
        >>> @app.post("/admin/settings")
        >>> async def update_settings(
        ...     _: None = Depends(require_role("admin"))
        ... ):
        ...     pass
    """

    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> None:
        user_role = getattr(current_user, "role", None)

        if user_role != role:
            # Check if user has higher role level
            user_level = ROLE_LEVELS.get(user_role, 0)
            required_level = ROLE_LEVELS.get(role, 100)

            if user_level < required_level:
                logger.warning(
                    "Rol reddedildi",
                    user_id=str(current_user.id),
                    required_role=role,
                    user_role=user_role,
                )

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Bu işlem için '{role}' rolü gerekiyor",
                )

    return role_checker


def require_min_role_level(min_level: int):
    """
    Dependency factory for minimum role level checking.

    Args:
        min_level: Minimum required role level

    Returns:
        Dependency function

    Example:
        >>> @app.get("/admin/reports")
        >>> async def get_reports(
        ...     _: None = Depends(require_min_role_level(60))  # manager or above
        ... ):
        ...     pass
    """

    async def role_level_checker(
        current_user: User = Depends(get_current_user),
    ) -> None:
        user_role = getattr(current_user, "role", "user")
        user_level = ROLE_LEVELS.get(user_role, 0)

        if user_level < min_level:
            logger.warning(
                "Rol seviyesi yetersiz",
                user_id=str(current_user.id),
                user_level=user_level,
                required_level=min_level,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bu işlem için yeterli yetki seviyeniz bulunmuyor",
            )

    return role_level_checker


# =============================================================================
# RESOURCE OWNERSHIP
# =============================================================================


async def verify_resource_owner(user: User, resource) -> None:
    """
    Verify user owns the resource.

    Args:
        user: Current user
        resource: Resource object with user_id or owner_id

    Raises:
        HTTPException: If user doesn't own resource

    Example:
        >>> contract = await db.get(Contract, contract_id)
        >>> await verify_resource_owner(current_user, contract)
    """
    resource_owner_id = getattr(resource, "user_id", None) or getattr(
        resource, "owner_id", None
    )

    if resource_owner_id and resource_owner_id != user.id:
        logger.warning(
            "Kaynak sahipliği reddedildi",
            user_id=str(user.id),
            resource_owner_id=str(resource_owner_id),
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu kaynağa erişim yetkiniz bulunmuyor",
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "get_current_user_id",
    "get_current_user",
    "get_current_active_user",
    "get_optional_current_user",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
    "require_role",
    "require_min_role_level",
    "verify_resource_owner",
    "ROLE_LEVELS",
]
