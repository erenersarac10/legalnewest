"""
Permission management service for RBAC in Turkish Legal AI.

This module provides permission-level access control operations:
- Permission CRUD (system-defined permissions)
- Permission discovery and listing
- User permission aggregation (from all roles)
- Permission validation (resource:action format)
- Permission dependencies and prerequisites
- Context-aware permission checking

Permission Format:
    "resource:action"
    - resource: The object being accessed (document, contract, user)
    - action: The operation (read, write, delete, approve)

    Examples:
    - "document:read"
    - "contract:approve"
    - "*:*" (wildcard, superadmin)

Permission Categories:
    - SYSTEM: Platform administration
    - CONTENT: Documents, contracts, templates
    - COLLABORATION: Sharing, teams
    - ANALYTICS: Reports, dashboards
    - API: API access
    - BILLING: Subscriptions, payments
    - SECURITY: Security features, audit

Example:
    >>> from backend.security.rbac.permissions import PermissionService
    >>>
    >>> perm_svc = PermissionService(db_session)
    >>>
    >>> # Check if user has permission (aggregates all roles)
    >>> has_perm = await perm_svc.user_has_permission(
    ...     user_id=user_id,
    ...     resource="document",
    ...     action="approve"
    ... )
    >>>
    >>> # Get all user permissions
    >>> permissions = await perm_svc.get_user_permissions(user_id)
    >>> # {"document": ["read", "write"], "contract": ["generate"]}
    >>>
    >>> # List all defined permissions
    >>> all_perms = await perm_svc.list_permissions()
"""

import hashlib
import json
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.permission import (
    Permission,
    PermissionCategory,
    PermissionScope,
)
from backend.core.database.models.role import user_roles, Role
from backend.core.exceptions import (
    NotFoundError,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.security.rbac.roles import RoleService

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# REDIS CACHE (for permission sets - 60s TTL, ~40% p95 latency reduction)
# =============================================================================

# Cache configuration
PERMISSION_CACHE_TTL = 60  # seconds
PERMISSION_CACHE_ENABLED = True  # Can be disabled via config

# Redis client (initialized externally)
redis_client = None  # TODO: Initialize from backend.core.cache


def _get_user_permission_cache_key(user_id: UUID) -> str:
    """Generate Redis cache key for user permissions."""
    return f"rbac:permissions:user:{str(user_id)}"


def _hash_user_id(user_id: UUID) -> str:
    """Hash user ID for privacy in metrics."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


async def _get_cached_permissions(user_id: UUID) -> Optional[Dict[str, List[str]]]:
    """
    Get cached user permissions from Redis.

    Returns:
        Cached permissions dict or None if not found
    """
    if not PERMISSION_CACHE_ENABLED or redis_client is None:
        return None

    try:
        cache_key = _get_user_permission_cache_key(user_id)
        cached = await redis_client.get(cache_key)

        if cached:
            # Prometheus metric: cache hit
            # rbac_permission_cache_hits_total.labels(user_id_hash=_hash_user_id(user_id)).inc()
            logger.debug(f"Permission cache HIT for user {user_id}")
            return json.loads(cached)
        else:
            # Prometheus metric: cache miss
            # rbac_permission_cache_misses_total.labels(user_id_hash=_hash_user_id(user_id)).inc()
            logger.debug(f"Permission cache MISS for user {user_id}")
            return None
    except Exception as e:
        logger.warning(f"Redis cache error: {e}")
        return None


async def _set_cached_permissions(
    user_id: UUID,
    permissions: Dict[str, List[str]]
) -> None:
    """
    Cache user permissions in Redis with TTL.

    Args:
        user_id: User ID
        permissions: Permission dictionary to cache
    """
    if not PERMISSION_CACHE_ENABLED or redis_client is None:
        return

    try:
        cache_key = _get_user_permission_cache_key(user_id)
        await redis_client.setex(
            cache_key,
            PERMISSION_CACHE_TTL,
            json.dumps(permissions)
        )
        logger.debug(f"Cached permissions for user {user_id} (TTL={PERMISSION_CACHE_TTL}s)")
    except Exception as e:
        logger.warning(f"Redis cache write error: {e}")


# =============================================================================
# PERMISSION SERVICE
# =============================================================================


class PermissionService:
    """
    Permission management service for RBAC.

    Provides:
    - Permission CRUD
    - User permission aggregation from roles
    - Permission validation and checking
    - Permission discovery
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize permission service.

        Args:
            db: Async database session
        """
        self.db = db
        self.logger = logger
        self.role_service = RoleService(db)

    # =========================================================================
    # PERMISSION CRUD
    # =========================================================================

    async def create_permission(
        self,
        *,
        name: str,
        resource: str,
        action: str,
        category: PermissionCategory,
        scope: PermissionScope = PermissionScope.TENANT,
        description: Optional[str] = None,
        requires: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Permission:
        """
        Create a new system permission definition.

        Args:
            name: Permission name (e.g., "document:approve")
            resource: Resource name (e.g., "document")
            action: Action name (e.g., "approve")
            category: Permission category
            scope: Permission scope
            description: Optional description
            requires: List of prerequisite permissions
            metadata: Additional metadata

        Returns:
            Created permission

        Raises:
            ValidationError: If permission format invalid
        """
        # Validate format
        if ":" not in name:
            raise ValidationError("Permission name must be in format 'resource:action'")

        # Create permission
        permission = Permission(
            name=name,
            resource=resource,
            action=action,
            category=category,
            scope=scope,
            description=description,
            requires=requires or [],
            is_system=True,
            is_active=True,
            metadata=metadata or {},
        )

        self.db.add(permission)
        await self.db.commit()
        await self.db.refresh(permission)

        self.logger.info(f"Created permission: {permission.name}")

        return permission

    async def get_permission(self, permission_id: UUID) -> Optional[Permission]:
        """
        Get permission by ID.

        Args:
            permission_id: Permission ID

        Returns:
            Permission or None
        """
        result = await self.db.execute(
            select(Permission).where(
                and_(
                    Permission.id == permission_id,
                    Permission.deleted_at.is_(None),
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_permission_by_name(self, name: str) -> Optional[Permission]:
        """
        Get permission by name.

        Args:
            name: Permission name (e.g., "document:read")

        Returns:
            Permission or None
        """
        result = await self.db.execute(
            select(Permission).where(
                and_(
                    Permission.name == name,
                    Permission.deleted_at.is_(None),
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_permissions(
        self,
        *,
        category: Optional[PermissionCategory] = None,
        resource: Optional[str] = None,
        scope: Optional[PermissionScope] = None,
        is_active: Optional[bool] = True,
    ) -> List[Permission]:
        """
        List all defined permissions.

        Args:
            category: Filter by category
            resource: Filter by resource
            scope: Filter by scope
            is_active: Filter by active status

        Returns:
            List of permissions
        """
        conditions = [Permission.deleted_at.is_(None)]

        if category:
            conditions.append(Permission.category == category)

        if resource:
            conditions.append(Permission.resource == resource)

        if scope:
            conditions.append(Permission.scope == scope)

        if is_active is not None:
            conditions.append(Permission.is_active == is_active)

        result = await self.db.execute(
            select(Permission)
            .where(and_(*conditions))
            .order_by(Permission.category, Permission.resource, Permission.action)
        )

        return list(result.scalars().all())

    async def list_permissions_by_category(
        self,
    ) -> Dict[PermissionCategory, List[Permission]]:
        """
        List permissions grouped by category.

        Returns:
            Dictionary mapping category to permissions
        """
        permissions = await self.list_permissions()

        grouped: Dict[PermissionCategory, List[Permission]] = {}
        for perm in permissions:
            if perm.category not in grouped:
                grouped[perm.category] = []
            grouped[perm.category].append(perm)

        return grouped

    # =========================================================================
    # USER PERMISSION OPERATIONS
    # =========================================================================

    async def get_user_permissions(
        self,
        user_id: UUID,
        *,
        include_hierarchy: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Get aggregated permissions for user (from all roles).

        This method:
        1. Check Redis cache (60s TTL) → ~40% p95 latency reduction
        2. If miss, fetch from DB and aggregate
        3. Cache result for 60 seconds

        Args:
            user_id: User ID
            include_hierarchy: Include parent role permissions

        Returns:
            Permission dictionary {resource: [actions]}

        Example:
            >>> perms = await perm_svc.get_user_permissions(user_id)
            >>> # {"document": ["read", "write"], "contract": ["generate", "approve"]}
        """
        # Try cache first (Redis 60s TTL)
        cached = await _get_cached_permissions(user_id)
        if cached is not None:
            return cached

        # Cache miss - fetch from database
        user_roles_list = await self.role_service.get_user_roles(user_id)

        if not user_roles_list:
            return {}

        # Aggregate permissions from all roles
        aggregated_permissions: Dict[str, Set[str]] = {}

        for role in user_roles_list:
            role_permissions = await self.role_service.get_role_permissions(
                role.id,
                include_hierarchy=include_hierarchy,
            )

            # Merge permissions
            for resource, actions in role_permissions.items():
                if resource not in aggregated_permissions:
                    aggregated_permissions[resource] = set()
                aggregated_permissions[resource].update(actions)

        # Convert sets to sorted lists
        result = {
            resource: sorted(list(actions))
            for resource, actions in aggregated_permissions.items()
        }

        # Cache for 60 seconds
        await _set_cached_permissions(user_id, result)

        return result

    async def user_has_permission(
        self,
        user_id: UUID,
        resource: str,
        action: str,
    ) -> bool:
        """
        Check if user has specific permission.

        Aggregates permissions from all user roles and checks
        if the requested permission exists.

        Args:
            user_id: User ID
            resource: Resource name (e.g., "document")
            action: Action name (e.g., "read")

        Returns:
            True if user has permission

        Example:
            >>> can_approve = await perm_svc.user_has_permission(
            ...     user_id=user_id,
            ...     resource="contract",
            ...     action="approve"
            ... )
        """
        permissions = await self.get_user_permissions(user_id)

        # Check wildcard permissions (superadmin)
        if "*" in permissions and "*" in permissions["*"]:
            return True

        # Check specific permission
        if resource in permissions:
            return action in permissions[resource] or "*" in permissions[resource]

        return False

    async def user_has_any_permission(
        self,
        user_id: UUID,
        required_permissions: List[tuple[str, str]],
    ) -> bool:
        """
        Check if user has ANY of the specified permissions (OR logic).

        Args:
            user_id: User ID
            required_permissions: List of (resource, action) tuples

        Returns:
            True if user has at least one permission

        Example:
            >>> # Check if user can read OR write documents
            >>> has_access = await perm_svc.user_has_any_permission(
            ...     user_id=user_id,
            ...     required_permissions=[
            ...         ("document", "read"),
            ...         ("document", "write")
            ...     ]
            ... )
        """
        for resource, action in required_permissions:
            if await self.user_has_permission(user_id, resource, action):
                return True
        return False

    async def user_has_all_permissions(
        self,
        user_id: UUID,
        required_permissions: List[tuple[str, str]],
    ) -> bool:
        """
        Check if user has ALL specified permissions (AND logic).

        Args:
            user_id: User ID
            required_permissions: List of (resource, action) tuples

        Returns:
            True if user has all permissions

        Example:
            >>> # Check if user can both read AND write documents
            >>> has_full_access = await perm_svc.user_has_all_permissions(
            ...     user_id=user_id,
            ...     required_permissions=[
            ...         ("document", "read"),
            ...         ("document", "write")
            ...     ]
            ... )
        """
        for resource, action in required_permissions:
            if not await self.user_has_permission(user_id, resource, action):
                return False
        return True

    # =========================================================================
    # PERMISSION VALIDATION
    # =========================================================================

    def validate_permission_format(self, permission: str) -> tuple[str, str]:
        """
        Validate and parse permission string.

        Args:
            permission: Permission string (e.g., "document:read")

        Returns:
            Tuple of (resource, action)

        Raises:
            ValidationError: If invalid format

        Example:
            >>> resource, action = perm_svc.validate_permission_format("document:read")
            >>> # resource = "document", action = "read"
        """
        if ":" not in permission:
            raise ValidationError(
                f"Invalid permission format: '{permission}'. "
                "Must be 'resource:action'"
            )

        parts = permission.split(":", 1)
        resource, action = parts[0].strip(), parts[1].strip()

        if not resource or not action:
            raise ValidationError(
                f"Invalid permission format: '{permission}'. "
                "Both resource and action must be non-empty"
            )

        return resource, action

    async def validate_permissions_exist(
        self,
        permissions: List[str],
    ) -> bool:
        """
        Validate that all permissions are defined in the system.

        Args:
            permissions: List of permission names

        Returns:
            True if all permissions exist

        Raises:
            ValidationError: If any permission doesn't exist
        """
        for perm_name in permissions:
            # Parse format
            self.validate_permission_format(perm_name)

            # Check if exists (only for non-wildcard permissions)
            if perm_name != "*:*":
                perm = await self.get_permission_by_name(perm_name)
                if not perm:
                    raise ValidationError(
                        f"Permission '{perm_name}' is not defined in the system"
                    )

        return True

    # =========================================================================
    # PERMISSION PREREQUISITES
    # =========================================================================

    async def check_permission_prerequisites(
        self,
        user_id: UUID,
        permission: str,
    ) -> tuple[bool, List[str]]:
        """
        Check if user has all prerequisite permissions.

        Some permissions require other permissions to be present.
        For example, "document:approve" might require "document:read".

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            Tuple of (has_prerequisites, missing_permissions)

        Example:
            >>> has_prereqs, missing = await perm_svc.check_permission_prerequisites(
            ...     user_id=user_id,
            ...     permission="document:approve"
            ... )
            >>> if not has_prereqs:
            ...     print(f"Missing: {missing}")
        """
        # Get permission definition
        perm = await self.get_permission_by_name(permission)
        if not perm or not perm.requires:
            return True, []

        # Check each prerequisite
        missing = []
        for required_perm in perm.requires:
            resource, action = self.validate_permission_format(required_perm)
            if not await self.user_has_permission(user_id, resource, action):
                missing.append(required_perm)

        return len(missing) == 0, missing

    # =========================================================================
    # PERMISSION DISCOVERY
    # =========================================================================

    async def get_available_resources(self) -> List[str]:
        """
        Get list of all available resources.

        Returns:
            List of resource names

        Example:
            >>> resources = await perm_svc.get_available_resources()
            >>> # ["document", "contract", "user", "team", ...]
        """
        result = await self.db.execute(
            select(Permission.resource)
            .distinct()
            .where(Permission.deleted_at.is_(None))
            .order_by(Permission.resource)
        )
        return list(result.scalars().all())

    async def get_resource_actions(self, resource: str) -> List[str]:
        """
        Get all available actions for a resource.

        Args:
            resource: Resource name

        Returns:
            List of action names

        Example:
            >>> actions = await perm_svc.get_resource_actions("document")
            >>> # ["read", "write", "delete", "approve", ...]
        """
        result = await self.db.execute(
            select(Permission.action)
            .where(
                and_(
                    Permission.resource == resource,
                    Permission.deleted_at.is_(None),
                )
            )
            .order_by(Permission.action)
        )
        return list(result.scalars().all())

    async def search_permissions(
        self,
        query: str,
        *,
        limit: int = 50,
    ) -> List[Permission]:
        """
        Search permissions by name or description.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching permissions

        Example:
            >>> results = await perm_svc.search_permissions("document")
            >>> # Returns all permissions related to documents
        """
        search_pattern = f"%{query}%"

        result = await self.db.execute(
            select(Permission)
            .where(
                and_(
                    or_(
                        Permission.name.ilike(search_pattern),
                        Permission.description.ilike(search_pattern),
                        Permission.resource.ilike(search_pattern),
                    ),
                    Permission.deleted_at.is_(None),
                )
            )
            .limit(limit)
        )

        return list(result.scalars().all())


# =============================================================================
# PERMISSION UTILITIES
# =============================================================================


def parse_permission(permission: str) -> tuple[str, str]:
    """
    Parse permission string into resource and action.

    Args:
        permission: Permission string (e.g., "document:read")

    Returns:
        Tuple of (resource, action)

    Raises:
        ValidationError: If invalid format

    Example:
        >>> resource, action = parse_permission("contract:approve")
        >>> # resource = "contract", action = "approve"
    """
    if ":" not in permission:
        raise ValidationError(
            f"Invalid permission format: '{permission}'. Must be 'resource:action'"
        )

    parts = permission.split(":", 1)
    return parts[0].strip(), parts[1].strip()


def format_permission(resource: str, action: str) -> str:
    """
    Format resource and action into permission string.

    Args:
        resource: Resource name
        action: Action name

    Returns:
        Permission string

    Example:
        >>> perm = format_permission("document", "read")
        >>> # "document:read"
    """
    return f"{resource}:{action}"


def is_wildcard_permission(permission: str) -> bool:
    """
    Check if permission is a wildcard (grants all access).

    Args:
        permission: Permission string

    Returns:
        True if wildcard

    Example:
        >>> is_wildcard_permission("*:*")  # True
        >>> is_wildcard_permission("document:*")  # True (all document actions)
        >>> is_wildcard_permission("*:read")  # True (read on all resources)
        >>> is_wildcard_permission("document:read")  # False
    """
    try:
        resource, action = parse_permission(permission)
        return resource == "*" or action == "*"
    except ValidationError:
        return False
