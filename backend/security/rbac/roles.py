"""
Role management service for RBAC in Turkish Legal AI.

This module provides comprehensive role management operations:
- Role CRUD (create, read, update, delete)
- Role assignment to users (with expiration support)
- Role hierarchy and permission inheritance
- System role vs custom role handling
- Tenant-scoped role management
- Role cloning from templates
- Bulk operations

Architecture:
    User → UserRole (association) → Role → Permissions (JSONB)

    Role Types:
    - SYSTEM: Built-in roles (immutable, cannot be deleted)
    - CUSTOM: Tenant-specific roles (fully customizable)
    - TEMPLATE: Role templates for quick setup

    Role Hierarchy:
    - Roles can inherit from parent roles
    - Child roles inherit all parent permissions
    - Permission aggregation from entire hierarchy chain

Example:
    >>> from backend.security.rbac.roles import RoleService
    >>>
    >>> # Create role service
    >>> role_svc = RoleService(db_session)
    >>>
    >>> # Create custom role
    >>> role = await role_svc.create_role(
    ...     tenant_id=tenant_id,
    ...     name="Senior Partner",
    ...     permissions={
    ...         "document": ["read", "write", "delete"],
    ...         "contract": ["generate", "approve", "sign"]
    ...     }
    ... )
    >>>
    >>> # Assign role to user
    >>> await role_svc.assign_role(
    ...     user_id=user_id,
    ...     role_id=role.id,
    ...     granted_by_id=admin_id,
    ...     expires_at=None  # Permanent
    ... )
    >>>
    >>> # Get aggregated permissions (including hierarchy)
    >>> permissions = await role_svc.get_role_permissions(role.id)
"""

import datetime
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

from sqlalchemy import and_, or_, select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from backend.core.database.models.role import (
    Role,
    RoleType,
    RoleScope,
    user_roles,
)
from backend.core.exceptions import (
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
    ConflictError,
)
from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ROLE SERVICE
# =============================================================================


class RoleService:
    """
    Role management service for RBAC.

    Provides comprehensive role operations including:
    - CRUD operations for roles
    - Role assignment/revocation
    - Permission aggregation with hierarchy
    - System vs custom role handling
    - Tenant isolation
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize role service.

        Args:
            db: Async database session
        """
        self.db = db
        self.logger = logger

    # =========================================================================
    # ROLE CRUD OPERATIONS
    # =========================================================================

    async def create_role(
        self,
        *,
        tenant_id: UUID,
        name: str,
        slug: str,
        permissions: Dict[str, List[str]],
        description: Optional[str] = None,
        role_type: RoleType = RoleType.CUSTOM,
        role_scope: RoleScope = RoleScope.TENANT,
        parent_id: Optional[UUID] = None,
        is_default: bool = False,
        priority: int = 0,
        settings: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Role:
        """
        Create a new role.

        Args:
            tenant_id: Tenant ID
            name: Role name
            slug: URL-safe identifier
            permissions: Permission dictionary {resource: [actions]}
            description: Optional description
            role_type: Role type (SYSTEM, CUSTOM, TEMPLATE)
            role_scope: Role scope (GLOBAL, TENANT, etc.)
            parent_id: Parent role ID for inheritance
            is_default: Auto-assign to new users
            priority: Role priority (higher = more privileged)
            settings: Additional settings
            metadata: Additional metadata

        Returns:
            Created role

        Raises:
            ConflictError: If role slug exists
            ValidationError: If invalid parameters
        """
        # Check if slug exists within tenant
        existing = await self.db.execute(
            select(Role).where(
                and_(
                    Role.tenant_id == tenant_id,
                    Role.slug == slug,
                    Role.deleted_at.is_(None),
                )
            )
        )
        if existing.scalar_one_or_none():
            raise ConflictError(f"Role with slug '{slug}' already exists")

        # Validate parent role if specified
        if parent_id:
            parent = await self.get_role(parent_id)
            if not parent or parent.tenant_id != tenant_id:
                raise ValidationError("Invalid parent role")

        # Create role
        role = Role(
            tenant_id=tenant_id,
            name=name,
            slug=slug,
            description=description,
            role_type=role_type,
            role_scope=role_scope,
            permissions=permissions,
            parent_id=parent_id,
            is_system=role_type == RoleType.SYSTEM,
            is_default=is_default,
            is_active=True,
            priority=priority,
            settings=settings or {},
            metadata=metadata or {},
        )

        self.db.add(role)
        await self.db.commit()
        await self.db.refresh(role)

        self.logger.info(
            f"Created role: {role.name} (slug={role.slug}, "
            f"type={role.role_type}, tenant={tenant_id})"
        )

        # Prometheus metric: role creation
        # rbac_role_created_total.labels(role_type=role_type, tenant_id=str(tenant_id)).inc()

        return role

    async def get_role(
        self,
        role_id: UUID,
        *,
        load_relationships: bool = False,
    ) -> Optional[Role]:
        """
        Get role by ID.

        Args:
            role_id: Role ID
            load_relationships: Load parent/children relationships

        Returns:
            Role or None if not found
        """
        query = select(Role).where(
            and_(
                Role.id == role_id,
                Role.deleted_at.is_(None),
            )
        )

        if load_relationships:
            query = query.options(
                selectinload(Role.parent),
                selectinload(Role.children),
            )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_role_by_slug(
        self,
        tenant_id: UUID,
        slug: str,
    ) -> Optional[Role]:
        """
        Get role by slug within tenant.

        Args:
            tenant_id: Tenant ID
            slug: Role slug

        Returns:
            Role or None
        """
        result = await self.db.execute(
            select(Role).where(
                and_(
                    Role.tenant_id == tenant_id,
                    Role.slug == slug,
                    Role.deleted_at.is_(None),
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_roles(
        self,
        tenant_id: UUID,
        *,
        role_type: Optional[RoleType] = None,
        is_active: Optional[bool] = None,
        include_system: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Role]:
        """
        List roles for tenant.

        Args:
            tenant_id: Tenant ID
            role_type: Filter by role type
            is_active: Filter by active status
            include_system: Include system roles
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            List of roles
        """
        conditions = [Role.deleted_at.is_(None)]

        # Tenant filter (system roles are tenant-agnostic)
        if include_system:
            conditions.append(
                or_(
                    Role.tenant_id == tenant_id,
                    Role.role_type == RoleType.SYSTEM,
                )
            )
        else:
            conditions.append(Role.tenant_id == tenant_id)

        if role_type:
            conditions.append(Role.role_type == role_type)

        if is_active is not None:
            conditions.append(Role.is_active == is_active)

        result = await self.db.execute(
            select(Role)
            .where(and_(*conditions))
            .order_by(Role.priority.desc(), Role.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        return list(result.scalars().all())

    async def update_role(
        self,
        role_id: UUID,
        **updates: Any,
    ) -> Role:
        """
        Update role.

        Args:
            role_id: Role ID
            **updates: Fields to update

        Returns:
            Updated role

        Raises:
            NotFoundError: If role not found
            PermissionDeniedError: If trying to modify system role
        """
        role = await self.get_role(role_id)
        if not role:
            raise NotFoundError(f"Role {role_id} not found")

        # System roles cannot be modified
        if role.is_system:
            raise PermissionDeniedError("Cannot modify system role")

        # Update fields
        for key, value in updates.items():
            if hasattr(role, key):
                setattr(role, key, value)

        await self.db.commit()
        await self.db.refresh(role)

        self.logger.info(f"Updated role: {role.name} (id={role_id})")

        return role

    async def delete_role(
        self,
        role_id: UUID,
        *,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete role (soft or hard delete).

        Args:
            role_id: Role ID
            hard_delete: Permanently delete (default: soft delete)

        Returns:
            True if deleted

        Raises:
            NotFoundError: If role not found
            PermissionDeniedError: If trying to delete system role
        """
        role = await self.get_role(role_id)
        if not role:
            raise NotFoundError(f"Role {role_id} not found")

        # System roles cannot be deleted
        if role.is_system:
            raise PermissionDeniedError("Cannot delete system role")

        if hard_delete:
            # Revoke all user assignments first
            await self.db.execute(
                delete(user_roles).where(user_roles.c.role_id == role_id)
            )
            await self.db.delete(role)
        else:
            # Soft delete
            role.deleted_at = datetime.datetime.utcnow()

        await self.db.commit()

        self.logger.info(
            f"Deleted role: {role.name} (id={role_id}, hard={hard_delete})"
        )

        # Prometheus metric: role deletion
        deletion_type = "hard" if hard_delete else "soft"
        # rbac_role_deleted_total.labels(
        #     role_type=str(role.role_type),
        #     tenant_id=str(role.tenant_id),
        #     deletion_type=deletion_type
        # ).inc()

        return True

    # =========================================================================
    # ROLE ASSIGNMENT OPERATIONS
    # =========================================================================

    async def assign_role(
        self,
        user_id: UUID,
        role_id: UUID,
        *,
        granted_by_id: Optional[UUID] = None,
        expires_at: Optional[datetime.datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Assign role to user.

        Args:
            user_id: User ID
            role_id: Role ID
            granted_by_id: Who granted this role
            expires_at: When role expires (None = permanent)
            metadata: Additional metadata (reason, conditions)

        Returns:
            True if assigned

        Raises:
            NotFoundError: If role not found
            ConflictError: If user already has this role
        """
        # Validate role exists
        role = await self.get_role(role_id)
        if not role:
            raise NotFoundError(f"Role {role_id} not found")

        # Check if already assigned
        existing = await self.db.execute(
            select(user_roles).where(
                and_(
                    user_roles.c.user_id == user_id,
                    user_roles.c.role_id == role_id,
                )
            )
        )
        if existing.scalar_one_or_none():
            raise ConflictError("User already has this role")

        # Insert assignment
        await self.db.execute(
            user_roles.insert().values(
                user_id=user_id,
                role_id=role_id,
                granted_at=datetime.datetime.utcnow(),
                granted_by_id=granted_by_id,
                expires_at=expires_at,
                metadata=metadata or {},
            )
        )

        # Update user count
        await self.db.execute(
            update(Role)
            .where(Role.id == role_id)
            .values(user_count=Role.user_count + 1)
        )

        await self.db.commit()

        self.logger.info(
            f"Assigned role {role.name} to user {user_id} "
            f"(expires={expires_at})"
        )

        # Prometheus metric: role assignment
        # rbac_role_assignment_total.labels(
        #     role_slug=role.slug,
        #     role_type=str(role.role_type),
        #     tenant_id=str(role.tenant_id)
        # ).inc()

        return True

    async def revoke_role(
        self,
        user_id: UUID,
        role_id: UUID,
    ) -> bool:
        """
        Revoke role from user.

        Args:
            user_id: User ID
            role_id: Role ID

        Returns:
            True if revoked
        """
        # Get role for metrics (before deletion)
        role = await self.get_role(role_id)

        # Delete assignment
        result = await self.db.execute(
            delete(user_roles).where(
                and_(
                    user_roles.c.user_id == user_id,
                    user_roles.c.role_id == role_id,
                )
            )
        )

        if result.rowcount == 0:
            return False

        # Update user count
        await self.db.execute(
            update(Role)
            .where(Role.id == role_id)
            .values(user_count=Role.user_count - 1)
        )

        await self.db.commit()

        self.logger.info(f"Revoked role {role_id} from user {user_id}")

        # Prometheus metric: role revocation
        if role:
            # rbac_role_revocation_total.labels(
            #     role_slug=role.slug,
            #     role_type=str(role.role_type),
            #     tenant_id=str(role.tenant_id)
            # ).inc()
            pass

        return True

    async def get_user_roles(
        self,
        user_id: UUID,
        *,
        include_expired: bool = False,
    ) -> List[Role]:
        """
        Get all roles assigned to user.

        Args:
            user_id: User ID
            include_expired: Include expired roles

        Returns:
            List of roles
        """
        conditions = [user_roles.c.user_id == user_id]

        if not include_expired:
            conditions.append(
                or_(
                    user_roles.c.expires_at.is_(None),
                    user_roles.c.expires_at > datetime.datetime.utcnow(),
                )
            )

        result = await self.db.execute(
            select(Role)
            .select_from(user_roles)
            .join(Role, Role.id == user_roles.c.role_id)
            .where(
                and_(
                    *conditions,
                    Role.deleted_at.is_(None),
                    Role.is_active.is_(True),
                )
            )
            .order_by(Role.priority.desc())
        )

        return list(result.scalars().all())

    # =========================================================================
    # PERMISSION OPERATIONS
    # =========================================================================

    async def get_role_permissions(
        self,
        role_id: UUID,
        *,
        include_hierarchy: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Get aggregated permissions for role (including hierarchy).

        Args:
            role_id: Role ID
            include_hierarchy: Include parent role permissions

        Returns:
            Permission dictionary {resource: [actions]}
        """
        permissions: Dict[str, Set[str]] = {}

        # Get role with parent hierarchy
        role = await self.get_role(role_id, load_relationships=True)
        if not role:
            return {}

        # Collect permissions from hierarchy (parent first, child overrides)
        roles_to_process = []

        if include_hierarchy:
            # Walk up the parent chain
            current = role
            while current:
                roles_to_process.insert(0, current)  # Prepend parent
                if current.parent_id:
                    current = await self.get_role(current.parent_id)
                else:
                    break
        else:
            roles_to_process = [role]

        # Aggregate permissions
        for r in roles_to_process:
            for resource, actions in r.permissions.items():
                if resource not in permissions:
                    permissions[resource] = set()
                permissions[resource].update(actions)

        # Convert sets to lists
        return {
            resource: sorted(list(actions))
            for resource, actions in permissions.items()
        }

    async def has_permission(
        self,
        role_id: UUID,
        resource: str,
        action: str,
    ) -> bool:
        """
        Check if role has specific permission.

        Args:
            role_id: Role ID
            resource: Resource name (e.g., "document")
            action: Action name (e.g., "read")

        Returns:
            True if role has permission
        """
        permissions = await self.get_role_permissions(role_id)

        # Check wildcard permissions
        if "*" in permissions and "*" in permissions["*"]:
            return True

        if resource in permissions:
            return action in permissions[resource] or "*" in permissions[resource]

        return False

    # =========================================================================
    # ROLE TEMPLATES
    # =========================================================================

    async def clone_from_template(
        self,
        template_id: UUID,
        tenant_id: UUID,
        *,
        name: str,
        slug: str,
        description: Optional[str] = None,
    ) -> Role:
        """
        Clone role from template.

        Args:
            template_id: Template role ID
            tenant_id: Target tenant ID
            name: New role name
            slug: New role slug
            description: Optional description

        Returns:
            Cloned role

        Raises:
            NotFoundError: If template not found
            ValidationError: If template is not a template type
        """
        template = await self.get_role(template_id)
        if not template:
            raise NotFoundError(f"Template {template_id} not found")

        if template.role_type != RoleType.TEMPLATE:
            raise ValidationError("Role is not a template")

        # Clone role
        return await self.create_role(
            tenant_id=tenant_id,
            name=name,
            slug=slug,
            description=description or template.description,
            role_type=RoleType.CUSTOM,
            role_scope=template.role_scope,
            permissions=template.permissions.copy(),
            priority=template.priority,
            settings=template.settings.copy() if template.settings else {},
        )
