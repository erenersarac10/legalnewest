"""
Role model for Role-Based Access Control (RBAC) in Turkish Legal AI.

This module provides the Role model for fine-grained permission management:
- Custom roles per tenant (organization-specific)
- Hierarchical role structure (role inheritance)
- Dynamic permission assignment
- System-defined vs custom roles
- Role templates for common use cases
- Audit trail for role changes

RBAC Architecture:
    User → Roles → Permissions → Resources
    
    A user can have multiple roles:
    - System role (LAWYER, ADMIN, etc.)
    - Custom roles (Senior Partner, Associate, Paralegal)
    
    Each role contains a set of permissions:
    - document:read, document:write
    - contract:generate, contract:approve
    - billing:view, billing:manage

Role Types:
    - SYSTEM: Built-in roles (cannot be deleted)
    - CUSTOM: Tenant-specific roles (can be modified)
    - TEMPLATE: Role templates for quick setup

Example:
    >>> # Create custom role
    >>> role = Role(
    ...     name="Senior Partner",
    ...     role_type=RoleType.CUSTOM,
    ...     tenant_id=tenant_id,
    ...     permissions={
    ...         "document": ["read", "write", "delete"],
    ...         "contract": ["generate", "approve", "sign"],
    ...         "billing": ["view", "manage"]
    ...     }
    ... )
    >>> 
    >>> # Assign to user
    >>> user.roles.append(role)
    >>> user.has_permission("contract:approve")  # True
"""

import datetime
import enum
from time import timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    CheckConstraint,
    Index,
    UniqueConstraint,
    Table,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import text

from backend.core.constants import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
)
from backend.core.exceptions import (
    PermissionDeniedError,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
    AuditMixin,
    SoftDeleteMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class RoleType(str, enum.Enum):
    """
    Role type classification.
    
    Types:
    - SYSTEM: Built-in system roles (cannot be deleted, tenant-agnostic)
    - CUSTOM: Tenant-specific custom roles (fully customizable)
    - TEMPLATE: Role templates for quick setup (can be copied)
    """
    
    SYSTEM = "system"
    CUSTOM = "custom"
    TEMPLATE = "template"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.SYSTEM: "Sistem Rolü",
            self.CUSTOM: "Özel Rol",
            self.TEMPLATE: "Şablon Rol",
        }
        return names.get(self, self.value)


class RoleScope(str, enum.Enum):
    """
    Role scope defines where the role can be applied.
    
    Scopes:
    - GLOBAL: Platform-wide (superadmin)
    - TENANT: Tenant-wide (all organizations)
    - ORGANIZATION: Organization-specific
    - TEAM: Team-specific
    """
    
    GLOBAL = "global"
    TENANT = "tenant"
    ORGANIZATION = "organization"
    TEAM = "team"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.GLOBAL: "Platform Geneli",
            self.TENANT: "Kuruluş Geneli",
            self.ORGANIZATION: "Organizasyon",
            self.TEAM: "Ekip",
        }
        return names.get(self, self.value)


# =============================================================================
# ASSOCIATION TABLE (Many-to-Many: User ↔ Role)
# =============================================================================

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        comment="Unique identifier",
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User ID",
    ),
    Column(
        "role_id",
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Role ID",
    ),
    Column(
        "granted_at",
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="When role was granted",
    ),
    Column(
        "granted_by_id",
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Who granted this role",
    ),
    Column(
        "expires_at",
        DateTime(timezone=True),
        nullable=True,
        comment="Role expiration (NULL = permanent)",
    ),
    Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (reason, conditions)",
    ),
    
    # Unique constraint: user can have each role only once
    UniqueConstraint("user_id", "role_id", name="uq_user_roles_user_role"),
    
    # Index for role assignments lookup
    Index("ix_user_roles_role_id", "role_id"),
    
    # Index for expiring roles
    Index(
        "ix_user_roles_expires",
        "expires_at",
        postgresql_where="expires_at IS NOT NULL",
    ),
)


# =============================================================================
# ROLE MODEL
# =============================================================================


class Role(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Role model for RBAC (Role-Based Access Control).
    
    Roles define sets of permissions that can be assigned to users:
    - System roles are built-in (ADMIN, LAWYER, etc.)
    - Custom roles are tenant-specific (Senior Partner, Associate)
    - Permissions are stored as JSON for flexibility
    - Supports role hierarchy (parent-child inheritance)
    
    Permission Format:
        {
            "resource": ["action1", "action2"],
            "document": ["read", "write", "delete"],
            "contract": ["generate", "approve"],
            "*": ["*"]  # Wildcard for all permissions
        }
    
    Attributes:
        name: Role name
        slug: URL-safe identifier
        role_type: Type of role (system, custom, template)
        role_scope: Where role applies (global, tenant, org, team)
        description: Role description
        
        permissions: JSON permission set
        parent_id: Parent role (for inheritance)
        parent: Parent role relationship
        children: Child roles
        
        is_system: System role (cannot be deleted)
        is_active: Role is active
        is_default: Auto-assign to new users
        
        priority: Role priority (higher = more privileged)
        user_count: Number of users with this role
        
        settings: JSON configuration
        metadata: Additional metadata
        
    Relationships:
        tenant: Parent tenant
        parent: Parent role (inheritance)
        children: Child roles
        users: Users with this role (many-to-many)
    """
    
    __tablename__ = "roles"
    
    # =========================================================================
    # IDENTITY
    # =========================================================================
    
    name = Column(
        String(MAX_NAME_LENGTH),
        nullable=False,
        comment="Role name (e.g., 'Senior Partner', 'Associate')",
    )
    
    slug = Column(
        String(100),
        nullable=False,
        index=True,
        comment="URL-safe identifier (unique within tenant)",
    )
    
    role_type = Column(
        Enum(RoleType, native_enum=False, length=50),
        nullable=False,
        default=RoleType.CUSTOM,
        index=True,
        comment="Role type (system, custom, template)",
    )
    
    role_scope = Column(
        Enum(RoleScope, native_enum=False, length=50),
        nullable=False,
        default=RoleScope.TENANT,
        comment="Role scope (global, tenant, organization, team)",
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Role description and purpose",
    )
    
    # =========================================================================
    # PERMISSIONS
    # =========================================================================
    
    permissions = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Permission set (resource:action mapping)",
    )
    
    # =========================================================================
    # HIERARCHY (Role Inheritance)
    # =========================================================================
    
    parent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Parent role (for permission inheritance)",
    )
    
    # Self-referential relationship
    parent = relationship(
        "Role",
        remote_side="Role.id",
        back_populates="children",
    )
    
    children = relationship(
        "Role",
        back_populates="parent",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    is_system = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="System role (cannot be deleted or renamed)",
    )
    
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Role is active and can be assigned",
    )
    
    is_default = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Auto-assign to new users in tenant",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    priority = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Role priority (higher = more privileged, used for conflict resolution)",
    )
    
    user_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of users with this role (denormalized)",
    )
    
    settings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Role-specific settings",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata",
    )
    
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    
    # Users relationship (many-to-many)
    users = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles",
        lazy="dynamic",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Unique slug within tenant (allow same slug across tenants)
        UniqueConstraint(
            "tenant_id",
            "slug",
            name="uq_roles_tenant_slug",
        ),
        # Index for system roles
        Index(
            "ix_roles_system",
            "is_system",
            postgresql_where="is_system = true",
        ),
        # Index for active roles
        Index(
            "ix_roles_active",
            "tenant_id",
            "is_active",
            postgresql_where="is_active = true AND deleted_at IS NULL",
        ),
        # Index for default roles
        Index(
            "ix_roles_default",
            "tenant_id",
            "is_default",
            postgresql_where="is_default = true",
        ),
        # Check: priority non-negative
        CheckConstraint(
            "priority >= 0",
            name="ck_roles_priority_positive",
        ),
        # Check: user_count non-negative
        CheckConstraint(
            "user_count >= 0",
            name="ck_roles_user_count_positive",
        ),
    )
    
    # =========================================================================
    # PERMISSION MANAGEMENT
    # =========================================================================
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if role has specific permission.
        
        Supports:
        - Exact match: "document:read"
        - Wildcard resource: "document:*"
        - Wildcard all: "*:*" or {"*": ["*"]}
        - Parent inheritance
        
        Args:
            permission: Permission string (format: "resource:action")
            
        Returns:
            bool: True if role has permission
            
        Example:
            >>> role.has_permission("document:read")
            True
            >>> role.has_permission("contract:approve")
            False
        """
        # Parse permission
        if ":" not in permission:
            logger.warning(
                "Invalid permission format (expected 'resource:action')",
                permission=permission,
            )
            return False
        
        resource, action = permission.split(":", 1)
        
        # Check current role permissions
        if self._check_permission_in_set(resource, action, self.permissions):
            return True
        
        # Check parent role (inheritance)
        if self.parent:
            return self.parent.has_permission(permission)
        
        return False
    
    def _check_permission_in_set(
        self,
        resource: str,
        action: str,
        permission_set: dict[str, list[str]],
    ) -> bool:
        """
        Check permission in a permission set.
        
        Args:
            resource: Resource name (e.g., "document")
            action: Action name (e.g., "read")
            permission_set: Permission dictionary
            
        Returns:
            bool: True if permission exists
        """
        # Wildcard all
        if "*" in permission_set and "*" in permission_set["*"]:
            return True
        
        # Exact resource match
        if resource in permission_set:
            actions = permission_set[resource]
            
            # Wildcard action
            if "*" in actions:
                return True
            
            # Exact action match
            if action in actions:
                return True
        
        # Wildcard resource
        if "*" in permission_set:
            actions = permission_set["*"]
            if action in actions or "*" in actions:
                return True
        
        return False
    
    def get_all_permissions(self) -> dict[str, list[str]]:
        """
        Get all permissions including inherited from parent.
        
        Returns:
            dict: Combined permission set
            
        Example:
            >>> permissions = role.get_all_permissions()
            >>> print(permissions)
            {
                "document": ["read", "write"],
                "contract": ["generate"]
            }
        """
        # Start with current permissions
        all_perms = dict(self.permissions)
        
        # Merge parent permissions (recursive)
        if self.parent:
            parent_perms = self.parent.get_all_permissions()
            
            for resource, actions in parent_perms.items():
                if resource in all_perms:
                    # Merge actions (remove duplicates)
                    all_perms[resource] = list(
                        set(all_perms[resource] + actions)
                    )
                else:
                    all_perms[resource] = actions
        
        return all_perms
    
    def add_permission(self, resource: str, action: str) -> None:
        """
        Add a permission to this role.
        
        Args:
            resource: Resource name (e.g., "document")
            action: Action name (e.g., "read")
            
        Example:
            >>> role.add_permission("document", "delete")
        """
        if resource not in self.permissions:
            self.permissions[resource] = []
        
        if action not in self.permissions[resource]:
            self.permissions[resource].append(action)
            
            # Mark as modified (SQLAlchemy JSON tracking)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "permissions")
            
            logger.info(
                "Permission added to role",
                role_id=str(self.id),
                role_name=self.name,
                resource=resource,
                action=action,
            )
    
    def remove_permission(self, resource: str, action: str) -> None:
        """
        Remove a permission from this role.
        
        Args:
            resource: Resource name
            action: Action name
            
        Example:
            >>> role.remove_permission("document", "delete")
        """
        if resource in self.permissions and action in self.permissions[resource]:
            self.permissions[resource].remove(action)
            
            # Remove resource key if no actions left
            if not self.permissions[resource]:
                del self.permissions[resource]
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "permissions")
            
            logger.info(
                "Permission removed from role",
                role_id=str(self.id),
                role_name=self.name,
                resource=resource,
                action=action,
            )
    
    def set_permissions(self, permissions: dict[str, list[str]]) -> None:
        """
        Set all permissions (replaces existing).
        
        Args:
            permissions: New permission set
            
        Example:
            >>> role.set_permissions({
            ...     "document": ["read", "write"],
            ...     "contract": ["generate"]
            ... })
        """
        self.permissions = permissions
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "permissions")
        
        logger.info(
            "Permissions set for role",
            role_id=str(self.id),
            role_name=self.name,
            permission_count=sum(len(actions) for actions in permissions.values()),
        )
    
    # =========================================================================
    # USER ASSIGNMENT
    # =========================================================================
    
    def assign_to_user(self, user_id: str, granted_by_id: str | None = None) -> None:
        """
        Assign this role to a user.
        
        Args:
            user_id: User UUID
            granted_by_id: Who granted this role (optional)
            
        Example:
            >>> role.assign_to_user(
            ...     str(user.id),
            ...     granted_by_id=str(admin.id)
            ... )
        """
        self.user_count += 1
        
        logger.info(
            "Role assigned to user",
            role_id=str(self.id),
            role_name=self.name,
            user_id=user_id,
            granted_by=granted_by_id,
        )
    
    def revoke_from_user(self, user_id: str) -> None:
        """
        Revoke this role from a user.
        
        Args:
            user_id: User UUID
            
        Example:
            >>> role.revoke_from_user(str(user.id))
        """
        self.user_count = max(0, self.user_count - 1)
        
        logger.info(
            "Role revoked from user",
            role_id=str(self.id),
            role_name=self.name,
            user_id=user_id,
        )
    
    # =========================================================================
    # SYSTEM ROLE PROTECTION
    # =========================================================================
    
    def can_delete(self) -> bool:
        """
        Check if role can be deleted.
        
        System roles cannot be deleted.
        
        Returns:
            bool: True if deletable
        """
        return not self.is_system
    
    def can_modify(self) -> bool:
        """
        Check if role can be modified.
        
        System roles have restricted modifications.
        
        Returns:
            bool: True if modifiable
        """
        return not self.is_system
    
    def require_modifiable(self) -> None:
        """
        Require role to be modifiable (raises if not).
        
        Raises:
            PermissionDeniedError: If role is system role
        """
        if not self.can_modify():
            raise PermissionDeniedError(
                message="Sistem rolleri değiştirilemez",
                resource="role",
                action="modify",
            )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("slug")
    def validate_slug(self, key: str, slug: str) -> str:
        """Validate slug format (URL-safe)."""
        import re
        
        slug_pattern = r"^[a-z0-9-]+$"
        if not re.match(slug_pattern, slug):
            raise ValidationError(
                message="Slug sadece küçük harf, rakam ve tire içerebilir",
                field="slug",
            )
        
        return slug
    
    @validates("parent_id")
    def validate_parent_id(self, key: str, parent_id: str | None) -> str | None:
        """Validate parent_id (prevent circular references)."""
        if parent_id and str(parent_id) == str(self.id):
            raise ValidationError(
                message="Rol kendisinin üst rolü olamaz",
                field="parent_id",
            )
        
        return parent_id
    
    @validates("permissions")
    def validate_permissions(
        self,
        key: str,
        permissions: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Validate permission format."""
        if not isinstance(permissions, dict):
            raise ValidationError(
                message="Permissions dict olmalıdır",
                field="permissions",
            )
        
        for resource, actions in permissions.items():
            if not isinstance(actions, list):
                raise ValidationError(
                    message=f"Actions for '{resource}' must be a list",
                    field="permissions",
                )
            
            if not all(isinstance(action, str) for action in actions):
                raise ValidationError(
                    message=f"All actions for '{resource}' must be strings",
                    field="permissions",
                )
        
        return permissions
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    @staticmethod
    def generate_slug(name: str) -> str:
        """
        Generate URL-safe slug from role name.
        
        Args:
            name: Role name
            
        Returns:
            str: URL-safe slug
        """
        import re
        import unicodedata
        
        slug = name.lower()
        
        # Turkish character replacements
        replacements = {
            'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
            'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c',
        }
        for tr_char, en_char in replacements.items():
            slug = slug.replace(tr_char, en_char)
        
        # Remove accents
        slug = unicodedata.normalize('NFKD', slug)
        slug = slug.encode('ascii', 'ignore').decode('ascii')
        
        # Replace non-alphanumeric with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        slug = re.sub(r'-+', '-', slug)
        
        return slug
    
    @staticmethod
    def create_system_roles() -> list["Role"]:
        """
        Create default system roles for a new tenant.
        
        Returns:
            list: List of system roles
            
        Example:
            >>> roles = Role.create_system_roles()
            >>> for role in roles:
            ...     db.add(role)
        """
        system_roles = [
            {
                "name": "Superadmin",
                "slug": "superadmin",
                "role_scope": RoleScope.GLOBAL,
                "priority": 1000,
                "permissions": {"*": ["*"]},  # All permissions
                "description": "Platform yöneticisi (tüm yetkiler)",
            },
            {
                "name": "Admin",
                "slug": "admin",
                "role_scope": RoleScope.TENANT,
                "priority": 900,
                "permissions": {
                    "user": ["read", "write", "delete"],
                    "organization": ["read", "write", "delete"],
                    "team": ["read", "write", "delete"],
                    "document": ["read", "write", "delete"],
                    "analytics": ["read"],
                },
                "description": "Kuruluş yöneticisi",
            },
            {
                "name": "Lawyer",
                "slug": "lawyer",
                "role_scope": RoleScope.TENANT,
                "priority": 500,
                "permissions": {
                    "document": ["read", "write"],
                    "contract": ["generate", "analyze"],
                    "chat": ["read", "write"],
                    "legal_research": ["read"],
                },
                "description": "Avukat",
            },
            {
                "name": "Paralegal",
                "slug": "paralegal",
                "role_scope": RoleScope.TENANT,
                "priority": 300,
                "permissions": {
                    "document": ["read", "write"],
                    "chat": ["read", "write"],
                },
                "description": "Hukuk asistanı",
            },
            {
                "name": "Guest",
                "slug": "guest",
                "role_scope": RoleScope.TENANT,
                "priority": 100,
                "permissions": {
                    "document": ["read"],
                    "chat": ["read"],
                },
                "description": "Misafir kullanıcı (sadece görüntüleme)",
            },
        ]
        
        # Note: Actual Role objects would be created in a service/migration
        # This is a template definition
        return system_roles
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Role(id={self.id}, name={self.name}, users={self.user_count})>"
    
    def to_dict(self, include_permissions: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_permissions: Include full permission set
            
        Returns:
            dict: Role data
        """
        data = super().to_dict()
        
        # Add computed fields
        data["type_display"] = self.role_type.display_name_tr
        data["scope_display"] = self.role_scope.display_name_tr
        data["can_delete"] = self.can_delete()
        data["can_modify"] = self.can_modify()
        
        if include_permissions:
            data["all_permissions"] = self.get_all_permissions()
        else:
            data.pop("permissions", None)
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Role",
    "RoleType",
    "RoleScope",
    "user_roles",
]