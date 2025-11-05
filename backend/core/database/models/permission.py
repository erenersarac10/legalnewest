"""
Permission model for fine-grained access control in Turkish Legal AI.

This module provides the Permission model for atomic authorization units:
- Granular permission definitions (resource:action pairs)
- Permission categories and grouping
- Dynamic permission discovery
- Permission dependencies and prerequisites
- Audit trail for permission usage
- Support for conditional permissions (context-based)

Permission Architecture:
    Permission = Atomic unit of access (e.g., "document:read")
    Role = Collection of permissions
    User = Has multiple roles → Inherits all permissions
    
    Permission Format: "resource:action"
    - resource: The object being accessed (document, contract, user)
    - action: The operation (read, write, delete, approve)

Permission Categories:
    - SYSTEM: Platform-level permissions (user management, billing)
    - CONTENT: Content-related (documents, contracts, templates)
    - COLLABORATION: Team/sharing permissions
    - ANALYTICS: Reporting and analytics
    - API: API access permissions

Permission Context:
    Some permissions may be context-dependent:
    - document:read → Requires document ownership OR team membership
    - contract:approve → Requires senior role AND document completion
    - billing:manage → Requires admin role AND billing feature enabled

Example:
    >>> # Define permission
    >>> perm = Permission(
    ...     name="document:approve",
    ...     resource="document",
    ...     action="approve",
    ...     category=PermissionCategory.CONTENT,
    ...     description="Approve finalized documents",
    ...     requires=["document:read", "document:write"]
    ... )
    >>> 
    >>> # Check if user has permission
    >>> if user.has_permission("document:approve"):
    ...     approve_document(doc_id)
"""

import enum
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates

from backend.core.constants import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
)
from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
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


class PermissionCategory(str, enum.Enum):
    """
    Permission category for logical grouping.
    
    Categories:
    - SYSTEM: System administration (users, tenants, settings)
    - CONTENT: Content management (documents, contracts, templates)
    - COLLABORATION: Team and sharing features
    - ANALYTICS: Reporting, dashboards, statistics
    - API: API access and integrations
    - BILLING: Subscription and payment management
    - SECURITY: Security and audit features
    """
    
    SYSTEM = "system"
    CONTENT = "content"
    COLLABORATION = "collaboration"
    ANALYTICS = "analytics"
    API = "api"
    BILLING = "billing"
    SECURITY = "security"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.SYSTEM: "Sistem Yönetimi",
            self.CONTENT: "İçerik Yönetimi",
            self.COLLABORATION: "İşbirliği",
            self.ANALYTICS: "Analitik",
            self.API: "API Erişimi",
            self.BILLING: "Faturalama",
            self.SECURITY: "Güvenlik",
        }
        return names.get(self, self.value)


class PermissionScope(str, enum.Enum):
    """
    Permission scope defines the boundary of access.
    
    Scopes:
    - GLOBAL: Platform-wide access
    - TENANT: Tenant-wide access
    - ORGANIZATION: Organization-specific
    - TEAM: Team-specific
    - OWNER: Own resources only
    """
    
    GLOBAL = "global"
    TENANT = "tenant"
    ORGANIZATION = "organization"
    TEAM = "team"
    OWNER = "owner"
    
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
            self.OWNER: "Sadece Kendi Kayıtları",
        }
        return names.get(self, self.value)


class PermissionEffect(str, enum.Enum):
    """
    Permission effect (allow or deny).
    
    Used for explicit deny rules (overrides allow).
    """
    
    ALLOW = "allow"
    DENY = "deny"
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# PERMISSION MODEL
# =============================================================================


class Permission(Base, BaseModelMixin, AuditMixin, SoftDeleteMixin):
    """
    Permission model for atomic authorization units.
    
    Permissions define specific access rights in the system:
    - Each permission has a unique identifier (resource:action)
    - Permissions are grouped into categories
    - Permissions can depend on other permissions
    - Conditional logic can be attached (JSON rules)
    - Audit trail tracks permission grants/revokes
    
    Permission Naming Convention:
        Format: "resource:action"
        Examples:
        - "document:read" → Read documents
        - "document:write" → Create/edit documents
        - "document:delete" → Delete documents
        - "contract:approve" → Approve contracts
        - "user:manage" → Manage users
        
    Wildcard Support:
        - "document:*" → All document actions
        - "*:read" → Read all resources
        - "*:*" → All permissions (superadmin)
    
    Attributes:
        name: Permission identifier (resource:action)
        resource: Resource type (document, contract, user)
        action: Action type (read, write, delete, approve)
        
        category: Permission category
        scope: Permission scope
        effect: Allow or deny (default: allow)
        
        display_name: Human-readable name
        description: Permission description
        
        requires: Prerequisite permissions (array)
        conflicts_with: Conflicting permissions (array)
        
        is_system: System permission (cannot be deleted)
        is_dangerous: Dangerous permission (requires confirmation)
        
        conditions: JSON conditions for context-based access
        metadata: Additional metadata
        
    Relationships:
        roles: Roles that include this permission (implicit via Role.permissions JSONB)
    """
    
    __tablename__ = "permissions"
    
    # =========================================================================
    # IDENTITY
    # =========================================================================
    
    name = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Permission identifier (resource:action format)",
    )
    
    resource = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Resource type (document, contract, user, etc.)",
    )
    
    action = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Action type (read, write, delete, approve, etc.)",
    )
    
    # =========================================================================
    # CLASSIFICATION
    # =========================================================================
    
    category = Column(
        Enum(PermissionCategory, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Permission category for grouping",
    )
    
    scope = Column(
        Enum(PermissionScope, native_enum=False, length=50),
        nullable=False,
        default=PermissionScope.TENANT,
        comment="Permission scope (global, tenant, organization, team, owner)",
    )
    
    effect = Column(
        Enum(PermissionEffect, native_enum=False, length=50),
        nullable=False,
        default=PermissionEffect.ALLOW,
        comment="Permission effect (allow or deny)",
    )
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    display_name = Column(
        String(MAX_NAME_LENGTH),
        nullable=False,
        comment="Human-readable permission name",
    )
    
    display_name_tr = Column(
        String(MAX_NAME_LENGTH),
        nullable=True,
        comment="Turkish display name",
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Detailed permission description",
    )
    
    # =========================================================================
    # DEPENDENCIES
    # =========================================================================
    
    requires = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Prerequisite permissions (must have all)",
    )
    
    conflicts_with = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Conflicting permissions (cannot have both)",
    )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    is_system = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="System permission (cannot be deleted)",
    )
    
    is_dangerous = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Dangerous permission (requires confirmation)",
    )
    
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Permission is active",
    )
    
    # =========================================================================
    # CONDITIONAL ACCESS
    # =========================================================================
    
    conditions = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Conditional logic for context-based access (JSON rules)",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (examples, notes, related docs)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for resource lookup
        Index("ix_permissions_resource", "resource"),
        
        # Index for category filtering
        Index("ix_permissions_category", "category"),
        
        # Index for system permissions
        Index(
            "ix_permissions_system",
            "is_system",
            postgresql_where="is_system = true",
        ),
        
        # Index for active permissions
        Index(
            "ix_permissions_active",
            "is_active",
            postgresql_where="is_active = true AND deleted_at IS NULL",
        ),
        
        # Composite index for resource:action lookup
        Index("ix_permissions_resource_action", "resource", "action"),
        
        # Check: name format (resource:action)
        CheckConstraint(
            "name ~ '^[a-z_]+:[a-z_]+$'",
            name="ck_permissions_name_format",
        ),
    )
    
    # =========================================================================
    # DEPENDENCY VALIDATION
    # =========================================================================
    
    def check_dependencies(self, user_permissions: list[str]) -> tuple[bool, list[str]]:
        """
        Check if all prerequisite permissions are met.
        
        Args:
            user_permissions: List of permissions user has
            
        Returns:
            tuple: (all_met: bool, missing: list[str])
            
        Example:
            >>> perm = Permission(
            ...     name="document:approve",
            ...     requires=["document:read", "document:write"]
            ... )
            >>> all_met, missing = perm.check_dependencies(user_perms)
            >>> if not all_met:
            ...     print(f"Missing: {missing}")
        """
        if not self.requires:
            return True, []
        
        missing = []
        for required_perm in self.requires:
            if required_perm not in user_permissions:
                missing.append(required_perm)
        
        return len(missing) == 0, missing
    
    def check_conflicts(self, user_permissions: list[str]) -> tuple[bool, list[str]]:
        """
        Check if any conflicting permissions exist.
        
        Args:
            user_permissions: List of permissions user has
            
        Returns:
            tuple: (has_conflicts: bool, conflicts: list[str])
            
        Example:
            >>> perm = Permission(
            ...     name="document:full_access",
            ...     conflicts_with=["document:read_only"]
            ... )
            >>> has_conflicts, conflicts = perm.check_conflicts(user_perms)
        """
        if not self.conflicts_with:
            return False, []
        
        conflicts = []
        for conflict_perm in self.conflicts_with:
            if conflict_perm in user_permissions:
                conflicts.append(conflict_perm)
        
        return len(conflicts) > 0, conflicts
    
    def evaluate_conditions(self, context: dict[str, Any]) -> bool:
        """
        Evaluate conditional access rules.
        
        Conditions are JSON rules that define context-based access:
        - User attributes (role, department, seniority)
        - Resource attributes (status, owner, sensitivity)
        - Time-based (business hours, expiration)
        - Feature flags (subscription plan, beta features)
        
        Args:
            context: Context dictionary with evaluation variables
            
        Returns:
            bool: True if conditions are met
            
        Example:
            >>> perm = Permission(
            ...     name="contract:approve",
            ...     conditions={
            ...         "user_role": ["senior_partner", "managing_partner"],
            ...         "document_status": "finalized",
            ...         "subscription_plan": ["professional", "enterprise"]
            ...     }
            ... )
            >>> 
            >>> context = {
            ...     "user_role": "senior_partner",
            ...     "document_status": "finalized",
            ...     "subscription_plan": "professional"
            ... }
            >>> perm.evaluate_conditions(context)  # True
        """
        if not self.conditions:
            return True  # No conditions = always allowed
        
        for key, expected_values in self.conditions.items():
            actual_value = context.get(key)
            
            # Check if actual value matches any expected value
            if isinstance(expected_values, list):
                if actual_value not in expected_values:
                    logger.debug(
                        "Condition not met",
                        permission=self.name,
                        condition_key=key,
                        expected=expected_values,
                        actual=actual_value,
                    )
                    return False
            else:
                if actual_value != expected_values:
                    logger.debug(
                        "Condition not met",
                        permission=self.name,
                        condition_key=key,
                        expected=expected_values,
                        actual=actual_value,
                    )
                    return False
        
        return True
    
    # =========================================================================
    # WILDCARD MATCHING
    # =========================================================================
    
    def matches(self, permission_name: str) -> bool:
        """
        Check if this permission matches a given permission string.
        
        Supports wildcard matching:
        - "document:*" matches "document:read", "document:write"
        - "*:read" matches "document:read", "contract:read"
        - "*:*" matches all permissions
        
        Args:
            permission_name: Permission to check (resource:action)
            
        Returns:
            bool: True if matches
            
        Example:
            >>> perm = Permission(name="document:*")
            >>> perm.matches("document:read")  # True
            >>> perm.matches("contract:read")  # False
        """
        if self.name == "*:*":
            return True
        
        if ":" not in permission_name:
            return False
        
        target_resource, target_action = permission_name.split(":", 1)
        
        # Wildcard resource
        if self.resource == "*":
            return self.action == "*" or self.action == target_action
        
        # Wildcard action
        if self.action == "*":
            return self.resource == target_resource
        
        # Exact match
        return self.name == permission_name
    
    # =========================================================================
    # SYSTEM PERMISSIONS
    # =========================================================================
    
    @staticmethod
    def create_system_permissions() -> list[dict[str, Any]]:
        """
        Create default system permissions.
        
        Returns:
            list: List of permission definitions
            
        Example:
            >>> permissions = Permission.create_system_permissions()
            >>> for perm_data in permissions:
            ...     perm = Permission(**perm_data)
            ...     db.add(perm)
        """
        permissions = [
            # ================================================================
            # SYSTEM PERMISSIONS
            # ================================================================
            {
                "name": "user:read",
                "resource": "user",
                "action": "read",
                "category": PermissionCategory.SYSTEM,
                "scope": PermissionScope.TENANT,
                "display_name": "View Users",
                "display_name_tr": "Kullanıcıları Görüntüle",
                "description": "View user profiles and information",
                "is_system": True,
            },
            {
                "name": "user:write",
                "resource": "user",
                "action": "write",
                "category": PermissionCategory.SYSTEM,
                "scope": PermissionScope.TENANT,
                "display_name": "Manage Users",
                "display_name_tr": "Kullanıcıları Yönet",
                "description": "Create and edit user accounts",
                "requires": ["user:read"],
                "is_system": True,
            },
            {
                "name": "user:delete",
                "resource": "user",
                "action": "delete",
                "category": PermissionCategory.SYSTEM,
                "scope": PermissionScope.TENANT,
                "display_name": "Delete Users",
                "display_name_tr": "Kullanıcıları Sil",
                "description": "Delete user accounts",
                "requires": ["user:read", "user:write"],
                "is_system": True,
                "is_dangerous": True,
            },
            
            # ================================================================
            # CONTENT PERMISSIONS
            # ================================================================
            {
                "name": "document:read",
                "resource": "document",
                "action": "read",
                "category": PermissionCategory.CONTENT,
                "scope": PermissionScope.OWNER,
                "display_name": "View Documents",
                "display_name_tr": "Belgeleri Görüntüle",
                "description": "View and download documents",
                "is_system": True,
            },
            {
                "name": "document:write",
                "resource": "document",
                "action": "write",
                "category": PermissionCategory.CONTENT,
                "scope": PermissionScope.OWNER,
                "display_name": "Create/Edit Documents",
                "display_name_tr": "Belge Oluştur/Düzenle",
                "description": "Upload, create, and edit documents",
                "requires": ["document:read"],
                "is_system": True,
            },
            {
                "name": "document:delete",
                "resource": "document",
                "action": "delete",
                "category": PermissionCategory.CONTENT,
                "scope": PermissionScope.OWNER,
                "display_name": "Delete Documents",
                "display_name_tr": "Belgeleri Sil",
                "description": "Delete documents permanently",
                "requires": ["document:read", "document:write"],
                "is_system": True,
                "is_dangerous": True,
            },
            {
                "name": "document:share",
                "resource": "document",
                "action": "share",
                "category": PermissionCategory.COLLABORATION,
                "scope": PermissionScope.OWNER,
                "display_name": "Share Documents",
                "display_name_tr": "Belgeleri Paylaş",
                "description": "Share documents with teams or users",
                "requires": ["document:read"],
                "is_system": True,
            },
            
            # ================================================================
            # CONTRACT PERMISSIONS
            # ================================================================
            {
                "name": "contract:generate",
                "resource": "contract",
                "action": "generate",
                "category": PermissionCategory.CONTENT,
                "scope": PermissionScope.TENANT,
                "display_name": "Generate Contracts",
                "display_name_tr": "Sözleşme Oluştur",
                "description": "Generate contracts using AI templates",
                "is_system": True,
            },
            {
                "name": "contract:analyze",
                "resource": "contract",
                "action": "analyze",
                "category": PermissionCategory.CONTENT,
                "scope": PermissionScope.TENANT,
                "display_name": "Analyze Contracts",
                "display_name_tr": "Sözleşme Analiz Et",
                "description": "Analyze contracts for risks and compliance",
                "is_system": True,
            },
            {
                "name": "contract:approve",
                "resource": "contract",
                "action": "approve",
                "category": PermissionCategory.CONTENT,
                "scope": PermissionScope.TEAM,
                "display_name": "Approve Contracts",
                "display_name_tr": "Sözleşmeleri Onayla",
                "description": "Approve finalized contracts",
                "requires": ["contract:generate", "contract:analyze"],
                "is_system": True,
                "conditions": {
                    "user_role": ["senior_partner", "managing_partner", "admin"],
                },
            },
            
            # ================================================================
            # TEAM PERMISSIONS
            # ================================================================
            {
                "name": "team:read",
                "resource": "team",
                "action": "read",
                "category": PermissionCategory.COLLABORATION,
                "scope": PermissionScope.ORGANIZATION,
                "display_name": "View Teams",
                "display_name_tr": "Ekipleri Görüntüle",
                "description": "View team information",
                "is_system": True,
            },
            {
                "name": "team:write",
                "resource": "team",
                "action": "write",
                "category": PermissionCategory.COLLABORATION,
                "scope": PermissionScope.ORGANIZATION,
                "display_name": "Manage Teams",
                "display_name_tr": "Ekipleri Yönet",
                "description": "Create and edit teams",
                "requires": ["team:read"],
                "is_system": True,
            },
            {
                "name": "team:delete",
                "resource": "team",
                "action": "delete",
                "category": PermissionCategory.COLLABORATION,
                "scope": PermissionScope.ORGANIZATION,
                "display_name": "Delete Teams",
                "display_name_tr": "Ekipleri Sil",
                "description": "Delete teams",
                "requires": ["team:read", "team:write"],
                "is_system": True,
                "is_dangerous": True,
            },
            
            # ================================================================
            # ANALYTICS PERMISSIONS
            # ================================================================
            {
                "name": "analytics:read",
                "resource": "analytics",
                "action": "read",
                "category": PermissionCategory.ANALYTICS,
                "scope": PermissionScope.TENANT,
                "display_name": "View Analytics",
                "display_name_tr": "Analitiği Görüntüle",
                "description": "View usage statistics and reports",
                "is_system": True,
            },
            {
                "name": "analytics:export",
                "resource": "analytics",
                "action": "export",
                "category": PermissionCategory.ANALYTICS,
                "scope": PermissionScope.TENANT,
                "display_name": "Export Analytics",
                "display_name_tr": "Analitik Dışa Aktar",
                "description": "Export analytics data and reports",
                "requires": ["analytics:read"],
                "is_system": True,
            },
            
            # ================================================================
            # API PERMISSIONS
            # ================================================================
            {
                "name": "api:read",
                "resource": "api",
                "action": "read",
                "category": PermissionCategory.API,
                "scope": PermissionScope.TENANT,
                "display_name": "API Read Access",
                "display_name_tr": "API Okuma Erişimi",
                "description": "Read-only API access",
                "is_system": True,
            },
            {
                "name": "api:write",
                "resource": "api",
                "action": "write",
                "category": PermissionCategory.API,
                "scope": PermissionScope.TENANT,
                "display_name": "API Write Access",
                "display_name_tr": "API Yazma Erişimi",
                "description": "Full API access (read/write)",
                "requires": ["api:read"],
                "is_system": True,
            },
            
            # ================================================================
            # BILLING PERMISSIONS
            # ================================================================
            {
                "name": "billing:read",
                "resource": "billing",
                "action": "read",
                "category": PermissionCategory.BILLING,
                "scope": PermissionScope.TENANT,
                "display_name": "View Billing",
                "display_name_tr": "Faturaları Görüntüle",
                "description": "View subscription and billing information",
                "is_system": True,
            },
            {
                "name": "billing:manage",
                "resource": "billing",
                "action": "manage",
                "category": PermissionCategory.BILLING,
                "scope": PermissionScope.TENANT,
                "display_name": "Manage Billing",
                "display_name_tr": "Faturaları Yönet",
                "description": "Manage subscriptions and payment methods",
                "requires": ["billing:read"],
                "is_system": True,
            },
            
            # ================================================================
            # SECURITY PERMISSIONS
            # ================================================================
            {
                "name": "audit:read",
                "resource": "audit",
                "action": "read",
                "category": PermissionCategory.SECURITY,
                "scope": PermissionScope.TENANT,
                "display_name": "View Audit Logs",
                "display_name_tr": "Denetim Kayıtlarını Görüntüle",
                "description": "View audit trail and security logs",
                "is_system": True,
            },
            {
                "name": "security:manage",
                "resource": "security",
                "action": "manage",
                "category": PermissionCategory.SECURITY,
                "scope": PermissionScope.TENANT,
                "display_name": "Manage Security",
                "display_name_tr": "Güvenlik Yönetimi",
                "description": "Manage security settings and policies",
                "requires": ["audit:read"],
                "is_system": True,
                "is_dangerous": True,
            },
        ]
        
        return permissions
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("name")
    def validate_name(self, key: str, name: str) -> str:
        """Validate permission name format (resource:action)."""
        import re
        
        # Format: resource:action (lowercase, underscores allowed)
        name_pattern = r"^[a-z_]+:[a-z_]+$"
        if not re.match(name_pattern, name):
            raise ValidationError(
                message="Permission name format: 'resource:action' (lowercase, underscores)",
                field="name",
            )
        
        # Extract resource and action for consistency
        resource, action = name.split(":", 1)
        if self.resource and self.resource != resource:
            raise ValidationError(
                message=f"Resource mismatch: name='{resource}' but resource='{self.resource}'",
                field="name",
            )
        if self.action and self.action != action:
            raise ValidationError(
                message=f"Action mismatch: name='{action}' but action='{self.action}'",
                field="name",
            )
        
        return name.lower()
    
    @validates("resource", "action")
    def validate_resource_action(self, key: str, value: str) -> str:
        """Validate resource and action format."""
        import re
        
        # Lowercase, underscores, no spaces
        pattern = r"^[a-z_]+$"
        if not re.match(pattern, value):
            raise ValidationError(
                message=f"{key} must be lowercase with underscores only",
                field=key,
            )
        
        return value.lower()
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Permission(name={self.name}, category={self.category})>"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add display fields
        data["category_display"] = self.category.display_name_tr
        data["scope_display"] = self.scope.display_name_tr
        data["display"] = self.display_name_tr or self.display_name
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Permission",
    "PermissionCategory",
    "PermissionScope",
    "PermissionEffect",
]