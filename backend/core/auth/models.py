"""
RBAC Database Models - Harvey/Legora %100 Production Security.

World-class Role-Based Access Control for Turkish Legal AI:
- Multi-tenant security architecture
- Granular permission system
- Tenant-scoped data isolation
- Audit-ready access tracking
- JWT token management
- Session management

Why RBAC?
    Without: All users have same access → security nightmare
    With: Role-based permissions → secure multi-tenant platform

    Impact: Enterprise-grade security! 🔐

Security Architecture:
    [User] → [TenantMembership] → [Tenant]
       ↓           ↓
    [UserRole] → [Role] → [RolePermission] → [Permission]
       ↓
    [Session] → [RefreshToken]

Permission Model:
    - resource:action format (e.g., "documents:read", "search:execute")
    - Hierarchical inheritance (admin inherits all permissions)
    - Tenant-scoped (permissions apply within tenant context)
    - Time-based expiration support

Built-in Roles:
    - superadmin: Platform-wide access
    - tenant_admin: Full tenant access
    - legal_analyst: Read + search + analyze
    - legal_researcher: Read + search
    - viewer: Read-only access

Tenant Isolation:
    - Row-level security (RLS) via PostgreSQL
    - Request-scoped tenant context
    - Cross-tenant query prevention
    - Data leak protection

Features:
    - Password hashing (bcrypt)
    - JWT token management
    - Session tracking
    - IP-based restrictions
    - MFA support (future)
    - OAuth2 integration (future)

Performance:
    - Indexed lookups O(1)
    - Permission cache (Redis)
    - Session cache (Redis)
    - Lazy loading relationships

KVKK/GDPR Compliance:
    - User consent tracking
    - Data retention policies
    - Right to deletion
    - Audit trail integration
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict
from enum import Enum as PyEnum
import uuid

from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime, ForeignKey,
    Index, CheckConstraint, UniqueConstraint, Table
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property

from backend.core.database.models import Base


# =============================================================================
# ENUMS
# =============================================================================


class UserStatusEnum(str, PyEnum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class TenantStatusEnum(str, PyEnum):
    """Tenant status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    ARCHIVED = "archived"


class PermissionScopeEnum(str, PyEnum):
    """Permission scope level."""
    GLOBAL = "global"  # Platform-wide
    TENANT = "tenant"  # Tenant-scoped
    RESOURCE = "resource"  # Resource-specific


# =============================================================================
# TENANT MODELS
# =============================================================================


class Tenant(Base):
    """
    Tenant model for multi-tenancy.

    Harvey/Legora %100: Enterprise tenant isolation.

    Features:
    - Tenant-scoped data isolation
    - Subscription management
    - Usage tracking
    - Custom settings per tenant
    - Quota management
    """

    __tablename__ = "tenants"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Tenant UUID"
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[TenantStatusEnum] = mapped_column(
        String(50),
        default=TenantStatusEnum.TRIAL,
        nullable=False
    )

    # Contact
    contact_email: Mapped[str] = mapped_column(String(255), nullable=False)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Subscription
    subscription_tier: Mapped[str] = mapped_column(
        String(50),
        default="trial",
        nullable=False,
        comment="trial | basic | professional | enterprise"
    )
    subscription_start: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    subscription_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Quotas (monthly limits)
    quota_documents: Mapped[int] = mapped_column(
        Integer,
        default=1000,
        nullable=False,
        comment="Max documents per month"
    )
    quota_searches: Mapped[int] = mapped_column(
        Integer,
        default=10000,
        nullable=False,
        comment="Max searches per month"
    )
    quota_api_calls: Mapped[int] = mapped_column(
        Integer,
        default=100000,
        nullable=False,
        comment="Max API calls per month"
    )

    # Usage tracking (current period)
    usage_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    usage_searches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    usage_api_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    usage_reset_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.utcnow() + timedelta(days=30),
        nullable=False
    )

    # Settings (JSONB for flexibility)
    settings: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Custom tenant settings"
    )

    # Features (enabled features for this tenant)
    enabled_features: Mapped[list] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
        comment="Feature flags: ['advanced_search', 'rag', 'citations', ...]"
    )

    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    memberships: Mapped[List["TenantMembership"]] = relationship(
        "TenantMembership",
        back_populates="tenant",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_tenants_name", "name"),
        Index("ix_tenants_status", "status"),
        Index("ix_tenants_subscription_tier", "subscription_tier"),
    )

    def __repr__(self) -> str:
        return f"<Tenant(id={self.id}, name='{self.name}', status='{self.status}')>"

    @hybrid_property
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatusEnum.ACTIVE

    def check_quota(self, resource: str) -> bool:
        """Check if tenant has quota available for resource."""
        quota_map = {
            "documents": (self.usage_documents, self.quota_documents),
            "searches": (self.usage_searches, self.quota_searches),
            "api_calls": (self.usage_api_calls, self.quota_api_calls),
        }

        if resource not in quota_map:
            return True

        usage, quota = quota_map[resource]
        return usage < quota

    def increment_usage(self, resource: str) -> None:
        """Increment usage counter for resource."""
        if resource == "documents":
            self.usage_documents += 1
        elif resource == "searches":
            self.usage_searches += 1
        elif resource == "api_calls":
            self.usage_api_calls += 1


# =============================================================================
# USER MODELS
# =============================================================================


class User(Base):
    """
    User model for authentication and authorization.

    Harvey/Legora %100: Secure user management.

    Features:
    - Password hashing (bcrypt)
    - Email verification
    - Multi-tenant membership
    - Session management
    - Audit trail
    """

    __tablename__ = "users"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="User UUID"
    )

    # Authentication
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # Profile
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Status
    status: Mapped[UserStatusEnum] = mapped_column(
        String(50),
        default=UserStatusEnum.PENDING_VERIFICATION,
        nullable=False
    )
    is_superadmin: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Platform superadmin flag"
    )

    # Verification
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    verification_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Password management
    password_changed_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    password_reset_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    password_reset_expires: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Security
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Account locked until this time"
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_login_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    # Preferences
    preferences: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="User preferences and settings"
    )

    # KVKK/GDPR
    consent_marketing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consent_data_processing: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    data_retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Data deletion scheduled for this date"
    )

    # Relationships
    memberships: Mapped[List["TenantMembership"]] = relationship(
        "TenantMembership",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    sessions: Mapped[List["Session"]] = relationship(
        "Session",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_username", "username"),
        Index("ix_users_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>"

    @hybrid_property
    def is_active(self) -> bool:
        """Check if user is active."""
        return self.status == UserStatusEnum.ACTIVE

    @hybrid_property
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until

    def record_login(self, ip_address: str) -> None:
        """Record successful login."""
        self.last_login_at = datetime.utcnow()
        self.last_login_ip = ip_address
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_failed_login(self, max_attempts: int = 5, lockout_minutes: int = 30) -> None:
        """Record failed login attempt and lock account if needed."""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.locked_until = datetime.utcnow() + timedelta(minutes=lockout_minutes)


class TenantMembership(Base):
    """
    User-Tenant membership model.

    Harvey/Legora %100: Multi-tenant user management.

    Features:
    - User can belong to multiple tenants
    - Default tenant selection
    - Invitation workflow
    - Membership metadata
    """

    __tablename__ = "tenant_memberships"

    # Composite primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )

    # Membership status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Default tenant for user"
    )

    # Invitation
    invited_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User ID who sent invitation"
    )
    invited_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memberships")
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="memberships")

    # Indexes
    __table_args__ = (
        Index("ix_memberships_user_id", "user_id"),
        Index("ix_memberships_tenant_id", "tenant_id"),
        UniqueConstraint("user_id", "tenant_id", name="uq_user_tenant"),
    )

    def __repr__(self) -> str:
        return f"<TenantMembership(user_id={self.user_id}, tenant_id={self.tenant_id})>"


# =============================================================================
# RBAC MODELS
# =============================================================================


class Role(Base):
    """
    Role model for RBAC.

    Harvey/Legora %100: Hierarchical role management.

    Features:
    - Built-in roles (superadmin, tenant_admin, analyst, researcher, viewer)
    - Custom roles per tenant
    - Role hierarchy
    - Permission inheritance
    """

    __tablename__ = "roles"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Basic info
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Role name (e.g., 'tenant_admin', 'legal_analyst')"
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scope
    is_system_role: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Built-in system role (cannot be deleted)"
    )
    tenant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=True,
        comment="NULL for system roles, tenant ID for custom roles"
    )

    # Priority (higher = more permissions)
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Role priority (1000=superadmin, 900=tenant_admin, 500=analyst, 100=viewer)"
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="role",
        cascade="all, delete-orphan"
    )
    role_permissions: Mapped[List["RolePermission"]] = relationship(
        "RolePermission",
        back_populates="role",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_roles_name", "name"),
        Index("ix_roles_tenant_id", "tenant_id"),
        UniqueConstraint("name", "tenant_id", name="uq_role_name_tenant"),
    )

    def __repr__(self) -> str:
        return f"<Role(id={self.id}, name='{self.name}', priority={self.priority})>"


class Permission(Base):
    """
    Permission model for RBAC.

    Harvey/Legora %100: Granular permission management.

    Format: resource:action
    Examples:
        - documents:read
        - documents:write
        - documents:delete
        - search:execute
        - search:advanced
        - analytics:view
        - users:manage
        - settings:configure
    """

    __tablename__ = "permissions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Permission data
    resource: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Resource name (e.g., 'documents', 'search', 'users')"
    )
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Action name (e.g., 'read', 'write', 'delete', 'execute')"
    )

    # Computed field
    code: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="Full permission code (resource:action)"
    )

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scope: Mapped[PermissionScopeEnum] = mapped_column(
        String(50),
        default=PermissionScopeEnum.TENANT,
        nullable=False
    )

    # Relationships
    role_permissions: Mapped[List["RolePermission"]] = relationship(
        "RolePermission",
        back_populates="permission",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_permissions_code", "code"),
        Index("ix_permissions_resource", "resource"),
    )

    def __repr__(self) -> str:
        return f"<Permission(code='{self.code}')>"


class RolePermission(Base):
    """
    Role-Permission mapping (many-to-many).

    Harvey/Legora %100: Fine-grained permission assignment.
    """

    __tablename__ = "role_permissions"

    # Composite primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="CASCADE"),
        nullable=False
    )

    # Metadata
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    granted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User ID who granted permission"
    )

    # Relationships
    role: Mapped["Role"] = relationship("Role", back_populates="role_permissions")
    permission: Mapped["Permission"] = relationship("Permission", back_populates="role_permissions")

    # Indexes
    __table_args__ = (
        Index("ix_role_permissions_role_id", "role_id"),
        Index("ix_role_permissions_permission_id", "permission_id"),
        UniqueConstraint("role_id", "permission_id", name="uq_role_permission"),
    )

    def __repr__(self) -> str:
        return f"<RolePermission(role_id={self.role_id}, permission_id={self.permission_id})>"


class UserRole(Base):
    """
    User-Role mapping (many-to-many).

    Harvey/Legora %100: Tenant-scoped role assignment.

    Features:
    - User can have different roles in different tenants
    - Time-based role expiration
    - Role assignment audit trail
    """

    __tablename__ = "user_roles"

    # Composite primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="Tenant context for this role"
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Time-based expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Role expires at this time (NULL = permanent)"
    )

    # Audit
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    assigned_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User ID who assigned role"
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="user_roles")
    role: Mapped["Role"] = relationship("Role", back_populates="user_roles")

    # Indexes
    __table_args__ = (
        Index("ix_user_roles_user_id", "user_id"),
        Index("ix_user_roles_role_id", "role_id"),
        Index("ix_user_roles_tenant_id", "tenant_id"),
        UniqueConstraint("user_id", "role_id", "tenant_id", name="uq_user_role_tenant"),
    )

    def __repr__(self) -> str:
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id}, tenant_id={self.tenant_id})>"

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if role assignment is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


# =============================================================================
# SESSION MODELS
# =============================================================================


class Session(Base):
    """
    User session model for JWT token management.

    Harvey/Legora %100: Secure session tracking.

    Features:
    - JWT token tracking
    - Refresh token management
    - IP-based session tracking
    - Device fingerprinting
    - Session revocation
    """

    __tablename__ = "sessions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    # Token data
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str] = mapped_column(Text, nullable=False)
    token_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Session data
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device_fingerprint: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Timestamps
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")

    # Indexes
    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_access_token", "access_token", postgresql_using="hash"),
        Index("ix_sessions_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, user_id={self.user_id}, active={self.is_active})>"

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.token_expires_at

    def revoke(self, reason: str = "user_logout") -> None:
        """Revoke session."""
        self.is_active = False
        self.revoked_at = datetime.utcnow()
        self.revoked_reason = reason


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "UserStatusEnum",
    "TenantStatusEnum",
    "PermissionScopeEnum",
    # Tenant
    "Tenant",
    # User
    "User",
    "TenantMembership",
    # RBAC
    "Role",
    "Permission",
    "RolePermission",
    "UserRole",
    # Session
    "Session",
]
