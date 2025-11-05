"""
Database models package initialization.

This module exports all database models for easy importing:
- Base classes and mixins
- All entity models
- Association tables
- Enums

Usage:
    >>> from backend.core.database.models import User, Tenant, Document
    >>> from backend.core.database.models import Base, BaseModelMixin
    >>> from backend.core.database.models import UserRole, DocumentType
"""

# =============================================================================
# BASE CLASSES & MIXINS
# =============================================================================

from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
    AuditMixin,
    SoftDeleteMixin,
    TimestampMixin,
)

# =============================================================================
# USER & AUTHENTICATION MODELS
# =============================================================================

from backend.core.database.models.user import (
    User,
    UserRole,
    UserStatus,
    OnboardingStatus,
)

from backend.core.database.models.session import (
    Session,
)

from backend.core.database.models.api_key import (
    APIKey,
)

# =============================================================================
# TENANT & ORGANIZATION MODELS
# =============================================================================

from backend.core.database.models.tenant import (
    Tenant,
    TenantStatus,
    SubscriptionPlan,
    BillingCycle,
)

from backend.core.database.models.organization import (
    Organization,
    OrganizationType,
    OrganizationStatus,
)

from backend.core.database.models.team import (
    Team,
    TeamType,
    TeamStatus,
    TeamMemberRole,
    team_members,
)

# =============================================================================
# RBAC MODELS
# =============================================================================

from backend.core.database.models.role import (
    Role,
    RoleType,
    RoleScope,
    user_roles,
)

from backend.core.database.models.permission import (
    Permission,
    PermissionCategory,
    PermissionScope,
    PermissionEffect,
)

# =============================================================================
# DOCUMENT MODELS
# =============================================================================

from backend.core.database.models.document import (
    Document,
    DocumentType,
    ProcessingStatus,
    AccessLevel,
    SecurityClassification,
)

from backend.core.database.models.document_version import (
    DocumentVersion,
    ChangeType,
    VersionStatus,
    StorageStrategy,
)

# =============================================================================
# CHAT MODELS
# =============================================================================

from backend.core.database.models.chat_session import (
    ChatSession,
    ChatStatus,
    ChatMode,
    chat_session_documents,
)

from backend.core.database.models.chat_message import (
    ChatMessage,
    MessageRole,
    MessageStatus,
    ContentType,
)

# =============================================================================
# ALL MODELS LIST
# =============================================================================

__all__ = [
    # Base classes
    "Base",
    "BaseModelMixin",
    "TenantMixin",
    "AuditMixin",
    "SoftDeleteMixin",
    "TimestampMixin",
    
    # User models
    "User",
    "UserRole",
    "UserStatus",
    "OnboardingStatus",
    
    # Authentication models
    "Session",
    "APIKey",
    
    # Tenant models
    "Tenant",
    "TenantStatus",
    "SubscriptionPlan",
    "BillingCycle",
    
    # Organization models
    "Organization",
    "OrganizationType",
    "OrganizationStatus",
    
    # Team models
    "Team",
    "TeamType",
    "TeamStatus",
    "TeamMemberRole",
    "team_members",
    
    # RBAC models
    "Role",
    "RoleType",
    "RoleScope",
    "user_roles",
    "Permission",
    "PermissionCategory",
    "PermissionScope",
    "PermissionEffect",
    
    # Document models
    "Document",
    "DocumentType",
    "ProcessingStatus",
    "AccessLevel",
    "SecurityClassification",
    "DocumentVersion",
    "ChangeType",
    "VersionStatus",
    "StorageStrategy",
    
    # Chat models
    "ChatSession",
    "ChatStatus",
    "ChatMode",
    "chat_session_documents",
    "ChatMessage",
    "MessageRole",
    "MessageStatus",
    "ContentType",
]
