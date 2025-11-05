"""
Tenant model for multi-tenant isolation in Turkish Legal AI.

This module provides the Tenant model for organization/workspace management:
- Multi-tenant data isolation
- Subscription and billing management
- Organization settings and configuration
- Usage limits and quotas
- KVKK compliance settings
- White-labeling support
- API access configuration

Tenant Structure:
- Each tenant represents an isolated workspace (law firm, legal department)
- All user data is scoped to tenant
- Row-Level Security (RLS) enforces isolation
- Separate storage buckets per tenant
- Independent encryption keys (optional)

Example:
    >>> tenant = Tenant(
    ...     name="Yilmaz Hukuk Bürosu",
    ...     domain="yilmaz-hukuk.com",
    ...     subscription_plan=SubscriptionPlan.PROFESSIONAL
    ... )
    >>> tenant.is_trial_active
    True
"""

import enum
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Enum,
    Integer,
    String,
    Text,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates

from backend.core.constants import (
    DEFAULT_TRIAL_DAYS,
    MAX_DOMAIN_LENGTH,
    MAX_NAME_LENGTH,
    MAX_ORGANIZATION_SIZE,
)
from backend.core.exceptions import (
    QuotaExceededError,
    SubscriptionExpiredError,
    ValidationError,
)
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


class SubscriptionPlan(str, enum.Enum):
    """
    Subscription plans with different feature sets and limits.
    
    Plans:
    - FREE: Basic features, limited usage
    - TRIAL: Full features for trial period
    - STARTER: Small teams, moderate usage
    - PROFESSIONAL: Medium teams, high usage
    - ENTERPRISE: Large teams, unlimited usage
    - CUSTOM: Custom pricing and features
    """
    
    FREE = "free"
    TRIAL = "trial"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name(self) -> str:
        """Human-readable plan name in Turkish."""
        names = {
            self.FREE: "Ücretsiz",
            self.TRIAL: "Deneme",
            self.STARTER: "Başlangıç",
            self.PROFESSIONAL: "Profesyonel",
            self.ENTERPRISE: "Kurumsal",
            self.CUSTOM: "Özel",
        }
        return names.get(self, self.value)
    
    @property
    def limits(self) -> dict[str, int]:
        """Get resource limits for this plan."""
        limits_map = {
            self.FREE: {
                "max_users": 2,
                "max_documents": 100,
                "max_storage_gb": 1,
                "max_api_calls_per_month": 1000,
                "max_llm_tokens_per_month": 100000,
                "max_chat_sessions": 50,
            },
            self.TRIAL: {
                "max_users": 10,
                "max_documents": 1000,
                "max_storage_gb": 10,
                "max_api_calls_per_month": 10000,
                "max_llm_tokens_per_month": 1000000,
                "max_chat_sessions": 500,
            },
            self.STARTER: {
                "max_users": 5,
                "max_documents": 5000,
                "max_storage_gb": 50,
                "max_api_calls_per_month": 50000,
                "max_llm_tokens_per_month": 5000000,
                "max_chat_sessions": 2000,
            },
            self.PROFESSIONAL: {
                "max_users": 25,
                "max_documents": 50000,
                "max_storage_gb": 500,
                "max_api_calls_per_month": 500000,
                "max_llm_tokens_per_month": 50000000,
                "max_chat_sessions": 10000,
            },
            self.ENTERPRISE: {
                "max_users": -1,  # Unlimited
                "max_documents": -1,
                "max_storage_gb": -1,
                "max_api_calls_per_month": -1,
                "max_llm_tokens_per_month": -1,
                "max_chat_sessions": -1,
            },
            self.CUSTOM: {
                "max_users": -1,  # Configured per tenant
                "max_documents": -1,
                "max_storage_gb": -1,
                "max_api_calls_per_month": -1,
                "max_llm_tokens_per_month": -1,
                "max_chat_sessions": -1,
            },
        }
        return limits_map.get(self, limits_map[self.FREE])
    
    @property
    def features(self) -> list[str]:
        """Get available features for this plan."""
        feature_map = {
            self.FREE: [
                "chat",
                "document_upload",
                "basic_analysis",
            ],
            self.TRIAL: [
                "chat",
                "document_upload",
                "document_analysis",
                "contract_generation",
                "legal_research",
                "turkish_law_corpus",
                "api_access",
                "email_support",
            ],
            self.STARTER: [
                "chat",
                "document_upload",
                "document_analysis",
                "contract_generation",
                "legal_research",
                "turkish_law_corpus",
                "api_access",
                "email_support",
            ],
            self.PROFESSIONAL: [
                "chat",
                "document_upload",
                "document_analysis",
                "contract_generation",
                "legal_research",
                "turkish_law_corpus",
                "precedent_matching",
                "advanced_analytics",
                "api_access",
                "priority_support",
                "custom_templates",
                "team_collaboration",
            ],
            self.ENTERPRISE: [
                "chat",
                "document_upload",
                "document_analysis",
                "contract_generation",
                "legal_research",
                "turkish_law_corpus",
                "precedent_matching",
                "advanced_analytics",
                "api_access",
                "priority_support",
                "custom_templates",
                "team_collaboration",
                "white_labeling",
                "sso",
                "dedicated_support",
                "custom_integrations",
                "on_premise_deployment",
            ],
            self.CUSTOM: [
                "*",  # All features, negotiated
            ],
        }
        return feature_map.get(self, feature_map[self.FREE])


class TenantStatus(str, enum.Enum):
    """Tenant lifecycle status."""
    
    ACTIVE = "active"              # Active subscription
    TRIAL = "trial"                # Trial period
    SUSPENDED = "suspended"        # Payment issue or policy violation
    EXPIRED = "expired"            # Subscription expired
    CANCELLED = "cancelled"        # User cancelled
    
    def __str__(self) -> str:
        return self.value


class OrganizationType(str, enum.Enum):
    """Turkish legal organization types."""
    
    LAW_FIRM = "law_firm"                    # Avukatlık Bürosu
    CORPORATE_LEGAL = "corporate_legal"      # Şirket Hukuk Departmanı
    PUBLIC_INSTITUTION = "public_institution"  # Kamu Kurumu
    COURT = "court"                          # Mahkeme
    PROSECUTORS_OFFICE = "prosecutors_office"  # Savcılık
    NOTARY = "notary"                        # Noterlik
    BAR_ASSOCIATION = "bar_association"      # Baro
    INDIVIDUAL = "individual"                # Bireysel Kullanıcı
    OTHER = "other"                          # Diğer
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# TENANT MODEL
# =============================================================================


class Tenant(Base, BaseModelMixin, AuditMixin, SoftDeleteMixin):
    """
    Tenant (organization/workspace) model for multi-tenant isolation.
    
    Each tenant represents an isolated workspace with:
    - Own users, documents, and data
    - Separate storage and encryption
    - Independent subscription and billing
    - Custom settings and configuration
    
    Attributes:
        name: Organization name
        slug: URL-safe identifier
        domain: Optional custom domain
        organization_type: Type of legal organization
        tax_id: Turkish tax ID (Vergi Kimlik No)
        address: Organization address
        city: City
        country: Country (default: TR)
        phone: Contact phone
        email: Contact email
        website: Organization website
        
        subscription_plan: Current subscription plan
        status: Tenant status
        trial_ends_at: Trial period end date
        subscription_starts_at: Subscription start date
        subscription_ends_at: Subscription end date
        
        settings: JSON settings (theme, features, etc.)
        limits: Custom resource limits (overrides plan defaults)
        usage: Current usage statistics
        metadata: Additional metadata
        
        is_active: Quick active check
        is_verified: Email/domain verification
        
    Relationships:
        users: Organization users
        documents: Organization documents
        chat_sessions: Organization chat sessions
    """
    
    __tablename__ = "tenants"
    
    # =========================================================================
    # ORGANIZATION IDENTITY
    # =========================================================================
    
    name = Column(
        String(MAX_NAME_LENGTH),
        nullable=False,
        comment="Organization name",
    )
    
    slug = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="URL-safe identifier (auto-generated from name)",
    )
    
    domain = Column(
        String(MAX_DOMAIN_LENGTH),
        unique=True,
        nullable=True,
        index=True,
        comment="Custom domain (e.g., yilmaz-hukuk.com)",
    )
    
    organization_type = Column(
        Enum(OrganizationType, native_enum=False, length=50),
        nullable=True,
        comment="Type of legal organization",
    )
    
    # =========================================================================
    # CONTACT INFORMATION
    # =========================================================================
    
    tax_id = Column(
        String(20),
        nullable=True,
        unique=True,
        index=True,
        comment="Turkish tax ID (Vergi Kimlik No)",
    )
    
    address = Column(
        Text,
        nullable=True,
        comment="Organization address",
    )
    
    city = Column(
        String(100),
        nullable=True,
        comment="City",
    )
    
    country = Column(
        String(2),
        nullable=False,
        default="TR",
        comment="Country code (ISO 3166-1 alpha-2)",
    )
    
    phone = Column(
        String(20),
        nullable=True,
        comment="Contact phone number",
    )
    
    email = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Organization contact email",
    )
    
    website = Column(
        String(255),
        nullable=True,
        comment="Organization website",
    )
    
    # =========================================================================
    # SUBSCRIPTION & BILLING
    # =========================================================================
    
    subscription_plan = Column(
        Enum(SubscriptionPlan, native_enum=False, length=50),
        nullable=False,
        default=SubscriptionPlan.TRIAL,
        index=True,
        comment="Current subscription plan",
    )
    
    status = Column(
        Enum(TenantStatus, native_enum=False, length=50),
        nullable=False,
        default=TenantStatus.TRIAL,
        index=True,
        comment="Tenant status",
    )
    
    trial_ends_at = Column(
        Date,
        nullable=True,
        comment="Trial period end date",
    )
    
    subscription_starts_at = Column(
        Date,
        nullable=True,
        comment="Paid subscription start date",
    )
    
    subscription_ends_at = Column(
        Date,
        nullable=True,
        index=True,
        comment="Subscription end date (NULL = active)",
    )
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    settings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Tenant settings (theme, language, features)",
    )
    
    limits = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Custom resource limits (overrides plan defaults)",
    )
    
    usage = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Current usage statistics (users, documents, tokens)",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata",
    )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Quick active check",
    )
    
    is_verified = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Email/domain verification status",
    )
    
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    
    # Users relationship (one-to-many)
    users = relationship(
        "User",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # Documents relationship (one-to-many)
    documents = relationship(
        "Document",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # Chat sessions relationship (one-to-many)
    chat_sessions = relationship(
        "ChatSession",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Unique slug
        UniqueConstraint("slug", name="uq_tenants_slug"),
        # Index for active tenants
        Index(
            "ix_tenants_active",
            "status",
            "is_active",
            postgresql_where="status = 'active' AND is_active = true AND deleted_at IS NULL",
        ),
        # Index for trial expiration
        Index(
            "ix_tenants_trial_expires",
            "trial_ends_at",
            postgresql_where="status = 'trial'",
        ),
        # Check: email format
        CheckConstraint(
            "email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$'",
            name="ck_tenants_email_format",
        ),
        # Check: country code (2 chars)
        CheckConstraint(
            "length(country) = 2",
            name="ck_tenants_country_length",
        ),
    )
    
    # =========================================================================
    # SUBSCRIPTION MANAGEMENT
    # =========================================================================
    
    @property
    def is_trial_active(self) -> bool:
        """Check if trial period is active and not expired."""
        if self.status != TenantStatus.TRIAL:
            return False
        
        if not self.trial_ends_at:
            return False
        
        today = datetime.now(timezone.utc).date()
        return self.trial_ends_at >= today
    
    @property
    def is_subscription_active(self) -> bool:
        """Check if paid subscription is active."""
        if self.status not in (TenantStatus.ACTIVE, TenantStatus.TRIAL):
            return False
        
        # Trial is considered active
        if self.is_trial_active:
            return True
        
        # Check paid subscription
        if not self.subscription_ends_at:
            return True  # No end date = active
        
        today = datetime.now(timezone.utc).date()
        return self.subscription_ends_at >= today
    
    @property
    def days_until_expiration(self) -> int | None:
        """Get days until subscription/trial expires."""
        today = datetime.now(timezone.utc).date()
        
        if self.status == TenantStatus.TRIAL and self.trial_ends_at:
            delta = self.trial_ends_at - today
            return max(0, delta.days)
        
        if self.subscription_ends_at:
            delta = self.subscription_ends_at - today
            return max(0, delta.days)
        
        return None  # No expiration
    
    def start_trial(self, trial_days: int = DEFAULT_TRIAL_DAYS) -> None:
        """
        Start trial period.
        
        Args:
            trial_days: Trial duration in days (default: 14)
        """
        self.status = TenantStatus.TRIAL
        self.subscription_plan = SubscriptionPlan.TRIAL
        self.trial_ends_at = datetime.now(timezone.utc).date() + timedelta(days=trial_days)
        self.is_active = True
        
        logger.info(
            "Trial started",
            tenant_id=str(self.id),
            tenant_name=self.name,
            trial_ends_at=self.trial_ends_at.isoformat(),
        )
    
    def upgrade_to_paid(
        self,
        plan: SubscriptionPlan,
        duration_months: int = 12,
    ) -> None:
        """
        Upgrade to paid subscription.
        
        Args:
            plan: Subscription plan
            duration_months: Subscription duration in months
        """
        self.subscription_plan = plan
        self.status = TenantStatus.ACTIVE
        self.subscription_starts_at = datetime.now(timezone.utc).date()
        self.subscription_ends_at = (
            self.subscription_starts_at + timedelta(days=duration_months * 30)
        )
        self.trial_ends_at = None  # Clear trial
        self.is_active = True
        
        logger.info(
            "Subscription upgraded",
            tenant_id=str(self.id),
            tenant_name=self.name,
            plan=plan.value,
            ends_at=self.subscription_ends_at.isoformat(),
        )
    
    def suspend(self, reason: str | None = None) -> None:
        """
        Suspend tenant (payment issue or policy violation).
        
        Args:
            reason: Suspension reason
        """
        self.status = TenantStatus.SUSPENDED
        self.is_active = False
        
        if reason:
            self.metadata["suspension_reason"] = reason
            self.metadata["suspended_at"] = datetime.now(timezone.utc).isoformat()
        
        logger.warning(
            "Tenant suspended",
            tenant_id=str(self.id),
            tenant_name=self.name,
            reason=reason,
        )
    
    def reactivate(self) -> None:
        """Reactivate suspended tenant."""
        self.status = TenantStatus.ACTIVE
        self.is_active = True
        
        if "suspension_reason" in self.metadata:
            self.metadata["reactivated_at"] = datetime.now(timezone.utc).isoformat()
        
        logger.info(
            "Tenant reactivated",
            tenant_id=str(self.id),
            tenant_name=self.name,
        )
    
    # =========================================================================
    # QUOTA & LIMITS
    # =========================================================================
    
    def get_limit(self, resource: str) -> int:
        """
        Get limit for a specific resource.
        
        Checks custom limits first, then falls back to plan defaults.
        
        Args:
            resource: Resource name (e.g., "max_users")
            
        Returns:
            int: Limit value (-1 = unlimited)
        """
        # Check custom limits
        if resource in self.limits:
            return self.limits[resource]
        
        # Fallback to plan limits
        plan_limits = self.subscription_plan.limits
        return plan_limits.get(resource, 0)
    
    def get_usage(self, resource: str) -> int:
        """
        Get current usage for a resource.
        
        Args:
            resource: Resource name (e.g., "users_count")
            
        Returns:
            int: Current usage
        """
        return self.usage.get(resource, 0)
    
    def check_quota(self, resource: str, requested: int = 1) -> bool:
        """
        Check if quota is available for resource.
        
        Args:
            resource: Resource name (e.g., "max_users")
            requested: Requested amount
            
        Returns:
            bool: True if quota available
        """
        limit = self.get_limit(resource)
        
        # Unlimited
        if limit == -1:
            return True
        
        # Get current usage
        usage_key = resource.replace("max_", "") + "_count"
        current = self.get_usage(usage_key)
        
        return (current + requested) <= limit
    
    def increment_usage(self, resource: str, amount: int = 1) -> None:
        """
        Increment usage counter.
        
        Args:
            resource: Resource name (e.g., "users_count")
            amount: Amount to increment
            
        Raises:
            QuotaExceededError: If quota exceeded
        """
        # Check limit
        limit_key = f"max_{resource.replace('_count', '')}"
        if not self.check_quota(limit_key, amount):
            raise QuotaExceededError(
                message=f"Quota exceeded for {resource}",
                resource=resource,
                limit=self.get_limit(limit_key),
                current=self.get_usage(resource),
            )
        
        # Increment
        if resource not in self.usage:
            self.usage[resource] = 0
        
        self.usage[resource] += amount
        
        logger.debug(
            "Usage incremented",
            tenant_id=str(self.id),
            resource=resource,
            amount=amount,
            new_total=self.usage[resource],
        )
    
    def decrement_usage(self, resource: str, amount: int = 1) -> None:
        """
        Decrement usage counter.
        
        Args:
            resource: Resource name
            amount: Amount to decrement
        """
        if resource in self.usage:
            self.usage[resource] = max(0, self.usage[resource] - amount)
    
    # =========================================================================
    # FEATURES
    # =========================================================================
    
    def has_feature(self, feature: str) -> bool:
        """
        Check if tenant has access to feature.
        
        Args:
            feature: Feature name (e.g., "api_access")
            
        Returns:
            bool: True if feature available
        """
        # Check if subscription active
        if not self.is_subscription_active:
            return False
        
        # Get plan features
        features = self.subscription_plan.features
        
        # Wildcard = all features
        if "*" in features:
            return True
        
        return feature in features
    
    def require_feature(self, feature: str) -> None:
        """
        Require feature access (raises if not available).
        
        Args:
            feature: Feature name
            
        Raises:
            SubscriptionExpiredError: If subscription expired
            QuotaExceededError: If feature not available
        """
        if not self.is_subscription_active:
            raise SubscriptionExpiredError(
                message="Abonelik süresi dolmuş",
                tenant_id=str(self.id),
            )
        
        if not self.has_feature(feature):
            raise QuotaExceededError(
                message=f"Bu özellik mevcut planınızda bulunmuyor: {feature}",
                resource=feature,
            )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("email")
    def validate_email(self, key: str, email: str) -> str:
        """Validate email format."""
        import re
        
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email.lower()):
            raise ValidationError(
                message="Geçersiz email formatı",
                field="email",
            )
        
        return email.lower()
    
    @validates("slug")
    def validate_slug(self, key: str, slug: str) -> str:
        """Validate slug format (URL-safe)."""
        import re
        
        # Only lowercase, numbers, hyphens
        slug_pattern = r"^[a-z0-9-]+$"
        if not re.match(slug_pattern, slug):
            raise ValidationError(
                message="Slug sadece küçük harf, rakam ve tire içerebilir",
                field="slug",
            )
        
        return slug
    
    @validates("domain")
    def validate_domain(self, key: str, domain: str | None) -> str | None:
        """Validate domain format."""
        if not domain:
            return None
        
        import re
        
        # Basic domain validation
        domain_pattern = r"^([a-z0-9-]+\.)+[a-z]{2,}$"
        if not re.match(domain_pattern, domain.lower()):
            raise ValidationError(
                message="Geçersiz domain formatı",
                field="domain",
            )
        
        return domain.lower()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    @staticmethod
    def generate_slug(name: str) -> str:
        """
        Generate URL-safe slug from organization name.
        
        Args:
            name: Organization name
            
        Returns:
            str: URL-safe slug
        """
        import re
        import unicodedata
        
        # Convert to lowercase
        slug = name.lower()
        
        # Remove Turkish characters
        replacements = {
            'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        }
        for tr_char, en_char in replacements.items():
            slug = slug.replace(tr_char, en_char)
        
        # Remove accents
        slug = unicodedata.normalize('NFKD', slug)
        slug = slug.encode('ascii', 'ignore').decode('ascii')
        
        # Replace spaces and special chars with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        # Remove consecutive hyphens
        slug = re.sub(r'-+', '-', slug)
        
        return slug
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Tenant(id={self.id}, name={self.name}, plan={self.subscription_plan})>"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add computed fields
        data["is_trial_active"] = self.is_trial_active
        data["is_subscription_active"] = self.is_subscription_active
        data["days_until_expiration"] = self.days_until_expiration
        data["plan_display"] = self.subscription_plan.display_name
        data["plan_limits"] = self.subscription_plan.limits
        data["plan_features"] = self.subscription_plan.features
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Tenant",
    "SubscriptionPlan",
    "TenantStatus",
    "OrganizationType",
]