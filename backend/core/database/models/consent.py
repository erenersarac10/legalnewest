"""
Consent model for KVKK consent management in Turkish Legal AI.

This module provides the Consent model for tracking user consents:
- Explicit consent tracking (KVKK Article 5/2-a)
- Granular consent management (by purpose and category)
- Consent withdrawal support
- Consent history and audit trail
- Multi-version consent texts
- Consent expiration and renewal
- KVKK-compliant consent documentation

KVKK Requirements:
    - Article 5/2-a: Explicit consent required
    - Article 3: Consent must be freely given, specific, informed
    - Consent must be verifiable (who, when, what, how)
    - Withdrawal must be as easy as giving consent
    - Consent records must be kept for 6 years

Consent Types:
    - MARKETING: Marketing communications
    - ANALYTICS: Usage analytics and tracking
    - THIRD_PARTY: Third-party data sharing
    - PROFILING: Automated decision making
    - LOCATION: Location data processing
    - SENSITIVE: Sensitive personal data (race, religion, health)

Consent Channels:
    - WEB: Website consent form
    - MOBILE: Mobile app consent
    - EMAIL: Email confirmation
    - CALL: Phone call
    - IN_PERSON: Physical document

Example:
    >>> # Record user consent
    >>> consent = Consent.record_consent(
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ...     consent_type=ConsentType.MARKETING,
    ...     purpose="E-posta ile pazarlama iletişimi",
    ...     data_categories=["iletişim", "müşteri işlem"],
    ...     channel=ConsentChannel.WEB,
    ...     consent_text="Pazarlama e-postaları almayı kabul ediyorum",
    ...     ip_address=request.client.host,
    ...     user_agent=request.headers.get("User-Agent")
    ... )
    >>> 
    >>> # Withdraw consent
    >>> consent.withdraw(
    ...     reason="Kullanıcı tarafından iptal edildi",
    ...     channel=ConsentChannel.WEB
    ... )
"""

import enum
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID as UUIDType

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
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
    SoftDeleteMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ConsentType(str, enum.Enum):
    """
    Consent type classification.
    
    Types:
    - MARKETING: Marketing communications (email, SMS, push)
    - ANALYTICS: Usage analytics and behavioral tracking
    - THIRD_PARTY: Data sharing with third parties
    - PROFILING: Automated decision making and profiling
    - LOCATION: Location data processing
    - SENSITIVE: Sensitive personal data (health, religion, etc.)
    - COOKIES: Cookie consent (technical, analytics, marketing)
    - RESEARCH: Data usage for research purposes
    """
    
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    THIRD_PARTY = "third_party"
    PROFILING = "profiling"
    LOCATION = "location"
    SENSITIVE = "sensitive"
    COOKIES = "cookies"
    RESEARCH = "research"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.MARKETING: "Pazarlama İletişimi",
            self.ANALYTICS: "Analitik ve İstatistik",
            self.THIRD_PARTY: "Üçüncü Taraf Paylaşımı",
            self.PROFILING: "Otomatik Karar ve Profilleme",
            self.LOCATION: "Konum Verisi",
            self.SENSITIVE: "Özel Nitelikli Kişisel Veri",
            self.COOKIES: "Çerez Kullanımı",
            self.RESEARCH: "Araştırma Amaçlı Kullanım",
        }
        return names.get(self, self.value)


class ConsentStatus(str, enum.Enum):
    """Consent status."""
    
    GRANTED = "granted"          # Consent given
    WITHDRAWN = "withdrawn"      # Consent withdrawn
    EXPIRED = "expired"          # Consent expired (if has expiration)
    PENDING = "pending"          # Waiting for confirmation (double opt-in)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.GRANTED: "Verildi",
            self.WITHDRAWN: "Geri Çekildi",
            self.EXPIRED: "Süresi Doldu",
            self.PENDING: "Bekliyor",
        }
        return names.get(self, self.value)


class ConsentChannel(str, enum.Enum):
    """
    Channel through which consent was obtained.
    
    Channels:
    - WEB: Website consent form
    - MOBILE: Mobile app
    - EMAIL: Email confirmation link
    - CALL: Phone call
    - IN_PERSON: Physical document/signature
    - API: API request
    """
    
    WEB = "web"
    MOBILE = "mobile"
    EMAIL = "email"
    CALL = "call"
    IN_PERSON = "in_person"
    API = "api"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.WEB: "Web Sitesi",
            self.MOBILE: "Mobil Uygulama",
            self.EMAIL: "E-posta",
            self.CALL: "Telefon",
            self.IN_PERSON: "Yüz Yüze",
            self.API: "API",
        }
        return names.get(self, self.value)


# =============================================================================
# CONSENT MODEL
# =============================================================================


class Consent(Base, BaseModelMixin, TenantMixin, SoftDeleteMixin):
    """
    Consent model for KVKK-compliant consent management.
    
    Tracks user consents for data processing:
    - Explicit consent recording
    - Granular consent by purpose
    - Consent withdrawal
    - Consent history
    - Verifiable audit trail
    
    KVKK Compliance:
        - Article 5/2-a: Explicit consent
        - Article 3: Freely given, specific, informed
        - Verifiable records (who, when, what, how)
        - Easy withdrawal process
        - 6 year retention
    
    Attributes:
        user_id: User who gave consent
        user: User relationship
        
        consent_type: Type of consent
        purpose: Specific processing purpose
        data_categories: Data categories covered (array)
        
        status: Current status (granted, withdrawn, expired)
        
        granted_at: When consent was given
        withdrawn_at: When consent was withdrawn
        expires_at: Consent expiration date (if applicable)
        
        consent_text: Consent text shown to user
        consent_version: Version of consent text
        
        channel: How consent was obtained
        
        ip_address: IP address when consent given
        user_agent: User agent string
        geolocation: Geographic location (JSON)
        
        is_explicit: Explicit consent (not pre-checked)
        is_informed: User was properly informed
        
        withdrawal_reason: Why consent was withdrawn
        withdrawal_channel: How consent was withdrawn
        
        parent_consent_id: Parent consent (for renewals/updates)
        parent_consent: Parent relationship
        
        metadata: Additional context (form data, etc.)
        
        proof_document_id: Document proving consent (if physical)
        
    Relationships:
        tenant: Parent tenant
        user: User who gave consent
        parent_consent: Previous consent version
    """
    
    __tablename__ = "consents"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who gave consent",
    )
    
    user = relationship(
        "User",
        back_populates="consents",
    )
    
    # =========================================================================
    # CONSENT CLASSIFICATION
    # =========================================================================
    
    consent_type = Column(
        Enum(ConsentType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of consent (marketing, analytics, etc.)",
    )
    
    purpose = Column(
        Text,
        nullable=False,
        comment="Specific processing purpose (must be clear and specific)",
    )
    
    data_categories = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Data categories covered by this consent",
    )
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    status = Column(
        Enum(ConsentStatus, native_enum=False, length=50),
        nullable=False,
        default=ConsentStatus.GRANTED,
        index=True,
        comment="Current consent status",
    )
    
    # =========================================================================
    # TIMESTAMPS
    # =========================================================================
    
    granted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="When consent was given",
    )
    
    withdrawn_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When consent was withdrawn",
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Consent expiration date (if applicable)",
    )
    
    # =========================================================================
    # CONSENT TEXT & VERSION
    # =========================================================================
    
    consent_text = Column(
        Text,
        nullable=False,
        comment="Exact consent text shown to user (for proof)",
    )
    
    consent_version = Column(
        String(50),
        nullable=False,
        comment="Version of consent text (e.g., 'v1.0', '2025-01-15')",
    )
    
    # =========================================================================
    # CHANNEL & METHOD
    # =========================================================================
    
    channel = Column(
        Enum(ConsentChannel, native_enum=False, length=50),
        nullable=False,
        comment="Channel through which consent was obtained",
    )
    
    # =========================================================================
    # VERIFICATION DATA
    # =========================================================================
    
    ip_address = Column(
        String(45),  # IPv6 max length
        nullable=True,
        comment="IP address when consent was given",
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="User agent string",
    )
    
    geolocation = Column(
        JSONB,
        nullable=True,
        comment="Geographic location data (city, country)",
    )
    
    # =========================================================================
    # CONSENT QUALITY FLAGS
    # =========================================================================
    
    is_explicit = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Explicit consent (not pre-checked, clear action required)",
    )
    
    is_informed = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="User was properly informed before giving consent",
    )
    
    # =========================================================================
    # WITHDRAWAL INFORMATION
    # =========================================================================
    
    withdrawal_reason = Column(
        Text,
        nullable=True,
        comment="Reason for withdrawal",
    )
    
    withdrawal_channel = Column(
        Enum(ConsentChannel, native_enum=False, length=50),
        nullable=True,
        comment="Channel used to withdraw consent",
    )
    
    # =========================================================================
    # CONSENT HISTORY (Parent-Child)
    # =========================================================================
    
    parent_consent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("consents.id", ondelete="SET NULL"),
        nullable=True,
        comment="Previous consent (for renewals/updates)",
    )
    
    parent_consent = relationship(
        "Consent",
        remote_side="Consent.id",
        backref="child_consents",
        foreign_keys=[parent_consent_id],
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context (form fields, referrer, etc.)",
    )
    
    proof_document_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Document ID if physical proof exists (scanned form)",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user's active consents
        Index(
            "ix_consents_user_active",
            "user_id",
            "consent_type",
            "status",
            postgresql_where="status = 'granted' AND deleted_at IS NULL",
        ),
        
        # Index for consent type queries
        Index(
            "ix_consents_type",
            "tenant_id",
            "consent_type",
            "status",
        ),
        
        # Index for expiring consents
        Index(
            "ix_consents_expiring",
            "expires_at",
            postgresql_where="expires_at IS NOT NULL AND status = 'granted'",
        ),
        
        # Index for withdrawn consents
        Index(
            "ix_consents_withdrawn",
            "user_id",
            "withdrawn_at",
            postgresql_where="status = 'withdrawn'",
        ),
        
        # Index for consent history
        Index(
            "ix_consents_history",
            "user_id",
            "consent_type",
            "granted_at",
        ),
    )
    
    # =========================================================================
    # CONSENT RECORDING
    # =========================================================================
    
    @classmethod
    def record_consent(
        cls,
        user_id: UUIDType,
        tenant_id: UUIDType,
        consent_type: ConsentType,
        purpose: str,
        consent_text: str,
        consent_version: str,
        data_categories: list[str],
        channel: ConsentChannel,
        ip_address: str | None = None,
        user_agent: str | None = None,
        is_explicit: bool = True,
        is_informed: bool = True,
        expires_in_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Consent":
        """
        Record a new consent.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant UUID
            consent_type: Type of consent
            purpose: Processing purpose
            consent_text: Exact text shown to user
            consent_version: Version identifier
            data_categories: Data categories covered
            channel: How consent was obtained
            ip_address: IP address
            user_agent: User agent
            is_explicit: Explicit consent flag
            is_informed: Informed consent flag
            expires_in_days: Expiration in days (optional)
            metadata: Additional context
            
        Returns:
            Consent: New consent record
            
        Example:
            >>> consent = Consent.record_consent(
            ...     user_id=user.id,
            ...     tenant_id=tenant.id,
            ...     consent_type=ConsentType.MARKETING,
            ...     purpose="E-posta ile pazarlama iletişimi göndermek",
            ...     consent_text="Pazarlama e-postaları almayı kabul ediyorum",
            ...     consent_version="v2.0",
            ...     data_categories=["iletişim", "müşteri işlem"],
            ...     channel=ConsentChannel.WEB,
            ...     ip_address="192.168.1.100",
            ...     is_explicit=True,
            ...     is_informed=True
            ... )
        """
        # Calculate expiration if specified
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        consent = cls(
            user_id=user_id,
            tenant_id=tenant_id,
            consent_type=consent_type,
            purpose=purpose,
            consent_text=consent_text,
            consent_version=consent_version,
            data_categories=data_categories,
            channel=channel,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent,
            is_explicit=is_explicit,
            is_informed=is_informed,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        logger.info(
            "Consent recorded",
            consent_id=str(consent.id),
            user_id=str(user_id),
            consent_type=consent_type.value,
            channel=channel.value,
        )
        
        return consent
    
    # =========================================================================
    # CONSENT MANAGEMENT
    # =========================================================================
    
    def withdraw(
        self,
        reason: str | None = None,
        channel: ConsentChannel | None = None,
    ) -> None:
        """
        Withdraw consent.
        
        Args:
            reason: Withdrawal reason
            channel: How consent was withdrawn
            
        Example:
            >>> consent.withdraw(
            ...     reason="Kullanıcı tarafından iptal edildi",
            ...     channel=ConsentChannel.WEB
            ... )
        """
        self.status = ConsentStatus.WITHDRAWN
        self.withdrawn_at = datetime.now(timezone.utc)
        self.withdrawal_reason = reason
        self.withdrawal_channel = channel
        
        logger.info(
            "Consent withdrawn",
            consent_id=str(self.id),
            user_id=str(self.user_id),
            consent_type=self.consent_type.value,
            reason=reason,
        )
    
    def renew(
        self,
        new_consent_text: str | None = None,
        new_version: str | None = None,
        expires_in_days: int | None = None,
    ) -> "Consent":
        """
        Renew consent (create new consent record).
        
        Args:
            new_consent_text: Updated consent text (if changed)
            new_version: New version identifier
            expires_in_days: New expiration period
            
        Returns:
            Consent: New consent record
            
        Example:
            >>> new_consent = consent.renew(
            ...     new_version="v3.0",
            ...     expires_in_days=365
            ... )
        """
        # Create new consent record
        new_consent = Consent.record_consent(
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            consent_type=self.consent_type,
            purpose=self.purpose,
            consent_text=new_consent_text or self.consent_text,
            consent_version=new_version or self.consent_version,
            data_categories=self.data_categories,
            channel=self.channel,
            is_explicit=self.is_explicit,
            is_informed=self.is_informed,
            expires_in_days=expires_in_days,
        )
        
        # Link to parent
        new_consent.parent_consent_id = self.id
        
        # Expire old consent
        self.status = ConsentStatus.EXPIRED
        
        logger.info(
            "Consent renewed",
            old_consent_id=str(self.id),
            new_consent_id=str(new_consent.id),
            user_id=str(self.user_id),
        )
        
        return new_consent
    
    def mark_expired(self) -> None:
        """Mark consent as expired."""
        self.status = ConsentStatus.EXPIRED
        
        logger.info(
            "Consent expired",
            consent_id=str(self.id),
            user_id=str(self.user_id),
        )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def is_valid(self) -> bool:
        """
        Check if consent is currently valid.
        
        Returns:
            bool: True if consent is granted and not expired
        """
        if self.status != ConsentStatus.GRANTED:
            return False
        
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        
        return True
    
    def is_expired(self) -> bool:
        """Check if consent has expired."""
        if not self.expires_at:
            return False
        
        return datetime.now(timezone.utc) > self.expires_at
    
    def days_until_expiration(self) -> int | None:
        """
        Get days until expiration.
        
        Returns:
            int | None: Days remaining or None if no expiration
        """
        if not self.expires_at:
            return None
        
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    # =========================================================================
    # QUERY HELPERS
    # =========================================================================
    
    @classmethod
    def get_active_consent(
        cls,
        user_id: UUIDType,
        consent_type: ConsentType,
    ) -> "Consent | None":
        """
        Get user's active consent for specific type.
        
        Args:
            user_id: User UUID
            consent_type: Consent type
            
        Returns:
            Consent | None: Active consent or None
            
        Note:
            This would be implemented in a service/repository layer
        """
        # Placeholder for query logic
        # session.query(Consent).filter(
        #     Consent.user_id == user_id,
        #     Consent.consent_type == consent_type,
        #     Consent.status == ConsentStatus.GRANTED,
        #     Consent.deleted_at.is_(None)
        # ).first()
        return None
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("purpose")
    def validate_purpose(self, key: str, purpose: str) -> str:
        """Validate purpose."""
        if not purpose or not purpose.strip():
            raise ValidationError(
                message="Purpose cannot be empty",
                field="purpose",
            )
        
        return purpose.strip()
    
    @validates("consent_text")
    def validate_consent_text(self, key: str, consent_text: str) -> str:
        """Validate consent text."""
        if not consent_text or not consent_text.strip():
            raise ValidationError(
                message="Consent text cannot be empty",
                field="consent_text",
            )
        
        return consent_text.strip()
    
    @validates("data_categories")
    def validate_data_categories(
        self,
        key: str,
        data_categories: list[str],
    ) -> list[str]:
        """Validate data categories."""
        if not data_categories:
            raise ValidationError(
                message="At least one data category must be specified",
                field="data_categories",
            )
        
        return data_categories
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Consent("
            f"id={self.id}, "
            f"type={self.consent_type.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add display names
        data["consent_type_display"] = self.consent_type.display_name_tr
        data["status_display"] = self.status.display_name_tr
        data["channel_display"] = self.channel.display_name_tr
        
        if self.withdrawal_channel:
            data["withdrawal_channel_display"] = self.withdrawal_channel.display_name_tr
        
        # Add computed fields
        data["is_valid"] = self.is_valid()
        data["is_expired"] = self.is_expired()
        data["days_until_expiration"] = self.days_until_expiration()
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Consent",
    "ConsentType",
    "ConsentStatus",
    "ConsentChannel",
]