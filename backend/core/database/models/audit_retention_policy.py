"""
Audit Retention Policy Database Model for Turkish Legal AI.

This module defines retention policies for audit data:
- GDPR / KVKK compliance (data retention requirements)
- Automatic data expiration
- Multi-tier retention (hot/warm/cold/archive)
- Legal hold support
- Tenant-specific policies

Retention Tiers:
    - HOT: Active data (0-30 days) - Fast SSD storage
    - WARM: Recent data (30-180 days) - Standard storage
    - COLD: Archival data (180-365 days) - Glacier/S3 IA
    - ARCHIVE: Long-term archive (1-7 years) - Deep Glacier

Compliance Frameworks:
    - GDPR: 30 days to 7 years (depending on data type)
    - KVKK: Turkish data protection law
    - SOC 2: 1 year minimum
    - ISO 27001: 1 year minimum

Example:
    >>> from backend.core.database.models.audit_retention_policy import AuditRetentionPolicy
    >>>
    >>> # Create GDPR-compliant policy
    >>> policy = AuditRetentionPolicy(
    ...     tenant_id=tenant_id,
    ...     name="GDPR Standard Retention",
    ...     retention_days=365,
    ...     compliance_framework="GDPR"
    ... )
"""

import datetime
import enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

from backend.core.database.base import Base


# =============================================================================
# ENUMS
# =============================================================================


class RetentionTier(str, enum.Enum):
    """Data retention storage tiers."""

    HOT = "hot"  # 0-30 days: Fast SSD, high cost
    WARM = "warm"  # 30-180 days: Standard storage
    COLD = "cold"  # 180-365 days: S3 Glacier/IA
    ARCHIVE = "archive"  # 1-7 years: Deep Glacier


class ComplianceFramework(str, enum.Enum):
    """Compliance frameworks for retention requirements."""

    GDPR = "gdpr"  # EU General Data Protection Regulation
    KVKK = "kvkk"  # Turkish Personal Data Protection Law
    SOC2 = "soc2"  # SOC 2 Type II
    ISO27001 = "iso27001"  # ISO 27001
    HIPAA = "hipaa"  # Health Insurance Portability
    PCI_DSS = "pci_dss"  # Payment Card Industry
    CUSTOM = "custom"  # Custom compliance requirements


class DataCategory(str, enum.Enum):
    """Data categories for retention policies."""

    AUDIT_LOG = "audit_log"  # Audit logs
    ACCESS_LOG = "access_log"  # Access logs
    CHANGE_LOG = "change_log"  # Change logs
    SECURITY_EVENT = "security_event"  # Security events
    USER_DATA = "user_data"  # User personal data
    DOCUMENT_DATA = "document_data"  # Document metadata
    FINANCIAL_DATA = "financial_data"  # Financial transactions
    COMMUNICATION = "communication"  # Email, chat, etc.


# =============================================================================
# AUDIT RETENTION POLICY MODEL
# =============================================================================


class AuditRetentionPolicy(Base):
    """
    Audit retention policy model for data lifecycle management.

    This model defines how long different types of audit data
    should be retained based on compliance requirements.

    Features:
    - Multi-tier retention (hot/warm/cold/archive)
    - Automatic expiration
    - Legal hold support
    - Compliance framework mapping
    - Tenant-specific policies
    """

    __tablename__ = "audit_retention_policies"

    # =========================================================================
    # PRIMARY KEY
    # =========================================================================

    id: UUID = Column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        index=True,
    )

    # =========================================================================
    # TENANT & OWNERSHIP
    # =========================================================================

    tenant_id: UUID = Column(
        PGUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_by_id: UUID = Column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # =========================================================================
    # POLICY CONFIGURATION
    # =========================================================================

    name: str = Column(
        String(255),
        nullable=False,
        doc="Policy name",
    )

    description: Optional[str] = Column(
        Text,
        nullable=True,
        doc="Policy description",
    )

    data_category: DataCategory = Column(
        Enum(DataCategory),
        nullable=False,
        index=True,
        doc="Type of data this policy applies to",
    )

    # =========================================================================
    # RETENTION CONFIGURATION
    # =========================================================================

    # Total retention period in days
    retention_days: int = Column(
        Integer,
        nullable=False,
        doc="Total retention period in days",
    )

    # Tier transition configuration
    hot_tier_days: int = Column(
        Integer,
        nullable=False,
        default=30,
        doc="Days in HOT tier (fast SSD)",
    )

    warm_tier_days: int = Column(
        Integer,
        nullable=False,
        default=150,
        doc="Days in WARM tier (standard storage)",
    )

    cold_tier_days: int = Column(
        Integer,
        nullable=False,
        default=185,
        doc="Days in COLD tier (Glacier/S3 IA)",
    )

    # After total retention_days, data is deleted
    # archive_tier handles anything beyond cold_tier_days

    # =========================================================================
    # COMPLIANCE
    # =========================================================================

    compliance_framework: ComplianceFramework = Column(
        Enum(ComplianceFramework),
        nullable=False,
        index=True,
        doc="Compliance framework this policy satisfies",
    )

    # Legal requirements reference
    legal_reference: Optional[str] = Column(
        Text,
        nullable=True,
        doc="Legal/regulatory reference (article, section)",
    )

    # Minimum retention required by law
    minimum_retention_days: int = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Minimum retention required by law (cannot delete before this)",
    )

    # =========================================================================
    # LEGAL HOLD
    # =========================================================================

    supports_legal_hold: bool = Column(
        Boolean,
        nullable=False,
        default=True,
        doc="Can this data be placed on legal hold?",
    )

    # =========================================================================
    # DELETION POLICY
    # =========================================================================

    auto_delete_enabled: bool = Column(
        Boolean,
        nullable=False,
        default=True,
        doc="Automatically delete data after retention period",
    )

    # Require manual approval before deletion
    require_approval: bool = Column(
        Boolean,
        nullable=False,
        default=False,
        doc="Require manual approval before deletion",
    )

    # Deletion method
    secure_delete: bool = Column(
        Boolean,
        nullable=False,
        default=True,
        doc="Use secure deletion (overwrite, DOD 5220.22-M)",
    )

    # =========================================================================
    # POLICY STATUS
    # =========================================================================

    is_active: bool = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        doc="Is this policy currently active?",
    )

    is_system: bool = Column(
        Boolean,
        nullable=False,
        default=False,
        doc="Is this a system-defined policy (immutable)?",
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    # Number of records currently under this policy
    records_count: int = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of records under this policy",
    )

    # Total storage size in bytes
    storage_bytes: int = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Total storage used by records under this policy",
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata: Dict[str, Any] = Column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Additional metadata",
    )

    # =========================================================================
    # AUDIT FIELDS
    # =========================================================================

    created_at: datetime.datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.datetime.utcnow,
        index=True,
    )

    updated_at: datetime.datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    deleted_at: Optional[datetime.datetime] = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Soft delete timestamp",
    )

    # =========================================================================
    # INDEXES & CONSTRAINTS
    # =========================================================================

    __table_args__ = (
        # Tenant + category unique constraint
        Index(
            "ix_audit_retention_policies_tenant_category",
            "tenant_id",
            "data_category",
            unique=True,
            postgresql_where=Column("deleted_at").is_(None),
        ),
        # Active policies index
        Index(
            "ix_audit_retention_policies_active",
            "is_active",
            "compliance_framework",
            postgresql_where=Column("is_active") == True,
        ),
        # Compliance framework index
        Index(
            "ix_audit_retention_policies_compliance",
            "compliance_framework",
            "data_category",
        ),
    )

    # =========================================================================
    # METHODS
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<AuditRetentionPolicy(id={self.id}, name='{self.name}', "
            f"category={self.data_category}, days={self.retention_days})>"
        )

    def get_tier_for_age(self, age_days: int) -> RetentionTier:
        """
        Determine storage tier based on data age.

        Args:
            age_days: Age of data in days

        Returns:
            Appropriate RetentionTier
        """
        if age_days <= self.hot_tier_days:
            return RetentionTier.HOT
        elif age_days <= self.warm_tier_days:
            return RetentionTier.WARM
        elif age_days <= self.cold_tier_days:
            return RetentionTier.COLD
        else:
            return RetentionTier.ARCHIVE

    def should_delete(self, age_days: int) -> bool:
        """
        Check if data should be deleted based on age.

        Args:
            age_days: Age of data in days

        Returns:
            True if data should be deleted
        """
        if not self.auto_delete_enabled:
            return False

        # Cannot delete before minimum retention
        if age_days < self.minimum_retention_days:
            return False

        # Delete if past total retention period
        return age_days >= self.retention_days

    def is_within_minimum_retention(self, age_days: int) -> bool:
        """
        Check if data is within minimum legal retention period.

        Args:
            age_days: Age of data in days

        Returns:
            True if within minimum retention
        """
        return age_days < self.minimum_retention_days

    def calculate_expiration_date(
        self,
        created_at: datetime.datetime,
    ) -> datetime.datetime:
        """
        Calculate expiration date for data.

        Args:
            created_at: Data creation timestamp

        Returns:
            Expiration datetime
        """
        return created_at + datetime.timedelta(days=self.retention_days)
