"""
Compliance Record model for KVKK compliance tracking in Turkish Legal AI.

This module provides the ComplianceRecord model for tracking compliance activities:
- KVKK compliance documentation
- Data processing records (VVT - Veri İşleme Envanteri)
- Privacy impact assessments (DPIA)
- Data breach notifications
- Audit findings and remediation
- Compliance certifications
- Third-party assessments
- Legal basis documentation

KVKK Requirements:
    - Article 10: Right to information
    - Article 11: Right to access
    - Article 12: Right to correction
    - Article 13: Right to deletion
    - Article 14: Right to objection
    - Data Processing Inventory (VVT)
    - Data Protection Impact Assessment (DPIA)
    - Breach notification (72 hours)

Record Types:
    - VVT: Data Processing Inventory
    - DPIA: Data Protection Impact Assessment
    - BREACH: Data breach notification
    - AUDIT: Audit findings
    - CERTIFICATION: Compliance certifications
    - LEGAL_BASIS: Legal basis documentation
    - CONSENT: Consent management records
    - THIRD_PARTY: Third-party processor assessments

Example:
    >>> # Create VVT record
    >>> record = ComplianceRecord(
    ...     tenant_id=tenant.id,
    ...     record_type=ComplianceRecordType.VVT,
    ...     title="Kullanıcı Verileri İşleme Envanteri",
    ...     description="Platform kullanıcı verilerinin işleme amaçları",
    ...     status=ComplianceStatus.ACTIVE,
    ...     legal_basis=["KVKK Madde 5/2-a", "KVKK Madde 5/2-ç"],
    ...     data_categories=["kimlik", "iletişim", "müşteri işlem"],
    ...     retention_period_months=72,  # 6 years
    ...     responsible_person_id=dpo_user.id
    ... )
    >>> 
    >>> # Create breach notification
    >>> breach = ComplianceRecord(
    ...     tenant_id=tenant.id,
    ...     record_type=ComplianceRecordType.BREACH,
    ...     title="Veri İhlali Bildirimi - 2025-001",
    ...     severity=ComplianceSeverity.HIGH,
    ...     status=ComplianceStatus.REPORTED,
    ...     breach_date=datetime.now(timezone.utc),
    ...     notification_required=True
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


class ComplianceRecordType(str, enum.Enum):
    """
    Compliance record type.
    
    Types:
    - VVT: Veri İşleme Envanteri (Data Processing Inventory)
    - DPIA: Data Protection Impact Assessment
    - BREACH: Data breach notification
    - AUDIT: Compliance audit findings
    - CERTIFICATION: Compliance certifications (ISO 27001, etc.)
    - LEGAL_BASIS: Legal basis documentation
    - CONSENT: Consent management records
    - THIRD_PARTY: Third-party processor assessments
    - POLICY: Privacy policy updates
    - TRAINING: Staff training records
    """
    
    VVT = "vvt"                          # Veri İşleme Envanteri
    DPIA = "dpia"                        # Data Protection Impact Assessment
    BREACH = "breach"                    # Data breach
    AUDIT = "audit"                      # Audit finding
    CERTIFICATION = "certification"      # Certification (ISO 27001)
    LEGAL_BASIS = "legal_basis"          # Legal basis documentation
    CONSENT = "consent"                  # Consent records
    THIRD_PARTY = "third_party"          # Third-party assessment
    POLICY = "policy"                    # Policy update
    TRAINING = "training"                # Training record
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.VVT: "Veri İşleme Envanteri",
            self.DPIA: "Veri Koruma Etki Değerlendirmesi",
            self.BREACH: "Veri İhlali",
            self.AUDIT: "Denetim Bulgusu",
            self.CERTIFICATION: "Sertifikasyon",
            self.LEGAL_BASIS: "Hukuki Dayanak",
            self.CONSENT: "Onay Kaydı",
            self.THIRD_PARTY: "Üçüncü Taraf Değerlendirmesi",
            self.POLICY: "Politika Güncellemesi",
            self.TRAINING: "Eğitim Kaydı",
        }
        return names.get(self, self.value)


class ComplianceStatus(str, enum.Enum):
    """Compliance record status."""
    
    DRAFT = "draft"                  # Draft (being prepared)
    ACTIVE = "active"                # Active/approved
    UNDER_REVIEW = "under_review"    # Under review
    REPORTED = "reported"            # Reported to authority (KVKK)
    RESOLVED = "resolved"            # Issue resolved
    ARCHIVED = "archived"            # Archived (historical)
    EXPIRED = "expired"              # Expired certification/policy
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.DRAFT: "Taslak",
            self.ACTIVE: "Aktif",
            self.UNDER_REVIEW: "İnceleme Altında",
            self.REPORTED: "Bildirildi",
            self.RESOLVED: "Çözüldü",
            self.ARCHIVED: "Arşivlendi",
            self.EXPIRED: "Süresi Doldu",
        }
        return names.get(self, self.value)


class ComplianceSeverity(str, enum.Enum):
    """Compliance issue severity."""
    
    LOW = "low"              # Low impact
    MEDIUM = "medium"        # Medium impact
    HIGH = "high"            # High impact
    CRITICAL = "critical"    # Critical impact (requires immediate action)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.LOW: "Düşük",
            self.MEDIUM: "Orta",
            self.HIGH: "Yüksek",
            self.CRITICAL: "Kritik",
        }
        return names.get(self, self.value)


class LegalBasis(str, enum.Enum):
    """
    KVKK legal basis for data processing.
    
    KVKK Article 5/2:
    """
    
    EXPLICIT_CONSENT = "explicit_consent"              # Açık rıza (5/2-a)
    LAW = "law"                                        # Kanunda açıkça öngörülme (5/2-b)
    CONTRACT = "contract"                              # Sözleşme ifası (5/2-c)
    LEGAL_OBLIGATION = "legal_obligation"              # Hukuki yükümlülük (5/2-ç)
    VITAL_INTEREST = "vital_interest"                  # İlgili kişinin yaşamı (5/2-d)
    PUBLIC_DISCLOSURE = "public_disclosure"            # İlgili kişi tarafından alenileştirilme (5/2-e)
    LEGITIMATE_INTEREST = "legitimate_interest"        # Meşru menfaat (5/2-f)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name with KVKK article reference."""
        names = {
            self.EXPLICIT_CONSENT: "Açık Rıza (KVKK 5/2-a)",
            self.LAW: "Kanunda Öngörülme (KVKK 5/2-b)",
            self.CONTRACT: "Sözleşme İfası (KVKK 5/2-c)",
            self.LEGAL_OBLIGATION: "Hukuki Yükümlülük (KVKK 5/2-ç)",
            self.VITAL_INTEREST: "Yaşamsal Çıkar (KVKK 5/2-d)",
            self.PUBLIC_DISCLOSURE: "Aleniyet (KVKK 5/2-e)",
            self.LEGITIMATE_INTEREST: "Meşru Menfaat (KVKK 5/2-f)",
        }
        return names.get(self, self.value)


# =============================================================================
# COMPLIANCE RECORD MODEL
# =============================================================================


class ComplianceRecord(Base, BaseModelMixin, TenantMixin, AuditMixin, SoftDeleteMixin):
    """
    Compliance Record model for KVKK compliance tracking.
    
    Tracks all compliance-related activities:
    - Data processing inventory (VVT)
    - Privacy impact assessments (DPIA)
    - Data breach notifications
    - Audit findings
    - Certifications
    - Legal basis documentation
    
    KVKK Compliance:
        - Article 10: Right to information
        - Article 16: Data Processing Inventory
        - Article 12: Breach notification (72 hours)
        - 6 year retention for legal records
    
    Attributes:
        record_type: Type of compliance record
        title: Record title
        description: Detailed description
        
        status: Current status
        severity: Issue severity (if applicable)
        
        legal_basis: KVKK legal basis (array)
        data_categories: Data categories processed (array)
        processing_purposes: Processing purposes (array)
        
        responsible_person_id: DPO or responsible person
        responsible_person: User relationship
        
        start_date: Record effective date
        end_date: Record expiration date
        review_date: Next review date
        
        retention_period_months: Data retention period
        
        breach_date: Date of breach (if applicable)
        breach_discovered_date: When breach was discovered
        notification_required: Requires KVKK notification
        notification_date: When authority was notified
        affected_count: Number of affected individuals
        
        findings: Audit findings (JSON)
        recommendations: Recommendations (JSON)
        remediation_plan: Remediation steps (JSON)
        
        documents: Related document IDs (array)
        evidence: Evidence/attachments (JSON)
        
        metadata: Additional metadata
        
        review_notes: Review notes/comments
        
    Relationships:
        tenant: Parent tenant
        responsible_person: DPO or responsible user
    """
    
    __tablename__ = "compliance_records"
    
    # =========================================================================
    # RECORD CLASSIFICATION
    # =========================================================================
    
    record_type = Column(
        Enum(ComplianceRecordType, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Type of compliance record",
    )
    
    title = Column(
        String(500),
        nullable=False,
        comment="Record title",
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Detailed description of the record",
    )
    
    # =========================================================================
    # STATUS & SEVERITY
    # =========================================================================
    
    status = Column(
        Enum(ComplianceStatus, native_enum=False, length=50),
        nullable=False,
        default=ComplianceStatus.DRAFT,
        index=True,
        comment="Record status",
    )
    
    severity = Column(
        Enum(ComplianceSeverity, native_enum=False, length=50),
        nullable=True,
        comment="Issue severity (for breaches, findings)",
    )
    
    # =========================================================================
    # LEGAL BASIS & DATA CATEGORIES
    # =========================================================================
    
    legal_basis = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="KVKK legal basis for processing (Article 5/2)",
    )
    
    data_categories = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Data categories processed (kimlik, iletişim, müşteri işlem, etc.)",
    )
    
    processing_purposes = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Data processing purposes",
    )
    
    # =========================================================================
    # RESPONSIBILITY
    # =========================================================================
    
    responsible_person_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Data Protection Officer or responsible person",
    )
    
    responsible_person = relationship(
        "User",
        back_populates="compliance_records",
        foreign_keys=[responsible_person_id],
    )
    
    # =========================================================================
    # DATES
    # =========================================================================
    
    start_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Record effective/start date",
    )
    
    end_date = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Record expiration/end date",
    )
    
    review_date = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Next review date",
    )
    
    # =========================================================================
    # RETENTION
    # =========================================================================
    
    retention_period_months = Column(
        Integer,
        nullable=True,
        comment="Data retention period in months (KVKK requirement)",
    )
    
    # =========================================================================
    # BREACH INFORMATION
    # =========================================================================
    
    breach_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Date when breach occurred",
    )
    
    breach_discovered_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Date when breach was discovered",
    )
    
    notification_required = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Requires notification to KVKK/authorities",
    )
    
    notification_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Date when authorities were notified",
    )
    
    affected_count = Column(
        Integer,
        nullable=True,
        comment="Number of affected individuals (for breaches)",
    )
    
    # =========================================================================
    # FINDINGS & REMEDIATION
    # =========================================================================
    
    findings = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Audit findings or identified issues (array of objects)",
    )
    
    recommendations = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Recommendations for improvement (array)",
    )
    
    remediation_plan = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Remediation plan with steps and timeline",
    )
    
    # =========================================================================
    # DOCUMENTATION
    # =========================================================================
    
    documents = Column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
        comment="Related document IDs (evidence, reports)",
    )
    
    evidence = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Evidence and attachments metadata (array)",
    )
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (certifications, third-party info, etc.)",
    )
    
    review_notes = Column(
        Text,
        nullable=True,
        comment="Review notes and comments",
    )
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for record type queries
        Index(
            "ix_compliance_records_type",
            "tenant_id",
            "record_type",
            "status",
        ),
        
        # Index for breach notifications
        Index(
            "ix_compliance_records_breach",
            "record_type",
            "breach_date",
            postgresql_where="record_type = 'breach'",
        ),
        
        # Index for review dates (upcoming reviews)
        Index(
            "ix_compliance_records_review",
            "review_date",
            postgresql_where="review_date IS NOT NULL AND status = 'active'",
        ),
        
        # Index for expiration dates
        Index(
            "ix_compliance_records_expiration",
            "end_date",
            postgresql_where="end_date IS NOT NULL",
        ),
        
        # Index for active VVT records
        Index(
            "ix_compliance_records_vvt",
            "tenant_id",
            "status",
            postgresql_where="record_type = 'vvt' AND status = 'active'",
        ),
        
        # Check: retention period positive
        CheckConstraint(
            "retention_period_months IS NULL OR retention_period_months > 0",
            name="ck_compliance_records_retention",
        ),
        
        # Check: affected count non-negative
        CheckConstraint(
            "affected_count IS NULL OR affected_count >= 0",
            name="ck_compliance_records_affected_count",
        ),
    )
    
    # =========================================================================
    # RECORD CREATION
    # =========================================================================
    
    @classmethod
    def create_vvt_record(
        cls,
        tenant_id: UUIDType,
        title: str,
        description: str,
        data_categories: list[str],
        processing_purposes: list[str],
        legal_basis: list[str],
        retention_period_months: int,
        responsible_person_id: UUIDType,
    ) -> "ComplianceRecord":
        """
        Create VVT (Data Processing Inventory) record.
        
        Example:
            >>> vvt = ComplianceRecord.create_vvt_record(
            ...     tenant_id=tenant.id,
            ...     title="Kullanıcı Verileri İşleme Envanteri",
            ...     description="Platform kullanıcılarının kişisel verilerinin işlenmesi",
            ...     data_categories=["kimlik", "iletişim", "müşteri işlem"],
            ...     processing_purposes=["Sözleşme ifası", "Müşteri ilişkileri"],
            ...     legal_basis=["KVKK 5/2-c", "KVKK 5/2-f"],
            ...     retention_period_months=72,
            ...     responsible_person_id=dpo.id
            ... )
        """
        record = cls(
            tenant_id=tenant_id,
            record_type=ComplianceRecordType.VVT,
            title=title,
            description=description,
            status=ComplianceStatus.ACTIVE,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            legal_basis=legal_basis,
            retention_period_months=retention_period_months,
            responsible_person_id=responsible_person_id,
            start_date=datetime.now(timezone.utc),
            review_date=datetime.now(timezone.utc) + timedelta(days=365),  # Annual review
        )
        
        logger.info(
            "VVT record created",
            record_id=str(record.id),
            tenant_id=str(tenant_id),
            data_categories=data_categories,
        )
        
        return record
    
    @classmethod
    def create_breach_record(
        cls,
        tenant_id: UUIDType,
        title: str,
        description: str,
        severity: ComplianceSeverity,
        breach_date: datetime,
        affected_count: int,
        notification_required: bool,
        responsible_person_id: UUIDType,
    ) -> "ComplianceRecord":
        """
        Create data breach notification record.
        
        Example:
            >>> breach = ComplianceRecord.create_breach_record(
            ...     tenant_id=tenant.id,
            ...     title="Veri İhlali Bildirimi - 2025-001",
            ...     description="Yetkisiz erişim tespit edildi",
            ...     severity=ComplianceSeverity.HIGH,
            ...     breach_date=datetime.now(timezone.utc),
            ...     affected_count=150,
            ...     notification_required=True,
            ...     responsible_person_id=dpo.id
            ... )
        """
        record = cls(
            tenant_id=tenant_id,
            record_type=ComplianceRecordType.BREACH,
            title=title,
            description=description,
            status=ComplianceStatus.UNDER_REVIEW,
            severity=severity,
            breach_date=breach_date,
            breach_discovered_date=datetime.now(timezone.utc),
            affected_count=affected_count,
            notification_required=notification_required,
            responsible_person_id=responsible_person_id,
        )
        
        logger.warning(
            "Data breach record created",
            record_id=str(record.id),
            tenant_id=str(tenant_id),
            severity=severity.value,
            affected_count=affected_count,
        )
        
        return record
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def activate(self) -> None:
        """Activate record."""
        self.status = ComplianceStatus.ACTIVE
        if not self.start_date:
            self.start_date = datetime.now(timezone.utc)
        
        logger.info(
            "Compliance record activated",
            record_id=str(self.id),
            record_type=self.record_type.value,
        )
    
    def submit_for_review(self) -> None:
        """Submit record for review."""
        self.status = ComplianceStatus.UNDER_REVIEW
        
        logger.info(
            "Compliance record submitted for review",
            record_id=str(self.id),
        )
    
    def mark_resolved(self, resolution_notes: str | None = None) -> None:
        """Mark issue as resolved."""
        self.status = ComplianceStatus.RESOLVED
        
        if resolution_notes:
            if "resolution" not in self.metadata:
                self.metadata["resolution"] = {}
            self.metadata["resolution"]["notes"] = resolution_notes
            self.metadata["resolution"]["resolved_at"] = datetime.now(timezone.utc).isoformat()
            
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(self, "metadata")
        
        logger.info(
            "Compliance record resolved",
            record_id=str(self.id),
        )
    
    def archive(self) -> None:
        """Archive record."""
        self.status = ComplianceStatus.ARCHIVED
        
        logger.info(
            "Compliance record archived",
            record_id=str(self.id),
        )
    
    # =========================================================================
    # BREACH MANAGEMENT
    # =========================================================================
    
    def notify_authority(self, notification_details: dict[str, Any]) -> None:
        """
        Record that authorities (KVKK) have been notified.
        
        Args:
            notification_details: Notification details (method, reference, etc.)
        """
        self.notification_date = datetime.now(timezone.utc)
        self.status = ComplianceStatus.REPORTED
        
        if "notification" not in self.metadata:
            self.metadata["notification"] = {}
        self.metadata["notification"].update(notification_details)
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(self, "metadata")
        
        logger.warning(
            "Breach notification sent to authority",
            record_id=str(self.id),
            notification_date=self.notification_date.isoformat(),
        )
    
    def is_notification_overdue(self) -> bool:
        """
        Check if breach notification is overdue (72 hours).
        
        Returns:
            bool: True if notification required and overdue
        """
        if not self.notification_required:
            return False
        
        if self.notification_date:
            return False  # Already notified
        
        if not self.breach_discovered_date:
            return False
        
        # KVKK requires notification within 72 hours
        deadline = self.breach_discovered_date + timedelta(hours=72)
        return datetime.now(timezone.utc) > deadline
    
    # =========================================================================
    # REVIEW MANAGEMENT
    # =========================================================================
    
    def schedule_review(self, months: int = 12) -> None:
        """
        Schedule next review.
        
        Args:
            months: Months until next review (default: 12)
        """
        self.review_date = datetime.now(timezone.utc) + timedelta(days=months * 30)
        
        logger.info(
            "Review scheduled",
            record_id=str(self.id),
            review_date=self.review_date.isoformat(),
        )
    
    def is_review_due(self) -> bool:
        """Check if review is due."""
        if not self.review_date:
            return False
        
        return datetime.now(timezone.utc) >= self.review_date
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("title")
    def validate_title(self, key: str, title: str) -> str:
        """Validate title."""
        if not title or not title.strip():
            raise ValidationError(
                message="Title cannot be empty",
                field="title",
            )
        
        return title.strip()
    
    @validates("description")
    def validate_description(self, key: str, description: str) -> str:
        """Validate description."""
        if not description or not description.strip():
            raise ValidationError(
                message="Description cannot be empty",
                field="description",
            )
        
        return description.strip()
    
    @validates("retention_period_months")
    def validate_retention_period(
        self,
        key: str,
        retention_period_months: int | None,
    ) -> int | None:
        """Validate retention period."""
        if retention_period_months is not None and retention_period_months <= 0:
            raise ValidationError(
                message="Retention period must be positive",
                field="retention_period_months",
            )
        
        return retention_period_months
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ComplianceRecord("
            f"id={self.id}, "
            f"type={self.record_type.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        
        # Add display names
        data["record_type_display"] = self.record_type.display_name_tr
        data["status_display"] = self.status.display_name_tr
        
        if self.severity:
            data["severity_display"] = self.severity.display_name_tr
        
        # Add computed fields
        data["is_review_due"] = self.is_review_due()
        
        if self.record_type == ComplianceRecordType.BREACH:
            data["is_notification_overdue"] = self.is_notification_overdue()
        
        # Format legal basis with Turkish names
        if self.legal_basis:
            try:
                data["legal_basis_display"] = [
                    LegalBasis(basis).display_name_tr
                    for basis in self.legal_basis
                ]
            except ValueError:
                data["legal_basis_display"] = self.legal_basis
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ComplianceRecord",
    "ComplianceRecordType",
    "ComplianceStatus",
    "ComplianceSeverity",
    "LegalBasis",
]