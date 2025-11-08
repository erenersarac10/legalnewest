"""
Audit Report Database Model for Turkish Legal AI.

This module defines the AuditReport model for compliance reporting:
- SOC 2 / ISO 27001 audit reports
- Scheduled/on-demand report generation
- Multiple report formats (PDF, JSON, CSV)
- Retention policy compliance
- Tenant-scoped reporting
- Export capabilities

Report Types:
    - COMPLIANCE: SOC 2 / ISO 27001 compliance reports
    - ACCESS: User access audit reports
    - CHANGE: System change audit reports
    - SECURITY: Security event reports
    - CUSTOM: Custom query-based reports

Example:
    >>> from backend.core.database.models.audit_report import AuditReport
    >>>
    >>> # Create compliance report
    >>> report = AuditReport(
    ...     tenant_id=tenant_id,
    ...     report_type=AuditReportType.COMPLIANCE,
    ...     title="SOC 2 Q4 2025 Audit Report",
    ...     parameters={"start_date": "2025-10-01", "end_date": "2025-12-31"}
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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from backend.core.database.base import Base


# =============================================================================
# ENUMS
# =============================================================================


class AuditReportType(str, enum.Enum):
    """Audit report types."""

    COMPLIANCE = "compliance"  # SOC 2 / ISO 27001
    ACCESS = "access"  # User access audit
    CHANGE = "change"  # System change audit
    SECURITY = "security"  # Security event audit
    DATA_RETENTION = "data_retention"  # Data retention compliance
    CUSTOM = "custom"  # Custom query reports


class AuditReportStatus(str, enum.Enum):
    """Audit report generation status."""

    PENDING = "pending"  # Queued for generation
    GENERATING = "generating"  # Currently generating
    COMPLETED = "completed"  # Successfully generated
    FAILED = "failed"  # Generation failed
    ARCHIVED = "archived"  # Archived (past retention)


class AuditReportFormat(str, enum.Enum):
    """Audit report output formats."""

    PDF = "pdf"  # PDF document
    JSON = "json"  # JSON export
    CSV = "csv"  # CSV export
    XLSX = "xlsx"  # Excel export
    HTML = "html"  # HTML report


# =============================================================================
# AUDIT REPORT MODEL
# =============================================================================


class AuditReport(Base):
    """
    Audit report model for compliance and audit trail reporting.

    This model stores metadata for audit reports:
    - Report configuration
    - Generation status
    - Output storage
    - Retention policy compliance
    - Tenant isolation

    Relationships:
        - tenant: The tenant this report belongs to
        - created_by: User who created the report
        - retention_policy: Data retention policy applied
    """

    __tablename__ = "audit_reports"

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
        index=True,
    )

    # =========================================================================
    # REPORT CONFIGURATION
    # =========================================================================

    report_type: AuditReportType = Column(
        Enum(AuditReportType),
        nullable=False,
        index=True,
        doc="Type of audit report",
    )

    title: str = Column(
        String(255),
        nullable=False,
        doc="Report title",
    )

    description: Optional[str] = Column(
        Text,
        nullable=True,
        doc="Detailed report description",
    )

    # Report parameters (filters, date ranges, etc.)
    parameters: Dict[str, Any] = Column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Report generation parameters (filters, date ranges)",
    )

    # =========================================================================
    # GENERATION STATUS
    # =========================================================================

    status: AuditReportStatus = Column(
        Enum(AuditReportStatus),
        nullable=False,
        default=AuditReportStatus.PENDING,
        index=True,
        doc="Report generation status",
    )

    format: AuditReportFormat = Column(
        Enum(AuditReportFormat),
        nullable=False,
        default=AuditReportFormat.PDF,
        doc="Output format",
    )

    # =========================================================================
    # EXECUTION METADATA
    # =========================================================================

    started_at: Optional[datetime.datetime] = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Generation start timestamp",
    )

    completed_at: Optional[datetime.datetime] = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Generation completion timestamp",
    )

    duration_seconds: Optional[int] = Column(
        Integer,
        nullable=True,
        doc="Generation duration in seconds",
    )

    # Error information (if failed)
    error_message: Optional[str] = Column(
        Text,
        nullable=True,
        doc="Error message if generation failed",
    )

    # =========================================================================
    # OUTPUT & STORAGE
    # =========================================================================

    # File storage path (S3, local, etc.)
    file_path: Optional[str] = Column(
        String(500),
        nullable=True,
        doc="File storage path (S3 key or local path)",
    )

    file_size_bytes: Optional[int] = Column(
        Integer,
        nullable=True,
        doc="File size in bytes",
    )

    # Report summary statistics
    summary: Dict[str, Any] = Column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Report summary statistics (total events, users, etc.)",
    )

    # =========================================================================
    # RETENTION & COMPLIANCE
    # =========================================================================

    retention_policy_id: Optional[UUID] = Column(
        PGUUID(as_uuid=True),
        ForeignKey("audit_retention_policies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Associated retention policy",
    )

    expires_at: Optional[datetime.datetime] = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Report expiration date (based on retention policy)",
    )

    # =========================================================================
    # SCHEDULING
    # =========================================================================

    is_scheduled: bool = Column(
        Boolean,
        nullable=False,
        default=False,
        doc="Is this a scheduled recurring report?",
    )

    schedule_cron: Optional[str] = Column(
        String(100),
        nullable=True,
        doc="Cron expression for scheduled reports",
    )

    next_run_at: Optional[datetime.datetime] = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Next scheduled run timestamp",
    )

    # =========================================================================
    # COMPLIANCE FLAGS
    # =========================================================================

    is_compliance_report: bool = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Is this a compliance-required report (SOC 2, ISO 27001)?",
    )

    compliance_framework: Optional[str] = Column(
        String(100),
        nullable=True,
        doc="Compliance framework (SOC2, ISO27001, GDPR, KVKK)",
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
        # Tenant isolation index
        Index(
            "ix_audit_reports_tenant_status",
            "tenant_id",
            "status",
        ),
        # Scheduled reports index
        Index(
            "ix_audit_reports_scheduled",
            "is_scheduled",
            "next_run_at",
            postgresql_where=Column("is_scheduled") == True,  # Partial index
        ),
        # Compliance reports index
        Index(
            "ix_audit_reports_compliance",
            "tenant_id",
            "compliance_framework",
            postgresql_where=Column("is_compliance_report") == True,
        ),
        # Expiration index for cleanup
        Index(
            "ix_audit_reports_expiration",
            "expires_at",
            postgresql_where=Column("deleted_at").is_(None),
        ),
        # Type + status composite index
        Index(
            "ix_audit_reports_type_status",
            "report_type",
            "status",
        ),
    )

    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================

    # Relationships defined in main tenant/user models
    # tenant = relationship("Tenant", back_populates="audit_reports")
    # created_by = relationship("User", back_populates="audit_reports")
    # retention_policy = relationship("AuditRetentionPolicy")

    # =========================================================================
    # METHODS
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<AuditReport(id={self.id}, type={self.report_type}, "
            f"status={self.status}, title='{self.title}')>"
        )

    def is_expired(self) -> bool:
        """Check if report has expired based on retention policy."""
        if not self.expires_at:
            return False
        return datetime.datetime.utcnow() >= self.expires_at

    def is_pending(self) -> bool:
        """Check if report is pending generation."""
        return self.status == AuditReportStatus.PENDING

    def is_completed(self) -> bool:
        """Check if report generation completed successfully."""
        return self.status == AuditReportStatus.COMPLETED

    def mark_started(self) -> None:
        """Mark report generation as started."""
        self.status = AuditReportStatus.GENERATING
        self.started_at = datetime.datetime.utcnow()

    def mark_completed(
        self,
        file_path: str,
        file_size_bytes: int,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark report generation as completed.

        Args:
            file_path: Storage path for generated report
            file_size_bytes: File size in bytes
            summary: Optional summary statistics
        """
        self.status = AuditReportStatus.COMPLETED
        self.completed_at = datetime.datetime.utcnow()
        self.file_path = file_path
        self.file_size_bytes = file_size_bytes

        if self.started_at:
            self.duration_seconds = int(
                (self.completed_at - self.started_at).total_seconds()
            )

        if summary:
            self.summary = summary

    def mark_failed(self, error_message: str) -> None:
        """
        Mark report generation as failed.

        Args:
            error_message: Error description
        """
        self.status = AuditReportStatus.FAILED
        self.completed_at = datetime.datetime.utcnow()
        self.error_message = error_message

        if self.started_at:
            self.duration_seconds = int(
                (self.completed_at - self.started_at).total_seconds()
            )

    def calculate_next_run(self) -> None:
        """Calculate next scheduled run based on cron expression."""
        if not self.is_scheduled or not self.schedule_cron:
            self.next_run_at = None
            return

        # TODO: Parse cron expression and calculate next run
        # For now, default to 24 hours from now
        self.next_run_at = datetime.datetime.utcnow() + datetime.timedelta(days=1)
