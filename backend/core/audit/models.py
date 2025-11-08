"""
Audit Log Database Models - Harvey/Legora %100 KVKK/GDPR Compliance.

World-class audit logging for Turkish Legal AI:
- Complete audit trail (who, what, when, where, why)
- KVKK/GDPR compliance
- Tamper-proof logging
- Compliance reporting
- Data access tracking
- Security event logging

Why Audit Logs?
    Without: No accountability → compliance violations, security blind spots
    With: Complete audit trail → KVKK/GDPR compliant, forensic ready

    Impact: Legal compliance + security monitoring! 📋

Audit Architecture:
    [Request] → [Middleware] → [AuditService] → [AuditLog DB + Archive]
                                                         ↓
                                                   [Retention Policy]

Audit Categories:
    - Authentication (login, logout, password change)
    - Authorization (permission grant/revoke, role assignment)
    - Data Access (read, search, export)
    - Data Modification (create, update, delete)
    - System Events (config change, error, security alert)
    - Compliance (KVKK consent, data deletion, export)

Log Levels:
    - INFO: Normal operations (login, search)
    - WARNING: Suspicious activity (failed login, permission denied)
    - ERROR: System errors (service failure, data corruption)
    - CRITICAL: Security incidents (breach attempt, unauthorized access)

KVKK/GDPR Requirements:
    - Log all personal data access (Article 12)
    - Log all data modifications (Article 13)
    - Log all data deletions (Article 17)
    - Log all data exports (Article 20)
    - Retain logs for 6 years minimum
    - Provide audit reports on request

Tamper Protection:
    - Append-only logs
    - Cryptographic hashing (SHA-256 chain)
    - Periodic archival to immutable storage
    - Digital signatures (optional)

Performance:
    - Async logging (non-blocking)
    - Batch inserts (10-50 logs/batch)
    - Automatic archival (monthly)
    - Indexed queries (user, resource, timestamp)
    - Retention policy (auto-delete after N years)

Features:
    - Request context capture
    - IP geolocation
    - User agent parsing
    - Diff tracking (before/after)
    - Related log grouping
    - Export to SIEM systems
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum as PyEnum
import uuid
import json

from sqlalchemy import (
    Column, String, Text, Integer, DateTime, ForeignKey,
    Index, CheckConstraint
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property

from backend.core.database.models import Base


# =============================================================================
# ENUMS
# =============================================================================


class AuditActionEnum(str, PyEnum):
    """Audit action types."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    PASSWORD_CHANGE = "auth.password_change"
    PASSWORD_RESET = "auth.password_reset"
    EMAIL_VERIFY = "auth.email_verify"

    # Authorization
    PERMISSION_GRANT = "authz.permission_grant"
    PERMISSION_REVOKE = "authz.permission_revoke"
    ROLE_ASSIGN = "authz.role_assign"
    ROLE_REMOVE = "authz.role_remove"
    PERMISSION_DENIED = "authz.permission_denied"

    # Data Access
    DOCUMENT_READ = "data.document_read"
    DOCUMENT_SEARCH = "data.document_search"
    DOCUMENT_EXPORT = "data.document_export"
    DOCUMENT_DOWNLOAD = "data.document_download"

    # Data Modification
    DOCUMENT_CREATE = "data.document_create"
    DOCUMENT_UPDATE = "data.document_update"
    DOCUMENT_DELETE = "data.document_delete"

    # User Management
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    USER_SUSPEND = "user.suspend"
    USER_ACTIVATE = "user.activate"

    # Tenant Management
    TENANT_CREATE = "tenant.create"
    TENANT_UPDATE = "tenant.update"
    TENANT_DELETE = "tenant.delete"
    TENANT_SUSPEND = "tenant.suspend"

    # System Events
    SYSTEM_CONFIG_CHANGE = "system.config_change"
    SYSTEM_ERROR = "system.error"
    SYSTEM_ALERT = "system.alert"

    # Compliance (KVKK/GDPR)
    DATA_CONSENT_GIVEN = "compliance.consent_given"
    DATA_CONSENT_REVOKED = "compliance.consent_revoked"
    DATA_EXPORT_REQUEST = "compliance.data_export_request"
    DATA_DELETE_REQUEST = "compliance.data_delete_request"
    DATA_ACCESS_LOG = "compliance.data_access_log"


class AuditSeverityEnum(str, PyEnum):
    """Audit log severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditStatusEnum(str, PyEnum):
    """Audit log status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


# =============================================================================
# AUDIT LOG MODEL
# =============================================================================


class AuditLog(Base):
    """
    Audit log model for compliance and security.

    Harvey/Legora %100: Complete audit trail.

    Features:
    - Comprehensive logging (who, what, when, where, why)
    - KVKK/GDPR compliance
    - Tamper detection via hash chain
    - Fast queries via indexes
    - Auto-archival support
    """

    __tablename__ = "audit_logs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Audit log UUID"
    )

    # WHO (Actor)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User who performed action (NULL for system actions)"
    )
    username: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Username at time of action (denormalized for audit trail)"
    )
    tenant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Tenant context"
    )

    # WHAT (Action)
    action: Mapped[AuditActionEnum] = mapped_column(
        String(100),
        nullable=False,
        comment="Action performed (e.g., 'auth.login', 'data.document_read')"
    )
    resource_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Resource type (e.g., 'document', 'user', 'tenant')"
    )
    resource_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Resource ID (e.g., document ID, user ID)"
    )
    resource_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable resource name"
    )

    # WHEN (Timestamp)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="Action timestamp (UTC)"
    )

    # WHERE (Context)
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),
        nullable=True,
        comment="Client IP address (IPv4/IPv6)"
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Client user agent"
    )
    request_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Request ID for correlation"
    )
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Session ID"
    )

    # WHY/HOW (Details)
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Human-readable description"
    )
    details: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional details (request params, filters, etc.)"
    )

    # Status
    status: Mapped[AuditStatusEnum] = mapped_column(
        String(50),
        default=AuditStatusEnum.SUCCESS,
        nullable=False
    )
    severity: Mapped[AuditSeverityEnum] = mapped_column(
        String(50),
        default=AuditSeverityEnum.INFO,
        nullable=False
    )

    # Result
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if action failed"
    )
    response_code: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="HTTP response code"
    )

    # Change tracking (for data modifications)
    old_value: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Previous value (for updates/deletes)"
    )
    new_value: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="New value (for creates/updates)"
    )

    # Tamper detection
    hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hash of log entry (tamper detection)"
    )
    previous_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Hash of previous log entry (chain verification)"
    )

    # Archival
    archived: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Archived to long-term storage"
    )
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Log can be deleted after this date (compliance)"
    )

    # Indexes
    __table_args__ = (
        # Fast user audit queries
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_user_id_timestamp", "user_id", "timestamp"),

        # Fast resource audit queries
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_resource_timestamp", "resource_type", "resource_id", "timestamp"),

        # Fast tenant audit queries
        Index("ix_audit_logs_tenant_id", "tenant_id"),
        Index("ix_audit_logs_tenant_id_timestamp", "tenant_id", "timestamp"),

        # Fast action queries
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_action_timestamp", "action", "timestamp"),

        # Security monitoring
        Index("ix_audit_logs_severity", "severity"),
        Index("ix_audit_logs_status", "status"),
        Index("ix_audit_logs_ip_address", "ip_address"),

        # Archival queries
        Index("ix_audit_logs_archived", "archived"),
        Index("ix_audit_logs_retention", "retention_until"),

        # Request correlation
        Index("ix_audit_logs_request_id", "request_id"),
        Index("ix_audit_logs_session_id", "session_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, action='{self.action}', "
            f"user_id={self.user_id}, timestamp='{self.timestamp}')>"
        )

    @hybrid_property
    def is_security_event(self) -> bool:
        """Check if this is a security-related event."""
        security_actions = {
            AuditActionEnum.LOGIN_FAILED,
            AuditActionEnum.PERMISSION_DENIED,
            AuditActionEnum.SYSTEM_ALERT,
        }
        return self.action in security_actions or self.severity in [
            AuditSeverityEnum.ERROR,
            AuditSeverityEnum.CRITICAL
        ]

    @hybrid_property
    def is_compliance_event(self) -> bool:
        """Check if this is a KVKK/GDPR compliance event."""
        return self.action.value.startswith("compliance.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "user_id": str(self.user_id) if self.user_id else None,
            "username": self.username,
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "description": self.description,
            "status": self.status.value,
            "severity": self.severity.value,
            "ip_address": self.ip_address,
            "details": self.details,
            "error_message": self.error_message,
            "response_code": self.response_code,
        }

    def to_json(self) -> str:
        """Convert to JSON string for export."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# AUDIT ARCHIVE MODEL
# =============================================================================


class AuditArchive(Base):
    """
    Archived audit logs for long-term retention.

    Harvey/Legora %100: Compliance-grade archival.

    Features:
    - Monthly archival batches
    - Compressed storage
    - Immutable records
    - Compliance retention (6+ years)
    """

    __tablename__ = "audit_archives"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Archive metadata
    archive_date: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="Start of archived period"
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="End of archived period"
    )

    # Archive data
    log_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of logs in archive"
    )
    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Path to archived file (S3/local)"
    )
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="File size in bytes"
    )
    file_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash of archive file"
    )

    # Compression
    compression: Mapped[str] = mapped_column(
        String(50),
        default="gzip",
        nullable=False,
        comment="Compression algorithm (gzip, bzip2, zstd)"
    )

    # Retention
    retention_until: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="Archive can be deleted after this date"
    )

    # Verification
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="Last integrity verification"
    )
    verification_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        comment="pending | verified | failed"
    )

    # Indexes
    __table_args__ = (
        Index("ix_audit_archives_archive_date", "archive_date"),
        Index("ix_audit_archives_period", "period_start", "period_end"),
        Index("ix_audit_archives_retention", "retention_until"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditArchive(id={self.id}, period={self.period_start.date()}-{self.period_end.date()}, "
            f"logs={self.log_count})>"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "AuditActionEnum",
    "AuditSeverityEnum",
    "AuditStatusEnum",
    # Models
    "AuditLog",
    "AuditArchive",
]
