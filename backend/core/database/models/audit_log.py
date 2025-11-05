"""
Audit Log model for comprehensive activity tracking in Turkish Legal AI.

This module provides the AuditLog model for KVKK-compliant audit trails:
- Complete user activity tracking
- Data access logs (who accessed what, when)
- System event tracking
- Security event monitoring
- Change history (before/after values)
- IP address and geolocation tracking
- KVKK compliance (6 year retention)
- Immutable audit records

Audit Categories:
    - Authentication (login, logout, password change)
    - Authorization (permission changes, role assignments)
    - Data Access (document view, download, share)
    - Data Modification (create, update, delete)
    - System Events (config changes, errors)
    - Security Events (failed logins, suspicious activity)

KVKK Requirements:
    - 6 year retention period for legal records
    - Immutable logs (cannot be edited)
    - Complete audit trail for data access
    - User consent tracking
    - Data request tracking
    - Breach notification logs

Example:
    >>> # Log user login
    >>> audit = AuditLog.log_event(
    ...     tenant_id=tenant.id,
    ...     user_id=user.id,
    ...     action=AuditAction.USER_LOGIN,
    ...     category=AuditCategory.AUTHENTICATION,
    ...     resource_type="user",
    ...     resource_id=str(user.id),
    ...     ip_address=request.client.host,
    ...     user_agent=request.headers.get("User-Agent"),
    ...     status=AuditStatus.SUCCESS
    ... )
    >>> 
    >>> # Log document access
    >>> audit = AuditLog.log_document_access(
    ...     tenant_id=tenant.id,
    ...     user_id=user.id,
    ...     document_id=doc.id,
    ...     action=AuditAction.DOCUMENT_VIEW,
    ...     ip_address=request.client.host
    ... )
"""

import enum
from datetime import datetime, timezone
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
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger
from backend.core.database.models.base import (
    Base,
    BaseModelMixin,
    TenantMixin,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AuditCategory(str, enum.Enum):
    """
    Audit event category for grouping.
    
    Categories:
    - AUTHENTICATION: Login, logout, password changes
    - AUTHORIZATION: Permission changes, role assignments
    - DATA_ACCESS: View, download, export
    - DATA_MODIFICATION: Create, update, delete
    - SYSTEM: Configuration, settings, maintenance
    - SECURITY: Failed logins, suspicious activity
    - COMPLIANCE: KVKK actions, consent, data requests
    """
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.AUTHENTICATION: "Kimlik Doğrulama",
            self.AUTHORIZATION: "Yetkilendirme",
            self.DATA_ACCESS: "Veri Erişimi",
            self.DATA_MODIFICATION: "Veri Değişikliği",
            self.SYSTEM: "Sistem",
            self.SECURITY: "Güvenlik",
            self.COMPLIANCE: "Uyumluluk",
        }
        return names.get(self, self.value)


class AuditAction(str, enum.Enum):
    """
    Specific audit actions.
    
    Actions are grouped by category but defined comprehensively.
    """
    
    # Authentication
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    
    # Authorization
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    
    # Data Access
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_DOWNLOAD = "document_download"
    DOCUMENT_EXPORT = "document_export"
    DOCUMENT_SHARE = "document_share"
    
    # Data Modification
    DOCUMENT_CREATE = "document_create"
    DOCUMENT_UPDATE = "document_update"
    DOCUMENT_DELETE = "document_delete"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    
    # System
    CONFIG_CHANGE = "config_change"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    
    # Security
    LOGIN_FAILED = "login_failed"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCOUNT_LOCKED = "account_locked"
    SESSION_REVOKED = "session_revoked"
    
    # Compliance (KVKK)
    CONSENT_GIVEN = "consent_given"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_REQUEST = "data_request"
    DATA_EXPORT_REQUEST = "data_export_request"
    DATA_DELETION_REQUEST = "data_deletion_request"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.USER_LOGIN: "Kullanıcı Girişi",
            self.USER_LOGOUT: "Kullanıcı Çıkışı",
            self.PASSWORD_CHANGE: "Şifre Değişikliği",
            self.PASSWORD_RESET: "Şifre Sıfırlama",
            self.DOCUMENT_VIEW: "Belge Görüntüleme",
            self.DOCUMENT_DOWNLOAD: "Belge İndirme",
            self.DOCUMENT_CREATE: "Belge Oluşturma",
            self.DOCUMENT_UPDATE: "Belge Güncelleme",
            self.DOCUMENT_DELETE: "Belge Silme",
            self.CONSENT_GIVEN: "Onay Verildi",
            self.CONSENT_WITHDRAWN: "Onay Geri Çekildi",
            self.DATA_REQUEST: "Veri Talebi",
        }
        return names.get(self, self.value.replace("_", " ").title())


class AuditStatus(str, enum.Enum):
    """Audit event status."""
    
    SUCCESS = "success"          # Action completed successfully
    FAILURE = "failure"          # Action failed
    PARTIAL = "partial"          # Partially completed
    PENDING = "pending"          # In progress
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name_tr(self) -> str:
        """Turkish display name."""
        names = {
            self.SUCCESS: "Başarılı",
            self.FAILURE: "Başarısız",
            self.PARTIAL: "Kısmi",
            self.PENDING: "Bekliyor",
        }
        return names.get(self, self.value)


class SeverityLevel(str, enum.Enum):
    """Event severity level."""
    
    INFO = "info"                # Informational
    WARNING = "warning"          # Warning
    ERROR = "error"              # Error
    CRITICAL = "critical"        # Critical security event
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# AUDIT LOG MODEL
# =============================================================================


class AuditLog(Base, BaseModelMixin, TenantMixin):
    """
    Audit Log model for comprehensive activity tracking.
    
    Immutable audit records for:
    - User activity tracking
    - Data access monitoring
    - Security event logging
    - KVKK compliance
    - Change history
    
    KVKK Compliance:
        - 6 year retention period
        - Immutable records (no updates/deletes)
        - Complete audit trail
        - IP tracking and geolocation
        - User agent tracking
        - Before/after values for changes
    
    Attributes:
        user_id: Who performed the action (nullable for system events)
        user: User relationship
        
        category: Event category
        action: Specific action performed
        status: Action result (success, failure)
        severity: Event severity level
        
        resource_type: Type of resource (document, user, etc.)
        resource_id: Resource UUID
        resource_name: Resource display name
        
        description: Human-readable description
        
        changes: Before/after values for modifications (JSON)
        metadata: Additional context data (JSON)
        
        ip_address: Client IP address
        ip_location: Geographic location (JSON)
        user_agent: HTTP User-Agent
        
        session_id: Related session UUID
        request_id: Request correlation ID
        
        duration_ms: Action duration in milliseconds
        
        error_message: Error details if failed
        stack_trace: Exception stack trace
        
    Relationships:
        tenant: Parent tenant
        user: User who performed action
        
    Note:
        No SoftDeleteMixin - audit logs are never deleted (KVKK requirement)
        No AuditMixin - audit logs don't track their own changes
    """
    
    __tablename__ = "audit_logs"
    
    # =========================================================================
    # USER RELATIONSHIP
    # =========================================================================
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,  # System events may not have a user
        index=True,
        comment="User who performed the action (NULL for system events)",
    )
    
    user = relationship(
        "User",
        back_populates="audit_logs",
    )
    
    # =========================================================================
    # EVENT CLASSIFICATION
    # =========================================================================
    
    category = Column(
        Enum(AuditCategory, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Event category (authentication, data_access, etc.)",
    )
    
    action = Column(
        Enum(AuditAction, native_enum=False, length=50),
        nullable=False,
        index=True,
        comment="Specific action performed",
    )
    
    status = Column(
        Enum(AuditStatus, native_enum=False, length=50),
        nullable=False,
        default=AuditStatus.SUCCESS,
        index=True,
        comment="Action result (success, failure, partial)",
    )
    
    severity = Column(
        Enum(SeverityLevel, native_enum=False, length=50),
        nullable=False,
        default=SeverityLevel.INFO,
        index=True,
        comment="Event severity level",
    )
    
    # =========================================================================
    # RESOURCE INFORMATION
    # =========================================================================
    
    resource_type = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Type of resource affected (document, user, team, etc.)",
    )
    
    resource_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Resource UUID",
    )
    
    resource_name = Column(
        String(255),
        nullable=True,
        comment="Resource display name (for readability)",
    )
    
    # =========================================================================
    # DESCRIPTION
    # =========================================================================
    
    description = Column(
        Text,
        nullable=False,
        comment="Human-readable event description",
    )
    
    # =========================================================================
    # CHANGE TRACKING
    # =========================================================================
    
    changes = Column(
        JSONB,
        nullable=True,
        comment="Before/after values for modifications",
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context data (request params, etc.)",
    )
    
    # =========================================================================
    # NETWORK INFORMATION
    # =========================================================================
    
    ip_address = Column(
        String(45),  # IPv6 max length
        nullable=True,
        index=True,
        comment="Client IP address",
    )
    
    ip_location = Column(
        JSONB,
        nullable=True,
        comment="Geographic location (city, country, lat/lon)",
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="HTTP User-Agent header",
    )
    
    # =========================================================================
    # REQUEST CORRELATION
    # =========================================================================
    
    session_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Related session UUID",
    )
    
    request_id = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Request correlation ID (for tracing)",
    )
    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    
    duration_ms = Column(
        Integer,
        nullable=True,
        comment="Action duration in milliseconds",
    )
    
    # =========================================================================
    # ERROR INFORMATION
    # =========================================================================
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if action failed",
    )
    
    stack_trace = Column(
        Text,
        nullable=True,
        comment="Exception stack trace (for debugging)",
    )
    
    # =========================================================================
    # TIMESTAMP (from BaseModelMixin)
    # =========================================================================
    # created_at is inherited and indexed
    # This is the event timestamp
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    __table_args__ = (
        # Index for user activity queries
        Index(
            "ix_audit_logs_user_activity",
            "user_id",
            "created_at",
        ),
        
        # Index for resource tracking
        Index(
            "ix_audit_logs_resource",
            "resource_type",
            "resource_id",
            "created_at",
        ),
        
        # Index for category filtering
        Index(
            "ix_audit_logs_category",
            "category",
            "action",
            "created_at",
        ),
        
        # Index for failed events
        Index(
            "ix_audit_logs_failures",
            "status",
            "created_at",
            postgresql_where="status IN ('failure', 'partial')",
        ),
        
        # Index for security events
        Index(
            "ix_audit_logs_security",
            "category",
            "severity",
            "created_at",
            postgresql_where="category = 'security' OR severity IN ('error', 'critical')",
        ),
        
        # Index for tenant queries with time range
        Index(
            "ix_audit_logs_tenant_time",
            "tenant_id",
            "created_at",
        ),
        
        # Index for IP tracking (security analysis)
        Index(
            "ix_audit_logs_ip",
            "ip_address",
            "created_at",
        ),
    )
    
    # =========================================================================
    # AUDIT LOG CREATION
    # =========================================================================
    
    @classmethod
    def log_event(
        cls,
        tenant_id: UUIDType,
        category: AuditCategory,
        action: AuditAction,
        description: str,
        user_id: UUIDType | None = None,
        status: AuditStatus = AuditStatus.SUCCESS,
        severity: SeverityLevel = SeverityLevel.INFO,
        resource_type: str | None = None,
        resource_id: UUIDType | None = None,
        resource_name: str | None = None,
        changes: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        session_id: UUIDType | None = None,
        request_id: str | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        stack_trace: str | None = None,
    ) -> "AuditLog":
        """
        Log an audit event.
        
        Args:
            tenant_id: Tenant UUID
            category: Event category
            action: Specific action
            description: Human-readable description
            user_id: User who performed action (optional)
            status: Action result
            severity: Event severity
            resource_type: Type of resource
            resource_id: Resource UUID
            resource_name: Resource display name
            changes: Before/after values
            metadata: Additional context
            ip_address: Client IP
            user_agent: User agent
            session_id: Session UUID
            request_id: Request correlation ID
            duration_ms: Action duration
            error_message: Error details
            stack_trace: Exception trace
            
        Returns:
            AuditLog: New audit log entry
            
        Example:
            >>> audit = AuditLog.log_event(
            ...     tenant_id=tenant.id,
            ...     user_id=user.id,
            ...     category=AuditCategory.DATA_ACCESS,
            ...     action=AuditAction.DOCUMENT_VIEW,
            ...     description=f"User viewed document: {doc.name}",
            ...     resource_type="document",
            ...     resource_id=doc.id,
            ...     resource_name=doc.name,
            ...     ip_address=request.client.host,
            ...     user_agent=request.headers.get("User-Agent")
            ... )
        """
        audit = cls(
            tenant_id=tenant_id,
            user_id=user_id,
            category=category,
            action=action,
            status=status,
            severity=severity,
            description=description,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            changes=changes,
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            duration_ms=duration_ms,
            error_message=error_message,
            stack_trace=stack_trace,
        )
        
        logger.info(
            "Audit event logged",
            audit_id=str(audit.id),
            category=category.value,
            action=action.value,
            user_id=str(user_id) if user_id else None,
            resource_type=resource_type,
            status=status.value,
        )
        
        return audit
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    @classmethod
    def log_authentication(
        cls,
        tenant_id: UUIDType,
        user_id: UUIDType,
        action: AuditAction,
        status: AuditStatus,
        ip_address: str | None = None,
        user_agent: str | None = None,
        error_message: str | None = None,
    ) -> "AuditLog":
        """
        Log authentication event.
        
        Example:
            >>> AuditLog.log_authentication(
            ...     tenant_id=tenant.id,
            ...     user_id=user.id,
            ...     action=AuditAction.USER_LOGIN,
            ...     status=AuditStatus.SUCCESS,
            ...     ip_address="192.168.1.100"
            ... )
        """
        severity = SeverityLevel.WARNING if status == AuditStatus.FAILURE else SeverityLevel.INFO
        
        return cls.log_event(
            tenant_id=tenant_id,
            user_id=user_id,
            category=AuditCategory.AUTHENTICATION,
            action=action,
            description=f"{action.display_name_tr}: {status.display_name_tr}",
            status=status,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message,
        )
    
    @classmethod
    def log_document_access(
        cls,
        tenant_id: UUIDType,
        user_id: UUIDType,
        document_id: UUIDType,
        action: AuditAction,
        document_name: str | None = None,
        ip_address: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AuditLog":
        """
        Log document access event.
        
        Example:
            >>> AuditLog.log_document_access(
            ...     tenant_id=tenant.id,
            ...     user_id=user.id,
            ...     document_id=doc.id,
            ...     action=AuditAction.DOCUMENT_DOWNLOAD,
            ...     document_name=doc.name,
            ...     ip_address=request.client.host
            ... )
        """
        return cls.log_event(
            tenant_id=tenant_id,
            user_id=user_id,
            category=AuditCategory.DATA_ACCESS,
            action=action,
            description=f"{action.display_name_tr}: {document_name or 'Document'}",
            resource_type="document",
            resource_id=document_id,
            resource_name=document_name,
            ip_address=ip_address,
            metadata=metadata,
        )
    
    @classmethod
    def log_data_modification(
        cls,
        tenant_id: UUIDType,
        user_id: UUIDType,
        action: AuditAction,
        resource_type: str,
        resource_id: UUIDType,
        resource_name: str | None = None,
        changes: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> "AuditLog":
        """
        Log data modification event with change tracking.
        
        Example:
            >>> AuditLog.log_data_modification(
            ...     tenant_id=tenant.id,
            ...     user_id=user.id,
            ...     action=AuditAction.USER_UPDATE,
            ...     resource_type="user",
            ...     resource_id=target_user.id,
            ...     changes={
            ...         "email": {
            ...             "old": "old@example.com",
            ...             "new": "new@example.com"
            ...         }
            ...     }
            ... )
        """
        return cls.log_event(
            tenant_id=tenant_id,
            user_id=user_id,
            category=AuditCategory.DATA_MODIFICATION,
            action=action,
            description=f"{action.display_name_tr}: {resource_name or resource_type}",
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            changes=changes,
            ip_address=ip_address,
        )
    
    @classmethod
    def log_security_event(
        cls,
        tenant_id: UUIDType,
        action: AuditAction,
        description: str,
        user_id: UUIDType | None = None,
        severity: SeverityLevel = SeverityLevel.WARNING,
        ip_address: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AuditLog":
        """
        Log security event.
        
        Example:
            >>> AuditLog.log_security_event(
            ...     tenant_id=tenant.id,
            ...     action=AuditAction.LOGIN_FAILED,
            ...     description="Multiple failed login attempts",
            ...     severity=SeverityLevel.WARNING,
            ...     ip_address="192.168.1.100",
            ...     metadata={"attempt_count": 5}
            ... )
        """
        return cls.log_event(
            tenant_id=tenant_id,
            user_id=user_id,
            category=AuditCategory.SECURITY,
            action=action,
            description=description,
            severity=severity,
            ip_address=ip_address,
            metadata=metadata,
        )
    
    @classmethod
    def log_compliance_event(
        cls,
        tenant_id: UUIDType,
        user_id: UUIDType,
        action: AuditAction,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> "AuditLog":
        """
        Log KVKK compliance event.
        
        Example:
            >>> AuditLog.log_compliance_event(
            ...     tenant_id=tenant.id,
            ...     user_id=user.id,
            ...     action=AuditAction.DATA_EXPORT_REQUEST,
            ...     description="User requested data export",
            ...     metadata={"request_type": "full_export"}
            ... )
        """
        return cls.log_event(
            tenant_id=tenant_id,
            user_id=user_id,
            category=AuditCategory.COMPLIANCE,
            action=action,
            description=description,
            severity=SeverityLevel.INFO,
            metadata=metadata,
        )
    
    # =========================================================================
    # QUERY HELPERS
    # =========================================================================
    
    def get_change_summary(self) -> str:
        """
        Get human-readable change summary.
        
        Returns:
            str: Change summary
        """
        if not self.changes:
            return "No changes recorded"
        
        changes_list = []
        for field, values in self.changes.items():
            if isinstance(values, dict) and "old" in values and "new" in values:
                changes_list.append(
                    f"{field}: '{values['old']}' → '{values['new']}'"
                )
        
        return ", ".join(changes_list) if changes_list else "Changes recorded"
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @validates("description")
    def validate_description(self, key: str, description: str) -> str:
        """Validate description."""
        if not description or not description.strip():
            raise ValidationError(
                message="Description cannot be empty",
                field="description",
            )
        
        return description.strip()
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<AuditLog("
            f"id={self.id}, "
            f"action={self.action.value}, "
            f"status={self.status.value}"
            f")>"
        )
    
    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_sensitive: Include sensitive data (stack traces, etc.)
            
        Returns:
            dict: Audit log data
        """
        data = super().to_dict()
        
        # Add display names
        data["category_display"] = self.category.display_name_tr
        data["action_display"] = self.action.display_name_tr
        data["status_display"] = self.status.display_name_tr
        data["severity_display"] = self.severity.value.upper()
        
        # Add change summary
        data["change_summary"] = self.get_change_summary()
        
        # Remove sensitive data by default
        if not include_sensitive:
            data.pop("stack_trace", None)
        
        return data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "AuditLog",
    "AuditCategory",
    "AuditAction",
    "AuditStatus",
    "SeverityLevel",
]