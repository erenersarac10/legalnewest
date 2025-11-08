"""
Audit Logging Module - Harvey/Legora %100 KVKK/GDPR Compliance.

Production-ready audit logging for Turkish Legal AI:
- Complete audit trail (who, what, when, where, why)
- KVKK/GDPR compliance
- Tamper-proof hash chain
- Async non-blocking logging
- Compliance reports

Usage:
    >>> from backend.core.audit import AuditService, AuditActionEnum
    >>>
    >>> # Create service
    >>> audit = AuditService(db_session)
    >>>
    >>> # Log action
    >>> await audit.log_action(
    ...     action=AuditActionEnum.DOCUMENT_READ,
    ...     resource_type="document",
    ...     resource_id="rg:12345",
    ...     description="User read document",
    ...     user_id=user.id,
    ...     tenant_id=tenant.id,
    ... )
    >>>
    >>> # Get audit trail
    >>> logs = await audit.get_user_audit_trail(user.id)
    >>>
    >>> # Get compliance report
    >>> report = await audit.get_compliance_report(
    ...     tenant_id=tenant.id,
    ...     start_date=start,
    ...     end_date=end
    ... )
"""

from backend.core.audit.models import (
    AuditLog,
    AuditArchive,
    AuditActionEnum,
    AuditSeverityEnum,
    AuditStatusEnum,
)
from backend.core.audit.service import (
    AuditService,
    get_audit_service,
)


__all__ = [
    # Models
    "AuditLog",
    "AuditArchive",
    "AuditActionEnum",
    "AuditSeverityEnum",
    "AuditStatusEnum",
    # Service
    "AuditService",
    "get_audit_service",
]
