"""
Compliance Audit Log Database Model for Turkish Legal AI.

Tracks GDPR/KVKK compliance events for audit trails.

Event Types: Data access, consent, cross-border transfers, data subject rights
Compliance: GDPR Article 30, KVKK Article 12
Immutable: Records cannot be modified once created
"""

import datetime
import enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

from backend.core.database.base import Base


class ComplianceEventType(str, enum.Enum):
    """Compliance event types."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"
    RETENTION_POLICY_APPLIED = "retention_policy_applied"
    SUBJECT_ACCESS_REQUEST = "subject_access_request"


class LegalBasis(str, enum.Enum):
    """GDPR legal basis (Article 6)."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class ComplianceAuditLog(Base):
    """Compliance audit log (GDPR/KVKK). Immutable."""

    __tablename__ = "compliance_audit_logs"

    id: UUID = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    tenant_id: UUID = Column(PGUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    data_subject_id: UUID = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    processor_id: Optional[UUID] = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    event_type: ComplianceEventType = Column(Enum(ComplianceEventType), nullable=False, index=True)
    compliance_framework: str = Column(String(50), nullable=False, index=True)
    legal_basis: Optional[LegalBasis] = Column(Enum(LegalBasis), nullable=True)
    legal_reference: Optional[str] = Column(String(255), nullable=True)

    data_categories: Dict[str, Any] = Column(JSONB, nullable=False, default=list)
    processing_purpose: Optional[str] = Column(Text, nullable=True)
    recipients: Dict[str, Any] = Column(JSONB, nullable=False, default=list)

    source_country: Optional[str] = Column(String(2), nullable=True)
    destination_country: Optional[str] = Column(String(2), nullable=True)
    transfer_mechanism: Optional[str] = Column(String(255), nullable=True)

    consent_id: Optional[UUID] = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    consent_version: Optional[str] = Column(String(50), nullable=True)

    ip_address: Optional[str] = Column(String(45), nullable=True)
    user_agent: Optional[str] = Column(Text, nullable=True)
    request_id: Optional[str] = Column(String(100), nullable=True, index=True)

    description: Optional[str] = Column(Text, nullable=True)
    metadata: Dict[str, Any] = Column(JSONB, nullable=False, default=dict)

    created_at: datetime.datetime = Column(DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_compliance_audit_logs_tenant_event", "tenant_id", "event_type"),
        Index("ix_compliance_audit_logs_data_subject", "data_subject_id", "created_at"),
        Index("ix_compliance_audit_logs_framework", "compliance_framework", "event_type"),
    )

    def __repr__(self) -> str:
        return f"<ComplianceAuditLog(id={self.id}, event={self.event_type})>"
