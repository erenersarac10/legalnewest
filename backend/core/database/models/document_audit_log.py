"""
Document Audit Log Database Model for Turkish Legal AI.

Tracks all document lifecycle events for legal compliance.

Events: Create, read, update, delete, share, download, print
Use Cases: Contract audit trail, evidence tracking, legal hold
Immutable: Records cannot be modified once created
"""

import datetime
import enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

from backend.core.database.base import Base


class DocumentEventType(str, enum.Enum):
    """Document lifecycle events."""
    CREATED = "created"
    VIEWED = "viewed"
    UPDATED = "updated"
    DELETED = "deleted"
    SHARED = "shared"
    DOWNLOADED = "downloaded"
    PRINTED = "printed"
    EXPORTED = "exported"
    PERMISSIONS_CHANGED = "permissions_changed"
    MOVED = "moved"
    COPIED = "copied"
    LOCKED = "locked"
    UNLOCKED = "unlocked"


class DocumentAuditLog(Base):
    """Document audit log for legal compliance. Immutable."""

    __tablename__ = "document_audit_logs"

    id: UUID = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    tenant_id: UUID = Column(PGUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    document_id: UUID = Column(PGUUID(as_uuid=True), nullable=False, index=True, doc="Document being audited")
    document_type: Optional[str] = Column(String(100), nullable=True, doc="Contract, template, etc.")
    document_title: Optional[str] = Column(String(500), nullable=True)

    user_id: UUID = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type: DocumentEventType = Column(Enum(DocumentEventType), nullable=False, index=True)

    # Change tracking
    changes: Dict[str, Any] = Column(JSONB, nullable=False, default=dict, doc="Field-level changes (before/after)")
    version_number: Optional[int] = Column(Integer, nullable=True, doc="Document version")

    # Share tracking
    shared_with_user_id: Optional[UUID] = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    permission_level: Optional[str] = Column(String(50), nullable=True, doc="read, write, admin")

    # Request context
    ip_address: Optional[str] = Column(String(45), nullable=True)
    user_agent: Optional[str] = Column(Text, nullable=True)
    request_id: Optional[str] = Column(String(100), nullable=True, index=True)

    description: Optional[str] = Column(Text, nullable=True)
    metadata: Dict[str, Any] = Column(JSONB, nullable=False, default=dict)

    created_at: datetime.datetime = Column(DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_document_audit_logs_document", "document_id", "created_at"),
        Index("ix_document_audit_logs_user", "user_id", "event_type"),
        Index("ix_document_audit_logs_tenant_event", "tenant_id", "event_type"),
    )

    def __repr__(self) -> str:
        return f"<DocumentAuditLog(id={self.id}, doc={self.document_id}, event={self.event_type})>"
