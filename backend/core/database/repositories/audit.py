"""
Audit Repository for Turkish Legal AI.

This module provides database repository layer for audit operations.
Repository pattern provides a clean separation between business logic
and data access.

Features:
    - Create audit log entries
    - Query audit logs with filters
    - Batch operations
    - Transaction management
    - Type-safe CRUD operations

Example:
    >>> from backend.core.database.repositories.audit import AuditRepository
    >>>
    >>> async with get_db() as db:
    ...     repo = AuditRepository(db)
    ...
    ...     # Create compliance log
    ...     log = await repo.create_compliance_log(
    ...         tenant_id=tenant_id,
    ...         event_type=ComplianceEventType.DATA_ACCESS,
    ...         data_subject_id=user_id,
    ...         compliance_framework="GDPR"
    ...     )
    ...
    ...     # Query logs
    ...     logs = await repo.find_compliance_logs(
    ...         tenant_id=tenant_id,
    ...         start_date=start,
    ...         limit=100
    ...     )
"""

import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.audit_report import AuditReport
from backend.core.database.models.audit_retention_policy import AuditRetentionPolicy
from backend.core.database.models.compliance_audit_log import (
    ComplianceAuditLog,
    ComplianceEventType,
    LegalBasis,
)
from backend.core.database.models.document_audit_log import (
    DocumentAuditLog,
    DocumentEventType,
)
from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# AUDIT REPOSITORY
# =============================================================================


class AuditRepository:
    """
    Audit repository for database operations.

    This repository provides clean data access layer for audit operations.
    All methods are tenant-scoped for multi-tenancy security.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        self.db = db

    # =========================================================================
    # COMPLIANCE AUDIT LOG OPERATIONS
    # =========================================================================

    async def create_compliance_log(
        self,
        tenant_id: UUID,
        event_type: ComplianceEventType,
        compliance_framework: str,
        *,
        data_subject_id: Optional[UUID] = None,
        processor_id: Optional[UUID] = None,
        legal_basis: Optional[LegalBasis] = None,
        processing_purpose: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ComplianceAuditLog:
        """
        Create compliance audit log entry.

        Args:
            tenant_id: Tenant ID
            event_type: Compliance event type
            compliance_framework: Framework (GDPR, KVKK, etc.)
            data_subject_id: Data subject (user) ID
            processor_id: Processor ID
            legal_basis: GDPR legal basis
            processing_purpose: Processing purpose
            ip_address: Client IP
            user_agent: User agent
            request_id: Request ID
            description: Description
            metadata: Additional metadata

        Returns:
            Created ComplianceAuditLog
        """
        log = ComplianceAuditLog(
            tenant_id=tenant_id,
            data_subject_id=data_subject_id,
            processor_id=processor_id,
            event_type=event_type,
            compliance_framework=compliance_framework,
            legal_basis=legal_basis,
            data_categories=[],
            processing_purpose=processing_purpose,
            recipients=[],
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            description=description,
            metadata=metadata or {},
        )

        self.db.add(log)
        await self.db.flush()

        return log

    async def find_compliance_logs(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        event_type: Optional[ComplianceEventType] = None,
        data_subject_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ComplianceAuditLog]:
        """
        Find compliance audit logs with filters.

        Args:
            tenant_id: Tenant ID
            start_date: Start date filter
            end_date: End date filter
            event_type: Event type filter
            data_subject_id: Data subject filter
            limit: Results limit
            offset: Pagination offset

        Returns:
            List of ComplianceAuditLog records
        """
        query = select(ComplianceAuditLog).where(
            ComplianceAuditLog.tenant_id == tenant_id
        )

        if start_date:
            query = query.where(ComplianceAuditLog.created_at >= start_date)
        if end_date:
            query = query.where(ComplianceAuditLog.created_at <= end_date)
        if event_type:
            query = query.where(ComplianceAuditLog.event_type == event_type)
        if data_subject_id:
            query = query.where(ComplianceAuditLog.data_subject_id == data_subject_id)

        query = query.order_by(desc(ComplianceAuditLog.created_at))
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count_compliance_logs(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> int:
        """
        Count compliance audit logs.

        Args:
            tenant_id: Tenant ID
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Total count
        """
        query = select(func.count(ComplianceAuditLog.id)).where(
            ComplianceAuditLog.tenant_id == tenant_id
        )

        if start_date:
            query = query.where(ComplianceAuditLog.created_at >= start_date)
        if end_date:
            query = query.where(ComplianceAuditLog.created_at <= end_date)

        result = await self.db.execute(query)
        return result.scalar() or 0

    # =========================================================================
    # DOCUMENT AUDIT LOG OPERATIONS
    # =========================================================================

    async def create_document_log(
        self,
        tenant_id: UUID,
        document_id: UUID,
        event_type: DocumentEventType,
        *,
        user_id: Optional[UUID] = None,
        document_title: Optional[str] = None,
        document_type: Optional[str] = None,
        changes: Optional[dict] = None,
        version_number: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> DocumentAuditLog:
        """
        Create document audit log entry.

        Args:
            tenant_id: Tenant ID
            document_id: Document ID
            event_type: Document event type
            user_id: User ID
            document_title: Document title
            document_type: Document type
            changes: Field-level changes
            version_number: Document version
            ip_address: Client IP
            user_agent: User agent
            request_id: Request ID
            description: Description
            metadata: Additional metadata

        Returns:
            Created DocumentAuditLog
        """
        log = DocumentAuditLog(
            tenant_id=tenant_id,
            document_id=document_id,
            document_type=document_type,
            document_title=document_title,
            user_id=user_id,
            event_type=event_type,
            changes=changes or {},
            version_number=version_number,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            description=description,
            metadata=metadata or {},
        )

        self.db.add(log)
        await self.db.flush()

        return log

    async def find_document_logs(
        self,
        tenant_id: UUID,
        *,
        document_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        event_type: Optional[DocumentEventType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentAuditLog]:
        """
        Find document audit logs with filters.

        Args:
            tenant_id: Tenant ID
            document_id: Document ID filter
            user_id: User ID filter
            event_type: Event type filter
            limit: Results limit
            offset: Pagination offset

        Returns:
            List of DocumentAuditLog records
        """
        query = select(DocumentAuditLog).where(
            DocumentAuditLog.tenant_id == tenant_id
        )

        if document_id:
            query = query.where(DocumentAuditLog.document_id == document_id)
        if user_id:
            query = query.where(DocumentAuditLog.user_id == user_id)
        if event_type:
            query = query.where(DocumentAuditLog.event_type == event_type)

        query = query.order_by(desc(DocumentAuditLog.created_at))
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    # =========================================================================
    # RETENTION POLICY OPERATIONS
    # =========================================================================

    async def get_active_retention_policy(
        self,
        tenant_id: UUID,
        data_category: str,
    ) -> Optional[AuditRetentionPolicy]:
        """
        Get active retention policy for tenant and category.

        Args:
            tenant_id: Tenant ID
            data_category: Data category

        Returns:
            AuditRetentionPolicy or None
        """
        query = (
            select(AuditRetentionPolicy)
            .where(AuditRetentionPolicy.tenant_id == tenant_id)
            .where(AuditRetentionPolicy.data_category == data_category)
            .where(AuditRetentionPolicy.is_active == True)
            .where(AuditRetentionPolicy.deleted_at.is_(None))
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    # =========================================================================
    # AUDIT REPORT OPERATIONS
    # =========================================================================

    async def create_audit_report(
        self,
        tenant_id: UUID,
        report_type: str,
        title: str,
        *,
        created_by_id: Optional[UUID] = None,
        description: Optional[str] = None,
        parameters: Optional[dict] = None,
        format: str = "PDF",
    ) -> AuditReport:
        """
        Create audit report entry.

        Args:
            tenant_id: Tenant ID
            report_type: Report type
            title: Report title
            created_by_id: Creator user ID
            description: Description
            parameters: Report parameters
            format: Output format

        Returns:
            Created AuditReport
        """
        from backend.core.database.models.audit_report import (
            AuditReportFormat,
            AuditReportStatus,
            AuditReportType,
        )

        report = AuditReport(
            tenant_id=tenant_id,
            created_by_id=created_by_id,
            report_type=AuditReportType(report_type),
            title=title,
            description=description,
            parameters=parameters or {},
            status=AuditReportStatus.PENDING,
            format=AuditReportFormat(format),
        )

        self.db.add(report)
        await self.db.flush()

        return report

    async def get_report_by_id(
        self,
        report_id: UUID,
        tenant_id: UUID,
    ) -> Optional[AuditReport]:
        """
        Get audit report by ID (tenant-scoped).

        Args:
            report_id: Report ID
            tenant_id: Tenant ID

        Returns:
            AuditReport or None
        """
        query = (
            select(AuditReport)
            .where(AuditReport.id == report_id)
            .where(AuditReport.tenant_id == tenant_id)
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    async def bulk_create_compliance_logs(
        self,
        logs: List[ComplianceAuditLog],
    ) -> None:
        """
        Bulk insert compliance audit logs.

        Args:
            logs: List of ComplianceAuditLog objects

        Note:
            This is more efficient for large batches (100+ records).
        """
        self.db.add_all(logs)
        await self.db.flush()

        logger.info(
            f"Bulk created {len(logs)} compliance audit logs",
            extra={"count": len(logs)},
        )
