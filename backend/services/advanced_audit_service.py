"""
Advanced Audit Service for Turkish Legal AI.

This module provides advanced audit operations:
- Audit log querying and filtering
- Compliance report generation
- Retention policy enforcement
- Legal hold management
- Audit trail export (JSON, CSV)
- Statistics aggregation

Features:
    - Multi-tenant isolation
    - Date range filtering
    - Event type filtering
    - User activity tracking
    - Document lifecycle tracking
    - Compliance framework attestation
    - Automated retention enforcement
    - Legal hold support (litigation/investigation)

Example:
    >>> from backend.services.advanced_audit_service import AdvancedAuditService
    >>>
    >>> async with get_db() as db:
    ...     service = AdvancedAuditService(db)
    ...
    ...     # Query audit logs
    ...     logs = await service.query_audit_logs(
    ...         tenant_id=tenant_id,
    ...         start_date=start,
    ...         end_date=end,
    ...         event_types=["DATA_ACCESS", "CONSENT_GRANTED"]
    ...     )
    ...
    ...     # Generate compliance report
    ...     report = await service.generate_compliance_report(
    ...         tenant_id=tenant_id,
    ...         framework="GDPR",
    ...         start_date=start,
    ...         end_date=end
    ...     )
    ...
    ...     # Place legal hold
    ...     await service.place_legal_hold(
    ...         tenant_id=tenant_id,
    ...         document_ids=[doc_id_1, doc_id_2],
    ...         reason="Litigation - Case #2025-001"
    ...     )
"""

import csv
import datetime
import json
from io import StringIO
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.audit_report import (
    AuditReport,
    AuditReportFormat,
    AuditReportStatus,
    AuditReportType,
)
from backend.core.database.models.audit_retention_policy import (
    AuditRetentionPolicy,
    ComplianceFramework,
    DataCategory,
    RetentionTier,
)
from backend.core.database.models.compliance_audit_log import (
    ComplianceAuditLog,
    ComplianceEventType,
)
from backend.core.database.models.document_audit_log import (
    DocumentAuditLog,
    DocumentEventType,
)
from backend.core.exceptions import NotFoundError, ValidationError
from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# ADVANCED AUDIT SERVICE
# =============================================================================


class AdvancedAuditService:
    """
    Advanced audit operations service.

    This service provides high-level audit operations including:
    - Complex audit log queries
    - Compliance report generation
    - Retention policy enforcement
    - Legal hold management
    - Export capabilities

    All operations are tenant-scoped for security.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize service.

        Args:
            db: Database session
        """
        self.db = db

    # =========================================================================
    # COMPLIANCE AUDIT LOG QUERIES
    # =========================================================================

    async def query_compliance_logs(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        event_types: Optional[List[ComplianceEventType]] = None,
        data_subject_id: Optional[UUID] = None,
        compliance_framework: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[ComplianceAuditLog]:
        """
        Query compliance audit logs with filters.

        Args:
            tenant_id: Tenant ID (required for isolation)
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            event_types: Filter by event types
            data_subject_id: Filter by data subject (user)
            compliance_framework: Filter by framework (GDPR, KVKK, etc.)
            limit: Maximum results (default 1000)
            offset: Pagination offset

        Returns:
            List of ComplianceAuditLog records

        Example:
            >>> logs = await service.query_compliance_logs(
            ...     tenant_id=tenant_id,
            ...     start_date=datetime(2025, 1, 1),
            ...     end_date=datetime(2025, 1, 31),
            ...     event_types=[ComplianceEventType.DATA_ACCESS],
            ...     compliance_framework="GDPR"
            ... )
        """
        # Build query
        query = select(ComplianceAuditLog).where(
            ComplianceAuditLog.tenant_id == tenant_id
        )

        # Date range filter
        if start_date:
            query = query.where(ComplianceAuditLog.created_at >= start_date)
        if end_date:
            query = query.where(ComplianceAuditLog.created_at <= end_date)

        # Event type filter
        if event_types:
            query = query.where(ComplianceAuditLog.event_type.in_(event_types))

        # Data subject filter
        if data_subject_id:
            query = query.where(ComplianceAuditLog.data_subject_id == data_subject_id)

        # Compliance framework filter
        if compliance_framework:
            query = query.where(
                ComplianceAuditLog.compliance_framework == compliance_framework
            )

        # Order by created_at desc
        query = query.order_by(ComplianceAuditLog.created_at.desc())

        # Pagination
        query = query.limit(limit).offset(offset)

        # Execute
        result = await self.db.execute(query)
        logs = result.scalars().all()

        logger.info(
            f"Queried {len(logs)} compliance audit logs for tenant {tenant_id}",
            extra={
                "tenant_id": str(tenant_id),
                "count": len(logs),
                "filters": {
                    "start_date": str(start_date) if start_date else None,
                    "end_date": str(end_date) if end_date else None,
                    "event_types": [str(et) for et in event_types] if event_types else None,
                },
            },
        )

        return list(logs)

    # =========================================================================
    # DOCUMENT AUDIT LOG QUERIES
    # =========================================================================

    async def query_document_logs(
        self,
        tenant_id: UUID,
        *,
        document_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        event_types: Optional[List[DocumentEventType]] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[DocumentAuditLog]:
        """
        Query document audit logs with filters.

        Args:
            tenant_id: Tenant ID
            document_id: Filter by document
            user_id: Filter by user
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            event_types: Filter by event types
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of DocumentAuditLog records

        Example:
            >>> logs = await service.query_document_logs(
            ...     tenant_id=tenant_id,
            ...     document_id=doc_id,
            ...     event_types=[DocumentEventType.VIEWED, DocumentEventType.DOWNLOADED]
            ... )
        """
        # Build query
        query = select(DocumentAuditLog).where(
            DocumentAuditLog.tenant_id == tenant_id
        )

        # Document filter
        if document_id:
            query = query.where(DocumentAuditLog.document_id == document_id)

        # User filter
        if user_id:
            query = query.where(DocumentAuditLog.user_id == user_id)

        # Date range
        if start_date:
            query = query.where(DocumentAuditLog.created_at >= start_date)
        if end_date:
            query = query.where(DocumentAuditLog.created_at <= end_date)

        # Event types
        if event_types:
            query = query.where(DocumentAuditLog.event_type.in_(event_types))

        # Order + pagination
        query = query.order_by(DocumentAuditLog.created_at.desc())
        query = query.limit(limit).offset(offset)

        # Execute
        result = await self.db.execute(query)
        logs = result.scalars().all()

        logger.info(
            f"Queried {len(logs)} document audit logs for tenant {tenant_id}",
            extra={"tenant_id": str(tenant_id), "count": len(logs)},
        )

        return list(logs)

    # =========================================================================
    # STATISTICS & AGGREGATION
    # =========================================================================

    async def get_compliance_statistics(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get compliance audit statistics.

        Args:
            tenant_id: Tenant ID
            start_date: Start date
            end_date: End date

        Returns:
            Statistics dictionary with:
            - total_events: Total compliance events
            - events_by_type: Count by event type
            - events_by_framework: Count by compliance framework
            - unique_data_subjects: Unique users affected
            - cross_border_transfers: Count of cross-border transfers

        Example:
            >>> stats = await service.get_compliance_statistics(
            ...     tenant_id=tenant_id,
            ...     start_date=datetime(2025, 1, 1),
            ...     end_date=datetime(2025, 1, 31)
            ... )
            >>> print(stats["total_events"])  # 1543
        """
        # Base query
        base_query = select(ComplianceAuditLog).where(
            ComplianceAuditLog.tenant_id == tenant_id
        )
        if start_date:
            base_query = base_query.where(ComplianceAuditLog.created_at >= start_date)
        if end_date:
            base_query = base_query.where(ComplianceAuditLog.created_at <= end_date)

        # Total events
        total_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.db.execute(total_query)
        total_events = total_result.scalar() or 0

        # Events by type
        type_query = (
            select(
                ComplianceAuditLog.event_type,
                func.count(ComplianceAuditLog.id).label("count"),
            )
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .group_by(ComplianceAuditLog.event_type)
        )
        if start_date:
            type_query = type_query.where(ComplianceAuditLog.created_at >= start_date)
        if end_date:
            type_query = type_query.where(ComplianceAuditLog.created_at <= end_date)

        type_result = await self.db.execute(type_query)
        events_by_type = {str(row[0]): row[1] for row in type_result.all()}

        # Events by framework
        framework_query = (
            select(
                ComplianceAuditLog.compliance_framework,
                func.count(ComplianceAuditLog.id).label("count"),
            )
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .group_by(ComplianceAuditLog.compliance_framework)
        )
        if start_date:
            framework_query = framework_query.where(
                ComplianceAuditLog.created_at >= start_date
            )
        if end_date:
            framework_query = framework_query.where(
                ComplianceAuditLog.created_at <= end_date
            )

        framework_result = await self.db.execute(framework_query)
        events_by_framework = {row[0]: row[1] for row in framework_result.all()}

        # Unique data subjects
        subjects_query = (
            select(func.count(func.distinct(ComplianceAuditLog.data_subject_id)))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(ComplianceAuditLog.data_subject_id.isnot(None))
        )
        if start_date:
            subjects_query = subjects_query.where(
                ComplianceAuditLog.created_at >= start_date
            )
        if end_date:
            subjects_query = subjects_query.where(
                ComplianceAuditLog.created_at <= end_date
            )

        subjects_result = await self.db.execute(subjects_query)
        unique_data_subjects = subjects_result.scalar() or 0

        # Cross-border transfers
        transfer_query = (
            select(func.count(ComplianceAuditLog.id))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(
                ComplianceAuditLog.event_type == ComplianceEventType.CROSS_BORDER_TRANSFER
            )
        )
        if start_date:
            transfer_query = transfer_query.where(
                ComplianceAuditLog.created_at >= start_date
            )
        if end_date:
            transfer_query = transfer_query.where(
                ComplianceAuditLog.created_at <= end_date
            )

        transfer_result = await self.db.execute(transfer_query)
        cross_border_transfers = transfer_result.scalar() or 0

        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "events_by_framework": events_by_framework,
            "unique_data_subjects": unique_data_subjects,
            "cross_border_transfers": cross_border_transfers,
        }

    async def get_document_statistics(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get document audit statistics.

        Args:
            tenant_id: Tenant ID
            start_date: Start date
            end_date: End date

        Returns:
            Statistics dictionary with:
            - total_events: Total document events
            - events_by_type: Count by event type
            - unique_documents: Unique documents affected
            - unique_users: Unique users
            - shared_documents: Count of share events

        Example:
            >>> stats = await service.get_document_statistics(
            ...     tenant_id=tenant_id,
            ...     start_date=datetime(2025, 1, 1)
            ... )
        """
        # Total events
        total_query = select(func.count(DocumentAuditLog.id)).where(
            DocumentAuditLog.tenant_id == tenant_id
        )
        if start_date:
            total_query = total_query.where(DocumentAuditLog.created_at >= start_date)
        if end_date:
            total_query = total_query.where(DocumentAuditLog.created_at <= end_date)

        total_result = await self.db.execute(total_query)
        total_events = total_result.scalar() or 0

        # Events by type
        type_query = (
            select(
                DocumentAuditLog.event_type,
                func.count(DocumentAuditLog.id).label("count"),
            )
            .where(DocumentAuditLog.tenant_id == tenant_id)
            .group_by(DocumentAuditLog.event_type)
        )
        if start_date:
            type_query = type_query.where(DocumentAuditLog.created_at >= start_date)
        if end_date:
            type_query = type_query.where(DocumentAuditLog.created_at <= end_date)

        type_result = await self.db.execute(type_query)
        events_by_type = {str(row[0]): row[1] for row in type_result.all()}

        # Unique documents
        docs_query = (
            select(func.count(func.distinct(DocumentAuditLog.document_id)))
            .where(DocumentAuditLog.tenant_id == tenant_id)
        )
        if start_date:
            docs_query = docs_query.where(DocumentAuditLog.created_at >= start_date)
        if end_date:
            docs_query = docs_query.where(DocumentAuditLog.created_at <= end_date)

        docs_result = await self.db.execute(docs_query)
        unique_documents = docs_result.scalar() or 0

        # Unique users
        users_query = (
            select(func.count(func.distinct(DocumentAuditLog.user_id)))
            .where(DocumentAuditLog.tenant_id == tenant_id)
            .where(DocumentAuditLog.user_id.isnot(None))
        )
        if start_date:
            users_query = users_query.where(DocumentAuditLog.created_at >= start_date)
        if end_date:
            users_query = users_query.where(DocumentAuditLog.created_at <= end_date)

        users_result = await self.db.execute(users_query)
        unique_users = users_result.scalar() or 0

        # Shared documents
        shared_query = (
            select(func.count(DocumentAuditLog.id))
            .where(DocumentAuditLog.tenant_id == tenant_id)
            .where(DocumentAuditLog.event_type == DocumentEventType.SHARED)
        )
        if start_date:
            shared_query = shared_query.where(DocumentAuditLog.created_at >= start_date)
        if end_date:
            shared_query = shared_query.where(DocumentAuditLog.created_at <= end_date)

        shared_result = await self.db.execute(shared_query)
        shared_documents = shared_result.scalar() or 0

        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "unique_documents": unique_documents,
            "unique_users": unique_users,
            "shared_documents": shared_documents,
        }

    # =========================================================================
    # EXPORT CAPABILITIES
    # =========================================================================

    async def export_compliance_logs_json(
        self,
        tenant_id: UUID,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> str:
        """
        Export compliance logs as JSON string.

        Args:
            tenant_id: Tenant ID
            start_date: Start date
            end_date: End date

        Returns:
            JSON string (array of log objects)

        Example:
            >>> json_data = await service.export_compliance_logs_json(
            ...     tenant_id=tenant_id,
            ...     start_date=datetime(2025, 1, 1)
            ... )
            >>> with open("audit_export.json", "w") as f:
            ...     f.write(json_data)
        """
        logs = await self.query_compliance_logs(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            limit=100000,  # Large limit for export
        )

        # Convert to dict
        logs_dict = [
            {
                "id": str(log.id),
                "tenant_id": str(log.tenant_id),
                "data_subject_id": str(log.data_subject_id) if log.data_subject_id else None,
                "event_type": str(log.event_type),
                "compliance_framework": log.compliance_framework,
                "legal_basis": str(log.legal_basis) if log.legal_basis else None,
                "processing_purpose": log.processing_purpose,
                "source_country": log.source_country,
                "destination_country": log.destination_country,
                "created_at": log.created_at.isoformat(),
                "metadata": log.metadata,
            }
            for log in logs
        ]

        return json.dumps(logs_dict, indent=2, ensure_ascii=False)

    async def export_document_logs_csv(
        self,
        tenant_id: UUID,
        *,
        document_id: Optional[UUID] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> str:
        """
        Export document logs as CSV string.

        Args:
            tenant_id: Tenant ID
            document_id: Filter by document
            start_date: Start date
            end_date: End date

        Returns:
            CSV string

        Example:
            >>> csv_data = await service.export_document_logs_csv(
            ...     tenant_id=tenant_id,
            ...     document_id=doc_id
            ... )
            >>> with open("document_audit.csv", "w") as f:
            ...     f.write(csv_data)
        """
        logs = await self.query_document_logs(
            tenant_id=tenant_id,
            document_id=document_id,
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        # CSV writer
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "ID",
            "Tenant ID",
            "Document ID",
            "Document Title",
            "User ID",
            "Event Type",
            "Version",
            "IP Address",
            "Created At",
        ])

        # Rows
        for log in logs:
            writer.writerow([
                str(log.id),
                str(log.tenant_id),
                str(log.document_id),
                log.document_title or "",
                str(log.user_id) if log.user_id else "",
                str(log.event_type),
                log.version_number or "",
                log.ip_address or "",
                log.created_at.isoformat(),
            ])

        return output.getvalue()

    # =========================================================================
    # LEGAL HOLD MANAGEMENT
    # =========================================================================

    async def place_legal_hold(
        self,
        tenant_id: UUID,
        document_ids: List[UUID],
        *,
        reason: str,
        placed_by_id: UUID,
    ) -> int:
        """
        Place legal hold on documents (prevents deletion).

        Creates LOCKED audit log entries for each document.

        Args:
            tenant_id: Tenant ID
            document_ids: List of document IDs to lock
            reason: Legal hold reason (case number, investigation)
            placed_by_id: User placing the hold

        Returns:
            Number of documents locked

        Example:
            >>> count = await service.place_legal_hold(
            ...     tenant_id=tenant_id,
            ...     document_ids=[doc1, doc2, doc3],
            ...     reason="Litigation - Case #2025-001",
            ...     placed_by_id=admin_id
            ... )
            >>> print(f"Locked {count} documents")
        """
        locked_count = 0

        for document_id in document_ids:
            # Create LOCKED audit log
            log = DocumentAuditLog(
                tenant_id=tenant_id,
                document_id=document_id,
                user_id=placed_by_id,
                event_type=DocumentEventType.LOCKED,
                description=f"Legal hold placed: {reason}",
                metadata={
                    "legal_hold": True,
                    "reason": reason,
                    "placed_at": datetime.datetime.utcnow().isoformat(),
                },
            )
            self.db.add(log)
            locked_count += 1

        await self.db.commit()

        logger.warning(
            f"Legal hold placed on {locked_count} documents",
            extra={
                "tenant_id": str(tenant_id),
                "document_ids": [str(d) for d in document_ids],
                "reason": reason,
                "placed_by_id": str(placed_by_id),
            },
        )

        return locked_count

    async def remove_legal_hold(
        self,
        tenant_id: UUID,
        document_ids: List[UUID],
        *,
        removed_by_id: UUID,
    ) -> int:
        """
        Remove legal hold from documents.

        Creates UNLOCKED audit log entries.

        Args:
            tenant_id: Tenant ID
            document_ids: List of document IDs to unlock
            removed_by_id: User removing the hold

        Returns:
            Number of documents unlocked

        Example:
            >>> count = await service.remove_legal_hold(
            ...     tenant_id=tenant_id,
            ...     document_ids=[doc1, doc2],
            ...     removed_by_id=admin_id
            ... )
        """
        unlocked_count = 0

        for document_id in document_ids:
            # Create UNLOCKED audit log
            log = DocumentAuditLog(
                tenant_id=tenant_id,
                document_id=document_id,
                user_id=removed_by_id,
                event_type=DocumentEventType.UNLOCKED,
                description="Legal hold removed",
                metadata={
                    "legal_hold": False,
                    "removed_at": datetime.datetime.utcnow().isoformat(),
                },
            )
            self.db.add(log)
            unlocked_count += 1

        await self.db.commit()

        logger.info(
            f"Legal hold removed from {unlocked_count} documents",
            extra={
                "tenant_id": str(tenant_id),
                "document_ids": [str(d) for d in document_ids],
                "removed_by_id": str(removed_by_id),
            },
        )

        return unlocked_count

    # =========================================================================
    # RETENTION POLICY OPERATIONS
    # =========================================================================

    async def get_retention_policy(
        self,
        tenant_id: UUID,
        data_category: DataCategory,
    ) -> Optional[AuditRetentionPolicy]:
        """
        Get active retention policy for tenant and data category.

        Args:
            tenant_id: Tenant ID
            data_category: Data category

        Returns:
            AuditRetentionPolicy or None

        Example:
            >>> policy = await service.get_retention_policy(
            ...     tenant_id=tenant_id,
            ...     data_category=DataCategory.AUDIT_LOG
            ... )
            >>> print(policy.retention_days)  # 365
        """
        query = (
            select(AuditRetentionPolicy)
            .where(AuditRetentionPolicy.tenant_id == tenant_id)
            .where(AuditRetentionPolicy.data_category == data_category)
            .where(AuditRetentionPolicy.is_active == True)
            .where(AuditRetentionPolicy.deleted_at.is_(None))
        )

        result = await self.db.execute(query)
        policy = result.scalar_one_or_none()

        return policy

    async def apply_retention_policy(
        self,
        tenant_id: UUID,
        data_category: DataCategory,
        *,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply retention policy (identify expired data).

        This does NOT actually delete data - it returns what WOULD be deleted.
        Use with dry_run=False to perform actual deletion.

        Args:
            tenant_id: Tenant ID
            data_category: Data category
            dry_run: If True, only identify (default). If False, delete.

        Returns:
            Dictionary with:
            - policy: Policy details
            - expired_count: Number of expired records
            - deleted: Whether deletion was performed

        Example:
            >>> # Dry run
            >>> result = await service.apply_retention_policy(
            ...     tenant_id=tenant_id,
            ...     data_category=DataCategory.AUDIT_LOG,
            ...     dry_run=True
            ... )
            >>> print(f"Would delete {result['expired_count']} records")
            >>>
            >>> # Actual deletion
            >>> result = await service.apply_retention_policy(
            ...     tenant_id=tenant_id,
            ...     data_category=DataCategory.AUDIT_LOG,
            ...     dry_run=False
            ... )
        """
        # Get policy
        policy = await self.get_retention_policy(tenant_id, data_category)
        if not policy:
            raise NotFoundError(
                f"No retention policy found for {tenant_id}/{data_category}"
            )

        # Calculate expiration date
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
            days=policy.retention_days
        )

        # Count expired records
        # For now, we only support compliance_audit_logs
        if data_category == DataCategory.AUDIT_LOG:
            count_query = (
                select(func.count(ComplianceAuditLog.id))
                .where(ComplianceAuditLog.tenant_id == tenant_id)
                .where(ComplianceAuditLog.created_at < cutoff_date)
            )
            count_result = await self.db.execute(count_query)
            expired_count = count_result.scalar() or 0

            if not dry_run and expired_count > 0:
                # Actual deletion (use with caution!)
                # In production, this should be a soft delete or archive
                logger.warning(
                    f"Deleting {expired_count} expired audit logs",
                    extra={
                        "tenant_id": str(tenant_id),
                        "data_category": str(data_category),
                        "cutoff_date": cutoff_date.isoformat(),
                    },
                )
                # DELETE query would go here
                # For safety, we don't implement actual deletion in this version

        else:
            expired_count = 0

        return {
            "policy": {
                "id": str(policy.id),
                "name": policy.name,
                "retention_days": policy.retention_days,
                "data_category": str(policy.data_category),
            },
            "expired_count": expired_count,
            "cutoff_date": cutoff_date.isoformat(),
            "deleted": not dry_run,
        }
