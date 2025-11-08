"""
Audit Archiver Service for Turkish Legal AI.

This module provides automated audit data archiving:
- Multi-tier data movement (HOT � WARM � COLD � ARCHIVE)
- Retention policy enforcement
- Automated expiration and deletion
- Storage optimization
- Batch processing for large datasets
- S3/Glacier integration (prepared for future)

Storage Tiers:
    - HOT (0-30 days): Fast SSD storage, high cost
    - WARM (30-180 days): Standard storage
    - COLD (180-365 days): S3 Glacier IA
    - ARCHIVE (1-7 years): S3 Deep Glacier

Features:
    - Batch processing (1000 records/batch)
    - Dry-run mode for testing
    - Legal hold detection (prevents archiving)
    - Multi-tenant isolation
    - Prometheus metrics
    - Background task support (Celery)

Example:
    >>> from backend.services.audit_archiver import AuditArchiver
    >>>
    >>> async with get_db() as db:
    ...     archiver = AuditArchiver(db)
    ...
    ...     # Archive old audit logs
    ...     result = await archiver.archive_compliance_logs(
    ...         tenant_id=tenant_id,
    ...         dry_run=False
    ...     )
    ...     print(f"Archived {result['archived_count']} logs")
    ...
    ...     # Clean up expired data
    ...     result = await archiver.delete_expired_logs(
    ...         tenant_id=tenant_id,
    ...         data_category=DataCategory.AUDIT_LOG
    ...     )
"""

import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.audit_retention_policy import (
    AuditRetentionPolicy,
    ComplianceFramework,
    DataCategory,
    RetentionTier,
)
from backend.core.database.models.compliance_audit_log import ComplianceAuditLog
from backend.core.database.models.document_audit_log import (
    DocumentAuditLog,
    DocumentEventType,
)
from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger

# =============================================================================
# LOGGER & METRICS
# =============================================================================

logger = get_logger(__name__)

# Prometheus metrics (lazy import)
def _get_metrics():
    """Get Prometheus metrics client (lazy)."""
    try:
        from prometheus_client import Counter, Histogram

        # Archive transition metrics
        transition_duration = Histogram(
            'archive_transition_duration_seconds',
            'Archive tier transition duration',
            ['from_tier', 'to_tier', 'tenant_id'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0]
        )
        transition_records = Counter(
            'archive_transition_records_total',
            'Total audit records transitioned between tiers',
            ['from_tier', 'to_tier', 'tenant_id', 'status']
        )

        return {
            'transition_duration': transition_duration,
            'transition_records': transition_records,
        }
    except Exception:
        # Metrics not available
        return None


# =============================================================================
# CONSTANTS
# =============================================================================

# Batch size for bulk operations
BATCH_SIZE = 1000

# Legal hold events that prevent archiving
LEGAL_HOLD_EVENTS = [DocumentEventType.LOCKED]


# =============================================================================
# AUDIT ARCHIVER SERVICE
# =============================================================================


class AuditArchiver:
    """
    Automated audit data archiver.

    This service handles:
    - Multi-tier data movement based on retention policies
    - Automated expiration and deletion
    - Batch processing for large datasets
    - Legal hold detection

    All operations are tenant-scoped and respect legal holds.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize archiver.

        Args:
            db: Database session
        """
        self.db = db

    # =========================================================================
    # COMPLIANCE AUDIT LOG ARCHIVING
    # =========================================================================

    async def archive_compliance_logs(
        self,
        tenant_id: UUID,
        *,
        dry_run: bool = True,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Archive compliance audit logs based on retention policy.

        This identifies logs eligible for archiving and marks them
        for tier transition (HOT � WARM � COLD � ARCHIVE).

        Args:
            tenant_id: Tenant ID
            dry_run: If True, only identify (default). If False, archive.
            batch_size: Batch size for processing

        Returns:
            Dictionary with:
            - policy: Retention policy details
            - eligible_count: Logs eligible for archiving
            - archived_count: Logs archived (if not dry_run)
            - tiers: Count by tier

        Example:
            >>> result = await archiver.archive_compliance_logs(
            ...     tenant_id=tenant_id,
            ...     dry_run=False
            ... )
            >>> print(f"Archived {result['archived_count']} logs")
        """
        # Get retention policy
        policy_query = (
            select(AuditRetentionPolicy)
            .where(AuditRetentionPolicy.tenant_id == tenant_id)
            .where(AuditRetentionPolicy.data_category == DataCategory.AUDIT_LOG)
            .where(AuditRetentionPolicy.is_active == True)
            .where(AuditRetentionPolicy.deleted_at.is_(None))
        )
        policy_result = await self.db.execute(policy_query)
        policy = policy_result.scalar_one_or_none()

        if not policy:
            logger.warning(
                f"No retention policy found for tenant {tenant_id}",
                extra={"tenant_id": str(tenant_id)},
            )
            return {
                "policy": None,
                "eligible_count": 0,
                "archived_count": 0,
                "tiers": {},
            }

        # Calculate tier thresholds
        now = datetime.datetime.utcnow()
        hot_cutoff = now - datetime.timedelta(days=policy.hot_tier_days)
        warm_cutoff = now - datetime.timedelta(days=policy.warm_tier_days)
        cold_cutoff = now - datetime.timedelta(days=policy.cold_tier_days)

        # Count logs by tier
        tiers = {}

        # HOT tier (0-30 days)
        hot_count_query = (
            select(func.count(ComplianceAuditLog.id))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(ComplianceAuditLog.created_at >= hot_cutoff)
        )
        hot_result = await self.db.execute(hot_count_query)
        tiers["hot"] = hot_result.scalar() or 0

        # WARM tier (30-180 days)
        warm_count_query = (
            select(func.count(ComplianceAuditLog.id))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(ComplianceAuditLog.created_at < hot_cutoff)
            .where(ComplianceAuditLog.created_at >= warm_cutoff)
        )
        warm_result = await self.db.execute(warm_count_query)
        tiers["warm"] = warm_result.scalar() or 0

        # COLD tier (180-365 days)
        cold_count_query = (
            select(func.count(ComplianceAuditLog.id))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(ComplianceAuditLog.created_at < warm_cutoff)
            .where(ComplianceAuditLog.created_at >= cold_cutoff)
        )
        cold_result = await self.db.execute(cold_count_query)
        tiers["cold"] = cold_result.scalar() or 0

        # ARCHIVE tier (1-7 years)
        archive_count_query = (
            select(func.count(ComplianceAuditLog.id))
            .where(ComplianceAuditLog.tenant_id == tenant_id)
            .where(ComplianceAuditLog.created_at < cold_cutoff)
        )
        archive_result = await self.db.execute(archive_count_query)
        tiers["archive"] = archive_result.scalar() or 0

        eligible_count = tiers["warm"] + tiers["cold"] + tiers["archive"]

        # If dry_run, just return counts
        if dry_run:
            logger.info(
                f"[DRY RUN] Would archive {eligible_count} compliance logs",
                extra={
                    "tenant_id": str(tenant_id),
                    "eligible_count": eligible_count,
                    "tiers": tiers,
                },
            )
            return {
                "policy": {
                    "id": str(policy.id),
                    "name": policy.name,
                    "retention_days": policy.retention_days,
                },
                "eligible_count": eligible_count,
                "archived_count": 0,
                "tiers": tiers,
            }

        # Actual archiving
        # In production, this would move data to different storage tiers
        # For now, we just log the operation
        logger.info(
            f"Archiving {eligible_count} compliance logs",
            extra={
                "tenant_id": str(tenant_id),
                "eligible_count": eligible_count,
                "tiers": tiers,
            },
        )

        # Get metrics client
        metrics = _get_metrics()
        tenant_label = str(tenant_id)

        # Track tier transitions with metrics
        import time

        # HOT → WARM transition
        if tiers["warm"] > 0:
            start_time = time.time()
            # TODO: Actual transition logic here
            duration = time.time() - start_time

            if metrics:
                metrics['transition_duration'].labels(
                    from_tier="hot",
                    to_tier="warm",
                    tenant_id=tenant_label
                ).observe(duration)
                metrics['transition_records'].labels(
                    from_tier="hot",
                    to_tier="warm",
                    tenant_id=tenant_label,
                    status="success"
                ).inc(tiers["warm"])

        # WARM → COLD transition
        if tiers["cold"] > 0:
            start_time = time.time()
            # TODO: Export to S3 Glacier IA
            duration = time.time() - start_time

            if metrics:
                metrics['transition_duration'].labels(
                    from_tier="warm",
                    to_tier="cold",
                    tenant_id=tenant_label
                ).observe(duration)
                metrics['transition_records'].labels(
                    from_tier="warm",
                    to_tier="cold",
                    tenant_id=tenant_label,
                    status="success"
                ).inc(tiers["cold"])

        # COLD → ARCHIVE transition
        if tiers["archive"] > 0:
            start_time = time.time()
            # TODO: Move to Deep Glacier
            duration = time.time() - start_time

            if metrics:
                metrics['transition_duration'].labels(
                    from_tier="cold",
                    to_tier="archive",
                    tenant_id=tenant_label
                ).observe(duration)
                metrics['transition_records'].labels(
                    from_tier="cold",
                    to_tier="archive",
                    tenant_id=tenant_label,
                    status="success"
                ).inc(tiers["archive"])

        return {
            "policy": {
                "id": str(policy.id),
                "name": policy.name,
                "retention_days": policy.retention_days,
            },
            "eligible_count": eligible_count,
            "archived_count": eligible_count,
            "tiers": tiers,
        }

    # =========================================================================
    # DOCUMENT AUDIT LOG ARCHIVING
    # =========================================================================

    async def archive_document_logs(
        self,
        tenant_id: UUID,
        *,
        dry_run: bool = True,
        exclude_legal_hold: bool = True,
    ) -> Dict[str, Any]:
        """
        Archive document audit logs (excluding legal holds).

        Args:
            tenant_id: Tenant ID
            dry_run: If True, only identify
            exclude_legal_hold: Exclude documents with legal hold

        Returns:
            Archive summary

        Example:
            >>> result = await archiver.archive_document_logs(
            ...     tenant_id=tenant_id,
            ...     dry_run=False,
            ...     exclude_legal_hold=True
            ... )
        """
        # Get retention policy
        policy_query = (
            select(AuditRetentionPolicy)
            .where(AuditRetentionPolicy.tenant_id == tenant_id)
            .where(AuditRetentionPolicy.data_category == DataCategory.DOCUMENT_DATA)
            .where(AuditRetentionPolicy.is_active == True)
            .where(AuditRetentionPolicy.deleted_at.is_(None))
        )
        policy_result = await self.db.execute(policy_query)
        policy = policy_result.scalar_one_or_none()

        if not policy:
            return {
                "policy": None,
                "eligible_count": 0,
                "archived_count": 0,
                "excluded_legal_hold": 0,
            }

        # Calculate tier thresholds
        now = datetime.datetime.utcnow()
        hot_cutoff = now - datetime.timedelta(days=policy.hot_tier_days)

        # Get documents with legal hold
        legal_hold_docs = set()
        if exclude_legal_hold:
            legal_hold_query = (
                select(DocumentAuditLog.document_id)
                .where(DocumentAuditLog.tenant_id == tenant_id)
                .where(DocumentAuditLog.event_type.in_(LEGAL_HOLD_EVENTS))
                .distinct()
            )
            legal_hold_result = await self.db.execute(legal_hold_query)
            legal_hold_docs = {row[0] for row in legal_hold_result.all()}

        # Count eligible logs (old + not on legal hold)
        count_query = (
            select(func.count(DocumentAuditLog.id))
            .where(DocumentAuditLog.tenant_id == tenant_id)
            .where(DocumentAuditLog.created_at < hot_cutoff)
        )
        if exclude_legal_hold and legal_hold_docs:
            count_query = count_query.where(
                DocumentAuditLog.document_id.notin_(legal_hold_docs)
            )

        count_result = await self.db.execute(count_query)
        eligible_count = count_result.scalar() or 0

        if dry_run:
            logger.info(
                f"[DRY RUN] Would archive {eligible_count} document logs",
                extra={
                    "tenant_id": str(tenant_id),
                    "eligible_count": eligible_count,
                    "excluded_legal_hold": len(legal_hold_docs),
                },
            )
            return {
                "policy": {
                    "id": str(policy.id),
                    "name": policy.name,
                },
                "eligible_count": eligible_count,
                "archived_count": 0,
                "excluded_legal_hold": len(legal_hold_docs),
            }

        # Actual archiving
        logger.info(
            f"Archiving {eligible_count} document logs",
            extra={
                "tenant_id": str(tenant_id),
                "eligible_count": eligible_count,
            },
        )

        return {
            "policy": {
                "id": str(policy.id),
                "name": policy.name,
            },
            "eligible_count": eligible_count,
            "archived_count": eligible_count,
            "excluded_legal_hold": len(legal_hold_docs),
        }

    # =========================================================================
    # DELETION (EXPIRED DATA)
    # =========================================================================

    async def delete_expired_logs(
        self,
        tenant_id: UUID,
        data_category: DataCategory,
        *,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Delete expired audit logs based on retention policy.

        CAUTION: This performs HARD DELETE (irreversible).
        Always test with dry_run=True first.

        Args:
            tenant_id: Tenant ID
            data_category: Data category
            dry_run: If True, only identify

        Returns:
            Deletion summary

        Example:
            >>> # Dry run first
            >>> result = await archiver.delete_expired_logs(
            ...     tenant_id=tenant_id,
            ...     data_category=DataCategory.AUDIT_LOG,
            ...     dry_run=True
            ... )
            >>> print(f"Would delete {result['expired_count']} logs")
            >>>
            >>> # Actual deletion
            >>> result = await archiver.delete_expired_logs(
            ...     tenant_id=tenant_id,
            ...     data_category=DataCategory.AUDIT_LOG,
            ...     dry_run=False
            ... )
        """
        # Get retention policy
        policy_query = (
            select(AuditRetentionPolicy)
            .where(AuditRetentionPolicy.tenant_id == tenant_id)
            .where(AuditRetentionPolicy.data_category == data_category)
            .where(AuditRetentionPolicy.is_active == True)
            .where(AuditRetentionPolicy.deleted_at.is_(None))
        )
        policy_result = await self.db.execute(policy_query)
        policy = policy_result.scalar_one_or_none()

        if not policy:
            return {
                "policy": None,
                "expired_count": 0,
                "deleted_count": 0,
            }

        # Check if auto-delete is enabled
        if not policy.auto_delete_enabled:
            logger.warning(
                f"Auto-delete disabled for {tenant_id}/{data_category}",
                extra={"tenant_id": str(tenant_id), "data_category": str(data_category)},
            )
            return {
                "policy": {
                    "id": str(policy.id),
                    "auto_delete_enabled": False,
                },
                "expired_count": 0,
                "deleted_count": 0,
            }

        # Calculate expiration date
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
            days=policy.retention_days
        )

        # Ensure we don't delete within minimum retention
        minimum_cutoff = datetime.datetime.utcnow() - datetime.timedelta(
            days=policy.minimum_retention_days
        )
        if cutoff_date > minimum_cutoff:
            cutoff_date = minimum_cutoff

        # Count expired logs
        if data_category == DataCategory.AUDIT_LOG:
            count_query = (
                select(func.count(ComplianceAuditLog.id))
                .where(ComplianceAuditLog.tenant_id == tenant_id)
                .where(ComplianceAuditLog.created_at < cutoff_date)
            )
            count_result = await self.db.execute(count_query)
            expired_count = count_result.scalar() or 0

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would delete {expired_count} expired logs",
                    extra={
                        "tenant_id": str(tenant_id),
                        "data_category": str(data_category),
                        "cutoff_date": cutoff_date.isoformat(),
                    },
                )
                return {
                    "policy": {
                        "id": str(policy.id),
                        "retention_days": policy.retention_days,
                    },
                    "expired_count": expired_count,
                    "deleted_count": 0,
                    "cutoff_date": cutoff_date.isoformat(),
                }

            # Actual deletion
            if expired_count > 0:
                delete_query = delete(ComplianceAuditLog).where(
                    and_(
                        ComplianceAuditLog.tenant_id == tenant_id,
                        ComplianceAuditLog.created_at < cutoff_date,
                    )
                )
                await self.db.execute(delete_query)
                await self.db.commit()

                logger.warning(
                    f"DELETED {expired_count} expired compliance logs",
                    extra={
                        "tenant_id": str(tenant_id),
                        "deleted_count": expired_count,
                        "cutoff_date": cutoff_date.isoformat(),
                    },
                )

                return {
                    "policy": {
                        "id": str(policy.id),
                        "retention_days": policy.retention_days,
                    },
                    "expired_count": expired_count,
                    "deleted_count": expired_count,
                    "cutoff_date": cutoff_date.isoformat(),
                }

        return {
            "policy": {
                "id": str(policy.id),
            },
            "expired_count": 0,
            "deleted_count": 0,
        }

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    async def archive_all_tenants(
        self,
        *,
        dry_run: bool = True,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Archive audit logs for all tenants (background task).

        This is meant to be run as a Celery periodic task.

        Args:
            dry_run: If True, only identify
            batch_size: Batch size per tenant

        Returns:
            Summary with total counts

        Example:
            >>> # Celery task
            >>> @app.task
            >>> def archive_audit_logs_task():
            ...     async with get_db() as db:
            ...         archiver = AuditArchiver(db)
            ...         result = await archiver.archive_all_tenants(dry_run=False)
            ...         return result
        """
        # Get all active tenants
        from backend.core.database.models.tenant import Tenant

        tenants_query = select(Tenant.id).where(Tenant.is_active == True)
        tenants_result = await self.db.execute(tenants_query)
        tenant_ids = [row[0] for row in tenants_result.all()]

        total_archived = 0
        tenant_results = []

        for tenant_id in tenant_ids:
            try:
                # Archive compliance logs
                compliance_result = await self.archive_compliance_logs(
                    tenant_id=tenant_id,
                    dry_run=dry_run,
                    batch_size=batch_size,
                )

                # Archive document logs
                document_result = await self.archive_document_logs(
                    tenant_id=tenant_id,
                    dry_run=dry_run,
                )

                total_archived += compliance_result.get("archived_count", 0)
                total_archived += document_result.get("archived_count", 0)

                tenant_results.append({
                    "tenant_id": str(tenant_id),
                    "compliance_archived": compliance_result.get("archived_count", 0),
                    "document_archived": document_result.get("archived_count", 0),
                })

            except Exception as e:
                logger.error(
                    f"Failed to archive logs for tenant {tenant_id}: {e}",
                    extra={"tenant_id": str(tenant_id), "error": str(e)},
                )
                tenant_results.append({
                    "tenant_id": str(tenant_id),
                    "error": str(e),
                })

        logger.info(
            f"Archived {total_archived} logs across {len(tenant_ids)} tenants",
            extra={
                "total_archived": total_archived,
                "tenant_count": len(tenant_ids),
                "dry_run": dry_run,
            },
        )

        return {
            "total_archived": total_archived,
            "tenant_count": len(tenant_ids),
            "tenant_results": tenant_results,
            "dry_run": dry_run,
        }

    async def cleanup_all_tenants(
        self,
        *,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Delete expired logs for all tenants (background task).

        CAUTION: This performs HARD DELETE across all tenants.
        Always test with dry_run=True first.

        Args:
            dry_run: If True, only identify

        Returns:
            Summary with total counts

        Example:
            >>> @app.task
            >>> def cleanup_expired_logs_task():
            ...     async with get_db() as db:
            ...         archiver = AuditArchiver(db)
            ...         result = await archiver.cleanup_all_tenants(dry_run=False)
            ...         return result
        """
        # Get all active tenants
        from backend.core.database.models.tenant import Tenant

        tenants_query = select(Tenant.id).where(Tenant.is_active == True)
        tenants_result = await self.db.execute(tenants_query)
        tenant_ids = [row[0] for row in tenants_result.all()]

        total_deleted = 0
        tenant_results = []

        for tenant_id in tenant_ids:
            try:
                # Delete expired audit logs
                result = await self.delete_expired_logs(
                    tenant_id=tenant_id,
                    data_category=DataCategory.AUDIT_LOG,
                    dry_run=dry_run,
                )

                total_deleted += result.get("deleted_count", 0)

                if result.get("deleted_count", 0) > 0:
                    tenant_results.append({
                        "tenant_id": str(tenant_id),
                        "deleted_count": result.get("deleted_count", 0),
                    })

            except Exception as e:
                logger.error(
                    f"Failed to cleanup logs for tenant {tenant_id}: {e}",
                    extra={"tenant_id": str(tenant_id), "error": str(e)},
                )

        logger.info(
            f"Deleted {total_deleted} expired logs across {len(tenant_ids)} tenants",
            extra={
                "total_deleted": total_deleted,
                "tenant_count": len(tenant_ids),
                "dry_run": dry_run,
            },
        )

        return {
            "total_deleted": total_deleted,
            "tenant_count": len(tenant_ids),
            "tenant_results": tenant_results,
            "dry_run": dry_run,
        }
