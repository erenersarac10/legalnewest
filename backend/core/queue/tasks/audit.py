"""
Celery Audit Tasks for Turkish Legal AI.

This module provides background tasks for audit operations:
- Scheduled audit log archiving
- Expired data cleanup
- Report generation
- Batch processing
- Retention policy enforcement

Tasks are scheduled via Celery Beat:
    - archive_audit_logs: Daily at 02:00 UTC
    - cleanup_expired_logs: Weekly on Sunday at 03:00 UTC
    - generate_compliance_report: Monthly on 1st at 04:00 UTC

Example:
    >>> # Manual task execution
    >>> from backend.core.queue.tasks.audit import archive_audit_logs
    >>>
    >>> # Trigger archiving for specific tenant
    >>> result = archive_audit_logs.delay(tenant_id=str(tenant_id))
    >>>
    >>> # Get result
    >>> result.get(timeout=300)  # 5 minute timeout
"""

import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from celery import shared_task
from celery.utils.log import get_task_logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.audit_retention_policy import DataCategory
from backend.core.database.session import get_db
from backend.services.advanced_audit_service import AdvancedAuditService
from backend.services.audit_archiver import AuditArchiver

# =============================================================================
# LOGGER & METRICS
# =============================================================================

logger = get_task_logger(__name__)

# Prometheus metrics (lazy import to avoid circular dependency)
def _get_metrics():
    """Get Prometheus metrics client (lazy)."""
    try:
        from prometheus_client import Counter

        # Task failure metrics
        audit_task_failure_total = Counter(
            'audit_task_failure_total',
            'Total audit task failures',
            ['task_name', 'failure_reason', 'tenant_id']
        )
        audit_task_retry_total = Counter(
            'audit_task_retry_total',
            'Total audit task retries',
            ['task_name', 'retry_count', 'tenant_id']
        )
        audit_task_success_total = Counter(
            'audit_task_success_total',
            'Total successful audit task completions',
            ['task_name', 'tenant_id']
        )

        return {
            'failure': audit_task_failure_total,
            'retry': audit_task_retry_total,
            'success': audit_task_success_total,
        }
    except Exception:
        # Metrics not available (testing environment)
        return None


# =============================================================================
# ARCHIVING TASKS
# =============================================================================


@shared_task(
    name="audit.archive_logs",
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
)
def archive_audit_logs(
    self,
    tenant_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Archive audit logs based on retention policies.

    This task moves old audit logs to cold storage (S3 Glacier, etc.)
    based on tenant retention policies.

    Schedule: Daily at 02:00 UTC

    Args:
        tenant_id: Tenant ID (optional, archives all if not specified)
        dry_run: If True, only preview (don't actually archive)

    Returns:
        Archive summary with counts

    Example:
        >>> # Archive all tenants
        >>> archive_audit_logs.delay()
        >>>
        >>> # Archive specific tenant
        >>> archive_audit_logs.delay(tenant_id="uuid-here")
    """
    logger.info(
        f"Starting audit log archiving (dry_run={dry_run})",
        extra={"tenant_id": tenant_id, "dry_run": dry_run},
    )

    metrics = _get_metrics()
    task_name = "archive_logs"
    tenant_label = tenant_id or "all_tenants"

    try:
        # Run async function in sync context
        import asyncio

        result = asyncio.run(
            _archive_logs_async(
                tenant_id=UUID(tenant_id) if tenant_id else None,
                dry_run=dry_run,
            )
        )

        logger.info(
            f"Audit log archiving completed: {result['total_archived']} logs archived",
            extra=result,
        )

        # Track success metric
        if metrics:
            metrics['success'].labels(
                task_name=task_name,
                tenant_id=tenant_label
            ).inc()

        return result

    except Exception as exc:
        logger.error(
            f"Audit log archiving failed: {exc}",
            extra={"error": str(exc), "tenant_id": tenant_id},
        )

        # Track failure metric
        if metrics:
            metrics['failure'].labels(
                task_name=task_name,
                failure_reason=type(exc).__name__,
                tenant_id=tenant_label
            ).inc()

        # Track retry metric
        retry_num = self.request.retries + 1
        if metrics and retry_num <= self.max_retries:
            metrics['retry'].labels(
                task_name=task_name,
                retry_count=str(retry_num),
                tenant_id=tenant_label
            ).inc()

        # Retry with exponential backoff
        raise self.retry(exc=exc)


async def _archive_logs_async(
    tenant_id: Optional[UUID],
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Async helper for archive_audit_logs task.

    Args:
        tenant_id: Tenant ID or None for all tenants
        dry_run: Preview mode

    Returns:
        Archive summary
    """
    async for db in get_db():
        try:
            archiver = AuditArchiver(db)

            if tenant_id:
                # Archive single tenant
                compliance_result = await archiver.archive_compliance_logs(
                    tenant_id=tenant_id,
                    dry_run=dry_run,
                )
                document_result = await archiver.archive_document_logs(
                    tenant_id=tenant_id,
                    dry_run=dry_run,
                )

                return {
                    "tenant_id": str(tenant_id),
                    "total_archived": compliance_result.get("archived_count", 0)
                    + document_result.get("archived_count", 0),
                    "compliance": compliance_result,
                    "documents": document_result,
                    "dry_run": dry_run,
                }
            else:
                # Archive all tenants
                result = await archiver.archive_all_tenants(dry_run=dry_run)
                return result

        finally:
            break  # Exit after first session


# =============================================================================
# CLEANUP TASKS
# =============================================================================


@shared_task(
    name="audit.cleanup_expired_logs",
    bind=True,
    max_retries=2,
    default_retry_delay=600,  # 10 minutes
)
def cleanup_expired_logs(
    self,
    tenant_id: Optional[str] = None,
    dry_run: bool = True,  # Default to dry_run for safety
) -> Dict[str, Any]:
    """
    Delete expired audit logs based on retention policies.

    ⚠️  CAUTION: This performs HARD DELETE (irreversible).
    Always use dry_run=True first!

    Schedule: Weekly on Sunday at 03:00 UTC

    Args:
        tenant_id: Tenant ID (optional, cleans all if not specified)
        dry_run: If True, only preview (default: True for safety)

    Returns:
        Deletion summary

    Example:
        >>> # Dry run (preview)
        >>> cleanup_expired_logs.delay(dry_run=True)
        >>>
        >>> # Actual deletion (DANGEROUS!)
        >>> cleanup_expired_logs.delay(dry_run=False)
    """
    logger.warning(
        f"Starting expired log cleanup (dry_run={dry_run})",
        extra={"tenant_id": tenant_id, "dry_run": dry_run},
    )

    metrics = _get_metrics()
    task_name = "cleanup_logs"
    tenant_label = tenant_id or "all_tenants"

    try:
        import asyncio

        result = asyncio.run(
            _cleanup_logs_async(
                tenant_id=UUID(tenant_id) if tenant_id else None,
                dry_run=dry_run,
            )
        )

        if not dry_run:
            logger.warning(
                f"DELETED {result['total_deleted']} expired logs",
                extra=result,
            )
        else:
            logger.info(
                f"Would delete {result.get('total_deleted', 0)} expired logs",
                extra=result,
            )

        # Track success metric
        if metrics:
            metrics['success'].labels(
                task_name=task_name,
                tenant_id=tenant_label
            ).inc()

        return result

    except Exception as exc:
        logger.error(
            f"Log cleanup failed: {exc}",
            extra={"error": str(exc), "tenant_id": tenant_id},
        )

        # Track failure & retry metrics
        if metrics:
            metrics['failure'].labels(
                task_name=task_name,
                failure_reason=type(exc).__name__,
                tenant_id=tenant_label
            ).inc()

            retry_num = self.request.retries + 1
            if retry_num <= self.max_retries:
                metrics['retry'].labels(
                    task_name=task_name,
                    retry_count=str(retry_num),
                    tenant_id=tenant_label
                ).inc()

        raise self.retry(exc=exc)


async def _cleanup_logs_async(
    tenant_id: Optional[UUID],
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Async helper for cleanup_expired_logs task.

    Args:
        tenant_id: Tenant ID or None for all tenants
        dry_run: Preview mode

    Returns:
        Deletion summary
    """
    async for db in get_db():
        try:
            archiver = AuditArchiver(db)

            if tenant_id:
                # Cleanup single tenant
                result = await archiver.delete_expired_logs(
                    tenant_id=tenant_id,
                    data_category=DataCategory.AUDIT_LOG,
                    dry_run=dry_run,
                )
                return {
                    "tenant_id": str(tenant_id),
                    "total_deleted": result.get("deleted_count", 0),
                    "result": result,
                    "dry_run": dry_run,
                }
            else:
                # Cleanup all tenants
                result = await archiver.cleanup_all_tenants(dry_run=dry_run)
                return result

        finally:
            break


# =============================================================================
# REPORT GENERATION TASKS
# =============================================================================


@shared_task(
    name="audit.generate_compliance_report",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def generate_compliance_report(
    self,
    tenant_id: str,
    start_date: str,  # ISO 8601 format
    end_date: str,  # ISO 8601 format
    compliance_framework: str = "GDPR",
) -> Dict[str, Any]:
    """
    Generate compliance audit report.

    This task generates a comprehensive compliance report
    for the specified time period.

    Schedule: Monthly on 1st at 04:00 UTC (automatic)
    Can also be triggered manually via API.

    Args:
        tenant_id: Tenant ID
        start_date: Start date (ISO 8601, e.g., "2025-01-01T00:00:00Z")
        end_date: End date (ISO 8601)
        compliance_framework: Framework (GDPR, KVKK, etc.)

    Returns:
        Report generation summary

    Example:
        >>> generate_compliance_report.delay(
        ...     tenant_id="uuid-here",
        ...     start_date="2025-01-01T00:00:00Z",
        ...     end_date="2025-01-31T23:59:59Z",
        ...     compliance_framework="GDPR"
        ... )
    """
    logger.info(
        f"Generating compliance report for {tenant_id}",
        extra={
            "tenant_id": tenant_id,
            "start_date": start_date,
            "end_date": end_date,
            "framework": compliance_framework,
        },
    )

    metrics = _get_metrics()
    task_name = "generate_report"

    try:
        import asyncio

        result = asyncio.run(
            _generate_report_async(
                tenant_id=UUID(tenant_id),
                start_date=datetime.datetime.fromisoformat(start_date.replace("Z", "+00:00")),
                end_date=datetime.datetime.fromisoformat(end_date.replace("Z", "+00:00")),
                compliance_framework=compliance_framework,
            )
        )

        logger.info(
            f"Compliance report generated: {result['total_events']} events",
            extra=result,
        )

        # Track success metric
        if metrics:
            metrics['success'].labels(
                task_name=task_name,
                tenant_id=tenant_id
            ).inc()

        return result

    except Exception as exc:
        logger.error(
            f"Report generation failed: {exc}",
            extra={"error": str(exc), "tenant_id": tenant_id},
        )

        # Track failure & retry metrics
        if metrics:
            metrics['failure'].labels(
                task_name=task_name,
                failure_reason=type(exc).__name__,
                tenant_id=tenant_id
            ).inc()

            retry_num = self.request.retries + 1
            if retry_num <= self.max_retries:
                metrics['retry'].labels(
                    task_name=task_name,
                    retry_count=str(retry_num),
                    tenant_id=tenant_id
                ).inc()

        raise self.retry(exc=exc)


async def _generate_report_async(
    tenant_id: UUID,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    compliance_framework: str,
) -> Dict[str, Any]:
    """
    Async helper for generate_compliance_report task.

    Args:
        tenant_id: Tenant ID
        start_date: Start date
        end_date: End date
        compliance_framework: Framework

    Returns:
        Report data
    """
    async for db in get_db():
        try:
            service = AdvancedAuditService(db)

            # Get statistics
            stats = await service.get_compliance_statistics(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
            )

            # Export JSON
            json_data = await service.export_compliance_logs_json(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
            )

            # TODO: Save to S3 or file system
            # For now, just return stats

            return {
                "tenant_id": str(tenant_id),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "compliance_framework": compliance_framework,
                "total_events": stats["total_events"],
                "events_by_type": stats["events_by_type"],
                "unique_data_subjects": stats["unique_data_subjects"],
                "json_size_bytes": len(json_data),
            }

        finally:
            break


# =============================================================================
# BATCH PROCESSING TASKS
# =============================================================================


@shared_task(
    name="audit.process_batch_logs",
    bind=True,
)
def process_batch_logs(
    self,
    log_data: List[Dict[str, Any]],
    tenant_id: str,
) -> Dict[str, Any]:
    """
    Process batch of audit logs (bulk insert).

    This task is useful for importing historical audit data
    or bulk processing from external systems.

    Args:
        log_data: List of log entries (dicts)
        tenant_id: Tenant ID

    Returns:
        Processing summary

    Example:
        >>> logs = [
        ...     {
        ...         "event_type": "DATA_ACCESS",
        ...         "compliance_framework": "GDPR",
        ...         "data_subject_id": "user-uuid",
        ...         # ... other fields
        ...     },
        ...     # ... more logs
        ... ]
        >>> process_batch_logs.delay(log_data=logs, tenant_id="tenant-uuid")
    """
    logger.info(
        f"Processing batch of {len(log_data)} audit logs",
        extra={"tenant_id": tenant_id, "count": len(log_data)},
    )

    metrics = _get_metrics()
    task_name = "process_batch"

    try:
        import asyncio

        result = asyncio.run(
            _process_batch_async(
                log_data=log_data,
                tenant_id=UUID(tenant_id),
            )
        )

        logger.info(
            f"Batch processed: {result['processed_count']} logs",
            extra=result,
        )

        # Track success metric
        if metrics:
            metrics['success'].labels(
                task_name=task_name,
                tenant_id=tenant_id
            ).inc()

        return result

    except Exception as exc:
        logger.error(
            f"Batch processing failed: {exc}",
            extra={"error": str(exc), "tenant_id": tenant_id},
        )

        # Track failure metric
        if metrics:
            metrics['failure'].labels(
                task_name=task_name,
                failure_reason=type(exc).__name__,
                tenant_id=tenant_id
            ).inc()

        raise


async def _process_batch_async(
    log_data: List[Dict[str, Any]],
    tenant_id: UUID,
) -> Dict[str, Any]:
    """
    Async helper for process_batch_logs task.

    Args:
        log_data: Log entries
        tenant_id: Tenant ID

    Returns:
        Processing summary
    """
    from backend.core.database.models.compliance_audit_log import ComplianceEventType
    from backend.core.database.repositories.audit import AuditRepository

    async for db in get_db():
        try:
            repo = AuditRepository(db)

            # Create logs
            logs = []
            for data in log_data:
                from backend.core.database.models.compliance_audit_log import (
                    ComplianceAuditLog,
                )

                log = ComplianceAuditLog(
                    tenant_id=tenant_id,
                    event_type=ComplianceEventType(data["event_type"]),
                    compliance_framework=data["compliance_framework"],
                    data_subject_id=UUID(data["data_subject_id"])
                    if data.get("data_subject_id")
                    else None,
                    # ... other fields
                    metadata=data.get("metadata", {}),
                )
                logs.append(log)

            # Bulk insert
            await repo.bulk_create_compliance_logs(logs)
            await db.commit()

            return {
                "tenant_id": str(tenant_id),
                "processed_count": len(logs),
                "success": True,
            }

        except Exception as e:
            await db.rollback()
            raise

        finally:
            break


# =============================================================================
# SCHEDULED TASK CONFIGURATION
# =============================================================================

# Celery Beat schedule (add to celery.py config)
AUDIT_TASK_SCHEDULE = {
    "archive-audit-logs-daily": {
        "task": "audit.archive_logs",
        "schedule": "crontab(hour=2, minute=0)",  # 02:00 UTC daily
        "kwargs": {"dry_run": False},
    },
    "cleanup-expired-logs-weekly": {
        "task": "audit.cleanup_expired_logs",
        "schedule": "crontab(day_of_week=0, hour=3, minute=0)",  # Sunday 03:00 UTC
        "kwargs": {"dry_run": False},  # Change to False after testing
    },
    "generate-monthly-compliance-reports": {
        "task": "audit.generate_compliance_report",
        "schedule": "crontab(day_of_month=1, hour=4, minute=0)",  # 1st of month, 04:00 UTC
        # Note: This would need tenant_id parameter, so it's commented out
        # In production, you'd generate this dynamically for each tenant
    },
}
