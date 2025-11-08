"""
Audit API Routes for Turkish Legal AI.

This module provides REST API endpoints for audit operations:
- Query compliance audit logs
- Query document audit logs
- Get audit statistics
- Export audit logs (JSON, CSV)
- Legal hold management
- Retention policy operations

All endpoints require authentication and appropriate permissions.

Example:
    >>> # Query compliance logs
    >>> GET /api/v1/audit/compliance?start_date=2025-01-01&event_type=DATA_ACCESS
    >>>
    >>> # Export document logs
    >>> GET /api/v1/audit/documents/export/csv?document_id=123...
    >>>
    >>> # Place legal hold
    >>> POST /api/v1/audit/legal-hold
    >>> {
    ...     "document_ids": ["doc1", "doc2"],
    ...     "reason": "Litigation - Case #2025-001"
    ... }
"""

import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.models.audit_retention_policy import DataCategory
from backend.core.database.models.compliance_audit_log import ComplianceEventType
from backend.core.database.models.document_audit_log import DocumentEventType
from backend.core.database.session import get_db
from backend.security.rbac.context import get_current_tenant_id, get_current_user_id
from backend.security.rbac.decorators import require_permission
from backend.services.advanced_audit_service import AdvancedAuditService
from backend.services.audit_archiver import AuditArchiver

# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/audit",
    tags=["audit"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ComplianceLogResponse(BaseModel):
    """Compliance audit log response."""

    id: UUID
    tenant_id: UUID
    data_subject_id: Optional[UUID]
    event_type: str
    compliance_framework: str
    legal_basis: Optional[str]
    processing_purpose: Optional[str]
    ip_address: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class DocumentLogResponse(BaseModel):
    """Document audit log response."""

    id: UUID
    tenant_id: UUID
    document_id: UUID
    document_title: Optional[str]
    user_id: Optional[UUID]
    event_type: str
    version_number: Optional[int]
    ip_address: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Audit statistics response."""

    total_events: int
    events_by_type: dict
    events_by_framework: Optional[dict] = None
    unique_data_subjects: Optional[int] = None
    unique_documents: Optional[int] = None
    unique_users: Optional[int] = None


class LegalHoldRequest(BaseModel):
    """Legal hold request."""

    document_ids: List[UUID] = Field(..., description="Document IDs to lock")
    reason: str = Field(..., description="Legal hold reason")


class LegalHoldResponse(BaseModel):
    """Legal hold response."""

    locked_count: int
    document_ids: List[UUID]


class RetentionPolicyResponse(BaseModel):
    """Retention policy response."""

    id: UUID
    name: str
    retention_days: int
    data_category: str
    compliance_framework: str
    hot_tier_days: int
    warm_tier_days: int
    cold_tier_days: int


# =============================================================================
# COMPLIANCE AUDIT LOG ENDPOINTS
# =============================================================================


@router.get(
    "/compliance",
    response_model=List[ComplianceLogResponse],
    summary="Query compliance audit logs",
    description="Query compliance audit logs with filters (RBAC: audit:read required)",
)
@require_permission("audit", "read")
async def query_compliance_logs(
    start_date: Optional[datetime.datetime] = Query(
        None, description="Start date (ISO 8601)"
    ),
    end_date: Optional[datetime.datetime] = Query(
        None, description="End date (ISO 8601)"
    ),
    event_type: Optional[ComplianceEventType] = Query(
        None, description="Filter by event type"
    ),
    compliance_framework: Optional[str] = Query(
        None, description="Filter by framework (GDPR, KVKK, etc.)"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Results limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> List[ComplianceLogResponse]:
    """
    Query compliance audit logs.

    **Permissions**: Requires `audit:read` permission.

    **Query Parameters**:
    - start_date: Start date (inclusive)
    - end_date: End date (inclusive)
    - event_type: Filter by event type
    - compliance_framework: Filter by framework
    - limit: Maximum results (1-1000)
    - offset: Pagination offset

    **Returns**: List of compliance audit log records.
    """
    service = AdvancedAuditService(db)

    event_types = [event_type] if event_type else None

    logs = await service.query_compliance_logs(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        compliance_framework=compliance_framework,
        limit=limit,
        offset=offset,
    )

    return [ComplianceLogResponse.from_orm(log) for log in logs]


@router.get(
    "/compliance/statistics",
    response_model=StatisticsResponse,
    summary="Get compliance audit statistics",
    description="Get aggregated compliance audit statistics",
)
@require_permission("audit", "read")
async def get_compliance_statistics(
    start_date: Optional[datetime.datetime] = Query(None),
    end_date: Optional[datetime.datetime] = Query(None),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> StatisticsResponse:
    """
    Get compliance audit statistics.

    **Permissions**: Requires `audit:read` permission.

    **Returns**: Aggregated statistics (total events, events by type, etc.).
    """
    service = AdvancedAuditService(db)

    stats = await service.get_compliance_statistics(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
    )

    return StatisticsResponse(**stats)


@router.get(
    "/compliance/export/json",
    summary="Export compliance logs as JSON",
    description="Export compliance audit logs as JSON file",
)
@require_permission("audit", "export")
async def export_compliance_logs_json(
    start_date: Optional[datetime.datetime] = Query(None),
    end_date: Optional[datetime.datetime] = Query(None),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Export compliance logs as JSON.

    **Permissions**: Requires `audit:export` permission.

    **Returns**: JSON file download.
    """
    service = AdvancedAuditService(db)

    json_data = await service.export_compliance_logs_json(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
    )

    # Return as downloadable file
    return Response(
        content=json_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=compliance_audit_{tenant_id}.json"
        },
    )


# =============================================================================
# DOCUMENT AUDIT LOG ENDPOINTS
# =============================================================================


@router.get(
    "/documents",
    response_model=List[DocumentLogResponse],
    summary="Query document audit logs",
    description="Query document audit logs with filters",
)
@require_permission("audit", "read")
async def query_document_logs(
    document_id: Optional[UUID] = Query(None, description="Filter by document ID"),
    user_id: Optional[UUID] = Query(None, description="Filter by user ID"),
    start_date: Optional[datetime.datetime] = Query(None),
    end_date: Optional[datetime.datetime] = Query(None),
    event_type: Optional[DocumentEventType] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> List[DocumentLogResponse]:
    """
    Query document audit logs.

    **Permissions**: Requires `audit:read` permission.

    **Returns**: List of document audit log records.
    """
    service = AdvancedAuditService(db)

    event_types = [event_type] if event_type else None

    logs = await service.query_document_logs(
        tenant_id=tenant_id,
        document_id=document_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        limit=limit,
        offset=offset,
    )

    return [DocumentLogResponse.from_orm(log) for log in logs]


@router.get(
    "/documents/statistics",
    response_model=StatisticsResponse,
    summary="Get document audit statistics",
    description="Get aggregated document audit statistics",
)
@require_permission("audit", "read")
async def get_document_statistics(
    start_date: Optional[datetime.datetime] = Query(None),
    end_date: Optional[datetime.datetime] = Query(None),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> StatisticsResponse:
    """
    Get document audit statistics.

    **Permissions**: Requires `audit:read` permission.
    """
    service = AdvancedAuditService(db)

    stats = await service.get_document_statistics(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
    )

    return StatisticsResponse(**stats)


@router.get(
    "/documents/export/csv",
    summary="Export document logs as CSV",
    description="Export document audit logs as CSV file",
)
@require_permission("audit", "export")
async def export_document_logs_csv(
    document_id: Optional[UUID] = Query(None),
    start_date: Optional[datetime.datetime] = Query(None),
    end_date: Optional[datetime.datetime] = Query(None),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Export document logs as CSV.

    **Permissions**: Requires `audit:export` permission.

    **Returns**: CSV file download.
    """
    service = AdvancedAuditService(db)

    csv_data = await service.export_document_logs_csv(
        tenant_id=tenant_id,
        document_id=document_id,
        start_date=start_date,
        end_date=end_date,
    )

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=document_audit_{tenant_id}.csv"
        },
    )


# =============================================================================
# LEGAL HOLD ENDPOINTS
# =============================================================================


@router.post(
    "/legal-hold",
    response_model=LegalHoldResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Place legal hold on documents",
    description="Place legal hold (prevents deletion during litigation/investigation)",
)
@require_permission("audit", "legal_hold")
async def place_legal_hold(
    request: LegalHoldRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> LegalHoldResponse:
    """
    Place legal hold on documents.

    **Permissions**: Requires `audit:legal_hold` permission.

    Legal holds prevent documents from being deleted during
    litigation, investigations, or regulatory audits.

    **Request Body**:
    ```json
    {
        "document_ids": ["uuid1", "uuid2"],
        "reason": "Litigation - Case #2025-001"
    }
    ```

    **Returns**: Number of documents locked.
    """
    service = AdvancedAuditService(db)

    locked_count = await service.place_legal_hold(
        tenant_id=tenant_id,
        document_ids=request.document_ids,
        reason=request.reason,
        placed_by_id=user_id,
    )

    return LegalHoldResponse(
        locked_count=locked_count,
        document_ids=request.document_ids,
    )


@router.delete(
    "/legal-hold",
    response_model=LegalHoldResponse,
    summary="Remove legal hold from documents",
    description="Remove legal hold (allows deletion again)",
)
@require_permission("audit", "legal_hold")
async def remove_legal_hold(
    document_ids: List[UUID] = Query(..., description="Document IDs to unlock"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> LegalHoldResponse:
    """
    Remove legal hold from documents.

    **Permissions**: Requires `audit:legal_hold` permission.

    **Query Parameters**:
    - document_ids: List of document UUIDs to unlock

    **Returns**: Number of documents unlocked.
    """
    service = AdvancedAuditService(db)

    unlocked_count = await service.remove_legal_hold(
        tenant_id=tenant_id,
        document_ids=document_ids,
        removed_by_id=user_id,
    )

    return LegalHoldResponse(
        locked_count=unlocked_count,
        document_ids=document_ids,
    )


# =============================================================================
# RETENTION POLICY ENDPOINTS
# =============================================================================


@router.get(
    "/retention-policy/{data_category}",
    response_model=RetentionPolicyResponse,
    summary="Get retention policy for data category",
    description="Get active retention policy for tenant and data category",
)
@require_permission("audit", "read")
async def get_retention_policy(
    data_category: DataCategory,
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> RetentionPolicyResponse:
    """
    Get retention policy.

    **Permissions**: Requires `audit:read` permission.

    **Path Parameters**:
    - data_category: Data category (audit_log, document_data, etc.)

    **Returns**: Active retention policy details.
    """
    service = AdvancedAuditService(db)

    policy = await service.get_retention_policy(
        tenant_id=tenant_id,
        data_category=data_category,
    )

    if not policy:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No retention policy found for {data_category}",
        )

    return RetentionPolicyResponse(
        id=policy.id,
        name=policy.name,
        retention_days=policy.retention_days,
        data_category=str(policy.data_category),
        compliance_framework=str(policy.compliance_framework),
        hot_tier_days=policy.hot_tier_days,
        warm_tier_days=policy.warm_tier_days,
        cold_tier_days=policy.cold_tier_days,
    )


@router.post(
    "/archive",
    summary="Archive audit logs (admin only)",
    description="Trigger audit log archiving based on retention policy",
)
@require_permission("audit", "archive")
async def archive_logs(
    dry_run: bool = Query(True, description="Dry run (don't actually archive)"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Archive audit logs.

    **Permissions**: Requires `audit:archive` permission (admin only).

    **CAUTION**: Use dry_run=True first to preview what will be archived.

    **Query Parameters**:
    - dry_run: If true, only preview (default: true)

    **Returns**: Archive summary (counts by tier).
    """
    archiver = AuditArchiver(db)

    # Archive compliance logs
    compliance_result = await archiver.archive_compliance_logs(
        tenant_id=tenant_id,
        dry_run=dry_run,
    )

    # Archive document logs
    document_result = await archiver.archive_document_logs(
        tenant_id=tenant_id,
        dry_run=dry_run,
    )

    return {
        "dry_run": dry_run,
        "compliance": compliance_result,
        "documents": document_result,
    }


@router.delete(
    "/expired",
    summary="Delete expired audit logs (admin only)",
    description="Delete expired audit logs based on retention policy",
)
@require_permission("audit", "delete")
async def delete_expired_logs(
    data_category: DataCategory = Query(..., description="Data category to clean up"),
    dry_run: bool = Query(True, description="Dry run (don't actually delete)"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Delete expired audit logs.

    **Permissions**: Requires `audit:delete` permission (superadmin only).

    **⚠️ DANGER**: This performs HARD DELETE (irreversible).
    Always use dry_run=True first!

    **Query Parameters**:
    - data_category: Data category to clean up
    - dry_run: If true, only preview (default: true)

    **Returns**: Deletion summary.
    """
    archiver = AuditArchiver(db)

    result = await archiver.delete_expired_logs(
        tenant_id=tenant_id,
        data_category=data_category,
        dry_run=dry_run,
    )

    return result
