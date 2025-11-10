"""
Bulk Review API Routes for Harvey/Legora Turkish Legal AI Platform.

This module provides REST API endpoints for bulk document review and processing:
- Bulk Job Management: Create, monitor, control high-volume document processing
- Processing Strategies: FAST, BALANCED, THOROUGH, CUSTOM workflows
- Progress Tracking: Real-time status, progress percentage, ETA
- Quality Control: Confidence scoring, manual review queue
- Manual Review: Approve/reject documents requiring human review
- Analytics: Job statistics, performance metrics, success rates

Bulk review is designed for high-volume document processing scenarios:
- Due diligence (M&A, 100s-1000s of documents)
- Contract portfolio analysis
- Legal discovery (e-discovery)
- Compliance audits
- Case file digitization

Example Usage:
    >>> # Create bulk job for 500 contracts
    >>> POST /api/v1/bulk-review/jobs
    >>> {
    ...     "name": "Q4 2025 Contract Review",
    ...     "document_ids": [...500 UUIDs...],
    ...     "strategy": "balanced",
    ...     "concurrency": 20,
    ...     "workflow": "contract_review"
    ... }
    >>>
    >>> # Monitor job progress
    >>> GET /api/v1/bulk-review/jobs/{job_id}
    >>> # Response: {"progress": 45.2, "status": "processing", "eta": "15 minutes"}
    >>>
    >>> # Review low-confidence document
    >>> POST /api/v1/bulk-review/jobs/{job_id}/documents/{doc_id}/approve
    >>> {"notes": "Verified manually - contract is valid"}

Performance:
    - 100 documents: 5-10 minutes (FAST strategy)
    - 1000 documents: 30-60 minutes (FAST strategy)
    - Configurable concurrency: 1-50 parallel workers
    - Progress updates every 10 documents
    - Real-time status via polling or WebSocket

Security:
    - Requires 'bulk_review:write' permission to create jobs
    - Requires 'bulk_review:read' permission to view status
    - Requires 'bulk_review:review' permission to approve/reject documents
    - All operations are tenant-isolated and audit-logged

Author: Harvey/Legora CTO
Date: 2025-11-10
Lines: 650+
"""

import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.session import get_db
from backend.core.exceptions import BulkProcessingError, ValidationError
from backend.security.rbac.context import get_current_tenant_id, get_current_user_id
from backend.security.rbac.decorators import require_permission
from backend.services.bulk_review_orchestrator import (
    BulkReviewOrchestrator,
    ProcessingStrategy,
    JobStatus,
    DocumentStatus,
    Priority,
)

# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter(
    prefix="/bulk-review",
    tags=["bulk-review"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class CreateBulkJobRequest(BaseModel):
    """Create bulk job request."""

    name: str = Field(..., description="Job name", min_length=3, max_length=255)
    document_ids: List[str] = Field(..., description="Document UUIDs to process", min_items=1)
    strategy: str = Field(
        "balanced",
        description="Processing strategy (fast, balanced, thorough, custom)"
    )
    workflow: Optional[str] = Field(None, description="Workflow template name")
    concurrency: int = Field(10, ge=1, le=50, description="Parallel workers (1-50)")
    tags: Optional[List[str]] = Field(None, description="Job tags")

    @validator('strategy')
    def validate_strategy(cls, v):
        """Validate strategy."""
        valid_strategies = ['fast', 'balanced', 'thorough', 'custom']
        if v.lower() not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {", ".join(valid_strategies)}')
        return v.lower()

    @validator('document_ids')
    def validate_document_ids(cls, v):
        """Validate document IDs."""
        if len(v) > 10000:
            raise ValueError('Maximum 10,000 documents per job')
        return v


class BulkJobResponse(BaseModel):
    """Bulk job response."""

    job_id: str
    name: str
    status: str
    strategy: str
    progress: float
    total_documents: int
    processed_documents: int
    failed_documents: int
    needs_review_count: int
    avg_confidence: Optional[float] = None
    estimated_time_remaining: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    analytics: Dict[str, Any] = {}


class BulkJobResultsResponse(BaseModel):
    """Bulk job results response."""

    job_id: str
    name: str
    status: str
    progress: float
    total_documents: int
    completed: int
    failed: int
    needs_review: int
    avg_confidence: float
    estimated_time_remaining: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    analytics: Dict[str, Any]
    failed_tasks: Optional[List[Dict[str, Any]]] = None


class ApproveDocumentRequest(BaseModel):
    """Approve document request."""

    notes: Optional[str] = Field(None, description="Approval notes", max_length=1000)


class RejectDocumentRequest(BaseModel):
    """Reject document request."""

    reason: str = Field(..., description="Rejection reason", min_length=10, max_length=1000)


class JobControlResponse(BaseModel):
    """Job control response (pause, resume, cancel)."""

    message: str
    job_id: str
    status: str


class DocumentReviewResponse(BaseModel):
    """Document review response (approve/reject)."""

    message: str
    job_id: str
    document_id: str
    status: str


# =============================================================================
# BULK JOB MANAGEMENT ENDPOINTS
# =============================================================================


@router.post(
    "/jobs",
    response_model=BulkJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create bulk processing job",
    description="Create a new bulk document processing job (requires bulk_review:write)",
)
@require_permission("bulk_review", "write")
async def create_bulk_job(
    request: CreateBulkJobRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> BulkJobResponse:
    """
    Create a new bulk processing job.

    **Permissions**: Requires 'bulk_review:write' permission.

    **Request Body**:
    ```json
    {
        "name": "Q4 2025 Contract Review",
        "document_ids": ["uuid1", "uuid2", ...],
        "strategy": "balanced",
        "workflow": "contract_review",
        "concurrency": 20,
        "tags": ["contracts", "q4-2025"]
    }
    ```

    **Processing Strategies**:
    - **fast**: Parallel processing with auto-approval (best for simple documents)
    - **balanced**: Mixed auto/manual review (default, recommended)
    - **thorough**: Full manual review for high accuracy
    - **custom**: User-defined workflow

    **Returns**: Created job with job_id for tracking.

    **Performance**:
    - 100 docs: ~5-10 min (fast)
    - 1000 docs: ~30-60 min (fast)
    - Configurable concurrency: 1-50 workers
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)

        # Parse document IDs
        document_ids = [UUID(doc_id) for doc_id in request.document_ids]

        # Parse strategy
        strategy = ProcessingStrategy(request.strategy)

        # Create job
        job = await orchestrator.create_bulk_job(
            name=request.name,
            document_ids=document_ids,
            strategy=strategy,
            workflow=request.workflow,
            concurrency=request.concurrency,
            user_id=user_id,
            tenant_id=tenant_id,
            tags=request.tags,
        )

        return BulkJobResponse(
            job_id=str(job.id),
            name=job.name,
            status=job.status.value,
            strategy=job.strategy.value,
            progress=job.progress(),
            total_documents=job.total_documents,
            processed_documents=job.processed_documents,
            failed_documents=job.failed_documents,
            needs_review_count=job.needs_review_count,
            avg_confidence=job.avg_confidence() if job.processed_documents > 0 else None,
            estimated_time_remaining=None,  # Not started yet
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=None,
            analytics=job.analytics,
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except BulkProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create bulk job: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get(
    "/jobs/{job_id}",
    response_model=BulkJobResponse,
    summary="Get bulk job status",
    description="Get real-time status of a bulk processing job (requires bulk_review:read)",
)
@require_permission("bulk_review", "read")
async def get_bulk_job_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> BulkJobResponse:
    """
    Get bulk job status and progress.

    **Permissions**: Requires 'bulk_review:read' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID

    **Returns**: Job status with real-time progress, ETA, and analytics.

    **Status Values**:
    - pending: Job queued, not started yet
    - processing: Job is actively processing documents
    - review: Processing complete, awaiting manual review
    - completed: All processing and reviews complete
    - failed: Job failed (check error details)
    - cancelled: Job was cancelled by user
    - paused: Job is paused (can be resumed)

    **Example Response**:
    ```json
    {
        "job_id": "uuid",
        "status": "processing",
        "progress": 45.2,
        "total_documents": 500,
        "processed_documents": 226,
        "estimated_time_remaining": "12 minutes",
        "avg_confidence": 0.87
    }
    ```
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        job = await orchestrator.get_job_status(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        # Verify tenant access
        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        eta = job.estimated_time_remaining()

        return BulkJobResponse(
            job_id=str(job.id),
            name=job.name,
            status=job.status.value,
            strategy=job.strategy.value,
            progress=job.progress(),
            total_documents=job.total_documents,
            processed_documents=job.processed_documents,
            failed_documents=job.failed_documents,
            needs_review_count=job.needs_review_count,
            avg_confidence=job.avg_confidence() if job.processed_documents > 0 else None,
            estimated_time_remaining=str(eta) if eta else None,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            analytics=job.analytics,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}",
        )


@router.get(
    "/jobs/{job_id}/results",
    response_model=BulkJobResultsResponse,
    summary="Get bulk job results",
    description="Get detailed results and analytics for completed job (requires bulk_review:read)",
)
@require_permission("bulk_review", "read")
async def get_bulk_job_results(
    job_id: UUID,
    include_failed: bool = Query(False, description="Include failed task details"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> BulkJobResultsResponse:
    """
    Get bulk job results and detailed analytics.

    **Permissions**: Requires 'bulk_review:read' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID

    **Query Parameters**:
        - include_failed: If true, include details of failed tasks (default: false)

    **Returns**: Comprehensive results with analytics.

    **Analytics Include**:
    - Status distribution (completed, failed, needs_review)
    - Confidence score distribution (high, medium, low)
    - Processing performance (avg time, total time, throughput)
    - Success rate percentage
    - Documents per minute

    **Example Response**:
    ```json
    {
        "job_id": "uuid",
        "status": "completed",
        "completed": 487,
        "failed": 13,
        "needs_review": 0,
        "avg_confidence": 0.89,
        "analytics": {
            "success_rate": 97.4,
            "avg_processing_time_ms": 2340,
            "documents_per_minute": 18.5
        }
    }
    ```
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        
        # Get job to verify tenant access
        job = await orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        # Get results
        results = await orchestrator.get_job_results(
            job_id=job_id,
            include_failed=include_failed,
        )

        return BulkJobResultsResponse(**results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job results: {str(e)}",
        )


# =============================================================================
# JOB CONTROL ENDPOINTS
# =============================================================================


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=JobControlResponse,
    summary="Cancel bulk job",
    description="Cancel a running bulk job (requires bulk_review:write)",
)
@require_permission("bulk_review", "write")
async def cancel_bulk_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> JobControlResponse:
    """
    Cancel a running bulk job.

    **Permissions**: Requires 'bulk_review:write' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID

    **Returns**: Confirmation with updated status.

    **Note**: Documents already processed will remain processed.
    Only pending documents will be cancelled.
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        
        # Verify job exists and tenant access
        job = await orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        # Cancel job
        await orchestrator.cancel_job(job_id)

        return JobControlResponse(
            message="Bulk job cancelled successfully",
            job_id=str(job_id),
            status="cancelled",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}",
        )


@router.post(
    "/jobs/{job_id}/pause",
    response_model=JobControlResponse,
    summary="Pause bulk job",
    description="Pause a running bulk job (requires bulk_review:write)",
)
@require_permission("bulk_review", "write")
async def pause_bulk_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> JobControlResponse:
    """
    Pause a running bulk job.

    **Permissions**: Requires 'bulk_review:write' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID

    **Returns**: Confirmation with updated status.

    **Note**: Job can be resumed later using /resume endpoint.
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        
        job = await orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        await orchestrator.pause_job(job_id)

        return JobControlResponse(
            message="Bulk job paused successfully",
            job_id=str(job_id),
            status="paused",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause job: {str(e)}",
        )


@router.post(
    "/jobs/{job_id}/resume",
    response_model=JobControlResponse,
    summary="Resume bulk job",
    description="Resume a paused bulk job (requires bulk_review:write)",
)
@require_permission("bulk_review", "write")
async def resume_bulk_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> JobControlResponse:
    """
    Resume a paused bulk job.

    **Permissions**: Requires 'bulk_review:write' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID

    **Returns**: Confirmation with updated status.
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        
        job = await orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        if job.status != JobStatus.PAUSED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is not in paused state",
            )

        await orchestrator.resume_job(job_id)

        return JobControlResponse(
            message="Bulk job resumed successfully",
            job_id=str(job_id),
            status="processing",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume job: {str(e)}",
        )


# =============================================================================
# MANUAL REVIEW ENDPOINTS
# =============================================================================


@router.post(
    "/jobs/{job_id}/documents/{document_id}/approve",
    response_model=DocumentReviewResponse,
    summary="Approve document in review",
    description="Approve a document requiring manual review (requires bulk_review:review)",
)
@require_permission("bulk_review", "review")
async def approve_document(
    job_id: UUID,
    document_id: UUID,
    request: ApproveDocumentRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> DocumentReviewResponse:
    """
    Approve a document in manual review queue.

    **Permissions**: Requires 'bulk_review:review' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID
        - document_id: Document UUID to approve

    **Request Body**:
    ```json
    {
        "notes": "Manually verified - contract is valid and complete"
    }
    ```

    **Returns**: Confirmation of approval.

    **Note**: Once all documents in review are approved/rejected,
    the job will automatically transition to completed status.
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        
        # Verify job exists and tenant access
        job = await orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        # Approve document
        await orchestrator.approve_document(
            job_id=job_id,
            document_id=document_id,
            reviewer_id=user_id,
            notes=request.notes,
        )

        return DocumentReviewResponse(
            message="Document approved successfully",
            job_id=str(job_id),
            document_id=str(document_id),
            status="approved",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve document: {str(e)}",
        )


@router.post(
    "/jobs/{job_id}/documents/{document_id}/reject",
    response_model=DocumentReviewResponse,
    summary="Reject document in review",
    description="Reject a document requiring manual review (requires bulk_review:review)",
)
@require_permission("bulk_review", "review")
async def reject_document(
    job_id: UUID,
    document_id: UUID,
    request: RejectDocumentRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> DocumentReviewResponse:
    """
    Reject a document in manual review queue.

    **Permissions**: Requires 'bulk_review:review' permission.

    **Path Parameters**:
        - job_id: Bulk job UUID
        - document_id: Document UUID to reject

    **Request Body**:
    ```json
    {
        "reason": "Document is incomplete - missing signature pages and exhibits"
    }
    ```

    **Returns**: Confirmation of rejection.

    **Note**: Rejected documents will be marked as rejected and excluded
    from final results. Provide a clear reason for rejection.
    """
    try:
        orchestrator = BulkReviewOrchestrator(db_session=db)
        
        # Verify job exists and tenant access
        job = await orchestrator.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bulk job not found: {job_id}",
            )

        if job.tenant_id and job.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job",
            )

        # Reject document
        await orchestrator.reject_document(
            job_id=job_id,
            document_id=document_id,
            reviewer_id=user_id,
            reason=request.reason,
        )

        return DocumentReviewResponse(
            message="Document rejected successfully",
            job_id=str(job_id),
            document_id=str(document_id),
            status="rejected",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reject document: {str(e)}",
        )
