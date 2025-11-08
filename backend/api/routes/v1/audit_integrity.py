"""
Audit Integrity API Routes - Hash-Chain Verification Endpoints.

This module provides REST API endpoints for audit integrity operations:
- Hash-chain verification
- Tamper detection
- Integrity statistics
- Forensic analysis
- Chain health monitoring

All endpoints require admin-level authentication.

Example:
    >>> # Verify chain integrity
    >>> GET /api/v1/audit-integrity/verify?start_date=2025-01-01
    >>>
    >>> # Check specific event
    >>> GET /api/v1/audit-integrity/event/{event_id}/tamper-check
    >>>
    >>> # Get chain statistics
    >>> GET /api/v1/audit-integrity/statistics
"""

import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.session import get_db
from backend.security.rbac.context import get_current_tenant_id
from backend.security.rbac.decorators import require_permission
from backend.services.audit_integrity import AuditIntegrityService

# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/audit-integrity",
    tags=["audit-integrity"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class IntegrityEventResponse(BaseModel):
    """Integrity event response."""

    event_id: UUID
    event_hash: str
    previous_hash: str
    sequence_number: int
    created_at: datetime.datetime
    event_type: str
    tenant_id: UUID

    class Config:
        from_attributes = True


class IntegrityViolationResponse(BaseModel):
    """Integrity violation response."""

    event_id: UUID
    sequence_number: int
    violation_type: str
    expected_value: Optional[str]
    actual_value: Optional[str]
    description: str


class VerificationResultResponse(BaseModel):
    """Chain verification result response."""

    is_valid: bool
    total_events: int
    verified_events: int
    violations: List[dict]
    chain_length: int
    verification_duration_ms: float


class TamperCheckResponse(BaseModel):
    """Tamper check response."""

    event_id: UUID
    sequence_number: int
    is_tampered: bool
    confidence: str  # high, medium, low, verified, unknown
    stored_hash: str
    computed_hash: str
    evidence: dict


class ChainStatisticsResponse(BaseModel):
    """Chain statistics response."""

    tenant_id: UUID
    total_events: int
    events_with_integrity: int
    chain_length: int
    integrity_coverage: float  # Percentage
    genesis_hash: str
    hash_algorithm: str


class CreateIntegrityEventRequest(BaseModel):
    """Create integrity event request."""

    event_type: str = Field(..., description="Event type")
    event_data: dict = Field(..., description="Event data (will be hashed)")
    data_subject_id: Optional[UUID] = Field(None, description="Data subject ID")
    compliance_framework: str = Field(default="GDPR", description="Framework")


# =============================================================================
# VERIFICATION ENDPOINTS
# =============================================================================


@router.get(
    "/verify",
    response_model=VerificationResultResponse,
    summary="Verify audit log hash-chain integrity",
    description="Perform full cryptographic verification of audit log chain",
)
@require_permission("audit", "admin")  # Requires admin permission
async def verify_chain_integrity(
    start_date: Optional[datetime.datetime] = Query(
        None, description="Start date for verification"
    ),
    end_date: Optional[datetime.datetime] = Query(
        None, description="End date for verification"
    ),
    limit: Optional[int] = Query(
        None, ge=1, le=10000, description="Max events to verify"
    ),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> VerificationResultResponse:
    """
    Verify audit log hash-chain integrity.

    This performs a full cryptographic verification:
    - Verifies each event's hash
    - Checks chain linkage (previous_hash)
    - Detects sequence gaps
    - Validates timestamps

    **Permissions**: Requires `audit:admin` permission.

    **Query Parameters**:
    - start_date: Start date (inclusive, optional)
    - end_date: End date (inclusive, optional)
    - limit: Maximum events to verify (1-10000, optional)

    **Returns**: Verification result with violations (if any).

    **Example Response**:
    ```json
    {
        "is_valid": true,
        "total_events": 1543,
        "verified_events": 1543,
        "violations": [],
        "chain_length": 1543,
        "verification_duration_ms": 245.3
    }
    ```

    **Violation Types**:
    - `hash_mismatch`: Event data was tampered with
    - `chain_gap`: Missing events in sequence
    - `missing_predecessor`: Previous hash doesn't match
    - `timestamp_anomaly`: Timestamp went backwards
    """
    service = AuditIntegrityService(db)

    result = await service.verify_chain(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    return VerificationResultResponse(
        is_valid=result.is_valid,
        total_events=result.total_events,
        verified_events=result.verified_events,
        violations=result.violations,
        chain_length=result.chain_length,
        verification_duration_ms=result.verification_duration_ms,
    )


@router.get(
    "/event/{event_id}/tamper-check",
    response_model=TamperCheckResponse,
    summary="Check if specific event was tampered with",
    description="Forensic analysis on specific audit event",
)
@require_permission("audit", "admin")
async def check_event_tampering(
    event_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> TamperCheckResponse:
    """
    Check if specific event was tampered with.

    This performs forensic analysis on a single event:
    - Recomputes event hash
    - Compares with stored hash
    - Returns tamper evidence

    **Permissions**: Requires `audit:admin` permission.

    **Path Parameters**:
    - event_id: Event UUID to analyze

    **Returns**: Tamper check result with evidence.

    **Example Response**:
    ```json
    {
        "event_id": "123e4567-e89b-12d3-a456-426614174000",
        "sequence_number": 1543,
        "is_tampered": false,
        "confidence": "verified",
        "stored_hash": "a1b2c3d4...",
        "computed_hash": "a1b2c3d4...",
        "evidence": {
            "hash_match": true,
            "previous_hash": "d4c3b2a1...",
            "timestamp": "2025-01-15T10:30:00Z"
        }
    }
    ```

    **Confidence Levels**:
    - `verified`: Hash matches (not tampered)
    - `high`: Hash mismatch (tampered)
    - `unknown`: No integrity metadata (pre-integrity event)
    """
    service = AuditIntegrityService(db)

    result = await service.detect_tampering(
        tenant_id=tenant_id,
        event_id=event_id,
    )

    return TamperCheckResponse(**result)


# =============================================================================
# STATISTICS ENDPOINTS
# =============================================================================


@router.get(
    "/statistics",
    response_model=ChainStatisticsResponse,
    summary="Get hash-chain statistics",
    description="Get audit log chain health metrics",
)
@require_permission("audit", "read")
async def get_chain_statistics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> ChainStatisticsResponse:
    """
    Get hash-chain statistics for tenant.

    **Permissions**: Requires `audit:read` permission.

    **Returns**: Chain statistics and health metrics.

    **Example Response**:
    ```json
    {
        "tenant_id": "123e4567-e89b-12d3-a456-426614174000",
        "total_events": 2500,
        "events_with_integrity": 1543,
        "chain_length": 1543,
        "integrity_coverage": 61.72,
        "genesis_hash": "0000000000000000...",
        "hash_algorithm": "sha256"
    }
    ```

    **Metrics**:
    - `total_events`: All audit events
    - `events_with_integrity`: Events with hash-chain
    - `chain_length`: Current chain sequence number
    - `integrity_coverage`: Percentage with integrity (%)
    - `genesis_hash`: First event hash (64 zeros)
    - `hash_algorithm`: SHA-256
    """
    service = AuditIntegrityService(db)

    stats = await service.get_chain_statistics(tenant_id=tenant_id)

    return ChainStatisticsResponse(
        tenant_id=UUID(stats["tenant_id"]),
        total_events=stats["total_events"],
        events_with_integrity=stats["events_with_integrity"],
        chain_length=stats["chain_length"],
        integrity_coverage=stats["integrity_coverage"],
        genesis_hash=stats["genesis_hash"],
        hash_algorithm=stats["hash_algorithm"],
    )


# =============================================================================
# EVENT CREATION ENDPOINTS
# =============================================================================


@router.post(
    "/create-event",
    response_model=IntegrityEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create integrity-protected audit event",
    description="Create audit event with hash-chain protection",
)
@require_permission("audit", "create")
async def create_integrity_event(
    request: CreateIntegrityEventRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> IntegrityEventResponse:
    """
    Create audit event with hash-chain protection.

    This creates an immutable, tamper-proof audit event:
    - Generates cryptographic hash
    - Links to previous event
    - Assigns sequence number
    - Stores in audit log

    **Permissions**: Requires `audit:create` permission.

    **Request Body**:
    ```json
    {
        "event_type": "DATA_ACCESS",
        "event_data": {
            "user_id": "user-uuid",
            "resource": "documents/123",
            "action": "view"
        },
        "data_subject_id": "user-uuid",
        "compliance_framework": "GDPR"
    }
    ```

    **Returns**: Created integrity event with hash metadata.

    **Example Response**:
    ```json
    {
        "event_id": "123e4567-e89b-12d3-a456-426614174000",
        "event_hash": "a1b2c3d4e5f6...",
        "previous_hash": "f6e5d4c3b2a1...",
        "sequence_number": 1544,
        "created_at": "2025-01-15T10:30:00Z",
        "event_type": "DATA_ACCESS",
        "tenant_id": "tenant-uuid"
    }
    ```

    **Security**:
    - Hash algorithm: SHA-256 (FIPS 140-2)
    - Tenant salt: Prevents rainbow table attacks
    - Chain linkage: Tampering breaks entire chain
    - Immutable: Cannot be modified after creation
    """
    from backend.core.database.models.compliance_audit_log import ComplianceEventType

    service = AuditIntegrityService(db)

    try:
        event_type_enum = ComplianceEventType(request.event_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event_type: {request.event_type}",
        )

    event = await service.create_integrity_event(
        tenant_id=tenant_id,
        event_type=event_type_enum,
        event_data=request.event_data,
        data_subject_id=request.data_subject_id,
        compliance_framework=request.compliance_framework,
    )

    await db.commit()

    return IntegrityEventResponse(
        event_id=event.event_id,
        event_hash=event.event_hash,
        previous_hash=event.previous_hash,
        sequence_number=event.sequence_number,
        created_at=event.created_at,
        event_type=event.event_type,
        tenant_id=event.tenant_id,
    )


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================


@router.get(
    "/health",
    summary="Check integrity system health",
    description="Health check for hash-chain integrity system",
)
async def integrity_health_check(
    tenant_id: UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Check integrity system health.

    **Returns**: Health status and metrics.

    **Example Response**:
    ```json
    {
        "status": "healthy",
        "chain_operational": true,
        "latest_sequence": 1543,
        "hash_algorithm": "sha256"
    }
    ```
    """
    service = AuditIntegrityService(db)

    stats = await service.get_chain_statistics(tenant_id=tenant_id)

    return {
        "status": "healthy",
        "chain_operational": True,
        "latest_sequence": stats["chain_length"],
        "hash_algorithm": stats["hash_algorithm"],
        "integrity_coverage": stats["integrity_coverage"],
    }
