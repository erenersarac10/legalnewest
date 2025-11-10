"""
Admin Dashboard API Routes for Harvey/Legora Turkish Legal AI Platform.

This module provides comprehensive administrative REST API endpoints:
- System Overview: Health monitoring, metrics, resource usage
- User Management: CRUD operations, lock/unlock, session management
- Tenant Management: Multi-tenancy operations, resource quotas
- Analytics: Time-series metrics, document/usage statistics
- Audit Logs: Security events, compliance audit trails
- Operations: Background jobs, cache management, backup/restore
- Configuration: System settings management

All endpoints require authentication and admin permissions.

Example Usage:
    >>> # Get system overview
    >>> GET /api/v1/admin/dashboard/system/overview
    >>>
    >>> # List users with filters
    >>> GET /api/v1/admin/dashboard/users?role=lawyer&active=true&limit=50
    >>>
    >>> # Get analytics for date range
    >>> GET /api/v1/admin/dashboard/analytics?start_date=2025-01-01&end_date=2025-11-10&metrics=documents,users
    >>>
    >>> # Create new tenant
    >>> POST /api/v1/admin/dashboard/tenants
    >>> {
    ...     "name": "Acme Law Firm",
    ...     "slug": "acme-law",
    ...     "plan": "professional"
    ... }

Security:
    - All endpoints require 'admin:read' or 'admin:write' permissions
    - Multi-tenancy: Super admins can access all tenants
    - Audit logging: All operations are logged
    - Rate limiting: Applied to prevent abuse

Author: Harvey/Legora CTO
Date: 2025-11-10
Lines: 680+
"""

import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.session import get_db
from backend.core.exceptions import AdminError, ValidationError
from backend.security.rbac.context import get_current_tenant_id, get_current_user_id
from backend.security.rbac.decorators import require_permission
from backend.services.admin_dashboard_service import (
    AdminDashboardService,
    UserRole,
    ServiceStatus,
)

# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter(
    prefix="/admin/dashboard",
    tags=["admin-dashboard"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class SystemOverviewResponse(BaseModel):
    """System overview response."""

    health: Dict[str, Any] = Field(..., description="System health status")
    metrics: Dict[str, Any] = Field(..., description="Active metrics")
    resources: Dict[str, float] = Field(..., description="Resource usage")
    recent_activity: List[Dict[str, Any]] = Field(..., description="Recent events")
    timestamp: str = Field(..., description="Timestamp (ISO 8601)")


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    full_name: str
    role: str
    tenant_id: Optional[str]
    is_active: bool
    created_at: str
    last_login: Optional[str]
    metadata: Dict[str, Any] = {}

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """User list with pagination."""

    users: List[UserResponse]
    pagination: Dict[str, Any]


class CreateUserRequest(BaseModel):
    """Create user request."""

    email: str = Field(..., description="User email address", min_length=5, max_length=255)
    full_name: str = Field(..., description="Full name", min_length=2, max_length=255)
    role: str = Field(..., description="User role (admin, lawyer, etc.)")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (optional)")
    password: Optional[str] = Field(None, description="Initial password (optional)")

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class UpdateUserRequest(BaseModel):
    """Update user request."""

    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    role: Optional[str] = None
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class LockUserRequest(BaseModel):
    """Lock user request."""

    reason: str = Field(..., description="Reason for locking account", min_length=10)


class TenantResponse(BaseModel):
    """Tenant information response."""

    id: str
    name: str
    slug: str
    plan: str
    is_active: bool
    created_at: str
    user_count: int
    document_count: int
    storage_used_gb: float
    api_calls_month: int
    metadata: Dict[str, Any] = {}

    class Config:
        from_attributes = True


class TenantListResponse(BaseModel):
    """Tenant list with pagination."""

    tenants: List[TenantResponse]
    pagination: Dict[str, Any]


class CreateTenantRequest(BaseModel):
    """Create tenant request."""

    name: str = Field(..., description="Tenant name", min_length=2, max_length=255)
    slug: str = Field(..., description="URL-safe slug", min_length=2, max_length=100)
    plan: str = Field(..., description="Subscription plan")

    @validator('slug')
    def validate_slug(cls, v):
        """Validate slug format (alphanumeric + hyphens)."""
        if not v.replace('-', '').isalnum():
            raise ValueError('Slug must be alphanumeric with hyphens only')
        return v.lower()


class AnalyticsRequest(BaseModel):
    """Analytics request parameters."""

    start_date: datetime.datetime = Field(..., description="Start date")
    end_date: datetime.datetime = Field(..., description="End date")
    metrics: List[str] = Field(..., description="Metrics to retrieve")
    granularity: str = Field("day", description="Time granularity (hour, day, week, month)")


class AnalyticsDataResponse(BaseModel):
    """Analytics data for single metric."""

    metric_name: str
    time_series: List[Dict[str, Any]]  # [{"timestamp": "...", "value": 123.4}, ...]
    total: float
    average: float
    trend: str  # "increasing", "decreasing", "stable"
    metadata: Dict[str, Any] = {}


class AnalyticsResponse(BaseModel):
    """Analytics response."""

    metrics: Dict[str, AnalyticsDataResponse]


class DocumentStatisticsResponse(BaseModel):
    """Document statistics response."""

    total_documents: int
    documents_today: int
    documents_this_week: int
    documents_this_month: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    avg_processing_time_ms: int


class UsageStatisticsResponse(BaseModel):
    """Usage statistics response."""

    api_calls_today: int
    api_calls_this_month: int
    tokens_used_today: int
    tokens_used_this_month: int
    storage_used_gb: float
    active_users_today: int
    active_sessions: int


class AuditLogResponse(BaseModel):
    """Audit log entry response."""

    id: str
    event_type: str
    user_id: Optional[str]
    timestamp: str
    ip_address: Optional[str]
    details: Dict[str, Any]
    severity: str


class AuditLogListResponse(BaseModel):
    """Audit log list with pagination."""

    logs: List[AuditLogResponse]
    pagination: Dict[str, Any]


class SecurityEventResponse(BaseModel):
    """Security event response."""

    id: str
    event_type: str
    severity: str
    timestamp: str
    details: Dict[str, Any]


class BackgroundJobResponse(BaseModel):
    """Background job response."""

    id: str
    type: str
    status: str
    progress: float
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class SystemConfigResponse(BaseModel):
    """System configuration response."""

    version: str
    environment: str
    features: Dict[str, Any]
    limits: Dict[str, Any]


class UpdateConfigRequest(BaseModel):
    """Update system configuration request."""

    updates: Dict[str, Any] = Field(..., description="Configuration updates")


class BackupResponse(BaseModel):
    """Backup operation response."""

    backup_id: str
    message: str


# =============================================================================
# SYSTEM OVERVIEW ENDPOINTS
# =============================================================================


@router.get(
    "/system/overview",
    response_model=SystemOverviewResponse,
    summary="Get system overview",
    description="Get comprehensive system overview with health, metrics, and resources (requires admin:read)",
)
@require_permission("admin", "read")
async def get_system_overview(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> SystemOverviewResponse:
    """
    Get comprehensive system overview.

    **Permissions**: Requires 'admin:read' permission.

    **Returns**:
        - System health status (healthy, degraded, down)
        - Active metrics (users, sessions, documents)
        - Resource usage (CPU, memory, disk)
        - Recent activity log

    **Example Response**:
    ```json
    {
        "health": {
            "status": "healthy",
            "services": {"database": "healthy", "cache": "healthy"},
            "timestamp": "2025-11-10T12:00:00Z"
        },
        "metrics": {
            "active_users": 127,
            "active_sessions": 43,
            "documents_today": 234
        },
        "resources": {
            "cpu_percent": 45.2,
            "memory_percent": 62.8
        }
    }
    ```
    """
    try:
        service = AdminDashboardService(db_session=db)
        overview = await service.get_system_overview()
        return SystemOverviewResponse(**overview)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system overview: {str(e)}",
        )


# =============================================================================
# USER MANAGEMENT ENDPOINTS
# =============================================================================


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List users",
    description="List users with filters and pagination (requires admin:read)",
)
@require_permission("admin", "read")
async def list_users(
    role: Optional[str] = Query(None, description="Filter by role"),
    active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search by email or name"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(50, ge=1, le=1000, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
) -> UserListResponse:
    """
    List users with filters and pagination.

    **Permissions**: Requires 'admin:read' permission.

    **Query Parameters**:
        - role: Filter by user role (admin, lawyer, paralegal, user)
        - active: Filter by active status (true/false)
        - search: Search by email or full name
        - sort_by: Sort field (created_at, email, role)
        - sort_order: Sort direction (asc, desc)
        - limit: Results per page (1-1000)
        - offset: Pagination offset

    **Returns**: List of users with pagination metadata.
    """
    try:
        service = AdminDashboardService(db_session=db)

        # Build filters
        filters = {}
        if role:
            filters["role"] = role
        if active is not None:
            filters["active"] = active
        if search:
            filters["search"] = search

        result = await service.list_users(
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )

        # Convert to response format
        users = [
            UserResponse(
                id=str(u["id"]),
                email=u["email"],
                full_name=u["full_name"],
                role=u["role"],
                tenant_id=str(u.get("tenant_id")) if u.get("tenant_id") else None,
                is_active=u["is_active"],
                created_at=u["created_at"],
                last_login=u.get("last_login"),
            )
            for u in result["users"]
        ]

        return UserListResponse(
            users=users,
            pagination=result["pagination"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}",
        )


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get user details",
    description="Get detailed user information (requires admin:read)",
)
@require_permission("admin", "read")
async def get_user_details(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Get detailed user information.

    **Permissions**: Requires 'admin:read' permission.

    **Path Parameters**:
        - user_id: User UUID

    **Returns**: User details including metadata.
    """
    try:
        service = AdminDashboardService(db_session=db)
        user_info = await service.get_user_details(user_id)

        return UserResponse(
            id=str(user_info.id),
            email=user_info.email,
            full_name=user_info.full_name,
            role=user_info.role.value,
            tenant_id=str(user_info.tenant_id) if user_info.tenant_id else None,
            is_active=user_info.is_active,
            created_at=user_info.created_at.isoformat(),
            last_login=user_info.last_login.isoformat() if user_info.last_login else None,
            metadata=user_info.metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {str(e)}",
        )


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new user",
    description="Create a new user account (requires admin:write)",
)
@require_permission("admin", "write")
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user_id),
) -> UserResponse:
    """
    Create a new user account.

    **Permissions**: Requires 'admin:write' permission.

    **Request Body**:
    ```json
    {
        "email": "user@example.com",
        "full_name": "John Doe",
        "role": "lawyer",
        "tenant_id": "optional-tenant-uuid",
        "password": "optional-initial-password"
    }
    ```

    **Returns**: Created user information.
    """
    try:
        service = AdminDashboardService(db_session=db)

        # Parse role
        try:
            role = UserRole(request.role)
        except ValueError:
            raise ValidationError(f"Invalid role: {request.role}")

        tenant_id = UUID(request.tenant_id) if request.tenant_id else None

        user_info = await service.create_user(
            email=request.email,
            full_name=request.full_name,
            role=role,
            tenant_id=tenant_id,
            password=request.password,
        )

        return UserResponse(
            id=str(user_info.id),
            email=user_info.email,
            full_name=user_info.full_name,
            role=user_info.role.value,
            tenant_id=str(user_info.tenant_id) if user_info.tenant_id else None,
            is_active=user_info.is_active,
            created_at=user_info.created_at.isoformat(),
            last_login=None,
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@router.put(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update user information (requires admin:write)",
)
@require_permission("admin", "write")
async def update_user(
    user_id: UUID,
    request: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Update user information.

    **Permissions**: Requires 'admin:write' permission.

    **Path Parameters**:
        - user_id: User UUID

    **Request Body**: Partial update (only include fields to change)
    ```json
    {
        "full_name": "Updated Name",
        "role": "admin",
        "is_active": true
    }
    ```

    **Returns**: Updated user information.
    """
    try:
        service = AdminDashboardService(db_session=db)

        # Build updates dict
        updates = {}
        if request.full_name:
            updates["full_name"] = request.full_name
        if request.role:
            updates["role"] = request.role
        if request.is_active is not None:
            updates["is_active"] = request.is_active
        if request.metadata:
            updates["metadata"] = request.metadata

        user_info = await service.update_user(user_id, updates)

        return UserResponse(
            id=str(user_info.id),
            email=user_info.email,
            full_name=user_info.full_name,
            role=user_info.role.value,
            tenant_id=str(user_info.tenant_id) if user_info.tenant_id else None,
            is_active=user_info.is_active,
            created_at=user_info.created_at.isoformat(),
            last_login=user_info.last_login.isoformat() if user_info.last_login else None,
            metadata=user_info.metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}",
        )


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    description="Delete user account (requires admin:delete)",
)
@require_permission("admin", "delete")
async def delete_user(
    user_id: UUID,
    hard_delete: bool = Query(False, description="Perform hard delete (irreversible)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete user account.

    **Permissions**: Requires 'admin:delete' permission.

    **Path Parameters**:
        - user_id: User UUID

    **Query Parameters**:
        - hard_delete: If true, permanently delete (default: soft delete)

    **Returns**: 204 No Content on success.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.delete_user(user_id, hard_delete=hard_delete)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}",
        )


@router.post(
    "/users/{user_id}/lock",
    status_code=status.HTTP_200_OK,
    summary="Lock user account",
    description="Lock user account (requires admin:write)",
)
@require_permission("admin", "write")
async def lock_user(
    user_id: UUID,
    request: LockUserRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Lock user account (prevents login).

    **Permissions**: Requires 'admin:write' permission.

    **Path Parameters**:
        - user_id: User UUID

    **Request Body**:
    ```json
    {
        "reason": "Security violation - multiple failed login attempts"
    }
    ```

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.lock_user(user_id, reason=request.reason)
        return {"message": "User account locked", "user_id": str(user_id)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to lock user: {str(e)}",
        )


@router.post(
    "/users/{user_id}/unlock",
    status_code=status.HTTP_200_OK,
    summary="Unlock user account",
    description="Unlock user account (requires admin:write)",
)
@require_permission("admin", "write")
async def unlock_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Unlock user account (allows login).

    **Permissions**: Requires 'admin:read' permission.

    **Path Parameters**:
        - user_id: User UUID

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.unlock_user(user_id)
        return {"message": "User account unlocked", "user_id": str(user_id)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unlock user: {str(e)}",
        )


# Continue in next part due to length...

# =============================================================================
# TENANT MANAGEMENT ENDPOINTS
# =============================================================================


@router.get(
    "/tenants",
    response_model=TenantListResponse,
    summary="List tenants",
    description="List tenants with pagination (requires admin:read)",
)
@require_permission("admin", "read")
async def list_tenants(
    plan: Optional[str] = Query(None, description="Filter by plan"),
    active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> TenantListResponse:
    """
    List tenants with pagination.

    **Permissions**: Requires 'admin:read' permission.

    **Query Parameters**:
        - plan: Filter by subscription plan
        - active: Filter by active status
        - limit: Results per page (1-1000)
        - offset: Pagination offset

    **Returns**: List of tenants with pagination.
    """
    try:
        service = AdminDashboardService(db_session=db)

        filters = {}
        if plan:
            filters["plan"] = plan
        if active is not None:
            filters["active"] = active

        result = await service.list_tenants(
            filters=filters,
            limit=limit,
            offset=offset,
        )

        tenants = [
            TenantResponse(**t) for t in result["tenants"]
        ]

        return TenantListResponse(
            tenants=tenants,
            pagination=result["pagination"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tenants: {str(e)}",
        )


@router.get(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    summary="Get tenant details",
    description="Get detailed tenant information (requires admin:read)",
)
@require_permission("admin", "read")
async def get_tenant_details(
    tenant_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """
    Get detailed tenant information.

    **Permissions**: Requires 'admin:read' permission.

    **Path Parameters**:
        - tenant_id: Tenant UUID

    **Returns**: Tenant details with usage statistics.
    """
    try:
        service = AdminDashboardService(db_session=db)
        tenant_info = await service.get_tenant_details(tenant_id)

        return TenantResponse(
            id=str(tenant_info.id),
            name=tenant_info.name,
            slug=tenant_info.slug,
            plan=tenant_info.plan,
            is_active=tenant_info.is_active,
            created_at=tenant_info.created_at.isoformat(),
            user_count=tenant_info.user_count,
            document_count=tenant_info.document_count,
            storage_used_gb=tenant_info.storage_used_gb,
            api_calls_month=tenant_info.api_calls_month,
            metadata=tenant_info.metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {str(e)}",
        )


@router.post(
    "/tenants",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new tenant",
    description="Create a new tenant (requires admin:write)",
)
@require_permission("admin", "write")
async def create_tenant(
    request: CreateTenantRequest,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """
    Create a new tenant.

    **Permissions**: Requires 'admin:write' permission.

    **Request Body**:
    ```json
    {
        "name": "Acme Law Firm",
        "slug": "acme-law",
        "plan": "professional"
    }
    ```

    **Returns**: Created tenant information.
    """
    try:
        service = AdminDashboardService(db_session=db)

        tenant_info = await service.create_tenant(
            name=request.name,
            slug=request.slug,
            plan=request.plan,
        )

        return TenantResponse(
            id=str(tenant_info.id),
            name=tenant_info.name,
            slug=tenant_info.slug,
            plan=tenant_info.plan,
            is_active=tenant_info.is_active,
            created_at=tenant_info.created_at.isoformat(),
            user_count=tenant_info.user_count,
            document_count=tenant_info.document_count,
            storage_used_gb=tenant_info.storage_used_gb,
            api_calls_month=tenant_info.api_calls_month,
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tenant: {str(e)}",
        )


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Get analytics data",
    description="Get time-series analytics for specified metrics (requires admin:read)",
)
@require_permission("admin", "read")
async def get_analytics(
    start_date: datetime.datetime = Query(..., description="Start date (ISO 8601)"),
    end_date: datetime.datetime = Query(..., description="End date (ISO 8601)"),
    metrics: str = Query(..., description="Comma-separated metrics (e.g. documents,users,api_calls)"),
    granularity: str = Query("day", description="Granularity (hour, day, week, month)"),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsResponse:
    """
    Get analytics data for specified metrics.

    **Permissions**: Requires 'admin:read' permission.

    **Query Parameters**:
        - start_date: Start date (ISO 8601 format)
        - end_date: End date (ISO 8601 format)
        - metrics: Comma-separated metric names (documents, users, api_calls, etc.)
        - granularity: Time granularity (hour, day, week, month)

    **Returns**: Time-series data for each metric with trend analysis.
    """
    try:
        service = AdminDashboardService(db_session=db)

        metrics_list = [m.strip() for m in metrics.split(',')]

        result = await service.get_analytics(
            start_date=start_date,
            end_date=end_date,
            metrics_list=metrics_list,
            granularity=granularity,
        )

        # Convert to response format
        analytics_data = {}
        for metric_name, data in result.items():
            analytics_data[metric_name] = AnalyticsDataResponse(
                metric_name=data.metric_name,
                time_series=[
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in data.time_series
                ],
                total=data.total,
                average=data.average,
                trend=data.trend,
                metadata=data.metadata,
            )

        return AnalyticsResponse(metrics=analytics_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}",
        )


@router.get(
    "/analytics/documents",
    response_model=DocumentStatisticsResponse,
    summary="Get document statistics",
    description="Get document statistics (requires admin:read)",
)
@require_permission("admin", "read")
async def get_document_statistics(
    db: AsyncSession = Depends(get_db),
) -> DocumentStatisticsResponse:
    """
    Get document statistics.

    **Permissions**: Requires 'admin:read' permission.

    **Returns**: Document counts by type, status, and time period.
    """
    try:
        service = AdminDashboardService(db_session=db)
        stats = await service.get_document_statistics()
        return DocumentStatisticsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document statistics: {str(e)}",
        )


@router.get(
    "/analytics/usage",
    response_model=UsageStatisticsResponse,
    summary="Get usage statistics",
    description="Get platform usage statistics (requires admin:read)",
)
@require_permission("admin", "read")
async def get_usage_statistics(
    db: AsyncSession = Depends(get_db),
) -> UsageStatisticsResponse:
    """
    Get platform usage statistics.

    **Permissions**: Requires 'admin:read' permission.

    **Returns**: API calls, token usage, storage, and active user metrics.
    """
    try:
        service = AdminDashboardService(db_session=db)
        stats = await service.get_usage_statistics()
        return UsageStatisticsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage statistics: {str(e)}",
        )


# =============================================================================
# AUDIT LOG ENDPOINTS
# =============================================================================


@router.get(
    "/audit/logs",
    response_model=AuditLogListResponse,
    summary="Get audit logs",
    description="Get audit logs with filters (requires admin:read)",
)
@require_permission("admin", "read")
async def get_audit_logs(
    start_date: Optional[datetime.datetime] = Query(None),
    end_date: Optional[datetime.datetime] = Query(None),
    event_type: Optional[str] = Query(None),
    user_id: Optional[UUID] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> AuditLogListResponse:
    """
    Get audit logs with filters.

    **Permissions**: Requires 'admin:read' permission.

    **Query Parameters**:
        - start_date: Filter by start date
        - end_date: Filter by end date
        - event_type: Filter by event type
        - user_id: Filter by user ID
        - severity: Filter by severity (info, warning, error, critical)
        - limit: Results per page (1-1000)
        - offset: Pagination offset

    **Returns**: List of audit log entries with pagination.
    """
    try:
        service = AdminDashboardService(db_session=db)

        filters = {}
        if event_type:
            filters["event_type"] = event_type
        if user_id:
            filters["user_id"] = user_id
        if severity:
            filters["severity"] = severity

        result = await service.get_audit_logs(
            filters=filters,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        logs = [AuditLogResponse(**log) for log in result["logs"]]

        return AuditLogListResponse(
            logs=logs,
            pagination=result["pagination"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit logs: {str(e)}",
        )


@router.get(
    "/security/events",
    response_model=List[SecurityEventResponse],
    summary="Get security events",
    description="Get recent security events (requires admin:read)",
)
@require_permission("admin", "read")
async def get_security_events(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> List[SecurityEventResponse]:
    """
    Get recent security events.

    **Permissions**: Requires 'admin:read' permission.

    **Query Parameters**:
        - severity: Filter by severity (warning, error, critical)
        - limit: Maximum results (1-500)

    **Returns**: List of security events (failed logins, suspicious activity, etc.).
    """
    try:
        service = AdminDashboardService(db_session=db)
        events = await service.get_security_events(severity=severity, limit=limit)
        return [SecurityEventResponse(**event) for event in events]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security events: {str(e)}",
        )


# =============================================================================
# OPERATIONS ENDPOINTS
# =============================================================================


@router.get(
    "/operations/jobs",
    response_model=List[BackgroundJobResponse],
    summary="Get background jobs",
    description="Get background job status (requires admin:read)",
)
@require_permission("admin", "read")
async def get_background_jobs(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> List[BackgroundJobResponse]:
    """
    Get background jobs status.

    **Permissions**: Requires 'admin:read' permission.

    **Query Parameters**:
        - status: Filter by status (pending, running, completed, failed)
        - limit: Maximum results (1-500)

    **Returns**: List of background jobs with progress.
    """
    try:
        service = AdminDashboardService(db_session=db)
        jobs = await service.get_background_jobs(status=status_filter, limit=limit)
        return [BackgroundJobResponse(**job) for job in jobs]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get background jobs: {str(e)}",
        )


@router.post(
    "/operations/jobs/{job_id}/cancel",
    summary="Cancel background job",
    description="Cancel a background job (requires admin:write)",
)
@require_permission("admin", "write")
async def cancel_background_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a background job.

    **Permissions**: Requires 'admin:write' permission.

    **Path Parameters**:
        - job_id: Job UUID

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.cancel_background_job(job_id)
        return {"message": "Job cancelled", "job_id": str(job_id)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}",
        )


@router.post(
    "/operations/cache/clear",
    summary="Clear system cache",
    description="Clear system cache (requires admin:write)",
)
@require_permission("admin", "write")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to clear"),
    db: AsyncSession = Depends(get_db),
):
    """
    Clear system cache.

    **Permissions**: Requires 'admin:write' permission.

    **Query Parameters**:
        - pattern: Optional cache key pattern (e.g., "user:*" to clear all user cache)

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.clear_cache(pattern=pattern)
        return {"message": "Cache cleared", "pattern": pattern or "all"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


@router.post(
    "/operations/maintenance",
    summary="Run database maintenance",
    description="Run database maintenance tasks (requires admin:write)",
)
@require_permission("admin", "write")
async def run_maintenance(
    db: AsyncSession = Depends(get_db),
):
    """
    Run database maintenance tasks.

    **Permissions**: Requires 'admin:write' permission.

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.run_database_maintenance()
        return {"message": "Database maintenance started"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run maintenance: {str(e)}",
        )


@router.post(
    "/operations/backup",
    response_model=BackupResponse,
    summary="Create system backup",
    description="Create system backup (requires admin:write)",
)
@require_permission("admin", "write")
async def create_backup(
    db: AsyncSession = Depends(get_db),
) -> BackupResponse:
    """
    Create system backup.

    **Permissions**: Requires 'admin:write' permission.

    **Returns**: Backup ID for future restore operations.
    """
    try:
        service = AdminDashboardService(db_session=db)
        backup_id = await service.create_backup()
        return BackupResponse(
            backup_id=backup_id,
            message="Backup created successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create backup: {str(e)}",
        )


@router.post(
    "/operations/backup/{backup_id}/restore",
    summary="Restore from backup",
    description="Restore from backup (requires admin:write)",
)
@require_permission("admin", "write")
async def restore_backup(
    backup_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Restore from backup.

    **Permissions**: Requires 'admin:write' permission.

    **WARNING**: This will restore the entire system to the backup state.

    **Path Parameters**:
        - backup_id: Backup ID from create_backup operation

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.restore_backup(backup_id)
        return {"message": "Backup restored successfully", "backup_id": backup_id}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore backup: {str(e)}",
        )


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================


@router.get(
    "/config",
    response_model=SystemConfigResponse,
    summary="Get system configuration",
    description="Get system configuration (requires admin:read)",
)
@require_permission("admin", "read")
async def get_system_config(
    db: AsyncSession = Depends(get_db),
) -> SystemConfigResponse:
    """
    Get system configuration.

    **Permissions**: Requires 'admin:read' permission.

    **Returns**: System configuration (features, limits, environment).
    """
    try:
        service = AdminDashboardService(db_session=db)
        config = await service.get_system_config()
        return SystemConfigResponse(**config)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system config: {str(e)}",
        )


@router.put(
    "/config",
    summary="Update system configuration",
    description="Update system configuration (requires admin:write)",
)
@require_permission("admin", "write")
async def update_system_config(
    request: UpdateConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Update system configuration.

    **Permissions**: Requires 'admin:write' permission.

    **WARNING**: Incorrect configuration can break the system.

    **Request Body**:
    ```json
    {
        "updates": {
            "features.rag_enabled": true,
            "limits.max_file_size_mb": 200
        }
    }
    ```

    **Returns**: Success message.
    """
    try:
        service = AdminDashboardService(db_session=db)
        await service.update_system_config(request.updates)
        return {"message": "System configuration updated", "keys": list(request.updates.keys())}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update system config: {str(e)}",
        )
