"""
Tenant Dependencies for Turkish Legal AI Platform.

Provides FastAPI dependency injection for multi-tenancy.

Features:
- Tenant ID extraction
- Tenant object retrieval
- Tenant validation
- Tenant-scoped operations

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.database import get_db
from backend.core import get_logger, settings
from backend.core.database.models import Tenant

logger = get_logger(__name__)


def get_current_tenant_id(request: Request) -> Optional[UUID]:
    """
    Extract current tenant ID from request state.

    Tenant ID is set by TenantContextMiddleware.

    Args:
        request: FastAPI request

    Returns:
        Tenant UUID or None

    Example:
        @app.get("/data")
        async def get_data(tenant_id: UUID = Depends(get_current_tenant_id)):
            return {"tenant_id": str(tenant_id)}
    """
    tenant_id = getattr(request.state, "tenant_id", None)

    if not tenant_id:
        if settings.MULTI_TENANT_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant ID is required",
            )
        return None

    return UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id


async def get_current_tenant(
    db: AsyncSession = Depends(get_db),
    tenant_id: Optional[UUID] = Depends(get_current_tenant_id),
) -> Optional[Tenant]:
    """
    Get current tenant from database.

    Args:
        db: Database session
        tenant_id: Tenant ID from request

    Returns:
        Tenant model instance or None

    Raises:
        HTTPException: If tenant not found or inactive

    Example:
        @app.get("/tenant-info")
        async def get_tenant_info(tenant: Tenant = Depends(get_current_tenant)):
            return {
                "id": str(tenant.id),
                "name": tenant.name,
                "plan": tenant.subscription_plan,
            }
    """
    if not tenant_id:
        return None

    result = await db.execute(
        select(Tenant)
        .where(Tenant.id == tenant_id)
        .where(Tenant.is_deleted == False)  # noqa: E712
    )

    tenant = result.scalar_one_or_none()

    if not tenant:
        logger.warning("Tenant not found", tenant_id=str(tenant_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    if not tenant.is_active:
        logger.warning("Inactive tenant attempted access", tenant_id=str(tenant_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is inactive",
        )

    return tenant


def require_tenant_feature(feature: str):
    """
    Dependency factory for feature flag checking.

    Args:
        feature: Required feature name

    Returns:
        Dependency function

    Example:
        @app.post("/advanced-analysis")
        async def advanced_analysis(
            _: None = Depends(require_tenant_feature("advanced_analysis")),
        ):
            # Tenant has advanced_analysis feature enabled
            pass
    """

    async def feature_checker(
        tenant: Tenant = Depends(get_current_tenant),
    ) -> None:
        """Check if tenant has required feature."""
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant required for this feature",
            )

        # Check tenant's features (assumes Tenant model has features JSONB field)
        tenant_features = getattr(tenant, "features", {})

        if not tenant_features.get(feature, False):
            logger.warning(
                "Feature not enabled for tenant",
                tenant_id=str(tenant.id),
                feature=feature,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature '{feature}' not enabled for your plan",
            )

    return feature_checker


__all__ = [
    "get_current_tenant_id",
    "get_current_tenant",
    "require_tenant_feature",
]
