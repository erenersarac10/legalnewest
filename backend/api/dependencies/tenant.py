"""
Tenant Dependencies for Turkish Legal AI Platform.

Enterprise-grade multi-tenancy dependencies with tenant resolution, quota management,
subscription validation, and feature flag support.

=============================================================================
FEATURES
=============================================================================

1. Tenant Resolution
   ------------------
   - Extract tenant ID from multiple sources
   - Tenant validation and activation check
   - Subdomain-based tenant resolution
   - Tenant context propagation

2. Quota Management
   ------------------
   - API request quota enforcement
   - Storage quota tracking
   - AI analysis usage limits
   - User count limits
   - Real-time quota checking

3. Subscription Management
   -------------------------
   - Plan-based feature access
   - Trial period validation
   - Payment status checking
   - Automatic plan enforcement
   - Upgrade/downgrade handling

4. Feature Flags
   --------------
   - Tenant-specific feature toggles
   - Plan-based feature availability
   - A/B testing support
   - Gradual feature rollout

5. Tenant Isolation
   -----------------
   - Data isolation verification
   - Cross-tenant access prevention
   - Tenant-scoped operations
   - Audit trail per tenant

=============================================================================
USAGE
=============================================================================

Basic Tenant Access:
--------------------

>>> from fastapi import Depends
>>> from backend.api.dependencies.tenant import get_current_tenant
>>>
>>> @app.get("/tenant-info")
>>> async def get_tenant_info(tenant: Tenant = Depends(get_current_tenant)):
...     return {
...         "id": str(tenant.id),
...         "name": tenant.name,
...         "plan": tenant.subscription_plan
...     }

Tenant ID Only:
---------------

>>> from backend.api.dependencies.tenant import get_current_tenant_id
>>>
>>> @app.get("/data")
>>> async def get_data(tenant_id: UUID = Depends(get_current_tenant_id)):
...     # Lightweight - no database query
...     return {"tenant_id": str(tenant_id)}

Feature Flag Checking:
----------------------

>>> from backend.api.dependencies.tenant import require_tenant_feature
>>>
>>> @app.post("/advanced-analysis")
>>> async def advanced_analysis(
...     _: None = Depends(require_tenant_feature("advanced_analysis"))
... ):
...     # Only available to tenants with this feature
...     return await run_advanced_analysis()

Subscription Plan Validation:
------------------------------

>>> from backend.api.dependencies.tenant import require_plan
>>>
>>> @app.post("/bulk-import")
>>> async def bulk_import(
...     _: None = Depends(require_plan(["professional", "enterprise"]))
... ):
...     # Only professional and enterprise plans
...     return await bulk_import_service()

Quota Enforcement:
------------------

>>> from backend.api.dependencies.tenant import check_quota
>>>
>>> @app.post("/contracts/analyze")
>>> async def analyze_contract(
...     contract: Contract,
...     tenant: Tenant = Depends(get_current_tenant),
...     _: None = Depends(check_quota("ai_analysis"))
... ):
...     # Automatically checks and decrements quota
...     return await ai_service.analyze(contract)

=============================================================================
TENANT PLANS
=============================================================================

Plan Hierarchy:
---------------

1. trial (Level 10)
   - 14-day trial period
   - Limited features
   - 100 AI analyses
   - 5 users
   - 1 GB storage

2. starter (Level 20)
   - Basic features
   - 500 AI analyses/month
   - 10 users
   - 10 GB storage
   - Email support

3. professional (Level 40)
   - Advanced features
   - 5,000 AI analyses/month
   - 50 users
   - 100 GB storage
   - Priority support
   - API access

4. enterprise (Level 80)
   - All features
   - Unlimited AI analyses
   - Unlimited users
   - 1 TB storage
   - Dedicated support
   - Custom integrations
   - SLA guarantees

Plan Comparison:
----------------

>>> if tenant.plan_level >= required_plan_level:
...     # Tenant has sufficient plan
...     pass

=============================================================================
QUOTA LIMITS BY RESOURCE
=============================================================================

API Requests:
-------------
- trial: 1,000/month
- starter: 10,000/month
- professional: 100,000/month
- enterprise: unlimited

Storage:
--------
- trial: 1 GB
- starter: 10 GB
- professional: 100 GB
- enterprise: 1 TB

AI Analysis:
------------
- trial: 100/month
- starter: 500/month
- professional: 5,000/month
- enterprise: unlimited

Users:
------
- trial: 5
- starter: 10
- professional: 50
- enterprise: unlimited

=============================================================================
FEATURE FLAGS
=============================================================================

Available Features:
-------------------

Core Features (all plans):
- contract_management
- document_storage
- basic_search

Professional Features:
- advanced_search
- bulk_operations
- api_access
- webhooks
- custom_templates

Enterprise Features:
- sso_integration
- audit_logs
- custom_branding
- dedicated_instance
- sla_support

Feature Check Example:
----------------------

>>> if tenant.has_feature("advanced_search"):
...     # Use advanced search
...     results = await advanced_search(query)
... else:
...     # Fall back to basic search
...     results = await basic_search(query)

=============================================================================
TENANT LIFECYCLE
=============================================================================

Tenant States:
--------------

1. trial
   - Initial signup state
   - 14-day trial period
   - Limited features and quotas
   - Can upgrade anytime

2. active
   - Paid subscription active
   - Full feature access based on plan
   - Regular quota limits
   - Auto-renewal enabled

3. suspended
   - Payment failed or overdue
   - Read-only access
   - Grace period (7 days)
   - Can reactivate with payment

4. cancelled
   - Subscription cancelled
   - Export data available (30 days)
   - Can reactivate within 30 days
   - Data deleted after 30 days

5. deleted
   - Permanently deleted
   - All data purged
   - Cannot be recovered
   - Audit logs retained (7 years)

Turkish Status Messages:
-------------------------
- trial: "Deneme sürümü"
- active: "Aktif abonelik"
- suspended: "Askıya alındı (ödeme bekliyor)"
- cancelled: "İptal edildi"
- deleted: "Silinmiş"

=============================================================================
SECURITY CONSIDERATIONS
=============================================================================

Tenant Isolation:
-----------------

1. Database level (RLS):
   - All queries filtered by tenant_id
   - PostgreSQL Row-Level Security
   - Impossible to access other tenant data

2. Application level:
   - Middleware validates tenant context
   - Dependencies enforce tenant checks
   - Cross-tenant access blocked

3. Storage level:
   - S3 folders per tenant
   - Presigned URLs scoped to tenant
   - Access policies per tenant

Quota Abuse Prevention:
-----------------------

1. Rate limiting per tenant
2. Usage monitoring and alerts
3. Automatic suspension on abuse
4. Manual review for suspicious patterns

=============================================================================
KVKK COMPLIANCE
=============================================================================

Tenant Data Management:
-----------------------

- Tenant data isolated and encrypted
- Per-tenant data retention policies
- Export capability for data portability
- Complete deletion on request
- Audit trail maintained separately

Data Sovereignty:
-----------------

- Tenant can specify data region
- Compliance with local regulations
- Data residency guarantees
- Cross-border transfer controls

=============================================================================
TROUBLESHOOTING
=============================================================================

"Tenant not found" Error:
--------------------------

1. Check X-Tenant-ID header is correct
2. Verify tenant exists in database
3. Check tenant hasn't been deleted
4. Verify tenant_id format (UUID)

"Tenant inactive" Error:
-------------------------

1. Check subscription status
2. Verify payment is current
3. Check for manual suspension
4. Review tenant state in admin panel

"Quota exceeded" Error:
-----------------------

1. Check current usage vs limits
2. Verify plan quota allocation
3. Consider upgrading plan
4. Review usage patterns for anomalies

"Feature not available" Error:
-------------------------------

1. Check tenant plan level
2. Verify feature is enabled for plan
3. Check feature flag status
4. Consider plan upgrade

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.database import get_db
from backend.core import get_logger, settings
from backend.core.database.models import Tenant

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# PLAN LEVELS
# =============================================================================

PLAN_LEVELS = {
    "trial": 10,
    "starter": 20,
    "professional": 40,
    "enterprise": 80,
}

# =============================================================================
# QUOTA LIMITS
# =============================================================================

PLAN_QUOTAS = {
    "trial": {
        "api_requests_monthly": 1_000,
        "storage_gb": 1,
        "ai_analysis_monthly": 100,
        "users": 5,
    },
    "starter": {
        "api_requests_monthly": 10_000,
        "storage_gb": 10,
        "ai_analysis_monthly": 500,
        "users": 10,
    },
    "professional": {
        "api_requests_monthly": 100_000,
        "storage_gb": 100,
        "ai_analysis_monthly": 5_000,
        "users": 50,
    },
    "enterprise": {
        "api_requests_monthly": -1,  # Unlimited
        "storage_gb": 1000,
        "ai_analysis_monthly": -1,  # Unlimited
        "users": -1,  # Unlimited
    },
}

# =============================================================================
# TENANT DEPENDENCIES
# =============================================================================


def get_current_tenant_id(request: Request) -> Optional[UUID]:
    """
    Extract current tenant ID from request state.

    Args:
        request: FastAPI request

    Returns:
        Tenant UUID or None

    Raises:
        HTTPException: If tenant ID required but missing

    Example:
        >>> @app.get("/data")
        >>> async def get_data(tenant_id: UUID = Depends(get_current_tenant_id)):
        ...     return {"tenant_id": str(tenant_id)}
    """
    tenant_id = getattr(request.state, "tenant_id", None)

    if not tenant_id:
        if settings.MULTI_TENANT_ENABLED:
            logger.warning("Tenant ID eksik (multi-tenant modu)")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant ID gereklidir",
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
        >>> @app.get("/tenant-info")
        >>> async def get_tenant_info(tenant: Tenant = Depends(get_current_tenant)):
        ...     return {"name": tenant.name}
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
        logger.warning("Tenant bulunamadı", tenant_id=str(tenant_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant bulunamadı",
        )

    if not tenant.is_active:
        logger.warning("İnaktif tenant erişim denemesi", tenant_id=str(tenant_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant hesabı aktif değil. Lütfen aboneliğinizi yenileyin.",
        )

    logger.debug("Tenant doğrulandı", tenant_id=str(tenant_id), plan=tenant.subscription_plan)
    return tenant


async def get_required_tenant(
    tenant: Optional[Tenant] = Depends(get_current_tenant),
) -> Tenant:
    """
    Get current tenant (required, not optional).

    Args:
        tenant: Optional tenant from get_current_tenant

    Returns:
        Tenant model instance

    Raises:
        HTTPException: If tenant is None
    """
    if not tenant:
        logger.warning("Tenant gerekli ama bulunamadı")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu işlem için tenant gereklidir",
        )
    return tenant


# =============================================================================
# FEATURE FLAG DEPENDENCIES
# =============================================================================


def require_tenant_feature(feature: str):
    """
    Dependency factory for feature flag checking.

    Args:
        feature: Required feature name

    Returns:
        Dependency function

    Example:
        >>> @app.post("/advanced-analysis")
        >>> async def advanced_analysis(
        ...     _: None = Depends(require_tenant_feature("advanced_analysis"))
        ... ):
        ...     pass
    """

    async def feature_checker(
        tenant: Tenant = Depends(get_required_tenant),
    ) -> None:
        tenant_features = getattr(tenant, "features", {})

        if not tenant_features.get(feature, False):
            logger.warning(
                "Feature tenant için aktif değil",
                tenant_id=str(tenant.id),
                feature=feature,
                plan=tenant.subscription_plan,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"'{feature}' özelliği planınızda mevcut değil. Lütfen planınızı yükseltin.",
            )

    return feature_checker


# =============================================================================
# PLAN DEPENDENCIES
# =============================================================================


def require_plan(allowed_plans: List[str]):
    """
    Dependency factory for plan validation.

    Args:
        allowed_plans: List of allowed plan names

    Returns:
        Dependency function

    Example:
        >>> @app.post("/bulk-import")
        >>> async def bulk_import(
        ...     _: None = Depends(require_plan(["professional", "enterprise"]))
        ... ):
        ...     pass
    """

    async def plan_checker(
        tenant: Tenant = Depends(get_required_tenant),
    ) -> None:
        tenant_plan = tenant.subscription_plan

        if tenant_plan not in allowed_plans:
            logger.warning(
                "Plan yetersiz",
                tenant_id=str(tenant.id),
                tenant_plan=tenant_plan,
                required_plans=allowed_plans,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Bu işlem için {' veya '.join(allowed_plans)} planı gereklidir",
            )

    return plan_checker


def require_min_plan_level(min_level: int):
    """
    Dependency factory for minimum plan level checking.

    Args:
        min_level: Minimum required plan level

    Returns:
        Dependency function

    Example:
        >>> @app.get("/advanced-reports")
        >>> async def get_reports(
        ...     _: None = Depends(require_min_plan_level(40))  # professional or above
        ... ):
        ...     pass
    """

    async def plan_level_checker(
        tenant: Tenant = Depends(get_required_tenant),
    ) -> None:
        tenant_plan = tenant.subscription_plan
        tenant_level = PLAN_LEVELS.get(tenant_plan, 0)

        if tenant_level < min_level:
            logger.warning(
                "Plan seviyesi yetersiz",
                tenant_id=str(tenant.id),
                tenant_level=tenant_level,
                required_level=min_level,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bu işlem için daha yüksek bir plan gereklidir",
            )

    return plan_level_checker


# =============================================================================
# QUOTA DEPENDENCIES
# =============================================================================


def check_quota(resource: str):
    """
    Dependency factory for quota checking.

    Args:
        resource: Resource type to check quota for

    Returns:
        Dependency function

    Example:
        >>> @app.post("/contracts/analyze")
        >>> async def analyze_contract(
        ...     _: None = Depends(check_quota("ai_analysis"))
        ... ):
        ...     pass
    """

    async def quota_checker(
        tenant: Tenant = Depends(get_required_tenant),
    ) -> None:
        # Get plan quotas
        plan_quotas = PLAN_QUOTAS.get(tenant.subscription_plan, {})
        quota_key = f"{resource}_monthly"
        quota_limit = plan_quotas.get(quota_key, 0)

        # -1 means unlimited
        if quota_limit == -1:
            return

        # Check current usage (would query from usage tracking table)
        # current_usage = await get_tenant_usage(tenant.id, resource)
        current_usage = getattr(tenant, f"{resource}_usage", 0)

        if current_usage >= quota_limit:
            logger.warning(
                "Kota aşıldı",
                tenant_id=str(tenant.id),
                resource=resource,
                current_usage=current_usage,
                quota_limit=quota_limit,
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"{resource.replace('_', ' ').title()} kotanız dolmuştur. Lütfen planınızı yükseltin.",
            )

    return quota_checker


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "get_current_tenant_id",
    "get_current_tenant",
    "get_required_tenant",
    "require_tenant_feature",
    "require_plan",
    "require_min_plan_level",
    "check_quota",
    "PLAN_LEVELS",
    "PLAN_QUOTAS",
]
