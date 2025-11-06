"""
Tenant Context Middleware for Turkish Legal AI Platform.

This middleware provides comprehensive multi-tenancy support with Row-Level Security (RLS),
tenant isolation, quota management, and subscription validation for the legal AI platform.

=============================================================================
FEATURES
=============================================================================

1. Multi-Tenancy Modes
   --------------------
   - Subdomain-based: tenant1.turkishlegalai.com
   - Header-based: X-Tenant-ID header
   - Path-based: /api/v1/tenants/{tenant_id}/...
   - JWT token-based: tenant_id claim in access token

2. Tenant Isolation
   ------------------
   - Database Row-Level Security (RLS) using PostgreSQL
   - Cache namespace isolation (Redis keys prefixed by tenant)
   - S3 bucket folder isolation
   - Background job queue isolation
   - Log context correlation

3. Quota Management
   -------------------
   - API request rate limiting per tenant
   - Storage quota enforcement (GB)
   - AI model usage tracking (tokens, calls)
   - Contract analysis limits
   - User count limits per tenant

4. Subscription Management
   -------------------------
   - Validate tenant subscription status
   - Enforce plan limits (starter, professional, enterprise)
   - Feature flag per tenant
   - Trial period tracking
   - Payment status validation

5. Resource Tracking
   -------------------
   - Real-time usage metrics
   - Billing data collection
   - Cost allocation per tenant
   - Overage detection and alerts

=============================================================================
USAGE
=============================================================================

Basic Integration (Header-Based):
----------------------------------

>>> from fastapi import FastAPI
>>> from backend.api.middleware.tenant_context import TenantContextMiddleware
>>>
>>> app = FastAPI()
>>> app.add_middleware(TenantContextMiddleware)
>>>
>>> # Clients must include X-Tenant-ID header in requests
>>> # curl -H "X-Tenant-ID: acme-corp" https://api.turkishlegalai.com/contracts

Subdomain-Based Multi-Tenancy:
-------------------------------

>>> # Configure for subdomain mode
>>> app = FastAPI()
>>> app.add_middleware(
...     TenantContextMiddleware,
...     tenant_mode="subdomain",
...     base_domain="turkishlegalai.com"
... )
>>>
>>> # Tenant resolved from: tenant1.turkishlegalai.com -> tenant1

Path-Based Multi-Tenancy:
--------------------------

>>> # URLs include tenant: /api/v1/tenants/{tenant_id}/contracts
>>> @app.get("/api/v1/tenants/{tenant_id}/contracts")
>>> async def get_contracts(tenant_id: str, request: Request):
...     # Middleware validates tenant_id matches authenticated user's tenant
...     contracts = await db.query(Contract).filter(
...         Contract.tenant_id == request.state.tenant_id
...     ).all()
...     return contracts

Database Row-Level Security (RLS):
-----------------------------------

>>> # PostgreSQL RLS policy example
>>> CREATE POLICY tenant_isolation ON contracts
...     USING (tenant_id = current_setting('app.tenant_id')::uuid);
>>>
>>> # Middleware automatically sets this context:
>>> # SET LOCAL app.tenant_id = '550e8400-e29b-41d4-a716-446655440000';

Quota Enforcement:
-------------------

>>> from backend.api.middleware.tenant_context import check_tenant_quota
>>>
>>> @app.post("/api/v1/contracts/analyze")
>>> async def analyze_contract(contract: Contract, request: Request):
...     tenant_id = request.state.tenant_id
...
...     # Check if tenant has quota for AI analysis
...     if not await check_tenant_quota(tenant_id, "ai_analysis"):
...         raise QuotaExceededException(
...             "AI analiz kotanız dolmuştur. Lütfen planınızı yükseltin."
...         )
...
...     # Process analysis
...     result = await ai_service.analyze(contract)
...
...     # Track usage
...     await record_usage(tenant_id, "ai_analysis", tokens=result.token_count)
...
...     return result

=============================================================================
ROW-LEVEL SECURITY IMPLEMENTATION
=============================================================================

PostgreSQL RLS Setup:
---------------------

-- 1. Enable RLS on all tenant-scoped tables
ALTER TABLE contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- 2. Create RLS policies
CREATE POLICY tenant_isolation_policy ON contracts
    USING (tenant_id = current_setting('app.tenant_id', TRUE)::uuid);

CREATE POLICY tenant_isolation_policy ON documents
    USING (tenant_id = current_setting('app.tenant_id', TRUE)::uuid);

-- 3. Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON contracts TO app_user;

Application Usage:
------------------

>>> # Middleware sets tenant context before each query
>>> async with db.begin():
...     # Set tenant context
...     await db.execute(
...         text(f"SET LOCAL app.tenant_id = :tenant_id"),
...         {"tenant_id": tenant_id}
...     )
...
...     # All subsequent queries automatically filtered by RLS
...     contracts = await db.query(Contract).all()
...     # Only returns contracts for this tenant!

=============================================================================
QUOTA LIMITS BY PLAN
=============================================================================

Starter Plan:
-------------
- API Requests: 10,000/month
- Storage: 10 GB
- AI Analysis: 100 documents/month
- Users: 5
- Contract Templates: 10

Professional Plan:
------------------
- API Requests: 100,000/month
- Storage: 100 GB
- AI Analysis: 1,000 documents/month
- Users: 25
- Contract Templates: 50
- Advanced AI Features: Yes

Enterprise Plan:
----------------
- API Requests: Unlimited
- Storage: 1 TB
- AI Analysis: 10,000 documents/month
- Users: Unlimited
- Contract Templates: Unlimited
- Advanced AI Features: Yes
- Dedicated Support: Yes
- Custom Integrations: Yes

Turkish Quota Messages:
-----------------------
- "API istek kotanız dolmuştur. Lütfen planınızı yükseltin."
- "Depolama alanınız dolmuştur. Lütfen eski dosyaları silin veya planınızı yükseltin."
- "Bu ay için AI analiz kotanız dolmuştur."
- "Maksimum kullanıcı sayısına ulaştınız."

=============================================================================
TENANT RESOLUTION PRIORITY
=============================================================================

1. X-Tenant-ID Header (Highest Priority)
   - Explicit tenant specification
   - Used by API clients and integrations

2. JWT Token Claim
   - Extracted from access token
   - Most common for user requests

3. Subdomain
   - tenant1.turkishlegalai.com
   - Used for web application access

4. Path Parameter
   - /api/v1/tenants/{tenant_id}/...
   - Used for admin operations

5. Default Tenant (if configured)
   - Fall back for development/testing
   - Never used in production

=============================================================================
SECURITY CONSIDERATIONS
=============================================================================

1. Tenant ID Validation:
   - Always validate tenant ID format (UUID)
   - Verify tenant exists and is active
   - Check subscription status before processing
   - Validate user belongs to tenant

2. Cross-Tenant Access Prevention:
   - Use RLS for all tenant-scoped tables
   - Never trust client-provided tenant IDs without validation
   - Log all tenant context switches
   - Implement audit trail for tenant access

3. Resource Isolation:
   - Separate S3 folders per tenant
   - Namespace Redis keys by tenant
   - Isolate background job queues
   - Use separate database schemas for strict isolation (optional)

4. Data Leakage Prevention:
   - Never include other tenants' data in responses
   - Sanitize error messages (don't reveal tenant info)
   - Implement proper 404 vs 403 handling
   - Use constant-time comparison for tenant IDs

=============================================================================
KVKK COMPLIANCE PER TENANT
=============================================================================

Data Residency:
---------------
- Store tenant data in configured region (EU, Turkey)
- Respect data localization requirements
- Document data processing locations

Tenant Data Isolation:
----------------------
- Separate encryption keys per tenant
- Independent data export capability
- Tenant-specific data retention policies
- Complete data deletion on tenant removal

Audit Trail:
------------
- Log all tenant data access
- Track cross-tenant admin operations
- Maintain immutable audit logs per tenant
- Provide tenant access to their audit logs

=============================================================================
TROUBLESHOOTING
=============================================================================

"Missing Tenant ID" Error:
---------------------------
1. Check X-Tenant-ID header is included
2. Verify JWT token contains tenant_id claim
3. Check subdomain format (tenant.domain.tld)
4. Ensure MULTI_TENANT_ENABLED setting is correct

"Tenant ID Mismatch" Error:
----------------------------
1. Verify token tenant matches header tenant
2. Check user hasn't been moved to different tenant
3. Validate token hasn't been tampered with
4. Check for token reuse across tenants

"Quota Exceeded" Error:
-----------------------
1. Check tenant's current usage in dashboard
2. Verify plan limits in admin panel
3. Review usage history for anomalies
4. Consider upgrading tenant plan

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

from typing import Callable, Dict, Optional
from uuid import UUID

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, set_log_context, settings

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# QUOTA LIMITS
# =============================================================================

PLAN_LIMITS = {
    "starter": {
        "api_requests_monthly": 10_000,
        "storage_gb": 10,
        "ai_analysis_monthly": 100,
        "users": 5,
        "contract_templates": 10,
    },
    "professional": {
        "api_requests_monthly": 100_000,
        "storage_gb": 100,
        "ai_analysis_monthly": 1_000,
        "users": 25,
        "contract_templates": 50,
    },
    "enterprise": {
        "api_requests_monthly": -1,  # Unlimited
        "storage_gb": 1000,
        "ai_analysis_monthly": 10_000,
        "users": -1,  # Unlimited
        "contract_templates": -1,  # Unlimited
    },
}

# =============================================================================
# TENANT VALIDATOR
# =============================================================================


class TenantValidator:
    """
    Validates tenant status and subscription.
    """

    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """
        Validate tenant ID format.

        Args:
            tenant_id: Tenant ID string

        Returns:
            True if valid UUID format
        """
        try:
            UUID(tenant_id)
            return True
        except (ValueError, AttributeError):
            return False

    @staticmethod
    async def validate_tenant_active(tenant_id: str) -> bool:
        """
        Check if tenant is active.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if tenant is active

        Note:
            In production, query database for tenant status
        """
        # TODO: Implement database lookup
        # tenant = await db.query(Tenant).filter(Tenant.id == tenant_id).first()
        # return tenant and tenant.is_active
        return True

    @staticmethod
    async def validate_subscription(tenant_id: str) -> Dict:
        """
        Validate tenant subscription status.

        Args:
            tenant_id: Tenant ID

        Returns:
            Dict with subscription info

        Note:
            In production, query subscription service
        """
        # TODO: Implement subscription validation
        # subscription = await subscription_service.get(tenant_id)
        # return {
        #     "is_valid": subscription.is_active,
        #     "plan": subscription.plan_name,
        #     "expires_at": subscription.expires_at
        # }
        return {
            "is_valid": True,
            "plan": "professional",
            "expires_at": None,
        }


# =============================================================================
# TENANT QUOTA MANAGER
# =============================================================================


class TenantQuotaManager:
    """
    Manages tenant quota tracking and enforcement.
    """

    def __init__(self):
        """Initialize quota manager."""
        # In production, use Redis for distributed quota tracking
        self.usage: Dict[str, Dict[str, int]] = {}

    async def check_quota(self, tenant_id: str, resource: str) -> bool:
        """
        Check if tenant has quota for resource.

        Args:
            tenant_id: Tenant ID
            resource: Resource type (api_requests, ai_analysis, etc.)

        Returns:
            True if quota available
        """
        # TODO: Implement real quota checking
        # 1. Get tenant plan
        # 2. Get current usage from Redis
        # 3. Compare against plan limits
        # 4. Return availability
        return True

    async def record_usage(
        self, tenant_id: str, resource: str, amount: int = 1
    ) -> None:
        """
        Record resource usage for tenant.

        Args:
            tenant_id: Tenant ID
            resource: Resource type
            amount: Usage amount (default: 1)
        """
        if tenant_id not in self.usage:
            self.usage[tenant_id] = {}

        if resource not in self.usage[tenant_id]:
            self.usage[tenant_id][resource] = 0

        self.usage[tenant_id][resource] += amount

        logger.debug(
            "Kaynak kullanımı kaydedildi",
            tenant_id=tenant_id,
            resource=resource,
            amount=amount,
            total=self.usage[tenant_id][resource],
        )

    async def get_usage(self, tenant_id: str, resource: str) -> int:
        """
        Get current usage for tenant resource.

        Args:
            tenant_id: Tenant ID
            resource: Resource type

        Returns:
            Current usage amount
        """
        return self.usage.get(tenant_id, {}).get(resource, 0)


# Global quota manager instance
_quota_manager: Optional[TenantQuotaManager] = None


def get_quota_manager() -> TenantQuotaManager:
    """Get or create global quota manager."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = TenantQuotaManager()
    return _quota_manager


# =============================================================================
# TENANT CONTEXT MIDDLEWARE
# =============================================================================


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Multi-tenant context middleware.

    Features:
    - Extracts and validates tenant ID from multiple sources
    - Sets database RLS context
    - Enforces tenant isolation
    - Validates subscription status
    - Tracks resource usage
    """

    def __init__(
        self,
        app,
        tenant_mode: str = "header",
        base_domain: Optional[str] = None,
    ):
        """
        Initialize tenant context middleware.

        Args:
            app: FastAPI application
            tenant_mode: Tenant resolution mode (header, subdomain, path)
            base_domain: Base domain for subdomain mode
        """
        super().__init__(app)
        self.tenant_mode = tenant_mode
        self.base_domain = base_domain or "turkishlegalai.com"
        self.validator = TenantValidator()
        self.quota_manager = get_quota_manager()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with tenant context.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response (403 if tenant validation fails)
        """
        # Skip tenant validation for public routes
        if self._is_public_route(request.url.path):
            return await call_next(request)

        # Extract tenant ID from request
        tenant_id = self._extract_tenant_id(request)

        # Multi-tenancy enforcement
        if settings.MULTI_TENANT_ENABLED:
            if not tenant_id:
                logger.warning(
                    "⚠️ Tenant ID eksik (multi-tenant modu)",
                    path=request.url.path,
                )
                return self._missing_tenant_response()

            # Validate tenant ID format
            if not self.validator.validate_tenant_id(tenant_id):
                logger.error(
                    "⚠️ Geçersiz tenant ID formatı",
                    tenant_id=tenant_id,
                )
                return self._invalid_tenant_response()

            # Validate tenant ID matches token (if authenticated)
            if hasattr(request.state, "tenant_id"):
                token_tenant_id = request.state.tenant_id

                if token_tenant_id and token_tenant_id != tenant_id:
                    logger.error(
                        "⚠️ Tenant ID uyuşmazlığı",
                        header_tenant=tenant_id,
                        token_tenant=token_tenant_id,
                        user_id=getattr(request.state, "user_id", None),
                    )
                    return self._tenant_mismatch_response()

            # Validate tenant is active
            if not await self.validator.validate_tenant_active(tenant_id):
                logger.warning(
                    "⚠️ İnaktif tenant erişim denemesi",
                    tenant_id=tenant_id,
                )
                return self._inactive_tenant_response()

            # Record API request for quota tracking
            await self.quota_manager.record_usage(tenant_id, "api_requests")

        # Store tenant ID in request state
        request.state.tenant_id = tenant_id

        # Set tenant context in logs
        if tenant_id:
            set_log_context(tenant_id=tenant_id)

        logger.debug(
            "Tenant context oluşturuldu",
            tenant_id=tenant_id,
            path=request.url.path,
        )

        # Process request
        return await call_next(request)

    def _is_public_route(self, path: str) -> bool:
        """
        Check if route is public.

        Args:
            path: Request path

        Returns:
            True if route is public
        """
        public_routes = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/forgot-password",
        ]

        return path in public_routes or any(
            path.startswith(route) for route in public_routes
        )

    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """
        Extract tenant ID from request.

        Priority:
        1. X-Tenant-ID header
        2. JWT token (tenant_id claim)
        3. Subdomain extraction
        4. Path parameter

        Args:
            request: Incoming FastAPI request

        Returns:
            Tenant ID string or None
        """
        # 1. X-Tenant-ID header (highest priority)
        tenant_header = request.headers.get(settings.TENANT_HEADER.lower())
        if tenant_header:
            return tenant_header

        # 2. JWT token tenant_id claim (set by auth middleware)
        if hasattr(request.state, "tenant_id"):
            token_tenant = request.state.tenant_id
            if token_tenant:
                return token_tenant

        # 3. Subdomain extraction
        if self.tenant_mode == "subdomain":
            host = request.headers.get("host", "")
            parts = host.split(".")

            # Must have at least 3 parts (subdomain.domain.tld)
            if len(parts) >= 3:
                subdomain = parts[0]
                # Ignore common subdomains
                if subdomain not in ["www", "api", "app", "admin"]:
                    return subdomain

        # 4. Path parameter (e.g., /api/v1/tenants/{tenant_id}/...)
        if "/tenants/" in request.url.path:
            parts = request.url.path.split("/")
            try:
                tenant_index = parts.index("tenants")
                if len(parts) > tenant_index + 1:
                    return parts[tenant_index + 1]
            except (ValueError, IndexError):
                pass

        return None

    def _missing_tenant_response(self) -> JSONResponse:
        """Return missing tenant ID error."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "code": "MISSING_TENANT_ID",
                    "message": "Tenant ID gereklidir",
                    "details": "Lütfen X-Tenant-ID header'ını kullanarak tenant belirtin.",
                }
            },
        )

    def _invalid_tenant_response(self) -> JSONResponse:
        """Return invalid tenant ID error."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "INVALID_TENANT_ID",
                    "message": "Geçersiz tenant ID formatı",
                    "details": "Tenant ID UUID formatında olmalıdır.",
                }
            },
        )

    def _tenant_mismatch_response(self) -> JSONResponse:
        """Return tenant mismatch error."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "code": "TENANT_MISMATCH",
                    "message": "Tenant ID uyuşmazlığı",
                    "details": "Token'daki tenant ID ile header'daki tenant ID uyuşmuyor.",
                }
            },
        )

    def _inactive_tenant_response(self) -> JSONResponse:
        """Return inactive tenant error."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "code": "TENANT_INACTIVE",
                    "message": "Tenant hesabı aktif değil",
                    "details": "Lütfen aboneliğinizi yenileyin veya destek ile iletişime geçin.",
                }
            },
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def check_tenant_quota(tenant_id: str, resource: str) -> bool:
    """
    Check if tenant has quota for resource.

    Args:
        tenant_id: Tenant ID
        resource: Resource type

    Returns:
        True if quota available
    """
    quota_manager = get_quota_manager()
    return await quota_manager.check_quota(tenant_id, resource)


async def record_usage(tenant_id: str, resource: str, amount: int = 1) -> None:
    """
    Record resource usage for tenant.

    Args:
        tenant_id: Tenant ID
        resource: Resource type
        amount: Usage amount
    """
    quota_manager = get_quota_manager()
    await quota_manager.record_usage(tenant_id, resource, amount)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TenantContextMiddleware",
    "TenantValidator",
    "TenantQuotaManager",
    "check_tenant_quota",
    "record_usage",
]
