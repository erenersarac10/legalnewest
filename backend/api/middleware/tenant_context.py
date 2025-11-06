"""
Tenant Context Middleware for Turkish Legal AI Platform.

Multi-tenancy support with Row-Level Security (RLS) context.

Features:
- Extracts tenant ID from request
- Sets PostgreSQL RLS context
- Validates tenant permissions
- Enforces tenant isolation

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, set_log_context, settings

logger = get_logger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Multi-tenant context middleware.

    Extracts and validates tenant ID from:
    1. X-Tenant-ID header
    2. JWT token payload (tenant_id claim)
    3. Subdomain (e.g., acme.turkishlegalai.com)

    Sets tenant context for:
    - Database RLS (Row-Level Security)
    - Cache namespacing
    - Log correlation
    """

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

        # Multi-tenancy is optional in settings
        if settings.MULTI_TENANT_ENABLED:
            if not tenant_id:
                logger.warning(
                    "Missing tenant ID in multi-tenant mode",
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": {
                            "code": "MISSING_TENANT_ID",
                            "message": "Tenant ID is required",
                            "details": "Lütfen X-Tenant-ID header'ını veya subdomain kullanarak tenant belirtin.",
                        }
                    },
                )

            # Validate tenant ID matches token (if user is authenticated)
            if hasattr(request.state, "tenant_id"):
                token_tenant_id = request.state.tenant_id

                if token_tenant_id and token_tenant_id != tenant_id:
                    logger.error(
                        "Tenant ID mismatch",
                        header_tenant=tenant_id,
                        token_tenant=token_tenant_id,
                        user_id=getattr(request.state, "user_id", None),
                    )
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": {
                                "code": "TENANT_MISMATCH",
                                "message": "Tenant ID mismatch",
                                "details": "Token'daki tenant ID ile header'daki tenant ID uyuşmuyor.",
                            }
                        },
                    )

        # Store tenant ID in request state
        request.state.tenant_id = tenant_id

        # Set tenant context in logs
        if tenant_id:
            set_log_context(tenant_id=tenant_id)

        # Set tenant context for database RLS
        # This will be used in database session to execute:
        # SET LOCAL app.tenant_id = 'tenant-uuid'
        # (Implemented in database session context)

        logger.debug(
            "Tenant context established",
            tenant_id=tenant_id,
            path=request.url.path,
        )

        # Process request
        return await call_next(request)

    def _is_public_route(self, path: str) -> bool:
        """Check if route is public."""
        public_routes = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
        ]

        return path in public_routes or any(
            path.startswith(route) for route in public_routes
        )

    def _extract_tenant_id(self, request: Request) -> str | None:
        """
        Extract tenant ID from request.

        Priority:
        1. X-Tenant-ID header
        2. JWT token (tenant_id claim)
        3. Subdomain extraction

        Args:
            request: Incoming FastAPI request

        Returns:
            Tenant ID string or None
        """
        # 1. X-Tenant-ID header
        tenant_header = request.headers.get(settings.TENANT_HEADER.lower())
        if tenant_header:
            return tenant_header

        # 2. JWT token tenant_id claim (set by auth middleware)
        if hasattr(request.state, "tenant_id"):
            token_tenant = request.state.tenant_id
            if token_tenant:
                return token_tenant

        # 3. Subdomain extraction (e.g., acme.turkishlegalai.com -> acme)
        if settings.TENANT_MODE == "subdomain":
            host = request.headers.get("host", "")
            parts = host.split(".")

            # Must have at least 3 parts (subdomain.domain.tld)
            if len(parts) >= 3:
                subdomain = parts[0]
                # Ignore www, api, app subdomains
                if subdomain not in ["www", "api", "app"]:
                    return subdomain

        return None


__all__ = ["TenantContextMiddleware"]
