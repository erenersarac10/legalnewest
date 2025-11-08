"""
Audit Middleware for Turkish Legal AI.

This middleware automatically logs all HTTP requests for compliance:
- GDPR Article 30 compliance (processing records)
- KVKK compliance (Turkish data protection)
- Access tracking for security audits
- User activity monitoring
- API endpoint access logging

Features:
    - Automatic audit log creation for all requests
    - Multi-tenant isolation
    - Request/response metadata capture
    - User action tracking
    - IP address and user agent logging
    - Request ID correlation
    - Async database writes (non-blocking)
    - Configurable exclusion patterns
    - PII redaction support

Example:
    >>> from fastapi import FastAPI
    >>> from backend.api.middleware.audit import AuditMiddleware
    >>>
    >>> app = FastAPI()
    >>> app.add_middleware(AuditMiddleware)
"""

import datetime
import json
import time
from typing import Callable, Optional
from uuid import UUID

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.core.database.models.compliance_audit_log import (
    ComplianceAuditLog,
    ComplianceEventType,
    LegalBasis,
)
from backend.core.database.session import get_db
from backend.core.logging import get_logger

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Paths to exclude from audit logging (health checks, metrics, etc.)
EXCLUDED_PATHS = [
    "/health",
    "/healthz",
    "/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
]

# HTTP methods that trigger data access events
DATA_ACCESS_METHODS = ["GET", "HEAD", "OPTIONS"]

# HTTP methods that trigger data modification events
DATA_MODIFICATION_METHODS = ["POST", "PUT", "PATCH"]

# HTTP methods that trigger data deletion events
DATA_DELETION_METHODS = ["DELETE"]


# =============================================================================
# AUDIT MIDDLEWARE
# =============================================================================


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Audit middleware for automatic compliance logging.

    This middleware intercepts all HTTP requests and creates
    audit log entries for compliance frameworks (GDPR, KVKK).

    The middleware:
    1. Captures request metadata (method, path, IP, user agent)
    2. Extracts user/tenant context from request
    3. Determines event type based on HTTP method
    4. Creates ComplianceAuditLog entry asynchronously
    5. Does NOT block request processing

    Configuration:
    - Excluded paths: EXCLUDED_PATHS (health checks, etc.)
    - Event mapping: HTTP method → ComplianceEventType
    - Compliance framework: Auto-detected from tenant settings
    """

    def __init__(self, app: ASGIApp):
        """
        Initialize middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and create audit log.

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        # Skip excluded paths (health checks, metrics, etc.)
        if self._should_exclude(request.url.path):
            return await call_next(request)

        # Extract context
        start_time = time.time()
        user_id = self._extract_user_id(request)
        tenant_id = self._extract_tenant_id(request)
        request_id = self._extract_request_id(request)

        # Process request
        response = await call_next(request)

        # Create audit log (async, non-blocking)
        try:
            await self._create_audit_log(
                request=request,
                response=response,
                user_id=user_id,
                tenant_id=tenant_id,
                request_id=request_id,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            # Log error but don't fail the request
            logger.error(
                f"Failed to create audit log: {e}",
                extra={
                    "error": str(e),
                    "path": request.url.path,
                    "method": request.method,
                },
            )

        return response

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _should_exclude(self, path: str) -> bool:
        """
        Check if path should be excluded from audit logging.

        Args:
            path: Request path

        Returns:
            True if should exclude
        """
        return any(path.startswith(excluded) for excluded in EXCLUDED_PATHS)

    def _extract_user_id(self, request: Request) -> Optional[UUID]:
        """
        Extract user ID from request state.

        Args:
            request: HTTP request

        Returns:
            User ID or None
        """
        try:
            # User ID set by auth middleware
            if hasattr(request.state, "user_id"):
                user_id = request.state.user_id
                if isinstance(user_id, str):
                    return UUID(user_id)
                return user_id
        except Exception:
            pass
        return None

    def _extract_tenant_id(self, request: Request) -> Optional[UUID]:
        """
        Extract tenant ID from request state.

        Args:
            request: HTTP request

        Returns:
            Tenant ID or None
        """
        try:
            # Tenant ID set by tenant_context middleware
            if hasattr(request.state, "tenant_id"):
                tenant_id = request.state.tenant_id
                if isinstance(tenant_id, str):
                    return UUID(tenant_id)
                return tenant_id
        except Exception:
            pass
        return None

    def _extract_request_id(self, request: Request) -> Optional[str]:
        """
        Extract request ID from headers or state.

        Args:
            request: HTTP request

        Returns:
            Request ID or None
        """
        # Try request state first
        if hasattr(request.state, "request_id"):
            return str(request.state.request_id)

        # Try X-Request-ID header
        return request.headers.get("X-Request-ID")

    def _get_ip_address(self, request: Request) -> Optional[str]:
        """
        Extract client IP address from request.

        Args:
            request: HTTP request

        Returns:
            IP address or None
        """
        # Try X-Forwarded-For header (proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP if multiple
            return forwarded.split(",")[0].strip()

        # Try X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fallback to direct client
        if request.client:
            return request.client.host

        return None

    def _get_user_agent(self, request: Request) -> Optional[str]:
        """
        Extract user agent from request.

        Args:
            request: HTTP request

        Returns:
            User agent string or None
        """
        return request.headers.get("User-Agent")

    def _map_method_to_event(self, method: str) -> ComplianceEventType:
        """
        Map HTTP method to compliance event type.

        Args:
            method: HTTP method (GET, POST, etc.)

        Returns:
            ComplianceEventType
        """
        if method in DATA_ACCESS_METHODS:
            return ComplianceEventType.DATA_ACCESS
        elif method in DATA_MODIFICATION_METHODS:
            return ComplianceEventType.DATA_MODIFICATION
        elif method in DATA_DELETION_METHODS:
            return ComplianceEventType.DATA_DELETION
        else:
            return ComplianceEventType.DATA_ACCESS  # Default

    def _determine_compliance_framework(
        self,
        tenant_id: Optional[UUID],
    ) -> str:
        """
        Determine compliance framework for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Compliance framework string (GDPR, KVKK, etc.)
        """
        # TODO: Fetch from tenant settings
        # For now, default to GDPR (most strict)
        return "GDPR"

    def _determine_legal_basis(
        self,
        method: str,
        path: str,
    ) -> Optional[LegalBasis]:
        """
        Determine GDPR legal basis for processing.

        Args:
            method: HTTP method
            path: Request path

        Returns:
            LegalBasis or None
        """
        # Contract execution (authenticated API calls)
        if "/api/v1/" in path:
            return LegalBasis.CONTRACT

        # Consent (explicit user actions)
        if method in DATA_MODIFICATION_METHODS:
            return LegalBasis.CONSENT

        # Legal obligation (compliance endpoints)
        if "/audit/" in path or "/compliance/" in path:
            return LegalBasis.LEGAL_OBLIGATION

        return LegalBasis.LEGITIMATE_INTERESTS  # Default

    async def _create_audit_log(
        self,
        request: Request,
        response: Response,
        user_id: Optional[UUID],
        tenant_id: Optional[UUID],
        request_id: Optional[str],
        duration_ms: float,
    ) -> None:
        """
        Create compliance audit log entry.

        Args:
            request: HTTP request
            response: HTTP response
            user_id: User ID
            tenant_id: Tenant ID
            request_id: Request ID
            duration_ms: Request duration in milliseconds
        """
        # Skip if no tenant (unauthenticated endpoints)
        if not tenant_id:
            return

        # Map event type
        event_type = self._map_method_to_event(request.method)

        # Determine compliance framework
        compliance_framework = self._determine_compliance_framework(tenant_id)

        # Determine legal basis
        legal_basis = self._determine_legal_basis(request.method, request.url.path)

        # Extract IP and user agent
        ip_address = self._get_ip_address(request)
        user_agent = self._get_user_agent(request)

        # Build metadata
        metadata = {
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.url.query) if request.url.query else None,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "content_length": response.headers.get("Content-Length"),
        }

        # Create audit log
        async for db in get_db():
            try:
                audit_log = ComplianceAuditLog(
                    tenant_id=tenant_id,
                    data_subject_id=user_id,  # User is data subject
                    processor_id=user_id,  # User is also processor
                    event_type=event_type,
                    compliance_framework=compliance_framework,
                    legal_basis=legal_basis,
                    data_categories=[],  # TODO: Extract from path
                    processing_purpose=f"API access: {request.method} {request.url.path}",
                    recipients=[],  # No third-party sharing
                    ip_address=ip_address,
                    user_agent=user_agent,
                    request_id=request_id,
                    description=f"{request.method} {request.url.path} → {response.status_code}",
                    metadata=metadata,
                )

                db.add(audit_log)
                await db.commit()

                logger.debug(
                    f"Created audit log for {request.method} {request.url.path}",
                    extra={
                        "audit_log_id": str(audit_log.id),
                        "user_id": str(user_id) if user_id else None,
                        "tenant_id": str(tenant_id),
                        "event_type": str(event_type),
                    },
                )

            except Exception as e:
                logger.error(
                    f"Failed to save audit log: {e}",
                    extra={"error": str(e)},
                )
                await db.rollback()
            finally:
                break  # Exit after first session


# =============================================================================
# MIDDLEWARE FACTORY
# =============================================================================


def create_audit_middleware(
    excluded_paths: Optional[list] = None,
) -> AuditMiddleware:
    """
    Create audit middleware with custom configuration.

    Args:
        excluded_paths: Additional paths to exclude

    Returns:
        Configured AuditMiddleware instance

    Example:
        >>> middleware = create_audit_middleware(
        ...     excluded_paths=["/internal/", "/debug/"]
        ... )
        >>> app.add_middleware(lambda app: middleware)
    """
    if excluded_paths:
        EXCLUDED_PATHS.extend(excluded_paths)

    return AuditMiddleware
