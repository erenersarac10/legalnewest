"""
Security Headers Middleware for Turkish Legal AI Platform.

Adds security headers to all responses for defense-in-depth.

Features:
- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- X-Frame-Options (clickjacking protection)
- X-Content-Type-Options (MIME sniffing prevention)
- Referrer-Policy
- Permissions-Policy

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, settings

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware for enhanced protection.

    Implements OWASP security header recommendations.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request and add security headers to response.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response with security headers
        """
        # Process request
        response = await call_next(request)

        # =====================================================================
        # Content Security Policy (CSP)
        # =====================================================================
        # Restricts resource loading to prevent XSS attacks
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' https://api.turkishlegalai.com",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]

        # Relaxed CSP for development
        if settings.ENVIRONMENT == "development":
            csp_directives.append("upgrade-insecure-requests")
            response.headers["Content-Security-Policy-Report-Only"] = "; ".join(
                csp_directives
            )
        else:
            response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # =====================================================================
        # HTTP Strict Transport Security (HSTS)
        # =====================================================================
        # Forces HTTPS connections for 1 year
        if settings.ENVIRONMENT != "development":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # =====================================================================
        # X-Frame-Options
        # =====================================================================
        # Prevents clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"

        # =====================================================================
        # X-Content-Type-Options
        # =====================================================================
        # Prevents MIME-type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # =====================================================================
        # X-XSS-Protection (legacy, but still useful)
        # =====================================================================
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # =====================================================================
        # Referrer-Policy
        # =====================================================================
        # Controls referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # =====================================================================
        # Permissions-Policy (formerly Feature-Policy)
        # =====================================================================
        # Restricts browser features
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()",
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)

        # =====================================================================
        # X-Permitted-Cross-Domain-Policies
        # =====================================================================
        # Restricts Adobe Flash/PDF cross-domain requests
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # =====================================================================
        # Server Header Removal (hide server technology)
        # =====================================================================
        if "server" in response.headers:
            del response.headers["server"]

        return response


__all__ = ["SecurityHeadersMiddleware"]
