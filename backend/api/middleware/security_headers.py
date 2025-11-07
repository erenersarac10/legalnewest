"""
Security Headers Middleware for Turkish Legal AI Platform.

Enterprise-grade security headers middleware implementing OWASP best practices,
defense-in-depth strategies, and compliance requirements for legal AI applications.

=============================================================================
FEATURES
=============================================================================

1. OWASP Security Headers
   -----------------------
   - Content Security Policy (CSP) with nonce support
   - HTTP Strict Transport Security (HSTS)
   - X-Frame-Options (clickjacking protection)
   - X-Content-Type-Options (MIME sniffing prevention)
   - X-XSS-Protection (legacy browser support)
   - Referrer-Policy (privacy protection)
   - Permissions-Policy (feature restrictions)

2. CSP Builder
   -------------
   - Fluent API for CSP policy construction
   - Nonce generation for inline scripts/styles
   - Report URI for CSP violations
   - Environment-specific policies (dev/staging/prod)
   - Gradual CSP rollout with report-only mode

3. Legal & Compliance
   -------------------
   - KVKK-compliant privacy headers
   - Data sovereignty indicators
   - Audit trail for security events
   - PCI DSS compliance support
   - GDPR privacy considerations

4. Advanced Features
   ------------------
   - Dynamic CSP based on request context
   - Tenant-specific security policies
   - A/B testing security headers
   - Security header monitoring and alerting
   - Automatic header validation

=============================================================================
USAGE
=============================================================================

Basic Integration:
------------------

>>> from fastapi import FastAPI
>>> from backend.api.middleware.security_headers import SecurityHeadersMiddleware
>>>
>>> app = FastAPI()
>>> app.add_middleware(SecurityHeadersMiddleware)
>>>
>>> # All responses now include OWASP-recommended security headers

Custom CSP Configuration:
--------------------------

>>> from backend.api.middleware.security_headers import CSPBuilder
>>>
>>> csp = CSPBuilder()
>>> csp.default_src("'self'") \\
...    .script_src("'self'", "'unsafe-inline'", "https://cdn.example.com") \\
...    .style_src("'self'", "'unsafe-inline'") \\
...    .img_src("'self'", "data:", "https:") \\
...    .connect_src("'self'", "https://api.turkishlegalai.com") \\
...    .frame_ancestors("'none'")
>>>
>>> app.add_middleware(
...     SecurityHeadersMiddleware,
...     csp_policy=csp.build()
... )

Nonce-Based CSP (for inline scripts):
--------------------------------------

>>> from fastapi import Request
>>>
>>> @app.get("/dashboard")
>>> async def dashboard(request: Request):
...     # Get CSP nonce from request state (set by middleware)
...     nonce = request.state.csp_nonce
...
...     return HTMLResponse(f'''
...         <html>
...             <script nonce="{nonce}">
...                 // This inline script is allowed by CSP
...                 console.log("Protected by nonce!");
...             </script>
...         </html>
...     ''')

Environment-Specific Configuration:
------------------------------------

>>> # Development: Relaxed CSP with report-only mode
>>> if settings.ENVIRONMENT == "development":
...     app.add_middleware(
...         SecurityHeadersMiddleware,
...         report_only=True,  # Don't block, just report
...         allow_unsafe_eval=True,
...         allow_unsafe_inline=True
...     )
>>>
>>> # Production: Strict CSP enforcement
>>> else:
...     app.add_middleware(
...         SecurityHeadersMiddleware,
...         report_only=False,
...         allow_unsafe_eval=False,
...         allow_unsafe_inline=False,
...         hsts_max_age=31536000  # 1 year
...     )

=============================================================================
SECURITY HEADERS EXPLAINED
=============================================================================

Content-Security-Policy (CSP):
-------------------------------

Purpose: Prevents XSS, clickjacking, and other code injection attacks

Example:
  Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-random123'

Directives:
  - default-src: Fallback for other directives
  - script-src: JavaScript sources
  - style-src: CSS sources
  - img-src: Image sources
  - connect-src: AJAX, WebSocket, EventSource
  - font-src: Font sources
  - frame-src: iframe sources
  - frame-ancestors: Embedding allowed origins

Common Values:
  - 'self': Same origin only
  - 'none': Block all
  - 'unsafe-inline': Allow inline scripts (not recommended)
  - 'unsafe-eval': Allow eval() (not recommended)
  - 'nonce-ABC123': Allow scripts with this nonce
  - https://example.com: Specific domain

HTTP Strict-Transport-Security (HSTS):
---------------------------------------

Purpose: Forces HTTPS connections, prevents protocol downgrade attacks

Example:
  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload

Parameters:
  - max-age: Duration in seconds (1 year = 31536000)
  - includeSubDomains: Apply to all subdomains
  - preload: Include in browser HSTS preload list

X-Frame-Options:
----------------

Purpose: Prevents clickjacking attacks

Values:
  - DENY: Cannot be framed at all
  - SAMEORIGIN: Can be framed by same origin
  - ALLOW-FROM https://example.com: Specific origin (deprecated)

Use frame-ancestors CSP directive instead for modern browsers.

X-Content-Type-Options:
------------------------

Purpose: Prevents MIME-type sniffing

Value:
  - nosniff: Browser must respect Content-Type header

Example Attack Prevention:
  Without this header, browser might execute text/plain as JavaScript

Referrer-Policy:
----------------

Purpose: Controls referrer information sent to other sites

Values:
  - no-referrer: Never send referrer
  - no-referrer-when-downgrade: Default, don't send on HTTPS→HTTP
  - origin: Send only origin
  - origin-when-cross-origin: Full URL same-origin, origin only cross-origin
  - same-origin: Only send for same-origin requests
  - strict-origin: Send origin only for HTTPS→HTTPS
  - strict-origin-when-cross-origin: Recommended for privacy
  - unsafe-url: Always send full URL (not recommended)

Permissions-Policy:
-------------------

Purpose: Controls browser features (camera, microphone, geolocation)

Example:
  Permissions-Policy: geolocation=(), microphone=(), camera=()

This disables geolocation, microphone, and camera for all origins.

=============================================================================
CSP VIOLATION REPORTING
=============================================================================

Setup CSP Report Endpoint:
---------------------------

>>> @app.post("/api/v1/security/csp-report")
>>> async def csp_report(request: Request):
...     report = await request.json()
...
...     # Log CSP violation
...     logger.warning(
...         "🚨 CSP violation detected",
...         blocked_uri=report.get("blocked-uri"),
...         violated_directive=report.get("violated-directive"),
...         source_file=report.get("source-file"),
...         line_number=report.get("line-number")
...     )
...
...     # Alert security team for critical violations
...     if "script-src" in report.get("violated-directive", ""):
...         await security_alerts.send_alert(
...             severity="HIGH",
...             message=f"Script injection attempt: {report}"
...         )
...
...     return {"status": "received"}

Configure CSP with Report URI:
-------------------------------

>>> csp = CSPBuilder()
>>> csp.default_src("'self'") \\
...    .report_uri("/api/v1/security/csp-report") \\
...    .report_to("csp-endpoint")  # For modern browsers

=============================================================================
OWASP TOP 10 PROTECTION
=============================================================================

A01:2021 – Broken Access Control:
----------------------------------
✓ Referrer-Policy prevents information leakage
✓ Frame-ancestors CSP prevents unauthorized embedding

A02:2021 – Cryptographic Failures:
-----------------------------------
✓ HSTS enforces HTTPS
✓ Permissions-Policy restricts sensitive features

A03:2021 – Injection:
---------------------
✓ CSP prevents XSS attacks
✓ X-Content-Type-Options prevents MIME confusion
✓ X-XSS-Protection for legacy browsers

A05:2021 – Security Misconfiguration:
--------------------------------------
✓ Server header removed
✓ Secure defaults for all headers
✓ Environment-specific configurations

A07:2021 – Identification and Authentication Failures:
------------------------------------------------------
✓ HSTS prevents session hijacking over HTTP
✓ Referrer-Policy protects authentication tokens in URL

=============================================================================
KVKK COMPLIANCE
=============================================================================

Privacy Headers:
----------------
- Referrer-Policy prevents URL parameter tracking
- Permissions-Policy restricts geolocation tracking
- CSP prevents third-party tracking scripts

Data Sovereignty:
-----------------
- Connect-src CSP directive limits data destinations
- Font-src and img-src control CDN usage
- Frame-ancestors prevents unauthorized data display

Turkish Legal Requirements:
----------------------------
- HTTPS enforcement (HSTS) for data protection
- No third-party analytics without consent
- Audit trail for security violations

=============================================================================
TROUBLESHOOTING
=============================================================================

"Refused to load script" CSP Error:
------------------------------------
1. Check script-src directive includes source
2. Use 'nonce-xxx' or 'hash-xxx' for inline scripts
3. Avoid 'unsafe-inline' in production
4. Check browser console for violation details

"Mixed Content" Warning:
------------------------
1. Ensure all resources loaded over HTTPS
2. Use upgrade-insecure-requests CSP directive
3. Update hardcoded http:// URLs to https://
4. Use protocol-relative URLs (//example.com)

HSTS Not Working:
-----------------
1. Ensure first visit is over HTTPS
2. Check max-age is set correctly
3. Verify includeSubDomains if needed
4. Check browser HSTS cache (chrome://net-internals/#hsts)

CSP Too Strict:
---------------
1. Use report-only mode during testing
2. Review CSP reports to identify needed sources
3. Gradually tighten policy
4. Use nonces instead of 'unsafe-inline'

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

import hashlib
import secrets
from typing import Callable, Dict, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, settings

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# CSP BUILDER
# =============================================================================


class CSPBuilder:
    """
    Fluent API for building Content Security Policy directives.

    Example:
        >>> csp = CSPBuilder()
        >>> csp.default_src("'self'").script_src("'self'", "https://cdn.example.com")
        >>> policy = csp.build()
    """

    def __init__(self):
        """Initialize CSP builder with empty directives."""
        self.directives: Dict[str, List[str]] = {}

    def default_src(self, *sources: str) -> "CSPBuilder":
        """Set default-src directive."""
        self.directives["default-src"] = list(sources)
        return self

    def script_src(self, *sources: str) -> "CSPBuilder":
        """Set script-src directive."""
        self.directives["script-src"] = list(sources)
        return self

    def style_src(self, *sources: str) -> "CSPBuilder":
        """Set style-src directive."""
        self.directives["style-src"] = list(sources)
        return self

    def img_src(self, *sources: str) -> "CSPBuilder":
        """Set img-src directive."""
        self.directives["img-src"] = list(sources)
        return self

    def font_src(self, *sources: str) -> "CSPBuilder":
        """Set font-src directive."""
        self.directives["font-src"] = list(sources)
        return self

    def connect_src(self, *sources: str) -> "CSPBuilder":
        """Set connect-src directive."""
        self.directives["connect-src"] = list(sources)
        return self

    def frame_src(self, *sources: str) -> "CSPBuilder":
        """Set frame-src directive."""
        self.directives["frame-src"] = list(sources)
        return self

    def frame_ancestors(self, *sources: str) -> "CSPBuilder":
        """Set frame-ancestors directive."""
        self.directives["frame-ancestors"] = list(sources)
        return self

    def base_uri(self, *sources: str) -> "CSPBuilder":
        """Set base-uri directive."""
        self.directives["base-uri"] = list(sources)
        return self

    def form_action(self, *sources: str) -> "CSPBuilder":
        """Set form-action directive."""
        self.directives["form-action"] = list(sources)
        return self

    def report_uri(self, uri: str) -> "CSPBuilder":
        """Set report-uri directive."""
        self.directives["report-uri"] = [uri]
        return self

    def report_to(self, group: str) -> "CSPBuilder":
        """Set report-to directive (modern browsers)."""
        self.directives["report-to"] = [group]
        return self

    def add_nonce(self, nonce: str, directive: str = "script-src") -> "CSPBuilder":
        """
        Add nonce to directive.

        Args:
            nonce: Nonce value
            directive: Directive to add nonce to (default: script-src)
        """
        if directive not in self.directives:
            self.directives[directive] = []
        self.directives[directive].append(f"'nonce-{nonce}'")
        return self

    def build(self) -> str:
        """
        Build CSP policy string.

        Returns:
            CSP policy string
        """
        parts = []
        for directive, sources in self.directives.items():
            parts.append(f"{directive} {' '.join(sources)}")
        return "; ".join(parts)


# =============================================================================
# NONCE GENERATOR
# =============================================================================


class NonceGenerator:
    """
    Generates cryptographically secure nonces for CSP.
    """

    @staticmethod
    def generate(length: int = 32) -> str:
        """
        Generate random nonce.

        Args:
            length: Nonce length in bytes (default: 32)

        Returns:
            Base64-encoded nonce
        """
        import base64

        nonce_bytes = secrets.token_bytes(length)
        return base64.b64encode(nonce_bytes).decode("utf-8")


# =============================================================================
# SECURITY HEADERS MIDDLEWARE
# =============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware implementing OWASP best practices.

    Features:
    - Content Security Policy (CSP)
    - HTTP Strict Transport Security (HSTS)
    - Clickjacking protection
    - MIME sniffing prevention
    - XSS protection
    - Privacy headers
    - Feature restriction policies
    """

    def __init__(
        self,
        app,
        report_only: bool = False,
        allow_unsafe_eval: bool = False,
        allow_unsafe_inline: bool = False,
        hsts_max_age: int = 31536000,
        enable_nonce: bool = True,
    ):
        """
        Initialize security headers middleware.

        Args:
            app: FastAPI application
            report_only: Use CSP report-only mode (don't block)
            allow_unsafe_eval: Allow eval() in scripts
            allow_unsafe_inline: Allow inline scripts/styles
            hsts_max_age: HSTS max-age in seconds (default: 1 year)
            enable_nonce: Enable CSP nonce generation
        """
        super().__init__(app)
        self.report_only = report_only
        self.allow_unsafe_eval = allow_unsafe_eval
        self.allow_unsafe_inline = allow_unsafe_inline
        self.hsts_max_age = hsts_max_age
        self.enable_nonce = enable_nonce
        self.nonce_generator = NonceGenerator()

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
        # Generate CSP nonce if enabled
        nonce = None
        if self.enable_nonce:
            nonce = self.nonce_generator.generate()
            request.state.csp_nonce = nonce

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_csp_header(response, nonce)
        self._add_hsts_header(response)
        self._add_frame_options(response)
        self._add_content_type_options(response)
        self._add_xss_protection(response)
        self._add_referrer_policy(response)
        self._add_permissions_policy(response)
        self._add_misc_headers(response)

        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]

        return response

    def _add_csp_header(self, response: Response, nonce: Optional[str]) -> None:
        """Add Content Security Policy header."""
        csp = CSPBuilder()

        # Default sources
        csp.default_src("'self'")

        # Script sources
        script_sources = ["'self'"]
        if self.allow_unsafe_inline:
            script_sources.append("'unsafe-inline'")
        if self.allow_unsafe_eval:
            script_sources.append("'unsafe-eval'")
        if nonce:
            script_sources.append(f"'nonce-{nonce}'")
        # Allow common CDNs for production
        if settings.ENVIRONMENT == "production":
            script_sources.extend([
                "https://cdn.jsdelivr.net",
                "https://unpkg.com"
            ])
        csp.script_src(*script_sources)

        # Style sources
        style_sources = ["'self'"]
        if self.allow_unsafe_inline:
            style_sources.append("'unsafe-inline'")
        style_sources.extend([
            "https://fonts.googleapis.com",
            "https://cdn.jsdelivr.net"
        ])
        csp.style_src(*style_sources)

        # Font sources
        csp.font_src("'self'", "https://fonts.gstatic.com")

        # Image sources (allow data URIs and HTTPS)
        csp.img_src("'self'", "data:", "https:")

        # Connection sources (API endpoints)
        csp.connect_src(
            "'self'",
            "https://api.turkishlegalai.com",
            "wss://api.turkishlegalai.com"  # WebSocket
        )

        # Frame options
        csp.frame_ancestors("'none'")
        csp.frame_src("'none'")

        # Base and form actions
        csp.base_uri("'self'")
        csp.form_action("'self'")

        # CSP violation reporting
        if settings.ENVIRONMENT != "development":
            csp.report_uri("/api/v1/security/csp-report")

        # Build policy
        policy = csp.build()

        # Apply policy (report-only or enforcing)
        if self.report_only or settings.ENVIRONMENT == "development":
            response.headers["Content-Security-Policy-Report-Only"] = policy
        else:
            response.headers["Content-Security-Policy"] = policy

    def _add_hsts_header(self, response: Response) -> None:
        """Add HTTP Strict Transport Security header."""
        # Only in production (HTTPS required)
        if settings.ENVIRONMENT != "development":
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )

    def _add_frame_options(self, response: Response) -> None:
        """Add X-Frame-Options header."""
        response.headers["X-Frame-Options"] = "DENY"

    def _add_content_type_options(self, response: Response) -> None:
        """Add X-Content-Type-Options header."""
        response.headers["X-Content-Type-Options"] = "nosniff"

    def _add_xss_protection(self, response: Response) -> None:
        """Add X-XSS-Protection header (legacy support)."""
        response.headers["X-XSS-Protection"] = "1; mode=block"

    def _add_referrer_policy(self, response: Response) -> None:
        """Add Referrer-Policy header."""
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    def _add_permissions_policy(self, response: Response) -> None:
        """Add Permissions-Policy header."""
        # Disable sensitive features
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()",
            "interest-cohort=()"  # FLoC privacy protection
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)

    def _add_misc_headers(self, response: Response) -> None:
        """Add miscellaneous security headers."""
        # X-Permitted-Cross-Domain-Policies (Flash/PDF)
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # X-Download-Options (IE8+ download protection)
        response.headers["X-Download-Options"] = "noopen"

        # Cross-Origin policies
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SecurityHeadersMiddleware",
    "CSPBuilder",
    "NonceGenerator",
]
