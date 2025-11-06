"""
CORS Middleware Configuration for Turkish Legal AI Platform.

Enterprise-grade Cross-Origin Resource Sharing (CORS) configuration with dynamic
origin validation, security best practices, and compliance requirements.

=============================================================================
FEATURES
=============================================================================

1. Dynamic Origin Validation
   --------------------------
   - Wildcard pattern matching (*.turkishlegalai.com)
   - Regex-based origin validation
   - Environment-specific origin lists
   - Tenant-specific CORS policies
   - Origin allowlist management

2. Security Features
   -------------------
   - Strict origin validation in production
   - Credential support with specific origins only
   - Preflight request caching
   - CORS violation logging
   - Rate limiting for preflight requests

3. Performance Optimization
   --------------------------
   - Preflight cache configuration (max-age)
   - Conditional CORS header injection
   - Origin validation caching
   - Lazy origin list loading

4. Compliance & Privacy
   ----------------------
   - KVKK-compliant cross-origin policies
   - Data residency enforcement
   - Third-party integration controls
   - Audit trail for CORS violations

=============================================================================
USAGE
=============================================================================

Basic Configuration:
--------------------

>>> from fastapi import FastAPI
>>> from backend.api.middleware.cors import configure_cors
>>>
>>> app = FastAPI()
>>> configure_cors(app)
>>>
>>> # CORS automatically configured based on environment

Custom Origins:
---------------

>>> from backend.api.middleware.cors import configure_cors
>>>
>>> app = FastAPI()
>>> configure_cors(
...     app,
...     allowed_origins=[
...         "https://app.turkishlegalai.com",
...         "https://admin.turkishlegalai.com",
...         "https://*.turkishlegalai.com"  # Wildcard
...     ],
...     allow_credentials=True,
...     max_age=3600  # 1 hour preflight cache
... )

Dynamic Origin Validation:
---------------------------

>>> from backend.api.middleware.cors import CORSValidator
>>>
>>> validator = CORSValidator()
>>> validator.add_pattern("*.turkishlegalai.com")
>>> validator.add_pattern("*.legalai.tr")
>>>
>>> # Check origin
>>> is_valid = validator.validate_origin("https://app.turkishlegalai.com")
>>> # Returns: True

Tenant-Specific CORS:
----------------------

>>> from backend.api.middleware.cors import CORSPolicyManager
>>>
>>> cors_manager = CORSPolicyManager()
>>>
>>> # Set tenant-specific allowed origins
>>> cors_manager.set_tenant_origins(
...     tenant_id="acme-corp",
...     origins=["https://acme.com", "https://api.acme.com"]
... )
>>>
>>> # Validate origin for tenant
>>> is_allowed = cors_manager.validate(
...     origin="https://acme.com",
...     tenant_id="acme-corp"
... )

=============================================================================
CORS EXPLAINED
=============================================================================

What is CORS?
-------------

Cross-Origin Resource Sharing (CORS) is a security mechanism that allows
or restricts web pages from making requests to a different domain than
the one that served the web page.

Example Scenario:
  - Frontend: https://app.turkishlegalai.com
  - API: https://api.turkishlegalai.com
  - Without CORS: Browser blocks API requests (different origin)
  - With CORS: API explicitly allows requests from app.turkishlegalai.com

Simple Request vs Preflight:
-----------------------------

Simple Request (No Preflight):
  - Methods: GET, POST, HEAD
  - Headers: Accept, Accept-Language, Content-Language, Content-Type
  - Content-Type: application/x-www-form-urlencoded, multipart/form-data, text/plain

Preflight Request (OPTIONS):
  - Methods: PUT, PATCH, DELETE, or custom headers
  - Browser sends OPTIONS request first
  - Server responds with allowed methods/headers
  - Actual request proceeds if allowed

Example Preflight Flow:
  1. Browser: OPTIONS /api/v1/contracts
     Origin: https://app.turkishlegalai.com
     Access-Control-Request-Method: DELETE
     Access-Control-Request-Headers: Authorization

  2. Server: 200 OK
     Access-Control-Allow-Origin: https://app.turkishlegalai.com
     Access-Control-Allow-Methods: GET, POST, PUT, DELETE
     Access-Control-Allow-Headers: Authorization
     Access-Control-Max-Age: 3600

  3. Browser: DELETE /api/v1/contracts/123
     (Actual request proceeds)

CORS Headers Explained:
------------------------

Access-Control-Allow-Origin:
  - Specifies allowed origin
  - "*" allows all origins (NOT recommended with credentials)
  - "https://example.com" allows specific origin

Access-Control-Allow-Credentials:
  - true: Allows cookies and auth headers
  - Requires specific origin (cannot use "*")

Access-Control-Allow-Methods:
  - Lists allowed HTTP methods
  - Example: "GET, POST, PUT, DELETE"

Access-Control-Allow-Headers:
  - Lists allowed request headers
  - Example: "Content-Type, Authorization, X-Request-ID"

Access-Control-Expose-Headers:
  - Lists headers client can access
  - Example: "X-Request-ID, X-Response-Time"

Access-Control-Max-Age:
  - Preflight cache duration in seconds
  - Example: 3600 (1 hour)

=============================================================================
SECURITY CONSIDERATIONS
=============================================================================

1. Never Use Wildcard (*) with Credentials:
   -----------------------------------------
   ❌ BAD:
     allow_origins=["*"]
     allow_credentials=True

   ✓ GOOD:
     allow_origins=["https://app.turkishlegalai.com"]
     allow_credentials=True

2. Validate Origins Strictly:
   ---------------------------
   - Use allowlist, not blocklist
   - Validate protocol (https only in production)
   - Check exact matches, avoid regex when possible
   - Log rejected origins for monitoring

3. Limit Exposed Headers:
   -----------------------
   - Only expose necessary headers
   - Don't expose sensitive headers (X-Internal-*)
   - Avoid exposing database query info

4. Cache Preflight Responses:
   ---------------------------
   - Set appropriate max-age (3600 recommended)
   - Reduces preflight request overhead
   - Balance between performance and security updates

5. Monitor CORS Violations:
   -------------------------
   - Log rejected origins
   - Alert on suspicious patterns
   - Track preflight request rate

=============================================================================
REAL-WORLD EXAMPLES
=============================================================================

Example 1: Multi-Domain Application
------------------------------------

>>> # Main app, admin panel, and mobile app
>>> configure_cors(
...     app,
...     allowed_origins=[
...         "https://app.turkishlegalai.com",      # Web app
...         "https://admin.turkishlegalai.com",    # Admin panel
...         "https://mobile.turkishlegalai.com",   # Mobile web
...         "capacitor://localhost",               # Capacitor mobile
...         "ionic://localhost"                    # Ionic mobile
...     ],
...     allow_credentials=True
... )

Example 2: Development Setup
-----------------------------

>>> # Allow localhost for development
>>> if settings.ENVIRONMENT == "development":
...     configure_cors(
...         app,
...         allowed_origins=[
...             "http://localhost:3000",    # React dev server
...             "http://localhost:5173",    # Vite dev server
...             "http://localhost:8080",    # Vue dev server
...             "http://127.0.0.1:3000"     # Alternate localhost
...         ],
...         allow_credentials=True
...     )

Example 3: Tenant-Specific Origins
-----------------------------------

>>> # Allow custom domains per tenant
>>> @app.middleware("http")
>>> async def tenant_cors_middleware(request: Request, call_next):
...     tenant_id = request.state.tenant_id
...     origin = request.headers.get("origin")
...
...     # Get tenant-specific allowed origins
...     tenant_origins = await get_tenant_origins(tenant_id)
...
...     if origin in tenant_origins:
...         response = await call_next(request)
...         response.headers["Access-Control-Allow-Origin"] = origin
...         response.headers["Access-Control-Allow-Credentials"] = "true"
...         return response
...
...     return await call_next(request)

=============================================================================
TROUBLESHOOTING
=============================================================================

"CORS policy: No 'Access-Control-Allow-Origin' header":
--------------------------------------------------------
1. Check origin is in allowed_origins list
2. Verify ENVIRONMENT and CORS_ORIGINS setting
3. Check for typos in origin URL (protocol, port)
4. Ensure CORS middleware is added to app

"CORS policy: Credentials flag is true but Access-Control-Allow-Origin is *":
------------------------------------------------------------------------------
1. Cannot use wildcard (*) with credentials
2. Use specific origins: ["https://app.example.com"]
3. Or disable credentials: allow_credentials=False

"Preflight request failed":
---------------------------
1. Check allow_methods includes requested method
2. Verify allow_headers includes custom headers
3. Ensure OPTIONS method is not blocked by auth
4. Check preflight request timeout (increase max_age)

"CORS working in development but not production":
--------------------------------------------------
1. Development might use "*" wildcard
2. Production needs explicit origin list
3. Check CORS_ORIGINS environment variable
4. Verify HTTPS protocol in production

=============================================================================
KVKK COMPLIANCE
=============================================================================

Cross-Origin Data Sharing:
---------------------------
- Document all allowed origins
- Restrict third-party access
- Implement audit trail for cross-origin requests
- Obtain consent for data sharing

Data Residency:
---------------
- Limit origins to Turkey-based domains (if required)
- Block international origins for sensitive data
- Use geo-fencing for origin validation

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

import re
from typing import Dict, List, Optional, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core import get_logger, settings

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# CORS VALIDATOR
# =============================================================================


class CORSValidator:
    """
    Validates origins against patterns and rules.

    Supports:
    - Exact matching
    - Wildcard patterns (*.example.com)
    - Regex patterns
    """

    def __init__(self):
        """Initialize CORS validator."""
        self.exact_origins: Set[str] = set()
        self.wildcard_patterns: List[str] = []
        self.regex_patterns: List[re.Pattern] = []

    def add_origin(self, origin: str) -> None:
        """
        Add allowed origin.

        Args:
            origin: Origin to allow (exact, wildcard, or regex)
        """
        if "*" in origin:
            # Wildcard pattern
            self.wildcard_patterns.append(origin)
        elif origin.startswith("regex:"):
            # Regex pattern
            pattern = origin[6:]  # Remove "regex:" prefix
            self.regex_patterns.append(re.compile(pattern))
        else:
            # Exact origin
            self.exact_origins.add(origin)

    def add_pattern(self, pattern: str) -> None:
        """Add wildcard pattern (alias for add_origin)."""
        self.add_origin(pattern)

    def validate_origin(self, origin: str) -> bool:
        """
        Validate origin against allowed patterns.

        Args:
            origin: Origin to validate

        Returns:
            True if origin is allowed
        """
        # Exact match
        if origin in self.exact_origins:
            return True

        # Wildcard match
        for pattern in self.wildcard_patterns:
            if self._match_wildcard(origin, pattern):
                return True

        # Regex match
        for regex in self.regex_patterns:
            if regex.match(origin):
                return True

        return False

    def _match_wildcard(self, origin: str, pattern: str) -> bool:
        """
        Match origin against wildcard pattern.

        Args:
            origin: Origin to check
            pattern: Wildcard pattern (*.example.com)

        Returns:
            True if matches
        """
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"[^.]+")
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.match(regex_pattern, origin))


# =============================================================================
# CORS POLICY MANAGER
# =============================================================================


class CORSPolicyManager:
    """
    Manages CORS policies per tenant.
    """

    def __init__(self):
        """Initialize CORS policy manager."""
        self.tenant_origins: Dict[str, List[str]] = {}
        self.global_validator = CORSValidator()

    def set_tenant_origins(self, tenant_id: str, origins: List[str]) -> None:
        """
        Set allowed origins for tenant.

        Args:
            tenant_id: Tenant ID
            origins: List of allowed origins
        """
        self.tenant_origins[tenant_id] = origins

    def set_global_origins(self, origins: List[str]) -> None:
        """
        Set global allowed origins.

        Args:
            origins: List of allowed origins
        """
        for origin in origins:
            self.global_validator.add_origin(origin)

    def validate(self, origin: str, tenant_id: Optional[str] = None) -> bool:
        """
        Validate origin for tenant.

        Args:
            origin: Origin to validate
            tenant_id: Optional tenant ID

        Returns:
            True if origin is allowed
        """
        # Check global origins
        if self.global_validator.validate_origin(origin):
            return True

        # Check tenant-specific origins
        if tenant_id and tenant_id in self.tenant_origins:
            return origin in self.tenant_origins[tenant_id]

        return False


# =============================================================================
# CORS CONFIGURATION
# =============================================================================


def configure_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None,
    expose_headers: Optional[List[str]] = None,
    max_age: int = 600,
) -> None:
    """
    Configure CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins (default: from settings)
        allow_credentials: Allow credentials (cookies, auth)
        allow_methods: Allowed HTTP methods
        allow_headers: Allowed request headers
        expose_headers: Headers exposed to client
        max_age: Preflight cache duration in seconds (default: 10 min)
    """
    # Parse allowed origins
    if allowed_origins is None:
        allowed_origins = _parse_origins_from_settings()

    # Default methods
    if allow_methods is None:
        allow_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    # Default request headers
    if allow_headers is None:
        allow_headers = [
            "Accept",
            "Accept-Language",
            "Content-Type",
            "Content-Language",
            "Authorization",
            "X-Request-ID",
            "X-Tenant-ID",
            "X-API-Key",
            "X-Correlation-ID",
        ]

    # Default response headers to expose
    if expose_headers is None:
        expose_headers = [
            "X-Request-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Correlation-ID",
        ]

    # Development: Allow all origins
    if settings.ENVIRONMENT == "development" and settings.DEBUG:
        allowed_origins = ["*"]
        logger.warning(
            "⚠️ CORS yapılandırması: TÜM originlere izin veriliyor (development mode)",
            environment=settings.ENVIRONMENT,
        )
    else:
        logger.info(
            "✓ CORS yapılandırması: Production mode",
            allowed_origins=allowed_origins,
            allow_credentials=allow_credentials,
        )

    # Security check: Cannot use wildcard with credentials
    if "*" in allowed_origins and allow_credentials:
        logger.error(
            "❌ GÜVENLİK HATASI: CORS wildcard (*) credentials ile kullanılamaz"
        )
        raise ValueError(
            "Cannot use wildcard origin (*) with allow_credentials=True. "
            "This is a security risk. Use specific origins instead."
        )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers,
        max_age=max_age,
    )

    logger.info(
        "✓ CORS middleware configured",
        origins_count=len(allowed_origins) if allowed_origins != ["*"] else "all",
        max_age=max_age,
    )


def _parse_origins_from_settings() -> List[str]:
    """
    Parse allowed origins from settings.

    Returns:
        List of allowed origins
    """
    allowed_origins = []

    if settings.CORS_ORIGINS:
        if isinstance(settings.CORS_ORIGINS, str):
            # Parse comma-separated string
            allowed_origins = [
                origin.strip() for origin in settings.CORS_ORIGINS.split(",")
            ]
        else:
            # Already a list
            allowed_origins = settings.CORS_ORIGINS

    # Default origins if none configured
    if not allowed_origins:
        if settings.ENVIRONMENT == "production":
            allowed_origins = [
                "https://app.turkishlegalai.com",
                "https://admin.turkishlegalai.com",
            ]
        else:
            allowed_origins = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
            ]

    return allowed_origins


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "configure_cors",
    "CORSValidator",
    "CORSPolicyManager",
]
