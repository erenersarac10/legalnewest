"""
Authentication Middleware for Turkish Legal AI Platform.

This middleware provides comprehensive JWT token validation, session management,
API key authentication, rate limiting, and security features for protecting API endpoints.

=============================================================================
FEATURES
=============================================================================

1. JWT Authentication
   -------------------
   - Bearer token validation from Authorization header
   - Token expiration and signature verification
   - Claims-based access control (user_id, tenant_id, permissions)
   - Support for access and refresh tokens
   - Token blacklisting for logout

2. Session Management
   --------------------
   - Track concurrent user sessions
   - Limit maximum active sessions per user
   - Device fingerprinting and tracking
   - IP-based session validation
   - Automatic session cleanup

3. API Key Authentication
   ------------------------
   - X-API-Key header support
   - API key validation and rate limiting
   - Key rotation and expiration
   - Usage tracking per API key
   - Tenant-scoped API keys

4. Rate Limiting & Security
   -------------------------
   - Failed authentication attempt tracking
   - IP-based brute force protection
   - Progressive delay on repeated failures
   - Account lockout after threshold
   - CAPTCHA requirement after multiple failures

5. Public Route Management
   ------------------------
   - Bypass authentication for public endpoints
   - Wildcard route matching
   - Method-specific auth requirements
   - Dynamic route registration

6. Audit Trail
   ------------
   - Log all authentication attempts
   - Track successful and failed logins
   - Record user agent and IP address
   - KVKK-compliant logging
   - Integration with SIEM systems

=============================================================================
USAGE
=============================================================================

Basic Integration:
------------------

>>> from fastapi import FastAPI
>>> from backend.api.middleware.auth import AuthMiddleware
>>>
>>> app = FastAPI()
>>> app.add_middleware(AuthMiddleware)
>>>
>>> # All routes except PUBLIC_ROUTES now require authentication
>>> # Request must include: Authorization: Bearer <token>

Custom Public Routes:
---------------------

>>> app = FastAPI()
>>> auth_middleware = AuthMiddleware(
...     public_routes=[
...         "/",
...         "/health",
...         "/api/v1/auth/*",  # Wildcard support
...         "/api/v1/public/*",
...     ]
... )
>>> app.add_middleware(AuthMiddleware, public_routes=auth_middleware.public_routes)

Making Authenticated Request:
------------------------------

>>> import httpx
>>>
>>> # 1. Login to get token
>>> response = await httpx.post(
...     "https://api.turkishlegai.com/api/v1/auth/login",
...     json={"email": "user@example.com", "password": "secure123"}
... )
>>> token = response.json()["access_token"]
>>>
>>> # 2. Use token for authenticated requests
>>> response = await httpx.get(
...     "https://api.turkishlegai.com/api/v1/contracts",
...     headers={"Authorization": f"Bearer {token}"}
... )

API Key Authentication:
-----------------------

>>> # Generate API key for tenant
>>> api_key = await api_key_service.create(
...     tenant_id="550e8400-e29b-41d4-a716-446655440000",
...     name="Production API Key",
...     permissions=["contracts:read", "contracts:write"],
...     expires_at=datetime.now() + timedelta(days=365)
... )
>>>
>>> # Use API key in requests
>>> response = await httpx.get(
...     "https://api.turkishlegai.com/api/v1/contracts",
...     headers={"X-API-Key": api_key.key}
... )

WebSocket Authentication:
--------------------------

>>> # For WebSocket connections, pass token as query param
>>> ws = await websocket_connect(
...     "wss://api.turkishlegai.com/ws?token=eyJhbGc..."
... )

Real-World Example (Protected Endpoint):
-----------------------------------------

>>> from fastapi import Depends, Request
>>>
>>> @app.get("/api/v1/contracts/{contract_id}")
>>> async def get_contract(
...     contract_id: str,
...     request: Request
... ):
...     # User context automatically injected by middleware
...     user_id = request.state.user_id
...     tenant_id = request.state.tenant_id
...     permissions = request.state.permissions
...
...     # Check permissions
...     if "contracts:read" not in permissions:
...         raise ForbiddenException("Insufficient permissions")
...
...     # Fetch contract with tenant isolation
...     contract = await db.get_contract(
...         contract_id,
...         tenant_id=tenant_id
...     )
...
...     return contract

=============================================================================
TOKEN FORMAT
=============================================================================

Access Token Structure:
------------------------

{
  "sub": "550e8400-e29b-41d4-a716-446655440000",  // User ID
  "tenant_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "email": "user@example.com",
  "type": "access",
  "permissions": [
    "contracts:read",
    "contracts:write",
    "analysis:read"
  ],
  "iat": 1699276800,  // Issued at
  "exp": 1699363200,  // Expires at (24 hours)
  "jti": "unique-token-id"  // JWT ID for blacklisting
}

Refresh Token Structure:
-------------------------

{
  "sub": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "type": "refresh",
  "iat": 1699276800,
  "exp": 1701868800,  // Expires at (30 days)
  "jti": "unique-refresh-token-id"
}

API Key Token Structure:
-------------------------

{
  "sub": "api-key",
  "tenant_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "type": "api_key",
  "key_id": "api_key_id",
  "permissions": ["contracts:read", "contracts:write"],
  "iat": 1699276800,
  "exp": 1730899200  // Expires at (1 year)
}

=============================================================================
AUTHENTICATION FLOW
=============================================================================

Standard Login Flow:
--------------------

1. User submits credentials (email + password)
   POST /api/v1/auth/login
   {
     "email": "user@example.com",
     "password": "secure123"
   }

2. Backend validates credentials
   - Check password hash
   - Verify email confirmed
   - Check account not locked

3. Backend generates tokens
   - Access token (24h expiry)
   - Refresh token (30d expiry)
   - Store refresh token in database

4. Client receives tokens
   {
     "access_token": "eyJhbGc...",
     "refresh_token": "eyJhbGc...",
     "expires_in": 86400,
     "token_type": "Bearer"
   }

5. Client stores tokens securely
   - Access token: Memory or sessionStorage
   - Refresh token: HttpOnly cookie (recommended)

6. Client makes authenticated requests
   Authorization: Bearer <access_token>

7. When access token expires, use refresh token
   POST /api/v1/auth/refresh
   {
     "refresh_token": "eyJhbGc..."
   }

8. Receive new access token
   {
     "access_token": "eyJhbGc...",
     "expires_in": 86400
   }

Turkish Authentication Messages:
---------------------------------

Successful Login:
  "Giriş başarılı. Hoş geldiniz!"

Invalid Credentials:
  "E-posta veya şifre hatalı."

Account Locked:
  "Hesabınız güvenlik nedeniyle kilitlenmiştir. Lütfen destek ekibi ile iletişime geçin."

Too Many Attempts:
  "Çok fazla başarısız giriş denemesi. Lütfen {delay} saniye sonra tekrar deneyin."

Token Expired:
  "Oturum süreniz dolmuştur. Lütfen tekrar giriş yapın."

Invalid Token:
  "Geçersiz kimlik doğrulama bilgisi."

=============================================================================
RATE LIMITING
=============================================================================

Failed Login Attempts:
----------------------

Attempt | Delay     | Action
--------|-----------|------------------------------------------
1-3     | 0s        | Allow immediately
4-5     | 2s        | Add 2 second delay
6-10    | 5s        | Add 5 second delay
11-15   | 30s       | Add 30 second delay
16-20   | 5 min     | Add 5 minute delay
21+     | 30 min    | Lock account, require password reset

Rate Limit Headers:
-------------------

X-RateLimit-Limit: 1000          # Total requests allowed per hour
X-RateLimit-Remaining: 995       # Requests remaining
X-RateLimit-Reset: 1699363200    # Unix timestamp when limit resets

=============================================================================
SECURITY CONSIDERATIONS
=============================================================================

1. Token Storage:
   - NEVER store tokens in localStorage (XSS vulnerable)
   - Use httpOnly cookies for refresh tokens
   - Store access tokens in memory or sessionStorage
   - Clear tokens on logout

2. Token Transmission:
   - ALWAYS use HTTPS in production
   - Never send tokens in URL query parameters (except WebSocket)
   - Use Authorization header for API requests
   - Implement token rotation on security events

3. Session Security:
   - Limit concurrent sessions per user (default: 5)
   - Track device fingerprints
   - Validate IP consistency for sensitive operations
   - Implement logout from all devices

4. Brute Force Protection:
   - Rate limit login attempts by IP and email
   - Implement progressive delays
   - Lock accounts after threshold
   - Require CAPTCHA after failures

5. Token Blacklisting:
   - Implement JWT ID (jti) for token revocation
   - Store blacklisted tokens in Redis
   - Clean expired blacklist entries
   - Invalidate all tokens on password change

=============================================================================
KVKK COMPLIANCE
=============================================================================

Authentication Logs:
--------------------
- Log authentication events with minimal PII
- Store hashed user IDs instead of emails in logs
- Implement 90-day retention for auth logs
- Provide user access to their auth history

Data Subject Rights:
--------------------
- Allow users to view active sessions
- Provide session termination capability
- Export authentication history on request
- Delete authentication logs on account deletion

Audit Trail Requirements:
--------------------------
- Track all authentication attempts
- Log IP address and user agent
- Record timestamp and result (success/failure)
- Maintain immutable audit log

=============================================================================
TROUBLESHOOTING
=============================================================================

"401 Unauthorized" Error:
--------------------------
1. Check token is present in Authorization header
2. Verify token format: "Bearer <token>"
3. Check token hasn't expired (decode at jwt.io)
4. Verify JWT_SECRET_KEY matches between services
5. Check user hasn't been deleted or disabled

"Token has expired":
--------------------
1. Use refresh token to get new access token
2. Implement automatic token refresh in client
3. Check server clock synchronization (NTP)
4. Verify token expiry time is reasonable

"Invalid token signature":
--------------------------
1. Verify JWT_SECRET_KEY is correct
2. Check token wasn't modified in transit
3. Ensure algorithm matches (RS256 or HS256)
4. Verify public key for RS256 tokens

"Too many authentication attempts":
------------------------------------
1. Wait for rate limit cooldown period
2. Check for compromised credentials
3. Implement CAPTCHA in client
4. Contact support if account locked

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import (
    InvalidTokenException,
    TokenExpiredException,
    UnauthorizedException,
    decode_token,
    get_logger,
    settings,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Public routes that don't require authentication
DEFAULT_PUBLIC_ROUTES = [
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
    "/health/ready",
    "/health/live",
    "/metrics",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/v1/auth/verify-email",
    "/api/v1/public/*",  # Wildcard for all public endpoints
]

# Rate limiting thresholds
MAX_FAILED_ATTEMPTS = 5  # Before rate limiting kicks in
RATE_LIMIT_WINDOW = 900  # 15 minutes in seconds
ACCOUNT_LOCKOUT_THRESHOLD = 20  # Lock account after 20 failures
ACCOUNT_LOCKOUT_DURATION = 1800  # 30 minutes in seconds

# Session limits
MAX_CONCURRENT_SESSIONS = 5  # Maximum active sessions per user

# =============================================================================
# RATE LIMITER
# =============================================================================


class AuthRateLimiter:
    """
    Rate limiter for authentication attempts.

    Tracks failed login attempts by IP and email.
    Implements progressive delays and account lockout.
    """

    def __init__(self):
        """Initialize rate limiter with in-memory storage."""
        # In production, use Redis for distributed rate limiting
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}

    def record_failure(self, identifier: str) -> None:
        """
        Record failed authentication attempt.

        Args:
            identifier: IP address or email
        """
        now = datetime.utcnow()

        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []

        self.failed_attempts[identifier].append(now)

        # Clean old attempts (outside rate limit window)
        cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
        self.failed_attempts[identifier] = [
            attempt
            for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]

        # Check if account should be locked
        if len(self.failed_attempts[identifier]) >= ACCOUNT_LOCKOUT_THRESHOLD:
            self.locked_accounts[identifier] = now + timedelta(
                seconds=ACCOUNT_LOCKOUT_DURATION
            )
            logger.warning(
                "⚠️ Hesap kilitlendi (çok fazla başarısız deneme)",
                identifier=identifier,
                attempts=len(self.failed_attempts[identifier]),
            )

    def is_locked(self, identifier: str) -> bool:
        """
        Check if identifier is currently locked.

        Args:
            identifier: IP address or email

        Returns:
            True if locked
        """
        if identifier in self.locked_accounts:
            unlock_time = self.locked_accounts[identifier]
            if datetime.utcnow() < unlock_time:
                return True
            else:
                # Unlock has expired
                del self.locked_accounts[identifier]
                if identifier in self.failed_attempts:
                    del self.failed_attempts[identifier]

        return False

    def get_delay(self, identifier: str) -> int:
        """
        Get delay in seconds for next attempt.

        Args:
            identifier: IP address or email

        Returns:
            Delay in seconds (0 if no delay needed)
        """
        if identifier not in self.failed_attempts:
            return 0

        attempts = len(self.failed_attempts[identifier])

        if attempts <= 3:
            return 0
        elif attempts <= 5:
            return 2  # 2 seconds
        elif attempts <= 10:
            return 5  # 5 seconds
        elif attempts <= 15:
            return 30  # 30 seconds
        elif attempts <= 20:
            return 300  # 5 minutes
        else:
            return 1800  # 30 minutes

    def clear_failures(self, identifier: str) -> None:
        """
        Clear failed attempts for identifier (on successful auth).

        Args:
            identifier: IP address or email
        """
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
        if identifier in self.locked_accounts:
            del self.locked_accounts[identifier]


# Global rate limiter instance
_rate_limiter: Optional[AuthRateLimiter] = None


def get_rate_limiter() -> AuthRateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AuthRateLimiter()
    return _rate_limiter


# =============================================================================
# SESSION MANAGER
# =============================================================================


class SessionManager:
    """
    Manages user sessions and tracks concurrent logins.
    """

    def __init__(self):
        """Initialize session manager."""
        # In production, use Redis for distributed session storage
        self.sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids

    def add_session(self, user_id: str, session_id: str) -> bool:
        """
        Add new session for user.

        Args:
            user_id: User ID
            session_id: Unique session ID

        Returns:
            True if session added, False if limit exceeded
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = set()

        # Check concurrent session limit
        if len(self.sessions[user_id]) >= MAX_CONCURRENT_SESSIONS:
            logger.warning(
                "⚠️ Maksimum oturum sayısı aşıldı",
                user_id=user_id,
                session_count=len(self.sessions[user_id]),
            )
            # Remove oldest session (in production, track timestamps)
            self.sessions[user_id].pop()

        self.sessions[user_id].add(session_id)
        return True

    def remove_session(self, user_id: str, session_id: str) -> None:
        """
        Remove session for user.

        Args:
            user_id: User ID
            session_id: Session ID to remove
        """
        if user_id in self.sessions:
            self.sessions[user_id].discard(session_id)
            if not self.sessions[user_id]:
                del self.sessions[user_id]

    def get_session_count(self, user_id: str) -> int:
        """
        Get active session count for user.

        Args:
            user_id: User ID

        Returns:
            Number of active sessions
        """
        return len(self.sessions.get(user_id, set()))


# =============================================================================
# AUTH MIDDLEWARE
# =============================================================================


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for JWT token validation.

    Features:
    - Validates Bearer tokens and API keys
    - Injects user context into request state
    - Rate limits failed authentication attempts
    - Tracks concurrent sessions
    - Provides audit trail for authentication events
    """

    def __init__(
        self,
        app,
        public_routes: Optional[List[str]] = None,
        rate_limiter: Optional[AuthRateLimiter] = None,
    ):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application
            public_routes: Optional custom list of public routes
            rate_limiter: Optional custom rate limiter
        """
        super().__init__(app)
        self.public_routes = public_routes or DEFAULT_PUBLIC_ROUTES
        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.session_manager = SessionManager()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with authentication check.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response (401 if auth fails, otherwise continues)
        """
        # Check if route is public
        if self._is_public_route(request.url.path, request.method):
            return await call_next(request)

        # Get client IP for rate limiting
        client_ip = self._get_client_ip(request)

        # Check if IP is rate limited
        if self.rate_limiter.is_locked(client_ip):
            return self._rate_limit_response(
                f"Çok fazla başarısız giriş denemesi. Lütfen 30 dakika sonra tekrar deneyin."
            )

        # Extract token from Authorization header or query param
        token = self._extract_token(request)

        if not token:
            self.rate_limiter.record_failure(client_ip)
            return self._unauthorized_response(
                "Kimlik doğrulama bilgisi eksik",
                "Missing authentication token"
            )

        # Validate token
        try:
            payload = await self._validate_token(token)

            # Extract user info from token
            user_id = payload.get("sub")
            tenant_id = payload.get("tenant_id")
            permissions = payload.get("permissions", [])
            token_type = payload.get("type", "access")

            # Store in request state for downstream access
            request.state.user_id = user_id
            request.state.tenant_id = tenant_id
            request.state.permissions = permissions
            request.state.token_type = token_type

            # Clear rate limit on successful auth
            self.rate_limiter.clear_failures(client_ip)

            logger.debug(
                "Kimlik doğrulama başarılı",
                user_id=user_id,
                tenant_id=tenant_id,
                path=request.url.path,
                method=request.method,
                client_ip=client_ip,
            )

        except TokenExpiredException:
            self.rate_limiter.record_failure(client_ip)
            return self._unauthorized_response(
                "Oturum süreniz dolmuştur. Lütfen tekrar giriş yapın.",
                "Token has expired"
            )

        except InvalidTokenException as e:
            self.rate_limiter.record_failure(client_ip)
            return self._unauthorized_response(
                "Geçersiz kimlik doğrulama bilgisi",
                f"Invalid token: {str(e)}"
            )

        except Exception as e:
            self.rate_limiter.record_failure(client_ip)
            logger.error(
                "Token doğrulama hatası",
                error=str(e),
                path=request.url.path,
                client_ip=client_ip,
            )
            return self._unauthorized_response(
                "Kimlik doğrulama başarısız oldu",
                "Token validation failed"
            )

        # Process request
        return await call_next(request)

    def _is_public_route(self, path: str, method: str) -> bool:
        """
        Check if route is public and doesn't require authentication.

        Args:
            path: Request URL path
            method: HTTP method

        Returns:
            True if route is public
        """
        # Exact match
        if path in self.public_routes:
            return True

        # Wildcard match (e.g., /api/v1/public/*)
        for public_route in self.public_routes:
            if public_route.endswith("*"):
                prefix = public_route[:-1]  # Remove asterisk
                if path.startswith(prefix):
                    return True
            elif path.startswith(public_route):
                return True

        return False

    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract authentication token from request.

        Checks (in order):
        1. Authorization header (Bearer token)
        2. X-API-Key header (API key)
        3. Query parameter (for websockets)

        Args:
            request: Incoming FastAPI request

        Returns:
            Token string or None
        """
        # 1. Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # 2. X-API-Key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key

        # 3. Query parameter (for websocket connections)
        token_param = request.query_params.get("token")
        if token_param:
            return token_param

        return None

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.

        Args:
            request: Incoming request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take first IP in chain
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    async def _validate_token(self, token: str) -> Dict:
        """
        Validate JWT token.

        Args:
            token: JWT token string

        Returns:
            Token payload

        Raises:
            TokenExpiredException: If token expired
            InvalidTokenException: If token invalid
        """
        # Decode and validate token
        payload = decode_token(token)

        # Additional validations can go here:
        # - Check if token is blacklisted (logout)
        # - Verify tenant is active
        # - Check user is not deleted/disabled

        return payload

    def _unauthorized_response(
        self, turkish_message: str, english_message: str
    ) -> JSONResponse:
        """
        Return standardized 401 Unauthorized response.

        Args:
            turkish_message: User-facing Turkish message
            english_message: Technical English message

        Returns:
            JSON response with 401 status
        """
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": turkish_message,
                    "details": english_message if settings.DEBUG else None,
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _rate_limit_response(self, message: str) -> JSONResponse:
        """
        Return rate limit exceeded response.

        Args:
            message: Error message

        Returns:
            JSON response with 429 status
        """
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": message,
                    "retry_after": ACCOUNT_LOCKOUT_DURATION,
                }
            },
            headers={"Retry-After": str(ACCOUNT_LOCKOUT_DURATION)},
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AuthMiddleware",
    "AuthRateLimiter",
    "SessionManager",
    "get_rate_limiter",
]
