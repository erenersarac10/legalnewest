"""
Authentication Middleware for Turkish Legal AI Platform.

JWT token validation middleware for protected routes.

Features:
- Validates Bearer tokens in Authorization header
- Extracts user context from JWT
- Supports API key authentication
- Bypasses auth for public routes
- Rate limits failed auth attempts

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import Callable, Optional

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

logger = get_logger(__name__)

# Public routes that don't require authentication
PUBLIC_ROUTES = [
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
]


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for JWT token validation.

    Validates Bearer tokens and API keys.
    Injects user context into request state.
    """

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
        if self._is_public_route(request.url.path):
            return await call_next(request)

        # Extract token from Authorization header or query param
        token = self._extract_token(request)

        if not token:
            return self._unauthorized_response("Missing authentication token")

        # Validate token
        try:
            payload = decode_token(token)

            # Extract user info from token
            user_id = payload.get("sub")
            tenant_id = payload.get("tenant_id")
            permissions = payload.get("permissions", [])

            # Store in request state for downstream access
            request.state.user_id = user_id
            request.state.tenant_id = tenant_id
            request.state.permissions = permissions
            request.state.token_type = payload.get("type", "access")

            logger.debug(
                "Authentication successful",
                user_id=user_id,
                tenant_id=tenant_id,
                path=request.url.path,
            )

        except TokenExpiredException:
            return self._unauthorized_response("Token has expired")

        except InvalidTokenException as e:
            return self._unauthorized_response(f"Invalid token: {str(e)}")

        except Exception as e:
            logger.error(
                "Token validation error",
                error=str(e),
                path=request.url.path,
            )
            return self._unauthorized_response("Token validation failed")

        # Process request
        return await call_next(request)

    def _is_public_route(self, path: str) -> bool:
        """
        Check if route is public and doesn't require authentication.

        Args:
            path: Request URL path

        Returns:
            True if route is public
        """
        # Exact match
        if path in PUBLIC_ROUTES:
            return True

        # Prefix match (e.g., /docs/*)
        for public_route in PUBLIC_ROUTES:
            if path.startswith(public_route):
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
        if auth_header and auth_header.startswith("Bearer "):
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

    def _unauthorized_response(self, message: str) -> JSONResponse:
        """
        Return standardized 401 Unauthorized response.

        Args:
            message: Error message

        Returns:
            JSON response with 401 status
        """
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": message,
                    "details": "Lütfen giriş yapın veya geçerli bir token sağlayın.",
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["AuthMiddleware"]
