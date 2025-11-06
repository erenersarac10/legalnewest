"""
Request ID Middleware for Turkish Legal AI Platform.

Generates or extracts a unique request ID for every incoming request.
The request ID is used for distributed tracing and log correlation.

Features:
- Auto-generates UUID4 if not provided
- Accepts X-Request-ID header from clients
- Injects request ID into response headers
- Sets request ID in logging context

Author: Turkish Legal AI Team
License: Proprietary
"""

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, set_log_context

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and track unique request IDs.

    Adds X-Request-ID to both request state and response headers.
    Enables end-to-end request tracing across microservices.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with unique ID.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response with X-Request-ID header
        """
        # Extract or generate request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Store in request state for downstream access
        request.state.request_id = request_id

        # Set in logging context for all log statements
        set_log_context(request_id=request_id)

        # Process request
        response = await call_next(request)

        # Inject into response headers
        response.headers["X-Request-ID"] = request_id

        return response


__all__ = ["RequestIDMiddleware"]
