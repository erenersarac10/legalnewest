"""
Timing Middleware for Turkish Legal AI Platform.

Measures request processing time and adds performance metrics.

Features:
- Tracks end-to-end request latency
- Adds X-Response-Time header
- Logs slow requests (configurable threshold)
- Integrates with metrics collection

Author: Turkish Legal AI Team
License: Proprietary
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, settings

logger = get_logger(__name__)

# Slow request threshold in seconds
SLOW_REQUEST_THRESHOLD = 5.0


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to measure and report request processing time.

    Adds X-Response-Time header with millisecond precision.
    Logs warnings for requests exceeding threshold.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with timing metrics.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response with X-Response-Time header
        """
        # Start timer
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        # Add response header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Log slow requests
        if duration > SLOW_REQUEST_THRESHOLD:
            logger.warning(
                "Slow request detected",
                method=request.method,
                path=request.url.path,
                duration_ms=f"{duration_ms:.2f}",
                threshold_ms=SLOW_REQUEST_THRESHOLD * 1000,
            )

        # Emit metrics (if observability is enabled)
        if settings.OBSERVABILITY_ENABLED:
            # from backend.observability import metrics
            # metrics.request_duration.observe(duration, method=request.method, path=request.url.path)
            pass

        return response


__all__ = ["TimingMiddleware"]
