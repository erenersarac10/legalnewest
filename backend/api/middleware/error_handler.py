"""
Error Handler Middleware for Turkish Legal AI Platform.

Provides global exception handling with standardized error responses.

Features:
- Catches all unhandled exceptions
- Returns consistent error format
- Logs errors with full context
- Hides sensitive details in production
- KVKK-compliant error messages

Author: Turkish Legal AI Team
License: Proprietary
"""

import traceback
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import (
    BaseAppException,
    HTTPException,
    get_logger,
    settings,
)

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.

    Converts all exceptions to standardized JSON error responses.
    Logs errors with request context for debugging.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with error handling.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response (success or error JSON)
        """
        try:
            # Process request normally
            response = await call_next(request)
            return response

        except BaseAppException as e:
            # Known application exceptions (business logic errors)
            logger.warning(
                "Application exception",
                exception=e.__class__.__name__,
                message=str(e),
                status_code=e.status_code,
                path=request.url.path,
                method=request.method,
            )

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.error_code or e.__class__.__name__,
                        "message": str(e),
                        "details": e.details if settings.DEBUG else None,
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
            )

        except HTTPException as e:
            # FastAPI HTTP exceptions
            logger.warning(
                "HTTP exception",
                status_code=e.status_code,
                detail=e.detail,
                path=request.url.path,
                method=request.method,
            )

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": f"HTTP_{e.status_code}",
                        "message": e.detail,
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
            )

        except Exception as e:
            # Unhandled exceptions (500 errors)
            error_id = getattr(request.state, "request_id", "unknown")

            logger.error(
                "Unhandled exception",
                exception=e.__class__.__name__,
                message=str(e),
                path=request.url.path,
                method=request.method,
                error_id=error_id,
                traceback=traceback.format_exc() if settings.DEBUG else None,
            )

            # Different messages for dev vs prod
            if settings.DEBUG:
                message = f"{e.__class__.__name__}: {str(e)}"
                details = {"traceback": traceback.format_exc()}
            else:
                message = "Bir hata oluştu. Lütfen daha sonra tekrar deneyin."
                details = None

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": message,
                        "details": details,
                        "request_id": error_id,
                        "support": "Sorun devam ederse destek@turkishlegai.com ile iletişime geçin.",
                    }
                },
            )


__all__ = ["ErrorHandlerMiddleware"]
