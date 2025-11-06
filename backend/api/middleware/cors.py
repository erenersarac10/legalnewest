"""
CORS Middleware Configuration for Turkish Legal AI Platform.

Configures Cross-Origin Resource Sharing (CORS) policies.

Features:
- Configurable allowed origins
- Credential support for authenticated requests
- Preflight request handling
- Production-safe defaults

Author: Turkish Legal AI Team
License: Proprietary
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core import get_logger, settings

logger = get_logger(__name__)


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.

    In development: Allows all origins for easier testing
    In production: Restricts to configured allowed origins

    Args:
        app: FastAPI application instance
    """
    # Parse allowed origins from settings
    allowed_origins = []

    if settings.CORS_ORIGINS:
        if isinstance(settings.CORS_ORIGINS, str):
            allowed_origins = [
                origin.strip() for origin in settings.CORS_ORIGINS.split(",")
            ]
        else:
            allowed_origins = settings.CORS_ORIGINS

    # Development: Allow all origins
    if settings.ENVIRONMENT == "development" and settings.DEBUG:
        allowed_origins = ["*"]
        logger.warning(
            "CORS configured for development - allowing all origins",
            environment=settings.ENVIRONMENT,
        )
    else:
        logger.info(
            "CORS configured for production",
            allowed_origins=allowed_origins,
        )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-Tenant-ID",
            "X-API-Key",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
        max_age=600,  # Preflight cache: 10 minutes
    )


__all__ = ["configure_cors"]
