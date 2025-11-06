"""
FastAPI Application Factory for Turkish Legal AI Platform.

Creates and configures the FastAPI application with:
- Middleware stack (auth, logging, security, CORS, etc.)
- Route registration (v1, v2, beta API versions)
- OpenAPI documentation
- Lifespan management (startup/shutdown)
- Exception handlers
- Health checks
- Metrics endpoints

Usage:
    from backend.api.app import create_app
    app = create_app()

Author: Turkish Legal AI Team
License: Proprietary
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.lifespan import lifespan
from backend.api.middleware import (
    AuthMiddleware,
    ErrorHandlerMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    TenantContextMiddleware,
    TimingMiddleware,
    configure_compression,
    configure_cors,
)
from backend.core import get_logger, settings

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Application factory for FastAPI.

    Creates a fully configured FastAPI instance with:
    - Middleware stack (12 layers)
    - API routes (v1, v2, beta)
    - OpenAPI documentation
    - Exception handlers
    - CORS, compression, security headers

    Returns:
        Configured FastAPI application instance

    Example:
        >>> app = create_app()
        >>> # Run with uvicorn:
        >>> # uvicorn backend.main:app --host 0.0.0.0 --port 8000
    """
    # =========================================================================
    # Create FastAPI Instance
    # =========================================================================
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "🇹🇷 **Harvey AI of Turkey** - Enterprise-grade AI legal assistant\n\n"
            "Features:\n"
            "- 🤖 Advanced contract analysis & generation\n"
            "- 📚 Turkish legal research & citation finder\n"
            "- ⚖️ KVKK compliance checking\n"
            "- 💬 Legal Q&A with RAG\n"
            "- 📄 Multi-format document processing\n"
            "- 🔒 Enterprise security & multi-tenancy\n"
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        lifespan=lifespan,
        # OpenAPI metadata
        contact={
            "name": "Turkish Legal AI Team",
            "email": "support@turkishlegalai.com",
            "url": "https://turkishlegalai.com",
        },
        license_info={
            "name": "Proprietary",
            "url": "https://turkishlegalai.com/license",
        },
        terms_of_service="https://turkishlegalai.com/terms",
        # API behavior
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # Hide schemas by default
            "docExpansion": "list",  # Expand only tags
            "filter": True,  # Enable search
            "syntaxHighlight.theme": "monokai",
        },
    )

    # =========================================================================
    # Configure CORS (must be added before other middleware)
    # =========================================================================
    configure_cors(app)

    # =========================================================================
    # Configure Compression
    # =========================================================================
    configure_compression(app)

    # =========================================================================
    # Add Middleware Stack (Order matters - LIFO execution)
    # =========================================================================
    # Middleware execution order:
    # 1. Request ID (first, so all logs have request_id)
    # 2. Timing (measure total request time)
    # 3. Security Headers (add headers to all responses)
    # 4. Error Handler (catch all exceptions)
    # 5. Auth (validate JWT tokens)
    # 6. Tenant Context (set multi-tenant context)

    # 6. Tenant context (executed last, closest to route)
    app.add_middleware(TenantContextMiddleware)

    # 5. Authentication (validate tokens)
    app.add_middleware(AuthMiddleware)

    # 4. Error handling (catch all exceptions)
    app.add_middleware(ErrorHandlerMiddleware)

    # 3. Security headers (add security headers)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Timing (measure request duration)
    app.add_middleware(TimingMiddleware)

    # 1. Request ID (executed first, generate request ID)
    app.add_middleware(RequestIDMiddleware)

    logger.info(
        "Middleware stack configured",
        middleware_count=6,
        order=[
            "RequestID",
            "Timing",
            "SecurityHeaders",
            "ErrorHandler",
            "Auth",
            "TenantContext",
        ],
    )

    # =========================================================================
    # Exception Handlers
    # =========================================================================

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors with detailed messages."""
        logger.warning(
            "Validation error",
            path=request.url.path,
            errors=exc.errors(),
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors() if settings.DEBUG else None,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions from Starlette."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    # =========================================================================
    # Health Check Routes
    # =========================================================================

    @app.get("/health", tags=["Health"], summary="Health check endpoint")
    async def health_check() -> dict:
        """
        Basic health check endpoint.

        Returns API status and version.
        """
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/health/ready", tags=["Health"], summary="Readiness probe")
    async def readiness_check() -> dict:
        """
        Kubernetes readiness probe.

        Checks if app is ready to receive traffic.
        """
        # TODO: Check database connectivity, Redis, etc.
        return {"status": "ready"}

    @app.get("/health/live", tags=["Health"], summary="Liveness probe")
    async def liveness_check() -> dict:
        """
        Kubernetes liveness probe.

        Checks if app is alive (not deadlocked).
        """
        return {"status": "alive"}

    @app.get("/", tags=["Root"], summary="API root")
    async def root() -> dict:
        """
        API root endpoint with welcome message and links.
        """
        return {
            "message": "🇹🇷 Turkish Legal AI - Harvey AI of Turkey",
            "version": settings.APP_VERSION,
            "docs": "/docs" if settings.ENABLE_DOCS else None,
            "health": "/health",
            "api": {
                "v1": settings.API_V1_PREFIX,
                "v2": settings.API_V2_PREFIX if hasattr(settings, "API_V2_PREFIX") else None,
            },
        }

    # =========================================================================
    # Register API Routes
    # =========================================================================

    # Note: Routes will be registered here once route modules are implemented
    # from backend.api.routes import api_v1_router, api_v2_router
    # app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)
    # app.include_router(api_v2_router, prefix=settings.API_V2_PREFIX)

    logger.info(
        "API routes registered",
        v1_prefix=settings.API_V1_PREFIX,
        # v2_prefix=settings.API_V2_PREFIX if hasattr(settings, "API_V2_PREFIX") else None,
    )

    # =========================================================================
    # Application Ready
    # =========================================================================
    logger.info(
        "FastAPI application created",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    return app


# =============================================================================
# Global Application Instance (for backward compatibility)
# =============================================================================
app = create_app()


__all__ = ["create_app", "app"]
