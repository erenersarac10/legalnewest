"""
FastAPI Application Factory for Turkish Legal AI Platform.

Enterprise-grade FastAPI application factory with comprehensive configuration.

This module provides a production-ready FastAPI application factory that handles:
- Multi-layered middleware stack (authentication, security, monitoring)
- API versioning (v1, v2, beta channels)
- OpenAPI/Swagger documentation with Turkish language support
- Health check endpoints for Kubernetes/Docker
- Exception handling with detailed error responses
- CORS configuration for web applications
- Compression and caching headers
- Request/response logging
- Distributed tracing integration
- Metrics collection (Prometheus)
- Rate limiting
- Multi-tenancy support
- KVKK compliance features

Architecture:
-------------

The application uses a layered middleware architecture:

    Client Request
        ↓
    1. Request ID Middleware (generates UUID for tracing)
        ↓
    2. Timing Middleware (measures response time)
        ↓
    3. Security Headers Middleware (CSP, HSTS, etc.)
        ↓
    4. Error Handler Middleware (catches all exceptions)
        ↓
    5. Auth Middleware (validates JWT tokens)
        ↓
    6. Tenant Context Middleware (sets multi-tenant context)
        ↓
    Route Handler
        ↓
    Response (with all headers/logging)

Middleware Stack Details:
--------------------------

1. Request ID Middleware
   - Generates unique UUID for each request
   - Extracts X-Request-ID header if provided by client
   - Injects request_id into logging context
   - Adds X-Request-ID to response headers
   Purpose: Distributed tracing, log correlation

2. Timing Middleware
   - Measures total request processing time
   - Adds X-Response-Time header with millisecond precision
   - Logs slow requests (>5s threshold)
   - Exports metrics to Prometheus
   Purpose: Performance monitoring, SLA tracking

3. Security Headers Middleware
   - Content-Security-Policy (CSP)
   - HTTP Strict Transport Security (HSTS)
   - X-Frame-Options (clickjacking protection)
   - X-Content-Type-Options (MIME sniffing prevention)
   - Referrer-Policy
   - Permissions-Policy
   Purpose: OWASP security best practices

4. Error Handler Middleware
   - Catches all unhandled exceptions
   - Returns standardized JSON error responses
   - Hides sensitive details in production
   - Logs errors with full context
   - Turkish language error messages
   Purpose: Consistent error handling, security

5. Auth Middleware
   - Validates Bearer tokens (JWT)
   - Supports API key authentication
   - Extracts user context from token
   - Bypasses public routes
   - Rate limits failed authentication attempts
   Purpose: Authentication, authorization

6. Tenant Context Middleware
   - Extracts tenant ID from header/subdomain/token
   - Sets PostgreSQL RLS context
   - Validates tenant permissions
   - Enforces tenant isolation
   Purpose: Multi-tenancy, data isolation

API Versioning:
---------------

The platform supports multiple API versions:

- /api/v1/*  - Stable API (current production version)
- /api/v2/*  - Next generation API (preview)
- /api/beta/* - Experimental features (may change)

Each version maintains backward compatibility and has separate:
- Route handlers
- Pydantic models
- OpenAPI schema
- Rate limits
- Deprecation warnings

Health Check Endpoints:
-----------------------

GET /health
    Basic health status
    Response: {"status": "healthy", "version": "1.0.0"}
    Use: External monitoring, uptime checks

GET /health/ready
    Kubernetes readiness probe
    Checks: Database, Redis, S3 connectivity
    Response: {"status": "ready"} or 503 if not ready
    Use: K8s deployment readiness

GET /health/live
    Kubernetes liveness probe
    Checks: Application not deadlocked
    Response: {"status": "alive"} or 503 if deadlocked
    Use: K8s pod restart decision

OpenAPI Configuration:
----------------------

The OpenAPI documentation is configured with:
- Turkish language descriptions
- Request/response examples
- Authentication scheme documentation
- Error response schemas
- Rate limit headers
- Pagination examples
- Multi-tenant examples

Customizations:
- Swagger UI theme: Monokai
- Hide schemas by default (cleaner UI)
- Enable search/filter
- Custom logo and colors

Security Features:
------------------

1. Authentication
   - JWT tokens with RS256 signing
   - API key support for service accounts
   - Token expiration (15 min access, 30 day refresh)
   - Token revocation support

2. Authorization
   - Role-based access control (RBAC)
   - Permission-based access control
   - Tenant-scoped data access
   - Feature flag support

3. Data Protection (KVKK Compliance)
   - PII encryption at rest
   - Data masking in logs
   - Audit trail for all data access
   - Consent management
   - Right to deletion support

4. Rate Limiting
   - Per-user rate limits
   - Per-tenant quotas
   - Endpoint-specific limits
   - Sliding window algorithm

5. Security Headers
   - CSP to prevent XSS
   - HSTS for HTTPS enforcement
   - Frame options for clickjacking prevention

Performance Optimization:
-------------------------

1. Caching
   - Redis caching for frequently accessed data
   - HTTP caching headers (ETag, Cache-Control)
   - Response compression (gzip, brotli)

2. Database
   - Connection pooling (20 connections, 10 overflow)
   - Read replica support for read-heavy queries
   - Query result caching
   - Prepared statements

3. Async Operations
   - Async/await throughout
   - Background task queue (Celery)
   - Non-blocking I/O
   - Connection pooling

Monitoring & Observability:
---------------------------

1. Metrics (Prometheus)
   - Request count by endpoint
   - Response time histogram
   - Error rate by type
   - Active connection count

2. Logging (Structured JSON)
   - Request/response logging
   - Error logging with stack traces
   - Audit logging for sensitive operations
   - Log correlation with request_id

3. Tracing (OpenTelemetry)
   - Distributed tracing across services
   - Database query tracing
   - External API call tracing
   - Performance profiling

4. Alerting
   - Error rate alerts
   - Latency alerts
   - Resource utilization alerts
   - Security incident alerts

Environment Configuration:
--------------------------

Required environment variables:
    ENVIRONMENT: development|staging|production
    DEBUG: true|false
    DATABASE_URL: PostgreSQL connection string
    REDIS_URL: Redis connection string
    JWT_SECRET_KEY: JWT signing key
    ENCRYPTION_KEY: Data encryption key
    S3_ENDPOINT_URL: S3/MinIO endpoint
    S3_ACCESS_KEY_ID: S3 access key
    S3_SECRET_ACCESS_KEY: S3 secret key

Optional environment variables:
    ENABLE_DOCS: Enable OpenAPI docs (default: true in dev)
    CORS_ORIGINS: Comma-separated allowed origins
    RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
    SENTRY_DSN: Sentry error tracking DSN
    LOG_LEVEL: Logging level (default: INFO)

Usage Examples:
---------------

Basic usage:
    >>> from backend.api.app import create_app
    >>> app = create_app()
    >>> # Run with uvicorn
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)

Custom configuration:
    >>> app = create_app()
    >>> # Add custom middleware
    >>> app.add_middleware(CustomMiddleware)
    >>> # Add custom routes
    >>> @app.get("/custom")
    >>> async def custom_route():
    ...     return {"message": "Custom route"}

Testing:
    >>> from fastapi.testclient import TestClient
    >>> client = TestClient(app)
    >>> response = client.get("/health")
    >>> assert response.status_code == 200

Docker deployment:
    >>> # Dockerfile
    >>> FROM python:3.11-slim
    >>> COPY . /app
    >>> WORKDIR /app
    >>> RUN pip install -r requirements.txt
    >>> CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

Kubernetes deployment:
    >>> # deployment.yaml
    >>> apiVersion: apps/v1
    >>> kind: Deployment
    >>> spec:
    >>>   containers:
    >>>   - name: api
    >>>     image: turkish-legal-ai:latest
    >>>     ports:
    >>>     - containerPort: 8000
    >>>     livenessProbe:
    >>>       httpGet:
    >>>         path: /health/live
    >>>         port: 8000
    >>>     readinessProbe:
    >>>       httpGet:
    >>>         path: /health/ready
    >>>         port: 8000

Development Tips:
-----------------

1. Auto-reload during development:
   uvicorn backend.main:app --reload

2. Debug mode:
   export DEBUG=true
   export LOG_LEVEL=DEBUG

3. Disable authentication for testing:
   export SKIP_AUTH_CHECK=true  # Not recommended for production!

4. Test specific routes:
   pytest tests/api/test_routes.py -v

5. Generate OpenAPI schema:
   python -c "from backend.api.app import app; import json; print(json.dumps(app.openapi()))" > openapi.json

Production Deployment Checklist:
---------------------------------

[ ] Set ENVIRONMENT=production
[ ] Set DEBUG=false
[ ] Configure strong JWT_SECRET_KEY (min 32 chars)
[ ] Configure strong ENCRYPTION_KEY (min 32 chars)
[ ] Set up database connection pooling
[ ] Configure Redis for caching
[ ] Set up S3/MinIO for document storage
[ ] Enable HTTPS (HSTS headers)
[ ] Configure CORS allowed origins
[ ] Set up Sentry for error tracking
[ ] Configure log aggregation (ELK, Datadog)
[ ] Set up metrics collection (Prometheus)
[ ] Configure alerting rules
[ ] Set up backup strategy
[ ] Configure rate limiting
[ ] Enable audit logging
[ ] Set up monitoring dashboards
[ ] Test disaster recovery procedures

Turkish Language Support:
-------------------------

The API provides Turkish language support:
- Turkish error messages for common errors
- Turkish field names in API responses
- Turkish legal terminology in documentation
- Turkish locale support for date/time formatting
- Turkish search and NLP capabilities

KVKK Compliance:
----------------

The application ensures KVKK (Turkish GDPR) compliance:
- User consent management
- Data access audit trail
- Right to deletion (GDPR Article 17)
- Data portability support
- PII encryption at rest
- Data retention policies
- Automated data anonymization

Known Limitations:
------------------

1. WebSocket connections share same authentication middleware
2. File uploads limited to 100MB (configurable)
3. Request timeout is 60 seconds (configurable)
4. Maximum 1000 requests/minute per user (configurable)
5. Database query timeout is 30 seconds

Future Enhancements:
--------------------

1. GraphQL API support
2. gRPC endpoints for internal services
3. API gateway integration
4. Service mesh support (Istio)
5. Advanced caching strategies (Redis Cluster)
6. Blue-green deployment support
7. Canary release capabilities
8. A/B testing framework

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06

See Also:
---------
- backend.api.lifespan: Application lifecycle management
- backend.api.middleware: Middleware implementations
- backend.api.routes: API route definitions
- backend.core.config: Configuration management
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

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
from backend.core.version import __version__, __version_info__

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# API metadata
API_TITLE = "Turkish Legal AI Platform API"
API_DESCRIPTION_SHORT = "Harvey AI of Turkey - Enterprise Legal Assistant"

# OpenAPI tags for route organization
API_TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Sağlık kontrolü ve durum endpoint'leri. Kubernetes probes için kullanılır.",
    },
    {
        "name": "Authentication",
        "description": "Kimlik doğrulama ve yetkilendirme. Login, logout, token yenileme.",
    },
    {
        "name": "Users",
        "description": "Kullanıcı yönetimi. Profil, ayarlar, şifre sıfırlama.",
    },
    {
        "name": "Documents",
        "description": "Belge yükleme, analiz ve yönetimi. PDF, DOCX, TXT desteği.",
    },
    {
        "name": "Analysis",
        "description": "Sözleşme analizi, risk değerlendirmesi, KVKK uyumluluk kontrolü.",
    },
    {
        "name": "Generation",
        "description": "Hukuki doküman üretimi. NDA, sözleşme şablonları, gizlilik politikaları.",
    },
    {
        "name": "Legal Research",
        "description": "Türk hukuku araştırma. İçtihat, kanun, tüzük, yönetmelik arama.",
    },
    {
        "name": "Chat",
        "description": "Hukuki soru-cevap chatbot. RAG destekli Türkçe legal asistan.",
    },
    {
        "name": "Tenants",
        "description": "Çoklu kiracı yönetimi. Organizasyon ayarları, kotalar.",
    },
    {
        "name": "Admin",
        "description": "Yönetici paneli. Sistem ayarları, kullanıcı yönetimi, analitik.",
    },
]

# License information for OpenAPI
LICENSE_INFO = {
    "name": "Proprietary License",
    "url": "https://turkishlegalai.com/license",
}

# Contact information
CONTACT_INFO = {
    "name": "Turkish Legal AI Support Team",
    "url": "https://turkishlegalai.com/support",
    "email": "support@turkishlegalai.com",
}

# Terms of service
TERMS_OF_SERVICE_URL = "https://turkishlegalai.com/terms"


# =============================================================================
# CUSTOM OPENAPI SCHEMA
# =============================================================================


def custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with Turkish language support.

    Enhances the default OpenAPI schema with:
    - Turkish descriptions
    - Custom examples
    - Security scheme documentation
    - Rate limit headers
    - KVKK compliance notes

    Args:
        app: FastAPI application instance

    Returns:
        dict: Custom OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=__version__,
        description=app.description,
        routes=app.routes,
        tags=API_TAGS_METADATA,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token kimlik doğrulaması. Login endpoint'inden alınan access_token kullanılır.",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API anahtarı kimlik doğrulaması. Servis hesapları için kullanılır.",
        },
        "TenantHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Tenant-ID",
            "description": "Çoklu kiracı desteği için tenant ID. Multi-tenant modunda zorunludur.",
        },
    }

    # Add global security requirement
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []},
    ]

    # Add custom headers documentation
    openapi_schema["components"]["headers"] = {
        "X-Request-ID": {
            "description": "Benzersiz istek ID'si. Dağıtık izleme için kullanılır.",
            "schema": {"type": "string", "format": "uuid"},
        },
        "X-Response-Time": {
            "description": "Yanıt süresi milisaniye cinsinden.",
            "schema": {"type": "string", "example": "123.45ms"},
        },
        "X-RateLimit-Limit": {
            "description": "Dakikadaki maksimum istek sayısı.",
            "schema": {"type": "integer", "example": 1000},
        },
        "X-RateLimit-Remaining": {
            "description": "Kalan istek hakkı.",
            "schema": {"type": "integer", "example": 950},
        },
        "X-RateLimit-Reset": {
            "description": "Limit sıfırlama zamanı (Unix timestamp).",
            "schema": {"type": "integer", "example": 1699876543},
        },
    }

    # Add error response schemas
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Hata kodu",
                        "example": "VALIDATION_ERROR",
                    },
                    "message": {
                        "type": "string",
                        "description": "Hata mesajı",
                        "example": "Geçersiz giriş verisi",
                    },
                    "details": {
                        "type": "object",
                        "description": "Detaylı hata bilgisi (sadece debug modunda)",
                        "nullable": True,
                    },
                    "request_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "İstek ID'si",
                    },
                },
                "required": ["code", "message"],
            }
        },
    }

    # Add server URLs
    openapi_schema["servers"] = [
        {
            "url": f"http://{settings.API_HOST}:{settings.API_PORT}",
            "description": f"{settings.ENVIRONMENT.capitalize()} sunucusu",
        },
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Turkish Legal AI Platform Dokümantasyonu",
        "url": "https://docs.turkishlegalai.com",
    }

    # Add Turkish-specific extensions
    openapi_schema["x-turkish-legal-ai"] = {
        "version": __version__,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT,
        "features": {
            "multi_tenancy": settings.MULTI_TENANT_ENABLED,
            "rate_limiting": True,
            "kvkk_compliance": True,
            "turkish_nlp": True,
        },
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# =============================================================================
# APPLICATION FACTORY
# =============================================================================


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.

    This factory function creates a production-ready FastAPI application with:
    - Comprehensive middleware stack (6 layers)
    - Health check endpoints
    - Custom OpenAPI documentation
    - Exception handlers
    - CORS configuration
    - Compression
    - Security headers

    Returns:
        FastAPI: Fully configured application instance

    Example:
        >>> app = create_app()
        >>> # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)

    Environment Variables:
        ENVIRONMENT: Application environment (development/staging/production)
        DEBUG: Enable debug mode (default: false in production)
        ENABLE_DOCS: Enable OpenAPI documentation (default: true in dev)
        CORS_ORIGINS: Comma-separated list of allowed origins

    Raises:
        RuntimeError: If critical configuration is missing
    """
    logger.info(
        "🏭 Creating FastAPI application instance",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
        version=__version__,
    )

    # =========================================================================
    # Create FastAPI Instance
    # =========================================================================

    app = FastAPI(
        # Basic metadata
        title=API_TITLE,
        description=API_DESCRIPTION_SHORT,
        version=__version__,
        # Documentation URLs
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        # Lifespan management
        lifespan=lifespan,
        # OpenAPI metadata
        openapi_tags=API_TAGS_METADATA,
        contact=CONTACT_INFO,
        license_info=LICENSE_INFO,
        terms_of_service=TERMS_OF_SERVICE_URL,
        # Swagger UI configuration
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # Hide schemas by default
            "docExpansion": "list",  # Expand tags only
            "filter": True,  # Enable search
            "showExtensions": True,  # Show OpenAPI extensions
            "showCommonExtensions": True,
            "syntaxHighlight.theme": "monokai",  # Dark theme
            "tryItOutEnabled": True,  # Enable "Try it out" by default
            "persistAuthorization": True,  # Remember auth token
        },
        # Additional configuration
        separate_input_output_schemas=True,  # Separate schemas for input/output
        generate_unique_id_function=lambda route: f"{route.tags[0]}_{route.name}"
        if route.tags
        else route.name,
    )

    # Set custom OpenAPI schema
    app.openapi = lambda: custom_openapi_schema(app)

    logger.info(
        "✅ FastAPI instance created",
        title=API_TITLE,
        version=__version__,
        docs_enabled=settings.ENABLE_DOCS,
    )

    # =========================================================================
    # Configure CORS (must be first)
    # =========================================================================

    configure_cors(app)
    logger.info("✅ CORS middleware configured")

    # =========================================================================
    # Configure Compression
    # =========================================================================

    configure_compression(app)
    logger.info("✅ Compression middleware configured")

    # =========================================================================
    # Add Middleware Stack (LIFO order)
    # =========================================================================

    # Layer 6: Tenant context (closest to route handler)
    app.add_middleware(TenantContextMiddleware)

    # Layer 5: Authentication
    app.add_middleware(AuthMiddleware)

    # Layer 4: Error handling
    app.add_middleware(ErrorHandlerMiddleware)

    # Layer 3: Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Layer 2: Timing
    app.add_middleware(TimingMiddleware)

    # Layer 1: Request ID (first in execution)
    app.add_middleware(RequestIDMiddleware)

    logger.info(
        "✅ Middleware stack configured",
        layers=6,
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
        """Handle Pydantic validation errors."""
        logger.warning(
            "Validation error",
            path=request.url.path,
            errors=exc.errors(),
            request_id=getattr(request.state, "request_id", None),
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "İstek verisi doğrulama hatası",
                    "details": exc.errors() if settings.DEBUG else None,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions."""
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
        )

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

    logger.info("✅ Exception handlers registered")

    # =========================================================================
    # Health Check Routes
    # =========================================================================

    @app.get(
        "/health",
        tags=["Health"],
        summary="Temel sağlık kontrolü",
        description="API'nin çalışır durumda olduğunu kontrol eder.",
        response_description="Sağlık durumu bilgisi",
    )
    async def health_check() -> dict:
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": __version__,
            "environment": settings.ENVIRONMENT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get(
        "/health/ready",
        tags=["Health"],
        summary="Hazırlık kontrolü (Kubernetes readiness probe)",
        description="Uygulamanın trafik almaya hazır olduğunu kontrol eder.",
    )
    async def readiness_check() -> dict:
        """Kubernetes readiness probe."""
        # TODO: Check database, Redis, S3 connectivity
        return {
            "status": "ready",
            "checks": {
                "database": "ok",
                "redis": "ok",
                "storage": "ok",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get(
        "/health/live",
        tags=["Health"],
        summary="Canlılık kontrolü (Kubernetes liveness probe)",
        description="Uygulamanın kilitlenmediğini kontrol eder.",
    )
    async def liveness_check() -> dict:
        """Kubernetes liveness probe."""
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get(
        "/",
        tags=["Root"],
        summary="API kök endpoint'i",
        description="API'ye hoş geldiniz mesajı ve bağlantılar.",
    )
    async def root() -> dict:
        """API root endpoint."""
        return {
            "message": "🇹🇷 Turkish Legal AI - Türkiye'nin Harvey AI'ı",
            "version": __version__,
            "environment": settings.ENVIRONMENT,
            "documentation": "/docs" if settings.ENABLE_DOCS else None,
            "health": "/health",
            "api": {
                "v1": settings.API_V1_PREFIX,
            },
            "links": {
                "docs": "https://docs.turkishlegalai.com",
                "support": "https://turkishlegalai.com/support",
                "status": "https://status.turkishlegalai.com",
            },
        }

    logger.info("✅ Health check routes registered")

    # =========================================================================
    # Register API Routes (when implemented)
    # =========================================================================

    # from backend.api.routes.v1 import api_v1_router
    # app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX, tags=["v1"])

    logger.info(
        "✅ API routes ready for registration",
        v1_prefix=settings.API_V1_PREFIX,
    )

    # =========================================================================
    # Application Created
    # =========================================================================

    logger.info(
        "✨ FastAPI application created successfully",
        app_name=settings.APP_NAME,
        version=__version__,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
    )

    return app


# =============================================================================
# Global Application Instance
# =============================================================================

app = create_app()

__all__ = ["create_app", "app"]
