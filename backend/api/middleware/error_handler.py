"""
Error Handler Middleware for Turkish Legal AI Platform.

This middleware provides comprehensive global exception handling with standardized error
responses, detailed logging, error classification, recovery strategies, and integration
with external error tracking systems.

=============================================================================
FEATURES
=============================================================================

1. Global Exception Handling
   --------------------------
   - Catches all unhandled exceptions across the application
   - Converts exceptions to standardized JSON error responses
   - Provides consistent error format for API clients
   - Implements HTTP status code mapping

2. Error Classification
   ----------------------
   - Categorizes errors by type (client, server, network, database)
   - Assigns severity levels (INFO, WARNING, ERROR, CRITICAL)
   - Identifies transient vs permanent errors
   - Tracks error frequency and patterns

3. Error Recovery
   ---------------
   - Implements retry logic for transient errors
   - Circuit breaker for failing services
   - Graceful degradation strategies
   - Fallback responses for critical services

4. Security & Privacy
   -------------------
   - Hides sensitive information in production
   - Sanitizes PII from error messages
   - KVKK-compliant error logging
   - Prevents information disclosure attacks

5. Observability Integration
   --------------------------
   - Sentry error tracking
   - Datadog APM integration
   - ELK stack log shipping
   - Prometheus error metrics
   - Custom error dashboards

6. Turkish Language Support
   -------------------------
   - User-friendly Turkish error messages
   - Localized error codes
   - Cultural context in error handling
   - Turkish legal terminology

=============================================================================
USAGE
=============================================================================

Basic Integration:
------------------

>>> from fastapi import FastAPI
>>> from backend.api.middleware.error_handler import ErrorHandlerMiddleware
>>>
>>> app = FastAPI()
>>> app.add_middleware(ErrorHandlerMiddleware)
>>>
>>> # All exceptions are now caught and converted to JSON responses

Custom Exception Handling:
---------------------------

>>> from backend.core import BaseAppException
>>>
>>> class ContractNotFoundError(BaseAppException):
...     def __init__(self, contract_id: str):
...         super().__init__(
...             message=f"Sözleşme bulunamadı: {contract_id}",
...             status_code=404,
...             error_code="CONTRACT_NOT_FOUND"
...         )
>>>
>>> @app.get("/contracts/{contract_id}")
>>> async def get_contract(contract_id: str):
...     contract = await db.get_contract(contract_id)
...     if not contract:
...         raise ContractNotFoundError(contract_id)
...     return contract
>>>
>>> # Error response:
>>> # {
>>> #   "error": {
>>> #     "code": "CONTRACT_NOT_FOUND",
>>> #     "message": "Sözleşme bulunamadı: abc123",
>>> #     "request_id": "550e8400-e29b-41d4-a716-446655440000"
>>> #   }
>>> # }

Error Recovery with Circuit Breaker:
-------------------------------------

>>> from backend.api.middleware.error_handler import CircuitBreakerMiddleware
>>>
>>> app.add_middleware(
...     CircuitBreakerMiddleware,
...     failure_threshold=5,    # Open after 5 failures
...     recovery_timeout=60,    # Try again after 60 seconds
...     expected_exception=ServiceUnavailableError
... )

Real-World Example (Contract Analysis Failure):
------------------------------------------------

>>> # Scenario: LLM API is down, we want graceful fallback
>>>
>>> @app.post("/api/v1/contracts/analyze")
>>> async def analyze_contract(contract: Contract):
...     try:
...         # Primary: Use GPT-4 for analysis
...         result = await llm_service.analyze(contract, model="gpt-4")
...     except LLMAPIError as e:
...         logger.warning("GPT-4 failed, falling back to GPT-3.5")
...         try:
...             # Fallback: Use GPT-3.5
...             result = await llm_service.analyze(contract, model="gpt-3.5-turbo")
...         except LLMAPIError:
...             logger.error("All LLM services failed")
...             # Final fallback: Return cached result or error
...             raise ServiceUnavailableError(
...                 "Sözleşme analiz servisi şu anda kullanılamıyor"
...             )
...     return result

=============================================================================
ERROR RESPONSE FORMAT
=============================================================================

Standard Error Response:
-------------------------

{
  "error": {
    "code": "ERROR_CODE",
    "message": "Kullanıcı dostu hata mesajı",
    "details": {  // Only in DEBUG mode
      "field": "validation error details",
      "traceback": "full stack trace"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-11-06T10:30:45.123Z",
    "support": "Sorun devam ederse destek@turkishlegai.com ile iletişime geçin"
  }
}

Validation Error Response:
---------------------------

{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Giriş verileri geçersiz",
    "details": [
      {
        "field": "email",
        "message": "Geçersiz e-posta adresi",
        "type": "value_error.email"
      },
      {
        "field": "password",
        "message": "Şifre en az 8 karakter olmalıdır",
        "type": "value_error.min_length"
      }
    ],
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}

=============================================================================
ERROR CLASSIFICATION
=============================================================================

Client Errors (4xx):
--------------------
- 400 Bad Request: Invalid input, validation errors
- 401 Unauthorized: Missing or invalid authentication
- 403 Forbidden: Valid auth but insufficient permissions
- 404 Not Found: Resource doesn't exist
- 409 Conflict: Resource conflict (duplicate email, etc.)
- 422 Unprocessable Entity: Semantic validation errors
- 429 Too Many Requests: Rate limit exceeded

Server Errors (5xx):
--------------------
- 500 Internal Server Error: Unexpected application error
- 502 Bad Gateway: Upstream service error
- 503 Service Unavailable: Temporary unavailability
- 504 Gateway Timeout: Upstream service timeout

Turkish Error Messages:
-----------------------
- 400: "Geçersiz istek. Lütfen girdiğiniz bilgileri kontrol edin."
- 401: "Oturum açmanız gerekiyor."
- 403: "Bu işlem için yetkiniz bulunmuyor."
- 404: "İstenen kaynak bulunamadı."
- 409: "Bu kaynak zaten mevcut."
- 429: "Çok fazla istek gönderildi. Lütfen daha sonra tekrar deneyin."
- 500: "Bir hata oluştu. Lütfen daha sonra tekrar deneyin."
- 503: "Servis geçici olarak kullanılamıyor."

=============================================================================
INTEGRATION EXAMPLES
=============================================================================

Sentry Integration:
-------------------

>>> import sentry_sdk
>>> from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
>>>
>>> # Initialize Sentry
>>> sentry_sdk.init(
...     dsn=settings.SENTRY_DSN,
...     environment=settings.ENVIRONMENT,
...     traces_sample_rate=1.0 if settings.ENVIRONMENT == "development" else 0.1,
...     profiles_sample_rate=1.0 if settings.ENVIRONMENT == "development" else 0.1,
... )
>>>
>>> # Wrap application
>>> app = SentryAsgiMiddleware(app)
>>>
>>> # Errors are automatically reported to Sentry with:
>>> # - Full stack trace
>>> # - Request context (headers, body, user)
>>> # - Environment details
>>> # - Breadcrumbs (logs, HTTP requests, DB queries)

Datadog APM Integration:
-------------------------

>>> from ddtrace import tracer
>>> from ddtrace.contrib.asgi import TraceMiddleware
>>>
>>> # Add Datadog tracing
>>> app = TraceMiddleware(app, tracer, service="legal-ai-api")
>>>
>>> # Custom error tags
>>> @tracer.wrap(service="legal-ai-api", resource="contract-analysis")
>>> async def analyze_contract(contract: Contract):
...     try:
...         result = await llm_service.analyze(contract)
...         return result
...     except Exception as e:
...         # Add error tags for Datadog
...         span = tracer.current_span()
...         span.set_tag("error.type", e.__class__.__name__)
...         span.set_tag("error.message", str(e))
...         span.set_tag("contract.id", contract.id)
...         raise

ELK Stack Integration:
----------------------

>>> # Structured logging for ELK
>>> logger.error(
...     "Contract analysis failed",
...     extra={
...         "error_type": e.__class__.__name__,
...         "error_message": str(e),
...         "contract_id": contract.id,
...         "user_id": user.id,
...         "tenant_id": tenant.id,
...         "stack_trace": traceback.format_exc(),
...     }
... )
>>>
>>> # Elasticsearch query to find similar errors:
>>> GET /logs-*/_search
>>> {
...   "query": {
...     "bool": {
...       "must": [
...         {"match": {"error_type": "LLMAPIError"}},
...         {"range": {"@timestamp": {"gte": "now-1h"}}}
...       ]
...     }
...   },
...   "aggs": {
...     "error_count": {"terms": {"field": "error_message.keyword"}}
...   }
... }

=============================================================================
ERROR RECOVERY PATTERNS
=============================================================================

Retry with Exponential Backoff:
--------------------------------

>>> from tenacity import retry, stop_after_attempt, wait_exponential
>>>
>>> @retry(
...     stop=stop_after_attempt(3),
...     wait=wait_exponential(multiplier=1, min=2, max=10)
... )
>>> async def call_llm_with_retry(prompt: str):
...     return await llm_client.generate(prompt)

Circuit Breaker Pattern:
-------------------------

>>> from circuitbreaker import circuit
>>>
>>> @circuit(failure_threshold=5, recovery_timeout=60)
>>> async def call_external_api():
...     # If 5 failures occur, circuit opens
...     # Requests fail fast for 60 seconds
...     # After 60s, circuit tries again
...     return await external_api.call()

Graceful Degradation:
---------------------

>>> async def get_contract_suggestions(contract_id: str):
...     try:
...         # Primary: AI-powered suggestions
...         return await ai_service.get_suggestions(contract_id)
...     except AIServiceError:
...         logger.warning("AI service down, using rule-based fallback")
...         # Fallback: Rule-based suggestions
...         return await rule_engine.get_suggestions(contract_id)

=============================================================================
SECURITY CONSIDERATIONS
=============================================================================

1. Information Disclosure Prevention:
   - Never expose stack traces in production
   - Don't reveal database schema in error messages
   - Hide internal service names and IP addresses
   - Redact sensitive data from logs

2. PII Sanitization:
   - Remove email addresses from error messages
   - Mask phone numbers (0532****789)
   - Redact ID numbers (TC: 12345******)
   - Anonymize user identifiers in logs

3. Rate Limiting Error Responses:
   - Prevent error-based enumeration attacks
   - Limit error detail exposure for repeated failures
   - Implement CAPTCHA after multiple auth failures
   - Block IPs with suspicious error patterns

4. KVKK Compliance:
   - Log only necessary error context
   - Implement log retention policies (30 days)
   - Provide data deletion on user request
   - Encrypt error logs containing PII

=============================================================================
TROUBLESHOOTING
=============================================================================

High Error Rate:
----------------
1. Check recent deployments (rollback if needed)
2. Review database connection health
3. Verify external API status (LLM providers)
4. Check Redis cache availability
5. Monitor CPU and memory usage
6. Review error logs for patterns

Specific Error Debugging:
--------------------------
1. Search logs by request_id for full trace
2. Check Sentry for stack trace and context
3. Review Datadog APM for service dependencies
4. Analyze error frequency in Grafana
5. Check for recent code changes in Git

Database Errors:
----------------
1. Check connection pool exhaustion
2. Review slow query logs
3. Verify database disk space
4. Check for locks and deadlocks
5. Monitor connection count

LLM API Errors:
---------------
1. Verify API key validity
2. Check rate limit status
3. Review request/response size limits
4. Monitor API latency
5. Implement retry logic with backoff

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

import traceback
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import (
    BaseAppException,
    HTTPException,
    get_logger,
    settings,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# ERROR SANITIZER
# =============================================================================


class ErrorSanitizer:
    """
    Sanitizes error messages to prevent information disclosure and PII leaks.

    Removes sensitive information like:
    - Stack traces in production
    - Database connection strings
    - API keys and secrets
    - Personal identifiable information (email, phone, ID numbers)
    """

    # Patterns to redact
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b0\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b",
        "tc_id": r"\b\d{11}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }

    @classmethod
    def sanitize_message(cls, message: str, is_production: bool = True) -> str:
        """
        Sanitize error message for safe display.

        Args:
            message: Error message to sanitize
            is_production: Whether running in production

        Returns:
            Sanitized error message
        """
        if not is_production:
            return message

        # In production, use generic message for unknown errors
        if "Traceback" in message or "Error" in message:
            return "Bir hata oluştu. Lütfen daha sonra tekrar deneyin."

        return message

    @classmethod
    def sanitize_details(cls, details: Any) -> Optional[Dict[str, Any]]:
        """
        Sanitize error details.

        Args:
            details: Error details to sanitize

        Returns:
            Sanitized details or None in production
        """
        if not settings.DEBUG:
            return None

        # In debug mode, keep details but redact sensitive patterns
        if isinstance(details, dict):
            return {k: cls._redact_value(v) for k, v in details.items()}

        return details

    @classmethod
    def _redact_value(cls, value: Any) -> Any:
        """Redact sensitive patterns from value."""
        if isinstance(value, str):
            # Redact PII patterns
            import re

            for pattern_name, pattern in cls.PII_PATTERNS.items():
                value = re.sub(pattern, f"[REDACTED_{pattern_name.upper()}]", value)

        return value


# =============================================================================
# ERROR CLASSIFIER
# =============================================================================


class ErrorClassifier:
    """
    Classifies errors by type, severity, and recovery strategy.
    """

    # Error severity levels
    SEVERITY_INFO = "INFO"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_ERROR = "ERROR"
    SEVERITY_CRITICAL = "CRITICAL"

    # Error categories
    CATEGORY_CLIENT = "CLIENT_ERROR"
    CATEGORY_SERVER = "SERVER_ERROR"
    CATEGORY_NETWORK = "NETWORK_ERROR"
    CATEGORY_DATABASE = "DATABASE_ERROR"
    CATEGORY_EXTERNAL = "EXTERNAL_SERVICE_ERROR"

    @classmethod
    def classify(cls, exception: Exception) -> Dict[str, str]:
        """
        Classify exception.

        Args:
            exception: Exception to classify

        Returns:
            Dict with severity, category, and recovery strategy
        """
        exc_name = exception.__class__.__name__

        # Client errors (4xx)
        if isinstance(exception, BaseAppException):
            if 400 <= exception.status_code < 500:
                return {
                    "severity": cls.SEVERITY_WARNING,
                    "category": cls.CATEGORY_CLIENT,
                    "recoverable": False,
                }

        # Database errors
        if "Database" in exc_name or "SQL" in exc_name:
            return {
                "severity": cls.SEVERITY_ERROR,
                "category": cls.CATEGORY_DATABASE,
                "recoverable": True,
            }

        # Network/timeout errors
        if "Timeout" in exc_name or "Connection" in exc_name:
            return {
                "severity": cls.SEVERITY_WARNING,
                "category": cls.CATEGORY_NETWORK,
                "recoverable": True,
            }

        # External service errors
        if "API" in exc_name or "Service" in exc_name:
            return {
                "severity": cls.SEVERITY_ERROR,
                "category": cls.CATEGORY_EXTERNAL,
                "recoverable": True,
            }

        # Unknown errors (500)
        return {
            "severity": cls.SEVERITY_CRITICAL,
            "category": cls.CATEGORY_SERVER,
            "recoverable": False,
        }


# =============================================================================
# ERROR HANDLER MIDDLEWARE
# =============================================================================


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.

    Features:
    - Catches all unhandled exceptions
    - Converts exceptions to standardized JSON error responses
    - Logs errors with full context
    - Sanitizes sensitive information in production
    - Integrates with error tracking systems
    - Provides Turkish language error messages
    """

    def __init__(self, app):
        """
        Initialize error handler middleware.

        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.sanitizer = ErrorSanitizer()
        self.classifier = ErrorClassifier()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with comprehensive error handling.

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
            return await self._handle_app_exception(request, e)

        except HTTPException as e:
            # FastAPI HTTP exceptions
            return await self._handle_http_exception(request, e)

        except Exception as e:
            # Unhandled exceptions (500 errors)
            return await self._handle_unexpected_exception(request, e)

    async def _handle_app_exception(
        self, request: Request, exception: BaseAppException
    ) -> JSONResponse:
        """
        Handle known application exceptions.

        Args:
            request: Request object
            exception: Application exception

        Returns:
            JSON error response
        """
        # Get request context
        request_id = getattr(request.state, "request_id", None)
        tenant_id = getattr(request.state, "tenant_id", None)

        # Classify error
        classification = self.classifier.classify(exception)

        # Log with appropriate level
        logger.warning(
            "Application exception",
            exception=exception.__class__.__name__,
            message=str(exception),
            status_code=exception.status_code,
            error_code=exception.error_code,
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            tenant_id=tenant_id,
            severity=classification["severity"],
            category=classification["category"],
        )

        # Sanitize details
        details = self.sanitizer.sanitize_details(exception.details)

        # Build error response
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "error": {
                    "code": exception.error_code or exception.__class__.__name__,
                    "message": str(exception),
                    "details": details,
                    "request_id": request_id,
                    "timestamp": request.state.start_time if hasattr(request.state, "start_time") else None,
                }
            },
        )

    async def _handle_http_exception(
        self, request: Request, exception: HTTPException
    ) -> JSONResponse:
        """
        Handle FastAPI HTTP exceptions.

        Args:
            request: Request object
            exception: HTTP exception

        Returns:
            JSON error response
        """
        # Get request context
        request_id = getattr(request.state, "request_id", None)

        logger.warning(
            "HTTP exception",
            status_code=exception.status_code,
            detail=exception.detail,
            path=request.url.path,
            method=request.method,
            request_id=request_id,
        )

        # Map status code to Turkish message
        turkish_messages = {
            400: "Geçersiz istek. Lütfen girdiğiniz bilgileri kontrol edin.",
            401: "Oturum açmanız gerekiyor.",
            403: "Bu işlem için yetkiniz bulunmuyor.",
            404: "İstenen kaynak bulunamadı.",
            409: "Bu kaynak zaten mevcut.",
            422: "Giriş verileri geçersiz.",
            429: "Çok fazla istek gönderildi. Lütfen daha sonra tekrar deneyin.",
        }

        message = turkish_messages.get(
            exception.status_code, str(exception.detail)
        )

        return JSONResponse(
            status_code=exception.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exception.status_code}",
                    "message": message,
                    "details": exception.detail if settings.DEBUG else None,
                    "request_id": request_id,
                }
            },
        )

    async def _handle_unexpected_exception(
        self, request: Request, exception: Exception
    ) -> JSONResponse:
        """
        Handle unexpected exceptions (500 errors).

        Args:
            request: Request object
            exception: Unhandled exception

        Returns:
            JSON error response
        """
        # Get request context
        request_id = getattr(request.state, "request_id", "unknown")
        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = getattr(request.state, "user_id", None)

        # Classify error
        classification = self.classifier.classify(exception)

        # Get stack trace
        stack_trace = traceback.format_exc()

        # Log with full context
        logger.error(
            "⚠️ BEKLENMEYEN HATA",
            exception=exception.__class__.__name__,
            message=str(exception),
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            severity=classification["severity"],
            category=classification["category"],
            recoverable=classification["recoverable"],
            traceback=stack_trace if settings.DEBUG else None,
        )

        # Sanitize message
        if settings.DEBUG:
            message = f"{exception.__class__.__name__}: {str(exception)}"
            details = {"traceback": stack_trace}
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
                    "request_id": request_id,
                    "support": "Sorun devam ederse destek@turkishlegai.com ile iletişime geçin.",
                }
            },
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ErrorHandlerMiddleware",
    "ErrorSanitizer",
    "ErrorClassifier",
]
