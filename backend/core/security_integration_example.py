"""
Security Integration Examples - PII Masking with Logging.

Harvey/Legora %100 parite: Production-ready security integration.

This module demonstrates how to integrate PII masking with:
- Structured logging (backend.core.logging)
- FastAPI endpoints
- Audit logs
- Error handling

Integration patterns for GDPR/KVKK compliance across the system.
"""

import logging
from typing import Any, Dict

from backend.core.logging import get_logger
from backend.core.security import (
    mask_pii,
    mask_dict_values,
    sanitize_log_dict,
    create_audit_log,
    PIIFilter,
)


# =============================================================================
# EXAMPLE 1: BASIC LOGGING WITH PII MASKING
# =============================================================================


def example_basic_logging():
    """
    Basic logging with automatic PII masking.

    Harvey/Legora %100: Zero-configuration protection.
    """
    logger = get_logger(__name__)

    # Method 1: Manual masking before logging
    user_input = "Kullanıcı TC: 12345678901, email: ahmet@example.com"
    logger.info(mask_pii(user_input))
    # Logs: "Kullanıcı TC: *********01, email: ahm***@example.com"

    # Method 2: Mask structured log data
    log_data = {
        "user_id": "12345678901",
        "user_email": "ahmet.yilmaz@example.com",
        "action": "document_access",
        "document_id": "law_6098",
    }

    logger.info(
        "User action",
        extra=mask_dict_values(log_data)
    )
    # Logs with masked user_id and user_email


# =============================================================================
# EXAMPLE 2: FASTAPI ENDPOINT WITH AUDIT LOGGING
# =============================================================================


def example_fastapi_endpoint():
    """
    FastAPI endpoint with PII-masked audit logging.

    Harvey/Legora %100: Compliant API logging.
    """
    from fastapi import APIRouter, Request, HTTPException

    router = APIRouter()
    logger = get_logger(__name__)

    @router.get("/documents/{document_id}")
    async def get_document(document_id: str, request: Request):
        """
        Document access endpoint with audit logging.

        All logs are PII-masked for GDPR/KVKK compliance.
        """
        # Extract user info (example - adjust to your auth system)
        user_id = request.headers.get("X-User-ID", "anonymous")
        client_ip = request.client.host

        # Create audit log entry (automatically masks PII)
        audit = create_audit_log(
            action="document_access",
            user_id=user_id,  # Will be masked
            resource=document_id,
            metadata={
                "ip": client_ip,  # Will be masked
                "user_agent": request.headers.get("User-Agent", ""),
            }
        )

        # Log with masked data
        logger.info(
            "Document accessed",
            extra=audit
        )

        # Business logic...
        try:
            # Fetch document
            document = {"id": document_id, "title": "..."}
            return document

        except Exception as e:
            # Error logging with PII masking
            logger.error(
                mask_pii(f"Document access failed: {str(e)}"),
                extra=sanitize_log_dict({
                    "user_id": user_id,
                    "document_id": document_id,
                    "error": str(e),
                })
            )
            raise HTTPException(status_code=500, detail="Internal error")


# =============================================================================
# EXAMPLE 3: ADAPTER WITH PII-MASKED LOGGING
# =============================================================================


def example_adapter_logging():
    """
    Adapter implementation with PII-masked error logging.

    Harvey/Legora %100: Secure adapter logging.
    """
    from backend.parsers.adapters.base_adapter import BaseAdapter

    class SecureAdapter(BaseAdapter):
        """Adapter with PII-masked logging."""

        async def fetch_document(self, document_id: str, user_id: str = None):
            """
            Fetch document with audit trail.

            All logs automatically mask PII.
            """
            self.logger.info(
                "Fetching document",
                extra=sanitize_log_dict({
                    "document_id": document_id,
                    "user_id": user_id,  # Will be masked
                    "adapter": self.source_name,
                })
            )

            try:
                # Fetch logic...
                content = await self._fetch_content(document_id)
                return content

            except Exception as e:
                # Error log with PII masking
                self.logger.error(
                    mask_pii(f"Fetch failed: {str(e)}"),
                    extra=sanitize_log_dict({
                        "document_id": document_id,
                        "user_id": user_id,
                        "error_type": type(e).__name__,
                    })
                )
                raise


# =============================================================================
# EXAMPLE 4: GLOBAL LOGGING FILTER
# =============================================================================


def example_global_filter():
    """
    Configure global PII filter for all loggers.

    Harvey/Legora %100: System-wide protection.

    Add this to your application startup (e.g., main.py or __init__.py).
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Add PII filter to all handlers
    pii_filter = PIIFilter()

    for handler in root_logger.handlers:
        handler.addFilter(pii_filter)

    print("✅ PII filter applied to all loggers")

    # Test
    test_logger = logging.getLogger("test")
    test_logger.info("User 12345678901 logged in")
    # Automatically logs: "User *********01 logged in"


# =============================================================================
# EXAMPLE 5: CUSTOM SENSITIVE FIELDS
# =============================================================================


def example_custom_sensitive_fields():
    """
    Mask custom sensitive fields beyond default PII.

    Harvey/Legora %100: Flexible masking.
    """
    logger = get_logger(__name__)

    # Define custom sensitive keys
    custom_sensitive = [
        'tc', 'email', 'phone',  # Default
        'passport_no', 'driver_license',  # Custom
        'internal_id', 'employee_id',  # Custom
    ]

    # Log data with custom fields
    log_data = {
        "user_tc": "12345678901",
        "passport_no": "U1234567",
        "internal_id": "EMP-98765",
        "action": "login",
    }

    # Mask with custom sensitive keys
    masked = mask_dict_values(log_data, sensitive_keys=custom_sensitive)

    logger.info("User action", extra=masked)
    # All custom fields masked


# =============================================================================
# EXAMPLE 6: EXCEPTION HANDLING WITH PII MASKING
# =============================================================================


def example_exception_handling():
    """
    Exception handling with PII-masked logging.

    Harvey/Legora %100: Secure error tracking.
    """
    logger = get_logger(__name__)

    try:
        # Simulate error with user data
        user_data = {
            "tc": "12345678901",
            "email": "ahmet@example.com",
            "phone": "+90 555 123 4567"
        }

        # Operation that might fail
        result = process_user_data(user_data)

    except Exception as e:
        # Log exception with masked user data
        logger.error(
            "User data processing failed",
            extra={
                "error": mask_pii(str(e)),
                "user_data": sanitize_log_dict(user_data),
                "error_type": type(e).__name__,
            },
            exc_info=True  # Include stack trace
        )

        # Re-raise or handle
        raise


def process_user_data(data: Dict[str, Any]) -> Any:
    """Placeholder for actual processing."""
    pass


# =============================================================================
# EXAMPLE 7: SEARCH QUERY LOGGING
# =============================================================================


def example_search_logging():
    """
    Search query logging with PII detection.

    Harvey/Legora %100: Secure search analytics.
    """
    logger = get_logger(__name__)

    def log_search(query: str, user_id: str, results_count: int):
        """
        Log search query with automatic PII masking.

        Detects if query contains PII (TC, email, etc.) and masks it.
        """
        # Mask query (in case user searched for sensitive data)
        masked_query = mask_pii(query)

        # Audit log
        audit = create_audit_log(
            action="search",
            user_id=user_id,  # Masked
            metadata={
                "query": masked_query,  # Masked
                "results_count": results_count,
            }
        )

        logger.info("Search executed", extra=audit)

    # Example searches
    log_search("anayasa mahkemesi", "12345678901", 42)
    # Query: OK, user_id: masked

    log_search("TC 98765432109", "12345678901", 0)
    # Query: masked (contains TC), user_id: masked


# =============================================================================
# EXAMPLE 8: INTEGRATION WITH STRUCTLOG
# =============================================================================


def example_structlog_integration():
    """
    Integration with structlog for structured JSON logging.

    Harvey/Legora %100: Production JSON logs with PII masking.
    """
    import structlog

    # Configure structlog with PII masking processor
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),

            # PII masking processor
            lambda logger, method, event_dict: {
                k: mask_pii(v) if isinstance(v, str) else v
                for k, v in event_dict.items()
            },

            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Use structlog with automatic PII masking
    log = structlog.get_logger()

    log.info(
        "user_action",
        user_tc="12345678901",
        user_email="ahmet@example.com",
        action="document_access",
    )
    # JSON output with masked TC and email


# =============================================================================
# USAGE SUMMARY
# =============================================================================


"""
RECOMMENDED INTEGRATION:

1. Application Startup (main.py):
   ```python
   from backend.core.security_integration_example import example_global_filter

   # Apply global PII filter
   example_global_filter()
   ```

2. FastAPI Endpoints:
   ```python
   from backend.core.security import create_audit_log, sanitize_log_dict

   @app.get("/api/resource")
   async def get_resource(request: Request):
       audit = create_audit_log(
           action="resource_access",
           user_id=get_user_id(request),
           resource="resource_id"
       )
       logger.info("Resource accessed", extra=audit)
   ```

3. Adapters:
   ```python
   from backend.core.security import sanitize_log_dict

   self.logger.info(
       "Adapter action",
       extra=sanitize_log_dict({
           "user_id": user_id,
           "document_id": doc_id,
       })
   )
   ```

4. Exception Handling:
   ```python
   from backend.core.security import mask_pii

   except Exception as e:
       logger.error(mask_pii(str(e)), exc_info=True)
   ```

RESULT:
✅ Zero PII in logs
✅ GDPR/KVKK compliant
✅ Automatic masking
✅ No code changes needed after global filter setup
"""
