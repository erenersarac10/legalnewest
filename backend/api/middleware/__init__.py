"""
Middleware package for Turkish Legal AI Platform.

Exports all middleware classes and configuration functions.

Author: Turkish Legal AI Team
License: Proprietary
"""

from backend.api.middleware.auth import AuthMiddleware
from backend.api.middleware.compression import configure_compression
from backend.api.middleware.cors import configure_cors
from backend.api.middleware.error_handler import ErrorHandlerMiddleware
from backend.api.middleware.request_id import RequestIDMiddleware
from backend.api.middleware.security_headers import SecurityHeadersMiddleware
from backend.api.middleware.tenant_context import TenantContextMiddleware
from backend.api.middleware.timing import TimingMiddleware

__all__ = [
    "AuthMiddleware",
    "TenantContextMiddleware",
    "RequestIDMiddleware",
    "TimingMiddleware",
    "ErrorHandlerMiddleware",
    "SecurityHeadersMiddleware",
    "configure_cors",
    "configure_compression",
]
