"""
CORS Configuration - Harvey/Legora %100 Security Policy.

Secure cross-origin resource sharing for:
- Web frontend (React/Next.js)
- Mobile apps (iOS/Android)
- Third-party integrations
- Development/staging/production environments

Why CORS?
    Without: XSS attacks, unauthorized origins
    With: Whitelist-based security → Harvey-level protection

    Impact: OWASP Top 10 compliance! 🔒
"""

from typing import List
from enum import Enum


class CORSEnvironment(str, Enum):
    """CORS environment profiles."""

    DEVELOPMENT = "development"  # Localhost only
    STAGING = "staging"  # Staging domains
    PRODUCTION = "production"  # Production domains only


# Harvey/Legora %100: Environment-specific CORS policies
CORS_POLICIES = {
    CORSEnvironment.DEVELOPMENT: {
        "allow_origins": [
            "http://localhost:3000",  # React dev server
            "http://localhost:8000",  # Backend dev
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ],
        "allow_credentials": True,
        "allow_methods": ["*"],  # All methods in dev
        "allow_headers": ["*"],  # All headers in dev
        "max_age": 600,  # 10 minutes
    },
    CORSEnvironment.STAGING: {
        "allow_origins": [
            "https://staging.legalai.com.tr",
            "https://staging-app.legalai.com.tr",
        ],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-Tenant-ID",
        ],
        "max_age": 3600,  # 1 hour
    },
    CORSEnvironment.PRODUCTION: {
        "allow_origins": [
            "https://legalai.com.tr",
            "https://app.legalai.com.tr",
            "https://www.legalai.com.tr",
        ],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],  # No OPTIONS (preflight only)
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-Tenant-ID",
            "X-API-Key",
        ],
        "expose_headers": [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
        "max_age": 86400,  # 24 hours (reduce preflight requests)
    },
}


def get_cors_config(environment: str = "production") -> dict:
    """Get CORS config for environment."""
    env = CORSEnvironment(environment.lower())
    return CORS_POLICIES[env]


__all__ = ["CORSEnvironment", "CORS_POLICIES", "get_cors_config"]
