"""
Configuration Management - Harvey/Legora %100 Production Settings.

Environment-based configuration with feature flags:
- RBAC feature toggle
- Cache configuration
- Database settings
- API keys and secrets
- Performance tuning

Why Feature Flags?
    Without: Hard-coded features → difficult rollout/rollback
    With: Runtime toggles → gradual rollout → Harvey-level reliability

    Impact: Zero-downtime feature deployment! ⚙️

Feature Flags:
    - RBAC_ENABLED: Enable/disable RBAC (default: True)
    - CACHE_ENABLED: Enable/disable caching (default: True)
    - VECTOR_SEARCH_ENABLED: Enable/disable vector search (default: False)
    - CITATION_GRAPH_ENABLED: Enable/disable citation graph (default: False)

Configuration Sources (priority order):
    1. Environment variables
    2. .env file
    3. Default values

Usage:
    >>> from backend.core.config import settings
    >>>
    >>> if settings.RBAC_ENABLED:
    ...     # Enforce RBAC
    ...     check_permission(user, "documents:read")
    >>> else:
    ...     # Skip RBAC (development mode)
    ...     pass
"""

import os
from typing import Optional, List
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    Harvey/Legora %100: Production-ready configuration management.

    All settings can be overridden via environment variables.
    """

    # ==========================================================================
    # APPLICATION
    # ==========================================================================

    APP_NAME: str = "Turkish Legal AI"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"  # development, staging, production
    DEBUG: bool = False

    # ==========================================================================
    # FEATURE FLAGS (Harvey/Legora %100)
    # ==========================================================================

    # RBAC Feature Flag
    RBAC_ENABLED: bool = True
    RBAC_STRICT_MODE: bool = True  # If True, deny by default; if False, allow by default

    # Cache Feature Flag
    CACHE_ENABLED: bool = True
    CACHE_WARMING_ENABLED: bool = True
    CACHE_WARMING_ON_STARTUP: bool = True
    CACHE_WARMING_LIMIT: int = 1000

    # Vector Search Feature Flag
    VECTOR_SEARCH_ENABLED: bool = False
    VECTOR_DB_PROVIDER: str = "weaviate"  # weaviate, pinecone

    # Citation Graph Feature Flag
    CITATION_GRAPH_ENABLED: bool = False

    # Audit Logging Feature Flag
    AUDIT_LOGGING_ENABLED: bool = True
    AUDIT_BATCH_SIZE: int = 50
    AUDIT_HASH_CHAIN_ENABLED: bool = True

    # ==========================================================================
    # DATABASE
    # ==========================================================================

    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/legal_ai"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30

    # ==========================================================================
    # REDIS
    # ==========================================================================

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 300  # 5 minutes
    REDIS_PUBSUB_ENABLED: bool = True

    # ==========================================================================
    # NEO4J (Citation Graph)
    # ==========================================================================

    NEO4J_URI: str = "neo4j://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # ==========================================================================
    # WEAVIATE (Vector Search)
    # ==========================================================================

    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: Optional[str] = None

    # ==========================================================================
    # ELASTICSEARCH
    # ==========================================================================

    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_PREFIX: str = "legal_ai"

    # ==========================================================================
    # OPENAI
    # ==========================================================================

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536

    # ==========================================================================
    # JWT / AUTHENTICATION
    # ==========================================================================

    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Password Policy
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True

    # Rate Limiting
    RATE_LIMIT_LOGIN_MAX: int = 10
    RATE_LIMIT_LOGIN_WINDOW_SECONDS: int = 60
    RATE_LIMIT_REGISTER_MAX: int = 5
    RATE_LIMIT_REGISTER_WINDOW_SECONDS: int = 300

    # ==========================================================================
    # CORS
    # ==========================================================================

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # ==========================================================================
    # LOGGING
    # ==========================================================================

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json, text
    LOG_FILE: Optional[str] = None

    # ==========================================================================
    # PROMETHEUS METRICS
    # ==========================================================================

    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090

    # ==========================================================================
    # HEALTH CHECKS
    # ==========================================================================

    HEALTH_CHECK_ENABLED: bool = True
    STARTUP_TIMEOUT_SECONDS: int = 300  # 5 minutes

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance.

    Returns:
        Settings: Application settings

    Example:
        >>> from backend.core.config import get_settings
        >>>
        >>> settings = get_settings()
        >>> if settings.RBAC_ENABLED:
        ...     # Enforce RBAC
        ...     pass
    """
    global _settings

    if _settings is None:
        _settings = Settings()

    return _settings


# Convenience: settings instance
settings = get_settings()


# =============================================================================
# FEATURE FLAG HELPERS
# =============================================================================


def is_rbac_enabled() -> bool:
    """
    Check if RBAC is enabled.

    Returns:
        bool: True if RBAC enabled

    Example:
        >>> if is_rbac_enabled():
        ...     check_permission()
    """
    return settings.RBAC_ENABLED


def is_cache_enabled() -> bool:
    """
    Check if caching is enabled.

    Returns:
        bool: True if cache enabled
    """
    return settings.CACHE_ENABLED


def is_vector_search_enabled() -> bool:
    """
    Check if vector search is enabled.

    Returns:
        bool: True if vector search enabled
    """
    return settings.VECTOR_SEARCH_ENABLED


def is_citation_graph_enabled() -> bool:
    """
    Check if citation graph is enabled.

    Returns:
        bool: True if citation graph enabled
    """
    return settings.CITATION_GRAPH_ENABLED


def is_audit_logging_enabled() -> bool:
    """
    Check if audit logging is enabled.

    Returns:
        bool: True if audit logging enabled
    """
    return settings.AUDIT_LOGGING_ENABLED


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================


def is_production() -> bool:
    """Check if running in production."""
    return settings.APP_ENV == "production"


def is_development() -> bool:
    """Check if running in development."""
    return settings.APP_ENV == "development"


def is_staging() -> bool:
    """Check if running in staging."""
    return settings.APP_ENV == "staging"


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "is_rbac_enabled",
    "is_cache_enabled",
    "is_vector_search_enabled",
    "is_citation_graph_enabled",
    "is_audit_logging_enabled",
    "is_production",
    "is_development",
    "is_staging",
]
