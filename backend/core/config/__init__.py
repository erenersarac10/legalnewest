"""
Configuration management for Turkish Legal AI.

This module provides centralized configuration management using Pydantic Settings.
Configuration is loaded from:
1. Environment variables (.env file)
2. System environment
3. Default values (fallback)

Available configurations:
- settings: Main application settings
- database: Database configuration
- redis: Redis cache configuration
- s3: S3/MinIO storage configuration
- security: Security and cryptography
- logging: Logging configuration

Usage:
    >>> from backend.core.config import settings, db_config, redis_config
    >>> print(settings.APP_NAME)
    Turkish Legal AI
    >>> db_engine = db_config.create_write_engine()
"""

# =============================================================================
# MAIN SETTINGS
# =============================================================================

from backend.core.config.settings import (
    Settings,
    get_settings,
    settings,
)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

from backend.core.config.database import (
    DatabaseConfig,
    db_config,
    get_db_session,
    get_read_db_session,
)

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================

from backend.core.config.redis import (
    RedisConfig,
    cache_delete,
    cache_get,
    cache_set,
    get_redis,
    redis_config,
)

# =============================================================================
# S3 CONFIGURATION
# =============================================================================

from backend.core.config.s3 import (
    S3Config,
    s3_config,
)

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

from backend.core.config.security import (
    SecurityConfig,
    create_access_token,
    decode_token,
    hash_password,
    security_config,
    verify_password,
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

from backend.core.config.logging import (
    LoggingConfig,
    RequestLogger,
    clear_log_context,
    configure_logging,
    get_logger,
    logging_config,
    set_log_context,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Settings
    "Settings",
    "settings",
    "get_settings",
    # Database
    "DatabaseConfig",
    "db_config",
    "get_db_session",
    "get_read_db_session",
    # Redis
    "RedisConfig",
    "redis_config",
    "get_redis",
    "cache_set",
    "cache_get",
    "cache_delete",
    # S3
    "S3Config",
    "s3_config",
    # Security
    "SecurityConfig",
    "security_config",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_token",
    # Logging
    "LoggingConfig",
    "logging_config",
    "configure_logging",
    "get_logger",
    "set_log_context",
    "clear_log_context",
    "RequestLogger",
]

