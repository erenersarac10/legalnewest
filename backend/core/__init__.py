"""
Core module for Turkish Legal AI Backend.

This module provides foundational components:
- Configuration management
- Database utilities
- Cache management
- Security utilities
- Logging configuration
- Exception handling
- Common utilities
- Subscription & licensing
- Token optimization
- Source verification

All core functionality should be imported from here for consistency.

Usage:
    >>> from backend.core import settings, get_logger
    >>> from backend.core import cache, get_session
    >>> from backend.core import NotFoundException
    >>>
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started", version=settings.APP_VERSION)
"""

# =============================================================================
# VERSION INFO
# =============================================================================

from backend.core.version import (
    __version__,
    __version_info__,
)

# =============================================================================
# CONSTANTS
# =============================================================================

from backend.core.constants import *  # noqa: F403

# =============================================================================
# CONFIGURATION
# =============================================================================

from backend.core.config import (
    cache_delete,
    cache_get,
    cache_set,
    clear_log_context,
    configure_logging,
    create_access_token,
    db_config,
    decode_token,
    get_logger,
    get_redis,
    get_settings,
    hash_password,
    logging_config,
    redis_config,
    s3_config,
    security_config,
    set_log_context,
    settings,
    verify_password,
)

# =============================================================================
# DATABASE
# =============================================================================

from backend.core.database import (
    AsyncSession,
    Base,
    BaseModelMixin,
    DatabaseSession,
    FullAuditMixin,
    TenantModelMixin,
    get_read_session,
    get_session,
)

# =============================================================================
# CACHE
# =============================================================================

from backend.core.cache import (
    RedisCache,
    cache,
    cache_invalidate,
    cached,
    cached_method,
    rate_limit,
)

# =============================================================================
# SUBSCRIPTION & LICENSING
# =============================================================================

from backend.core.database.models import (
    Subscription,
    SubscriptionPlan,
    UsageQuota,
    UsageTracking,
    Payment,
    Invoice,
    PaymentMethod,
    BillingCycle,
    Company,
    License,
    LicenseAssignment,
    CompanyAdmin,
    EmployeeInvite,
)

# =============================================================================
# TOKEN OPTIMIZATION
# =============================================================================

from backend.core.database.models import (
    TokenUsage,
    CompressionLog,
    CacheEntry,
    OptimizationStrategy,
)

# =============================================================================
# SECURITY & VERIFICATION
# =============================================================================

from backend.core.database.models import (
    SourceVerification,
    PiiDetection,
    DataMasking,
    VerificationRule,
    SecurityIncident,
)

# =============================================================================
# EXCEPTIONS
# =============================================================================

from backend.core.exceptions import (
    BadRequestException,
    BaseAppException,
    BusinessLogicException,
    CacheException,
    ConflictException,
    DatabaseException,
    DocumentException,
    DocumentInvalidFormatException,
    DocumentParseException,
    DocumentTooLargeException,
    ForbiddenException,
    HTTPException,
    InsufficientPermissionsException,
    InternalServerException,
    InvalidIBANException,
    InvalidTCNoException,
    InvalidTokenException,
    KVKKComplianceException,
    LLMException,
    LLMInvalidResponseException,
    LLMQuotaExceededException,
    LLMTimeoutException,
    NotFoundException,
    OCRException,
    SecurityException,
    ServiceUnavailableException,
    TokenExpiredException,
    TooManyRequestsException,
    TurkishLegalException,
    UnauthorizedException,
    UnprocessableEntityException,
    ValidationException,
)

# =============================================================================
# UTILITIES
# =============================================================================

from backend.core.utils import (
    datetime_to_timestamp,
    format_datetime,
    format_file_size,
    generate_random_token,
    generate_uuid,
    get_file_extension,
    hash_string,
    is_allowed_file_type,
    is_valid_email,
    is_valid_url,
    is_valid_uuid,
    normalize_whitespace,
    safe_json_loads,
    slugify,
    timestamp_to_datetime,
    truncate,
    utc_now,
    validate_iban,
    validate_tc_no,
    validate_turkish_phone,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Settings
    "settings",
    "get_settings",
    # Config Managers
    "db_config",
    "redis_config",
    "s3_config",
    "security_config",
    "logging_config",
    # Logging
    "configure_logging",
    "get_logger",
    "set_log_context",
    "clear_log_context",
    # Security
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_token",
    # Database
    "Base",
    "AsyncSession",
    "BaseModelMixin",
    "FullAuditMixin",
    "TenantModelMixin",
    "get_session",
    "get_read_session",
    "DatabaseSession",
    # Cache
    "cache",
    "RedisCache",
    "cached",
    "cache_invalidate",
    "cached_method",
    "rate_limit",
    "get_redis",
    "cache_set",
    "cache_get",
    "cache_delete",
    # Subscription & Licensing
    "Subscription",
    "SubscriptionPlan",
    "UsageQuota",
    "UsageTracking",
    "Payment",
    "Invoice",
    "PaymentMethod",
    "BillingCycle",
    "Company",
    "License",
    "LicenseAssignment",
    "CompanyAdmin",
    "EmployeeInvite",
    # Token Optimization
    "TokenUsage",
    "CompressionLog",
    "CacheEntry",
    "OptimizationStrategy",
    # Security & Verification
    "SourceVerification",
    "PiiDetection",
    "DataMasking",
    "VerificationRule",
    "SecurityIncident",
    # Exceptions
    "BaseAppException",
    "HTTPException",
    "BadRequestException",
    "UnauthorizedException",
    "ForbiddenException",
    "NotFoundException",
    "ConflictException",
    "UnprocessableEntityException",
    "TooManyRequestsException",
    "InternalServerException",
    "ServiceUnavailableException",
    "DatabaseException",
    "CacheException",
    "SecurityException",
    "InvalidTokenException",
    "TokenExpiredException",
    "InsufficientPermissionsException",
    "BusinessLogicException",
    "ValidationException",
    "DocumentException",
    "DocumentTooLargeException",
    "DocumentInvalidFormatException",
    "DocumentParseException",
    "OCRException",
    "LLMException",
    "LLMTimeoutException",
    "LLMQuotaExceededException",
    "LLMInvalidResponseException",
    "TurkishLegalException",
    "InvalidTCNoException",
    "InvalidIBANException",
    "KVKKComplianceException",
    # Utilities
    "generate_uuid",
    "is_valid_uuid",
    "utc_now",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "format_datetime",
    "slugify",
    "truncate",
    "normalize_whitespace",
    "validate_tc_no",
    "validate_iban",
    "validate_turkish_phone",
    "hash_string",
    "generate_random_token",
    "get_file_extension",
    "is_allowed_file_type",
    "format_file_size",
    "safe_json_loads",
    "is_valid_email",
    "is_valid_url",
]