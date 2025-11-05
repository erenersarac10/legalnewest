"""
Global constants for Turkish Legal AI.

This module defines application-wide constants that are used across
all modules. These constants are immutable and provide a single source
of truth for configuration values, limits, and enumerations.

Constants are organized by category:
- Application metadata
- HTTP & API
- Database
- Security
- Document processing
- Rate limiting
- Turkish legal specifics
- Time & scheduling
"""
from typing import Final

from backend.core.version import (
    API_VERSION,
    CODENAME,
    __version__,
)

# =============================================================================
# APPLICATION METADATA
# =============================================================================

APP_NAME: Final[str] = "Turkish Legal AI"
"""Application display name."""

APP_SLUG: Final[str] = "turkish-legal-ai"
"""Application slug for URLs and identifiers."""

APP_DESCRIPTION: Final[str] = (
    "AI-powered legal assistant for Turkish law - The Harvey AI of Turkey"
)
"""Short application description."""

APP_VERSION: Final[str] = __version__
"""Current application version (from version.py)."""

APP_CODENAME: Final[str] = CODENAME
"""Release codename (from version.py)."""

# =============================================================================
# HTTP & API CONFIGURATION
# =============================================================================

# API Versioning
API_V1_STR: Final[str] = f"/api/{API_VERSION}"
"""API v1 prefix: /api/v1"""

API_V2_STR: Final[str] = "/api/v2"
"""API v2 prefix (future): /api/v2"""

API_BETA_STR: Final[str] = "/api/beta"
"""Beta API prefix: /api/beta"""

# HTTP Status Codes (commonly used)
HTTP_OK: Final[int] = 200
HTTP_CREATED: Final[int] = 201
HTTP_NO_CONTENT: Final[int] = 204
HTTP_BAD_REQUEST: Final[int] = 400
HTTP_UNAUTHORIZED: Final[int] = 401
HTTP_FORBIDDEN: Final[int] = 403
HTTP_NOT_FOUND: Final[int] = 404
HTTP_CONFLICT: Final[int] = 409
HTTP_UNPROCESSABLE_ENTITY: Final[int] = 422
HTTP_TOO_MANY_REQUESTS: Final[int] = 429
HTTP_INTERNAL_SERVER_ERROR: Final[int] = 500
HTTP_SERVICE_UNAVAILABLE: Final[int] = 503

# Request Limits
MAX_REQUEST_SIZE_MB: Final[int] = 100
"""Maximum HTTP request body size in megabytes."""

MAX_REQUEST_TIMEOUT_SECONDS: Final[int] = 300
"""Maximum request timeout (5 minutes)."""

MAX_CONCURRENT_REQUESTS: Final[int] = 1000
"""Maximum concurrent requests per instance."""

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Connection Limits
DEFAULT_DB_POOL_SIZE: Final[int] = 20
"""Default database connection pool size."""

DEFAULT_DB_MAX_OVERFLOW: Final[int] = 10
"""Maximum overflow connections."""

DEFAULT_DB_POOL_TIMEOUT: Final[int] = 30
"""Pool timeout in seconds."""

# Query Limits
DEFAULT_PAGE_SIZE: Final[int] = 50
"""Default pagination page size."""

MAX_PAGE_SIZE: Final[int] = 1000
"""Maximum pagination page size."""

MIN_PAGE_SIZE: Final[int] = 1
"""Minimum pagination page size."""

# Transaction Timeouts
DEFAULT_TRANSACTION_TIMEOUT: Final[int] = 30
"""Default transaction timeout in seconds."""

LONG_TRANSACTION_TIMEOUT: Final[int] = 300
"""Long-running transaction timeout (5 minutes)."""

# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Password Requirements
PASSWORD_MIN_LENGTH: Final[int] = 12
"""Minimum password length (KVKK/GDPR compliant)."""

PASSWORD_MAX_LENGTH: Final[int] = 128
"""Maximum password length."""

PASSWORD_BCRYPT_ROUNDS: Final[int] = 12
"""BCrypt work factor (2^12 iterations)."""

PASSWORD_ARGON2_TIME_COST: Final[int] = 3
"""Argon2 time cost parameter."""

PASSWORD_ARGON2_MEMORY_COST: Final[int] = 65536
"""Argon2 memory cost (64 MB)."""

PASSWORD_ARGON2_PARALLELISM: Final[int] = 4
"""Argon2 parallelism parameter."""

# JWT Configuration
JWT_ALGORITHM_DEFAULT: Final[str] = "RS256"
"""Default JWT signing algorithm."""

JWT_ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = 30
"""Access token expiration time."""

JWT_REFRESH_TOKEN_EXPIRE_DAYS: Final[int] = 7
"""Refresh token expiration time."""

JWT_ISSUER: Final[str] = "turkish-legal-ai"
"""JWT issuer claim."""

# Session Management
SESSION_COOKIE_NAME: Final[str] = "legal_ai_session"
"""Session cookie name."""

SESSION_MAX_AGE_SECONDS: Final[int] = 86400
"""Session max age (24 hours)."""

SESSION_IDLE_TIMEOUT_MINUTES: Final[int] = 30
"""Session idle timeout."""

# API Keys
API_KEY_LENGTH: Final[int] = 64
"""API key length in characters."""

API_KEY_PREFIX: Final[str] = "la_"
"""API key prefix: la_..."""

# =============================================================================
# RATE LIMITING
# =============================================================================

# Per-User Limits
RATE_LIMIT_PER_MINUTE: Final[int] = 60
"""Default requests per minute per user."""

RATE_LIMIT_PER_HOUR: Final[int] = 1000
"""Default requests per hour per user."""

RATE_LIMIT_PER_DAY: Final[int] = 10000
"""Default requests per day per user."""

# Per-Tenant Limits (SaaS)
TENANT_RATE_LIMIT_PER_MINUTE: Final[int] = 600
"""Requests per minute per tenant."""

TENANT_RATE_LIMIT_PER_DAY: Final[int] = 100000
"""Requests per day per tenant."""

# LLM-Specific Limits
LLM_RATE_LIMIT_PER_MINUTE: Final[int] = 20
"""LLM API calls per minute per user."""

LLM_RATE_LIMIT_PER_HOUR: Final[int] = 200
"""LLM API calls per hour per user."""

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

# File Size Limits
MAX_DOCUMENT_SIZE_MB: Final[int] = 100
"""Maximum document upload size."""

MAX_BATCH_DOCUMENT_SIZE_MB: Final[int] = 500
"""Maximum batch document upload size."""

# Document Limits
MAX_PAGES_PER_DOCUMENT: Final[int] = 500
"""Maximum pages per document."""

MAX_DOCUMENTS_PER_BATCH: Final[int] = 50
"""Maximum documents per batch upload."""

MAX_DOCUMENTS_PER_USER: Final[int] = 10000
"""Maximum documents per user account."""

# Processing Timeouts
DOCUMENT_PARSE_TIMEOUT_SECONDS: Final[int] = 300
"""Document parsing timeout (5 minutes)."""

OCR_TIMEOUT_SECONDS: Final[int] = 600
"""OCR processing timeout (10 minutes)."""

EMBEDDING_TIMEOUT_SECONDS: Final[int] = 600
"""Document embedding timeout (10 minutes)."""

# Supported Formats
SUPPORTED_DOCUMENT_TYPES: Final[tuple[str, ...]] = (
    "pdf",
    "docx",
    "doc",
    "txt",
    "rtf",
    "odt",
)
"""Supported document file extensions."""

SUPPORTED_IMAGE_TYPES: Final[tuple[str, ...]] = (
    "jpg",
    "jpeg",
    "png",
    "tiff",
    "bmp",
)
"""Supported image file extensions for OCR."""

# =============================================================================
# RAG (RETRIEVAL-AUGMENTED GENERATION)
# =============================================================================

# Chunking
DEFAULT_CHUNK_SIZE: Final[int] = 1000
"""Default text chunk size in characters."""

DEFAULT_CHUNK_OVERLAP: Final[int] = 200
"""Default chunk overlap in characters."""

MIN_CHUNK_SIZE: Final[int] = 100
"""Minimum chunk size."""

MAX_CHUNK_SIZE: Final[int] = 4000
"""Maximum chunk size."""

# Retrieval
DEFAULT_TOP_K: Final[int] = 5
"""Default number of retrieved chunks."""

MAX_TOP_K: Final[int] = 50
"""Maximum number of retrieved chunks."""

DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7
"""Default similarity score threshold."""

# Reranking
RERANK_TOP_K: Final[int] = 20
"""Number of candidates for reranking."""

# Hybrid Search
HYBRID_SEARCH_ALPHA: Final[float] = 0.5
"""Hybrid search balance (0=keyword, 1=semantic)."""

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Token Limits
MAX_PROMPT_TOKENS: Final[int] = 8000
"""Maximum tokens in prompt."""

MAX_COMPLETION_TOKENS: Final[int] = 4096
"""Maximum tokens in completion."""

MAX_TOTAL_TOKENS: Final[int] = 12000
"""Maximum total tokens (prompt + completion)."""

# Context Windows (by model)
GPT4_CONTEXT_WINDOW: Final[int] = 128000
"""GPT-4 Turbo context window."""

CLAUDE_CONTEXT_WINDOW: Final[int] = 200000
"""Claude 3 context window."""

GEMINI_CONTEXT_WINDOW: Final[int] = 1000000
"""Gemini Pro context window."""

# Generation Parameters
DEFAULT_TEMPERATURE: Final[float] = 0.7
"""Default LLM temperature."""

DEFAULT_TOP_P: Final[float] = 0.9
"""Default nucleus sampling threshold."""

# Timeouts
LLM_REQUEST_TIMEOUT_SECONDS: Final[int] = 60
"""LLM API request timeout."""

LLM_STREAMING_TIMEOUT_SECONDS: Final[int] = 300
"""LLM streaming timeout (5 minutes)."""

# Cost Limits
MAX_COST_PER_REQUEST_USD: Final[float] = 1.0
"""Maximum cost per single LLM request."""

# =============================================================================
# TURKISH LEGAL CONSTANTS
# =============================================================================

# Turkish Identification
TC_NO_LENGTH: Final[int] = 11
"""Turkish National ID (TC Kimlik No) length."""

TAX_ID_LENGTH: Final[int] = 10
"""Turkish Tax ID (Vergi Kimlik No) length."""

# Legal Document Types
TURKISH_LEGAL_DOCUMENT_TYPES: Final[tuple[str, ...]] = (
    "kanun",           # Law
    "kanun_hukmunde_kararname",  # Decree with force of law
    "tuzuk",           # Regulation
    "yonetmelik",      # By-law
    "teblig",          # Communiqué
    "genelge",         # Circular
    "yargitay_karari", # Supreme Court decision
    "danistay_karari", # Council of State decision
    "anayasa_mahkemesi_karari",  # Constitutional Court decision
)
"""Turkish legal document types."""

# Official Sources
RESMI_GAZETE_BASE_URL: Final[str] = "https://www.resmigazete.gov.tr"
"""Official Gazette (Resmi Gazete) base URL."""

MEVZUAT_GOV_BASE_URL: Final[str] = "https://www.mevzuat.gov.tr"
"""Legislation portal base URL."""

YARGITAY_BASE_URL: Final[str] = "https://karararama.yargitay.gov.tr"
"""Supreme Court decisions base URL."""

DANISTAY_BASE_URL: Final[str] = "https://karararama.danistay.gov.tr"
"""Council of State decisions base URL."""

KVKK_BASE_URL: Final[str] = "https://www.kvkk.gov.tr"
"""Personal Data Protection Authority base URL."""

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# TTL (Time To Live)
CACHE_SHORT_TTL: Final[int] = 300
"""Short-lived cache (5 minutes)."""

CACHE_MEDIUM_TTL: Final[int] = 3600
"""Medium-lived cache (1 hour)."""

CACHE_LONG_TTL: Final[int] = 86400
"""Long-lived cache (24 hours)."""

CACHE_PERMANENT_TTL: Final[int] = -1
"""Permanent cache (no expiration)."""

# Cache Keys
CACHE_KEY_PREFIX: Final[str] = "legal_ai"
"""Cache key namespace prefix."""

CACHE_VERSION: Final[str] = "v1"
"""Cache version for invalidation."""

# =============================================================================
# LOGGING & MONITORING
# =============================================================================

# Log Levels
LOG_LEVEL_DEBUG: Final[str] = "DEBUG"
LOG_LEVEL_INFO: Final[str] = "INFO"
LOG_LEVEL_WARNING: Final[str] = "WARNING"
LOG_LEVEL_ERROR: Final[str] = "ERROR"
LOG_LEVEL_CRITICAL: Final[str] = "CRITICAL"

# Log Formats
LOG_FORMAT_JSON: Final[str] = "json"
"""Structured JSON logging."""

LOG_FORMAT_TEXT: Final[str] = "text"
"""Human-readable text logging."""

# Metrics
METRICS_NAMESPACE: Final[str] = "turkish_legal_ai"
"""Prometheus metrics namespace."""

# =============================================================================
# TIME & SCHEDULING
# =============================================================================

# Timezone
DEFAULT_TIMEZONE: Final[str] = "Europe/Istanbul"
"""Default timezone (Turkey)."""

# Date Formats
DATE_FORMAT: Final[str] = "%Y-%m-%d"
"""ISO date format: 2025-10-30"""

DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Datetime format: 2025-10-30 12:00:00"""

ISO_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S%z"
"""ISO 8601 datetime format."""

# Cron Schedules
DAILY_BACKUP_SCHEDULE: Final[str] = "0 2 * * *"
"""Daily backup at 2 AM."""

HOURLY_CLEANUP_SCHEDULE: Final[str] = "0 * * * *"
"""Hourly cleanup at minute 0."""

WEEKLY_REPORT_SCHEDULE: Final[str] = "0 9 * * 1"
"""Weekly report Monday 9 AM."""

# =============================================================================
# KVKK / GDPR COMPLIANCE
# =============================================================================

# Data Retention
DEFAULT_DATA_RETENTION_DAYS: Final[int] = 365
"""Default data retention period (1 year)."""

AUDIT_LOG_RETENTION_DAYS: Final[int] = 2555
"""Audit log retention (7 years - KVKK requirement)."""

BACKUP_RETENTION_DAYS: Final[int] = 30
"""Backup retention period."""

# PII Types
PII_TYPES: Final[tuple[str, ...]] = (
    "TC_NO",
    "IBAN",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "PASSPORT",
    "DRIVING_LICENSE",
    "IP_ADDRESS",
)
"""Personally Identifiable Information types."""

# =============================================================================
# ERROR CODES
# =============================================================================

# Application Error Codes
ERR_INVALID_INPUT: Final[str] = "INVALID_INPUT"
ERR_UNAUTHORIZED: Final[str] = "UNAUTHORIZED"
ERR_FORBIDDEN: Final[str] = "FORBIDDEN"
ERR_NOT_FOUND: Final[str] = "NOT_FOUND"
ERR_CONFLICT: Final[str] = "CONFLICT"
ERR_RATE_LIMIT: Final[str] = "RATE_LIMIT_EXCEEDED"
ERR_INTERNAL: Final[str] = "INTERNAL_ERROR"
ERR_SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"

# Document Processing Errors
ERR_DOCUMENT_TOO_LARGE: Final[str] = "DOCUMENT_TOO_LARGE"
ERR_DOCUMENT_INVALID_FORMAT: Final[str] = "INVALID_FORMAT"
ERR_DOCUMENT_PARSE_FAILED: Final[str] = "PARSE_FAILED"
ERR_OCR_FAILED: Final[str] = "OCR_FAILED"

# LLM Errors
ERR_LLM_TIMEOUT: Final[str] = "LLM_TIMEOUT"
ERR_LLM_QUOTA_EXCEEDED: Final[str] = "LLM_QUOTA_EXCEEDED"
ERR_LLM_INVALID_RESPONSE: Final[str] = "LLM_INVALID_RESPONSE"

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Application
    "APP_NAME",
    "APP_SLUG",
    "APP_DESCRIPTION",
    "APP_VERSION",
    "APP_CODENAME",
    
    # API
    "API_V1_STR",
    "API_V2_STR",
    "API_BETA_STR",
    
    # HTTP Status
    "HTTP_OK",
    "HTTP_CREATED",
    "HTTP_BAD_REQUEST",
    "HTTP_UNAUTHORIZED",
    "HTTP_FORBIDDEN",
    "HTTP_NOT_FOUND",
    "HTTP_UNPROCESSABLE_ENTITY",
    "HTTP_TOO_MANY_REQUESTS",
    "HTTP_INTERNAL_SERVER_ERROR",
    
    # Limits
    "MAX_REQUEST_SIZE_MB",
    "MAX_DOCUMENT_SIZE_MB",
    "MAX_PAGES_PER_DOCUMENT",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    
    # Security
    "PASSWORD_MIN_LENGTH",
    "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    "API_KEY_PREFIX",
    
    # Rate Limiting
    "RATE_LIMIT_PER_MINUTE",
    "RATE_LIMIT_PER_HOUR",
    "LLM_RATE_LIMIT_PER_MINUTE",
    
    # RAG
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    
    # Turkish Legal
    "TC_NO_LENGTH",
    "RESMI_GAZETE_BASE_URL",
    "TURKISH_LEGAL_DOCUMENT_TYPES",
    
    # Cache
    "CACHE_SHORT_TTL",
    "CACHE_MEDIUM_TTL",
    "CACHE_LONG_TTL",
    
    # Time
    "DEFAULT_TIMEZONE",
    "DATE_FORMAT",
    "DATETIME_FORMAT",
    
    # KVKK
    "DEFAULT_DATA_RETENTION_DAYS",
    "AUDIT_LOG_RETENTION_DAYS",
    "PII_TYPES",
    
    # Errors
    "ERR_INVALID_INPUT",
    "ERR_UNAUTHORIZED",
    "ERR_NOT_FOUND",
    "ERR_RATE_LIMIT",
]