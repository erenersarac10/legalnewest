"""
Main application settings for Turkish Legal AI.
...
"""
import secrets
from functools import lru_cache
from typing import Any, Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.constants import (
    APP_DESCRIPTION,
    APP_NAME,
    API_V1_STR,
    DEFAULT_DB_MAX_OVERFLOW,
    DEFAULT_DB_POOL_SIZE,
    DEFAULT_DB_POOL_TIMEOUT,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM_DEFAULT,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
    PASSWORD_MIN_LENGTH,
)
from backend.core.version import __version__


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # =========================================================================
    # ENVIRONMENT
    # =========================================================================
    
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode (never use in production)",
    )
    
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    
    # =========================================================================
    # APPLICATION
    # =========================================================================
    
    APP_NAME: str = Field(
        default=APP_NAME,
        description="Application name",
    )
    
    APP_VERSION: str = Field(
        default=__version__,
        description="Application version",
    )
    
    API_V1_PREFIX: str = Field(
        default=API_V1_STR,
        description="API v1 prefix",
    )
    
    ALLOWED_HOSTS: list[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts for CORS",
    )
    
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )
    
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS",
    )
    
    # =========================================================================
    # DATABASE - PostgreSQL (URLs as strings, validated manually)
    # =========================================================================
    
    DATABASE_URL: str = Field(
        description="PostgreSQL database URL",
    )
    
    DATABASE_READ_URL: str | None = Field(
        default=None,
        description="Read replica database URL (optional)",
    )
    
    DATABASE_POOL_SIZE: int = Field(
        default=DEFAULT_DB_POOL_SIZE,
        description="Database connection pool size",
        ge=1,
        le=100,
    )
    
    DATABASE_MAX_OVERFLOW: int = Field(
        default=DEFAULT_DB_MAX_OVERFLOW,
        description="Maximum overflow connections",
        ge=0,
        le=50,
    )
    
    DATABASE_POOL_TIMEOUT: int = Field(
        default=DEFAULT_DB_POOL_TIMEOUT,
        description="Pool timeout in seconds",
        ge=1,
        le=300,
    )
    
    DATABASE_ECHO: bool = Field(
        default=False,
        description="Echo SQL queries (debug only)",
    )
    
    # =========================================================================
    # REDIS - Cache & Queue (URLs as strings)
    # =========================================================================
    
    REDIS_URL: str = Field(
        description="Redis connection URL",
    )
    
    REDIS_CACHE_DB: int = Field(
        default=0,
        description="Redis database for caching",
        ge=0,
        le=15,
    )
    
    REDIS_QUEUE_DB: int = Field(
        default=1,
        description="Redis database for queue",
        ge=0,
        le=15,
    )
    
    REDIS_SESSION_DB: int = Field(
        default=2,
        description="Redis database for sessions",
        ge=0,
        le=15,
    )
    
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        description="Maximum Redis connections",
        ge=1,
        le=1000,
    )
    
    # =========================================================================
    # CELERY - Task Queue
    # =========================================================================
    
    CELERY_BROKER_URL: str = Field(
        description="Celery broker URL",
    )
    
    CELERY_RESULT_BACKEND: str = Field(
        description="Celery result backend URL",
    )
    
    CELERY_TASK_TRACK_STARTED: bool = Field(
        default=True,
        description="Track task start time",
    )
    
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=3600,
        description="Task hard time limit in seconds",
        ge=60,
        le=86400,
    )
    
    # =========================================================================
    # STORAGE - S3 / MinIO (URL as string)
    # =========================================================================
    
    S3_ENDPOINT_URL: str = Field(
        description="S3 endpoint URL",
    )
    
    S3_ACCESS_KEY_ID: str = Field(
        description="S3 access key ID",
        min_length=1,
    )
    
    S3_SECRET_ACCESS_KEY: SecretStr = Field(
        description="S3 secret access key",
    )
    
    S3_BUCKET_NAME: str = Field(
        default="legal-documents",
        description="S3 bucket name",
        min_length=3,
        max_length=63,
    )
    
    S3_REGION: str = Field(
        default="us-east-1",
        description="S3 region",
    )
    
    S3_USE_SSL: bool = Field(
        default=False,
        description="Use SSL for S3 connections",
    )
    
    # =========================================================================
    # LLM PROVIDERS
    # =========================================================================
    
    # OpenAI
    OPENAI_API_KEY: SecretStr = Field(
        description="OpenAI API key",
    )
    
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="Default OpenAI model",
    )
    
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model",
    )
    
    OPENAI_MAX_TOKENS: int = Field(
        default=4096,
        description="Maximum completion tokens",
        ge=1,
        le=128000,
    )
    
    OPENAI_TEMPERATURE: float = Field(
        default=0.7,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )
    
    # Anthropic (Claude)
    ANTHROPIC_API_KEY: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (optional)",
    )
    
    ANTHROPIC_MODEL: str = Field(
        default="claude-3-opus-20240229",
        description="Default Anthropic model",
    )
    
    # Google (Gemini)
    GOOGLE_API_KEY: SecretStr | None = Field(
        default=None,
        description="Google API key (optional)",
    )
    
    GOOGLE_MODEL: str = Field(
        default="gemini-pro",
        description="Default Google model",
    )
    
    # Cohere
    COHERE_API_KEY: SecretStr | None = Field(
        default=None,
        description="Cohere API key (optional)",
    )
    
    COHERE_MODEL: str = Field(
        default="command-r-plus",
        description="Default Cohere model",
    )
    
    # LLM Configuration
    LLM_PROVIDER: Literal["openai", "anthropic", "google", "cohere"] = Field(
        default="openai",
        description="Primary LLM provider",
    )
    
    LLM_FALLBACK_ENABLED: bool = Field(
        default=True,
        description="Enable LLM fallback on failure",
    )
    
    LLM_RETRY_ATTEMPTS: int = Field(
        default=3,
        description="Number of retry attempts for LLM calls",
        ge=1,
        le=10,
    )
    
    LLM_TIMEOUT: int = Field(
        default=60,
        description="LLM request timeout in seconds",
        ge=10,
        le=300,
    )
    
    LLM_CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable LLM response caching",
    )
    
    LLM_CACHE_TTL: int = Field(
        default=3600,
        description="LLM cache TTL in seconds",
        ge=60,
        le=86400,
    )
    
    # =========================================================================
    # RAG CONFIGURATION
    # =========================================================================
    
    RAG_CHUNK_SIZE: int = Field(
        default=1000,
        description="Text chunk size for RAG",
        ge=100,
        le=4000,
    )
    
    RAG_CHUNK_OVERLAP: int = Field(
        default=200,
        description="Chunk overlap size",
        ge=0,
        le=1000,
    )
    
    RAG_TOP_K: int = Field(
        default=5,
        description="Number of chunks to retrieve",
        ge=1,
        le=50,
    )
    
    RAG_SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum similarity score",
        ge=0.0,
        le=1.0,
    )
    
    RAG_RERANK_ENABLED: bool = Field(
        default=True,
        description="Enable result reranking",
    )
    
    RAG_HYBRID_SEARCH: bool = Field(
        default=True,
        description="Enable hybrid (semantic + keyword) search",
    )
    
    # =========================================================================
    # SECURITY
    # =========================================================================
    
    # JWT
    JWT_SECRET_KEY: SecretStr = Field(
        description="JWT signing key (min 32 characters)",
    )
    
    JWT_ALGORITHM: str = Field(
        default=JWT_ALGORITHM_DEFAULT,
        description="JWT signing algorithm",
    )
    
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        description="Access token expiration in minutes",
        ge=5,
        le=1440,
    )
    
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=JWT_REFRESH_TOKEN_EXPIRE_DAYS,
        description="Refresh token expiration in days",
        ge=1,
        le=90,
    )
    
    # Encryption
    ENCRYPTION_KEY: SecretStr = Field(
        description="Data encryption key (32 bytes base64)",
    )
    
    MASTER_KEY: SecretStr = Field(
        description="Master encryption key",
    )
    
    # Password Policy
    PASSWORD_MIN_LENGTH: int = Field(
        default=PASSWORD_MIN_LENGTH,
        description="Minimum password length",
        ge=8,
        le=128,
    )
    
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(
        default=True,
        description="Require uppercase characters",
    )
    
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(
        default=True,
        description="Require lowercase characters",
    )
    
    PASSWORD_REQUIRE_DIGIT: bool = Field(
        default=True,
        description="Require digit characters",
    )
    
    PASSWORD_REQUIRE_SPECIAL: bool = Field(
        default=True,
        description="Require special characters",
    )
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="Requests per minute",
        ge=1,
        le=10000,
    )
    
    RATE_LIMIT_PER_HOUR: int = Field(
        default=1000,
        description="Requests per hour",
        ge=1,
        le=1000000,
    )
    
    RATE_LIMIT_PER_DAY: int = Field(
        default=10000,
        description="Requests per day",
        ge=1,
        le=10000000,
    )
    
    # =========================================================================
    # OBSERVABILITY
    # =========================================================================
    
    # Sentry
    SENTRY_DSN: str | None = Field(
        default=None,
        description="Sentry DSN for error tracking",
    )
    
    SENTRY_ENVIRONMENT: str = Field(
        default="development",
        description="Sentry environment tag",
    )
    
    SENTRY_TRACES_SAMPLE_RATE: float = Field(
        default=1.0,
        description="Sentry traces sample rate",
        ge=0.0,
        le=1.0,
    )
    
    # Prometheus
    PROMETHEUS_ENABLED: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    
    PROMETHEUS_PORT: int = Field(
        default=9090,
        description="Prometheus metrics port",
        ge=1024,
        le=65535,
    )
    
    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        default=None,
        description="OpenTelemetry OTLP endpoint",
    )
    
    OTEL_SERVICE_NAME: str = Field(
        default="turkish-legal-ai",
        description="OpenTelemetry service name",
    )
    
    OTEL_TRACES_ENABLED: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing",
    )
    
    OTEL_METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable OpenTelemetry metrics",
    )
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    LOG_FORMAT: Literal["json", "text"] = Field(
        default="json",
        description="Log format",
    )
    
    LOG_FILE_ENABLED: bool = Field(
        default=True,
        description="Enable file logging",
    )
    
    LOG_FILE_PATH: str = Field(
        default="/var/log/legal-ai/app.log",
        description="Log file path",
    )
    
    LOG_FILE_MAX_BYTES: int = Field(
        default=10485760,  # 10 MB
        description="Maximum log file size in bytes",
        ge=1048576,  # 1 MB
        le=104857600,  # 100 MB
    )
    
    LOG_FILE_BACKUP_COUNT: int = Field(
        default=10,
        description="Number of log file backups",
        ge=1,
        le=100,
    )
    
    # =========================================================================
    # FEATURE FLAGS
    # =========================================================================
    
    ENABLE_RAG: bool = Field(
        default=True,
        description="Enable RAG functionality",
    )
    
    ENABLE_CHAT: bool = Field(
        default=True,
        description="Enable chat functionality",
    )
    
    ENABLE_DOCUMENT_ANALYSIS: bool = Field(
        default=True,
        description="Enable document analysis",
    )
    
    ENABLE_CONTRACT_GENERATION: bool = Field(
        default=True,
        description="Enable contract generation",
    )
    
    ENABLE_LEGAL_QA: bool = Field(
        default=True,
        description="Enable legal Q&A",
    )
    
    ENABLE_TURKISH_PARSER: bool = Field(
        default=True,
        description="Enable Turkish legal parser",
    )
    
    # =========================================================================
    # PERFORMANCE
    # =========================================================================
    
    WORKERS: int = Field(
        default=4,
        description="Number of worker processes",
        ge=1,
        le=32,
    )
    
    WORKER_CONNECTIONS: int = Field(
        default=1000,
        description="Worker connections",
        ge=100,
        le=10000,
    )
    
    KEEPALIVE: int = Field(
        default=5,
        description="Keepalive timeout in seconds",
        ge=1,
        le=300,
    )
    
    TIMEOUT: int = Field(
        default=30,
        description="Request timeout in seconds",
        ge=10,
        le=300,
    )
    
    # =========================================================================
    # PYDANTIC SETTINGS CONFIGURATION
    # =========================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )
    
    # =========================================================================
    # VALIDATORS
    # =========================================================================
    
    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v: SecretStr) -> SecretStr:
        """Validate JWT secret key length."""
        secret = v.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        return v
    
    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with redis://")
        return v
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: Any) -> list[str]:
        """Parse allowed hosts from comma-separated string."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @model_validator(mode="after")
    def validate_environment_config(self) -> "Settings":
        """Validate environment-specific configuration."""
        # Production checks
        if self.ENVIRONMENT == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")
            if self.DATABASE_ECHO:
                raise ValueError("DATABASE_ECHO must be False in production")
            if self.LOG_LEVEL == "DEBUG":
                raise ValueError("LOG_LEVEL should not be DEBUG in production")
        
        # RAG chunk validation
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be less than RAG_CHUNK_SIZE"
            )
        
        return self
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.ENVIRONMENT == "staging"
    
    @property
    def redis_cache_url(self) -> str:
        """Get Redis URL for cache."""
        url = str(self.REDIS_URL)
        if "/" in url:
            base = url.rsplit("/", 1)[0]
            return f"{base}/{self.REDIS_CACHE_DB}"
        return f"{url}/{self.REDIS_CACHE_DB}"
    
    @property
    def redis_queue_url(self) -> str:
        """Get Redis URL for queue."""
        url = str(self.REDIS_URL)
        if "/" in url:
            base = url.rsplit("/", 1)[0]
            return f"{base}/{self.REDIS_QUEUE_DB}"
        return f"{url}/{self.REDIS_QUEUE_DB}"


# =============================================================================
# SETTINGS INSTANCE (Singleton)
# =============================================================================

@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Validated settings instance
    """
    return Settings()


# Create global settings instance
settings = get_settings()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Settings",
    "settings",
    "get_settings",
]