"""
Logging configuration for Turkish Legal AI.

This module provides comprehensive logging setup:
- Structured JSON logging (production)
- Human-readable text logging (development)
- Log rotation and retention
- Context injection (request_id, user_id, tenant_id)
- Multiple handlers (file, console, syslog)
- Integration with OpenTelemetry
- Sensitive data masking (KVKK/GDPR)

Logging Levels:
    DEBUG: Detailed diagnostic information
    INFO: General informational messages
    WARNING: Warning messages
    ERROR: Error messages
    CRITICAL: Critical failures

Log Structure:
    {
        "timestamp": "2025-10-30T12:00:00Z",
        "level": "INFO",
        "logger": "backend.api.routes",
        "message": "User login successful",
        "request_id": "abc-123",
        "user_id": "user_456",
        "tenant_id": "tenant_789",
        "extra": {...}
    }
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger

from backend.core.config.settings import settings
from backend.core.constants import (
    LOG_LEVEL_CRITICAL,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
)
from backend.core.version import __version__


class LoggingConfig:
    """
    Logging configuration and setup.
    
    Provides centralized logging configuration with:
    - Structured logging (JSON in production)
    - Context injection
    - Log masking for sensitive data
    - Multiple output handlers
    - Integration with monitoring systems
    """
    
    def __init__(self) -> None:
        """Initialize logging configuration."""
        self._configured = False
        self._sensitive_fields = {
            "password",
            "token",
            "api_key",
            "secret",
            "authorization",
            "jwt",
            "access_token",
            "refresh_token",
            "credit_card",
            "ssn",
            "tc_no",  # Turkish ID number
            "iban",
        }
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    def configure(self) -> None:
        """
        Configure application logging.
        
        Sets up:
        - Root logger
        - Structlog processors
        - Log handlers (console, file, etc.)
        - Log formatting
        
        Example:
            >>> from backend.core.config.logging import logging_config
            >>> logging_config.configure()
            >>> logger = logging.getLogger(__name__)
            >>> logger.info("Application started")
        """
        if self._configured:
            return
        
        # Configure standard library logging
        self._configure_standard_logging()
        
        # Configure structlog
        self._configure_structlog()
        
        self._configured = True
    
    def _configure_standard_logging(self) -> None:
        """Configure Python's standard logging module."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self._get_log_level())
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)
        
        # Add file handler (if enabled)
        if settings.LOG_FILE_ENABLED:
            file_handler = self._create_file_handler()
            if file_handler:
                root_logger.addHandler(file_handler)
        
        # Set levels for noisy libraries
        self._configure_third_party_loggers()
    
    def _configure_structlog(self) -> None:
        """Configure structlog for structured logging."""
        processors = [
            # Add log level
            structlog.stdlib.add_log_level,
            
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            
            # Add caller information (file, line, function)
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            
            # Mask sensitive data
            self._mask_sensitive_processor,
            
            # Add context (request_id, user_id, etc.)
            structlog.contextvars.merge_contextvars,
            
            # Stack trace formatting
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            
            # Format for output
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # =========================================================================
    # HANDLERS
    # =========================================================================
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """
        Create console (stdout) handler.
        
        Returns:
            logging.StreamHandler: Configured console handler
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._get_log_level())
        
        # Set formatter based on environment
        if settings.LOG_FORMAT == "json":
            formatter = self._create_json_formatter()
        else:
            formatter = self._create_text_formatter()
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self) -> logging.Handler | None:
        """
        Create rotating file handler.
        
        Returns:
            logging.Handler | None: Configured file handler or None if failed
        """
        try:
            # Ensure log directory exists
            log_path = Path(settings.LOG_FILE_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=settings.LOG_FILE_MAX_BYTES,
                backupCount=settings.LOG_FILE_BACKUP_COUNT,
                encoding='utf-8',
            )
            
            handler.setLevel(self._get_log_level())
            
            # Always use JSON for file logs
            formatter = self._create_json_formatter()
            handler.setFormatter(formatter)
            
            return handler
        except (OSError, PermissionError) as e:
            print(f"Failed to create log file handler: {e}", file=sys.stderr)
            return None
    
    # =========================================================================
    # FORMATTERS
    # =========================================================================
    
    def _create_json_formatter(self) -> jsonlogger.JsonFormatter:
        """
        Create JSON log formatter.
        
        Returns:
            jsonlogger.JsonFormatter: JSON formatter
        """
        return jsonlogger.JsonFormatter(
            fmt=(
                "%(timestamp)s %(level)s %(name)s %(message)s "
                "%(filename)s %(funcName)s %(lineno)d"
            ),
            rename_fields={
                "levelname": "level",
                "name": "logger",
                "asctime": "timestamp",
            },
        )
    
    def _create_text_formatter(self) -> logging.Formatter:
        """
        Create human-readable text formatter.
        
        Returns:
            logging.Formatter: Text formatter
        """
        return logging.Formatter(
            fmt=(
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    # =========================================================================
    # PROCESSORS
    # =========================================================================
    
    def _mask_sensitive_processor(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Structlog processor to mask sensitive data.
        
        Args:
            logger: Logger instance
            method_name: Log method name
            event_dict: Event dictionary
            
        Returns:
            dict: Event dict with masked sensitive fields
        """
        # Mask sensitive fields in event dict
        for key in list(event_dict.keys()):
            if any(sensitive in key.lower() for sensitive in self._sensitive_fields):
                event_dict[key] = "***MASKED***"
        
        # Mask sensitive data in nested dicts
        for key, value in event_dict.items():
            if isinstance(value, dict):
                event_dict[key] = self._mask_dict(value)
        
        return event_dict
    
    def _mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively mask sensitive fields in dictionary.
        
        Args:
            data: Dictionary to mask
            
        Returns:
            dict: Masked dictionary
        """
        masked = {}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self._sensitive_fields):
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [
                    self._mask_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked[key] = value
        
        return masked
    
    # =========================================================================
    # THIRD-PARTY LOGGERS
    # =========================================================================
    
    def _configure_third_party_loggers(self) -> None:
        """Configure log levels for third-party libraries."""
        # Reduce noise from verbose libraries
        noisy_loggers = {
            "urllib3": logging.WARNING,
            "boto3": logging.WARNING,
            "botocore": logging.WARNING,
            "s3transfer": logging.WARNING,
            "asyncio": logging.WARNING,
            "multipart": logging.WARNING,
            "sqlalchemy.engine": logging.WARNING,
            "sqlalchemy.pool": logging.WARNING,
            "redis": logging.WARNING,
            "celery": logging.INFO,
        }
        
        for logger_name, level in noisy_loggers.items():
            logging.getLogger(logger_name).setLevel(level)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _get_log_level(self) -> int:
        """
        Get numeric log level from settings.
        
        Returns:
            int: Logging level constant
        """
        level_map = {
            LOG_LEVEL_DEBUG: logging.DEBUG,
            LOG_LEVEL_INFO: logging.INFO,
            LOG_LEVEL_WARNING: logging.WARNING,
            LOG_LEVEL_ERROR: logging.ERROR,
            LOG_LEVEL_CRITICAL: logging.CRITICAL,
        }
        
        return level_map.get(settings.LOG_LEVEL, logging.INFO)
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """
        Get a structured logger.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            structlog.BoundLogger: Configured logger
            
        Example:
            >>> logger = logging_config.get_logger(__name__)
            >>> logger.info("message", user_id="123", action="login")
        """
        if not self._configured:
            self.configure()
        
        return structlog.get_logger(name)
    
    def set_context(self, **kwargs: Any) -> None:
        """
        Set context variables for all subsequent logs.
        
        Context is preserved across async boundaries.
        
        Args:
            **kwargs: Context key-value pairs
            
        Example:
            >>> logging_config.set_context(request_id="abc-123")
            >>> logger.info("Processing request")
            # Log will include request_id="abc-123"
        """
        structlog.contextvars.bind_contextvars(**kwargs)
    
    def clear_context(self) -> None:
        """Clear all context variables."""
        structlog.contextvars.clear_contextvars()
    
    def add_sensitive_field(self, field: str) -> None:
        """
        Add a field name to the sensitive fields list.
        
        Args:
            field: Field name to mask in logs
        """
        self._sensitive_fields.add(field.lower())


# =============================================================================
# GLOBAL LOGGING CONFIG INSTANCE
# =============================================================================

logging_config = LoggingConfig()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def configure_logging() -> None:
    """
    Configure application logging.
    
    Should be called once at application startup.
    
    Example:
        >>> from backend.core.config.logging import configure_logging
        >>> configure_logging()
    """
    logging_config.configure()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name (defaults to caller's module)
        
    Returns:
        structlog.BoundLogger: Configured logger
        
    Example:
        >>> from backend.core.config.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("User logged in", user_id="123")
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
    
    return logging_config.get_logger(name)


def set_log_context(**kwargs: Any) -> None:
    """
    Set logging context variables.
    
    Args:
        **kwargs: Context key-value pairs
        
    Example:
        >>> from backend.core.config.logging import set_log_context
        >>> set_log_context(request_id="abc", user_id="123")
    """
    logging_config.set_context(**kwargs)


def clear_log_context() -> None:
    """
    Clear all logging context variables.
    
    Example:
        >>> from backend.core.config.logging import clear_log_context
        >>> clear_log_context()
    """
    logging_config.clear_context()


# =============================================================================
# REQUEST LOGGING MIDDLEWARE HELPER
# =============================================================================

class RequestLogger:
    """
    Helper class for request logging in middleware.
    
    Automatically injects request context into logs.
    
    Example:
        >>> request_logger = RequestLogger(request)
        >>> request_logger.log_start()
        >>> # ... process request ...
        >>> request_logger.log_end(status_code=200)
    """
    
    def __init__(self, request_id: str, path: str, method: str) -> None:
        """
        Initialize request logger.
        
        Args:
            request_id: Unique request identifier
            path: Request path
            method: HTTP method
        """
        self.request_id = request_id
        self.path = path
        self.method = method
        self.logger = get_logger("backend.api.request")
        
        # Set context
        set_log_context(
            request_id=request_id,
            path=path,
            method=method,
        )
    
    def log_start(self, **extra: Any) -> None:
        """
        Log request start.
        
        Args:
            **extra: Additional context to log
        """
        self.logger.info(
            "Request started",
            **extra,
        )
    
    def log_end(self, status_code: int, duration_ms: float, **extra: Any) -> None:
        """
        Log request completion.
        
        Args:
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            **extra: Additional context to log
        """
        level = "info"
        if status_code >= 500:
            level = "error"
        elif status_code >= 400:
            level = "warning"
        
        log_method = getattr(self.logger, level)
        log_method(
            "Request completed",
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            **extra,
        )
    
    def log_error(self, error: Exception, **extra: Any) -> None:
        """
        Log request error.
        
        Args:
            error: Exception that occurred
            **extra: Additional context to log
        """
        self.logger.error(
            "Request failed",
            error=str(error),
            error_type=type(error).__name__,
            **extra,
            exc_info=True,
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "LoggingConfig",
    "logging_config",
    "configure_logging",
    "get_logger",
    "set_log_context",
    "clear_log_context",
    "RequestLogger",
]