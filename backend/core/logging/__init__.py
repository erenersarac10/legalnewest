"""
Logging utilities for Turkish Legal AI.

This module provides logging functionality beyond the base configuration:
- Structured logging helpers
- Context managers
- Log decorators
- Performance logging
- Audit logging
- Request/response logging

The base logging configuration is in backend.core.config.logging.
This module provides higher-level utilities built on top of that.

Usage:
    >>> from backend.core.logging import get_logger, log_execution_time
    >>> 
    >>> logger = get_logger(__name__)
    >>> 
    >>> @log_execution_time
    >>> async def slow_function():
    ...     await asyncio.sleep(1)
"""

# Re-export from config.logging for convenience
from backend.core.config.logging import (
    RequestLogger,
    clear_log_context,
    configure_logging,
    get_logger,
    logging_config,
    set_log_context,
)

# Additional exports will be added as we create more logging utilities

__all__ = [
    # Re-exported from config
    "configure_logging",
    "get_logger",
    "set_log_context",
    "clear_log_context",
    "logging_config",
    "RequestLogger",
]
