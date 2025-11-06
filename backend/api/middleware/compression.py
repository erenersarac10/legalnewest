"""
Compression Middleware Configuration for Turkish Legal AI Platform.

Configures gzip compression for response payloads.

Features:
- Automatic gzip compression
- Configurable minimum size
- Content-type filtering
- Compression level tuning

Author: Turkish Legal AI Team
License: Proprietary
"""

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from backend.core import get_logger

logger = get_logger(__name__)

# Minimum response size for compression (bytes)
MIN_SIZE = 1024  # 1KB

# Compression level (1-9, higher = better compression but slower)
COMPRESSION_LEVEL = 5


def configure_compression(app: FastAPI) -> None:
    """
    Configure gzip compression middleware.

    Compresses responses larger than 1KB to reduce bandwidth.
    Uses medium compression level (5) for balance between speed and size.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        GZipMiddleware,
        minimum_size=MIN_SIZE,
        compresslevel=COMPRESSION_LEVEL,
    )

    logger.info(
        "Compression configured",
        min_size_bytes=MIN_SIZE,
        compression_level=COMPRESSION_LEVEL,
    )


__all__ = ["configure_compression"]
