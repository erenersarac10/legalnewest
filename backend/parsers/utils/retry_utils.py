"""Retry Utilities - Harvey/Legora CTO-Level
Retry logic with exponential backoff for network operations.
"""
import time, logging
from typing import Callable, Optional, Tuple, Type
from functools import wraps

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, backoff_factor: float = 2.0, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1: raise
                    wait = backoff_factor ** attempt
                    logger.warning(f"Retry {attempt+1}/{max_attempts} after {wait}s: {e}")
                    time.sleep(wait)
        return wrapper
    return decorator

async def async_retry(max_attempts: int = 3, backoff_factor: float = 2.0):
    import asyncio
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1: raise
                    await asyncio.sleep(backoff_factor ** attempt)
        return wrapper
    return decorator

__all__ = ['retry', 'async_retry']
