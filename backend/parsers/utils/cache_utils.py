"""Cache Utilities - Harvey/Legora CTO-Level
TTL cache, LRU cache, persistent cache for parser performance.
"""
import time, hashlib, json, pickle
from typing import Any, Callable, Optional, Dict, Tuple
from functools import wraps, lru_cache
from pathlib import Path

class TTLCache:
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl, self._cache = default_ttl, {}
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry: return value
            del self._cache[key]
        return None
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._cache[key] = (value, time.time() + (ttl or self.default_ttl))
    def clear(self): self._cache.clear()

def cached(ttl: int = 3600):
    cache = TTLCache(ttl)
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hashlib.md5(repr((func.__name__, args, kwargs)).encode()).hexdigest()
            result = cache.get(key)
            if result is not None: return result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator

__all__ = ['TTLCache', 'cached', 'lru_cache']
