"""
Cache module for Turkish Legal AI.

This module provides caching utilities:
- Redis cache wrapper
- Cache decorators
- Cache invalidation
- Cache warming
- TTL management
- Cache keys management

Cache Strategy:
- L1: In-memory cache (Python dict, per-process)
- L2: Redis cache (shared across processes)
- Write-through caching
- Cache-aside pattern support

Usage:
    >>> from backend.core.cache import cache, cached
    >>> 
    >>> # Direct cache usage
    >>> await cache.set("user:123", user_data, ttl=3600)
    >>> user = await cache.get("user:123")
    >>> 
    >>> # Decorator usage
    >>> @cached(ttl=3600)
    >>> async def get_user(user_id: str):
    ...     return await db.get(User, user_id)
"""

# =============================================================================
# REDIS CACHE
# =============================================================================

from backend.core.cache.redis import (
    RedisCache,
    cache,
)

# =============================================================================
# CACHE DECORATORS
# =============================================================================

from backend.core.cache.decorators import (
    cache_invalidate,
    cached,
    cached_method,
    rate_limit,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Redis Cache
    "RedisCache",
    "cache",
    # Decorators
    "cached",
    "cache_invalidate",
    "cached_method",
    "rate_limit",
]