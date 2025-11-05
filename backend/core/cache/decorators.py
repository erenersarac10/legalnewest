"""
Cache decorators for Turkish Legal AI.

This module provides decorators for automatic caching:
- Function result caching
- Method caching with instance binding
- TTL management
- Cache key generation
- Cache invalidation
- Conditional caching

Features:
- Automatic key generation from function arguments
- Support for async functions
- Configurable TTL
- Cache warming
- Bypass for specific conditions

Usage:
    >>> from backend.core.cache.decorators import cached
    >>> 
    >>> @cached(ttl=3600, key_prefix="user")
    >>> async def get_user(user_id: str):
    ...     return await db.get(User, user_id)
"""
import functools
import hashlib
import inspect
import json
from typing import Any, Callable, ParamSpec, TypeVar

from backend.core.cache.redis import cache
from backend.core.constants import CACHE_MEDIUM_TTL

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# CACHE DECORATOR
# =============================================================================

def cached(
    ttl: int = CACHE_MEDIUM_TTL,
    key_prefix: str | None = None,
    key_builder: Callable[..., str] | None = None,
    condition: Callable[..., bool] | None = None,
    unless: Callable[[Any], bool] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for caching function results.
    
    Automatically caches function return values in Redis.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key (uses function name if None)
        key_builder: Custom function to build cache key
        condition: Function that returns True if caching should occur
        unless: Function that receives result and returns True to skip caching
        
    Returns:
        Decorated function
        
    Example:
        >>> @cached(ttl=3600, key_prefix="user")
        >>> async def get_user(user_id: str):
        ...     return await db.get(User, user_id)
        >>> 
        >>> # First call - hits database
        >>> user = await get_user("123")
        >>> 
        >>> # Second call - from cache
        >>> user = await get_user("123")
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Determine if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        # Get function name for default key prefix
        func_name = func.__name__
        prefix = key_prefix or func_name
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Check condition
                if condition and not condition(*args, **kwargs):
                    return await func(*args, **kwargs)
                
                # Build cache key
                cache_key = (
                    key_builder(*args, **kwargs)
                    if key_builder
                    else _build_cache_key(prefix, func, args, kwargs)
                )
                
                # Try to get from cache
                cached_result = await cache.get(cache_key)
                
                if cached_result is not None:
                    return cached_result
                
                # Call function
                result = await func(*args, **kwargs)
                
                # Check unless condition
                if unless and unless(result):
                    return result
                
                # Cache result
                await cache.set(cache_key, result, ttl=ttl)
                
                return result
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Check condition
                if condition and not condition(*args, **kwargs):
                    return func(*args, **kwargs)
                
                # Build cache key
                cache_key = (
                    key_builder(*args, **kwargs)
                    if key_builder
                    else _build_cache_key(prefix, func, args, kwargs)
                )
                
                # Try to get from cache (sync)
                # Note: This uses sync Redis client
                import asyncio
                cached_result = asyncio.run(cache.get(cache_key))
                
                if cached_result is not None:
                    return cached_result
                
                # Call function
                result = func(*args, **kwargs)
                
                # Check unless condition
                if unless and unless(result):
                    return result
                
                # Cache result
                asyncio.run(cache.set(cache_key, result, ttl=ttl))
                
                return result
            
            return sync_wrapper
    
    return decorator


# =============================================================================
# CACHE INVALIDATION DECORATOR
# =============================================================================

def cache_invalidate(
    key_prefix: str | None = None,
    key_builder: Callable[..., str] | None = None,
    pattern: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for cache invalidation.
    
    Automatically invalidates cache keys after function execution.
    
    Args:
        key_prefix: Prefix for cache key
        key_builder: Custom function to build cache key
        pattern: Pattern for bulk invalidation (e.g., "user:*")
        
    Returns:
        Decorated function
        
    Example:
        >>> @cache_invalidate(pattern="user:*")
        >>> async def update_user(user_id: str, data: dict):
        ...     await db.update(User, user_id, data)
        >>> 
        >>> # This will invalidate all "user:*" cache keys
        >>> await update_user("123", {"name": "John"})
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = inspect.iscoroutinefunction(func)
        func_name = func.__name__
        prefix = key_prefix or func_name
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Call function first
                result = await func(*args, **kwargs)
                
                # Invalidate cache
                if pattern:
                    await cache.delete_pattern(pattern)
                elif key_builder:
                    cache_key = key_builder(*args, **kwargs)
                    await cache.delete(cache_key)
                else:
                    cache_key = _build_cache_key(prefix, func, args, kwargs)
                    await cache.delete(cache_key)
                
                return result
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                result = func(*args, **kwargs)
                
                # Invalidate cache
                import asyncio
                if pattern:
                    asyncio.run(cache.delete_pattern(pattern))
                elif key_builder:
                    cache_key = key_builder(*args, **kwargs)
                    asyncio.run(cache.delete(cache_key))
                else:
                    cache_key = _build_cache_key(prefix, func, args, kwargs)
                    asyncio.run(cache.delete(cache_key))
                
                return result
            
            return sync_wrapper
    
    return decorator


# =============================================================================
# METHOD CACHE DECORATOR
# =============================================================================

def cached_method(
    ttl: int = CACHE_MEDIUM_TTL,
    key_prefix: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for caching method results (instance-aware).
    
    Similar to @cached but works with class methods and includes
    instance identity in cache key.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        
    Returns:
        Decorated method
        
    Example:
        >>> class UserService:
        ...     @cached_method(ttl=3600)
        ...     async def get_user(self, user_id: str):
        ...         return await self.db.get(User, user_id)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = inspect.iscoroutinefunction(func)
        func_name = func.__name__
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(
                self: Any,
                *args: P.args,
                **kwargs: P.kwargs
            ) -> T:
                # Build cache key with instance id
                instance_id = id(self)
                prefix = key_prefix or f"{self.__class__.__name__}:{func_name}"
                cache_key = _build_cache_key(
                    f"{prefix}:{instance_id}",
                    func,
                    args,
                    kwargs
                )
                
                # Try cache
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Call method
                result = await func(self, *args, **kwargs)
                
                # Cache result
                await cache.set(cache_key, result, ttl=ttl)
                
                return result
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
                instance_id = id(self)
                prefix = key_prefix or f"{self.__class__.__name__}:{func_name}"
                cache_key = _build_cache_key(
                    f"{prefix}:{instance_id}",
                    func,
                    args,
                    kwargs
                )
                
                import asyncio
                cached_result = asyncio.run(cache.get(cache_key))
                if cached_result is not None:
                    return cached_result
                
                result = func(self, *args, **kwargs)
                asyncio.run(cache.set(cache_key, result, ttl=ttl))
                
                return result
            
            return sync_wrapper
    
    return decorator


# =============================================================================
# RATE LIMITING DECORATOR
# =============================================================================

def rate_limit(
    max_calls: int,
    period: int,
    key_prefix: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for rate limiting function calls.
    
    Limits how many times a function can be called within a time period.
    
    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds
        key_prefix: Prefix for rate limit key
        
    Returns:
        Decorated function
        
    Raises:
        Exception: If rate limit exceeded
        
    Example:
        >>> @rate_limit(max_calls=10, period=60)
        >>> async def expensive_operation(user_id: str):
        ...     # Can only be called 10 times per minute per user
        ...     pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = inspect.iscoroutinefunction(func)
        func_name = func.__name__
        prefix = key_prefix or f"rate_limit:{func_name}"
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Build rate limit key
                cache_key = _build_cache_key(prefix, func, args, kwargs)
                
                # Get current count
                count = await cache.get(cache_key, default=0)
                
                if isinstance(count, int) and count >= max_calls:
                    raise Exception(
                        f"Çok fazla istek: Dakikada en fazla {max_calls} istek yapabilirsiniz"
                    )
                
                # Increment count
                new_count = await cache.incr(cache_key)
                
                # Set expiration on first call
                if new_count == 1:
                    await cache.expire(cache_key, period)
                
                # Call function
                return await func(*args, **kwargs)
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                import asyncio
                
                cache_key = _build_cache_key(prefix, func, args, kwargs)
                count = asyncio.run(cache.get(cache_key, default=0))
                
                if isinstance(count, int) and count >= max_calls:
                    raise Exception(
                        f"Çok fazla istek: Dakikada en fazla {max_calls} istek yapabilirsiniz"
                    )
                
                new_count = asyncio.run(cache.incr(cache_key))
                
                if new_count == 1:
                    asyncio.run(cache.expire(cache_key, period))
                
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_cache_key(
    prefix: str,
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> str:
    """
    Build cache key from function arguments using SHA256 hashing.
    
    Uses SHA256 instead of MD5 for better security practices, even though
    this is just for cache key generation (not cryptographic use).
    
    Args:
        prefix: Key prefix
        func: Function object
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        str: Cache key with format "prefix:hash"
        
    Example:
        >>> key = _build_cache_key("user", get_user, ("123",), {})
        >>> # Returns: "user:a1b2c3d4e5f6g7h8"
    """
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    
    # Create arguments dict
    args_dict = dict(bound_args.arguments)
    
    # Remove 'self' or 'cls' from class methods
    args_dict.pop('self', None)
    args_dict.pop('cls', None)
    
    # Create hash of arguments using SHA256 (best practice)
    args_str = json.dumps(args_dict, sort_keys=True, default=str)
    args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
    
    return cache.make_key(prefix, args_hash)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "cached",
    "cache_invalidate",
    "cached_method",
    "rate_limit",
]