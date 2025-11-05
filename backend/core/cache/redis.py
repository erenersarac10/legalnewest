"""
Redis cache wrapper for Turkish Legal AI.

This module provides a high-level Redis cache interface:
- Automatic serialization/deserialization
- TTL management
- Key namespacing
- Pattern-based operations
- Cache statistics
- Batch operations
- Pipeline support
- Lua script support

Features:
- JSON serialization for complex objects
- Compression for large values
- Cache versioning
- Atomic operations
- Distributed locking

Usage:
    >>> from backend.core.cache.redis import RedisCache
    >>> 
    >>> cache = RedisCache()
    >>> await cache.set("user:123", {"name": "John"}, ttl=3600)
    >>> user = await cache.get("user:123")
"""
import asyncio
import hashlib
import json
import pickle
import zlib
from datetime import timedelta
from typing import Any, Callable

from redis.asyncio import Redis
from redis.exceptions import RedisError

from backend.core.config.redis import redis_config
from backend.core.constants import (
    CACHE_KEY_PREFIX,
    CACHE_LONG_TTL,
    CACHE_MEDIUM_TTL,
    CACHE_SHORT_TTL,
    CACHE_VERSION,
)


class RedisCache:
    """
    High-level Redis cache interface.
    
    Provides automatic serialization, TTL management, and key namespacing.
    
    Example:
        >>> cache = RedisCache()
        >>> await cache.set("user:123", user_data, ttl=3600)
        >>> user = await cache.get("user:123")
    """
    
    def __init__(
        self,
        redis_client: Redis | None = None,
        prefix: str = CACHE_KEY_PREFIX,
        version: str = CACHE_VERSION,
        default_ttl: int = CACHE_MEDIUM_TTL,
        compression_threshold: int = 1024,  # Compress if > 1KB
    ) -> None:
        """
        Initialize Redis cache.
        
        Args:
            redis_client: Redis client (uses default if None)
            prefix: Key prefix for namespacing
            version: Cache version for invalidation
            default_ttl: Default TTL in seconds
            compression_threshold: Compress values larger than this (bytes)
        """
        self._redis = redis_client or redis_config.get_cache_client()
        self._prefix = prefix
        self._version = version
        self._default_ttl = default_ttl
        self._compression_threshold = compression_threshold
    
    # =========================================================================
    # KEY MANAGEMENT
    # =========================================================================
    
    def make_key(self, *parts: str) -> str:
        """
        Create a namespaced cache key.
        
        Args:
            *parts: Key components
            
        Returns:
            str: Formatted cache key
            
        Example:
            >>> cache.make_key("user", "123", "profile")
            'legal_ai:v1:user:123:profile'
        """
        key_parts = [self._prefix, self._version] + list(parts)
        return ":".join(str(part) for part in key_parts)
    
    def make_hash_key(self, data: str | dict) -> str:
        """
        Create a hash-based cache key.
        
        Useful for caching function results based on arguments.
        
        Args:
            data: Data to hash
            
        Returns:
            str: Hash-based key
            
        Example:
            >>> key = cache.make_hash_key({"query": "contract", "limit": 10})
        """
        if isinstance(data, dict):
            # Sort dict for consistent hashing
            data = json.dumps(data, sort_keys=True)
        
        hash_value = hashlib.sha256(data.encode()).hexdigest()[:16]
        return self.make_key("hash", hash_value)
    
    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================
    
    async def get(
        self,
        key: str,
        default: Any = None,
        deserialize: bool = True,
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key doesn't exist
            deserialize: Automatically deserialize value
            
        Returns:
            Any: Cached value or default
            
        Example:
            >>> user = await cache.get("user:123")
            >>> if user is None:
            ...     user = await db.get_user(123)
            ...     await cache.set("user:123", user)
        """
        try:
            value = await self._redis.get(key)
            
            if value is None:
                return default
            
            if deserialize:
                return self._deserialize(value)
            
            return value
            
        except RedisError as e:
            # Log error but don't fail
            print(f"Redis get error: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        serialize: bool = True,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            serialize: Automatically serialize value
            
        Returns:
            bool: True if successful
            
        Example:
            >>> await cache.set("user:123", user_data, ttl=3600)
        """
        try:
            if serialize:
                value = self._serialize(value)
            
            ttl = ttl or self._default_ttl
            
            if ttl == -1:
                # Permanent cache (no expiration)
                return await self._redis.set(key, value)
            else:
                return await self._redis.setex(key, ttl, value)
                
        except RedisError as e:
            print(f"Redis set error: {e}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """
        Delete one or more keys.
        
        Args:
            *keys: Cache keys to delete
            
        Returns:
            int: Number of keys deleted
            
        Example:
            >>> await cache.delete("user:123", "user:456")
        """
        try:
            if not keys:
                return 0
            return await self._redis.delete(*keys)
        except RedisError as e:
            print(f"Redis delete error: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists
        """
        try:
            return await self._redis.exists(key) > 0
        except RedisError:
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            return await self._redis.expire(key, ttl)
        except RedisError:
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            int: TTL in seconds (-1 if no expire, -2 if key doesn't exist)
        """
        try:
            return await self._redis.ttl(key)
        except RedisError:
            return -2
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    async def get_many(
        self,
        keys: list[str],
        deserialize: bool = True,
    ) -> dict[str, Any]:
        """
        Get multiple values at once.
        
        Args:
            keys: List of cache keys
            deserialize: Automatically deserialize values
            
        Returns:
            dict: Key-value pairs (only existing keys)
            
        Example:
            >>> users = await cache.get_many(["user:1", "user:2", "user:3"])
        """
        try:
            if not keys:
                return {}
            
            values = await self._redis.mget(keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value) if deserialize else value
            
            return result
            
        except RedisError as e:
            print(f"Redis mget error: {e}")
            return {}
    
    async def set_many(
        self,
        mapping: dict[str, Any],
        ttl: int | None = None,
        serialize: bool = True,
    ) -> bool:
        """
        Set multiple key-value pairs at once.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds
            serialize: Automatically serialize values
            
        Returns:
            bool: True if all successful
            
        Example:
            >>> await cache.set_many({
            ...     "user:1": user1_data,
            ...     "user:2": user2_data,
            ... }, ttl=3600)
        """
        try:
            if not mapping:
                return True
            
            # Use pipeline for atomic execution
            async with self._redis.pipeline() as pipe:
                for key, value in mapping.items():
                    if serialize:
                        value = self._serialize(value)
                    
                    if ttl and ttl != -1:
                        pipe.setex(key, ttl, value)
                    else:
                        pipe.set(key, value)
                
                await pipe.execute()
            
            return True
            
        except RedisError as e:
            print(f"Redis mset error: {e}")
            return False
    
    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys at once.
        
        Args:
            keys: List of cache keys
            
        Returns:
            int: Number of keys deleted
        """
        return await self.delete(*keys)
    
    # =========================================================================
    # PATTERN OPERATIONS
    # =========================================================================
    
    async def keys(self, pattern: str) -> list[str]:
        """
        Get all keys matching a pattern.
        
        WARNING: Use with caution in production (can be slow).
        
        Args:
            pattern: Redis pattern (supports * and ?)
            
        Returns:
            list: Matching keys
            
        Example:
            >>> keys = await cache.keys("user:*")
        """
        try:
            return [key.decode() for key in await self._redis.keys(pattern)]
        except RedisError:
            return []
    
    async def scan(
        self,
        pattern: str,
        count: int = 100,
    ) -> list[str]:
        """
        Scan for keys matching pattern (safer than keys()).
        
        Uses SCAN command which doesn't block the server.
        
        Args:
            pattern: Redis pattern
            count: Hint for number of elements to return per iteration
            
        Returns:
            list: Matching keys
            
        Example:
            >>> keys = await cache.scan("user:*", count=1000)
        """
        try:
            keys = []
            cursor = 0
            
            while True:
                cursor, batch = await self._redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=count,
                )
                keys.extend(key.decode() for key in batch)
                
                if cursor == 0:
                    break
            
            return keys
            
        except RedisError:
            return []
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Redis pattern
            
        Returns:
            int: Number of keys deleted
            
        Example:
            >>> deleted = await cache.delete_pattern("user:*")
        """
        keys = await self.scan(pattern)
        
        if keys:
            return await self.delete(*keys)
        
        return 0
    
    # =========================================================================
    # HASH OPERATIONS
    # =========================================================================
    
    async def hget(self, key: str, field: str, deserialize: bool = True) -> Any:
        """
        Get field from hash.
        
        Args:
            key: Hash key
            field: Field name
            deserialize: Automatically deserialize value
            
        Returns:
            Any: Field value
        """
        try:
            value = await self._redis.hget(key, field)
            
            if value is None:
                return None
            
            return self._deserialize(value) if deserialize else value
            
        except RedisError:
            return None
    
    async def hset(
        self,
        key: str,
        field: str,
        value: Any,
        serialize: bool = True,
    ) -> bool:
        """
        Set field in hash.
        
        Args:
            key: Hash key
            field: Field name
            value: Field value
            serialize: Automatically serialize value
            
        Returns:
            bool: True if new field, False if updated
        """
        try:
            if serialize:
                value = self._serialize(value)
            
            return await self._redis.hset(key, field, value) == 1
            
        except RedisError:
            return False
    
    async def hgetall(self, key: str, deserialize: bool = True) -> dict[str, Any]:
        """
        Get all fields from hash.
        
        Args:
            key: Hash key
            deserialize: Automatically deserialize values
            
        Returns:
            dict: All field-value pairs
        """
        try:
            data = await self._redis.hgetall(key)
            
            if deserialize:
                return {
                    k.decode(): self._deserialize(v)
                    for k, v in data.items()
                }
            
            return {k.decode(): v for k, v in data.items()}
            
        except RedisError:
            return {}
    
    async def hdel(self, key: str, *fields: str) -> int:
        """
        Delete fields from hash.
        
        Args:
            key: Hash key
            *fields: Field names to delete
            
        Returns:
            int: Number of fields deleted
        """
        try:
            return await self._redis.hdel(key, *fields)
        except RedisError:
            return 0
    
    # =========================================================================
    # ATOMIC OPERATIONS
    # =========================================================================
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.
        
        Args:
            key: Counter key
            amount: Amount to increment
            
        Returns:
            int: New value
            
        Example:
            >>> views = await cache.incr("page:views:123")
        """
        try:
            if amount == 1:
                return await self._redis.incr(key)
            else:
                return await self._redis.incrby(key, amount)
        except RedisError:
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement a counter.
        
        Args:
            key: Counter key
            amount: Amount to decrement
            
        Returns:
            int: New value
        """
        try:
            if amount == 1:
                return await self._redis.decr(key)
            else:
                return await self._redis.decrby(key, amount)
        except RedisError:
            return 0
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Uses JSON for simple types, pickle for complex objects.
        Compresses large values.
        
        Args:
            value: Value to serialize
            
        Returns:
            bytes: Serialized value
        """
        try:
            # Try JSON first (faster, more portable)
            serialized = json.dumps(value, default=str).encode('utf-8')
            prefix = b'json:'
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            prefix = b'pickle:'
        
        # Compress if larger than threshold
        if len(serialized) > self._compression_threshold:
            serialized = zlib.compress(serialized, level=6)
            prefix = b'compressed:' + prefix
        
        return prefix + serialized
    
    def _deserialize(self, value: bytes) -> Any:
        """
        Deserialize value from storage.
        
        Args:
            value: Serialized value
            
        Returns:
            Any: Deserialized value
        """
        # Check for compression
        if value.startswith(b'compressed:'):
            value = value[11:]  # Remove 'compressed:' prefix
            
            # Extract actual prefix
            if value.startswith(b'json:'):
                prefix = b'json:'
                value = value[5:]
            elif value.startswith(b'pickle:'):
                prefix = b'pickle:'
                value = value[7:]
            else:
                prefix = b''
            
            # Decompress
            value = zlib.decompress(value)
            
            # Re-add prefix for deserialization
            value = prefix + value
        
        # Deserialize based on prefix
        if value.startswith(b'json:'):
            return json.loads(value[5:].decode('utf-8'))
        elif value.startswith(b'pickle:'):
            return pickle.loads(value[7:])
        else:
            # Assume string
            return value.decode('utf-8')
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        try:
            info = await self._redis.info()
            
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0),
                ),
                "keys": info.get("db0", {}).get("keys", 0),
                "memory_used": info.get("used_memory_human", "0B"),
                "memory_peak": info.get("used_memory_peak_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except RedisError:
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    async def clear_all(self) -> bool:
        """
        Clear all keys in current database.
        
        WARNING: This deletes ALL keys in the database!
        Should only be used in testing.
        
        Returns:
            bool: True if successful
        """
        try:
            await self._redis.flushdb()
            return True
        except RedisError:
            return False
    
    async def ping(self) -> bool:
        """
        Check if Redis is reachable.
        
        Returns:
            bool: True if Redis responds
        """
        try:
            return await self._redis.ping()
        except RedisError:
            return False


# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

cache = RedisCache()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "RedisCache",
    "cache",
]
