"""
Redis configuration for Turkish Legal AI.

This module provides Redis-specific configuration and client management:
- Connection pooling
- Multiple database support (cache, queue, session)
- Health checks
- Sentinel support (high availability)
- Cluster support (horizontal scaling)

Redis is used for:
- Application caching
- Session storage
- Celery task queue
- Rate limiting
- Real-time features
"""
import asyncio
from typing import Any

from redis import ConnectionPool, Redis
from redis.asyncio import ConnectionPool as AsyncConnectionPool
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from backend.core.config.settings import settings
from backend.core.constants import (
    CACHE_KEY_PREFIX,
    CACHE_MEDIUM_TTL,
    CACHE_SHORT_TTL,
)


class RedisConfig:
    """
    Redis configuration and connection management.
    
    Manages multiple Redis connections for different purposes:
    - Cache DB (DB 0)
    - Queue DB (DB 1) 
    - Session DB (DB 2)
    
    Supports both sync and async Redis clients.
    """
    
    def __init__(self) -> None:
        """Initialize Redis configuration."""
        self._cache_pool: AsyncConnectionPool | None = None
        self._queue_pool: AsyncConnectionPool | None = None
        self._session_pool: AsyncConnectionPool | None = None
        
        self._cache_client: AsyncRedis | None = None
        self._queue_client: AsyncRedis | None = None
        self._session_client: AsyncRedis | None = None
        
        # Sync clients (for Celery, etc.)
        self._sync_cache_pool: ConnectionPool | None = None
        self._sync_queue_pool: ConnectionPool | None = None
    
    def get_connection_kwargs(self, db: int = 0) -> dict[str, Any]:
        """
        Get Redis connection parameters.
        
        Args:
            db: Redis database number
            
        Returns:
            dict: Connection parameters
        """
        # Parse Redis URL
        url = settings.REDIS_URL
        
        return {
            "db": db,
            "max_connections": settings.REDIS_MAX_CONNECTIONS,
            "decode_responses": True,  # Auto-decode bytes to strings
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "socket_keepalive": True,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        }
    
    # =========================================================================
    # ASYNC CLIENTS
    # =========================================================================
    
    def get_cache_client(self) -> AsyncRedis:
        """
        Get async Redis client for caching.
        
        Returns:
            AsyncRedis: Cache client (DB 0)
            
        Example:
            >>> redis = redis_config.get_cache_client()
            >>> await redis.set("key", "value", ex=3600)
            >>> value = await redis.get("key")
        """
        if self._cache_client is None:
            if self._cache_pool is None:
                self._cache_pool = AsyncConnectionPool.from_url(
                    str(settings.REDIS_URL),
                    **self.get_connection_kwargs(settings.REDIS_CACHE_DB),
                )
            
            self._cache_client = AsyncRedis(connection_pool=self._cache_pool)
        
        return self._cache_client
    
    def get_queue_client(self) -> AsyncRedis:
        """
        Get async Redis client for task queue.
        
        Returns:
            AsyncRedis: Queue client (DB 1)
            
        Example:
            >>> redis = redis_config.get_queue_client()
            >>> await redis.lpush("tasks", "task_data")
        """
        if self._queue_client is None:
            if self._queue_pool is None:
                self._queue_pool = AsyncConnectionPool.from_url(
                    str(settings.REDIS_URL),
                    **self.get_connection_kwargs(settings.REDIS_QUEUE_DB),
                )
            
            self._queue_client = AsyncRedis(connection_pool=self._queue_pool)
        
        return self._queue_client
    
    def get_session_client(self) -> AsyncRedis:
        """
        Get async Redis client for session storage.
        
        Returns:
            AsyncRedis: Session client (DB 2)
            
        Example:
            >>> redis = redis_config.get_session_client()
            >>> await redis.setex("session:user123", 3600, "session_data")
        """
        if self._session_client is None:
            if self._session_pool is None:
                self._session_pool = AsyncConnectionPool.from_url(
                    str(settings.REDIS_URL),
                    **self.get_connection_kwargs(settings.REDIS_SESSION_DB),
                )
            
            self._session_client = AsyncRedis(connection_pool=self._session_pool)
        
        return self._session_client
    
    # =========================================================================
    # SYNC CLIENTS (for Celery)
    # =========================================================================
    
    def get_sync_cache_client(self) -> Redis:
        """
        Get sync Redis client for caching.
        
        Used by synchronous code (e.g., Celery tasks).
        
        Returns:
            Redis: Sync cache client
        """
        if self._sync_cache_pool is None:
            self._sync_cache_pool = ConnectionPool.from_url(
                str(settings.REDIS_URL),
                **self.get_connection_kwargs(settings.REDIS_CACHE_DB),
            )
        
        return Redis(connection_pool=self._sync_cache_pool)
    
    def get_sync_queue_client(self) -> Redis:
        """
        Get sync Redis client for queue.
        
        Returns:
            Redis: Sync queue client
        """
        if self._sync_queue_pool is None:
            self._sync_queue_pool = ConnectionPool.from_url(
                str(settings.REDIS_URL),
                **self.get_connection_kwargs(settings.REDIS_QUEUE_DB),
            )
        
        return Redis(connection_pool=self._sync_queue_pool)
    
    # =========================================================================
    # HEALTH CHECKS
    # =========================================================================
    
    async def check_health(self) -> dict[str, Any]:
        """
        Check Redis health for all databases.
        
        Returns:
            dict: Health status for each database
            
        Example:
            >>> health = await redis_config.check_health()
            >>> print(health["cache"]["healthy"])
            True
        """
        result = {
            "cache": {"healthy": False, "error": None, "latency_ms": None},
            "queue": {"healthy": False, "error": None, "latency_ms": None},
            "session": {"healthy": False, "error": None, "latency_ms": None},
        }
        
        # Check cache DB
        try:
            redis = self.get_cache_client()
            start = asyncio.get_event_loop().time()
            await redis.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000
            
            result["cache"]["healthy"] = True
            result["cache"]["latency_ms"] = round(latency, 2)
        except (ConnectionError, TimeoutError, RedisError) as e:
            result["cache"]["error"] = str(e)
        
        # Check queue DB
        try:
            redis = self.get_queue_client()
            start = asyncio.get_event_loop().time()
            await redis.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000
            
            result["queue"]["healthy"] = True
            result["queue"]["latency_ms"] = round(latency, 2)
        except (ConnectionError, TimeoutError, RedisError) as e:
            result["queue"]["error"] = str(e)
        
        # Check session DB
        try:
            redis = self.get_session_client()
            start = asyncio.get_event_loop().time()
            await redis.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000
            
            result["session"]["healthy"] = True
            result["session"]["latency_ms"] = round(latency, 2)
        except (ConnectionError, TimeoutError, RedisError) as e:
            result["session"]["error"] = str(e)
        
        return result
    
    async def get_info(self) -> dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            dict: Redis server info
        """
        try:
            redis = self.get_cache_client()
            info = await redis.info()
            
            return {
                "version": info.get("redis_version"),
                "uptime_seconds": info.get("uptime_in_seconds"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace": await self._get_keyspace_info(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_keyspace_info(self) -> dict[str, Any]:
        """
        Get keyspace statistics for all databases.
        
        Returns:
            dict: Keyspace info per database
        """
        keyspace = {}
        
        for name, db_num in [
            ("cache", settings.REDIS_CACHE_DB),
            ("queue", settings.REDIS_QUEUE_DB),
            ("session", settings.REDIS_SESSION_DB),
        ]:
            try:
                if name == "cache":
                    redis = self.get_cache_client()
                elif name == "queue":
                    redis = self.get_queue_client()
                else:
                    redis = self.get_session_client()
                
                info = await redis.info("keyspace")
                db_key = f"db{db_num}"
                
                if db_key in info:
                    keyspace[name] = {
                        "keys": info[db_key].get("keys", 0),
                        "expires": info[db_key].get("expires", 0),
                    }
                else:
                    keyspace[name] = {"keys": 0, "expires": 0}
            except Exception:
                keyspace[name] = {"keys": 0, "expires": 0}
        
        return keyspace
    
    # =========================================================================
    # CACHE HELPERS
    # =========================================================================
    
    def make_cache_key(self, *parts: str) -> str:
        """
        Create a namespaced cache key.
        
        Args:
            *parts: Key components
            
        Returns:
            str: Formatted cache key
            
        Example:
            >>> redis_config.make_cache_key("user", "123", "profile")
            'legal_ai:user:123:profile'
        """
        return f"{CACHE_KEY_PREFIX}:{':'.join(parts)}"
    
    async def clear_cache_pattern(self, pattern: str) -> int:
        """
        Clear all cache keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (supports * and ?)
            
        Returns:
            int: Number of keys deleted
            
        Example:
            >>> await redis_config.clear_cache_pattern("user:*")
            42
        """
        redis = self.get_cache_client()
        keys = await redis.keys(pattern)
        
        if keys:
            return await redis.delete(*keys)
        
        return 0
    
    async def get_cache_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            int: TTL in seconds (-1 if no expire, -2 if key doesn't exist)
        """
        redis = self.get_cache_client()
        return await redis.ttl(key)
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    async def close_all(self) -> None:
        """
        Close all Redis connections.
        
        Should be called on application shutdown.
        """
        clients = [
            self._cache_client,
            self._queue_client,
            self._session_client,
        ]
        
        for client in clients:
            if client:
                await client.close()
        
        pools = [
            self._cache_pool,
            self._queue_pool,
            self._session_pool,
        ]
        
        for pool in pools:
            if pool:
                await pool.disconnect()
        
        # Close sync pools
        if self._sync_cache_pool:
            self._sync_cache_pool.disconnect()
        if self._sync_queue_pool:
            self._sync_queue_pool.disconnect()
        
        # Reset all references
        self._cache_client = None
        self._queue_client = None
        self._session_client = None
        self._cache_pool = None
        self._queue_pool = None
        self._session_pool = None
        self._sync_cache_pool = None
        self._sync_queue_pool = None


# =============================================================================
# GLOBAL REDIS CONFIG INSTANCE
# =============================================================================

redis_config = RedisConfig()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_redis() -> AsyncRedis:
    """
    Get default Redis client (cache).
    
    FastAPI dependency for Redis access.
    
    Returns:
        AsyncRedis: Cache client
        
    Example:
        @router.get("/cached")
        async def endpoint(redis: AsyncRedis = Depends(get_redis)):
            value = await redis.get("key")
            return {"value": value}
    """
    return redis_config.get_cache_client()


async def cache_set(
    key: str,
    value: str,
    ttl: int = CACHE_MEDIUM_TTL,
) -> bool:
    """
    Set a cache value with TTL.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
        
    Returns:
        bool: True if successful
    """
    redis = redis_config.get_cache_client()
    return await redis.setex(key, ttl, value)


async def cache_get(key: str) -> str | None:
    """
    Get a cached value.
    
    Args:
        key: Cache key
        
    Returns:
        str | None: Cached value or None
    """
    redis = redis_config.get_cache_client()
    return await redis.get(key)


async def cache_delete(key: str) -> int:
    """
    Delete a cache key.
    
    Args:
        key: Cache key
        
    Returns:
        int: Number of keys deleted (0 or 1)
    """
    redis = redis_config.get_cache_client()
    return await redis.delete(key)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "RedisConfig",
    "redis_config",
    "get_redis",
    "cache_set",
    "cache_get",
    "cache_delete",
]