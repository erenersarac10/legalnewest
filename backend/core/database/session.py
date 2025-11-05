"""
Database session management for Turkish Legal AI.

This module provides:
- Async session factory
- Session lifecycle management
- Transaction handling
- Context managers
- FastAPI dependencies
- Read/write session separation
- Connection pooling
- Tenant isolation helpers

Usage:
    >>> from backend.core.database.session import get_session
    >>> 
    >>> @router.get("/users")
    >>> async def get_users(db: AsyncSession = Depends(get_session)):
    ...     result = await db.execute(select(User))
    ...     return result.scalars().all()
"""
import contextvars
from datetime import datetime
from typing import Any, AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from backend.core.config.settings import settings
from backend.core.database.base import Base

# Context variable for tenant ID (thread-safe)
_tenant_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_id", default=None
)

# Global engine instances
_write_engine: AsyncEngine | None = None
_read_engine: AsyncEngine | None = None

# Global session factories
_write_session_factory: async_sessionmaker[AsyncSession] | None = None
_read_session_factory: async_sessionmaker[AsyncSession] | None = None


# =============================================================================
# ENGINE CREATION
# =============================================================================

def get_engine_config() -> dict[str, Any]:
    """
    Get SQLAlchemy engine configuration.
    
    Returns:
        dict: Engine configuration parameters
    """
    # Use NullPool for testing to avoid connection issues
    pool_class = NullPool if settings.ENVIRONMENT == "test" else QueuePool
    
    return {
        "echo": settings.DATABASE_ECHO,
        "echo_pool": settings.DEBUG and settings.DATABASE_ECHO,
        "pool_pre_ping": True,  # Verify connections before using
        "pool_size": settings.DATABASE_POOL_SIZE,
        "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "poolclass": pool_class,
        "connect_args": {
            "server_settings": {
                "application_name": f"turkish-legal-ai-{settings.ENVIRONMENT}",
                "jit": "off",  # Disable JIT for better compatibility
            },
            "command_timeout": 60,
            "timeout": 30,
        },
    }


def create_write_engine() -> AsyncEngine:
    """
    Create async engine for write operations.
    
    Returns:
        AsyncEngine: Write database engine
    """
    global _write_engine
    
    if _write_engine is None:
        _write_engine = create_async_engine(
            settings.DATABASE_URL,
            **get_engine_config(),
        )
        
        # Register event listeners
        _register_engine_listeners(_write_engine, "write")
    
    return _write_engine


def create_read_engine() -> AsyncEngine:
    """
    Create async engine for read operations.
    
    Uses read replica if configured, otherwise returns write engine.
    
    Returns:
        AsyncEngine: Read database engine
    """
    global _read_engine
    
    if settings.DATABASE_READ_URL:
        if _read_engine is None:
            _read_engine = create_async_engine(
                settings.DATABASE_READ_URL,
                **get_engine_config(),
            )
            _register_engine_listeners(_read_engine, "read")
        
        return _read_engine
    
    # Fallback to write engine
    return create_write_engine()


def _register_engine_listeners(engine: AsyncEngine, engine_type: str) -> None:
    """
    Register SQLAlchemy event listeners for monitoring.
    
    Args:
        engine: Database engine
        engine_type: Engine type ('write' or 'read')
    """
    @event.listens_for(engine.sync_engine, "connect")
    def receive_connect(dbapi_conn: Any, connection_record: Any) -> None:
        """Set up connection parameters on new connections."""
        # Set statement timeout (30 seconds)
        with dbapi_conn.cursor() as cursor:
            cursor.execute("SET statement_timeout = '30s'")
            
            # Set timezone to UTC
            cursor.execute("SET timezone = 'UTC'")
            
            if settings.DEBUG:
                print(f"[{engine_type}] New DB connection established")
    
    @event.listens_for(engine.sync_engine, "checkout")
    def receive_checkout(
        dbapi_conn: Any,
        connection_record: Any,
        connection_proxy: Any,
    ) -> None:
        """Handle connection checkout from pool."""
        if settings.DEBUG and settings.DATABASE_ECHO:
            print(f"[{engine_type}] Connection checked out from pool")
    
    @event.listens_for(engine.sync_engine, "checkin")
    def receive_checkin(dbapi_conn: Any, connection_record: Any) -> None:
        """Handle connection return to pool."""
        if settings.DEBUG and settings.DATABASE_ECHO:
            print(f"[{engine_type}] Connection returned to pool")


# =============================================================================
# SESSION FACTORIES
# =============================================================================

def get_write_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get async session factory for write operations.
    
    Returns:
        async_sessionmaker: Session factory
    """
    global _write_session_factory
    
    if _write_session_factory is None:
        engine = create_write_engine()
        
        _write_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    
    return _write_session_factory


def get_read_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get async session factory for read operations.
    
    Returns:
        async_sessionmaker: Session factory
    """
    global _read_session_factory
    
    if _read_session_factory is None:
        engine = create_read_engine()
        
        _read_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=True,  # Auto-flush for read operations
        )
    
    return _read_session_factory


# =============================================================================
# SESSION DEPENDENCIES
# =============================================================================

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions (write).
    
    Provides transactional session that automatically commits on success
    or rolls back on exception. Includes enhanced logging for monitoring.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        >>> @router.post("/users")
        >>> async def create_user(
        ...     user: UserCreate,
        ...     db: AsyncSession = Depends(get_session)
        ... ):
        ...     new_user = User(**user.dict())
        ...     db.add(new_user)
        ...     await db.commit()
        ...     return new_user
    """
    session_factory = get_write_session_factory()
    session_start = datetime.now()
    
    async with session_factory() as session:
        try:
            # Apply tenant filter if tenant context is set
            tenant_id = get_current_tenant_id()
            if tenant_id:
                await _apply_tenant_filter(session, tenant_id)
            
            yield session
            
            # Commit if no exceptions
            await session.commit()
            
            # Log successful transaction
            if settings.DEBUG:
                duration_ms = (datetime.now() - session_start).total_seconds() * 1000
                print(f"[DB] Session committed successfully (duration: {duration_ms:.2f}ms)")
            
        except Exception as e:
            # Rollback on any exception
            await session.rollback()
            
            # Log failed transaction
            duration_ms = (datetime.now() - session_start).total_seconds() * 1000
            print(
                f"[DB] Session rollback due to error: {type(e).__name__} "
                f"(duration: {duration_ms:.2f}ms)"
            )
            raise
        finally:
            # Always close session
            await session.close()


async def get_read_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for read-only database sessions.
    
    Uses read replica if configured, optimized for read operations.
    
    Yields:
        AsyncSession: Read-only database session
        
    Example:
        >>> @router.get("/users")
        >>> async def get_users(db: AsyncSession = Depends(get_read_session)):
        ...     result = await db.execute(select(User))
        ...     return result.scalars().all()
    """
    session_factory = get_read_session_factory()
    
    async with session_factory() as session:
        try:
            # Apply tenant filter if tenant context is set
            tenant_id = get_current_tenant_id()
            if tenant_id:
                await _apply_tenant_filter(session, tenant_id)
            
            yield session
            
        finally:
            await session.close()


async def get_transactional_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get session with explicit transaction control.
    
    Caller must manually commit or rollback.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        >>> async with get_transactional_session() as db:
        ...     user = User(email="test@example.com")
        ...     db.add(user)
        ...     await db.flush()  # Get ID without committing
        ...     
        ...     # Do other work...
        ...     
        ...     await db.commit()  # Explicit commit
    """
    session_factory = get_write_session_factory()
    
    async with session_factory() as session:
        try:
            # Apply tenant filter
            tenant_id = get_current_tenant_id()
            if tenant_id:
                await _apply_tenant_filter(session, tenant_id)
            
            yield session
            
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# TENANT ISOLATION
# =============================================================================

def set_current_tenant_id(tenant_id: str) -> None:
    """
    Set tenant ID for current context.
    
    This tenant ID will be used to filter all queries automatically.
    
    Args:
        tenant_id: Tenant identifier
        
    Example:
        >>> set_current_tenant_id("tenant_123")
        >>> # All subsequent queries will filter by tenant_id
    """
    _tenant_id_ctx.set(tenant_id)


def get_current_tenant_id() -> str | None:
    """
    Get current tenant ID from context.
    
    Returns:
        str | None: Current tenant ID or None
    """
    return _tenant_id_ctx.get()


def clear_current_tenant_id() -> None:
    """Clear tenant ID from context."""
    _tenant_id_ctx.set(None)


async def _apply_tenant_filter(session: AsyncSession, tenant_id: str) -> None:
    """
    Apply tenant filter to session using PostgreSQL RLS.
    
    Sets session variable that can be used by Row-Level Security policies.
    
    Args:
        session: Database session
        tenant_id: Tenant identifier
    """
    # Set PostgreSQL session variable for RLS
    await session.execute(
        text("SET app.current_tenant_id = :tenant_id"),
        {"tenant_id": tenant_id}
    )


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

async def create_all_tables() -> None:
    """
    Create all database tables.
    
    Should only be used in development/testing.
    Use Alembic migrations in production.
    """
    engine = create_write_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_tables() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data!
    Should only be used in testing.
    """
    engine = create_write_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def check_database_health() -> dict[str, Any]:
    """
    Check database connection health.
    
    Returns:
        dict: Health status
        
    Example:
        >>> health = await check_database_health()
        >>> print(health["write"]["healthy"])
        True
    """
    result = {
        "write": {"healthy": False, "error": None, "latency_ms": None},
        "read": {"healthy": False, "error": None, "latency_ms": None},
    }
    
    # Check write connection
    try:
        engine = create_write_engine()
        start = datetime.now()
        
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        latency = (datetime.now() - start).total_seconds() * 1000
        result["write"]["healthy"] = True
        result["write"]["latency_ms"] = round(latency, 2)
        
    except Exception as e:
        result["write"]["error"] = str(e)
    
    # Check read connection
    try:
        engine = create_read_engine()
        start = datetime.now()
        
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        latency = (datetime.now() - start).total_seconds() * 1000
        result["read"]["healthy"] = True
        result["read"]["latency_ms"] = round(latency, 2)
        
    except Exception as e:
        result["read"]["error"] = str(e)
    
    return result


async def get_pool_status() -> dict[str, Any]:
    """
    Get connection pool status.
    
    Returns:
        dict: Pool statistics
    """
    status = {}
    
    if _write_engine:
        pool = _write_engine.pool
        status["write"] = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
        }
    
    if _read_engine:
        pool = _read_engine.pool
        status["read"] = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
        }
    
    return status


async def dispose_engines() -> None:
    """
    Dispose all database engines.
    
    Should be called on application shutdown.
    """
    global _write_engine, _read_engine
    global _write_session_factory, _read_session_factory
    
    if _write_engine:
        await _write_engine.dispose()
        _write_engine = None
    
    if _read_engine:
        await _read_engine.dispose()
        _read_engine = None
    
    _write_session_factory = None
    _read_session_factory = None


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

class DatabaseSession:
    """
    Context manager for database sessions.
    
    Provides more control over session lifecycle than dependencies.
    
    Example:
        >>> async with DatabaseSession() as db:
        ...     user = await db.get(User, user_id)
        ...     user.name = "Updated"
        ...     # Auto-commits on exit
    """
    
    def __init__(self, read_only: bool = False) -> None:
        """
        Initialize database session context.
        
        Args:
            read_only: Use read-only session
        """
        self.read_only = read_only
        self.session: AsyncSession | None = None
    
    async def __aenter__(self) -> AsyncSession:
        """Enter context - create session."""
        factory = (
            get_read_session_factory()
            if self.read_only
            else get_write_session_factory()
        )
        
        self.session = factory()
        
        # Apply tenant filter
        tenant_id = get_current_tenant_id()
        if tenant_id:
            await _apply_tenant_filter(self.session, tenant_id)
        
        return self.session
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - commit or rollback."""
        if self.session:
            try:
                if exc_type is None and not self.read_only:
                    await self.session.commit()
                else:
                    await self.session.rollback()
            finally:
                await self.session.close()
                self.session = None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Engines
    "create_write_engine",
    "create_read_engine",
    # Session Factories
    "get_write_session_factory",
    "get_read_session_factory",
    # Dependencies
    "get_session",
    "get_read_session",
    "get_transactional_session",
    # Tenant Isolation
    "set_current_tenant_id",
    "get_current_tenant_id",
    "clear_current_tenant_id",
    # Utilities
    "create_all_tables",
    "drop_all_tables",
    "check_database_health",
    "get_pool_status",
    "dispose_engines",
    # Context Managers
    "DatabaseSession",
]