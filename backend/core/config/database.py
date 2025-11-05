"""
Database configuration for Turkish Legal AI.

This module provides database-specific configuration classes and utilities:
- Database URL parsing
- Connection pool settings
- Read/write splitting
- Health checks
- Migration helpers

Supports:
- PostgreSQL with asyncpg
- Connection pooling
- Read replicas
- pgvector extension
"""
from typing import Any

from sqlalchemy import event, pool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config.settings import settings


class DatabaseConfig:
    """
    Database configuration and connection management.
    
    Provides utilities for:
    - Creating database engines
    - Session management
    - Connection health checks
    - Pool monitoring
    """
    
    def __init__(self) -> None:
        """Initialize database configuration."""
        self.write_engine: AsyncEngine | None = None
        self.read_engine: AsyncEngine | None = None
        self.session_factory: sessionmaker | None = None
    
    def get_engine_config(self) -> dict[str, Any]:
        """
        Get SQLAlchemy engine configuration.
        
        Returns:
            dict: Engine configuration parameters
        """
        return {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": 3600,  # Recycle connections after 1 hour
            "pool_pre_ping": True,  # Verify connections before using
            "echo": settings.DATABASE_ECHO,
            "future": True,  # SQLAlchemy 2.0 style
            "poolclass": pool.NullPool if settings.ENVIRONMENT == "test" else pool.QueuePool,
        }
    
    def create_write_engine(self) -> AsyncEngine:
        """
        Create async engine for write operations.
        
        Returns:
            AsyncEngine: Write database engine
        """
        if self.write_engine is None:
            self.write_engine = create_async_engine(
                settings.DATABASE_URL,
                **self.get_engine_config(),
            )
            
            # Register connection pool listeners
            self._register_pool_listeners(self.write_engine)
        
        return self.write_engine
    
    def create_read_engine(self) -> AsyncEngine:
        """
        Create async engine for read operations.
        
        If no read replica is configured, returns write engine.
        
        Returns:
            AsyncEngine: Read database engine
        """
        if settings.DATABASE_READ_URL:
            if self.read_engine is None:
                self.read_engine = create_async_engine(
                    settings.DATABASE_READ_URL,
                    **self.get_engine_config(),
                )
                self._register_pool_listeners(self.read_engine)
            
            return self.read_engine
        
        # Fallback to write engine if no read replica
        return self.create_write_engine()
    
    def create_session_factory(self) -> sessionmaker:
        """
        Create async session factory.
        
        Returns:
            sessionmaker: Configured session factory
        """
        if self.session_factory is None:
            self.session_factory = sessionmaker(
                bind=self.create_write_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
        
        return self.session_factory
    
    def _register_pool_listeners(self, engine: AsyncEngine) -> None:
        """
        Register SQLAlchemy pool event listeners for monitoring.
        
        Args:
            engine: Database engine to monitor
        """
        @event.listens_for(engine.sync_engine.pool, "connect")
        def receive_connect(dbapi_conn: Any, connection_record: Any) -> None:
            """Log new database connections."""
            if settings.DEBUG:
                print(f"New DB connection: {id(dbapi_conn)}")
        
        @event.listens_for(engine.sync_engine.pool, "checkout")
        def receive_checkout(
            dbapi_conn: Any,
            connection_record: Any,
            connection_proxy: Any,
        ) -> None:
            """Log connection checkouts from pool."""
            if settings.DEBUG:
                print(f"DB connection checkout: {id(dbapi_conn)}")
        
        @event.listens_for(engine.sync_engine.pool, "checkin")
        def receive_checkin(dbapi_conn: Any, connection_record: Any) -> None:
            """Log connection returns to pool."""
            if settings.DEBUG:
                print(f"DB connection checkin: {id(dbapi_conn)}")
    
    async def check_health(self) -> dict[str, Any]:
        """
        Check database health.
        
        Returns:
            dict: Health check results
        """
        from sqlalchemy import text
        
        result = {
            "write": {"healthy": False, "error": None},
            "read": {"healthy": False, "error": None},
        }
        
        # Check write connection
        try:
            engine = self.create_write_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                result["write"]["healthy"] = True
        except Exception as e:
            result["write"]["error"] = str(e)
        
        # Check read connection
        try:
            engine = self.create_read_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                result["read"]["healthy"] = True
        except Exception as e:
            result["read"]["error"] = str(e)
        
        return result
    
    async def dispose_engines(self) -> None:
        """
        Dispose all database engines.
        
        Should be called on application shutdown.
        """
        if self.write_engine:
            await self.write_engine.dispose()
            self.write_engine = None
        
        if self.read_engine:
            await self.read_engine.dispose()
            self.read_engine = None
        
        self.session_factory = None
    
    def get_pool_status(self) -> dict[str, Any]:
        """
        Get connection pool status.
        
        Returns:
            dict: Pool statistics
        """
        status = {}
        
        if self.write_engine:
            pool = self.write_engine.pool
            status["write"] = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total": pool.size() + pool.overflow(),
            }
        
        if self.read_engine:
            pool = self.read_engine.pool
            status["read"] = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total": pool.size() + pool.overflow(),
            }
        
        return status


# =============================================================================
# GLOBAL DATABASE CONFIG INSTANCE
# =============================================================================

db_config = DatabaseConfig()


# =============================================================================
# DEPENDENCY INJECTION HELPERS
# =============================================================================

async def get_db_session() -> AsyncSession:
    """
    FastAPI dependency for database sessions.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    session_factory = db_config.create_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_read_db_session() -> AsyncSession:
    """
    FastAPI dependency for read-only database sessions.
    
    Uses read replica if configured, otherwise falls back to write engine.
    
    Yields:
        AsyncSession: Read-only database session
        
    Example:
        @router.get("/reports")
        async def get_reports(db: AsyncSession = Depends(get_read_db_session)):
            result = await db.execute(select(Report))
            return result.scalars().all()
    """
    engine = db_config.create_read_engine()
    session_factory = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DatabaseConfig",
    "db_config",
    "get_db_session",
    "get_read_db_session",
]