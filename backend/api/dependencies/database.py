"""
Database Session Dependencies for Turkish Legal AI Platform.

Provides FastAPI dependency injection for database sessions.

Features:
- Async database session management
- Automatic commit/rollback
- Connection pooling
- Read replica support
- Tenant context setting

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import DatabaseSession, get_session as core_get_session


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Provides an async database session with automatic cleanup.
    Sets tenant context for Row-Level Security (RLS) if multi-tenant is enabled.

    Args:
        request: FastAPI request (to extract tenant_id from state)

    Yields:
        AsyncSession: Database session

    Example:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    # Extract tenant context from request state (set by TenantContextMiddleware)
    tenant_id = getattr(request.state, "tenant_id", None)

    # Get database session from core
    async with DatabaseSession() as session:
        # Set tenant context for RLS
        if tenant_id:
            # Execute: SET LOCAL app.tenant_id = 'tenant-uuid'
            await session.execute(
                f"SET LOCAL app.tenant_id = '{tenant_id}'"
            )

        yield session


async def get_read_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for read-only database session.

    Uses read replica if configured, otherwise falls back to primary.
    Useful for read-heavy endpoints to distribute load.

    Args:
        request: FastAPI request

    Yields:
        AsyncSession: Read-only database session

    Example:
        @app.get("/reports")
        async def get_reports(db: AsyncSession = Depends(get_read_db)):
            # This uses read replica
            result = await db.execute(select(Report))
            return result.scalars().all()
    """
    tenant_id = getattr(request.state, "tenant_id", None)

    # Use read replica session from core
    # (Implementation would check settings.DATABASE_READ_REPLICA_ENABLED)
    async with DatabaseSession(read_only=True) as session:
        if tenant_id:
            await session.execute(
                f"SET LOCAL app.tenant_id = '{tenant_id}'"
            )

        yield session


__all__ = ["get_db", "get_read_db"]
