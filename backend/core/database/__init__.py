"""
Database module for Turkish Legal AI.

This module provides database utilities and ORM models:
- SQLAlchemy Base class
- Session management
- Connection pooling
- Migration helpers
- Model mixins
- Query helpers

Database Stack:
- PostgreSQL 15+ with asyncpg
- SQLAlchemy 2.0 (async)
- Alembic for migrations
- pgvector extension for embeddings

Usage:
    >>> from backend.core.database import Base, get_session
    >>> from backend.core.database import AsyncSession
    >>> 
    >>> # Use in FastAPI dependencies
    >>> @router.get("/users")
    >>> async def get_users(db: AsyncSession = Depends(get_session)):
    ...     result = await db.execute(select(User))
    ...     return result.scalars().all()
"""

# =============================================================================
# BASE CLASSES AND MIXINS
# =============================================================================

from backend.core.database.base import (
    AuditMixin,
    Base,
    BaseModelMixin,
    FullAuditMixin,
    SoftDeleteMixin,
    TenantMixin,
    TenantModelMixin,
    TimestampMixin,
    UUIDMixin,
    VersionMixin,
    generate_table_name,
)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

from backend.core.database.session import (
    AsyncSession,
    DatabaseSession,
    check_database_health,
    clear_current_tenant_id,
    create_all_tables,
    create_read_engine,
    create_write_engine,
    dispose_engines,
    drop_all_tables,
    get_current_tenant_id,
    get_pool_status,
    get_read_session,
    get_read_session_factory,
    get_session,
    get_transactional_session,
    get_write_session_factory,
    set_current_tenant_id,
)

# =============================================================================
# TYPE IMPORTS
# =============================================================================

from sqlalchemy.ext.asyncio import AsyncEngine

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base Classes
    "Base",
    # Mixins
    "UUIDMixin",
    "TimestampMixin",
    "SoftDeleteMixin",
    "TenantMixin",
    "AuditMixin",
    "VersionMixin",
    "BaseModelMixin",
    "FullAuditMixin",
    "TenantModelMixin",
    # Utilities
    "generate_table_name",
    # Session Management
    "AsyncSession",
    "AsyncEngine",
    "get_session",
    "get_read_session",
    "get_transactional_session",
    "get_write_session_factory",
    "get_read_session_factory",
    "DatabaseSession",
    # Engine Management
    "create_write_engine",
    "create_read_engine",
    "dispose_engines",
    # Tenant Isolation
    "set_current_tenant_id",
    "get_current_tenant_id",
    "clear_current_tenant_id",
    # Database Utilities
    "create_all_tables",
    "drop_all_tables",
    "check_database_health",
    "get_pool_status",
]