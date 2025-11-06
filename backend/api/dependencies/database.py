"""
Database Session Dependencies for Turkish Legal AI Platform.

Enterprise-grade database dependency injection with connection pooling, transaction
management, read/write splitting, and Row-Level Security (RLS) integration.

=============================================================================
FEATURES
=============================================================================

1. Session Management
   -------------------
   - Async database sessions with SQLAlchemy
   - Automatic commit/rollback on success/failure
   - Connection pooling with configurable limits
   - Session lifecycle management
   - Context manager support

2. Read/Write Splitting
   ----------------------
   - Automatic read replica routing
   - Load balancing across read replicas
   - Failover to primary on replica failure
   - Write-after-read consistency

3. Row-Level Security (RLS)
   -------------------------
   - Automatic tenant context injection
   - PostgreSQL RLS policy enforcement
   - Secure multi-tenant data isolation
   - Context variable management

4. Transaction Management
   -----------------------
   - ACID transaction guarantees
   - Savepoint support for nested transactions
   - Deadlock detection and retry logic
   - Transaction isolation levels

5. Performance Monitoring
   ------------------------
   - Query execution time tracking
   - Connection pool statistics
   - Slow query detection
   - Database health checks

=============================================================================
USAGE
=============================================================================

Basic Database Session:
-----------------------

>>> from fastapi import Depends
>>> from sqlalchemy import select
>>> from backend.api.dependencies.database import get_db
>>> from backend.db.models import Contract
>>>
>>> @app.get("/contracts")
>>> async def get_contracts(db: AsyncSession = Depends(get_db)):
...     result = await db.execute(select(Contract))
...     contracts = result.scalars().all()
...     return contracts

Read-Only Session (Read Replica):
----------------------------------

>>> from backend.api.dependencies.database import get_read_db
>>>
>>> @app.get("/reports")
>>> async def get_reports(db: AsyncSession = Depends(get_read_db)):
...     # This uses read replica for load distribution
...     result = await db.execute(select(Report))
...     return result.scalars().all()

Transaction Management:
-----------------------

>>> from backend.api.dependencies.database import get_transaction
>>>
>>> @app.post("/contracts")
>>> async def create_contract(
...     contract: ContractCreate,
...     db: AsyncSession = Depends(get_db)
... ):
...     # Automatic transaction management
...     new_contract = Contract(**contract.dict())
...     db.add(new_contract)
...     await db.commit()  # Commits automatically
...     await db.refresh(new_contract)
...     return new_contract

Multi-Tenant RLS:
-----------------

>>> # Tenant context automatically set by middleware
>>> @app.get("/contracts/{contract_id}")
>>> async def get_contract(
...     contract_id: str,
...     request: Request,
...     db: AsyncSession = Depends(get_db)
... ):
...     # RLS automatically filters by tenant_id from request.state
...     result = await db.execute(
...         select(Contract).where(Contract.id == contract_id)
...     )
...     contract = result.scalar_one_or_none()
...
...     if not contract:
...         raise HTTPException(404, "Contract not found")
...
...     return contract

Manual Transaction with Savepoints:
-----------------------------------

>>> async with get_db_transaction() as session:
...     # Create contract
...     contract = Contract(title="Test")
...     session.add(contract)
...
...     # Create savepoint
...     savepoint = await session.begin_nested()
...
...     try:
...         # Try to add clauses
...         for clause_data in clauses:
...             clause = Clause(**clause_data, contract_id=contract.id)
...             session.add(clause)
...         await savepoint.commit()
...     except Exception:
...         # Rollback to savepoint, keep contract
...         await savepoint.rollback()
...
...     await session.commit()

=============================================================================
CONNECTION POOLING
=============================================================================

Pool Configuration:
-------------------

Default Settings:
  - pool_size: 20 (max workers * 2)
  - max_overflow: 10 (additional connections)
  - pool_timeout: 30 seconds
  - pool_recycle: 3600 seconds (1 hour)
  - pool_pre_ping: True (validate connections)

Example Configuration:
  Total workers: 10 (uvicorn --workers 10)
  Pool size per worker: 20
  Max overflow: 10
  Total possible connections: 10 * (20 + 10) = 300

PostgreSQL max_connections should be set to:
  max_connections = workers * (pool_size + max_overflow) + 10
  max_connections = 10 * 30 + 10 = 310

Pool Monitoring:
----------------

>>> from backend.core.database import get_pool_stats
>>>
>>> stats = await get_pool_stats()
>>> print(f"Size: {stats['size']}")         # Current pool size
>>> print(f"Checked out: {stats['checked_out']}")  # Active connections
>>> print(f"Overflow: {stats['overflow']}")        # Overflow connections
>>> print(f"Queue: {stats['queue']}")              # Waiting requests

=============================================================================
READ/WRITE SPLITTING
=============================================================================

Architecture:
-------------

Primary Database (Write):
  - All INSERT, UPDATE, DELETE operations
  - Critical READ operations requiring consistency
  - Located in: eu-central-1 (Frankfurt)

Read Replicas (Read-Only):
  - SELECT operations
  - Reporting and analytics
  - Load balancing across multiple replicas
  - Located in: eu-central-1, eu-west-1

Replication Lag:
  - Typical lag: < 100ms
  - Maximum acceptable lag: 5 seconds
  - Monitoring and alerting on lag > 1 second

Usage Pattern:
--------------

Write Operations (use get_db):
  - User registration
  - Contract creation/updates
  - Document uploads
  - Payment processing

Read Operations (use get_read_db):
  - List contracts
  - Search documents
  - Generate reports
  - Dashboard analytics

=============================================================================
ROW-LEVEL SECURITY (RLS)
=============================================================================

PostgreSQL RLS Setup:
---------------------

-- Enable RLS on tables
ALTER TABLE contracts ENABLE ROW LEVEL SECURITY;

-- Create policy
CREATE POLICY tenant_isolation_policy ON contracts
  USING (tenant_id = current_setting('app.tenant_id', TRUE)::uuid);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON contracts TO app_user;

Automatic Context Injection:
-----------------------------

1. TenantContextMiddleware extracts tenant_id from request
2. Stores tenant_id in request.state
3. get_db dependency reads tenant_id from state
4. Executes: SET LOCAL app.tenant_id = 'uuid'
5. All subsequent queries filtered by RLS policy

Security Benefits:
------------------

✓ Impossible to query other tenants' data (enforced by database)
✓ Protection against SQL injection attacks
✓ No application-level filtering needed (database handles it)
✓ Audit trail at database level
✓ Performance: database-level filtering is optimized

=============================================================================
TRANSACTION ISOLATION LEVELS
=============================================================================

Available Levels:
-----------------

READ UNCOMMITTED:
  - Not supported by PostgreSQL
  - Falls back to READ COMMITTED

READ COMMITTED (Default):
  - Sees only committed data
  - Each statement sees fresh snapshot
  - Best for most operations

REPEATABLE READ:
  - Consistent snapshot for entire transaction
  - Prevents non-repeatable reads
  - Use for reports requiring consistency

SERIALIZABLE:
  - Strictest isolation
  - Serializes transactions
  - Use for critical financial operations

Usage:
------

>>> from sqlalchemy import text
>>>
>>> async with get_db_transaction(isolation_level="REPEATABLE READ") as session:
...     # All queries see same snapshot
...     contracts = await session.execute(select(Contract))
...     total = await session.execute(
...         text("SELECT COUNT(*) FROM contracts")
...     )
...     # Results are consistent

=============================================================================
DEADLOCK HANDLING
=============================================================================

Deadlock Detection:
-------------------

PostgreSQL automatically detects deadlocks and aborts one transaction.
Error: psycopg2.errors.DeadlockDetected

Retry Strategy:
---------------

>>> from tenacity import retry, stop_after_attempt, wait_exponential
>>> from sqlalchemy.exc import OperationalError
>>>
>>> @retry(
...     stop=stop_after_attempt(3),
...     wait=wait_exponential(multiplier=1, min=1, max=10),
...     retry=retry_if_exception_type(OperationalError)
... )
>>> async def create_contract_with_retry(contract_data, db):
...     contract = Contract(**contract_data)
...     db.add(contract)
...     await db.commit()
...     return contract

Prevention:
-----------

1. Access tables in consistent order
2. Keep transactions short
3. Use appropriate locking (FOR UPDATE)
4. Batch operations when possible
5. Monitor lock waits

=============================================================================
PERFORMANCE OPTIMIZATION
=============================================================================

Query Optimization:
-------------------

1. Use indexes:
   - CREATE INDEX idx_contracts_tenant ON contracts(tenant_id, created_at DESC)

2. Eager loading:
   - selectinload() for relationships
   - joinedload() for single relationships

3. Pagination:
   - Use LIMIT/OFFSET or cursor-based pagination
   - Avoid COUNT(*) on large tables

4. Batch operations:
   - bulk_insert_mappings() for inserts
   - bulk_update_mappings() for updates

Connection Pool Tuning:
-----------------------

High Traffic:
  - Increase pool_size (20 → 30)
  - Increase max_overflow (10 → 20)
  - Monitor queue size

Low Traffic:
  - Decrease pool_size (20 → 10)
  - Reduce max_overflow (10 → 5)
  - Increase pool_recycle (3600 → 7200)

=============================================================================
TROUBLESHOOTING
=============================================================================

"Too many connections":
-----------------------
1. Check PostgreSQL max_connections setting
2. Review pool_size * workers calculation
3. Monitor connection leaks (unclosed sessions)
4. Implement connection pool monitoring
5. Consider connection pooler (PgBouncer)

"SSL connection has been closed unexpectedly":
-----------------------------------------------
1. Enable pool_pre_ping (validates connections)
2. Reduce pool_recycle time (3600 → 1800)
3. Check network stability
4. Review firewall timeout settings

"Deadlock detected":
--------------------
1. Implement retry logic with exponential backoff
2. Review transaction ordering
3. Keep transactions short
4. Use NOWAIT or SKIP LOCKED
5. Monitor pg_stat_activity for locks

"Slow queries":
---------------
1. Enable query logging: log_min_duration_statement = 1000
2. Use EXPLAIN ANALYZE
3. Add appropriate indexes
4. Review N+1 query patterns
5. Consider materialized views

=============================================================================
KVKK COMPLIANCE
=============================================================================

Data Encryption:
----------------
- SSL/TLS for database connections (required)
- Encryption at rest (PostgreSQL transparent encryption)
- Connection string security (environment variables)

Access Control:
---------------
- Row-Level Security for tenant isolation
- Principle of least privilege (grant only needed permissions)
- Audit logging (pg_audit extension)
- Regular access reviews

Data Retention:
---------------
- Implement soft deletes (deleted_at column)
- Archive old data to cold storage
- Automated deletion workflows
- Compliance with retention policies

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import get_logger
from backend.core.database import DatabaseSession, get_session as core_get_session

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# DATABASE SESSION DEPENDENCIES
# =============================================================================


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Provides an async database session with automatic cleanup and RLS context.

    Args:
        request: FastAPI request (to extract tenant_id from state)

    Yields:
        AsyncSession: Database session with tenant context

    Example:
        >>> @app.get("/contracts")
        >>> async def get_contracts(db: AsyncSession = Depends(get_db)):
        ...     result = await db.execute(select(Contract))
        ...     return result.scalars().all()
    """
    # Extract tenant context from request state (set by TenantContextMiddleware)
    tenant_id = getattr(request.state, "tenant_id", None)
    request_id = getattr(request.state, "request_id", None)

    logger.debug(
        "Database session başlatılıyor",
        tenant_id=tenant_id,
        request_id=request_id,
    )

    # Get database session from core
    async with DatabaseSession() as session:
        try:
            # Set tenant context for RLS
            if tenant_id:
                await session.execute(
                    text("SET LOCAL app.tenant_id = :tenant_id"),
                    {"tenant_id": str(tenant_id)}
                )
                logger.debug("RLS context ayarlandı", tenant_id=tenant_id)

            # Set request context for logging
            if request_id:
                await session.execute(
                    text("SET LOCAL app.request_id = :request_id"),
                    {"request_id": request_id}
                )

            yield session

        except Exception as e:
            logger.error(
                "Database session hatası",
                error=str(e),
                tenant_id=tenant_id,
                request_id=request_id,
            )
            await session.rollback()
            raise
        finally:
            logger.debug("Database session kapatılıyor", request_id=request_id)


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
        >>> @app.get("/reports")
        >>> async def get_reports(db: AsyncSession = Depends(get_read_db)):
        ...     result = await db.execute(select(Report))
        ...     return result.scalars().all()
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    request_id = getattr(request.state, "request_id", None)

    logger.debug("Read-only session başlatılıyor", request_id=request_id)

    # Use read replica session from core
    async with DatabaseSession(read_only=True) as session:
        try:
            # Set tenant context
            if tenant_id:
                await session.execute(
                    text("SET LOCAL app.tenant_id = :tenant_id"),
                    {"tenant_id": str(tenant_id)}
                )

            # Set transaction to read-only
            await session.execute(text("SET TRANSACTION READ ONLY"))

            yield session

        except Exception as e:
            logger.error(
                "Read-only session hatası",
                error=str(e),
                tenant_id=tenant_id,
            )
            raise


@asynccontextmanager
async def get_db_transaction(
    tenant_id: Optional[str] = None,
    isolation_level: str = "READ COMMITTED"
) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for explicit transaction control.

    Args:
        tenant_id: Optional tenant ID for RLS
        isolation_level: Transaction isolation level

    Yields:
        AsyncSession: Database session in transaction

    Example:
        >>> async with get_db_transaction(tenant_id="abc") as session:
        ...     contract = Contract(title="Test")
        ...     session.add(contract)
        ...     await session.commit()
    """
    async with DatabaseSession() as session:
        # Set isolation level
        await session.execute(
            text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}")
        )

        # Set tenant context
        if tenant_id:
            await session.execute(
                text("SET LOCAL app.tenant_id = :tenant_id"),
                {"tenant_id": tenant_id}
            )

        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "get_db",
    "get_read_db",
    "get_db_transaction",
]
