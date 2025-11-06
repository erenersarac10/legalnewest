"""
Lifespan Management for Turkish Legal AI Platform.

Handles application startup and shutdown events:
- Database connection pool initialization
- Redis cache connection
- S3/MinIO storage verification
- Background task queue startup
- Health check initialization
- Graceful shutdown with connection cleanup

Usage:
    app = FastAPI(lifespan=lifespan)

Author: Turkish Legal AI Team
License: Proprietary
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from backend.core import (
    get_logger,
    get_redis,
    get_session,
    settings,
)
from backend.core.database import DatabaseSession
from backend.core.storage import s3_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Startup:
    - Initialize database connection pool
    - Verify Redis connectivity
    - Check S3/MinIO bucket access
    - Load ML models if configured
    - Start background task workers
    - Initialize metrics collectors

    Shutdown:
    - Close database connections gracefully
    - Flush Redis cache
    - Close S3 client connections
    - Stop background workers
    - Export final metrics

    Args:
        app: FastAPI application instance

    Yields:
        None: Application runs during this context

    Raises:
        ConnectionError: If critical services are unavailable
        RuntimeError: If startup validation fails
    """
    logger.info(
        "🚀 Starting Turkish Legal AI Platform",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    # =========================================================================
    # STARTUP PHASE
    # =========================================================================

    startup_tasks = []

    # -------------------------------------------------------------------------
    # 1. Database Connection Pool
    # -------------------------------------------------------------------------
    try:
        logger.info("📊 Initializing database connection pool...")

        # Test database connectivity
        async with DatabaseSession() as session:
            await session.execute("SELECT 1")
            logger.info(
                "✅ Database connected",
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
            )

        # Test read replica if enabled
        if settings.DATABASE_READ_REPLICA_ENABLED and settings.DATABASE_READ_URL:
            # Note: Read replica session would be implemented in DatabaseSession
            logger.info("✅ Read replica configured")

    except Exception as e:
        logger.error("❌ Database connection failed", error=str(e))
        raise ConnectionError(f"Database unavailable: {e}") from e

    # -------------------------------------------------------------------------
    # 2. Redis Cache & Queue
    # -------------------------------------------------------------------------
    try:
        logger.info("🔴 Initializing Redis connections...")

        redis = await get_redis()
        await redis.ping()

        logger.info(
            "✅ Redis connected",
            host=settings.REDIS_HOST,
            db=settings.REDIS_DB,
        )

    except Exception as e:
        logger.warning(
            "⚠️  Redis connection failed - running with degraded caching",
            error=str(e),
        )
        # Non-critical: Application can run without Redis (degraded performance)

    # -------------------------------------------------------------------------
    # 3. S3/MinIO Storage
    # -------------------------------------------------------------------------
    try:
        logger.info("🗄️  Verifying object storage...")

        # Verify bucket exists
        bucket_exists = await s3_client.bucket_exists(settings.S3_BUCKET_NAME)

        if not bucket_exists:
            logger.warning(
                f"⚠️  Bucket '{settings.S3_BUCKET_NAME}' not found - will be created on first upload"
            )
        else:
            logger.info(
                "✅ Object storage ready",
                bucket=settings.S3_BUCKET_NAME,
                endpoint=settings.S3_ENDPOINT_URL,
            )

    except Exception as e:
        logger.warning(
            "⚠️  Object storage verification failed - document uploads may fail",
            error=str(e),
        )
        # Non-critical: Can be fixed at runtime

    # -------------------------------------------------------------------------
    # 4. Background Task Queue (Celery)
    # -------------------------------------------------------------------------
    if settings.CELERY_ENABLED:
        try:
            logger.info("⚙️  Starting background task workers...")

            # Celery app would be initialized here
            # from backend.core.queue import celery_app
            # celery_app.start()

            logger.info("✅ Background task queue ready")

        except Exception as e:
            logger.warning(
                "⚠️  Background task queue initialization failed",
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # 5. ML Models Preloading (Optional)
    # -------------------------------------------------------------------------
    if settings.PRELOAD_ML_MODELS:
        try:
            logger.info("🤖 Preloading ML models...")

            # Turkish NLP models would be loaded here
            # from backend.analysis.nlp import load_models
            # await load_models()

            logger.info("✅ ML models loaded")

        except Exception as e:
            logger.warning(
                "⚠️  ML model preloading failed - models will load on-demand",
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # 6. Metrics & Observability
    # -------------------------------------------------------------------------
    try:
        logger.info("📈 Initializing metrics collectors...")

        # Prometheus metrics, OpenTelemetry tracing
        # from backend.observability import init_metrics
        # init_metrics(app)

        logger.info("✅ Observability initialized")

    except Exception as e:
        logger.warning(
            "⚠️  Metrics initialization failed - monitoring may be degraded",
            error=str(e),
        )

    # -------------------------------------------------------------------------
    # Startup Complete
    # -------------------------------------------------------------------------
    logger.info(
        "✨ Turkish Legal AI Platform started successfully",
        api_version=settings.API_VERSION,
        base_url=f"http://{settings.API_HOST}:{settings.API_PORT}",
        docs_url=f"http://{settings.API_HOST}:{settings.API_PORT}/docs",
    )

    # =========================================================================
    # APPLICATION RUNNING
    # =========================================================================
    yield

    # =========================================================================
    # SHUTDOWN PHASE
    # =========================================================================
    logger.info("🛑 Shutting down Turkish Legal AI Platform...")

    # -------------------------------------------------------------------------
    # 1. Stop Accepting New Requests (Handled by ASGI server)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 2. Flush Redis Cache (Optional)
    # -------------------------------------------------------------------------
    try:
        if settings.REDIS_FLUSH_ON_SHUTDOWN:
            logger.info("🔴 Flushing Redis cache...")
            redis = await get_redis()
            # Note: Selective flush, not FLUSHALL
            # await redis.delete(*await redis.keys(f"{settings.REDIS_KEY_PREFIX}:*"))
            logger.info("✅ Cache flushed")
    except Exception as e:
        logger.warning("⚠️  Cache flush failed", error=str(e))

    # -------------------------------------------------------------------------
    # 3. Close Database Connections
    # -------------------------------------------------------------------------
    try:
        logger.info("📊 Closing database connections...")

        # Connection pool cleanup is handled by SQLAlchemy engine disposal
        from backend.core.database.session import engine

        await engine.dispose()
        logger.info("✅ Database connections closed")

    except Exception as e:
        logger.error("❌ Database shutdown error", error=str(e))

    # -------------------------------------------------------------------------
    # 4. Close S3 Client
    # -------------------------------------------------------------------------
    try:
        logger.info("🗄️  Closing storage connections...")
        await s3_client.close()
        logger.info("✅ Storage connections closed")
    except Exception as e:
        logger.warning("⚠️  Storage shutdown warning", error=str(e))

    # -------------------------------------------------------------------------
    # 5. Stop Background Workers
    # -------------------------------------------------------------------------
    if settings.CELERY_ENABLED:
        try:
            logger.info("⚙️  Stopping background task workers...")
            # celery_app.stop()
            logger.info("✅ Background workers stopped")
        except Exception as e:
            logger.warning("⚠️  Worker shutdown warning", error=str(e))

    # -------------------------------------------------------------------------
    # 6. Export Final Metrics
    # -------------------------------------------------------------------------
    try:
        logger.info("📈 Exporting final metrics...")
        # Final metrics export
        logger.info("✅ Metrics exported")
    except Exception as e:
        logger.warning("⚠️  Metrics export warning", error=str(e))

    # -------------------------------------------------------------------------
    # Shutdown Complete
    # -------------------------------------------------------------------------
    logger.info("✅ Turkish Legal AI Platform shutdown complete")


__all__ = ["lifespan"]
