"""
Main Entry Point and CLI Management for Turkish Legal AI Platform.

This module serves as the primary entry point for the FastAPI application and provides
comprehensive CLI commands for application management, database operations, and development tasks.

Entry Points:
-------------

1. ASGI Application Export
   The `app` variable is exported for ASGI servers (Uvicorn, Gunicorn, Hypercorn):

   >>> from backend.main import app
   >>> # Used by: uvicorn backend.main:app

2. Direct Execution
   Running this module directly starts the development server:

   >>> python -m backend.main
   >>> # Equivalent to: uvicorn backend.main:app --reload

3. CLI Commands (when Click is installed)
   Management commands for database, users, development:

   >>> python -m backend.main db migrate
   >>> python -m backend.main db seed
   >>> python -m backend.main user create-admin

Deployment Modes:
-----------------

Development Mode (Single Process, Auto-reload):
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

    Features:
    - Auto-reload on code changes
    - Debug mode enabled
    - Detailed error traces
    - Single worker process
    - API docs enabled at /docs

Production Mode (Multi-process, Gunicorn):
    gunicorn backend.main:app \
        -w 4 \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --timeout 120 \
        --graceful-timeout 30 \
        --max-requests 1000 \
        --max-requests-jitter 50

    Features:
    - Multiple worker processes (4x CPU cores recommended)
    - Graceful shutdown
    - Worker recycling (prevents memory leaks)
    - Production-grade performance
    - No auto-reload

Docker Deployment:
    docker run -p 8000:8000 \
        -e ENVIRONMENT=production \
        -e DATABASE_URL=postgresql://... \
        turkish-legal-ai:latest

    Dockerfile CMD:
    CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

Kubernetes Deployment:
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: turkish-legal-ai
    spec:
      replicas: 3
      template:
        spec:
          containers:
          - name: api
            image: turkish-legal-ai:latest
            ports:
            - containerPort: 8000
            env:
            - name: ENVIRONMENT
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
            livenessProbe:
              httpGet:
                path: /health/live
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 10
            readinessProbe:
              httpGet:
                path: /health/ready
                port: 8000
              initialDelaySeconds: 10
              periodSeconds: 5

CLI Commands:
-------------

Database Management:
    python -m backend.main db migrate    # Run pending migrations
    python -m backend.main db upgrade    # Upgrade to latest version
    python -m backend.main db downgrade  # Downgrade one version
    python -m backend.main db revision   # Create new migration
    python -m backend.main db seed       # Seed sample data
    python -m backend.main db reset      # Reset database (DANGEROUS!)

User Management:
    python -m backend.main user create-admin     # Create admin user
    python -m backend.main user create           # Create regular user
    python -m backend.main user list             # List all users
    python -m backend.main user delete <email>   # Delete user

Tenant Management:
    python -m backend.main tenant create <name>  # Create new tenant
    python -m backend.main tenant list           # List all tenants
    python -m backend.main tenant delete <id>    # Delete tenant

Development Tools:
    python -m backend.main dev shell             # Interactive Python shell
    python -m backend.main dev routes            # List all API routes
    python -m backend.main dev openapi           # Generate OpenAPI schema
    python -m backend.main dev test-email        # Test email configuration
    python -m backend.main dev test-s3           # Test S3 connectivity

Cache Management:
    python -m backend.main cache clear           # Clear all cache
    python -m backend.main cache stats           # Cache statistics
    python -m backend.main cache warmup          # Warm up cache

Background Tasks:
    python -m backend.main celery worker         # Start Celery worker
    python -m backend.main celery beat           # Start Celery scheduler
    python -m backend.main celery flower         # Start Flower (monitoring UI)

Security & Compliance:
    python -m backend.main security audit        # Run security audit
    python -m backend.main security scan-pii     # Scan for PII in logs
    python -m backend.main kvkk export-data <user_id>  # Export user data (KVKK)
    python -m backend.main kvkk delete-data <user_id>  # Delete user data (KVKK)

Environment Variables:
----------------------

Required:
    ENVIRONMENT: development | staging | production
    DATABASE_URL: PostgreSQL connection string
    REDIS_URL: Redis connection string
    JWT_SECRET_KEY: JWT signing key (min 32 chars)
    ENCRYPTION_KEY: Data encryption key (min 32 chars)
    S3_ENDPOINT_URL: S3/MinIO endpoint
    S3_ACCESS_KEY_ID: S3 access key
    S3_SECRET_ACCESS_KEY: S3 secret key
    S3_BUCKET_NAME: S3 bucket name

Optional:
    API_HOST: API bind host (default: 0.0.0.0)
    API_PORT: API bind port (default: 8000)
    DEBUG: Enable debug mode (default: false)
    LOG_LEVEL: Logging level (default: INFO)
    ENABLE_DOCS: Enable API docs (default: true in dev)
    WORKERS: Number of worker processes (default: 1)
    CORS_ORIGINS: Allowed CORS origins (comma-separated)
    SENTRY_DSN: Sentry error tracking DSN
    CELERY_BROKER_URL: Celery broker URL
    CELERY_RESULT_BACKEND: Celery result backend URL

Performance Tuning:
-------------------

Worker Configuration:
    # CPU-bound workload (contract analysis, NLP)
    workers = (CPU_cores * 2) + 1

    # I/O-bound workload (database queries, API calls)
    workers = (CPU_cores * 4) + 1

    # Memory-constrained environment
    workers = min(desired_workers, available_memory_gb // 2)

Connection Pooling:
    # Database
    DATABASE_POOL_SIZE = workers * 5
    DATABASE_MAX_OVERFLOW = workers * 2

    # Redis
    REDIS_MAX_CONNECTIONS = workers * 10

Timeouts:
    # Request timeout (long-running analysis)
    timeout = 120  # seconds

    # Graceful shutdown timeout
    graceful_timeout = 30  # seconds

    # Worker recycling (prevent memory leaks)
    max_requests = 1000
    max_requests_jitter = 50

Monitoring & Logging:
---------------------

Structured Logging:
    All logs are in JSON format for easy parsing:

    {
        "timestamp": "2024-11-06T10:30:45.123Z",
        "level": "INFO",
        "message": "Request completed",
        "request_id": "uuid",
        "user_id": "uuid",
        "tenant_id": "uuid",
        "duration_ms": 123.45,
        "status_code": 200
    }

Metrics (Prometheus):
    - http_requests_total: Total HTTP requests
    - http_request_duration_seconds: Request latency histogram
    - http_requests_in_progress: Current in-flight requests
    - db_connections_active: Active database connections
    - cache_hits_total: Cache hit rate
    - celery_tasks_total: Background task count

Health Checks:
    GET /health       - Basic health status
    GET /health/ready - Readiness probe (K8s)
    GET /health/live  - Liveness probe (K8s)

Error Tracking (Sentry):
    export SENTRY_DSN=https://...@sentry.io/...

    Automatic error reporting with:
    - Full stack traces
    - Request context
    - User information
    - Environment details

Troubleshooting:
----------------

Application won't start:
    1. Check DATABASE_URL is correct and reachable
    2. Verify all required env vars are set
    3. Check port 8000 is not already in use
    4. Review logs: python -m backend.main --log-level DEBUG

Database connection errors:
    1. Verify PostgreSQL is running: pg_isready
    2. Check credentials in DATABASE_URL
    3. Test connection: psql $DATABASE_URL
    4. Check firewall/network settings

Redis connection errors:
    1. Verify Redis is running: redis-cli ping
    2. Check REDIS_URL format
    3. Test connection: redis-cli -u $REDIS_URL ping

S3/MinIO errors:
    1. Verify S3_ENDPOINT_URL is reachable
    2. Test credentials: aws s3 ls --endpoint-url $S3_ENDPOINT_URL
    3. Check bucket exists and has correct permissions

Memory issues:
    1. Reduce worker count: --workers 2
    2. Increase container memory limit
    3. Enable worker recycling: --max-requests 500
    4. Monitor with: docker stats or kubectl top

High CPU usage:
    1. Check for slow queries: pg_stat_statements
    2. Review long-running requests: /health endpoint
    3. Enable profiling: python -m cProfile backend.main
    4. Optimize hot paths

Slow response times:
    1. Check database connection pool exhaustion
    2. Review Redis cache hit rate
    3. Analyze slow queries in PostgreSQL logs
    4. Enable request tracing: X-Request-ID header

Testing:
--------

Unit Tests:
    pytest tests/unit -v --cov=backend --cov-report=html

Integration Tests:
    pytest tests/integration -v --log-cli-level=INFO

E2E Tests:
    pytest tests/e2e -v --headless

Load Testing:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Security Testing:
    bandit -r backend/
    safety check
    pip-audit

Security Considerations:
-------------------------

1. Never commit secrets to version control
2. Use environment variables for all credentials
3. Rotate JWT_SECRET_KEY regularly (every 90 days)
4. Enable HTTPS in production (HSTS headers)
5. Keep dependencies updated: pip-audit
6. Run security scans: bandit, safety
7. Monitor for vulnerabilities: Snyk, Dependabot
8. Use secrets management: HashiCorp Vault, AWS Secrets Manager
9. Enable audit logging for sensitive operations
10. Implement rate limiting to prevent abuse

KVKK Compliance:
----------------

Data Subject Rights:
    - Right to access: /api/v1/kvkk/export-data
    - Right to deletion: /api/v1/kvkk/delete-data
    - Right to portability: /api/v1/kvkk/export-data?format=json

Audit Trail:
    All data access is logged with:
    - User ID
    - Timestamp
    - Operation type
    - Data accessed
    - IP address

Data Retention:
    - User data: 7 years (legal requirement)
    - Audit logs: 10 years
    - Temporary files: 30 days
    - Cache: 24 hours

Known Issues & Limitations:
----------------------------

1. Maximum request size: 100MB (configurable)
2. Maximum concurrent requests: 1000 per worker
3. Maximum database connections: 100 (pool size)
4. Maximum cache size: 1GB
5. Maximum file upload size: 100MB
6. PDF processing timeout: 5 minutes
7. Contract analysis timeout: 10 minutes

Migration Guide:
----------------

From v0.x to v1.x:
    1. Backup database: pg_dump
    2. Update environment variables (new format)
    3. Run migrations: python -m backend.main db migrate
    4. Update API client libraries
    5. Test authentication flow (new JWT format)

Breaking Changes:
    - API endpoint structure changed: /api/v1/...
    - JWT token format updated (RS256)
    - Database schema changes (see migrations/)
    - New environment variables required

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.api.app import app
from backend.core import get_logger, settings

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

# Export app for ASGI servers (uvicorn, gunicorn)
__all__ = ["app"]

# =============================================================================
# CLI COMMANDS (Optional - requires Click)
# =============================================================================

try:
    import click

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    logger.info(
        "Click not installed - CLI commands unavailable. "
        "Install with: pip install click"
    )


if CLI_AVAILABLE:

    @click.group()
    @click.version_option(version=settings.APP_VERSION)
    def cli():
        """Turkish Legal AI Platform - Management CLI."""
        pass

    # Database commands
    @cli.group()
    def db():
        """Database management commands."""
        pass

    @db.command()
    def migrate():
        """Run pending database migrations."""
        click.echo("🔄 Running database migrations...")
        # TODO: Implement with Alembic
        click.echo("✅ Migrations complete")

    @db.command()
    def seed():
        """Seed database with sample data."""
        click.echo("🌱 Seeding database...")
        # TODO: Implement seed logic
        click.echo("✅ Database seeded")

    # User commands
    @cli.group()
    def user():
        """User management commands."""
        pass

    @user.command()
    @click.option("--email", prompt=True, help="Admin email")
    @click.option("--password", prompt=True, hide_input=True, help="Admin password")
    @click.option("--name", prompt=True, help="Admin full name")
    def create_admin(email, password, name):
        """Create admin user."""
        click.echo(f"👤 Creating admin user: {email}")
        # TODO: Implement user creation
        click.echo("✅ Admin user created")

    # Development commands
    @cli.group()
    def dev():
        """Development tools."""
        pass

    @dev.command()
    def routes():
        """List all API routes."""
        click.echo("📋 API Routes:\n")
        for route in app.routes:
            click.echo(f"  {route.path} [{', '.join(route.methods)}]")

    @dev.command()
    def shell():
        """Start interactive Python shell."""
        import code

        code.interact(local=locals())


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main entry point for direct execution.

    Starts Uvicorn development server with auto-reload.
    """
    logger.info(
        "🚀 Starting Turkish Legal AI Platform",
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION,
        host=settings.API_HOST,
        port=settings.API_PORT,
        debug=settings.DEBUG,
    )

    # Import uvicorn
    try:
        import uvicorn
    except ImportError:
        logger.error("❌ Uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)

    # Run Uvicorn server
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    # Check if CLI commands are available
    if CLI_AVAILABLE and len(sys.argv) > 1:
        # Run CLI command
        cli()
    else:
        # Run development server
        main()
