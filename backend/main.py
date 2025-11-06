"""
Main Entry Point for Turkish Legal AI Platform.

This module serves as the entry point for running the FastAPI application.

Usage:
    Development:
        uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

    Production:
        gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker \
            --bind 0.0.0.0:8000 --timeout 120 --graceful-timeout 30

    With environment:
        export ENVIRONMENT=production
        uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

Author: Turkish Legal AI Team
License: Proprietary
"""

from backend.api.app import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    from backend.core import settings

    # Run with uvicorn when executed directly
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
