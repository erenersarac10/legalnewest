"""
Pagination Dependencies for Turkish Legal AI Platform.

Provides FastAPI dependency injection for pagination.

Features:
- Page-based pagination
- Offset/limit pagination
- Configurable page size limits
- Cursor-based pagination support

Author: Turkish Legal AI Team
License: Proprietary
"""

from typing import Optional

from fastapi import Query
from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """
    Pagination parameters for list endpoints.

    Supports both page-based and offset-based pagination.
    """

    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed)",
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page (max 100)",
    )

    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Get limit (alias for page_size)."""
        return self.page_size


async def get_pagination(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    """
    FastAPI dependency for pagination parameters.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page (max 100)

    Returns:
        PaginationParams instance

    Example:
        @app.get("/users")
        async def list_users(
            pagination: PaginationParams = Depends(get_pagination)
        ):
            users = await db.query(User).offset(pagination.offset).limit(pagination.limit).all()
            return {
                "items": users,
                "page": pagination.page,
                "page_size": pagination.page_size,
            }
    """
    return PaginationParams(page=page, page_size=page_size)


class OffsetPaginationParams(BaseModel):
    """
    Offset-based pagination parameters.

    Direct offset/limit control without page abstraction.
    """

    offset: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip",
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum items to return (max 100)",
    )


async def get_offset_pagination(
    offset: int = Query(0, ge=0, description="Items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Items to return"),
) -> OffsetPaginationParams:
    """
    FastAPI dependency for offset-based pagination.

    Args:
        offset: Number of items to skip
        limit: Maximum items to return

    Returns:
        OffsetPaginationParams instance

    Example:
        @app.get("/documents")
        async def list_documents(
            pagination: OffsetPaginationParams = Depends(get_offset_pagination)
        ):
            docs = await db.query(Document).offset(pagination.offset).limit(pagination.limit).all()
            return {"items": docs}
    """
    return OffsetPaginationParams(offset=offset, limit=limit)


class CursorPaginationParams(BaseModel):
    """
    Cursor-based pagination parameters.

    More efficient for large datasets.
    """

    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor (opaque token)",
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items to return (max 100)",
    )


async def get_cursor_pagination(
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(20, ge=1, le=100, description="Items to return"),
) -> CursorPaginationParams:
    """
    FastAPI dependency for cursor-based pagination.

    Args:
        cursor: Pagination cursor (opaque token)
        limit: Items to return

    Returns:
        CursorPaginationParams instance

    Example:
        @app.get("/feed")
        async def get_feed(
            pagination: CursorPaginationParams = Depends(get_cursor_pagination)
        ):
            items, next_cursor = await get_feed_items(
                cursor=pagination.cursor,
                limit=pagination.limit
            )
            return {
                "items": items,
                "next_cursor": next_cursor,
            }
    """
    return CursorPaginationParams(cursor=cursor, limit=limit)


__all__ = [
    "PaginationParams",
    "OffsetPaginationParams",
    "CursorPaginationParams",
    "get_pagination",
    "get_offset_pagination",
    "get_cursor_pagination",
]
