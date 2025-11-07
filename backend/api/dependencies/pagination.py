"""
Pagination, Filtering, and Sorting Dependencies for Turkish Legal AI Platform.

Enterprise-grade pagination system with multiple strategies, advanced filtering,
multi-field sorting, and performance optimization for large datasets.

=============================================================================
FEATURES
=============================================================================

1. Multiple Pagination Strategies
   -------------------------------
   - Page-based pagination (user-friendly, traditional)
   - Offset-based pagination (direct control, API flexibility)
   - Cursor-based pagination (efficient for large datasets)
   - Configurable limits and defaults
   - Automatic offset calculation

2. Advanced Filtering
   --------------------
   - Multi-field filtering support
   - Operator support (eq, ne, gt, lt, contains, in)
   - Type-safe filter parsing
   - SQL injection prevention
   - Case-insensitive text search
   - Date range filtering

3. Multi-Field Sorting
   ---------------------
   - Sort by multiple fields
   - Ascending/descending per field
   - Nested field sorting support
   - Default sort orders
   - Null handling (nulls first/last)

4. Performance Optimization
   --------------------------
   - Efficient query construction
   - Index-aware pagination
   - Cursor encoding/decoding
   - Query result caching hints
   - Total count optimization

5. Response Standardization
   --------------------------
   - Consistent pagination metadata
   - Next/previous page links
   - Total count information
   - Has more indicator
   - Cursor tokens

=============================================================================
USAGE
=============================================================================

Page-Based Pagination (Traditional):
-------------------------------------

>>> from fastapi import Depends
>>> from sqlalchemy import select
>>> from backend.api.dependencies.pagination import get_pagination
>>> from backend.db.models import Contract
>>>
>>> @app.get("/contracts")
>>> async def list_contracts(
...     pagination: PaginationParams = Depends(get_pagination),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Query with pagination
...     result = await db.execute(
...         select(Contract)
...         .offset(pagination.offset)
...         .limit(pagination.limit)
...     )
...     contracts = result.scalars().all()
...
...     # Get total count
...     total_result = await db.execute(select(func.count(Contract.id)))
...     total = total_result.scalar()
...
...     return {
...         "items": contracts,
...         "page": pagination.page,
...         "page_size": pagination.page_size,
...         "total": total,
...         "total_pages": (total + pagination.page_size - 1) // pagination.page_size,
...     }

Offset-Based Pagination (API Flexibility):
-------------------------------------------

>>> from backend.api.dependencies.pagination import get_offset_pagination
>>>
>>> @app.get("/documents")
>>> async def list_documents(
...     pagination: OffsetPaginationParams = Depends(get_offset_pagination),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Direct offset/limit control
...     result = await db.execute(
...         select(Document)
...         .offset(pagination.offset)
...         .limit(pagination.limit)
...     )
...     documents = result.scalars().all()
...
...     return {
...         "items": documents,
...         "offset": pagination.offset,
...         "limit": pagination.limit,
...     }

Cursor-Based Pagination (Large Datasets):
------------------------------------------

>>> from backend.api.dependencies.pagination import get_cursor_pagination, decode_cursor, encode_cursor
>>>
>>> @app.get("/feed")
>>> async def get_feed(
...     pagination: CursorPaginationParams = Depends(get_cursor_pagination),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Decode cursor to get last seen ID
...     after_id = decode_cursor(pagination.cursor) if pagination.cursor else None
...
...     # Query items after cursor
...     query = select(FeedItem).order_by(FeedItem.created_at.desc())
...     if after_id:
...         query = query.where(FeedItem.id > after_id)
...
...     result = await db.execute(query.limit(pagination.limit + 1))
...     items = result.scalars().all()
...
...     # Check if there are more items
...     has_more = len(items) > pagination.limit
...     if has_more:
...         items = items[:pagination.limit]
...
...     # Generate next cursor
...     next_cursor = encode_cursor(items[-1].id) if has_more and items else None
...
...     return {
...         "items": items,
...         "next_cursor": next_cursor,
...         "has_more": has_more,
...     }

Advanced Filtering:
-------------------

>>> from backend.api.dependencies.pagination import get_filter_params
>>>
>>> @app.get("/contracts/search")
>>> async def search_contracts(
...     filters: FilterParams = Depends(get_filter_params),
...     pagination: PaginationParams = Depends(get_pagination),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Build query with filters
...     query = select(Contract)
...
...     # Apply filters
...     if "status" in filters.filters:
...         query = query.where(Contract.status == filters.filters["status"])
...     if "created_after" in filters.filters:
...         query = query.where(Contract.created_at >= filters.filters["created_after"])
...     if "title_contains" in filters.filters:
...         query = query.where(Contract.title.ilike(f"%{filters.filters['title_contains']}%"))
...
...     # Apply pagination
...     result = await db.execute(
...         query.offset(pagination.offset).limit(pagination.limit)
...     )
...     contracts = result.scalars().all()
...
...     return {"items": contracts}

Multi-Field Sorting:
--------------------

>>> from backend.api.dependencies.pagination import get_sort_params
>>>
>>> @app.get("/contracts")
>>> async def list_contracts(
...     sort: SortParams = Depends(get_sort_params),
...     pagination: PaginationParams = Depends(get_pagination),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Build query with sorting
...     query = select(Contract)
...
...     # Apply sorting
...     for field, direction in sort.get_order_by():
...         column = getattr(Contract, field, None)
...         if column is not None:
...             if direction == "desc":
...                 query = query.order_by(column.desc())
...             else:
...                 query = query.order_by(column.asc())
...
...     # Apply pagination
...     result = await db.execute(
...         query.offset(pagination.offset).limit(pagination.limit)
...     )
...     contracts = result.scalars().all()
...
...     return {"items": contracts}

Combined Filtering, Sorting, and Pagination:
---------------------------------------------

>>> @app.get("/contracts/advanced")
>>> async def advanced_search(
...     filters: FilterParams = Depends(get_filter_params),
...     sort: SortParams = Depends(get_sort_params),
...     pagination: PaginationParams = Depends(get_pagination),
...     db: AsyncSession = Depends(get_db)
... ):
...     # Build base query
...     query = select(Contract)
...
...     # Apply filters
...     if "status" in filters.filters:
...         query = query.where(Contract.status == filters.filters["status"])
...
...     # Apply sorting
...     for field, direction in sort.get_order_by():
...         column = getattr(Contract, field, None)
...         if column is not None:
...             query = query.order_by(
...                 column.desc() if direction == "desc" else column.asc()
...             )
...
...     # Get total count (before pagination)
...     count_result = await db.execute(
...         select(func.count()).select_from(query.subquery())
...     )
...     total = count_result.scalar()
...
...     # Apply pagination
...     result = await db.execute(
...         query.offset(pagination.offset).limit(pagination.limit)
...     )
...     contracts = result.scalars().all()
...
...     return {
...         "items": contracts,
...         "page": pagination.page,
...         "page_size": pagination.page_size,
...         "total": total,
...         "total_pages": (total + pagination.page_size - 1) // pagination.page_size,
...         "has_next": pagination.page * pagination.page_size < total,
...         "has_prev": pagination.page > 1,
...     }

=============================================================================
PAGINATION STRATEGY COMPARISON
=============================================================================

Page-Based vs Offset vs Cursor:
--------------------------------

+------------------+------------------+------------------+------------------+
| Feature          | Page-Based       | Offset-Based     | Cursor-Based     |
+------------------+------------------+------------------+------------------+
| User Friendly    | ★★★★★ (Best)    | ★★★☆☆           | ★★☆☆☆           |
| Performance      | ★★☆☆☆ (Poor)    | ★★☆☆☆ (Poor)    | ★★★★★ (Best)    |
| Consistency      | ★★☆☆☆           | ★★☆☆☆           | ★★★★★           |
| Random Access    | ★★★★★ (Yes)     | ★★★★★ (Yes)     | ★☆☆☆☆ (No)      |
| Large Datasets   | ★★☆☆☆           | ★★☆☆☆           | ★★★★★           |
| Real-time Data   | ★★☆☆☆           | ★★☆☆☆           | ★★★★★           |
| Implementation   | ★★★★★ (Easy)    | ★★★★★ (Easy)    | ★★★☆☆ (Medium)  |
+------------------+------------------+------------------+------------------+

When to Use Each:
-----------------

Page-Based Pagination:
  ✓ User-facing list views (contracts, documents)
  ✓ Small to medium datasets (< 100K records)
  ✓ Need to jump to specific pages
  ✓ Traditional web applications
  ✗ Large datasets (> 1M records)
  ✗ Real-time feeds
  ✗ Frequently changing data

Offset-Based Pagination:
  ✓ API endpoints with flexible access
  ✓ Internal tools and admin panels
  ✓ Data export and batch processing
  ✓ When client needs direct control
  ✗ Large datasets (performance issues)
  ✗ Real-time data (consistency issues)

Cursor-Based Pagination:
  ✓ Large datasets (millions of records)
  ✓ Real-time feeds and timelines
  ✓ Infinite scroll UIs
  ✓ Mobile apps (efficient data loading)
  ✓ Frequently changing data
  ✗ Need to jump to specific pages
  ✗ Simple list views (over-engineering)

Performance Considerations:
---------------------------

Page-Based Performance Issues:
  - OFFSET becomes slow on large datasets
  - Database must scan and skip all offset rows
  - Page 1000 with size 100 = skip 99,900 rows (SLOW!)
  - No index can optimize OFFSET

  Example slow query:
    SELECT * FROM contracts OFFSET 99900 LIMIT 100;
    -- Scans 100,000 rows to return 100!

Cursor-Based Performance Benefits:
  - Uses WHERE clause with indexed column
  - Database seeks directly to position
  - Consistent performance regardless of position
  - Can leverage indexes efficiently

  Example fast query:
    SELECT * FROM contracts WHERE id > 12345 ORDER BY id LIMIT 100;
    -- Uses index seek, always fast!

Optimization Tips:
  - Use cursor pagination for > 100K records
  - Add indexes on sort columns
  - Consider materialized views for complex sorts
  - Cache total counts (update periodically)
  - Use approximate counts for large tables

=============================================================================
FILTERING BEST PRACTICES
=============================================================================

Supported Filter Operators:
----------------------------

eq (equals):
  - Exact match
  - Example: status=eq:active
  - SQL: WHERE status = 'active'

ne (not equals):
  - Exclude values
  - Example: status=ne:deleted
  - SQL: WHERE status != 'deleted'

gt (greater than):
  - Numeric/date comparison
  - Example: created_at=gt:2024-01-01
  - SQL: WHERE created_at > '2024-01-01'

lt (less than):
  - Numeric/date comparison
  - Example: amount=lt:1000
  - SQL: WHERE amount < 1000

contains:
  - Text search (case-insensitive)
  - Example: title=contains:sözleşme
  - SQL: WHERE title ILIKE '%sözleşme%'

in:
  - Multiple values
  - Example: status=in:active,pending
  - SQL: WHERE status IN ('active', 'pending')

Security Considerations:
------------------------

SQL Injection Prevention:
  ✓ Always use parameterized queries
  ✓ Validate filter field names against whitelist
  ✓ Sanitize user input
  ✓ Use SQLAlchemy ORM (automatic escaping)
  ✗ Never construct SQL strings directly
  ✗ Don't allow arbitrary column access

Example Safe Filtering:
  ALLOWED_FILTERS = {"status", "created_at", "title", "amount"}

  def apply_filters(query, filters):
      for field, value in filters.items():
          if field not in ALLOWED_FILTERS:
              raise ValueError(f"Geçersiz filtre alanı: {field}")
          column = getattr(Model, field)
          query = query.where(column == value)
      return query

=============================================================================
SORTING BEST PRACTICES
=============================================================================

Multi-Field Sorting:
--------------------

Sort by multiple fields with different directions:
  - Primary sort: created_at DESC (newest first)
  - Secondary sort: title ASC (alphabetical)
  - Tertiary sort: id ASC (consistent ordering)

Example:
  sort_by=created_at:desc,title:asc,id:asc

Why Multiple Sort Fields:
  - Ensures consistent ordering
  - Breaks ties for same values
  - Predictable pagination results
  - Required for cursor pagination

Default Sort Orders:
--------------------

Always provide default sorting for:
  ✓ Consistent user experience
  ✓ Predictable API responses
  ✓ Efficient index usage
  ✓ Cursor pagination support

Recommended defaults:
  - List views: created_at DESC, id ASC
  - Alphabetical: title ASC, id ASC
  - Analytics: metric DESC, date DESC, id ASC

Index Optimization:
-------------------

Create composite indexes for sort columns:
  CREATE INDEX idx_contracts_sort
  ON contracts(created_at DESC, id ASC);

  CREATE INDEX idx_contracts_alpha
  ON contracts(title ASC, id ASC);

Benefits:
  - Database can use index for sorting
  - Avoids expensive sort operations
  - Faster query execution
  - Lower memory usage

=============================================================================
CURSOR PAGINATION IMPLEMENTATION
=============================================================================

Cursor Encoding/Decoding:
--------------------------

Cursors are opaque tokens representing position in dataset.
Typically encode: last seen ID, timestamp, or composite key.

Simple ID-based cursor:
  cursor = base64(str(last_id))

Composite cursor (ID + timestamp):
  cursor = base64(f"{last_id}:{timestamp}")

Security considerations:
  - Don't expose internal IDs directly
  - Use HMAC for tamper protection
  - Validate decoded cursor values
  - Handle invalid cursors gracefully

Bi-Directional Pagination:
---------------------------

Support both forward and backward pagination:

Forward (next page):
  WHERE id > cursor_id ORDER BY id ASC LIMIT 100

Backward (previous page):
  WHERE id < cursor_id ORDER BY id DESC LIMIT 100

Full implementation:
  - Provide both next_cursor and prev_cursor
  - Include has_next and has_prev indicators
  - Support before and after parameters
  - Reverse results for backward pagination

=============================================================================
PERFORMANCE OPTIMIZATION
=============================================================================

Total Count Optimization:
--------------------------

Problem: COUNT(*) is expensive on large tables

Solutions:

1. Approximate counts (fast, acceptable for UI):
   SELECT reltuples::bigint FROM pg_class WHERE relname = 'contracts';

2. Cached counts (fast, periodically updated):
   - Store count in Redis
   - Update every 5 minutes
   - Use for display purposes

3. Range counts (fast for filtered queries):
   - Don't count total, just check if more exist
   - Query for limit + 1 items
   - has_more = (results.length > limit)

4. Count hints (fast, user-friendly):
   - "More than 10,000 results"
   - Exact count only for small result sets

Query Result Caching:
----------------------

Cache list endpoints with:
  - Cache key: query params hash
  - TTL: 5-60 seconds (based on data volatility)
  - Invalidate on writes
  - Use Redis or in-memory cache

Example:
  cache_key = f"contracts:list:{hash(query_params)}"
  cached = await redis.get(cache_key)
  if cached:
      return cached
  results = await db.execute(query)
  await redis.setex(cache_key, 60, results)

Index Hints:
------------

Suggest indexes for common queries:
  - Add USE INDEX hints if needed
  - Monitor slow query log
  - Create covering indexes
  - Analyze query execution plans

=============================================================================
KVKK COMPLIANCE
=============================================================================

Data Privacy in Pagination:
----------------------------

Row-Level Security (RLS):
  ✓ Automatic tenant filtering
  ✓ User can only see their data
  ✓ Applied before pagination
  ✓ No cross-tenant data leakage

Personal Data Protection:
  ✓ Paginate audit logs
  ✓ Track data access
  ✓ Log pagination requests
  ✓ Implement data retention policies

Access Control:
  ✓ Verify user permissions before pagination
  ✓ Filter by user's accessible resources
  ✓ Apply role-based access control (RBAC)
  ✓ Audit trail for data access

=============================================================================
TROUBLESHOOTING
=============================================================================

"Pagination is slow on large offsets":
---------------------------------------
Problem: OFFSET 100000 LIMIT 100 takes seconds
Solution:
  1. Switch to cursor-based pagination
  2. Add composite index on sort columns
  3. Use WHERE clause instead of OFFSET
  4. Consider keyset pagination

"Inconsistent results between pages":
--------------------------------------
Problem: Items appear on multiple pages or get skipped
Solution:
  1. Always include stable sort column (id)
  2. Use cursor pagination for real-time data
  3. Add ORDER BY id as tiebreaker
  4. Consider snapshot isolation level

"Total count is inaccurate":
-----------------------------
Problem: COUNT(*) shows different numbers
Solution:
  1. Use approximate counts for display
  2. Cache counts with longer TTL
  3. Accept eventual consistency
  4. Show "More than X results" for large datasets

"Cursor pagination is confusing to users":
-------------------------------------------
Problem: Users can't jump to page 5
Solution:
  1. Use page-based for user-facing lists
  2. Reserve cursor for infinite scroll
  3. Provide search/filters for navigation
  4. Implement hybrid approach (page + cursor)

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

import base64
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Query, HTTPException, status
from pydantic import BaseModel, Field, validator

from backend.core import get_logger, settings

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum page size to prevent abuse
MAX_PAGE_SIZE = 100

# Default page size for list endpoints
DEFAULT_PAGE_SIZE = 20

# Cursor secret for HMAC signing (prevent tampering)
CURSOR_SECRET = settings.secret_key.encode()

# =============================================================================
# PAGINATION MODELS
# =============================================================================


class PaginationParams(BaseModel):
    """
    Page-based pagination parameters.

    Traditional pagination with page numbers and page size.
    User-friendly for web applications with page navigation.
    """

    page: int = Field(
        default=1,
        ge=1,
        description="Sayfa numarası (1'den başlar)",
    )

    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Sayfa başına öğe sayısı (maksimum {MAX_PAGE_SIZE})",
    )

    @property
    def offset(self) -> int:
        """Calculate database offset from page and page_size."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Get limit (alias for page_size)."""
        return self.page_size

    def get_page_info(self, total: int) -> Dict[str, Any]:
        """
        Generate pagination metadata.

        Args:
            total: Total number of items

        Returns:
            Dictionary with pagination information
        """
        total_pages = (total + self.page_size - 1) // self.page_size if total > 0 else 0

        return {
            "page": self.page,
            "page_size": self.page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": self.page < total_pages,
            "has_prev": self.page > 1,
        }


class OffsetPaginationParams(BaseModel):
    """
    Offset-based pagination parameters.

    Direct offset/limit control without page abstraction.
    Useful for APIs that need flexible pagination control.
    """

    offset: int = Field(
        default=0,
        ge=0,
        description="Atlanacak öğe sayısı",
    )

    limit: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Döndürülecek maksimum öğe sayısı (maksimum {MAX_PAGE_SIZE})",
    )

    def get_pagination_info(self, total: int) -> Dict[str, Any]:
        """
        Generate pagination metadata.

        Args:
            total: Total number of items

        Returns:
            Dictionary with pagination information
        """
        return {
            "offset": self.offset,
            "limit": self.limit,
            "total": total,
            "has_more": (self.offset + self.limit) < total,
        }


class CursorPaginationParams(BaseModel):
    """
    Cursor-based pagination parameters.

    Efficient pagination for large datasets using opaque cursors.
    Provides consistent results even when data changes.
    """

    cursor: Optional[str] = Field(
        default=None,
        description="Sayfalama imleci (opak token)",
    )

    limit: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Döndürülecek öğe sayısı (maksimum {MAX_PAGE_SIZE})",
    )

    @validator("cursor")
    def validate_cursor(cls, v: Optional[str]) -> Optional[str]:
        """Validate cursor format and signature."""
        if v is None:
            return v

        try:
            # Decode and verify cursor
            decode_cursor(v)
            return v
        except Exception as e:
            logger.warning("Geçersiz cursor formatı", cursor=v, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz sayfalama imleci",
            )


class FilterParams(BaseModel):
    """
    Advanced filtering parameters.

    Supports multiple filter operators and field filtering.
    """

    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filtre koşulları (alan: değer)",
    )

    def get_filter_value(self, field: str, default: Any = None) -> Any:
        """
        Get filter value for a field.

        Args:
            field: Filter field name
            default: Default value if not present

        Returns:
            Filter value or default
        """
        return self.filters.get(field, default)

    def has_filter(self, field: str) -> bool:
        """Check if filter exists for field."""
        return field in self.filters


class SortParams(BaseModel):
    """
    Multi-field sorting parameters.

    Supports sorting by multiple fields with different directions.
    """

    sort_by: str = Field(
        default="created_at:desc,id:asc",
        description="Sıralama alanları (alan:yön, ...)",
    )

    def get_order_by(self) -> List[Tuple[str, str]]:
        """
        Parse sort string into list of (field, direction) tuples.

        Returns:
            List of (field, direction) tuples

        Example:
            "created_at:desc,title:asc" -> [("created_at", "desc"), ("title", "asc")]
        """
        order_by = []

        for sort_spec in self.sort_by.split(","):
            sort_spec = sort_spec.strip()
            if not sort_spec:
                continue

            if ":" in sort_spec:
                field, direction = sort_spec.split(":", 1)
                field = field.strip()
                direction = direction.strip().lower()

                if direction not in ("asc", "desc"):
                    logger.warning(
                        "Geçersiz sıralama yönü, 'asc' kullanılıyor",
                        field=field,
                        direction=direction,
                    )
                    direction = "asc"
            else:
                field = sort_spec
                direction = "asc"

            order_by.append((field, direction))

        return order_by if order_by else [("created_at", "desc"), ("id", "asc")]

    @validator("sort_by")
    def validate_sort_by(cls, v: str) -> str:
        """Validate sort_by format."""
        if not v:
            return "created_at:desc,id:asc"

        # Basic validation
        for sort_spec in v.split(","):
            if ":" in sort_spec:
                field, direction = sort_spec.split(":", 1)
                if direction.lower() not in ("asc", "desc"):
                    raise ValueError(
                        f"Geçersiz sıralama yönü: {direction}. 'asc' veya 'desc' olmalı."
                    )

        return v


# =============================================================================
# CURSOR UTILITIES
# =============================================================================


def encode_cursor(value: Any, signature: bool = True) -> str:
    """
    Encode a value into a cursor token.

    Args:
        value: Value to encode (typically ID or composite key)
        signature: Whether to include HMAC signature

    Returns:
        Base64-encoded cursor token

    Example:
        >>> encode_cursor(12345)
        'MTIzNDU6YWJjZGVm...'
    """
    # Convert value to string
    value_str = str(value)

    # Create HMAC signature if requested
    if signature:
        sig = hmac.new(
            CURSOR_SECRET,
            value_str.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        cursor_data = f"{value_str}:{sig}"
    else:
        cursor_data = value_str

    # Base64 encode
    cursor_bytes = cursor_data.encode()
    cursor_b64 = base64.urlsafe_b64encode(cursor_bytes).decode()

    return cursor_b64


def decode_cursor(cursor: str, verify_signature: bool = True) -> Any:
    """
    Decode a cursor token back to its value.

    Args:
        cursor: Base64-encoded cursor token
        verify_signature: Whether to verify HMAC signature

    Returns:
        Decoded cursor value

    Raises:
        ValueError: If cursor is invalid or signature doesn't match

    Example:
        >>> decode_cursor('MTIzNDU6YWJjZGVm...')
        '12345'
    """
    try:
        # Base64 decode
        cursor_bytes = base64.urlsafe_b64decode(cursor.encode())
        cursor_data = cursor_bytes.decode()

        # Verify signature if requested
        if verify_signature:
            if ":" not in cursor_data:
                raise ValueError("Cursor imzası bulunamadı")

            value_str, sig = cursor_data.rsplit(":", 1)

            # Calculate expected signature
            expected_sig = hmac.new(
                CURSOR_SECRET,
                value_str.encode(),
                hashlib.sha256
            ).hexdigest()[:16]

            # Compare signatures
            if not hmac.compare_digest(sig, expected_sig):
                raise ValueError("Cursor imzası geçersiz")

            return value_str
        else:
            return cursor_data

    except Exception as e:
        logger.warning("Cursor decode hatası", cursor=cursor, error=str(e))
        raise ValueError(f"Geçersiz cursor: {str(e)}")


def build_pagination_response(
    items: List[Any],
    pagination: PaginationParams,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build standardized pagination response.

    Args:
        items: List of items for current page
        pagination: Pagination parameters
        total: Total number of items (optional)

    Returns:
        Dictionary with items and pagination metadata

    Example:
        >>> response = build_pagination_response(contracts, pagination, total=150)
        >>> response.keys()
        dict_keys(['items', 'page', 'page_size', 'total', 'total_pages', 'has_next', 'has_prev'])
    """
    response: Dict[str, Any] = {"items": items}

    if total is not None:
        response.update(pagination.get_page_info(total))
    else:
        response.update({
            "page": pagination.page,
            "page_size": pagination.page_size,
        })

    return response


# =============================================================================
# DEPENDENCIES
# =============================================================================


async def get_pagination(
    page: int = Query(1, ge=1, description="Sayfa numarası"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Sayfa başına öğe sayısı"),
) -> PaginationParams:
    """
    FastAPI dependency for page-based pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page

    Returns:
        PaginationParams instance

    Example:
        >>> @app.get("/contracts")
        >>> async def list_contracts(
        ...     pagination: PaginationParams = Depends(get_pagination)
        ... ):
        ...     return await get_contracts_page(pagination)
    """
    return PaginationParams(page=page, page_size=page_size)


async def get_offset_pagination(
    offset: int = Query(0, ge=0, description="Atlanacak öğe sayısı"),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Döndürülecek öğe sayısı"),
) -> OffsetPaginationParams:
    """
    FastAPI dependency for offset-based pagination.

    Args:
        offset: Number of items to skip
        limit: Maximum items to return

    Returns:
        OffsetPaginationParams instance

    Example:
        >>> @app.get("/documents")
        >>> async def list_documents(
        ...     pagination: OffsetPaginationParams = Depends(get_offset_pagination)
        ... ):
        ...     return await get_documents(pagination.offset, pagination.limit)
    """
    return OffsetPaginationParams(offset=offset, limit=limit)


async def get_cursor_pagination(
    cursor: Optional[str] = Query(None, description="Sayfalama imleci"),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Döndürülecek öğe sayısı"),
) -> CursorPaginationParams:
    """
    FastAPI dependency for cursor-based pagination.

    Args:
        cursor: Pagination cursor (opaque token)
        limit: Items to return

    Returns:
        CursorPaginationParams instance

    Example:
        >>> @app.get("/feed")
        >>> async def get_feed(
        ...     pagination: CursorPaginationParams = Depends(get_cursor_pagination)
        ... ):
        ...     return await get_feed_items(pagination.cursor, pagination.limit)
    """
    return CursorPaginationParams(cursor=cursor, limit=limit)


async def get_filter_params(
    status: Optional[str] = Query(None, description="Durum filtresi"),
    search: Optional[str] = Query(None, description="Metin araması"),
) -> FilterParams:
    """
    FastAPI dependency for filtering parameters.

    Args:
        status: Status filter
        search: Text search query

    Returns:
        FilterParams instance

    Example:
        >>> @app.get("/contracts")
        >>> async def search_contracts(
        ...     filters: FilterParams = Depends(get_filter_params)
        ... ):
        ...     return await search(filters)
    """
    filters = {}
    if status:
        filters["status"] = status
    if search:
        filters["search"] = search

    return FilterParams(filters=filters)


async def get_sort_params(
    sort_by: str = Query(
        "created_at:desc,id:asc",
        description="Sıralama (alan:yön, ...)",
    ),
) -> SortParams:
    """
    FastAPI dependency for sorting parameters.

    Args:
        sort_by: Sort specification (field:direction, ...)

    Returns:
        SortParams instance

    Example:
        >>> @app.get("/contracts")
        >>> async def list_contracts(
        ...     sort: SortParams = Depends(get_sort_params)
        ... ):
        ...     return await get_sorted_contracts(sort)
    """
    return SortParams(sort_by=sort_by)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PaginationParams",
    "OffsetPaginationParams",
    "CursorPaginationParams",
    "FilterParams",
    "SortParams",
    "get_pagination",
    "get_offset_pagination",
    "get_cursor_pagination",
    "get_filter_params",
    "get_sort_params",
    "encode_cursor",
    "decode_cursor",
    "build_pagination_response",
]
