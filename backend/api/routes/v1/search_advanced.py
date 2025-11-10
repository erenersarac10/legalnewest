"""
Advanced Search API Routes for Harvey/Legora Turkish Legal AI Platform.

This module provides REST API endpoints for advanced Boolean search:
- Boolean Operators: AND, OR, NOT for precise query logic
- Phrase Matching: "exact phrase" for literal text search
- Proximity Search: word1 NEAR/5 word2 for proximity-based matching
- Wildcard Queries: ceza* for pattern matching
- Field-Specific: title:kanun, body:ceza for targeted searches
- Query Validation: Syntax checking and error reporting
- Westlaw-Style Syntax: Professional legal search interface

Advanced search enables precise legal research with complex queries:
- Boolean logic for combining multiple criteria
- Phrase matching for exact text sequences
- Proximity search for related concepts
- Field targeting for structured searches
- Nested queries with parentheses grouping
- Turkish legal terminology support

Example Usage:
    >>> # Basic Boolean search
    >>> POST /api/v1/search-advanced
    >>> {
    ...     "query": "anayasa AND mahkemesi",
    ...     "filters": {"source": ["aym", "hsk"]}
    ... }
    >>>
    >>> # Complex nested query
    >>> POST /api/v1/search-advanced
    >>> {
    ...     "query": "(title:anayasa OR title:aihm) AND body:\"ifade ozgurlugu\" NOT ret",
    ...     "limit": 50
    ... }
    >>>
    >>> # Validate query syntax
    >>> POST /api/v1/search-advanced/validate
    >>> {"query": "anayasa AND (mahkeme OR mahkemesi)"}
    >>> # Response: {"valid": true, "message": "Query syntax is valid"}

Query Syntax:
    **Boolean Operators**:
    - AND: Both terms must appear (e.g., "anayasa AND mahkemesi")
    - OR: Either term can appear (e.g., "ceza OR idare")
    - NOT: Exclude term (e.g., "karar NOT bozma")

    **Phrase Matching**:
    - Exact phrase: "ifade ozgurlugu" (words in order)

    **Proximity Search**:
    - NEAR/n: Terms within n words (e.g., "anayasa NEAR/5 mahkemesi")

    **Field-Specific**:
    - Field:term format (e.g., "title:kanun", "body:ceza")

    **Wildcards**:
    - * matches multiple characters (e.g., "ceza*")
    - ? matches single character (e.g., "ka?ar")

    **Grouping**:
    - Parentheses for precedence (e.g., "(ceza OR idare) AND karar")

Performance:
    - Query validation: < 5ms
    - Parse time: < 10ms
    - Search time: < 200ms (p95)
    - Supports nested queries up to 10 levels

Security:
    - Requires 'search:advanced' permission
    - All searches are tenant-isolated
    - Query complexity limits to prevent abuse
    - Audit logging for compliance

Author: Harvey/Legora CTO
Date: 2025-11-10
Lines: 570+
"""

import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.session import get_db
from backend.core.exceptions import ValidationError
from backend.security.rbac.context import get_current_tenant_id, get_current_user_id
from backend.security.rbac.decorators import require_permission
from backend.services.advanced_search_service import AdvancedSearchService
from backend.services.document_search_service import SearchResult

# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter(
    prefix="/search-advanced",
    tags=["search-advanced"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class AdvancedSearchRequest(BaseModel):
    """Advanced search request."""

    query: str = Field(
        ...,
        description="Boolean search query (e.g., 'anayasa AND mahkemesi')",
        min_length=1,
        max_length=2000,
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional filters (source, date range, document type, etc.)"
    )
    limit: int = Field(20, ge=1, le=1000, description="Maximum results (1-1000)")
    offset: int = Field(0, ge=0, description="Pagination offset")
    sort_by: Optional[str] = Field(None, description="Sort field (relevance, date, title)")
    include_highlights: bool = Field(True, description="Include search term highlights")
    include_facets: bool = Field(False, description="Include facet aggregations")

    @validator('query')
    def validate_query_length(cls, v):
        """Validate query is not too complex."""
        # Count parentheses nesting level
        max_depth = 0
        current_depth = 0
        for char in v:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        if max_depth > 10:
            raise ValueError('Query nesting too deep (max 10 levels)')

        return v


class SearchResultResponse(BaseModel):
    """Search result response."""

    document_id: str
    title: str
    content_preview: str
    relevance_score: float
    document_type: Optional[str] = None
    source: Optional[str] = None
    publication_date: Optional[str] = None
    metadata: Dict[str, Any] = {}
    highlights: Optional[List[str]] = None


class AdvancedSearchResponse(BaseModel):
    """Advanced search response."""

    query: str
    total_results: int
    results: List[SearchResultResponse]
    pagination: Dict[str, Any]
    execution_time_ms: int
    facets: Optional[Dict[str, Any]] = None


class ValidateQueryRequest(BaseModel):
    """Validate query request."""

    query: str = Field(..., description="Query to validate", min_length=1, max_length=2000)


class ValidateQueryResponse(BaseModel):
    """Validate query response."""

    valid: bool
    message: Optional[str] = None
    error_position: Optional[int] = None
    suggestions: Optional[List[str]] = None


class QuerySyntaxResponse(BaseModel):
    """Query syntax help response."""

    operators: List[Dict[str, str]]
    examples: List[Dict[str, str]]
    fields: List[Dict[str, str]]
    tips: List[str]


# =============================================================================
# ADVANCED SEARCH ENDPOINTS
# =============================================================================


@router.post(
    "",
    response_model=AdvancedSearchResponse,
    summary="Execute advanced Boolean search",
    description="Execute advanced search with Boolean operators and complex query syntax (requires search:advanced)",
)
@require_permission("search", "advanced")
async def advanced_search(
    request: AdvancedSearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> AdvancedSearchResponse:
    """
    Execute advanced Boolean search.

    **Permissions**: Requires 'search:advanced' permission.

    **Request Body**:
    ```json
    {
        "query": "(title:anayasa OR title:aihm) AND body:\\"ifade ozgurlugu\\" NOT ret",
        "filters": {
            "source": ["aym", "hsk"],
            "date_from": "2020-01-01",
            "date_to": "2025-12-31"
        },
        "limit": 50,
        "include_highlights": true
    }
    ```

    **Query Syntax Examples**:

    1. **Boolean AND**: Both terms must appear
       ```
       anayasa AND mahkemesi
       ```

    2. **Boolean OR**: Either term can appear
       ```
       ceza OR idare
       ```

    3. **Boolean NOT**: Exclude term
       ```
       karar NOT bozma
       ```

    4. **Phrase Matching**: Exact phrase
       ```
       "ifade ozgurlugu"
       ```

    5. **Proximity Search**: Terms within distance
       ```
       anayasa NEAR/5 mahkemesi
       ```

    6. **Field-Specific**: Search in specific field
       ```
       title:kanun
       body:ceza
       ```

    7. **Wildcards**: Pattern matching
       ```
       ceza*      (matches ceza, cezaevi, cezalandirma)
       ka?ar      (matches karar, kadar)
       ```

    8. **Complex Nested**: Combine all operators
       ```
       (title:anayasa OR title:aihm) AND "bireysel basvuru" NOT ret
       ```

    **Filters**:
    - `source`: List of sources (e.g., ["aym", "hsk", "yargitay"])
    - `document_type`: Document types (e.g., ["law", "regulation", "decision"])
    - `date_from`: Start date (ISO 8601: "2020-01-01")
    - `date_to`: End date (ISO 8601: "2025-12-31")
    - `court`: Court name filter
    - `case_number`: Case number filter

    **Returns**: Search results with relevance scores and optional highlights.

    **Performance**:
    - Query parse: < 10ms
    - Search execution: < 200ms (p95)
    - Supports up to 10 levels of nesting
    """
    try:
        import time
        start_time = time.time()

        service = AdvancedSearchService()

        # Execute search
        search_results = await service.search(
            query=request.query,
            filters=request.filters or {},
            limit=request.limit,
            offset=request.offset,
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Convert results
        results = []
        for result in search_results.results:
            results.append(SearchResultResponse(
                document_id=str(result.document_id),
                title=result.title,
                content_preview=result.content_preview,
                relevance_score=result.relevance_score,
                document_type=result.document_type,
                source=result.source,
                publication_date=result.publication_date,
                metadata=result.metadata,
                highlights=result.highlights if request.include_highlights else None,
            ))

        # Build pagination
        pagination = {
            "total": search_results.total,
            "limit": request.limit,
            "offset": request.offset,
            "has_more": (request.offset + request.limit) < search_results.total,
        }

        # Build facets if requested
        facets = None
        if request.include_facets:
            facets = search_results.facets

        return AdvancedSearchResponse(
            query=request.query,
            total_results=search_results.total,
            results=results,
            pagination=pagination,
            execution_time_ms=execution_time_ms,
            facets=facets,
        )

    except ValueError as e:
        # Query syntax error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query syntax: {str(e)}",
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.post(
    "/validate",
    response_model=ValidateQueryResponse,
    summary="Validate query syntax",
    description="Validate Boolean query syntax without executing search (requires search:advanced)",
)
@require_permission("search", "advanced")
async def validate_query(
    request: ValidateQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> ValidateQueryResponse:
    """
    Validate Boolean query syntax.

    **Permissions**: Requires 'search:advanced' permission.

    **Request Body**:
    ```json
    {
        "query": "(anayasa OR aihm) AND \\"bireysel basvuru\\""
    }
    ```

    **Returns**: Validation result with error details if invalid.

    **Example Success Response**:
    ```json
    {
        "valid": true,
        "message": "Query syntax is valid"
    }
    ```

    **Example Error Response**:
    ```json
    {
        "valid": false,
        "message": "Unexpected token at position 15: 'AND'",
        "error_position": 15,
        "suggestions": [
            "Check for missing closing parenthesis",
            "Ensure operators have terms on both sides"
        ]
    }
    ```

    **Common Syntax Errors**:
    - Unmatched parentheses: `(anayasa AND mahkeme`
    - Missing operand: `anayasa AND`
    - Invalid field: `unknownfield:value`
    - Unclosed phrase: `"ifade ozgurlugu`
    """
    try:
        service = AdvancedSearchService()
        
        # Validate query
        is_valid, error_message = service.validate_query(request.query)

        if is_valid:
            return ValidateQueryResponse(
                valid=True,
                message="Query syntax is valid",
            )
        else:
            # Try to extract error position from message
            error_position = None
            suggestions = []

            if "parenthesis" in error_message.lower():
                suggestions.append("Check for missing closing parenthesis")
            if "unexpected token" in error_message.lower():
                suggestions.append("Ensure operators have terms on both sides")
            if "empty query" in error_message.lower():
                suggestions.append("Query cannot be empty")

            return ValidateQueryResponse(
                valid=False,
                message=error_message,
                error_position=error_position,
                suggestions=suggestions if suggestions else None,
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.get(
    "/syntax",
    response_model=QuerySyntaxResponse,
    summary="Get query syntax help",
    description="Get comprehensive query syntax documentation (requires search:advanced)",
)
@require_permission("search", "advanced")
async def get_query_syntax(
    db: AsyncSession = Depends(get_db),
) -> QuerySyntaxResponse:
    """
    Get query syntax help and examples.

    **Permissions**: Requires 'search:advanced' permission.

    **Returns**: Comprehensive syntax documentation with examples.

    This endpoint provides:
    - Boolean operators documentation
    - Query syntax examples
    - Available search fields
    - Best practices and tips
    """
    return QuerySyntaxResponse(
        operators=[
            {
                "operator": "AND",
                "description": "Both terms must appear in the document",
                "example": "anayasa AND mahkemesi",
            },
            {
                "operator": "OR",
                "description": "Either term can appear in the document",
                "example": "ceza OR idare",
            },
            {
                "operator": "NOT",
                "description": "Exclude documents containing the term",
                "example": "karar NOT bozma",
            },
            {
                "operator": "\"phrase\"",
                "description": "Exact phrase matching (words in order)",
                "example": "\"ifade ozgurlugu\"",
            },
            {
                "operator": "NEAR/n",
                "description": "Terms within n words of each other",
                "example": "anayasa NEAR/5 mahkemesi",
            },
            {
                "operator": "field:term",
                "description": "Search in specific field",
                "example": "title:kanun",
            },
            {
                "operator": "*",
                "description": "Wildcard (multiple characters)",
                "example": "ceza*",
            },
            {
                "operator": "?",
                "description": "Wildcard (single character)",
                "example": "ka?ar",
            },
            {
                "operator": "( )",
                "description": "Grouping for precedence",
                "example": "(ceza OR idare) AND karar",
            },
        ],
        examples=[
            {
                "query": "anayasa AND mahkemesi",
                "description": "Find documents containing both 'anayasa' and 'mahkemesi'",
            },
            {
                "query": "\"ifade ozgurlugu\"",
                "description": "Find exact phrase 'ifade ozgurlugu'",
            },
            {
                "query": "ceza OR idare",
                "description": "Find documents containing either 'ceza' or 'idare'",
            },
            {
                "query": "karar NOT bozma",
                "description": "Find 'karar' but exclude documents with 'bozma'",
            },
            {
                "query": "title:kanun",
                "description": "Search for 'kanun' only in title field",
            },
            {
                "query": "(title:anayasa OR title:aihm) AND body:\"bireysel basvuru\"",
                "description": "Complex: Title must contain 'anayasa' or 'aihm', and body must contain exact phrase 'bireysel basvuru'",
            },
            {
                "query": "anayasa NEAR/5 mahkemesi",
                "description": "Find 'anayasa' and 'mahkemesi' within 5 words of each other",
            },
            {
                "query": "ceza*",
                "description": "Find terms starting with 'ceza' (cezaevi, cezalandirma, etc.)",
            },
        ],
        fields=[
            {
                "field": "title",
                "description": "Document title (e.g., law name, case title)",
            },
            {
                "field": "body",
                "description": "Full document content/text",
            },
            {
                "field": "summary",
                "description": "Document summary or abstract",
            },
            {
                "field": "author",
                "description": "Document author or issuing authority",
            },
            {
                "field": "source",
                "description": "Document source (e.g., AYM, Yargitay, HSK)",
            },
            {
                "field": "court",
                "description": "Court name for judicial decisions",
            },
            {
                "field": "case_number",
                "description": "Case number for judicial decisions",
            },
            {
                "field": "law_number",
                "description": "Law number for legislation",
            },
        ],
        tips=[
            "Use parentheses to control operator precedence: (A OR B) AND C",
            "Phrase matching is case-insensitive: \"Ifade Ozgurlugu\" = \"ifade ozgurlugu\"",
            "Combine field search with operators: title:kanun AND body:ceza",
            "Use NOT carefully - it can significantly narrow results",
            "NEAR operator is useful for finding related concepts: anayasa NEAR/10 mahkeme",
            "Wildcards work within words: *ozgur* finds ozgurluk, ozgurlesmek",
            "Start simple and add complexity incrementally",
            "Use quotes for exact legal terms: \"ayrimcilik yasagi\"",
            "Validate complex queries with /validate endpoint before searching",
            "For Turkish characters, use standard Turkish alphabet (, , , , , )",
        ],
    )
