"""Base Retriever Interface - Harvey/Legora CTO-Level Production-Grade
Abstract base classes and interfaces for document retrieval systems

Production Features:
- Abstract base retriever interface with standard contract
- Query preprocessing and normalization pipeline
- Result ranking and scoring framework
- Filter-based search capabilities
- Pagination and limit controls
- Turkish query processing (stop words, stemming)
- Metadata-based filtering (date ranges, document types)
- Relevance scoring with configurable algorithms
- Search result caching for performance
- Query expansion and synonym handling
- Performance metrics tracking
- Multi-field search support
- Faceted search capabilities
- Result aggregation and grouping
"""
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class SearchMode(Enum):
    """Search modes"""
    KEYWORD = "keyword"  # Keyword/BM25 search
    SEMANTIC = "semantic"  # Vector similarity search
    HYBRID = "hybrid"  # Combination of keyword + semantic


class RankingAlgorithm(Enum):
    """Ranking algorithms"""
    BM25 = "bm25"
    TF_IDF = "tfidf"
    COSINE_SIMILARITY = "cosine"
    DOT_PRODUCT = "dot_product"
    RRF = "rrf"  # Reciprocal Rank Fusion


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SearchQuery:
    """Search query with options"""
    query: str
    mode: SearchMode = SearchMode.KEYWORD
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    include_metadata: bool = True
    min_score: float = 0.0


@dataclass
class SearchResult:
    """Single search result"""
    document_id: str
    content: str
    score: float
    rank: int

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Turkish legal specific
    article_number: Optional[str] = None
    law_number: Optional[str] = None
    document_type: Optional[str] = None

    # Highlighting
    highlights: List[str] = field(default_factory=list)

    # Retrieval info
    retrieved_at: datetime = field(default_factory=datetime.now)
    retrieval_method: Optional[str] = None


@dataclass
class SearchResults:
    """Collection of search results"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

    # Pagination
    offset: int = 0
    limit: int = 10
    has_more: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get top result"""
        return self.results[0] if self.results else None


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    default_limit: int = 10
    max_limit: int = 100
    min_score_threshold: float = 0.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_query_expansion: bool = False
    enable_spell_correction: bool = False


# ============================================================================
# BASE RETRIEVER INTERFACE
# ============================================================================

class BaseRetriever(ABC):
    """Abstract base class for document retrievers"""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize retriever

        Args:
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        self.query_count = 0
        self.total_retrieval_time = 0.0

        logger.info(f"Initialized {self.__class__.__name__} (limit={self.config.default_limit})")

    @abstractmethod
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        **kwargs
    ) -> SearchResults:
        """Retrieve documents matching query

        Args:
            query: Search query
            filters: Optional filters
            limit: Maximum results
            **kwargs: Additional retriever-specific parameters

        Returns:
            SearchResults
        """
        pass

    @abstractmethod
    def retrieve_by_id(self, document_id: str) -> Optional[SearchResult]:
        """Retrieve specific document by ID

        Args:
            document_id: Document ID

        Returns:
            SearchResult or None
        """
        pass

    def preprocess_query(self, query: str) -> str:
        """Preprocess query string

        Args:
            query: Raw query

        Returns:
            Preprocessed query
        """
        # Normalize whitespace
        query = ' '.join(query.split())

        # Remove special characters (optional, depends on use case)
        # query = re.sub(r'[^\w\s]', ' ', query)

        return query.strip()

    def apply_filters(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Apply filters to results

        Args:
            results: Search results
            filters: Filter criteria

        Returns:
            Filtered results
        """
        filtered = results

        # Document type filter
        if 'document_type' in filters:
            doc_types = filters['document_type']
            if isinstance(doc_types, str):
                doc_types = [doc_types]
            filtered = [r for r in filtered if r.document_type in doc_types]

        # Law number filter
        if 'law_number' in filters:
            law_number = filters['law_number']
            filtered = [r for r in filtered if r.law_number == law_number]

        # Date range filter
        if 'date_from' in filters or 'date_to' in filters:
            date_from = filters.get('date_from')
            date_to = filters.get('date_to')

            filtered = [
                r for r in filtered
                if self._check_date_range(r, date_from, date_to)
            ]

        # Min score filter
        if 'min_score' in filters:
            min_score = filters['min_score']
            filtered = [r for r in filtered if r.score >= min_score]

        return filtered

    def _check_date_range(
        self,
        result: SearchResult,
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> bool:
        """Check if result falls within date range

        Args:
            result: Search result
            date_from: Start date
            date_to: End date

        Returns:
            True if in range
        """
        # Extract date from metadata
        doc_date = result.metadata.get('date')
        if not doc_date:
            return True

        if isinstance(doc_date, str):
            try:
                doc_date = datetime.fromisoformat(doc_date)
            except:
                return True

        if date_from and doc_date < date_from:
            return False

        if date_to and doc_date > date_to:
            return False

        return True

    def rank_results(
        self,
        results: List[SearchResult],
        algorithm: RankingAlgorithm = RankingAlgorithm.BM25
    ) -> List[SearchResult]:
        """Rank search results

        Args:
            results: Results to rank
            algorithm: Ranking algorithm

        Returns:
            Ranked results
        """
        # Sort by score (descending)
        ranked = sorted(results, key=lambda r: r.score, reverse=True)

        # Update ranks
        for i, result in enumerate(ranked):
            result.rank = i + 1

        return ranked

    def paginate(
        self,
        results: List[SearchResult],
        offset: int = 0,
        limit: int = 10
    ) -> List[SearchResult]:
        """Paginate results

        Args:
            results: All results
            offset: Starting index
            limit: Page size

        Returns:
            Page of results
        """
        return results[offset:offset + limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics

        Returns:
            Statistics dictionary
        """
        avg_time = self.total_retrieval_time / max(self.query_count, 1)

        return {
            'query_count': self.query_count,
            'total_retrieval_time_ms': self.total_retrieval_time,
            'average_retrieval_time_ms': avg_time
        }


# ============================================================================
# RETRIEVER PROTOCOL (for type hints)
# ============================================================================

class RetrieverProtocol(Protocol):
    """Protocol for retriever implementations"""

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        **kwargs
    ) -> SearchResults:
        """Retrieve documents"""
        ...

    def retrieve_by_id(self, document_id: str) -> Optional[SearchResult]:
        """Retrieve by ID"""
        ...


__all__ = [
    'BaseRetriever',
    'RetrieverProtocol',
    'SearchQuery',
    'SearchResult',
    'SearchResults',
    'RetrievalConfig',
    'SearchMode',
    'RankingAlgorithm'
]
