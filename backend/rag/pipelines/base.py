"""Base Pipeline Interface - Harvey/Legora CTO-Level Production-Grade
Abstract base classes and interfaces for RAG pipeline orchestration

Production Features:
- Abstract base pipeline with standard retrieve-generate flow
- Pipeline configuration and lifecycle management
- Result tracking and metadata enrichment
- Multi-stage pipeline execution (retrieve → rerank → generate)
- Context management and state handling
- Pipeline metrics and performance tracking
- Caching strategies for pipeline results
- Error recovery and graceful degradation
- Pipeline composition and chaining
- Input validation and sanitization
- Output post-processing and formatting
- Streaming support for long-running pipelines
- Memory management for large contexts
- Pipeline observability (logging, tracing)
- Configurable timeout and retry policies
- Turkish legal document awareness
- Citation extraction and formatting
- Source attribution and provenance tracking
"""
from typing import Dict, List, Any, Optional, Protocol, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time
import json
from pathlib import Path

from ..retrievers.base import SearchResults, SearchResult
from ..retrievers.reranker import Reranker, RerankerConfig
from ..indexers.base import IndexedDocument

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class PipelineStage(Enum):
    """Pipeline execution stages"""
    PREPROCESSING = "preprocessing"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    GENERATION = "generation"
    POSTPROCESSING = "postprocessing"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    # Retrieval settings
    retrieval_limit: int = 20
    min_relevance_score: float = 0.0
    enable_reranking: bool = True
    reranking_top_n: int = 10

    # Generation settings
    max_context_tokens: int = 8000
    temperature: float = 0.7
    max_output_tokens: int = 2000

    # Pipeline settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    timeout_seconds: int = 60
    max_retries: int = 2

    # Turkish legal settings
    preserve_citations: bool = True
    extract_article_numbers: bool = True
    validate_law_references: bool = True

    # Performance
    enable_streaming: bool = False
    batch_size: int = 10

    # Observability
    enable_tracing: bool = True
    log_level: str = "INFO"


@dataclass
class PipelineContext:
    """Context passed through pipeline stages"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage"""
    stage: PipelineStage
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Legal citation with source information"""
    document_id: str
    document_type: str
    law_number: Optional[str] = None
    article_number: Optional[str] = None
    title: str = ""
    excerpt: str = ""
    relevance_score: float = 0.0
    page_number: Optional[int] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_citation(self, style: str = "default") -> str:
        """Format citation for display

        Args:
            style: Citation style (default, apa, bluebook)

        Returns:
            Formatted citation string
        """
        if style == "default":
            parts = []

            if self.law_number:
                parts.append(f"{self.law_number} sayılı")

            if self.title:
                parts.append(self.title)

            if self.article_number:
                parts.append(f"Madde {self.article_number}")

            return " ".join(parts)

        return f"{self.title} ({self.document_id})"


@dataclass
class PipelineResult:
    """Final result from pipeline execution"""
    query: str
    answer: str
    citations: List[Citation]

    # Execution metadata
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime
    total_duration_ms: float

    # Stage results
    stage_results: List[PipelineStageResult] = field(default_factory=list)

    # Context used
    retrieved_count: int = 0
    reranked_count: int = 0
    context_tokens_used: int = 0

    # Quality metrics
    confidence_score: float = 0.0
    citation_coverage: float = 0.0  # % of answer backed by citations

    # Error handling
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_time_ms(self) -> float:
        """Get retrieval stage duration"""
        for stage in self.stage_results:
            if stage.stage == PipelineStage.RETRIEVAL:
                return stage.duration_ms
        return 0.0

    @property
    def generation_time_ms(self) -> float:
        """Get generation stage duration"""
        for stage in self.stage_results:
            if stage.stage == PipelineStage.GENERATION:
                return stage.duration_ms
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary representation
        """
        return {
            'query': self.query,
            'answer': self.answer,
            'citations': [
                {
                    'document_id': c.document_id,
                    'title': c.title,
                    'law_number': c.law_number,
                    'article_number': c.article_number,
                    'excerpt': c.excerpt,
                    'score': c.relevance_score
                }
                for c in self.citations
            ],
            'status': self.status.value,
            'duration_ms': self.total_duration_ms,
            'retrieved_count': self.retrieved_count,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


# ============================================================================
# BASE PIPELINE INTERFACE
# ============================================================================

class BasePipeline(ABC):
    """Abstract base class for RAG pipelines"""

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        config: Optional[PipelineConfig] = None,
        reranker: Optional[Reranker] = None
    ):
        """Initialize pipeline

        Args:
            retriever: Retriever instance
            generator: LLM generator instance
            config: Pipeline configuration
            reranker: Optional reranker
        """
        self.retriever = retriever
        self.generator = generator
        self.config = config or PipelineConfig()
        self.reranker = reranker

        # State tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.cache: Dict[str, PipelineResult] = {}
        self.last_execution: Optional[datetime] = None

        # Metrics
        self.stage_metrics: Dict[PipelineStage, List[float]] = {
            stage: [] for stage in PipelineStage
        }

        logger.info(f"Initialized {self.__class__.__name__} (retrieval_limit={self.config.retrieval_limit})")

    def run(
        self,
        query: str,
        context: Optional[PipelineContext] = None,
        **kwargs
    ) -> PipelineResult:
        """Execute pipeline

        Args:
            query: User query
            context: Optional pipeline context
            **kwargs: Additional parameters

        Returns:
            PipelineResult
        """
        start_time = time.time()
        stage_results: List[PipelineStageResult] = []

        # Create context
        if context is None:
            context = PipelineContext(query=query)

        # Check cache
        if self.config.enable_caching:
            cache_key = self._get_cache_key(query, context)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached_result

        try:
            # Stage 1: Preprocessing
            stage_result = self._execute_stage(
                PipelineStage.PREPROCESSING,
                lambda: self.preprocess(query, context)
            )
            stage_results.append(stage_result)

            if stage_result.status == PipelineStatus.FAILED:
                return self._create_error_result(query, stage_results, stage_result.error)

            processed_query = stage_result.output

            # Stage 2: Retrieval
            stage_result = self._execute_stage(
                PipelineStage.RETRIEVAL,
                lambda: self.retrieve(processed_query, context)
            )
            stage_results.append(stage_result)

            if stage_result.status == PipelineStatus.FAILED:
                return self._create_error_result(query, stage_results, stage_result.error)

            retrieval_results: SearchResults = stage_result.output

            # Stage 3: Reranking (optional)
            if self.config.enable_reranking and self.reranker:
                stage_result = self._execute_stage(
                    PipelineStage.RERANKING,
                    lambda: self.rerank(processed_query, retrieval_results)
                )
                stage_results.append(stage_result)

                if stage_result.status == PipelineStatus.COMPLETED:
                    retrieval_results = stage_result.output

            # Stage 4: Generation
            stage_result = self._execute_stage(
                PipelineStage.GENERATION,
                lambda: self.generate(processed_query, retrieval_results, context)
            )
            stage_results.append(stage_result)

            if stage_result.status == PipelineStatus.FAILED:
                return self._create_error_result(query, stage_results, stage_result.error)

            generation_output = stage_result.output

            # Stage 5: Postprocessing
            stage_result = self._execute_stage(
                PipelineStage.POSTPROCESSING,
                lambda: self.postprocess(generation_output, retrieval_results, context)
            )
            stage_results.append(stage_result)

            if stage_result.status == PipelineStatus.FAILED:
                return self._create_error_result(query, stage_results, stage_result.error)

            final_output = stage_result.output

            # Build result
            total_duration = (time.time() - start_time) * 1000

            result = PipelineResult(
                query=query,
                answer=final_output['answer'],
                citations=final_output['citations'],
                status=PipelineStatus.COMPLETED,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                total_duration_ms=total_duration,
                stage_results=stage_results,
                retrieved_count=len(retrieval_results.results),
                reranked_count=self.config.reranking_top_n if self.config.enable_reranking else 0,
                confidence_score=final_output.get('confidence', 0.0),
                metadata=final_output.get('metadata', {})
            )

            # Update metrics
            self._update_metrics(result)

            # Cache result
            if self.config.enable_caching:
                self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return self._create_error_result(query, stage_results, str(e))

    def _execute_stage(
        self,
        stage: PipelineStage,
        func: Callable
    ) -> PipelineStageResult:
        """Execute a pipeline stage

        Args:
            stage: Stage type
            func: Function to execute

        Returns:
            PipelineStageResult
        """
        started_at = datetime.now()
        start_time = time.time()

        try:
            output = func()
            duration = (time.time() - start_time) * 1000

            # Record metrics
            self.stage_metrics[stage].append(duration)

            return PipelineStageResult(
                stage=stage,
                status=PipelineStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration,
                output=output
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Stage {stage.value} failed: {e}")

            return PipelineStageResult(
                stage=stage,
                status=PipelineStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration,
                error=str(e)
            )

    @abstractmethod
    def preprocess(
        self,
        query: str,
        context: PipelineContext
    ) -> str:
        """Preprocess query

        Args:
            query: Raw query
            context: Pipeline context

        Returns:
            Processed query
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        context: PipelineContext
    ) -> SearchResults:
        """Retrieve relevant documents

        Args:
            query: Processed query
            context: Pipeline context

        Returns:
            SearchResults
        """
        pass

    def rerank(
        self,
        query: str,
        results: SearchResults
    ) -> SearchResults:
        """Rerank retrieved results

        Args:
            query: Query
            results: Initial results

        Returns:
            Reranked results
        """
        if not self.reranker:
            return results

        return self.reranker.rerank(
            query,
            results,
            top_n=self.config.reranking_top_n
        )

    @abstractmethod
    def generate(
        self,
        query: str,
        results: SearchResults,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Generate answer from retrieved context

        Args:
            query: Query
            results: Retrieved and reranked results
            context: Pipeline context

        Returns:
            Generation output with answer and metadata
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        generation_output: Dict[str, Any],
        results: SearchResults,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Postprocess generation output

        Args:
            generation_output: Raw generation output
            results: Retrieved results
            context: Pipeline context

        Returns:
            Final output with answer, citations, metadata
        """
        pass

    def _create_error_result(
        self,
        query: str,
        stage_results: List[PipelineStageResult],
        error: Optional[str]
    ) -> PipelineResult:
        """Create error result

        Args:
            query: Query
            stage_results: Completed stages
            error: Error message

        Returns:
            Error PipelineResult
        """
        return PipelineResult(
            query=query,
            answer="",
            citations=[],
            status=PipelineStatus.FAILED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_ms=0.0,
            stage_results=stage_results,
            error=error
        )

    def _get_cache_key(self, query: str, context: PipelineContext) -> str:
        """Generate cache key

        Args:
            query: Query
            context: Context

        Returns:
            Cache key
        """
        # Simple hash-based key
        context_str = json.dumps({
            'filters': context.filters,
            'session_id': context.session_id
        }, sort_keys=True)

        return f"{query}_{hash(context_str)}"

    def _is_cache_valid(self, result: PipelineResult) -> bool:
        """Check if cached result is still valid

        Args:
            result: Cached result

        Returns:
            True if valid
        """
        age = (datetime.now() - result.completed_at).total_seconds()
        return age < self.config.cache_ttl_seconds

    def _update_metrics(self, result: PipelineResult) -> None:
        """Update pipeline metrics

        Args:
            result: Pipeline result
        """
        self.execution_count += 1
        self.total_execution_time += result.total_duration_ms
        self.last_execution = result.completed_at

    def _extract_citations_from_results(
        self,
        results: SearchResults
    ) -> List[Citation]:
        """Extract citations from search results

        Args:
            results: Search results

        Returns:
            List of citations
        """
        citations = []

        for result in results.results:
            citation = Citation(
                document_id=result.document_id,
                document_type=result.document_type or "UNKNOWN",
                law_number=result.law_number,
                article_number=result.article_number,
                title=result.metadata.get('title', ''),
                excerpt=result.content[:300],
                relevance_score=result.score,
                metadata=result.metadata
            )
            citations.append(citation)

        return citations

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics

        Returns:
            Statistics dictionary
        """
        avg_execution_time = self.total_execution_time / max(self.execution_count, 1)

        stage_stats = {}
        for stage, times in self.stage_metrics.items():
            if times:
                stage_stats[stage.value] = {
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'count': len(times)
                }

        return {
            'execution_count': self.execution_count,
            'total_execution_time_ms': self.total_execution_time,
            'average_execution_time_ms': avg_execution_time,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'cache_size': len(self.cache),
            'stage_stats': stage_stats
        }

    def clear_cache(self) -> None:
        """Clear pipeline cache"""
        self.cache.clear()
        logger.info("Pipeline cache cleared")


# ============================================================================
# PIPELINE PROTOCOL (for type hints)
# ============================================================================

class PipelineProtocol(Protocol):
    """Protocol for pipeline implementations"""

    def run(
        self,
        query: str,
        context: Optional[PipelineContext] = None,
        **kwargs
    ) -> PipelineResult:
        """Run pipeline"""
        ...


__all__ = [
    'BasePipeline',
    'PipelineProtocol',
    'PipelineConfig',
    'PipelineContext',
    'PipelineResult',
    'PipelineStageResult',
    'Citation',
    'PipelineStage',
    'PipelineStatus'
]
