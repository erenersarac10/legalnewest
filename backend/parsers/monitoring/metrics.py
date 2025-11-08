"""Parser Metrics - Harvey/Legora CTO-Level Production-Grade
Prometheus metrics for Turkish legal document parsing pipeline

Production Features:
- Prometheus Counter, Gauge, Histogram, Summary metrics
- Parsing success/failure tracking
- Processing time histograms
- Document type distribution
- Parser component metrics (extraction, structure, semantic, NER)
- Citation extraction metrics
- Validation error tracking
- Throughput monitoring
- Queue depth gauges
- Custom metric labels (document_type, source, parser_version)
- Grafana-ready metric naming
"""
from typing import Dict, Any, Optional
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# PROMETHEUS METRICS (with fallback)
# ============================================================================

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available - using mock metrics")


# ============================================================================
# MOCK METRICS (fallback when Prometheus not available)
# ============================================================================

class MockMetric:
    """Mock metric for when Prometheus is not available"""

    def __init__(self, name: str, description: str, labelnames: tuple = ()):
        self.name = name
        self.description = description
        self.labelnames = labelnames
        self._value = 0

    def inc(self, amount: float = 1):
        """Increment counter"""
        self._value += amount

    def dec(self, amount: float = 1):
        """Decrement gauge"""
        self._value -= amount

    def set(self, value: float):
        """Set gauge value"""
        self._value = value

    def observe(self, value: float):
        """Observe value (histogram/summary)"""
        pass

    def labels(self, **kwargs):
        """Return self for chaining"""
        return self

    def info(self, data: dict):
        """Set info data"""
        pass


# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Document processing counters
    DOCUMENTS_PROCESSED_TOTAL = Counter(
        'legal_parser_documents_processed_total',
        'Total number of documents processed',
        ['document_type', 'status']  # status: success, failed
    )

    DOCUMENTS_PARSED_TOTAL = Counter(
        'legal_parser_documents_parsed_total',
        'Total number of documents successfully parsed',
        ['document_type', 'source']
    )

    DOCUMENTS_FAILED_TOTAL = Counter(
        'legal_parser_documents_failed_total',
        'Total number of documents that failed parsing',
        ['document_type', 'error_type']
    )

    # Processing time histograms
    PARSE_DURATION_SECONDS = Histogram(
        'legal_parser_parse_duration_seconds',
        'Document parsing duration in seconds',
        ['document_type', 'parser_component'],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
    )

    EXTRACTION_DURATION_SECONDS = Histogram(
        'legal_parser_extraction_duration_seconds',
        'Text extraction duration in seconds',
        ['document_type', 'extraction_method'],
        buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )

    # Component-specific counters
    ARTICLES_EXTRACTED_TOTAL = Counter(
        'legal_parser_articles_extracted_total',
        'Total number of articles extracted',
        ['document_type']
    )

    CITATIONS_EXTRACTED_TOTAL = Counter(
        'legal_parser_citations_extracted_total',
        'Total number of citations extracted',
        ['citation_type']
    )

    ENTITIES_EXTRACTED_TOTAL = Counter(
        'legal_parser_entities_extracted_total',
        'Total number of named entities extracted',
        ['entity_type']
    )

    # Validation metrics
    VALIDATION_ERRORS_TOTAL = Counter(
        'legal_parser_validation_errors_total',
        'Total number of validation errors',
        ['error_code', 'severity']
    )

    VALIDATION_DURATION_SECONDS = Histogram(
        'legal_parser_validation_duration_seconds',
        'Validation duration in seconds',
        ['validator_type'],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    )

    # Queue and throughput gauges
    PARSE_QUEUE_SIZE = Gauge(
        'legal_parser_queue_size',
        'Number of documents in parsing queue',
        ['queue_type']
    )

    ACTIVE_PARSERS = Gauge(
        'legal_parser_active_parsers',
        'Number of active parser workers'
    )

    # Parser info
    PARSER_INFO = Info(
        'legal_parser_info',
        'Parser version and configuration info'
    )

else:
    # Mock metrics
    DOCUMENTS_PROCESSED_TOTAL = MockMetric(
        'legal_parser_documents_processed_total',
        'Total number of documents processed',
        ('document_type', 'status')
    )

    DOCUMENTS_PARSED_TOTAL = MockMetric(
        'legal_parser_documents_parsed_total',
        'Total number of documents successfully parsed',
        ('document_type', 'source')
    )

    DOCUMENTS_FAILED_TOTAL = MockMetric(
        'legal_parser_documents_failed_total',
        'Total number of documents that failed parsing',
        ('document_type', 'error_type')
    )

    PARSE_DURATION_SECONDS = MockMetric(
        'legal_parser_parse_duration_seconds',
        'Document parsing duration in seconds',
        ('document_type', 'parser_component')
    )

    EXTRACTION_DURATION_SECONDS = MockMetric(
        'legal_parser_extraction_duration_seconds',
        'Text extraction duration in seconds',
        ('document_type', 'extraction_method')
    )

    ARTICLES_EXTRACTED_TOTAL = MockMetric(
        'legal_parser_articles_extracted_total',
        'Total number of articles extracted',
        ('document_type',)
    )

    CITATIONS_EXTRACTED_TOTAL = MockMetric(
        'legal_parser_citations_extracted_total',
        'Total number of citations extracted',
        ('citation_type',)
    )

    ENTITIES_EXTRACTED_TOTAL = MockMetric(
        'legal_parser_entities_extracted_total',
        'Total number of named entities extracted',
        ('entity_type',)
    )

    VALIDATION_ERRORS_TOTAL = MockMetric(
        'legal_parser_validation_errors_total',
        'Total number of validation errors',
        ('error_code', 'severity')
    )

    VALIDATION_DURATION_SECONDS = MockMetric(
        'legal_parser_validation_duration_seconds',
        'Validation duration in seconds',
        ('validator_type',)
    )

    PARSE_QUEUE_SIZE = MockMetric(
        'legal_parser_queue_size',
        'Number of documents in parsing queue',
        ('queue_type',)
    )

    ACTIVE_PARSERS = MockMetric(
        'legal_parser_active_parsers',
        'Number of active parser workers'
    )

    PARSER_INFO = MockMetric(
        'legal_parser_info',
        'Parser version and configuration info'
    )


# ============================================================================
# METRIC HELPERS
# ============================================================================

def track_parsing_time(document_type: str = "unknown", component: str = "full"):
    """Decorator to track parsing time

    Args:
        document_type: Type of document being parsed
        component: Parser component (extraction, structure, semantic, etc.)

    Usage:
        @track_parsing_time(document_type="KANUN", component="structure")
        def parse_structure(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Record success
                PARSE_DURATION_SECONDS.labels(
                    document_type=document_type,
                    parser_component=component
                ).observe(duration)

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record failure
                PARSE_DURATION_SECONDS.labels(
                    document_type=document_type,
                    parser_component=component
                ).observe(duration)

                DOCUMENTS_FAILED_TOTAL.labels(
                    document_type=document_type,
                    error_type=type(e).__name__
                ).inc()

                raise

        return wrapper
    return decorator


def record_document_processed(
    document_type: str,
    status: str,
    source: str = "unknown"
) -> None:
    """Record a processed document

    Args:
        document_type: Type of document (KANUN, YONETMELIK, etc.)
        status: Processing status (success, failed)
        source: Document source
    """
    DOCUMENTS_PROCESSED_TOTAL.labels(
        document_type=document_type,
        status=status
    ).inc()

    if status == "success":
        DOCUMENTS_PARSED_TOTAL.labels(
            document_type=document_type,
            source=source
        ).inc()


def record_articles_extracted(document_type: str, count: int) -> None:
    """Record extracted articles

    Args:
        document_type: Type of document
        count: Number of articles extracted
    """
    ARTICLES_EXTRACTED_TOTAL.labels(
        document_type=document_type
    ).inc(count)


def record_citations_extracted(citation_type: str, count: int) -> None:
    """Record extracted citations

    Args:
        citation_type: Type of citation (EXPLICIT, IMPLICIT, etc.)
        count: Number of citations extracted
    """
    CITATIONS_EXTRACTED_TOTAL.labels(
        citation_type=citation_type
    ).inc(count)


def record_entities_extracted(entity_type: str, count: int) -> None:
    """Record extracted entities

    Args:
        entity_type: Type of entity (LAW, ARTICLE, DATE, etc.)
        count: Number of entities extracted
    """
    ENTITIES_EXTRACTED_TOTAL.labels(
        entity_type=entity_type
    ).inc(count)


def record_validation_error(error_code: str, severity: str) -> None:
    """Record validation error

    Args:
        error_code: Error code (MISSING_TITLE, INVALID_DATE, etc.)
        severity: Error severity (ERROR, WARNING, INFO)
    """
    VALIDATION_ERRORS_TOTAL.labels(
        error_code=error_code,
        severity=severity
    ).inc()


def update_queue_size(queue_type: str, size: int) -> None:
    """Update queue size gauge

    Args:
        queue_type: Type of queue (parsing, validation, indexing)
        size: Current queue size
    """
    PARSE_QUEUE_SIZE.labels(
        queue_type=queue_type
    ).set(size)


def update_active_parsers(count: int) -> None:
    """Update active parsers gauge

    Args:
        count: Number of active parsers
    """
    ACTIVE_PARSERS.set(count)


def set_parser_info(version: str, config: Dict[str, Any]) -> None:
    """Set parser version and config info

    Args:
        version: Parser version
        config: Configuration dict
    """
    info_data = {
        'version': version,
        **{k: str(v) for k, v in config.items()}
    }

    PARSER_INFO.info(info_data)


# ============================================================================
# METRICS CONTEXT MANAGER
# ============================================================================

class ParsingMetricsContext:
    """Context manager for tracking parsing metrics"""

    def __init__(
        self,
        document_type: str,
        component: str = "full",
        source: str = "unknown"
    ):
        """Initialize metrics context

        Args:
            document_type: Type of document
            component: Parser component
            source: Document source
        """
        self.document_type = document_type
        self.component = component
        self.source = source
        self.start_time = None

    def __enter__(self):
        """Enter context"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record metrics"""
        duration = time.time() - self.start_time

        # Record duration
        PARSE_DURATION_SECONDS.labels(
            document_type=self.document_type,
            parser_component=self.component
        ).observe(duration)

        # Record status
        if exc_type is None:
            # Success
            record_document_processed(
                self.document_type,
                "success",
                self.source
            )
        else:
            # Failure
            record_document_processed(
                self.document_type,
                "failed",
                self.source
            )

            DOCUMENTS_FAILED_TOTAL.labels(
                document_type=self.document_type,
                error_type=exc_type.__name__
            ).inc()

        return False  # Don't suppress exceptions


__all__ = [
    'DOCUMENTS_PROCESSED_TOTAL',
    'DOCUMENTS_PARSED_TOTAL',
    'DOCUMENTS_FAILED_TOTAL',
    'PARSE_DURATION_SECONDS',
    'EXTRACTION_DURATION_SECONDS',
    'ARTICLES_EXTRACTED_TOTAL',
    'CITATIONS_EXTRACTED_TOTAL',
    'ENTITIES_EXTRACTED_TOTAL',
    'VALIDATION_ERRORS_TOTAL',
    'VALIDATION_DURATION_SECONDS',
    'PARSE_QUEUE_SIZE',
    'ACTIVE_PARSERS',
    'PARSER_INFO',
    'track_parsing_time',
    'record_document_processed',
    'record_articles_extracted',
    'record_citations_extracted',
    'record_entities_extracted',
    'record_validation_error',
    'update_queue_size',
    'update_active_parsers',
    'set_parser_info',
    'ParsingMetricsContext',
    'PROMETHEUS_AVAILABLE'
]
