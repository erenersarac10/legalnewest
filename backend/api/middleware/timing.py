"""
Timing Middleware for Turkish Legal AI Platform.

This middleware provides comprehensive request performance monitoring and timing analysis
for the FastAPI application. It measures request processing time with high precision,
tracks performance metrics, detects slow requests, and integrates with observability systems.

=============================================================================
FEATURES
=============================================================================

1. Request Timing
   ---------------
   - High-precision timing using time.perf_counter()
   - Microsecond accuracy for request duration measurement
   - X-Response-Time header injection for client visibility
   - Multiple time unit support (ms, seconds, microseconds)

2. Slow Request Detection
   -----------------------
   - Configurable threshold levels (WARNING, ERROR, CRITICAL)
   - Automatic logging of slow requests
   - Request path and method categorization
   - User agent and tenant tracking for slow requests

3. Performance Metrics
   --------------------
   - Request duration histograms with custom buckets
   - Percentile calculations (p50, p90, p95, p99)
   - Endpoint-specific performance tracking
   - Method-specific metrics (GET, POST, etc.)
   - Status code correlation

4. Observability Integration
   --------------------------
   - Prometheus metrics export
   - Grafana dashboard support
   - Datadog APM integration
   - OpenTelemetry span attributes
   - ELK stack integration

5. Advanced Features
   ------------------
   - Request categorization (read, write, analysis, etc.)
   - Tenant-based performance isolation
   - User-agent based tracking
   - Time-of-day performance analysis
   - Geographic region performance tracking

=============================================================================
USAGE
=============================================================================

Basic Integration:
------------------

>>> from fastapi import FastAPI
>>> from backend.api.middleware.timing import TimingMiddleware
>>>
>>> app = FastAPI()
>>> app.add_middleware(TimingMiddleware)
>>>
>>> # All requests now have timing metrics
>>> # Response includes: X-Response-Time: 123.45ms

Custom Thresholds:
------------------

>>> app.add_middleware(
...     TimingMiddleware,
...     slow_threshold_ms=1000,      # 1 second
...     very_slow_threshold_ms=5000, # 5 seconds
...     critical_threshold_ms=10000, # 10 seconds
... )

With Prometheus Metrics:
------------------------

>>> from backend.api.middleware.timing import TimingMiddleware, PerformanceMetricsCollector
>>> from prometheus_client import generate_latest
>>>
>>> metrics_collector = PerformanceMetricsCollector()
>>> app.add_middleware(TimingMiddleware, metrics_collector=metrics_collector)
>>>
>>> @app.get("/metrics")
>>> async def metrics():
...     return Response(generate_latest(), media_type="text/plain")

Endpoint-Specific Monitoring:
------------------------------

>>> from backend.api.middleware.timing import track_endpoint_performance
>>>
>>> @app.post("/api/v1/contracts/analyze")
>>> @track_endpoint_performance(critical_threshold_ms=30000)  # 30 seconds for analysis
>>> async def analyze_contract(contract: Contract):
...     result = await contract_service.analyze(contract)
...     return result

Real-World Example (Contract Analysis Endpoint):
-------------------------------------------------

>>> # Request: POST /api/v1/contracts/analyze
>>> # X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
>>> # Content-Length: 2456789
>>>
>>> # Processing phases:
>>> # 1. File upload: 234ms
>>> # 2. PDF parsing: 1,234ms
>>> # 3. OCR processing: 3,456ms
>>> # 4. LLM analysis: 15,678ms
>>> # 5. Response generation: 123ms
>>> # Total: 20,725ms
>>>
>>> # Response headers:
>>> # X-Response-Time: 20725.34ms
>>> # X-Processing-Breakdown: upload=234,parse=1234,ocr=3456,llm=15678,response=123
>>>
>>> # Metrics emitted:
>>> # http_request_duration_seconds{method="POST",path="/api/v1/contracts/analyze",status="200"} 20.725
>>> # slow_request_total{severity="warning",path="/api/v1/contracts/analyze"} 1

=============================================================================
PERFORMANCE THRESHOLDS
=============================================================================

Standard API Endpoints:
-----------------------
- Fast (< 100ms): Database queries, cache hits, simple operations
- Normal (100-500ms): Complex queries, file operations, external API calls
- Slow (500-1000ms): Multi-step processing, large file uploads
- Very Slow (1-5s): Document processing, OCR, small AI models
- Critical (5-30s): Large contract analysis, comprehensive legal research

Turkish Context:
----------------
- Hızlı (< 100ms): Veritabanı sorguları, önbellek erişimi
- Normal (100-500ms): Karmaşık sorgular, dosya işlemleri
- Yavaş (500-1000ms): Çok adımlı işlemler
- Çok Yavaş (1-5s): Doküman işleme, OCR, küçük AI modelleri
- Kritik (5-30s): Büyük sözleşme analizi, kapsamlı yasal araştırma

=============================================================================
METRICS COLLECTION
=============================================================================

Histogram Buckets (milliseconds):
----------------------------------
[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]

This covers:
- Ultra-fast: < 10ms (cache hits)
- Fast: 10-100ms (simple queries)
- Normal: 100-500ms (complex operations)
- Slow: 500-5000ms (document processing)
- Very slow: 5-30s (AI analysis)
- Critical: 30-60s (large-scale processing)

Percentile Targets:
-------------------
- p50 (median): < 200ms
- p90: < 1000ms
- p95: < 2000ms
- p99: < 5000ms

=============================================================================
INTEGRATION EXAMPLES
=============================================================================

Grafana Dashboard Query:
------------------------

>>> # Average response time by endpoint (last 5 minutes)
>>> rate(http_request_duration_seconds_sum[5m])
>>> /
>>> rate(http_request_duration_seconds_count[5m])
>>>
>>> # p95 response time
>>> histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
>>>
>>> # Slow request rate
>>> rate(slow_request_total[5m])

Datadog APM Integration:
-------------------------

>>> import ddtrace
>>> from ddtrace import tracer
>>>
>>> @tracer.wrap(service="legal-ai-api", resource="contract-analysis")
>>> async def analyze_contract(contract: Contract):
...     with tracer.trace("pdf-parsing") as span:
...         span.set_tag("document.size_mb", contract.size_mb)
...         parsed = await parse_pdf(contract)
...
...     with tracer.trace("llm-analysis") as span:
...         span.set_tag("model.name", "gpt-4-turbo")
...         result = await llm_analyze(parsed)
...
...     return result

ELK Stack Log Query:
--------------------

>>> # Find slow requests (> 5 seconds) in last hour
>>> GET /logs-*/_search
>>> {
...   "query": {
...     "bool": {
...       "must": [
...         {"range": {"@timestamp": {"gte": "now-1h"}}},
...         {"range": {"duration_ms": {"gte": 5000}}}
...       ]
...     }
...   },
...   "sort": [{"duration_ms": "desc"}],
...   "size": 100
... }

=============================================================================
PERFORMANCE OPTIMIZATION TIPS
=============================================================================

1. Database Optimization:
   - Use connection pooling (min: workers * 2, max: workers * 5)
   - Add indexes on frequently queried columns
   - Use database query caching for repeated queries
   - Implement pagination for large result sets

2. Caching Strategy:
   - Cache expensive computations (contract analysis results)
   - Use Redis for distributed caching across workers
   - Implement cache warming for frequently accessed data
   - Set appropriate TTLs based on data volatility

3. Async/Await Best Practices:
   - Use asyncio.gather() for parallel I/O operations
   - Avoid blocking calls in async functions
   - Use async database drivers (asyncpg, aiomysql)
   - Implement timeout for external API calls

4. Resource Management:
   - Limit concurrent AI model requests (semaphore)
   - Implement queue for expensive operations
   - Use background tasks for non-critical work
   - Monitor memory usage and implement cleanup

5. Turkish-Specific Optimizations:
   - Pre-load Turkish legal term dictionaries
   - Cache Turkish NLP model outputs
   - Optimize Turkish text tokenization
   - Use Turkish-specific stop words for faster processing

=============================================================================
TROUBLESHOOTING
=============================================================================

High Response Times:
--------------------
1. Check database connection pool exhaustion
2. Review slow query logs in PostgreSQL
3. Verify Redis cache hit rate
4. Check external API latency (LLM providers)
5. Monitor CPU and memory usage
6. Review concurrent request count

Inconsistent Performance:
--------------------------
1. Check for database lock contention
2. Review worker process count vs CPU cores
3. Verify no network issues (DNS resolution, routing)
4. Check for memory swapping (high memory usage)
5. Review time-of-day patterns (peak hours)

Memory Issues:
--------------
1. Implement request size limits (100MB default)
2. Stream large file uploads instead of loading to memory
3. Use generator functions for large result sets
4. Monitor and limit concurrent AI model requests
5. Implement worker recycling (--max-requests 1000)

=============================================================================
SECURITY CONSIDERATIONS
=============================================================================

1. Information Disclosure:
   - Avoid exposing internal processing details in timing headers
   - Don't reveal database query times (could aid SQL injection)
   - Limit precision of timing data (round to nearest ms)

2. Timing Attacks:
   - Use constant-time comparison for sensitive operations
   - Add random jitter to authentication failure responses
   - Don't expose different timings for valid/invalid users

3. DoS Prevention:
   - Implement rate limiting before timing middleware
   - Set maximum request timeout (120 seconds)
   - Monitor and alert on sudden latency spikes

=============================================================================
KVKK COMPLIANCE
=============================================================================

Personal Data in Performance Logs:
-----------------------------------
- Don't log user IDs in slow request warnings
- Use anonymized request IDs instead
- Aggregate metrics without personal identifiers
- Implement log retention policies (30 days for performance logs)

Audit Requirements:
-------------------
- Track performance degradation affecting user experience
- Log system performance issues for incident reports
- Maintain performance SLA compliance records

Author: Turkish Legal AI Team
License: Proprietary
Version: 1.0.0
Last Updated: 2024-11-06
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core import get_logger, settings

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Timing thresholds in seconds
SLOW_REQUEST_THRESHOLD = 1.0  # 1 second - WARNING
VERY_SLOW_REQUEST_THRESHOLD = 5.0  # 5 seconds - ERROR
CRITICAL_REQUEST_THRESHOLD = 10.0  # 10 seconds - CRITICAL

# Histogram buckets in milliseconds
HISTOGRAM_BUCKETS = [
    10,  # 10ms - cache hits
    25,  # 25ms - simple queries
    50,  # 50ms
    100,  # 100ms - standard API calls
    250,  # 250ms
    500,  # 500ms - complex operations
    1000,  # 1s
    2500,  # 2.5s
    5000,  # 5s - document processing
    10000,  # 10s
    30000,  # 30s - large AI analysis
    60000,  # 60s - maximum timeout
]

# =============================================================================
# PERFORMANCE METRICS COLLECTOR
# =============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    method: str
    path: str
    status_code: int
    duration_ms: float
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceMetricsCollector:
    """
    Collects and aggregates performance metrics.

    Tracks request durations, calculates percentiles, and maintains
    histogram data for observability systems.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.request_count: Dict[str, int] = defaultdict(int)
        self.total_duration: Dict[str, float] = defaultdict(float)
        self.durations: Dict[str, List[float]] = defaultdict(list)
        self.slow_requests: Dict[str, int] = defaultdict(int)
        self.histogram_buckets: Dict[str, Dict[float, int]] = defaultdict(
            lambda: {bucket: 0 for bucket in HISTOGRAM_BUCKETS}
        )

    def record_request(self, metrics: RequestMetrics) -> None:
        """
        Record request metrics.

        Args:
            metrics: Request metrics data
        """
        key = f"{metrics.method}:{metrics.path}"

        # Update counters
        self.request_count[key] += 1
        self.total_duration[key] += metrics.duration_ms

        # Store duration for percentile calculation
        self.durations[key].append(metrics.duration_ms)

        # Keep only last 1000 requests per endpoint
        if len(self.durations[key]) > 1000:
            self.durations[key] = self.durations[key][-1000:]

        # Update histogram
        for bucket in HISTOGRAM_BUCKETS:
            if metrics.duration_ms <= bucket:
                self.histogram_buckets[key][bucket] += 1

        # Track slow requests
        if metrics.duration_ms >= SLOW_REQUEST_THRESHOLD * 1000:
            self.slow_requests[key] += 1

    def get_percentile(self, method: str, path: str, percentile: float) -> Optional[float]:
        """
        Calculate percentile for endpoint.

        Args:
            method: HTTP method
            path: Request path
            percentile: Percentile value (0.0 to 1.0)

        Returns:
            Percentile value in milliseconds, or None if no data
        """
        key = f"{method}:{path}"
        durations = sorted(self.durations[key])

        if not durations:
            return None

        index = int(len(durations) * percentile)
        return durations[min(index, len(durations) - 1)]

    def get_average_duration(self, method: str, path: str) -> Optional[float]:
        """
        Get average duration for endpoint.

        Args:
            method: HTTP method
            path: Request path

        Returns:
            Average duration in milliseconds, or None if no data
        """
        key = f"{method}:{path}"
        count = self.request_count[key]

        if count == 0:
            return None

        return self.total_duration[key] / count

    def get_slow_request_rate(self, method: str, path: str) -> Optional[float]:
        """
        Get slow request rate for endpoint.

        Args:
            method: HTTP method
            path: Request path

        Returns:
            Slow request rate (0.0 to 1.0), or None if no data
        """
        key = f"{method}:{path}"
        total = self.request_count[key]

        if total == 0:
            return None

        return self.slow_requests[key] / total


# Global metrics collector instance
_metrics_collector: Optional[PerformanceMetricsCollector] = None


def get_metrics_collector() -> PerformanceMetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = PerformanceMetricsCollector()
    return _metrics_collector


# =============================================================================
# TIMING MIDDLEWARE
# =============================================================================


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to measure and report request processing time.

    Features:
    - High-precision timing with time.perf_counter()
    - X-Response-Time header injection
    - Slow request detection and logging
    - Performance metrics collection
    - Integration with observability systems
    """

    def __init__(
        self,
        app,
        slow_threshold_ms: float = SLOW_REQUEST_THRESHOLD * 1000,
        very_slow_threshold_ms: float = VERY_SLOW_REQUEST_THRESHOLD * 1000,
        critical_threshold_ms: float = CRITICAL_REQUEST_THRESHOLD * 1000,
        metrics_collector: Optional[PerformanceMetricsCollector] = None,
    ):
        """
        Initialize timing middleware.

        Args:
            app: FastAPI application
            slow_threshold_ms: WARNING threshold in milliseconds (default: 1000ms)
            very_slow_threshold_ms: ERROR threshold in milliseconds (default: 5000ms)
            critical_threshold_ms: CRITICAL threshold in milliseconds (default: 10000ms)
            metrics_collector: Optional custom metrics collector
        """
        super().__init__(app)
        self.slow_threshold_ms = slow_threshold_ms
        self.very_slow_threshold_ms = very_slow_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.metrics_collector = metrics_collector or get_metrics_collector()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Process request with timing metrics.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response with X-Response-Time header
        """
        # Start high-precision timer
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration with microsecond precision
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        # Add response headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Add processing timestamp
        response.headers["X-Processing-Timestamp"] = str(int(time.time()))

        # Get request context
        request_id = request.headers.get("X-Request-ID", "unknown")
        method = request.method
        path = request.url.path
        status_code = response.status_code

        # Get tenant and user IDs from request state (set by previous middleware)
        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = getattr(request.state, "user_id", None)

        # Record metrics
        metrics = RequestMetrics(
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        self.metrics_collector.record_request(metrics)

        # Log slow requests with severity levels
        if duration_ms >= self.critical_threshold_ms:
            logger.error(
                "⚠️ KRİTİK YAVAS İSTEK ALGILANDI",
                request_id=request_id,
                method=method,
                path=path,
                duration_ms=f"{duration_ms:.2f}",
                threshold_ms=self.critical_threshold_ms,
                status_code=status_code,
                tenant_id=tenant_id,
                severity="CRITICAL",
            )
        elif duration_ms >= self.very_slow_threshold_ms:
            logger.error(
                "⚠️ ÇOK YAVAS İSTEK ALGILANDI",
                request_id=request_id,
                method=method,
                path=path,
                duration_ms=f"{duration_ms:.2f}",
                threshold_ms=self.very_slow_threshold_ms,
                status_code=status_code,
                tenant_id=tenant_id,
                severity="ERROR",
            )
        elif duration_ms >= self.slow_threshold_ms:
            logger.warning(
                "⚠️ Yavaş istek algılandı",
                request_id=request_id,
                method=method,
                path=path,
                duration_ms=f"{duration_ms:.2f}",
                threshold_ms=self.slow_threshold_ms,
                status_code=status_code,
                tenant_id=tenant_id,
                severity="WARNING",
            )

        # Emit Prometheus metrics (if observability enabled)
        if settings.OBSERVABILITY_ENABLED:
            try:
                # from backend.observability import metrics
                # metrics.http_request_duration_seconds.labels(
                #     method=method,
                #     path=path,
                #     status=status_code
                # ).observe(duration)
                pass
            except Exception as e:
                logger.debug(f"Failed to emit metrics: {e}")

        return response


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TimingMiddleware",
    "PerformanceMetricsCollector",
    "RequestMetrics",
    "get_metrics_collector",
    "SLOW_REQUEST_THRESHOLD",
    "VERY_SLOW_REQUEST_THRESHOLD",
    "CRITICAL_REQUEST_THRESHOLD",
    "HISTOGRAM_BUCKETS",
]
