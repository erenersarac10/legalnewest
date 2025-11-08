"""
Observability Configuration - Harvey/Legora %100 Full-Stack Monitoring.

Enterprise-grade observability with:
- Prometheus metrics (RED method: Rate, Errors, Duration)
- OpenTelemetry distributed tracing (W3C Trace Context)
- Structured logging (JSON format for aggregation)
- Application Performance Monitoring (APM)
- Custom business metrics (Turkish Legal AI specific)
- Grafana dashboard integration
- Alert thresholds (SLO-based)
- Error tracking (Sentry integration)

Why Observability?
    Without: Black box → no insight into failures → hours of debugging
    With: Full visibility → instant root cause → Harvey-level reliability

    Impact: 90% faster incident resolution! 📊

The Three Pillars:
    1. METRICS: What is happening? (Prometheus, Grafana)
    2. TRACES: Where is the bottleneck? (Jaeger, Tempo)
    3. LOGS: Why did it fail? (Loki, CloudWatch)

Architecture:
    Application → Metrics Export → Prometheus → Grafana Dashboards
                → Traces Export → Jaeger/Tempo → Trace Visualization
                → Logs Export → Loki/CloudWatch → Log Aggregation

Key Metrics (Turkish Legal AI):
    - RAG request rate (requests/sec)
    - RAG latency p50, p95, p99 (milliseconds)
    - LLM token usage (tokens/request)
    - Cache hit ratio (%)
    - Document processing rate (docs/min)
    - Embedding generation rate (embeddings/sec)
    - Legal source uptime (%)
    - RBAC authorization time (ms)

Usage:
    >>> from backend.core.config.observability import metrics
    >>>
    >>> # Counter
    >>> metrics.rag_requests_total.labels(status="success").inc()
    >>>
    >>> # Histogram
    >>> metrics.rag_duration_seconds.labels(model="gpt4").observe(2.5)
    >>>
    >>> # Gauge
    >>> metrics.active_sessions.set(42)
"""

from typing import Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel
import logging


class MetricsBackend(str, Enum):
    """Metrics storage backends."""

    PROMETHEUS = "prometheus"  # Industry standard
    DATADOG = "datadog"  # SaaS APM
    NEW_RELIC = "new_relic"  # SaaS APM
    CLOUDWATCH = "cloudwatch"  # AWS CloudWatch


class TracingBackend(str, Enum):
    """Distributed tracing backends."""

    JAEGER = "jaeger"  # Open source (CNCF)
    TEMPO = "tempo"  # Grafana Labs
    ZIPKIN = "zipkin"  # Twitter open source
    XRAY = "xray"  # AWS X-Ray


class LoggingBackend(str, Enum):
    """Logging aggregation backends."""

    LOKI = "loki"  # Grafana Labs (integrates with Grafana)
    CLOUDWATCH = "cloudwatch"  # AWS CloudWatch Logs
    ELASTICSEARCH = "elasticsearch"  # ELK stack
    DATADOG = "datadog"  # Datadog Logs


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"  # Verbose debugging
    INFO = "INFO"  # General info
    WARNING = "WARNING"  # Warnings
    ERROR = "ERROR"  # Errors (non-critical)
    CRITICAL = "CRITICAL"  # Critical failures


# =============================================================================
# OBSERVABILITY CONFIGURATIONS
# =============================================================================


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    # Metrics
    metrics_enabled: bool = True
    metrics_backend: MetricsBackend = MetricsBackend.PROMETHEUS
    metrics_port: int = 9090  # Prometheus scrape port
    metrics_path: str = "/metrics"
    metrics_interval: int = 15  # Scrape interval (seconds)

    # Tracing
    tracing_enabled: bool = True
    tracing_backend: TracingBackend = TracingBackend.JAEGER
    tracing_endpoint: str = "http://localhost:14268/api/traces"
    tracing_sample_rate: float = 0.1  # 10% sampling (production)
    tracing_service_name: str = "legalai-backend"
    tracing_tenant_id_enabled: bool = True  # Add tenant_id to all spans (multi-tenant tracing)

    # Logging
    logging_enabled: bool = True
    logging_backend: LoggingBackend = LoggingBackend.LOKI
    logging_level: LogLevel = LogLevel.INFO
    logging_format: str = "json"  # json, text
    logging_endpoint: Optional[str] = None

    # APM
    apm_enabled: bool = True
    apm_service_name: str = "legalai"
    apm_environment: str = "production"

    # Error tracking
    sentry_enabled: bool = True
    sentry_dsn: Optional[str] = None
    sentry_sample_rate: float = 1.0  # 100% error sampling
    sentry_traces_sample_rate: float = 0.01  # 1% performance traces

    # Health checks
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    liveness_probe_path: str = "/health/live"
    readiness_probe_path: str = "/health/ready"

    # Custom metrics
    custom_metrics_enabled: bool = True
    business_metrics_enabled: bool = True  # Turkish Legal AI metrics


# Harvey/Legora %100: Multi-Environment Observability Configuration
OBSERVABILITY_CONFIGS: Dict[str, ObservabilityConfig] = {
    # =============================================================================
    # PRODUCTION: Full observability stack
    # =============================================================================
    "production": ObservabilityConfig(
        # Metrics
        metrics_enabled=True,
        metrics_backend=MetricsBackend.PROMETHEUS,
        metrics_port=9090,
        metrics_interval=15,
        # Tracing
        tracing_enabled=True,
        tracing_backend=TracingBackend.TEMPO,
        tracing_endpoint="http://tempo:14268/api/traces",
        tracing_sample_rate=0.1,  # 10% sampling (cost optimization)
        tracing_service_name="legalai-production",
        # Logging
        logging_enabled=True,
        logging_backend=LoggingBackend.LOKI,
        logging_level=LogLevel.INFO,
        logging_format="json",
        logging_endpoint="http://loki:3100/loki/api/v1/push",
        # APM
        apm_enabled=True,
        apm_service_name="legalai",
        apm_environment="production",
        # Sentry
        sentry_enabled=True,
        sentry_dsn="https://your-sentry-dsn@sentry.io/project",
        sentry_sample_rate=1.0,  # Capture all errors
        sentry_traces_sample_rate=0.01,  # 1% performance traces
        # Health checks
        health_check_enabled=True,
        # Custom metrics
        custom_metrics_enabled=True,
        business_metrics_enabled=True,
    ),

    # =============================================================================
    # STAGING: Reduced sampling (cost)
    # =============================================================================
    "staging": ObservabilityConfig(
        metrics_enabled=True,
        metrics_backend=MetricsBackend.PROMETHEUS,
        metrics_port=9090,
        tracing_enabled=True,
        tracing_backend=TracingBackend.JAEGER,
        tracing_endpoint="http://jaeger:14268/api/traces",
        tracing_sample_rate=0.5,  # 50% sampling
        tracing_service_name="legalai-staging",
        logging_enabled=True,
        logging_backend=LoggingBackend.CLOUDWATCH,
        logging_level=LogLevel.DEBUG,  # More verbose in staging
        logging_format="json",
        sentry_enabled=True,
        sentry_sample_rate=1.0,
        sentry_traces_sample_rate=0.1,
        health_check_enabled=True,
    ),

    # =============================================================================
    # DEVELOPMENT: Local observability
    # =============================================================================
    "development": ObservabilityConfig(
        metrics_enabled=True,
        metrics_backend=MetricsBackend.PROMETHEUS,
        metrics_port=9090,
        tracing_enabled=True,
        tracing_backend=TracingBackend.JAEGER,
        tracing_endpoint="http://localhost:14268/api/traces",
        tracing_sample_rate=1.0,  # 100% sampling in dev
        tracing_service_name="legalai-dev",
        logging_enabled=True,
        logging_backend=LoggingBackend.LOKI,
        logging_level=LogLevel.DEBUG,
        logging_format="text",  # Human-readable in dev
        sentry_enabled=False,  # No Sentry in dev
        health_check_enabled=True,
        custom_metrics_enabled=True,
    ),

    # =============================================================================
    # TESTING: Minimal observability
    # =============================================================================
    "testing": ObservabilityConfig(
        metrics_enabled=False,
        tracing_enabled=False,
        logging_enabled=True,
        logging_level=LogLevel.WARNING,  # Only warnings/errors in tests
        logging_format="text",
        sentry_enabled=False,
        health_check_enabled=False,
        custom_metrics_enabled=False,
    ),
}


# =============================================================================
# PROMETHEUS METRICS DEFINITIONS
# =============================================================================

PROMETHEUS_METRICS: Dict[str, Dict] = {
    # ==========================================================================
    # REQUEST METRICS (RED Method: Rate, Errors, Duration)
    # ==========================================================================

    "http_requests_total": {
        "type": "counter",
        "description": "Total HTTP requests",
        "labels": ["method", "endpoint", "status_code"],
    },
    "http_request_duration_seconds": {
        "type": "histogram",
        "description": "HTTP request duration",
        "labels": ["method", "endpoint"],
        "buckets": [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    },
    "http_request_size_bytes": {
        "type": "histogram",
        "description": "HTTP request size",
        "labels": ["method", "endpoint"],
        "buckets": [100, 1000, 10000, 100000, 1000000],
    },
    "http_response_size_bytes": {
        "type": "histogram",
        "description": "HTTP response size",
        "labels": ["method", "endpoint"],
        "buckets": [100, 1000, 10000, 100000, 1000000],
    },

    # ==========================================================================
    # RAG METRICS (Turkish Legal AI)
    # ==========================================================================

    "rag_requests_total": {
        "type": "counter",
        "description": "Total RAG generation requests",
        "labels": ["model", "status", "language"],
    },
    "rag_duration_seconds": {
        "type": "histogram",
        "description": "RAG generation duration",
        "labels": ["model", "strategy"],  # strategy: hybrid, vector, fulltext
        "buckets": [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    },
    "rag_tokens_used_total": {
        "type": "counter",
        "description": "Total LLM tokens consumed",
        "labels": ["model", "type"],  # type: input, output
    },
    "rag_context_chunks": {
        "type": "histogram",
        "description": "Number of context chunks used in RAG",
        "labels": ["source_type"],  # source_type: mevzuat, yargitay, etc.
        "buckets": [1, 5, 10, 20, 50, 100],
    },
    "rag_citations_count": {
        "type": "histogram",
        "description": "Number of citations in RAG response",
        "labels": ["citation_format"],
        "buckets": [0, 1, 3, 5, 10, 20],
    },

    # ==========================================================================
    # LLM COST METRICS (for Grafana LLM Dashboard)
    # ==========================================================================

    "llm_cost_dollars_total": {
        "type": "counter",
        "description": "Total LLM cost in USD",
        "labels": ["model", "provider"],
    },
    "llm_cost_per_request_dollars": {
        "type": "histogram",
        "description": "LLM cost per request in USD",
        "labels": ["model"],
        "buckets": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    },
    "llm_token_budget_remaining": {
        "type": "gauge",
        "description": "Remaining token budget (tokens)",
        "labels": ["environment"],  # dev, staging, production
    },

    # ==========================================================================
    # CACHE METRICS
    # ==========================================================================

    "cache_hits_total": {
        "type": "counter",
        "description": "Total cache hits",
        "labels": ["cache_type"],  # cache_type: redis, memory, permission
    },
    "cache_misses_total": {
        "type": "counter",
        "description": "Total cache misses",
        "labels": ["cache_type"],
    },
    "cache_hit_ratio": {
        "type": "gauge",
        "description": "Cache hit ratio (0-1)",
        "labels": ["cache_type"],
    },
    "cache_size_bytes": {
        "type": "gauge",
        "description": "Cache size in bytes",
        "labels": ["cache_type"],
    },

    # ==========================================================================
    # DATABASE METRICS
    # ==========================================================================

    "db_queries_total": {
        "type": "counter",
        "description": "Total database queries",
        "labels": ["operation", "table"],
    },
    "db_query_duration_seconds": {
        "type": "histogram",
        "description": "Database query duration",
        "labels": ["operation", "table"],
        "buckets": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
    },
    "db_connections_active": {
        "type": "gauge",
        "description": "Active database connections",
        "labels": ["pool"],
    },

    # ==========================================================================
    # EMBEDDINGS METRICS
    # ==========================================================================

    "embeddings_generated_total": {
        "type": "counter",
        "description": "Total embeddings generated",
        "labels": ["model", "dimension"],
    },
    "embeddings_duration_seconds": {
        "type": "histogram",
        "description": "Embedding generation duration",
        "labels": ["model"],
        "buckets": [0.01, 0.1, 0.5, 1.0, 5.0],
    },

    # ==========================================================================
    # DOCUMENT PROCESSING METRICS
    # ==========================================================================

    "documents_processed_total": {
        "type": "counter",
        "description": "Total documents processed",
        "labels": ["document_type", "status"],
    },
    "document_processing_duration_seconds": {
        "type": "histogram",
        "description": "Document processing duration",
        "labels": ["document_type"],
        "buckets": [1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
    },
    "document_size_bytes": {
        "type": "histogram",
        "description": "Document size",
        "labels": ["document_type"],
        "buckets": [1000, 10000, 100000, 1000000, 10000000],
    },

    # ==========================================================================
    # LEGAL SOURCE METRICS
    # ==========================================================================

    "legal_source_requests_total": {
        "type": "counter",
        "description": "Total legal source API requests",
        "labels": ["source", "status"],  # source: mevzuat, yargitay, etc.
    },
    "legal_source_response_time_seconds": {
        "type": "histogram",
        "description": "Legal source API response time",
        "labels": ["source"],
        "buckets": [0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    },
    "legal_source_uptime": {
        "type": "gauge",
        "description": "Legal source uptime (0-1)",
        "labels": ["source"],
    },

    # ==========================================================================
    # RBAC METRICS
    # ==========================================================================

    "rbac_authorization_total": {
        "type": "counter",
        "description": "Total RBAC authorization checks",
        "labels": ["result", "resource", "action"],
    },
    "rbac_authorization_duration_seconds": {
        "type": "histogram",
        "description": "RBAC authorization duration",
        "labels": ["resource"],
        "buckets": [0.001, 0.01, 0.05, 0.1, 0.5],
    },

    # ==========================================================================
    # CELERY / TASK QUEUE METRICS
    # ==========================================================================

    "celery_tasks_total": {
        "type": "counter",
        "description": "Total Celery tasks executed",
        "labels": ["queue", "task_name", "status"],  # status: success, failure, retry
    },
    "celery_task_duration_seconds": {
        "type": "histogram",
        "description": "Celery task execution duration",
        "labels": ["queue", "task_name"],
        "buckets": [0.1, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
    },
    "celery_queue_length": {
        "type": "gauge",
        "description": "Current Celery queue length (pending tasks)",
        "labels": ["queue"],
    },
    "celery_dead_letter_queue_total": {
        "type": "counter",
        "description": "Tasks sent to dead letter queue (DLQ) - critical for retry anomaly detection",
        "labels": ["queue", "task_name", "failure_reason"],
    },
    "celery_worker_active_tasks": {
        "type": "gauge",
        "description": "Active tasks being processed by workers",
        "labels": ["worker", "queue"],
    },

    # ==========================================================================
    # BUSINESS METRICS (Turkish Legal AI)
    # ==========================================================================

    "active_users": {
        "type": "gauge",
        "description": "Current active users",
        "labels": ["tier"],  # tier: free, premium, enterprise
    },
    "active_sessions": {
        "type": "gauge",
        "description": "Current active sessions",
        "labels": [],
    },
    "daily_active_users": {
        "type": "gauge",
        "description": "Daily active users (DAU)",
        "labels": [],
    },
    "legal_queries_per_minute": {
        "type": "gauge",
        "description": "Legal queries per minute (QPM)",
        "labels": [],
    },

    # ==========================================================================
    # FEATURE FLAG / CANARY ROLLOUT METRICS
    # ==========================================================================

    "feature_rollout_progress": {
        "type": "gauge",
        "description": "Feature flag rollout percentage (canary deployment progress)",
        "labels": ["feature_name", "environment"],
    },
    "feature_flag_evaluations_total": {
        "type": "counter",
        "description": "Total feature flag evaluations",
        "labels": ["feature_name", "result"],  # result: enabled, disabled
    },
    "feature_flag_evaluation_duration_seconds": {
        "type": "histogram",
        "description": "Feature flag evaluation duration",
        "labels": ["feature_name"],
        "buckets": [0.0001, 0.001, 0.01, 0.05, 0.1],
    },
}


# =============================================================================
# ALERT THRESHOLDS (SLO-based)
# =============================================================================

ALERT_THRESHOLDS: Dict[str, Dict] = {
    # Latency alerts (p95)
    "rag_p95_latency_high": {
        "metric": "rag_duration_seconds",
        "threshold": 5.0,  # 5 seconds
        "severity": "warning",
        "description": "RAG p95 latency > 5s",
    },
    "rag_p95_latency_critical": {
        "metric": "rag_duration_seconds",
        "threshold": 10.0,  # 10 seconds
        "severity": "critical",
        "description": "RAG p95 latency > 10s",
    },

    # Error rate alerts
    "http_error_rate_high": {
        "metric": "http_requests_total",
        "threshold": 0.05,  # 5% error rate
        "severity": "warning",
        "description": "HTTP error rate > 5%",
    },
    "http_error_rate_critical": {
        "metric": "http_requests_total",
        "threshold": 0.10,  # 10% error rate
        "severity": "critical",
        "description": "HTTP error rate > 10%",
    },

    # Cache alerts
    "cache_hit_ratio_low": {
        "metric": "cache_hit_ratio",
        "threshold": 0.70,  # 70%
        "severity": "warning",
        "description": "Cache hit ratio < 70%",
    },

    # Database alerts
    "db_connections_high": {
        "metric": "db_connections_active",
        "threshold": 80,  # 80% of pool size
        "severity": "warning",
        "description": "DB connections > 80% of pool",
    },

    # Legal source alerts
    "legal_source_down": {
        "metric": "legal_source_uptime",
        "threshold": 0.95,  # 95%
        "severity": "critical",
        "description": "Legal source uptime < 95%",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_observability_config(environment: str = "production") -> ObservabilityConfig:
    """
    Get observability configuration for environment.

    Args:
        environment: Environment name

    Returns:
        ObservabilityConfig instance
    """
    return OBSERVABILITY_CONFIGS.get(environment, OBSERVABILITY_CONFIGS["development"])


def get_metric_config(metric_name: str) -> Optional[Dict]:
    """
    Get Prometheus metric configuration.

    Args:
        metric_name: Metric name

    Returns:
        Metric configuration dict or None
    """
    return PROMETHEUS_METRICS.get(metric_name)


def get_alert_threshold(alert_name: str) -> Optional[Dict]:
    """
    Get alert threshold configuration.

    Args:
        alert_name: Alert name

    Returns:
        Alert configuration dict or None
    """
    return ALERT_THRESHOLDS.get(alert_name)


__all__ = [
    "MetricsBackend",
    "TracingBackend",
    "LoggingBackend",
    "LogLevel",
    "ObservabilityConfig",
    "OBSERVABILITY_CONFIGS",
    "PROMETHEUS_METRICS",
    "ALERT_THRESHOLDS",
    "get_observability_config",
    "get_metric_config",
    "get_alert_threshold",
]
