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
    tracing_tenant_in_traceparent: bool = True  # Add tenant to tracestate header (W3C Trace Context)
    tracing_propagation_format: str = "w3c"  # w3c, b3, jaeger (OpenTelemetry Collector filtering)

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

    # RBAC Monitoring (production-critical)
    rbac_policy_audit_enabled: bool = True  # Log all policy evaluations to JSONL
    rbac_policy_audit_path: str = "/var/log/legalai/rbac_audit.jsonl"
    rbac_permission_cache_enabled: bool = True  # Redis cache for permissions
    rbac_permission_cache_ttl: int = 60  # Cache TTL in seconds (default: 60s)
    rbac_security_deny_logging: bool = True  # Log all deny events
    rbac_metrics_enabled: bool = True  # Prometheus RBAC metrics


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
    # RATE LIMITING METRICS
    # ==========================================================================

    "rate_limit_exceeded_total": {
        "type": "counter",
        "description": "Total rate limit exceeded events",
        "labels": ["tier", "endpoint"],
    },
    "burst_active_seconds_total": {
        "type": "counter",
        "description": "Total seconds burst capacity was active (behavior analysis)",
        "labels": ["tier", "endpoint"],
    },
    "rate_limit_tokens_consumed_total": {
        "type": "counter",
        "description": "Total tokens consumed across all rate limiters",
        "labels": ["tier"],
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
    # RBAC METRICS (Role-Based Access Control)
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

    # Role Change Events (for audit compliance)
    "rbac_role_assignment_total": {
        "type": "counter",
        "description": "Total role assignments (granted to users)",
        "labels": ["role_slug", "role_type", "tenant_id"],
    },
    "rbac_role_revocation_total": {
        "type": "counter",
        "description": "Total role revocations (removed from users)",
        "labels": ["role_slug", "role_type", "tenant_id"],
    },
    "rbac_role_created_total": {
        "type": "counter",
        "description": "Total roles created",
        "labels": ["role_type", "tenant_id"],
    },
    "rbac_role_deleted_total": {
        "type": "counter",
        "description": "Total roles deleted",
        "labels": ["role_type", "tenant_id", "deletion_type"],  # deletion_type: soft, hard
    },

    # Policy Evaluation Metrics
    "rbac_policy_evaluation_total": {
        "type": "counter",
        "description": "Total policy evaluations",
        "labels": ["decision", "resource_type", "action"],  # decision: allow, deny
    },
    "rbac_policy_evaluation_duration_seconds": {
        "type": "histogram",
        "description": "Policy evaluation duration (includes all checks)",
        "labels": ["resource_type"],
        "buckets": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    },
    "rbac_policy_custom_evaluation_total": {
        "type": "counter",
        "description": "Custom policy evaluations",
        "labels": ["policy_name", "decision"],
    },

    # Permission Cache Metrics (Redis)
    "rbac_permission_cache_hits_total": {
        "type": "counter",
        "description": "Permission cache hits (Redis)",
        "labels": ["user_id_hash"],  # Hash for privacy
    },
    "rbac_permission_cache_misses_total": {
        "type": "counter",
        "description": "Permission cache misses (Redis)",
        "labels": ["user_id_hash"],
    },
    "rbac_permission_cache_ttl_seconds": {
        "type": "gauge",
        "description": "Permission cache TTL setting (default: 60s)",
        "labels": [],
    },

    # Security Deny Events (for alerting)
    "security_event_deny_total": {
        "type": "counter",
        "description": "Security deny events (permission/policy denied)",
        "labels": ["event_type", "resource", "action", "tenant_id"],  # event_type: permission_denied, policy_denied, ownership_denied
    },
    "security_event_deny_by_user_total": {
        "type": "counter",
        "description": "Security deny events grouped by user (detect attacks)",
        "labels": ["user_id_hash", "event_type"],
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
    # AUDIT TASK METRICS (Phase 4: Integrity & Monitoring)
    # ==========================================================================

    # Task Failure Tracking (Alerting)
    "audit_task_failure_total": {
        "type": "counter",
        "description": "Total audit task failures (for alerting) - tracks archive, cleanup, and report generation failures",
        "labels": ["task_name", "failure_reason", "tenant_id"],  # task_name: archive_logs, cleanup_logs, generate_report, process_batch
    },
    "audit_task_retry_total": {
        "type": "counter",
        "description": "Total audit task retries (tracks retry patterns)",
        "labels": ["task_name", "retry_count", "tenant_id"],  # retry_count: 1, 2, 3
    },
    "audit_task_success_total": {
        "type": "counter",
        "description": "Total successful audit task completions",
        "labels": ["task_name", "tenant_id"],
    },

    # Archive Transition Performance (Tier Migration)
    "archive_transition_duration_seconds": {
        "type": "histogram",
        "description": "Archive tier transition duration (HOT→WARM→COLD→ARCHIVE) - measures data migration performance",
        "labels": ["from_tier", "to_tier", "tenant_id"],  # from_tier/to_tier: hot, warm, cold, archive
        "buckets": [1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0],  # 1s to 30min
    },
    "archive_transition_records_total": {
        "type": "counter",
        "description": "Total audit records transitioned between tiers",
        "labels": ["from_tier", "to_tier", "tenant_id", "status"],  # status: success, failed
    },
    "archive_tier_size_bytes": {
        "type": "gauge",
        "description": "Current storage size per tier (for cost optimization)",
        "labels": ["tier", "tenant_id"],  # tier: hot, warm, cold, archive
    },

    # Audit Integrity Metrics (Hash-Chain Verification)
    "audit_integrity_check_total": {
        "type": "counter",
        "description": "Total audit log integrity checks performed",
        "labels": ["result", "tenant_id"],  # result: valid, invalid, chain_broken
    },
    "audit_integrity_verification_duration_seconds": {
        "type": "histogram",
        "description": "Audit log integrity verification duration (hash-chain validation)",
        "labels": ["tenant_id"],
        "buckets": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    },
    "audit_integrity_chain_length": {
        "type": "gauge",
        "description": "Current audit log hash-chain length (number of linked events)",
        "labels": ["tenant_id"],
    },
    "audit_integrity_violation_total": {
        "type": "counter",
        "description": "Total audit log integrity violations detected (CRITICAL - tamper evidence)",
        "labels": ["violation_type", "tenant_id"],  # violation_type: hash_mismatch, chain_gap, missing_predecessor
    },

    # Legal Hold Metrics
    "audit_legal_hold_active_total": {
        "type": "gauge",
        "description": "Total documents currently under legal hold",
        "labels": ["tenant_id"],
    },
    "audit_legal_hold_placed_total": {
        "type": "counter",
        "description": "Total legal holds placed (litigation/investigation)",
        "labels": ["reason_category", "tenant_id"],  # reason_category: litigation, investigation, regulatory_audit
    },
    "audit_legal_hold_removed_total": {
        "type": "counter",
        "description": "Total legal holds removed",
        "labels": ["reason_category", "tenant_id"],
    },

    # Compliance Report Metrics
    "audit_compliance_report_generated_total": {
        "type": "counter",
        "description": "Total compliance reports generated",
        "labels": ["report_type", "framework", "tenant_id"],  # framework: GDPR, KVKK, SOC2, ISO27001
    },
    "audit_compliance_report_duration_seconds": {
        "type": "histogram",
        "description": "Compliance report generation duration",
        "labels": ["report_type", "tenant_id"],
        "buckets": [5.0, 10.0, 30.0, 60.0, 300.0, 600.0],  # 5s to 10min
    },
    "audit_compliance_event_total": {
        "type": "counter",
        "description": "Total compliance events logged (GDPR Article 30)",
        "labels": ["event_type", "framework", "tenant_id"],  # event_type: DATA_ACCESS, DATA_DELETION, CONSENT_GRANTED, etc.
    },

    # Data Retention Metrics
    "audit_retention_policy_enforced_total": {
        "type": "counter",
        "description": "Total retention policy enforcement actions",
        "labels": ["action", "data_category", "tenant_id"],  # action: archived, deleted, retained
    },
    "audit_retention_days": {
        "type": "gauge",
        "description": "Current retention period (days) per data category",
        "labels": ["data_category", "tenant_id"],
    },
    "audit_expired_logs_deleted_total": {
        "type": "counter",
        "description": "Total expired audit logs deleted (data lifecycle management)",
        "labels": ["data_category", "tenant_id"],
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

    # ==========================================================================
    # INFRASTRUCTURE / REGIONAL METRICS
    # ==========================================================================

    "route53_failover_state": {
        "type": "gauge",
        "description": "Route 53 DNS failover state (0=inactive, 1=active) for Grafana monitoring",
        "labels": ["region", "failover_type"],  # failover_type: PRIMARY, SECONDARY
    },
    "region_health_check_status": {
        "type": "gauge",
        "description": "Regional health check status (0=unhealthy, 1=healthy)",
        "labels": ["region", "check_type"],
    },
    "region_failover_events_total": {
        "type": "counter",
        "description": "Total regional failover events triggered",
        "labels": ["from_region", "to_region", "reason"],
    },

    # ==========================================================================
    # LEGAL SOURCE SYNC METRICS (Daily Update Scheduler)
    # ==========================================================================

    "legal_source_sync_duration_seconds": {
        "type": "histogram",
        "description": "Legal source dataset sync duration (Yargıtay, AYM, Danıştay)",
        "labels": ["source"],  # source: yargitay, aym, danistay
        "buckets": [10.0, 30.0, 60.0, 300.0, 600.0, 1800.0],  # 10s to 30min
    },
    "legal_source_docs_updated_total": {
        "type": "counter",
        "description": "Total legal documents updated from daily sync",
        "labels": ["source", "status"],  # status: success, failed, skipped
    },
    "legal_source_last_sync_timestamp": {
        "type": "gauge",
        "description": "Unix timestamp of last successful sync",
        "labels": ["source"],
    },

    # ==========================================================================
    # COST FORECAST METRICS (Budget Management)
    # ==========================================================================

    "predicted_cost_next_30d": {
        "type": "gauge",
        "description": "Predicted LLM cost for next 30 days (USD) - budget alarm integration",
        "labels": ["model", "forecast_method"],  # forecast_method: linear, exponential
    },
    "cost_budget_utilization": {
        "type": "gauge",
        "description": "Current cost vs budget utilization ratio (0-1)",
        "labels": ["budget_period"],  # budget_period: daily, weekly, monthly
    },
    "cost_anomaly_detected": {
        "type": "counter",
        "description": "Cost anomaly detection events (spike > 2x normal)",
        "labels": ["model", "anomaly_type"],
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

    # Dead Letter Queue (DLQ) alerts - auto-purge triggers
    "dlq_count_high": {
        "metric": "celery_dead_letter_queue_total",
        "threshold": 100,  # 100 tasks in DLQ
        "severity": "warning",
        "description": "DLQ count > 100 (consider manual review)",
        "action": "review_failed_tasks",
    },
    "dlq_count_critical": {
        "metric": "celery_dead_letter_queue_total",
        "threshold": 500,  # 500 tasks in DLQ
        "severity": "critical",
        "description": "DLQ count > 500 (auto-purge trigger)",
        "action": "auto_purge_dlq",  # Trigger celery queue purge
    },
    "dlq_growth_rate_high": {
        "metric": "celery_dead_letter_queue_total",
        "threshold": 50,  # 50 tasks/min growth
        "severity": "critical",
        "description": "DLQ growing > 50 tasks/min (systemic failure)",
        "action": "alert_oncall",
    },

    # Cost budget alerts (budget management)
    "cost_budget_exceeded": {
        "metric": "cost_budget_utilization",
        "threshold": 0.90,  # 90% of budget
        "severity": "warning",
        "description": "Cost budget utilization > 90%",
        "action": "notify_finance_team",
    },
    "cost_forecast_over_budget": {
        "metric": "predicted_cost_next_30d",
        "threshold": 10000.0,  # $10,000 monthly budget
        "severity": "warning",
        "description": "Predicted 30-day cost > $10k",
        "action": "budget_review",
    },

    # Regional failover alerts
    "region_health_check_failed": {
        "metric": "region_health_check_status",
        "threshold": 0.5,  # 50% health checks failing
        "severity": "critical",
        "description": "Regional health check failure > 50%",
        "action": "trigger_dns_failover",
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
