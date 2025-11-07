"""
Prometheus Metrics Endpoint for Legal AI System.

Harvey/Legora %100 parite: Production observability.

Exposes Prometheus-compatible metrics for:
- Adapter health (requests, errors, latency, cache hit ratio)
- Sync engine state (processed, failed, DLQ size)
- System health (uptime, memory, CPU)

Metrics Format: Prometheus text-based exposition format
https://prometheus.io/docs/instrumenting/exposition_formats/

Integration:
    Prometheus scrapes: GET /metrics every 15s
    Grafana dashboards: Visualize metrics
    AlertManager: Alert on SLO violations

Example metrics output:
    # HELP adapter_request_total Total requests per adapter
    # TYPE adapter_request_total counter
    adapter_request_total{adapter="resmi_gazete"} 1234
    adapter_request_total{adapter="yargitay"} 5678

    # HELP adapter_error_rate Error rate per adapter
    # TYPE adapter_error_rate gauge
    adapter_error_rate{adapter="resmi_gazete"} 0.0012

SLO Targets:
    - adapter_error_rate < 0.005 (0.5%)
    - adapter_cache_hit_ratio > 0.80 (80%)
    - adapter_latency_p95 < 3000ms
"""

from fastapi import APIRouter, Response
from datetime import datetime, timezone
import time
import psutil

from backend.parsers.adapters.adapter_factory import get_factory
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["observability"])


# =============================================================================
# PROMETHEUS METRICS HELPERS
# =============================================================================


def format_prometheus_metric(
    name: str,
    value: float,
    metric_type: str = "gauge",
    help_text: str = "",
    labels: dict = None,
) -> str:
    """
    Format single metric in Prometheus exposition format.

    Args:
        name: Metric name (e.g., "adapter_request_total")
        value: Metric value
        metric_type: "counter" or "gauge"
        help_text: Help description
        labels: Label dict (e.g., {"adapter": "resmi_gazete"})

    Returns:
        Prometheus-formatted metric string

    Example:
        >>> format_prometheus_metric(
        ...     "adapter_request_total",
        ...     1234,
        ...     "counter",
        ...     "Total requests",
        ...     {"adapter": "resmi_gazete"}
        ... )
        '# HELP adapter_request_total Total requests\\n
         # TYPE adapter_request_total counter\\n
         adapter_request_total{adapter="resmi_gazete"} 1234\\n'
    """
    lines = []

    # Add HELP and TYPE only once per metric name
    if help_text:
        lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")

    # Format labels
    if labels:
        label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
        lines.append(f"{name}{{{label_str}}} {value}")
    else:
        lines.append(f"{name} {value}")

    return "\n".join(lines) + "\n"


def collect_adapter_metrics() -> str:
    """
    Collect metrics from all adapters.

    Returns:
        Prometheus-formatted metrics string

    Metrics:
        - adapter_request_total: Total requests
        - adapter_error_total: Total errors
        - adapter_error_rate: Error rate (errors/requests)
        - adapter_cache_hit_total: Cache hits
        - adapter_cache_hit_ratio: Cache hit ratio
        - adapter_circuit_state: Circuit breaker state (0=closed, 1=open)
    """
    factory = get_factory()
    health_matrix = factory.get_health_matrix()

    metrics = []

    for adapter_name, health in health_matrix.items():
        labels = {"adapter": adapter_name}

        # Request metrics
        metrics.append(format_prometheus_metric(
            "adapter_request_total",
            health["requests_total"],
            "counter",
            "Total adapter requests",
            labels
        ))

        # Error metrics
        metrics.append(format_prometheus_metric(
            "adapter_error_total",
            health["errors_total"],
            "counter",
            "Total adapter errors",
            labels
        ))

        # Error rate (gauge)
        metrics.append(format_prometheus_metric(
            "adapter_error_rate",
            health["error_rate"],
            "gauge",
            "Adapter error rate (SLO target: <0.005)",
            labels
        ))

        # Cache metrics
        metrics.append(format_prometheus_metric(
            "adapter_cache_hit_total",
            health["cache_hits"],
            "counter",
            "Total cache hits",
            labels
        ))

        metrics.append(format_prometheus_metric(
            "adapter_cache_hit_ratio",
            health["cache_hit_ratio"],
            "gauge",
            "Cache hit ratio (SLO target: >0.80)",
            labels
        ))

        # Circuit breaker state (0=closed, 1=open, 0.5=half_open)
        circuit_value = {
            "closed": 0.0,
            "open": 1.0,
            "half_open": 0.5,
        }.get(health["circuit_state"], 0.0)

        metrics.append(format_prometheus_metric(
            "adapter_circuit_state",
            circuit_value,
            "gauge",
            "Circuit breaker state (0=closed, 1=open)",
            labels
        ))

    return "".join(metrics)


def collect_system_metrics() -> str:
    """
    Collect system-level metrics.

    Returns:
        Prometheus-formatted system metrics

    Metrics:
        - process_cpu_percent: CPU usage
        - process_memory_bytes: Memory usage
        - process_uptime_seconds: Uptime
    """
    metrics = []

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    metrics.append(format_prometheus_metric(
        "process_cpu_percent",
        cpu_percent,
        "gauge",
        "CPU usage percentage"
    ))

    # Memory usage
    process = psutil.Process()
    memory_bytes = process.memory_info().rss
    metrics.append(format_prometheus_metric(
        "process_memory_bytes",
        memory_bytes,
        "gauge",
        "Memory usage in bytes"
    ))

    # Uptime (would need to track start time)
    # For now, use placeholder
    metrics.append(format_prometheus_metric(
        "process_uptime_seconds",
        0,  # TODO: Track actual uptime
        "gauge",
        "Process uptime in seconds"
    ))

    return "".join(metrics)


# =============================================================================
# ROUTES
# =============================================================================


@router.get("")
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Harvey/Legora %100 parite: Production observability.

    Returns metrics in Prometheus text-based exposition format for:
    - Adapter health (request rate, error rate, cache efficiency)
    - System health (CPU, memory, uptime)

    SLO Targets:
        - adapter_error_rate < 0.005 (0.5%)
        - adapter_cache_hit_ratio > 0.80 (80%)

    Grafana Alert Rules:
        - Alert if adapter_error_rate > 0.01 for 5m
        - Alert if adapter_cache_hit_ratio < 0.60 for 10m
        - Alert if adapter_circuit_state == 1 (open)

    Example:
        ```bash
        curl http://localhost:8000/metrics
        ```

    Response:
        ```
        # HELP adapter_request_total Total adapter requests
        # TYPE adapter_request_total counter
        adapter_request_total{adapter="resmi_gazete"} 1234
        adapter_request_total{adapter="yargitay"} 5678

        # HELP adapter_error_rate Adapter error rate
        # TYPE adapter_error_rate gauge
        adapter_error_rate{adapter="resmi_gazete"} 0.0012
        ```
    """
    try:
        # Collect all metrics
        adapter_metrics = collect_adapter_metrics()
        system_metrics = collect_system_metrics()

        # Combine
        all_metrics = adapter_metrics + system_metrics

        return Response(
            content=all_metrics,
            media_type="text/plain; version=0.0.4"  # Prometheus format
        )

    except Exception as e:
        logger.error(
            "Failed to collect metrics",
            error=str(e)
        )
        # Return minimal error metric
        error_metric = format_prometheus_metric(
            "metrics_collection_error",
            1.0,
            "gauge",
            "Metrics collection failed"
        )
        return Response(
            content=error_metric,
            media_type="text/plain"
        )


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    Returns:
        JSON with status and timestamp

    Example:
        ```bash
        curl http://localhost:8000/metrics/health
        ```

    Response:
        ```json
        {
            "status": "healthy",
            "timestamp": "2024-11-07T12:00:00Z"
        }
        ```
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/adapters")
async def get_adapter_health():
    """
    Detailed adapter health matrix.

    Returns:
        JSON with detailed health for all adapters

    Example:
        ```bash
        curl http://localhost:8000/metrics/adapters
        ```

    Response:
        ```json
        {
            "resmi_gazete": {
                "status": "healthy",
                "circuit_state": "closed",
                "error_rate": 0.0012,
                "requests_total": 1234,
                "cache_hit_ratio": 0.8523
            }
        }
        ```
    """
    factory = get_factory()
    return factory.get_health_matrix()
