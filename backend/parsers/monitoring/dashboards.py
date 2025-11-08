"""Parser Dashboards - Harvey/Legora CTO-Level Production-Grade
Grafana dashboard configurations for Turkish legal document parsing monitoring

Production Features:
- Grafana dashboard JSON generation
- Multi-dashboard layouts (Overview, Performance, Quality, Alerting)
- Time series graphs for parsing metrics
- Gauge panels for real-time status
- Stat panels for aggregated totals
- Heatmap panels for temporal patterns
- Table panels for recent activity
- Alert state visualizations
- Template variables for filtering
- Dynamic panel generation
- Color schemes and thresholds
- Panel positioning and sizing
- Dashboard links and navigation
- Annotation support
- Turkish legal document-specific metrics
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DASHBOARD PANEL DEFINITIONS
# ============================================================================

@dataclass
class Panel:
    """Grafana panel definition"""
    panel_id: int
    title: str
    panel_type: str  # graph, stat, gauge, table, heatmap, etc.
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]  # x, y, w, h
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Convert panel to Grafana JSON format

        Returns:
            Panel JSON dict
        """
        panel_json = {
            "id": self.panel_id,
            "title": self.title,
            "type": self.panel_type,
            "targets": self.targets,
            "gridPos": self.grid_pos,
            "options": self.options,
            "fieldConfig": self.field_config
        }

        return panel_json


@dataclass
class Dashboard:
    """Grafana dashboard definition"""
    uid: str
    title: str
    description: str
    panels: List[Panel] = field(default_factory=list)
    template_variables: List[Dict[str, Any]] = field(default_factory=list)
    time_from: str = "now-6h"
    time_to: str = "now"
    refresh: str = "30s"
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """Convert dashboard to Grafana JSON format

        Returns:
            Dashboard JSON dict
        """
        dashboard_json = {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "timezone": "browser",
            "schemaVersion": 38,
            "version": 1,
            "refresh": self.refresh,
            "time": {
                "from": self.time_from,
                "to": self.time_to
            },
            "templating": {
                "list": self.template_variables
            },
            "panels": [panel.to_json() for panel in self.panels],
            "editable": True,
            "graphTooltip": 1,  # Shared crosshair
            "links": []
        }

        return dashboard_json


# ============================================================================
# DASHBOARD BUILDER
# ============================================================================

class DashboardBuilder:
    """Builds Grafana dashboards for parser monitoring"""

    def __init__(self, datasource: str = "Prometheus"):
        """Initialize dashboard builder

        Args:
            datasource: Grafana datasource name
        """
        self.datasource = datasource
        logger.debug(f"Initialized DashboardBuilder (datasource={datasource})")

    def build_overview_dashboard(self) -> Dashboard:
        """Build overview dashboard

        Returns:
            Dashboard with high-level metrics
        """
        dashboard = Dashboard(
            uid="legal-parser-overview",
            title="Legal Parser - Overview",
            description="High-level overview of Turkish legal document parsing",
            tags=["legal-parser", "overview"],
            time_from="now-6h",
            refresh="30s"
        )

        # Template variables
        dashboard.template_variables = [
            self._create_document_type_variable(),
            self._create_source_variable()
        ]

        # Panel 1: Total documents processed (Stat)
        dashboard.panels.append(Panel(
            panel_id=1,
            title="Documents Processed (Total)",
            panel_type="stat",
            targets=[{
                "expr": "sum(legal_parser_documents_processed_total)",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "textMode": "value_and_name",
                "colorMode": "background"
            },
            field_config={
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "blue"}
                        ]
                    }
                }
            }
        ))

        # Panel 2: Success rate (Gauge)
        dashboard.panels.append(Panel(
            panel_id=2,
            title="Success Rate",
            panel_type="gauge",
            targets=[{
                "expr": (
                    "sum(rate(legal_parser_documents_processed_total{status='success'}[5m])) / "
                    "sum(rate(legal_parser_documents_processed_total[5m])) * 100"
                ),
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 90, "color": "yellow"},
                            {"value": 95, "color": "green"}
                        ]
                    }
                }
            }
        ))

        # Panel 3: Current parsing rate (Stat)
        dashboard.panels.append(Panel(
            panel_id=3,
            title="Parsing Rate (docs/sec)",
            panel_type="stat",
            targets=[{
                "expr": "sum(rate(legal_parser_documents_processed_total[5m]))",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "textMode": "value_and_name",
                "colorMode": "value"
            },
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "decimals": 2,
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 0.1, "color": "yellow"},
                            {"value": 1, "color": "green"}
                        ]
                    }
                }
            }
        ))

        # Panel 4: Queue depth (Gauge)
        dashboard.panels.append(Panel(
            panel_id=4,
            title="Queue Depth",
            panel_type="gauge",
            targets=[{
                "expr": 'legal_parser_queue_size{queue_type="parsing"}',
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "min": 0,
                    "max": 2000,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 500, "color": "yellow"},
                            {"value": 1000, "color": "red"}
                        ]
                    }
                }
            }
        ))

        # Panel 5: Documents processed over time (Time series)
        dashboard.panels.append(Panel(
            panel_id=5,
            title="Documents Processed (Time Series)",
            panel_type="timeseries",
            targets=[{
                "expr": "sum(rate(legal_parser_documents_processed_total{status='success'}[5m])) by (document_type)",
                "legendFormat": "{{document_type}}",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["lastNotNull", "mean"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 10
                    }
                }
            }
        ))

        # Panel 6: Failures over time (Time series)
        dashboard.panels.append(Panel(
            panel_id=6,
            title="Failures (Time Series)",
            panel_type="timeseries",
            targets=[{
                "expr": "sum(rate(legal_parser_documents_processed_total{status='failed'}[5m])) by (document_type)",
                "legendFormat": "{{document_type}}",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom"
                }
            },
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 10
                    },
                    "color": {"mode": "palette-classic"}
                }
            }
        ))

        logger.debug("Built overview dashboard")
        return dashboard

    def build_performance_dashboard(self) -> Dashboard:
        """Build performance dashboard

        Returns:
            Dashboard with performance metrics
        """
        dashboard = Dashboard(
            uid="legal-parser-performance",
            title="Legal Parser - Performance",
            description="Performance metrics for Turkish legal document parsing",
            tags=["legal-parser", "performance"],
            time_from="now-6h",
            refresh="30s"
        )

        # Panel 1: Parsing duration p50, p95, p99 (Time series)
        dashboard.panels.append(Panel(
            panel_id=1,
            title="Parsing Duration (Percentiles)",
            panel_type="timeseries",
            targets=[
                {
                    "expr": "histogram_quantile(0.50, rate(legal_parser_parse_duration_seconds_bucket[5m]))",
                    "legendFormat": "p50",
                    "refId": "A",
                    "datasource": self.datasource
                },
                {
                    "expr": "histogram_quantile(0.95, rate(legal_parser_parse_duration_seconds_bucket[5m]))",
                    "legendFormat": "p95",
                    "refId": "B",
                    "datasource": self.datasource
                },
                {
                    "expr": "histogram_quantile(0.99, rate(legal_parser_parse_duration_seconds_bucket[5m]))",
                    "legendFormat": "p99",
                    "refId": "C",
                    "datasource": self.datasource
                }
            ],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["lastNotNull", "mean", "max"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth"
                    }
                }
            }
        ))

        # Panel 2: Parsing duration by component (Heatmap)
        dashboard.panels.append(Panel(
            panel_id=2,
            title="Parsing Duration Heatmap (by Component)",
            panel_type="heatmap",
            targets=[{
                "expr": "rate(legal_parser_parse_duration_seconds_bucket[5m])",
                "format": "heatmap",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            options={
                "calculate": True,
                "calculation": {
                    "xBuckets": {"mode": "size", "value": "1min"}
                },
                "color": {
                    "mode": "scheme",
                    "scheme": "Spectral"
                }
            }
        ))

        # Panel 3: Average parsing duration by document type (Bar gauge)
        dashboard.panels.append(Panel(
            panel_id=3,
            title="Avg Parsing Duration by Document Type",
            panel_type="bargauge",
            targets=[{
                "expr": "avg(rate(legal_parser_parse_duration_seconds_sum[5m])) by (document_type) / avg(rate(legal_parser_parse_duration_seconds_count[5m])) by (document_type)",
                "legendFormat": "{{document_type}}",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 8, "w": 12, "h": 6},
            options={
                "orientation": "horizontal",
                "displayMode": "gradient"
            },
            field_config={
                "defaults": {
                    "unit": "s",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 10, "color": "yellow"},
                            {"value": 30, "color": "red"}
                        ]
                    }
                }
            }
        ))

        # Panel 4: Active parsers (Stat)
        dashboard.panels.append(Panel(
            panel_id=4,
            title="Active Parsers",
            panel_type="stat",
            targets=[{
                "expr": "legal_parser_active_parsers",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 12, "y": 8, "w": 6, "h": 6},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "textMode": "value_and_name",
                "colorMode": "background"
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 1, "color": "yellow"},
                            {"value": 2, "color": "green"}
                        ]
                    }
                }
            }
        ))

        # Panel 5: Cache hit rate (Gauge)
        dashboard.panels.append(Panel(
            panel_id=5,
            title="Cache Hit Rate",
            panel_type="gauge",
            targets=[{
                "expr": "rate(legal_parser_cache_hits_total[5m]) / (rate(legal_parser_cache_hits_total[5m]) + rate(legal_parser_cache_misses_total[5m])) * 100",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 18, "y": 8, "w": 6, "h": 6},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 50, "color": "yellow"},
                            {"value": 80, "color": "green"}
                        ]
                    }
                }
            }
        ))

        logger.debug("Built performance dashboard")
        return dashboard

    def build_quality_dashboard(self) -> Dashboard:
        """Build data quality dashboard

        Returns:
            Dashboard with quality metrics
        """
        dashboard = Dashboard(
            uid="legal-parser-quality",
            title="Legal Parser - Data Quality",
            description="Data quality metrics for Turkish legal document parsing",
            tags=["legal-parser", "quality"],
            time_from="now-6h",
            refresh="1m"
        )

        # Panel 1: Validation errors by severity (Time series)
        dashboard.panels.append(Panel(
            panel_id=1,
            title="Validation Errors by Severity",
            panel_type="timeseries",
            targets=[{
                "expr": "sum(rate(legal_parser_validation_errors_total[5m])) by (severity)",
                "legendFormat": "{{severity}}",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["lastNotNull", "mean", "max"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 20,
                        "stacking": {"mode": "normal"}
                    }
                }
            }
        ))

        # Panel 2: Validation errors by type (Pie chart)
        dashboard.panels.append(Panel(
            panel_id=2,
            title="Validation Errors by Type",
            panel_type="piechart",
            targets=[{
                "expr": "sum(increase(legal_parser_validation_errors_total[1h])) by (error_type)",
                "legendFormat": "{{error_type}}",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "calcs": ["lastNotNull"]
                },
                "pieType": "donut"
            }
        ))

        # Panel 3: Articles extracted per document (Stat)
        dashboard.panels.append(Panel(
            panel_id=3,
            title="Avg Articles per Document",
            panel_type="stat",
            targets=[{
                "expr": "rate(legal_parser_articles_extracted_total[5m]) / rate(legal_parser_documents_processed_total[5m])",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 8, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "textMode": "value_and_name"
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "decimals": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 5, "color": "yellow"},
                            {"value": 10, "color": "green"}
                        ]
                    }
                }
            }
        ))

        # Panel 4: Citations extracted per document (Stat)
        dashboard.panels.append(Panel(
            panel_id=4,
            title="Avg Citations per Document",
            panel_type="stat",
            targets=[{
                "expr": "rate(legal_parser_citations_extracted_total[5m]) / rate(legal_parser_documents_processed_total[5m])",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 6, "y": 8, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "textMode": "value_and_name"
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "decimals": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "yellow"},
                            {"value": 1, "color": "green"}
                        ]
                    }
                }
            }
        ))

        # Panel 5: Average confidence score (Gauge)
        dashboard.panels.append(Panel(
            panel_id=5,
            title="Avg Confidence Score",
            panel_type="gauge",
            targets=[{
                "expr": "avg(legal_parser_confidence_score)",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 12, "y": 8, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 0.7, "color": "yellow"},
                            {"value": 0.9, "color": "green"}
                        ]
                    }
                }
            }
        ))

        # Panel 6: Document completeness (Gauge)
        dashboard.panels.append(Panel(
            panel_id=6,
            title="Document Completeness",
            panel_type="gauge",
            targets=[{
                "expr": "avg(legal_parser_document_completeness)",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 18, "y": 8, "w": 6, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            field_config={
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 0.8, "color": "yellow"},
                            {"value": 0.95, "color": "green"}
                        ]
                    }
                }
            }
        ))

        logger.debug("Built quality dashboard")
        return dashboard

    def build_alerts_dashboard(self) -> Dashboard:
        """Build alerts status dashboard

        Returns:
            Dashboard with alert status
        """
        dashboard = Dashboard(
            uid="legal-parser-alerts",
            title="Legal Parser - Alerts",
            description="Alert status and history for Turkish legal document parsing",
            tags=["legal-parser", "alerts"],
            time_from="now-24h",
            refresh="1m"
        )

        # Panel 1: Active alerts (Stat)
        dashboard.panels.append(Panel(
            panel_id=1,
            title="Active Alerts",
            panel_type="stat",
            targets=[{
                "expr": "count(ALERTS{alertstate='firing', component='parser'})",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 0, "w": 8, "h": 4},
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "textMode": "value_and_name",
                "colorMode": "background"
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 1, "color": "yellow"},
                            {"value": 3, "color": "red"}
                        ]
                    }
                }
            }
        ))

        # Panel 2: Alert firing history (Time series)
        dashboard.panels.append(Panel(
            panel_id=2,
            title="Alert Firing History",
            panel_type="timeseries",
            targets=[{
                "expr": "ALERTS{component='parser'}",
                "legendFormat": "{{alertname}} - {{severity}}",
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 8, "y": 0, "w": 16, "h": 8},
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom"
                }
            },
            field_config={
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 100
                    }
                }
            }
        ))

        # Panel 3: Alert details table
        dashboard.panels.append(Panel(
            panel_id=3,
            title="Alert Details",
            panel_type="table",
            targets=[{
                "expr": "ALERTS{component='parser'}",
                "format": "table",
                "instant": True,
                "refId": "A",
                "datasource": self.datasource
            }],
            grid_pos={"x": 0, "y": 8, "w": 24, "h": 8},
            options={
                "showHeader": True
            },
            field_config={
                "defaults": {},
                "overrides": []
            }
        ))

        logger.debug("Built alerts dashboard")
        return dashboard

    def _create_document_type_variable(self) -> Dict[str, Any]:
        """Create document type template variable

        Returns:
            Template variable config
        """
        return {
            "name": "document_type",
            "type": "query",
            "label": "Document Type",
            "query": "label_values(legal_parser_documents_processed_total, document_type)",
            "datasource": self.datasource,
            "refresh": 1,
            "multi": True,
            "includeAll": True,
            "allValue": ".*"
        }

    def _create_source_variable(self) -> Dict[str, Any]:
        """Create source template variable

        Returns:
            Template variable config
        """
        return {
            "name": "source",
            "type": "query",
            "label": "Source",
            "query": "label_values(legal_parser_documents_processed_total, source)",
            "datasource": self.datasource,
            "refresh": 1,
            "multi": True,
            "includeAll": True,
            "allValue": ".*"
        }


# ============================================================================
# DASHBOARD EXPORT
# ============================================================================

def generate_all_dashboards(
    datasource: str = "Prometheus",
    output_dir: Optional[str] = None
) -> Dict[str, Dashboard]:
    """Generate all dashboards

    Args:
        datasource: Grafana datasource name
        output_dir: Output directory for JSON files (optional)

    Returns:
        Dict of dashboard name to Dashboard object
    """
    builder = DashboardBuilder(datasource=datasource)

    dashboards = {
        'overview': builder.build_overview_dashboard(),
        'performance': builder.build_performance_dashboard(),
        'quality': builder.build_quality_dashboard(),
        'alerts': builder.build_alerts_dashboard()
    }

    # Save to files if output_dir provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        for name, dashboard in dashboards.items():
            file_path = os.path.join(output_dir, f"{name}_dashboard.json")
            save_dashboard_to_file(dashboard, file_path)

    logger.info(f"Generated {len(dashboards)} dashboards")

    return dashboards


def save_dashboard_to_file(dashboard: Dashboard, file_path: str) -> None:
    """Save dashboard to JSON file

    Args:
        dashboard: Dashboard object
        file_path: Output file path
    """
    dashboard_json = dashboard.to_json()

    # Wrap in Grafana import format
    export_json = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "inputs": [],
        "folderId": 0
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved dashboard to {file_path}")


def export_dashboards_to_grafana(
    dashboards: Dict[str, Dashboard],
    grafana_url: str,
    api_key: str
) -> Dict[str, bool]:
    """Export dashboards to Grafana via API

    Args:
        dashboards: Dict of dashboards
        grafana_url: Grafana URL
        api_key: Grafana API key

    Returns:
        Dict of dashboard name to success status
    """
    results = {}

    try:
        import requests
    except ImportError:
        logger.error("requests library not available - cannot export to Grafana")
        return {name: False for name in dashboards}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for name, dashboard in dashboards.items():
        try:
            dashboard_json = dashboard.to_json()

            payload = {
                "dashboard": dashboard_json,
                "overwrite": True,
                "folderId": 0
            }

            response = requests.post(
                f"{grafana_url}/api/dashboards/db",
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code in (200, 201):
                logger.info(f"Exported dashboard '{name}' to Grafana")
                results[name] = True
            else:
                logger.error(f"Failed to export dashboard '{name}': {response.status_code} {response.text}")
                results[name] = False

        except Exception as e:
            logger.error(f"Error exporting dashboard '{name}': {e}")
            results[name] = False

    return results


__all__ = [
    'Panel',
    'Dashboard',
    'DashboardBuilder',
    'generate_all_dashboards',
    'save_dashboard_to_file',
    'export_dashboards_to_grafana'
]
