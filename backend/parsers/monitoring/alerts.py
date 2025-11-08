"""Parser Alerts - Harvey/Legora CTO-Level Production-Grade
Alert rule definitions for Turkish legal document parsing monitoring

Production Features:
- Prometheus Alertmanager rule definitions
- High parsing failure rate alerts (>5%, >10%)
- Slow processing time alerts
- Queue depth alerts
- Validation error spike detection
- Parser downtime detection
- Data quality alerts
- YAML alert rule generation
- Alert severity levels (warning, critical)
- Custom alert labels and annotations
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ALERT RULE DEFINITIONS
# ============================================================================

@dataclass
class AlertRule:
    """Prometheus alert rule"""
    name: str
    expr: str  # PromQL expression
    duration: str  # e.g., "5m", "10m"
    severity: str  # warning, critical
    summary: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-compatible dict

        Returns:
            Dict for YAML serialization
        """
        rule = {
            'alert': self.name,
            'expr': self.expr,
            'for': self.duration,
            'labels': {
                'severity': self.severity,
                **self.labels
            },
            'annotations': {
                'summary': self.summary,
                'description': self.description,
                **self.annotations
            }
        }

        return rule


# ============================================================================
# PARSER ALERT RULES
# ============================================================================

# High parsing failure rate
HIGH_FAILURE_RATE_ALERT = AlertRule(
    name="HighParsingFailureRate",
    expr=(
        "(rate(legal_parser_documents_failed_total[5m]) / "
        "rate(legal_parser_documents_processed_total[5m])) > 0.05"
    ),
    duration="5m",
    severity="warning",
    summary="High parsing failure rate detected",
    description=(
        "Parsing failure rate is above 5% for the last 5 minutes. "
        "Current rate: {{ $value | humanizePercentage }}. "
        "This may indicate issues with document quality or parser bugs."
    ),
    labels={"component": "parser"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/high-failure-rate"
    }
)

# Critical parsing failure rate
CRITICAL_FAILURE_RATE_ALERT = AlertRule(
    name="CriticalParsingFailureRate",
    expr=(
        "(rate(legal_parser_documents_failed_total[5m]) / "
        "rate(legal_parser_documents_processed_total[5m])) > 0.10"
    ),
    duration="3m",
    severity="critical",
    summary="CRITICAL: Very high parsing failure rate",
    description=(
        "Parsing failure rate is above 10% for the last 3 minutes. "
        "Current rate: {{ $value | humanizePercentage }}. "
        "IMMEDIATE ACTION REQUIRED."
    ),
    labels={"component": "parser", "page": "oncall"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/critical-failure-rate"
    }
)

# Slow parsing
SLOW_PARSING_ALERT = AlertRule(
    name="SlowParsingDetected",
    expr=(
        "histogram_quantile(0.95, "
        "rate(legal_parser_parse_duration_seconds_bucket[10m])) > 60"
    ),
    duration="10m",
    severity="warning",
    summary="Slow parsing performance detected",
    description=(
        "95th percentile parsing duration is above 60 seconds. "
        "Current p95: {{ $value | humanizeDuration }}. "
        "This may indicate performance degradation."
    ),
    labels={"component": "parser"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/slow-parsing"
    }
)

# Queue backup
QUEUE_BACKUP_ALERT = AlertRule(
    name="ParsingQueueBackup",
    expr='legal_parser_queue_size{queue_type="parsing"} > 1000',
    duration="15m",
    severity="warning",
    summary="Parsing queue backup detected",
    description=(
        "Parsing queue has more than 1000 documents waiting for {{ $labels.queue_type }}. "
        "Current queue size: {{ $value }}. "
        "This may indicate insufficient parser capacity."
    ),
    labels={"component": "parser"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/queue-backup"
    }
)

# No parsing activity
NO_PARSING_ACTIVITY_ALERT = AlertRule(
    name="NoParsingActivity",
    expr='rate(legal_parser_documents_processed_total[10m]) == 0',
    duration="10m",
    severity="critical",
    summary="No parsing activity detected",
    description=(
        "No documents have been processed in the last 10 minutes. "
        "Parser may be down or stuck."
    ),
    labels={"component": "parser", "page": "oncall"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/no-activity"
    }
)

# High validation error rate
HIGH_VALIDATION_ERROR_RATE_ALERT = AlertRule(
    name="HighValidationErrorRate",
    expr=(
        'rate(legal_parser_validation_errors_total{severity="ERROR"}[5m]) > 10'
    ),
    duration="5m",
    severity="warning",
    summary="High validation error rate",
    description=(
        "Validation errors (severity=ERROR) are occurring at a rate above 10/sec. "
        "Current rate: {{ $value }} errors/sec. "
        "This may indicate data quality issues."
    ),
    labels={"component": "validator"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/validation-errors"
    }
)

# Parser workers down
PARSER_WORKERS_DOWN_ALERT = AlertRule(
    name="ParserWorkersDown",
    expr='legal_parser_active_parsers < 2',
    duration="5m",
    severity="critical",
    summary="Insufficient parser workers",
    description=(
        "Number of active parser workers is below minimum threshold (2). "
        "Current workers: {{ $value }}. "
        "This will severely impact throughput."
    ),
    labels={"component": "parser", "page": "oncall"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/workers-down"
    }
)

# Low article extraction rate
LOW_ARTICLE_EXTRACTION_ALERT = AlertRule(
    name="LowArticleExtractionRate",
    expr=(
        "(rate(legal_parser_articles_extracted_total[10m]) / "
        "rate(legal_parser_documents_processed_total[10m])) < 5"
    ),
    duration="10m",
    severity="warning",
    summary="Low article extraction rate",
    description=(
        "Average articles extracted per document is below 5. "
        "Current average: {{ $value | humanize }} articles/doc. "
        "This may indicate structure parsing issues."
    ),
    labels={"component": "parser"},
    annotations={
        "runbook_url": "https://docs.legal-parser.com/runbooks/low-extraction"
    }
)


# ============================================================================
# ALERT RULE GROUPS
# ============================================================================

def get_all_alert_rules() -> List[AlertRule]:
    """Get all defined alert rules

    Returns:
        List of AlertRule
    """
    return [
        HIGH_FAILURE_RATE_ALERT,
        CRITICAL_FAILURE_RATE_ALERT,
        SLOW_PARSING_ALERT,
        QUEUE_BACKUP_ALERT,
        NO_PARSING_ACTIVITY_ALERT,
        HIGH_VALIDATION_ERROR_RATE_ALERT,
        PARSER_WORKERS_DOWN_ALERT,
        LOW_ARTICLE_EXTRACTION_ALERT
    ]


def get_alert_rules_by_severity(severity: str) -> List[AlertRule]:
    """Get alert rules by severity

    Args:
        severity: Alert severity (warning, critical)

    Returns:
        List of matching AlertRule
    """
    return [rule for rule in get_all_alert_rules() if rule.severity == severity]


def generate_prometheus_alert_rules_yaml() -> str:
    """Generate Prometheus alert rules in YAML format

    Returns:
        YAML string with alert rules
    """
    rules = get_all_alert_rules()

    # Build YAML structure
    yaml_dict = {
        'groups': [
            {
                'name': 'legal_parser_alerts',
                'interval': '30s',
                'rules': [rule.to_yaml_dict() for rule in rules]
            }
        ]
    }

    # Convert to YAML string (manual, no PyYAML dependency)
    yaml_lines = ['groups:']
    yaml_lines.append('  - name: legal_parser_alerts')
    yaml_lines.append('    interval: 30s')
    yaml_lines.append('    rules:')

    for rule in rules:
        rule_dict = rule.to_yaml_dict()

        yaml_lines.append(f'      - alert: {rule_dict["alert"]}')
        yaml_lines.append(f'        expr: {rule_dict["expr"]}')
        yaml_lines.append(f'        for: {rule_dict["for"]}')

        # Labels
        yaml_lines.append('        labels:')
        for key, value in rule_dict['labels'].items():
            yaml_lines.append(f'          {key}: {value}')

        # Annotations
        yaml_lines.append('        annotations:')
        for key, value in rule_dict['annotations'].items():
            # Escape quotes in YAML
            escaped_value = value.replace('"', '\\"')
            yaml_lines.append(f'          {key}: "{escaped_value}"')

        yaml_lines.append('')  # Blank line between rules

    return '\n'.join(yaml_lines)


def save_alert_rules_to_file(file_path: str) -> None:
    """Save alert rules to YAML file

    Args:
        file_path: Output file path
    """
    yaml_content = generate_prometheus_alert_rules_yaml()

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    logger.info(f"Saved {len(get_all_alert_rules())} alert rules to {file_path}")


__all__ = [
    'AlertRule',
    'HIGH_FAILURE_RATE_ALERT',
    'CRITICAL_FAILURE_RATE_ALERT',
    'SLOW_PARSING_ALERT',
    'QUEUE_BACKUP_ALERT',
    'NO_PARSING_ACTIVITY_ALERT',
    'HIGH_VALIDATION_ERROR_RATE_ALERT',
    'PARSER_WORKERS_DOWN_ALERT',
    'LOW_ARTICLE_EXTRACTION_ALERT',
    'get_all_alert_rules',
    'get_alert_rules_by_severity',
    'generate_prometheus_alert_rules_yaml',
    'save_alert_rules_to_file'
]
