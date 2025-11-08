"""Regression Tester - Harvey/Legora CTO-Level Production-Grade
Automated regression testing for parsers

Production Features:
- Automated regression testing
- Compare current vs. baseline parsing results
- Detect regressions
- Performance benchmarking
- Test report generation
- CI/CD integration support
- Turkish legal document regression tests
- Production-grade reporting
"""
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class RegressionType(Enum):
    """Types of regressions"""
    ACCURACY_DROP = "ACCURACY_DROP"  # Accuracy decreased
    FIELD_MISSING = "FIELD_MISSING"  # Field now missing
    FIELD_CHANGED = "FIELD_CHANGED"  # Field value changed
    PERFORMANCE_DROP = "PERFORMANCE_DROP"  # Performance decreased
    ERROR_INTRODUCED = "ERROR_INTRODUCED"  # New errors introduced
    STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"  # Structure changed


class RegressionSeverity(Enum):
    """Regression severity levels"""
    CRITICAL = "CRITICAL"  # Breaking change
    HIGH = "HIGH"  # Significant regression
    MEDIUM = "MEDIUM"  # Moderate regression
    LOW = "LOW"  # Minor regression
    IMPROVEMENT = "IMPROVEMENT"  # Actually an improvement


@dataclass
class Regression:
    """Represents a detected regression"""
    regression_type: RegressionType
    severity: RegressionSeverity
    description: str
    test_case_id: str
    baseline_value: Any
    current_value: Any
    impact: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.regression_type.value}: {self.description}"


@dataclass
class BaselineSnapshot:
    """Baseline snapshot of parser output"""
    snapshot_id: str
    version: str
    test_case_id: str
    input_data: Any
    output_data: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'snapshot_id': self.snapshot_id,
            'version': self.version,
            'test_case_id': self.test_case_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineSnapshot':
        """Create from dictionary"""
        return cls(
            snapshot_id=data['snapshot_id'],
            version=data['version'],
            test_case_id=data['test_case_id'],
            input_data=data['input_data'],
            output_data=data['output_data'],
            performance_metrics=data.get('performance_metrics', {}),
            created_at=data.get('created_at'),
            metadata=data.get('metadata', {})
        )


@dataclass
class RegressionTestReport:
    """Complete regression test report"""
    total_tests: int
    passed_tests: int  # No regressions
    regressed_tests: int  # Tests with regressions
    improved_tests: int  # Tests that improved

    # Regressions by severity
    critical_regressions: int = 0
    high_regressions: int = 0
    medium_regressions: int = 0
    low_regressions: int = 0

    # All regressions
    regressions: List[Regression] = field(default_factory=list)

    # Performance comparison
    baseline_avg_time: float = 0.0
    current_avg_time: float = 0.0
    performance_delta: float = 0.0  # Negative = slower

    # Test metadata
    test_time: float = 0.0
    tested_at: Optional[str] = None
    baseline_version: Optional[str] = None
    current_version: Optional[str] = None

    @property
    def regression_rate(self) -> float:
        """Get regression rate"""
        return self.regressed_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def has_critical_regressions(self) -> bool:
        """Check if has critical regressions"""
        return self.critical_regressions > 0

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Regression Test Report")
        lines.append(f"=" * 50)
        lines.append(f"Total Tests: {self.total_tests}")
        lines.append(f"Passed: {self.passed_tests}")
        lines.append(f"Regressed: {self.regressed_tests} ({self.regression_rate*100:.1f}%)")
        lines.append(f"Improved: {self.improved_tests}")
        lines.append(f"")
        lines.append(f"Regressions by Severity:")
        lines.append(f"  Critical: {self.critical_regressions}")
        lines.append(f"  High: {self.high_regressions}")
        lines.append(f"  Medium: {self.medium_regressions}")
        lines.append(f"  Low: {self.low_regressions}")

        if self.baseline_avg_time > 0 and self.current_avg_time > 0:
            lines.append(f"\nPerformance:")
            lines.append(f"  Baseline: {self.baseline_avg_time:.3f}s")
            lines.append(f"  Current: {self.current_avg_time:.3f}s")
            perf_change = ((self.current_avg_time - self.baseline_avg_time) / self.baseline_avg_time) * 100
            direction = "slower" if perf_change > 0 else "faster"
            lines.append(f"  Change: {abs(perf_change):.1f}% {direction}")

        if self.regressions:
            lines.append(f"\nTop Regressions:")
            # Sort by severity
            severity_order = {
                RegressionSeverity.CRITICAL: 0,
                RegressionSeverity.HIGH: 1,
                RegressionSeverity.MEDIUM: 2,
                RegressionSeverity.LOW: 3,
                RegressionSeverity.IMPROVEMENT: 4
            }
            sorted_regressions = sorted(
                self.regressions,
                key=lambda r: severity_order[r.severity]
            )
            for regression in sorted_regressions[:5]:
                lines.append(f"  - {regression}")

        lines.append(f"\nTest Time: {self.test_time:.3f}s")

        return '\n'.join(lines)


class RegressionTester:
    """Regression Tester for Turkish Legal Document Parsers

    Automated regression testing:
    - Create baseline snapshots
    - Compare current output against baseline
    - Detect regressions
    - Performance benchmarking
    - CI/CD integration support

    Features:
    - Multiple baseline versions
    - Detailed regression analysis
    - Performance comparison
    - Turkish legal document specific checks
    - Export reports for CI/CD
    """

    def __init__(self, baseline_path: Optional[Path] = None):
        """Initialize Regression Tester

        Args:
            baseline_path: Path to baseline file (optional)
        """
        self.baseline_path = baseline_path
        self.baselines: Dict[str, BaselineSnapshot] = {}

        # Statistics
        self.stats = {
            'total_tests': 0,
            'total_regressions': 0,
            'regression_rate': 0.0,
            'test_time': 0.0,
        }

        # Load baseline if path provided
        if baseline_path and baseline_path.exists():
            self.load_baseline(baseline_path)

        logger.info("Initialized Regression Tester")

    def create_baseline(
        self,
        test_cases: List[Tuple[str, Any]],
        parser_func: Callable,
        version: str
    ) -> int:
        """Create baseline snapshots from test cases

        Args:
            test_cases: List of (test_case_id, input_data) tuples
            parser_func: Parser function to use
            version: Version identifier

        Returns:
            Number of baselines created
        """
        created = 0

        for test_case_id, input_data in test_cases:
            try:
                # Parse input and measure time
                start_time = time.time()
                output_data = parser_func(input_data)
                parse_time = time.time() - start_time

                # Create snapshot
                snapshot = BaselineSnapshot(
                    snapshot_id=f"{test_case_id}_{version}",
                    version=version,
                    test_case_id=test_case_id,
                    input_data=input_data,
                    output_data=output_data,
                    performance_metrics={'parse_time': parse_time}
                )

                self.baselines[test_case_id] = snapshot
                created += 1

            except Exception as e:
                logger.error(f"Failed to create baseline for {test_case_id}: {e}")

        logger.info(f"Created {created} baseline snapshots (version: {version})")
        return created

    def test(
        self,
        test_cases: List[Tuple[str, Any]],
        parser_func: Callable,
        current_version: str,
        **kwargs
    ) -> RegressionTestReport:
        """Run regression tests

        Args:
            test_cases: List of (test_case_id, input_data) tuples
            parser_func: Parser function to test
            current_version: Current version identifier
            **kwargs: Options
                - strict: Strict comparison mode (default: False)
                - performance_threshold: Performance degradation threshold (default: 0.2 = 20%)

        Returns:
            RegressionTestReport with results
        """
        start_time = time.time()

        strict = kwargs.get('strict', False)
        performance_threshold = kwargs.get('performance_threshold', 0.2)

        logger.info(f"Running regression tests ({len(test_cases)} test cases)")

        # Create report
        report = RegressionTestReport(
            total_tests=len(test_cases),
            passed_tests=0,
            regressed_tests=0,
            improved_tests=0,
            current_version=current_version
        )

        # Track performance
        baseline_times = []
        current_times = []

        for test_case_id, input_data in test_cases:
            # Check if baseline exists
            if test_case_id not in self.baselines:
                logger.warning(f"No baseline for test case: {test_case_id}")
                continue

            baseline = self.baselines[test_case_id]
            report.baseline_version = baseline.version

            try:
                # Parse current
                start_parse = time.time()
                current_output = parser_func(input_data)
                parse_time = time.time() - start_parse

                # Compare outputs
                regressions = self._compare_outputs(
                    test_case_id,
                    baseline.output_data,
                    current_output,
                    strict
                )

                # Check performance
                if 'parse_time' in baseline.performance_metrics:
                    baseline_time = baseline.performance_metrics['parse_time']
                    baseline_times.append(baseline_time)
                    current_times.append(parse_time)

                    # Check for performance regression
                    if parse_time > baseline_time * (1 + performance_threshold):
                        perf_regression = Regression(
                            regression_type=RegressionType.PERFORMANCE_DROP,
                            severity=RegressionSeverity.MEDIUM,
                            description=f"Performance degraded by {((parse_time - baseline_time) / baseline_time) * 100:.1f}%",
                            test_case_id=test_case_id,
                            baseline_value=baseline_time,
                            current_value=parse_time,
                            suggestion="Investigate parser performance"
                        )
                        regressions.append(perf_regression)

                # Process regressions
                if regressions:
                    # Count improvements vs regressions
                    improvements = [r for r in regressions if r.severity == RegressionSeverity.IMPROVEMENT]
                    actual_regressions = [r for r in regressions if r.severity != RegressionSeverity.IMPROVEMENT]

                    if actual_regressions:
                        report.regressed_tests += 1
                        report.regressions.extend(actual_regressions)

                        # Count by severity
                        for reg in actual_regressions:
                            if reg.severity == RegressionSeverity.CRITICAL:
                                report.critical_regressions += 1
                            elif reg.severity == RegressionSeverity.HIGH:
                                report.high_regressions += 1
                            elif reg.severity == RegressionSeverity.MEDIUM:
                                report.medium_regressions += 1
                            elif reg.severity == RegressionSeverity.LOW:
                                report.low_regressions += 1
                    elif improvements:
                        report.improved_tests += 1
                    else:
                        report.passed_tests += 1
                else:
                    report.passed_tests += 1

            except Exception as e:
                logger.error(f"Regression test failed for {test_case_id}: {e}")

                # Create error regression
                error_reg = Regression(
                    regression_type=RegressionType.ERROR_INTRODUCED,
                    severity=RegressionSeverity.CRITICAL,
                    description=f"Parser threw exception: {str(e)}",
                    test_case_id=test_case_id,
                    baseline_value="No error",
                    current_value=str(e)
                )
                report.regressions.append(error_reg)
                report.regressed_tests += 1
                report.critical_regressions += 1

        # Calculate performance metrics
        if baseline_times and current_times:
            report.baseline_avg_time = sum(baseline_times) / len(baseline_times)
            report.current_avg_time = sum(current_times) / len(current_times)
            report.performance_delta = (
                (report.current_avg_time - report.baseline_avg_time) / report.baseline_avg_time
            )

        # Finalize
        report.test_time = time.time() - start_time
        self._update_stats(report)

        logger.info(f"Regression testing complete: {report.regressed_tests} regressions found")

        return report

    def _compare_outputs(
        self,
        test_case_id: str,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        strict: bool
    ) -> List[Regression]:
        """Compare baseline and current outputs

        Args:
            test_case_id: Test case ID
            baseline: Baseline output
            current: Current output
            strict: Strict comparison mode

        Returns:
            List of regressions found
        """
        regressions = []

        # Check for missing fields
        baseline_fields = set(baseline.keys())
        current_fields = set(current.keys())

        missing_fields = baseline_fields - current_fields
        for field in missing_fields:
            regressions.append(Regression(
                regression_type=RegressionType.FIELD_MISSING,
                severity=RegressionSeverity.HIGH,
                description=f"Field '{field}' missing in current output",
                test_case_id=test_case_id,
                baseline_value=baseline[field],
                current_value=None,
                suggestion=f"Ensure parser extracts '{field}' field"
            ))

        # Check for changed fields
        for field in baseline_fields & current_fields:
            baseline_value = baseline[field]
            current_value = current[field]

            if not self._values_equal(baseline_value, current_value, strict):
                # Determine severity based on field importance
                severity = self._determine_field_change_severity(field, baseline_value, current_value)

                regressions.append(Regression(
                    regression_type=RegressionType.FIELD_CHANGED,
                    severity=severity,
                    description=f"Field '{field}' value changed",
                    test_case_id=test_case_id,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    impact=self._describe_change_impact(field, baseline_value, current_value)
                ))

        # Check for new fields (could be improvement)
        new_fields = current_fields - baseline_fields
        if new_fields and not strict:
            # New fields might be an improvement
            regressions.append(Regression(
                regression_type=RegressionType.STRUCTURAL_CHANGE,
                severity=RegressionSeverity.IMPROVEMENT,
                description=f"New fields added: {', '.join(new_fields)}",
                test_case_id=test_case_id,
                baseline_value=None,
                current_value=list(new_fields)
            ))

        return regressions

    def _values_equal(self, baseline: Any, current: Any, strict: bool) -> bool:
        """Compare two values for equality

        Args:
            baseline: Baseline value
            current: Current value
            strict: Strict comparison mode

        Returns:
            True if values are equal
        """
        # Exact match
        if baseline == current:
            return True

        # Type mismatch
        if type(baseline) != type(current):
            # Allow string/int equivalence
            if isinstance(baseline, (int, str)) and isinstance(current, (int, str)):
                try:
                    return str(baseline) == str(current)
                except:
                    return False
            return False

        # List comparison
        if isinstance(baseline, list) and isinstance(current, list):
            if len(baseline) != len(current):
                return False
            return all(self._values_equal(b, c, strict) for b, c in zip(baseline, current))

        # Dict comparison
        if isinstance(baseline, dict) and isinstance(current, dict):
            if strict and set(baseline.keys()) != set(current.keys()):
                return False
            return all(
                self._values_equal(baseline.get(k), current.get(k), strict)
                for k in baseline.keys()
            )

        # String comparison (case-insensitive for non-strict)
        if isinstance(baseline, str) and isinstance(current, str):
            if strict:
                return baseline == current
            else:
                return baseline.lower().strip() == current.lower().strip()

        return False

    def _determine_field_change_severity(
        self,
        field: str,
        baseline_value: Any,
        current_value: Any
    ) -> RegressionSeverity:
        """Determine severity of field change

        Args:
            field: Field name
            baseline_value: Baseline value
            current_value: Current value

        Returns:
            Regression severity
        """
        # Critical fields for Turkish legal documents
        critical_fields = [
            'law_number', 'regulation_number', 'decision_number',
            'court', 'publication_date', 'date'
        ]

        # High priority fields
        high_priority_fields = [
            'title', 'articles', 'decision_text', 'authority'
        ]

        if field in critical_fields:
            return RegressionSeverity.CRITICAL
        elif field in high_priority_fields:
            return RegressionSeverity.HIGH
        else:
            # Check magnitude of change
            if isinstance(baseline_value, (list, dict)) and isinstance(current_value, (list, dict)):
                # Structural changes are medium severity
                return RegressionSeverity.MEDIUM
            else:
                # Simple value changes are low severity
                return RegressionSeverity.LOW

    def _describe_change_impact(
        self,
        field: str,
        baseline_value: Any,
        current_value: Any
    ) -> str:
        """Describe the impact of a field change

        Args:
            field: Field name
            baseline_value: Baseline value
            current_value: Current value

        Returns:
            Impact description
        """
        if baseline_value is None and current_value is not None:
            return "Field now populated (improvement)"
        elif baseline_value is not None and current_value is None:
            return "Field now empty (regression)"
        elif isinstance(baseline_value, list) and isinstance(current_value, list):
            if len(current_value) > len(baseline_value):
                return f"List grew from {len(baseline_value)} to {len(current_value)} items"
            else:
                return f"List shrunk from {len(baseline_value)} to {len(current_value)} items"
        else:
            return "Value changed"

    def save_baseline(self, path: Path) -> None:
        """Save baseline to file

        Args:
            path: Path to save to
        """
        data = {
            'baselines': {
                test_id: baseline.to_dict()
                for test_id, baseline in self.baselines.items()
            },
            'metadata': {
                'total_baselines': len(self.baselines),
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved baseline to {path} ({len(self.baselines)} snapshots)")

    def load_baseline(self, path: Path) -> None:
        """Load baseline from file

        Args:
            path: Path to load from
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.baselines = {
            test_id: BaselineSnapshot.from_dict(baseline_data)
            for test_id, baseline_data in data.get('baselines', {}).items()
        }

        logger.info(f"Loaded baseline from {path} ({len(self.baselines)} snapshots)")

    def export_report(self, report: RegressionTestReport, path: Path, format: str = 'json') -> None:
        """Export regression test report

        Args:
            report: Regression test report
            path: Path to export to
            format: Export format ('json' or 'markdown')
        """
        if format == 'json':
            self._export_json_report(report, path)
        elif format == 'markdown':
            self._export_markdown_report(report, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json_report(self, report: RegressionTestReport, path: Path) -> None:
        """Export report as JSON"""
        data = {
            'summary': {
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'regressed_tests': report.regressed_tests,
                'improved_tests': report.improved_tests,
                'regression_rate': report.regression_rate,
                'has_critical_regressions': report.has_critical_regressions
            },
            'regressions_by_severity': {
                'critical': report.critical_regressions,
                'high': report.high_regressions,
                'medium': report.medium_regressions,
                'low': report.low_regressions
            },
            'performance': {
                'baseline_avg_time': report.baseline_avg_time,
                'current_avg_time': report.current_avg_time,
                'performance_delta': report.performance_delta
            },
            'regressions': [
                {
                    'type': r.regression_type.value,
                    'severity': r.severity.value,
                    'description': r.description,
                    'test_case_id': r.test_case_id,
                    'baseline_value': str(r.baseline_value),
                    'current_value': str(r.current_value),
                    'impact': r.impact
                }
                for r in report.regressions
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported JSON report to {path}")

    def _export_markdown_report(self, report: RegressionTestReport, path: Path) -> None:
        """Export report as Markdown"""
        lines = [
            "# Regression Test Report",
            "",
            "## Summary",
            f"- **Total Tests**: {report.total_tests}",
            f"- **Passed**: {report.passed_tests}",
            f"- **Regressed**: {report.regressed_tests} ({report.regression_rate*100:.1f}%)",
            f"- **Improved**: {report.improved_tests}",
            "",
            "## Regressions by Severity",
            f"- **Critical**: {report.critical_regressions}",
            f"- **High**: {report.high_regressions}",
            f"- **Medium**: {report.medium_regressions}",
            f"- **Low**: {report.low_regressions}",
            ""
        ]

        if report.regressions:
            lines.append("## Regressions")
            lines.append("")
            for r in report.regressions:
                lines.append(f"### {r.test_case_id}")
                lines.append(f"- **Type**: {r.regression_type.value}")
                lines.append(f"- **Severity**: {r.severity.value}")
                lines.append(f"- **Description**: {r.description}")
                lines.append(f"- **Baseline**: {r.baseline_value}")
                lines.append(f"- **Current**: {r.current_value}")
                if r.impact:
                    lines.append(f"- **Impact**: {r.impact}")
                lines.append("")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported Markdown report to {path}")

    def _update_stats(self, report: RegressionTestReport) -> None:
        """Update statistics"""
        self.stats['total_tests'] += report.total_tests
        self.stats['total_regressions'] += report.regressed_tests
        self.stats['test_time'] += report.test_time

        # Update regression rate
        if self.stats['total_tests'] > 0:
            self.stats['regression_rate'] = (
                self.stats['total_regressions'] / self.stats['total_tests']
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get tester statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_tests': 0,
            'total_regressions': 0,
            'regression_rate': 0.0,
            'test_time': 0.0,
        }
        logger.info("Statistics reset")


__all__ = [
    'RegressionTester',
    'RegressionTestReport',
    'Regression',
    'BaselineSnapshot',
    'RegressionType',
    'RegressionSeverity'
]
