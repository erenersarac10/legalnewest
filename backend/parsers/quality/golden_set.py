"""Golden Set Manager - Harvey/Legora CTO-Level Production-Grade
Manages golden dataset for quality assessment of parsers

Production Features:
- Golden dataset management
- Evaluate parsing accuracy against golden set
- Precision, recall, F1 metrics
- Error analysis
- Golden set versioning
- Test case management
- Turkish legal document test cases
- Production-grade evaluation
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics"""
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    F1_SCORE = "F1_SCORE"
    ACCURACY = "ACCURACY"
    FIELD_MATCH = "FIELD_MATCH"


@dataclass
class GoldenTestCase:
    """Represents a golden test case"""
    test_id: str
    document_type: str  # law, regulation, decision
    raw_input: Any  # Raw input data
    expected_output: Dict[str, Any]  # Expected parsed output
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_id': self.test_id,
            'document_type': self.document_type,
            'raw_input': self.raw_input,
            'expected_output': self.expected_output,
            'description': self.description,
            'tags': self.tags,
            'difficulty': self.difficulty,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenTestCase':
        """Create from dictionary"""
        return cls(
            test_id=data['test_id'],
            document_type=data['document_type'],
            raw_input=data['raw_input'],
            expected_output=data['expected_output'],
            description=data.get('description'),
            tags=data.get('tags', []),
            difficulty=data.get('difficulty', 'medium'),
            metadata=data.get('metadata', {})
        )


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a test case"""
    test_id: str
    passed: bool

    # Field-level metrics
    total_fields: int = 0
    matched_fields: int = 0
    missing_fields: int = 0
    extra_fields: int = 0
    incorrect_fields: int = 0

    # Content metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    # Error details
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_id': self.test_id,
            'passed': self.passed,
            'total_fields': self.total_fields,
            'matched_fields': self.matched_fields,
            'missing_fields': self.missing_fields,
            'extra_fields': self.extra_fields,
            'incorrect_fields': self.incorrect_fields,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'errors': self.errors
        }


@dataclass
class GoldenSetReport:
    """Complete evaluation report"""
    total_tests: int
    passed_tests: int
    failed_tests: int

    # Overall metrics
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    overall_accuracy: float = 0.0

    # Per-document-type metrics
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-difficulty metrics
    metrics_by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Individual test metrics
    test_metrics: List[EvaluationMetrics] = field(default_factory=list)

    # Evaluation metadata
    evaluation_time: float = 0.0
    evaluated_at: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        """Get pass rate"""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Golden Set Evaluation Report")
        lines.append(f"=" * 50)
        lines.append(f"Total Tests: {self.total_tests}")
        lines.append(f"Passed: {self.passed_tests} ({self.pass_rate*100:.1f}%)")
        lines.append(f"Failed: {self.failed_tests}")
        lines.append(f"")
        lines.append(f"Overall Metrics:")
        lines.append(f"  Precision: {self.overall_precision:.3f}")
        lines.append(f"  Recall: {self.overall_recall:.3f}")
        lines.append(f"  F1 Score: {self.overall_f1:.3f}")
        lines.append(f"  Accuracy: {self.overall_accuracy:.3f}")

        if self.metrics_by_type:
            lines.append(f"\nMetrics by Document Type:")
            for doc_type, metrics in self.metrics_by_type.items():
                lines.append(f"  {doc_type}:")
                lines.append(f"    Precision: {metrics.get('precision', 0.0):.3f}")
                lines.append(f"    Recall: {metrics.get('recall', 0.0):.3f}")
                lines.append(f"    F1: {metrics.get('f1', 0.0):.3f}")

        if self.metrics_by_difficulty:
            lines.append(f"\nMetrics by Difficulty:")
            for difficulty, metrics in self.metrics_by_difficulty.items():
                lines.append(f"  {difficulty}:")
                lines.append(f"    Pass Rate: {metrics.get('pass_rate', 0.0)*100:.1f}%")

        lines.append(f"\nEvaluation Time: {self.evaluation_time:.3f}s")

        return '\n'.join(lines)


class GoldenSetManager:
    """Golden Set Manager for Turkish Legal Document Parsing

    Manages golden test cases for quality assessment:
    - Store and load golden test cases
    - Evaluate parser output against golden set
    - Calculate precision, recall, F1 metrics
    - Track parser performance over time
    - Support versioning of golden sets

    Features:
    - Turkish legal document test cases
    - Multiple document types
    - Difficulty levels
    - Detailed error analysis
    - Version management
    """

    def __init__(self, golden_set_path: Optional[Path] = None, version: str = "1.0"):
        """Initialize Golden Set Manager

        Args:
            golden_set_path: Path to golden set file (optional)
            version: Golden set version
        """
        self.golden_set_path = golden_set_path
        self.version = version
        self.test_cases: List[GoldenTestCase] = []

        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'total_test_cases': 0,
            'average_pass_rate': 0.0,
            'evaluation_time': 0.0,
        }

        # Load golden set if path provided
        if golden_set_path and golden_set_path.exists():
            self.load_from_file(golden_set_path)

        logger.info(f"Initialized Golden Set Manager (version: {version})")

    def add_test_case(self, test_case: GoldenTestCase) -> None:
        """Add test case to golden set

        Args:
            test_case: Golden test case to add
        """
        # Check for duplicate ID
        if any(tc.test_id == test_case.test_id for tc in self.test_cases):
            logger.warning(f"Test case {test_case.test_id} already exists, replacing")
            self.test_cases = [tc for tc in self.test_cases if tc.test_id != test_case.test_id]

        self.test_cases.append(test_case)
        self.stats['total_test_cases'] = len(self.test_cases)
        logger.info(f"Added test case: {test_case.test_id}")

    def remove_test_case(self, test_id: str) -> bool:
        """Remove test case from golden set

        Args:
            test_id: Test case ID to remove

        Returns:
            True if removed, False if not found
        """
        initial_count = len(self.test_cases)
        self.test_cases = [tc for tc in self.test_cases if tc.test_id != test_id]

        if len(self.test_cases) < initial_count:
            self.stats['total_test_cases'] = len(self.test_cases)
            logger.info(f"Removed test case: {test_id}")
            return True
        else:
            logger.warning(f"Test case not found: {test_id}")
            return False

    def get_test_case(self, test_id: str) -> Optional[GoldenTestCase]:
        """Get test case by ID

        Args:
            test_id: Test case ID

        Returns:
            GoldenTestCase or None if not found
        """
        for tc in self.test_cases:
            if tc.test_id == test_id:
                return tc
        return None

    def filter_test_cases(
        self,
        document_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[GoldenTestCase]:
        """Filter test cases by criteria

        Args:
            document_type: Filter by document type
            difficulty: Filter by difficulty
            tags: Filter by tags (any match)

        Returns:
            List of matching test cases
        """
        filtered = self.test_cases

        if document_type:
            filtered = [tc for tc in filtered if tc.document_type == document_type]

        if difficulty:
            filtered = [tc for tc in filtered if tc.difficulty == difficulty]

        if tags:
            filtered = [
                tc for tc in filtered
                if any(tag in tc.tags for tag in tags)
            ]

        return filtered

    def evaluate(
        self,
        parser_func,
        **kwargs
    ) -> GoldenSetReport:
        """Evaluate parser against golden set

        Args:
            parser_func: Parser function that takes raw_input and returns parsed output
            **kwargs: Options
                - document_type: Only evaluate specific document type
                - difficulty: Only evaluate specific difficulty
                - strict: Strict evaluation mode (default: False)

        Returns:
            GoldenSetReport with evaluation results
        """
        start_time = time.time()

        # Filter test cases
        test_cases = self.filter_test_cases(
            document_type=kwargs.get('document_type'),
            difficulty=kwargs.get('difficulty')
        )

        if not test_cases:
            logger.warning("No test cases to evaluate")
            return GoldenSetReport(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                evaluation_time=time.time() - start_time
            )

        strict = kwargs.get('strict', False)

        logger.info(f"Evaluating parser against {len(test_cases)} test cases")

        # Create report
        report = GoldenSetReport(
            total_tests=len(test_cases),
            passed_tests=0,
            failed_tests=0
        )

        # Evaluate each test case
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_accuracies = []

        metrics_by_type: Dict[str, List[EvaluationMetrics]] = defaultdict(list)
        metrics_by_difficulty: Dict[str, List[EvaluationMetrics]] = defaultdict(list)

        for tc in test_cases:
            try:
                # Parse input
                parsed_output = parser_func(tc.raw_input)

                # Evaluate
                metrics = self._evaluate_test_case(tc, parsed_output, strict)

                report.test_metrics.append(metrics)

                if metrics.passed:
                    report.passed_tests += 1
                else:
                    report.failed_tests += 1

                # Collect metrics
                all_precisions.append(metrics.precision)
                all_recalls.append(metrics.recall)
                all_f1s.append(metrics.f1_score)
                all_accuracies.append(metrics.accuracy)

                # Group by type and difficulty
                metrics_by_type[tc.document_type].append(metrics)
                metrics_by_difficulty[tc.difficulty].append(metrics)

            except Exception as e:
                logger.error(f"Evaluation failed for test {tc.test_id}: {e}")
                report.failed_tests += 1

                # Create failed metrics
                failed_metrics = EvaluationMetrics(
                    test_id=tc.test_id,
                    passed=False,
                    errors=[f"Evaluation exception: {str(e)}"]
                )
                report.test_metrics.append(failed_metrics)

        # Calculate overall metrics
        if all_precisions:
            report.overall_precision = sum(all_precisions) / len(all_precisions)
            report.overall_recall = sum(all_recalls) / len(all_recalls)
            report.overall_f1 = sum(all_f1s) / len(all_f1s)
            report.overall_accuracy = sum(all_accuracies) / len(all_accuracies)

        # Calculate metrics by type
        for doc_type, metrics_list in metrics_by_type.items():
            type_precisions = [m.precision for m in metrics_list]
            type_recalls = [m.recall for m in metrics_list]
            type_f1s = [m.f1_score for m in metrics_list]

            report.metrics_by_type[doc_type] = {
                'precision': sum(type_precisions) / len(type_precisions) if type_precisions else 0.0,
                'recall': sum(type_recalls) / len(type_recalls) if type_recalls else 0.0,
                'f1': sum(type_f1s) / len(type_f1s) if type_f1s else 0.0,
                'count': len(metrics_list),
                'passed': sum(1 for m in metrics_list if m.passed)
            }

        # Calculate metrics by difficulty
        for difficulty, metrics_list in metrics_by_difficulty.items():
            passed_count = sum(1 for m in metrics_list if m.passed)
            report.metrics_by_difficulty[difficulty] = {
                'count': len(metrics_list),
                'passed': passed_count,
                'pass_rate': passed_count / len(metrics_list) if metrics_list else 0.0
            }

        # Finalize
        report.evaluation_time = time.time() - start_time
        self._update_stats(report)

        logger.info(f"Evaluation complete: {report.pass_rate*100:.1f}% pass rate")

        return report

    def _evaluate_test_case(
        self,
        test_case: GoldenTestCase,
        parsed_output: Dict[str, Any],
        strict: bool
    ) -> EvaluationMetrics:
        """Evaluate a single test case

        Args:
            test_case: Golden test case
            parsed_output: Parsed output to evaluate
            strict: Strict evaluation mode

        Returns:
            EvaluationMetrics
        """
        metrics = EvaluationMetrics(
            test_id=test_case.test_id,
            passed=False
        )

        expected = test_case.expected_output

        # Compare fields
        expected_fields = set(expected.keys())
        actual_fields = set(parsed_output.keys())

        metrics.total_fields = len(expected_fields)
        metrics.matched_fields = 0
        metrics.missing_fields = len(expected_fields - actual_fields)
        metrics.extra_fields = len(actual_fields - expected_fields)
        metrics.incorrect_fields = 0

        # Check each expected field
        for field in expected_fields:
            if field not in parsed_output:
                metrics.errors.append(f"Missing field: {field}")
                continue

            # Compare values
            expected_value = expected[field]
            actual_value = parsed_output[field]

            if self._compare_values(expected_value, actual_value, strict):
                metrics.matched_fields += 1
            else:
                metrics.incorrect_fields += 1
                metrics.errors.append(
                    f"Incorrect value for '{field}': expected {expected_value}, got {actual_value}"
                )

        # Report extra fields
        for field in actual_fields - expected_fields:
            if not strict:
                # Extra fields are okay in non-strict mode
                pass
            else:
                metrics.errors.append(f"Extra field: {field}")

        # Calculate metrics
        true_positives = metrics.matched_fields
        false_positives = metrics.incorrect_fields + (metrics.extra_fields if strict else 0)
        false_negatives = metrics.missing_fields

        # Precision = TP / (TP + FP)
        if true_positives + false_positives > 0:
            metrics.precision = true_positives / (true_positives + false_positives)
        else:
            metrics.precision = 0.0

        # Recall = TP / (TP + FN)
        if true_positives + false_negatives > 0:
            metrics.recall = true_positives / (true_positives + false_negatives)
        else:
            metrics.recall = 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        else:
            metrics.f1_score = 0.0

        # Accuracy = matched / total
        if metrics.total_fields > 0:
            metrics.accuracy = metrics.matched_fields / metrics.total_fields
        else:
            metrics.accuracy = 0.0

        # Determine if passed
        # Pass if: no missing fields, no incorrect fields, and (strict: no extra fields)
        if strict:
            metrics.passed = (
                metrics.missing_fields == 0 and
                metrics.incorrect_fields == 0 and
                metrics.extra_fields == 0
            )
        else:
            metrics.passed = (
                metrics.missing_fields == 0 and
                metrics.incorrect_fields == 0
            )

        return metrics

    def _compare_values(self, expected: Any, actual: Any, strict: bool) -> bool:
        """Compare expected and actual values

        Args:
            expected: Expected value
            actual: Actual value
            strict: Strict comparison mode

        Returns:
            True if values match
        """
        # Exact match
        if expected == actual:
            return True

        # Type mismatch
        if type(expected) != type(actual):
            # Allow string/int equivalence for numbers
            if isinstance(expected, (int, str)) and isinstance(actual, (int, str)):
                try:
                    return str(expected) == str(actual)
                except:
                    return False
            return False

        # List comparison
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(
                self._compare_values(e, a, strict)
                for e, a in zip(expected, actual)
            )

        # Dict comparison
        if isinstance(expected, dict) and isinstance(actual, dict):
            if strict:
                # Strict: all keys must match
                if set(expected.keys()) != set(actual.keys()):
                    return False
            else:
                # Non-strict: expected keys must be present
                if not all(k in actual for k in expected.keys()):
                    return False

            # Compare values for expected keys
            return all(
                self._compare_values(expected[k], actual.get(k), strict)
                for k in expected.keys()
            )

        # String comparison (case-insensitive for non-strict)
        if isinstance(expected, str) and isinstance(actual, str):
            if strict:
                return expected == actual
            else:
                return expected.lower().strip() == actual.lower().strip()

        # Default: exact match required
        return expected == actual

    def save_to_file(self, path: Path) -> None:
        """Save golden set to file

        Args:
            path: Path to save to
        """
        data = {
            'version': self.version,
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'metadata': {
                'total_test_cases': len(self.test_cases),
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved golden set to {path} ({len(self.test_cases)} test cases)")

    def load_from_file(self, path: Path) -> None:
        """Load golden set from file

        Args:
            path: Path to load from
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.version = data.get('version', '1.0')
        self.test_cases = [
            GoldenTestCase.from_dict(tc_data)
            for tc_data in data.get('test_cases', [])
        ]

        self.stats['total_test_cases'] = len(self.test_cases)

        logger.info(f"Loaded golden set from {path} ({len(self.test_cases)} test cases)")

    def export_failed_cases(self, report: GoldenSetReport, path: Path) -> None:
        """Export failed test cases for analysis

        Args:
            report: Evaluation report
            path: Path to export to
        """
        failed_cases = []

        for metrics in report.test_metrics:
            if not metrics.passed:
                test_case = self.get_test_case(metrics.test_id)
                if test_case:
                    failed_cases.append({
                        'test_case': test_case.to_dict(),
                        'metrics': metrics.to_dict()
                    })

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(failed_cases, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(failed_cases)} failed cases to {path}")

    def _update_stats(self, report: GoldenSetReport) -> None:
        """Update statistics"""
        self.stats['total_evaluations'] += 1
        self.stats['evaluation_time'] += report.evaluation_time

        # Update rolling average pass rate
        n = self.stats['total_evaluations']
        prev_avg = self.stats['average_pass_rate']
        self.stats['average_pass_rate'] = (
            ((n - 1) * prev_avg + report.pass_rate) / n
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_evaluations': 0,
            'total_test_cases': len(self.test_cases),
            'average_pass_rate': 0.0,
            'evaluation_time': 0.0,
        }
        logger.info("Statistics reset")


__all__ = [
    'GoldenSetManager',
    'GoldenTestCase',
    'GoldenSetReport',
    'EvaluationMetrics',
    'MetricType'
]
