"""Consistency Checker - Harvey/Legora CTO-Level Production-Grade
Checks consistency across multiple parsed documents for quality monitoring

Production Features:
- Cross-document consistency validation
- Batch consistency checking
- Statistical consistency analysis
- Consistency metrics
- Trend analysis
- Quality monitoring focus
- Turkish legal document conventions
- Production-grade reporting

Note: Different from validators/consistency_validator.py
- This module: Cross-document consistency for quality/monitoring
- Validator: Single document internal consistency
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import re
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)


class ConsistencyType(Enum):
    """Types of consistency checks"""
    FIELD_PRESENCE = "FIELD_PRESENCE"  # Field presence consistency
    FIELD_TYPE = "FIELD_TYPE"  # Field type consistency
    NAMING_CONVENTION = "NAMING_CONVENTION"  # Naming conventions
    VALUE_RANGE = "VALUE_RANGE"  # Value range consistency
    STRUCTURE = "STRUCTURE"  # Structural consistency
    TERMINOLOGY = "TERMINOLOGY"  # Terminology consistency


class ConsistencyLevel(Enum):
    """Consistency level categories"""
    HIGHLY_CONSISTENT = "HIGHLY_CONSISTENT"  # 90-100%
    CONSISTENT = "CONSISTENT"  # 75-90%
    MODERATELY_CONSISTENT = "MODERATELY_CONSISTENT"  # 50-75%
    INCONSISTENT = "INCONSISTENT"  # 25-50%
    HIGHLY_INCONSISTENT = "HIGHLY_INCONSISTENT"  # 0-25%


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue across documents"""
    consistency_type: ConsistencyType
    description: str
    affected_documents: int  # Number of documents affected
    total_documents: int  # Total documents checked
    consistency_rate: float  # 0.0 to 1.0
    examples: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def inconsistency_rate(self) -> float:
        """Get inconsistency rate"""
        return 1.0 - self.consistency_rate

    def __str__(self) -> str:
        return f"[{self.consistency_type.value}] {self.description} (consistency: {self.consistency_rate*100:.1f}%)"


@dataclass
class ConsistencyReport:
    """Complete consistency check report"""
    overall_consistency: float  # 0.0 to 1.0
    consistency_level: ConsistencyLevel
    issues: List[ConsistencyIssue] = field(default_factory=list)

    # Statistics
    total_documents: int = 0
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0

    # Breakdown by type
    consistency_by_type: Dict[str, float] = field(default_factory=dict)

    # Metadata
    check_time: float = 0.0
    checked_at: Optional[str] = None

    def add_issue(self, issue: ConsistencyIssue) -> None:
        """Add consistency issue"""
        self.issues.append(issue)

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Consistency Check: {self.consistency_level.value}")
        lines.append(f"Overall Consistency: {self.overall_consistency*100:.1f}%")
        lines.append(f"Documents Checked: {self.total_documents}")
        lines.append(f"Checks: {self.passed_checks}/{self.total_checks} passed")
        lines.append(f"Issues Found: {len(self.issues)}")
        lines.append(f"Check Time: {self.check_time:.3f}s")

        if self.consistency_by_type:
            lines.append(f"\nConsistency by Type:")
            for type_name, rate in sorted(self.consistency_by_type.items(), key=lambda x: x[1]):
                lines.append(f"  - {type_name}: {rate*100:.1f}%")

        if self.issues:
            lines.append(f"\nTop Issues:")
            # Sort by inconsistency rate
            sorted_issues = sorted(self.issues, key=lambda i: i.consistency_rate)
            for issue in sorted_issues[:5]:
                lines.append(f"  - {issue}")

        return '\n'.join(lines)


class ConsistencyChecker:
    """Consistency Checker for Turkish Legal Document Quality

    Checks consistency across multiple documents for quality monitoring:
    - Field presence consistency
    - Field type consistency
    - Naming convention consistency
    - Value range consistency
    - Structural consistency
    - Terminology consistency

    Features:
    - Cross-document analysis
    - Statistical consistency metrics
    - Trend detection
    - Turkish legal document conventions
    - Quality monitoring focus
    """

    # Expected field presence rates (what percentage of docs should have field)
    EXPECTED_FIELD_PRESENCE = {
        'law': {
            'law_number': 1.0,  # 100% of laws should have this
            'title': 1.0,
            'publication_date': 0.95,  # 95%+
            'articles': 0.95,
            'metadata': 0.8,  # 80%+
        },
        'regulation': {
            'regulation_number': 1.0,
            'title': 1.0,
            'authority': 0.95,
            'publication_date': 0.95,
            'articles': 0.95,
            'metadata': 0.8,
        },
        'decision': {
            'decision_number': 1.0,
            'court': 1.0,
            'date': 1.0,
            'subject': 0.9,
            'decision_text': 0.9,
            'metadata': 0.7,
        }
    }

    # Turkish legal terminology that should be consistent
    TERMINOLOGY_PATTERNS = {
        'article': [r'\bmadde\b', r'\bmd\.?\b'],
        'paragraph': [r'\bfıkra\b', r'\bf\.?\b'],
        'item': [r'\bbent\b', r'\bb\.?\b'],
    }

    def __init__(self, consistency_threshold: float = 0.75):
        """Initialize Consistency Checker

        Args:
            consistency_threshold: Minimum consistency rate to pass (default: 0.75)
        """
        self.consistency_threshold = consistency_threshold

        # Statistics
        self.stats = {
            'total_checks': 0,
            'total_documents_checked': 0,
            'average_consistency': 0.0,
            'check_time': 0.0,
        }

        logger.info(f"Initialized Consistency Checker (threshold: {consistency_threshold})")

    def check(self, documents: List[Any], **kwargs) -> ConsistencyReport:
        """Check consistency across documents

        Args:
            documents: List of parsed documents
            **kwargs: Options
                - strict: Strict checking mode (default: False)
                - min_documents: Minimum documents required (default: 2)

        Returns:
            ConsistencyReport with issues found
        """
        start_time = time.time()

        strict = kwargs.get('strict', False)
        min_documents = kwargs.get('min_documents', 2)

        logger.info(f"Checking consistency across {len(documents)} documents")

        # Create report
        report = ConsistencyReport(
            overall_consistency=1.0,
            consistency_level=ConsistencyLevel.HIGHLY_CONSISTENT,
            total_documents=len(documents)
        )

        # Validate input
        if len(documents) < min_documents:
            logger.warning(f"Insufficient documents for consistency check: {len(documents)} < {min_documents}")
            report.overall_consistency = 0.0
            report.consistency_level = ConsistencyLevel.HIGHLY_INCONSISTENT
            report.check_time = time.time() - start_time
            return report

        # Run consistency checks
        self._check_field_presence_consistency(documents, report, strict)
        self._check_field_type_consistency(documents, report, strict)
        self._check_naming_convention_consistency(documents, report, strict)
        self._check_value_range_consistency(documents, report)
        self._check_structural_consistency(documents, report)
        self._check_terminology_consistency(documents, report)

        # Calculate overall consistency
        if report.total_checks > 0:
            report.overall_consistency = report.passed_checks / report.total_checks
        else:
            report.overall_consistency = 1.0

        # Determine consistency level
        report.consistency_level = self._determine_consistency_level(report.overall_consistency)

        # Finalize
        report.check_time = time.time() - start_time
        self._update_stats(report)

        logger.info(f"Consistency check complete: {report.overall_consistency*100:.1f}% consistent")

        return report

    def _check_field_presence_consistency(
        self,
        documents: List[Any],
        report: ConsistencyReport,
        strict: bool
    ) -> None:
        """Check field presence consistency across documents"""

        if not documents:
            return

        # Group documents by type
        by_type = self._group_by_type(documents)

        for doc_type, docs in by_type.items():
            if doc_type not in self.EXPECTED_FIELD_PRESENCE:
                continue

            expected_fields = self.EXPECTED_FIELD_PRESENCE[doc_type]

            for field, expected_rate in expected_fields.items():
                # Count presence
                present_count = sum(
                    1 for doc in docs
                    if isinstance(doc, dict) and field in doc
                )

                actual_rate = present_count / len(docs) if docs else 0.0

                report.total_checks += 1

                # Check if meets expected rate
                tolerance = 0.05 if not strict else 0.0
                if actual_rate >= expected_rate - tolerance:
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1

                    # Add issue
                    issue = ConsistencyIssue(
                        consistency_type=ConsistencyType.FIELD_PRESENCE,
                        description=f"Field '{field}' presence inconsistent in {doc_type} documents",
                        affected_documents=len(docs) - present_count,
                        total_documents=len(docs),
                        consistency_rate=actual_rate,
                        suggestion=f"Expected {expected_rate*100:.0f}% of {doc_type} documents to have '{field}'"
                    )
                    report.add_issue(issue)

    def _check_field_type_consistency(
        self,
        documents: List[Any],
        report: ConsistencyReport,
        strict: bool
    ) -> None:
        """Check field type consistency across documents"""

        if not documents:
            return

        # Collect field types across all documents
        field_types: Dict[str, Counter] = defaultdict(Counter)

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            for field, value in doc.items():
                if value is not None:
                    field_types[field][type(value).__name__] += 1

        # Check consistency
        for field, type_counts in field_types.items():
            total_occurrences = sum(type_counts.values())

            if total_occurrences < 2:
                continue  # Skip fields that appear only once

            # Find most common type
            most_common_type, most_common_count = type_counts.most_common(1)[0]
            consistency_rate = most_common_count / total_occurrences

            report.total_checks += 1

            threshold = 0.9 if strict else 0.8

            if consistency_rate >= threshold:
                report.passed_checks += 1
            else:
                report.failed_checks += 1

                # Get examples of different types
                type_examples = [f"{t}: {c}" for t, c in type_counts.most_common()]

                issue = ConsistencyIssue(
                    consistency_type=ConsistencyType.FIELD_TYPE,
                    description=f"Field '{field}' has inconsistent types across documents",
                    affected_documents=total_occurrences - most_common_count,
                    total_documents=total_occurrences,
                    consistency_rate=consistency_rate,
                    examples=type_examples,
                    suggestion=f"Standardize '{field}' to type '{most_common_type}'"
                )
                report.add_issue(issue)

    def _check_naming_convention_consistency(
        self,
        documents: List[Any],
        report: ConsistencyReport,
        strict: bool
    ) -> None:
        """Check naming convention consistency"""

        if not documents:
            return

        # Collect all field names
        all_fields = set()
        for doc in documents:
            if isinstance(doc, dict):
                all_fields.update(doc.keys())

        # Check for mixed naming conventions (snake_case vs camelCase vs Turkish)
        snake_case_fields = [f for f in all_fields if '_' in f and f.islower()]
        camel_case_fields = [f for f in all_fields if not '_' in f and any(c.isupper() for c in f[1:])]
        turkish_fields = [
            f for f in all_fields
            if any(char in f for char in 'ğĞıİöÖüÜşŞçÇ')
        ]

        total_fields = len(all_fields)
        if total_fields == 0:
            return

        # Calculate predominant style
        styles = {
            'snake_case': len(snake_case_fields),
            'camelCase': len(camel_case_fields),
            'turkish': len(turkish_fields)
        }

        predominant_style = max(styles, key=styles.get)
        predominant_count = styles[predominant_style]

        consistency_rate = predominant_count / total_fields if total_fields > 0 else 0.0

        report.total_checks += 1

        threshold = 0.8 if strict else 0.7

        if consistency_rate >= threshold:
            report.passed_checks += 1
        else:
            report.failed_checks += 1

            issue = ConsistencyIssue(
                consistency_type=ConsistencyType.NAMING_CONVENTION,
                description="Mixed naming conventions across documents",
                affected_documents=len(documents),
                total_documents=len(documents),
                consistency_rate=consistency_rate,
                examples=[f"{style}: {count} fields" for style, count in styles.items() if count > 0],
                suggestion=f"Standardize to {predominant_style} naming convention"
            )
            report.add_issue(issue)

    def _check_value_range_consistency(
        self,
        documents: List[Any],
        report: ConsistencyReport
    ) -> None:
        """Check value range consistency for numeric fields"""

        if not documents:
            return

        # Collect numeric field values
        numeric_fields: Dict[str, List[float]] = defaultdict(list)

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            # Check article count
            if 'articles' in doc and isinstance(doc['articles'], list):
                numeric_fields['article_count'].append(len(doc['articles']))

            # Check other numeric fields
            for field, value in doc.items():
                if isinstance(value, (int, float)) and field not in ['article_count']:
                    numeric_fields[field].append(float(value))

        # Check for outliers using IQR method
        for field, values in numeric_fields.items():
            if len(values) < 4:
                continue  # Need at least 4 values for IQR

            # Calculate quartiles
            values_sorted = sorted(values)
            q1 = values_sorted[len(values_sorted) // 4]
            q3 = values_sorted[3 * len(values_sorted) // 4]
            iqr = q3 - q1

            # Outliers are values beyond 1.5 * IQR from quartiles
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            consistency_rate = 1.0 - (len(outliers) / len(values))

            report.total_checks += 1

            if consistency_rate >= 0.9:  # Less than 10% outliers
                report.passed_checks += 1
            else:
                report.failed_checks += 1

                issue = ConsistencyIssue(
                    consistency_type=ConsistencyType.VALUE_RANGE,
                    description=f"Field '{field}' has {len(outliers)} outlier values",
                    affected_documents=len(outliers),
                    total_documents=len(values),
                    consistency_rate=consistency_rate,
                    suggestion=f"Check documents with {field} outside range [{lower_bound:.2f}, {upper_bound:.2f}]",
                    metadata={
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'outlier_count': len(outliers)
                    }
                )
                report.add_issue(issue)

    def _check_structural_consistency(
        self,
        documents: List[Any],
        report: ConsistencyReport
    ) -> None:
        """Check structural consistency across documents"""

        if not documents:
            return

        # Group by type
        by_type = self._group_by_type(documents)

        for doc_type, docs in by_type.items():
            if len(docs) < 2:
                continue

            # Collect field sets from each document
            field_sets = []
            for doc in docs:
                if isinstance(doc, dict):
                    field_sets.append(set(doc.keys()))

            if not field_sets:
                continue

            # Find common fields
            common_fields = set.intersection(*field_sets)
            all_fields = set.union(*field_sets)

            # Calculate structural consistency
            # (what percentage of fields are common to all documents)
            if all_fields:
                consistency_rate = len(common_fields) / len(all_fields)
            else:
                consistency_rate = 1.0

            report.total_checks += 1

            if consistency_rate >= 0.7:  # At least 70% common fields
                report.passed_checks += 1
            else:
                report.failed_checks += 1

                # Find fields that are not common
                uncommon_fields = all_fields - common_fields
                examples = [
                    f"{field}: in {sum(1 for fs in field_sets if field in fs)}/{len(docs)} docs"
                    for field in sorted(uncommon_fields)[:5]
                ]

                issue = ConsistencyIssue(
                    consistency_type=ConsistencyType.STRUCTURE,
                    description=f"Structural inconsistency in {doc_type} documents",
                    affected_documents=len(docs),
                    total_documents=len(docs),
                    consistency_rate=consistency_rate,
                    examples=examples,
                    suggestion="Standardize document structure across all documents",
                    metadata={
                        'common_fields': len(common_fields),
                        'total_unique_fields': len(all_fields)
                    }
                )
                report.add_issue(issue)

    def _check_terminology_consistency(
        self,
        documents: List[Any],
        report: ConsistencyReport
    ) -> None:
        """Check terminology consistency across documents"""

        if not documents:
            return

        # Extract all text from documents
        all_texts = []
        for doc in documents:
            text = self._extract_text(doc)
            if text:
                all_texts.append(text)

        if not all_texts:
            return

        combined_text = ' '.join(all_texts)

        # Check each terminology pattern
        for term_name, patterns in self.TERMINOLOGY_PATTERNS.items():
            # Count usage of each pattern
            pattern_counts = {}
            total_matches = 0

            for pattern in patterns:
                matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                if matches > 0:
                    pattern_counts[pattern] = matches
                    total_matches += matches

            if total_matches < 5:
                continue  # Not enough data

            # Find most common pattern
            if pattern_counts:
                most_common_pattern = max(pattern_counts, key=pattern_counts.get)
                most_common_count = pattern_counts[most_common_pattern]
                consistency_rate = most_common_count / total_matches

                report.total_checks += 1

                if consistency_rate >= 0.8:  # 80% consistent
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1

                    examples = [
                        f"{pattern}: {count} times"
                        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1])
                    ]

                    issue = ConsistencyIssue(
                        consistency_type=ConsistencyType.TERMINOLOGY,
                        description=f"Inconsistent terminology for '{term_name}' across documents",
                        affected_documents=len(all_texts),
                        total_documents=len(all_texts),
                        consistency_rate=consistency_rate,
                        examples=examples,
                        suggestion=f"Standardize '{term_name}' references to '{most_common_pattern}'"
                    )
                    report.add_issue(issue)

    def _group_by_type(self, documents: List[Any]) -> Dict[str, List[Any]]:
        """Group documents by type"""
        by_type: Dict[str, List[Any]] = defaultdict(list)

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            doc_type = self._detect_document_type(doc)
            if doc_type:
                by_type[doc_type].append(doc)
            else:
                by_type['unknown'].append(doc)

        return by_type

    def _detect_document_type(self, data: Dict[str, Any]) -> Optional[str]:
        """Detect document type"""
        if 'law_number' in data or 'kanun_numarası' in data:
            return 'law'
        elif 'regulation_number' in data or 'yönetmelik_numarası' in data:
            return 'regulation'
        elif 'decision_number' in data or 'karar_numarası' in data:
            return 'decision'

        if 'metadata' in data and isinstance(data['metadata'], dict):
            doc_type = data['metadata'].get('document_type', '').lower()
            if 'kanun' in doc_type or 'law' in doc_type:
                return 'law'
            elif 'yönetmelik' in doc_type or 'regulation' in doc_type:
                return 'regulation'
            elif 'karar' in doc_type or 'decision' in doc_type:
                return 'decision'

        return None

    def _extract_text(self, data: Any) -> str:
        """Extract all text from document"""
        text_parts = []

        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            for field in ['content', 'text', 'title', 'decision_text']:
                if field in data and isinstance(data[field], str):
                    text_parts.append(data[field])

            if 'articles' in data and isinstance(data['articles'], list):
                for article in data['articles']:
                    if isinstance(article, dict) and 'content' in article:
                        text_parts.append(str(article['content']))

        return ' '.join(text_parts)

    def _determine_consistency_level(self, rate: float) -> ConsistencyLevel:
        """Determine consistency level from rate"""
        if rate >= 0.90:
            return ConsistencyLevel.HIGHLY_CONSISTENT
        elif rate >= 0.75:
            return ConsistencyLevel.CONSISTENT
        elif rate >= 0.50:
            return ConsistencyLevel.MODERATELY_CONSISTENT
        elif rate >= 0.25:
            return ConsistencyLevel.INCONSISTENT
        else:
            return ConsistencyLevel.HIGHLY_INCONSISTENT

    def _update_stats(self, report: ConsistencyReport) -> None:
        """Update statistics"""
        self.stats['total_checks'] += 1
        self.stats['total_documents_checked'] += report.total_documents
        self.stats['check_time'] += report.check_time

        # Update rolling average
        n = self.stats['total_checks']
        prev_avg = self.stats['average_consistency']
        self.stats['average_consistency'] = (
            ((n - 1) * prev_avg + report.overall_consistency) / n
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get checker statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_checks': 0,
            'total_documents_checked': 0,
            'average_consistency': 0.0,
            'check_time': 0.0,
        }
        logger.info("Statistics reset")


__all__ = [
    'ConsistencyChecker',
    'ConsistencyReport',
    'ConsistencyIssue',
    'ConsistencyType',
    'ConsistencyLevel'
]
