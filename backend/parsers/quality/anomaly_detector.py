"""Anomaly Detector - Harvey/Legora CTO-Level Production-Grade
Detects anomalies in parsed Turkish legal documents

Production Features:
- Statistical anomaly detection
- Pattern-based anomaly detection
- Turkish legal document anomaly patterns
- Outlier detection
- Anomaly scoring
- Alert generation
- Multi-method detection
- Baseline comparison
- Production-grade alerting
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import re
from collections import defaultdict
import math
import statistics

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    STRUCTURAL = "STRUCTURAL"  # Structure anomalies
    CONTENT = "CONTENT"  # Content anomalies
    STATISTICAL = "STATISTICAL"  # Statistical outliers
    PATTERN = "PATTERN"  # Pattern violations
    METADATA = "METADATA"  # Metadata anomalies
    TURKISH_LEGAL = "TURKISH_LEGAL"  # Turkish legal convention violations


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    CRITICAL = "CRITICAL"  # Critical anomaly
    HIGH = "HIGH"  # High severity
    MEDIUM = "MEDIUM"  # Medium severity
    LOW = "LOW"  # Low severity
    INFO = "INFO"  # Informational


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    location: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    score: float = 0.0  # Anomaly score (0-1, higher = more anomalous)
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value}] {self.anomaly_type.value}: {self.description}{loc}"


@dataclass
class AnomalyReport:
    """Complete anomaly detection report"""
    is_anomalous: bool
    anomalies: List[Anomaly] = field(default_factory=list)

    # Statistics
    total_checks: int = 0
    anomaly_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Overall anomaly score
    overall_score: float = 0.0  # 0-1, higher = more anomalous

    # Metadata
    detection_time: float = 0.0
    detected_at: Optional[str] = None

    def add_anomaly(self, anomaly: Anomaly) -> None:
        """Add anomaly to report"""
        self.anomalies.append(anomaly)
        self.anomaly_count += 1
        self.is_anomalous = True

        # Update severity counts
        if anomaly.severity == AnomalySeverity.CRITICAL:
            self.critical_count += 1
        elif anomaly.severity == AnomalySeverity.HIGH:
            self.high_count += 1
        elif anomaly.severity == AnomalySeverity.MEDIUM:
            self.medium_count += 1
        elif anomaly.severity == AnomalySeverity.LOW:
            self.low_count += 1

    def get_anomalies_by_type(self, anomaly_type: AnomalyType) -> List[Anomaly]:
        """Get anomalies of specific type"""
        return [a for a in self.anomalies if a.anomaly_type == anomaly_type]

    def get_anomalies_by_severity(self, severity: AnomalySeverity) -> List[Anomaly]:
        """Get anomalies of specific severity"""
        return [a for a in self.anomalies if a.severity == severity]

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Anomaly Detection: {'ANOMALIES FOUND' if self.is_anomalous else 'NORMAL'}")
        lines.append(f"Overall Score: {self.overall_score:.3f}")
        lines.append(f"Total Anomalies: {self.anomaly_count}")
        lines.append(f"Critical: {self.critical_count}, High: {self.high_count}, Medium: {self.medium_count}, Low: {self.low_count}")
        lines.append(f"Detection Time: {self.detection_time:.3f}s")

        if self.anomalies:
            lines.append(f"\nTop Anomalies:")
            # Sort by severity and score
            severity_order = {
                AnomalySeverity.CRITICAL: 0,
                AnomalySeverity.HIGH: 1,
                AnomalySeverity.MEDIUM: 2,
                AnomalySeverity.LOW: 3,
                AnomalySeverity.INFO: 4
            }
            sorted_anomalies = sorted(
                self.anomalies,
                key=lambda a: (severity_order[a.severity], -a.score)
            )
            for anomaly in sorted_anomalies[:5]:
                lines.append(f"  - {anomaly}")

        return '\n'.join(lines)


class AnomalyDetector:
    """Anomaly Detector for Turkish Legal Documents

    Detects anomalies in parsed documents:
    - Structural anomalies (missing fields, invalid structure)
    - Content anomalies (unusual text patterns)
    - Statistical outliers (unusual values)
    - Pattern violations (Turkish legal conventions)
    - Metadata anomalies

    Features:
    - Multiple detection methods
    - Baseline comparison
    - Turkish legal document patterns
    - Severity assessment
    - Alert generation
    """

    # Expected article count ranges by document type
    ARTICLE_COUNT_RANGES = {
        'law': (1, 500),  # Laws typically have 1-500 articles
        'regulation': (1, 300),  # Regulations 1-300 articles
        'decision': (0, 50),  # Decisions may have 0-50 articles
    }

    # Expected text length ranges (characters)
    TEXT_LENGTH_RANGES = {
        'law': (500, 1000000),
        'regulation': (500, 500000),
        'decision': (100, 100000),
    }

    # Turkish legal document patterns that should NOT appear
    BAD_PATTERNS = [
        r'<.*?>',  # HTML tags
        r'\?\?\?+',  # Multiple question marks
        r'XXX+',  # Placeholder text
        r'TODO|FIXME|HACK',  # Developer notes
        r'\[DELETE\]|\[REMOVE\]',  # Edit markers
        r'ğğğ+|ııı+|ööö+|üüü+|ççç+|şşş+',  # Keyboard mashing in Turkish
    ]

    # Turkish legal patterns that SHOULD appear (for laws/regulations)
    GOOD_PATTERNS = {
        'law': [r'\bmadde\b', r'\bkanun\b'],
        'regulation': [r'\bmadde\b', r'\byönetmelik\b'],
        'decision': [r'\bkarar\b', r'\bmahkeme\b'],
    }

    def __init__(self, baseline_stats: Optional[Dict[str, Any]] = None):
        """Initialize Anomaly Detector

        Args:
            baseline_stats: Baseline statistics for comparison
        """
        self.baseline_stats = baseline_stats or {}

        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_anomalies_found': 0,
            'anomaly_rate': 0.0,
            'detection_time': 0.0,
            'anomaly_type_counts': defaultdict(int),
        }

        logger.info("Initialized Anomaly Detector")

    def detect(self, data: Any, **kwargs) -> AnomalyReport:
        """Detect anomalies in document

        Args:
            data: Document data
            **kwargs: Options
                - threshold: Anomaly score threshold (default: 0.5)
                - strict: Strict detection mode (default: False)
                - methods: List of methods to use (default: all)

        Returns:
            AnomalyReport with detected anomalies
        """
        start_time = time.time()

        threshold = kwargs.get('threshold', 0.5)
        strict = kwargs.get('strict', False)
        methods = kwargs.get('methods', ['all'])

        logger.info("Starting anomaly detection")

        # Create report
        report = AnomalyReport(is_anomalous=False)

        # Run detection methods
        if 'all' in methods or 'structural' in methods:
            self._detect_structural_anomalies(data, report, strict)

        if 'all' in methods or 'content' in methods:
            self._detect_content_anomalies(data, report, strict)

        if 'all' in methods or 'statistical' in methods:
            self._detect_statistical_anomalies(data, report, threshold)

        if 'all' in methods or 'pattern' in methods:
            self._detect_pattern_anomalies(data, report, strict)

        if 'all' in methods or 'metadata' in methods:
            self._detect_metadata_anomalies(data, report, strict)

        if 'all' in methods or 'turkish_legal' in methods:
            self._detect_turkish_legal_anomalies(data, report, strict)

        # Calculate overall anomaly score
        if report.anomalies:
            report.overall_score = sum(a.score for a in report.anomalies) / len(report.anomalies)

        # Finalize
        report.detection_time = time.time() - start_time
        self._update_stats(report)

        logger.info(f"Detection complete: {report.anomaly_count} anomalies found")

        return report

    def detect_batch(self, data_list: List[Any], **kwargs) -> List[AnomalyReport]:
        """Detect anomalies in multiple documents

        Args:
            data_list: List of documents
            **kwargs: Options

        Returns:
            List of AnomalyReports
        """
        reports = []

        for i, data in enumerate(data_list):
            try:
                report = self.detect(data, **kwargs)
                reports.append(report)
            except Exception as e:
                logger.error(f"Detection failed for item {i}: {e}")
                # Create error report
                error_report = AnomalyReport(is_anomalous=True)
                error_report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.STRUCTURAL,
                    severity=AnomalySeverity.CRITICAL,
                    description=f"Detection failed with exception: {str(e)}",
                    score=1.0
                ))
                reports.append(error_report)

        logger.info(f"Batch detection complete: {len(reports)} documents")
        return reports

    def _detect_structural_anomalies(
        self,
        data: Any,
        report: AnomalyReport,
        strict: bool
    ) -> None:
        """Detect structural anomalies"""

        # Check if data is a dict
        if not isinstance(data, dict):
            report.add_anomaly(Anomaly(
                anomaly_type=AnomalyType.STRUCTURAL,
                severity=AnomalySeverity.CRITICAL,
                description=f"Document is not a dictionary (type: {type(data).__name__})",
                expected_value="dict",
                actual_value=type(data).__name__,
                score=1.0,
                suggestion="Ensure document is parsed as a dictionary"
            ))
            return

        # Detect document type
        doc_type = self._detect_document_type(data)

        # Check for required fields
        if doc_type:
            required_fields = self._get_required_fields(doc_type)
            for field in required_fields:
                if field not in data:
                    severity = AnomalySeverity.HIGH if strict else AnomalySeverity.MEDIUM
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.STRUCTURAL,
                        severity=severity,
                        description=f"Missing required field: {field}",
                        location="root",
                        expected_value=field,
                        score=0.8,
                        suggestion=f"Add '{field}' field to document"
                    ))

        # Check article count
        if 'articles' in data and isinstance(data['articles'], list):
            article_count = len(data['articles'])

            if doc_type and doc_type in self.ARTICLE_COUNT_RANGES:
                min_count, max_count = self.ARTICLE_COUNT_RANGES[doc_type]

                if article_count < min_count:
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.STRUCTURAL,
                        severity=AnomalySeverity.MEDIUM,
                        description=f"Unusually few articles for {doc_type}: {article_count}",
                        location="articles",
                        expected_value=f"at least {min_count}",
                        actual_value=article_count,
                        score=0.6,
                        suggestion="Verify document parsing is complete"
                    ))
                elif article_count > max_count:
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.STRUCTURAL,
                        severity=AnomalySeverity.LOW,
                        description=f"Unusually many articles for {doc_type}: {article_count}",
                        location="articles",
                        expected_value=f"at most {max_count}",
                        actual_value=article_count,
                        score=0.4
                    ))

            # Check for empty articles
            empty_count = sum(
                1 for a in data['articles']
                if isinstance(a, dict) and not a.get('content')
            )

            if empty_count > 0:
                ratio = empty_count / article_count
                if ratio > 0.1:  # More than 10% empty
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.STRUCTURAL,
                        severity=AnomalySeverity.HIGH,
                        description=f"{empty_count} articles ({ratio*100:.1f}%) have no content",
                        location="articles",
                        score=ratio,
                        suggestion="Verify article content is being parsed correctly"
                    ))

    def _detect_content_anomalies(
        self,
        data: Any,
        report: AnomalyReport,
        strict: bool
    ) -> None:
        """Detect content anomalies"""

        if not isinstance(data, dict):
            return

        # Extract text
        text = self._extract_text(data)
        doc_type = self._detect_document_type(data)

        if not text:
            report.add_anomaly(Anomaly(
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.HIGH,
                description="Document has no text content",
                score=0.9,
                suggestion="Verify document parsing is extracting text correctly"
            ))
            return

        # Check text length
        text_len = len(text)
        if doc_type and doc_type in self.TEXT_LENGTH_RANGES:
            min_len, max_len = self.TEXT_LENGTH_RANGES[doc_type]

            if text_len < min_len:
                report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.CONTENT,
                    severity=AnomalySeverity.MEDIUM,
                    description=f"Text unusually short for {doc_type}: {text_len} characters",
                    expected_value=f"at least {min_len}",
                    actual_value=text_len,
                    score=0.6,
                    suggestion="Verify document content is complete"
                ))
            elif text_len > max_len:
                report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.CONTENT,
                    severity=AnomalySeverity.LOW,
                    description=f"Text unusually long for {doc_type}: {text_len} characters",
                    expected_value=f"at most {max_len}",
                    actual_value=text_len,
                    score=0.3
                ))

        # Check for bad patterns
        for pattern in self.BAD_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                severity = AnomalySeverity.HIGH if strict else AnomalySeverity.MEDIUM
                report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.CONTENT,
                    severity=severity,
                    description=f"Found {len(matches)} instances of bad pattern: {pattern}",
                    score=min(len(matches) / 10, 1.0),
                    suggestion="Remove placeholder or invalid text",
                    metadata={'matches': matches[:5]}  # First 5 matches
                ))

        # Check character distribution (detect encoding issues)
        if self._has_encoding_issues(text):
            report.add_anomaly(Anomaly(
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Potential encoding issues detected",
                score=0.5,
                suggestion="Verify document encoding (should be UTF-8 for Turkish)"
            ))

    def _detect_statistical_anomalies(
        self,
        data: Any,
        report: AnomalyReport,
        threshold: float
    ) -> None:
        """Detect statistical outliers"""

        if not isinstance(data, dict):
            return

        if not self.baseline_stats:
            # No baseline to compare against
            return

        # Compare article count
        if 'articles' in data and isinstance(data['articles'], list):
            article_count = len(data['articles'])

            if 'article_count_mean' in self.baseline_stats:
                mean = self.baseline_stats['article_count_mean']
                std = self.baseline_stats.get('article_count_std', mean * 0.5)

                # Calculate z-score
                z_score = abs((article_count - mean) / std) if std > 0 else 0

                if z_score > 3:  # 3 standard deviations
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.STATISTICAL,
                        severity=AnomalySeverity.MEDIUM,
                        description=f"Article count is statistical outlier (z-score: {z_score:.2f})",
                        expected_value=f"mean: {mean:.0f} ± {std:.0f}",
                        actual_value=article_count,
                        score=min(z_score / 5, 1.0),
                        suggestion="Verify this document is of the expected type"
                    ))

        # Compare text length
        text = self._extract_text(data)
        if text and 'text_length_mean' in self.baseline_stats:
            text_len = len(text)
            mean = self.baseline_stats['text_length_mean']
            std = self.baseline_stats.get('text_length_std', mean * 0.5)

            z_score = abs((text_len - mean) / std) if std > 0 else 0

            if z_score > 3:
                report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=AnomalySeverity.LOW,
                    description=f"Text length is statistical outlier (z-score: {z_score:.2f})",
                    expected_value=f"mean: {mean:.0f} ± {std:.0f}",
                    actual_value=text_len,
                    score=min(z_score / 5, 1.0)
                ))

    def _detect_pattern_anomalies(
        self,
        data: Any,
        report: AnomalyReport,
        strict: bool
    ) -> None:
        """Detect pattern violations"""

        if not isinstance(data, dict):
            return

        text = self._extract_text(data)
        doc_type = self._detect_document_type(data)

        if not text or not doc_type:
            return

        # Check for expected patterns
        if doc_type in self.GOOD_PATTERNS:
            for pattern in self.GOOD_PATTERNS[doc_type]:
                matches = re.findall(pattern, text, re.IGNORECASE)

                if not matches:
                    severity = AnomalySeverity.MEDIUM if strict else AnomalySeverity.LOW
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.PATTERN,
                        severity=severity,
                        description=f"Missing expected pattern for {doc_type}: {pattern}",
                        score=0.5,
                        suggestion=f"Verify this is a valid {doc_type} document"
                    ))

    def _detect_metadata_anomalies(
        self,
        data: Any,
        report: AnomalyReport,
        strict: bool
    ) -> None:
        """Detect metadata anomalies"""

        if not isinstance(data, dict):
            return

        # Check for metadata field
        if 'metadata' not in data:
            if strict:
                report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.METADATA,
                    severity=AnomalySeverity.LOW,
                    description="Document has no metadata field",
                    score=0.3,
                    suggestion="Add metadata field with document information"
                ))
            return

        metadata = data['metadata']
        if not isinstance(metadata, dict):
            report.add_anomaly(Anomaly(
                anomaly_type=AnomalyType.METADATA,
                severity=AnomalySeverity.MEDIUM,
                description=f"Metadata is not a dictionary (type: {type(metadata).__name__})",
                score=0.6
            ))
            return

        # Check for empty metadata
        if not metadata:
            report.add_anomaly(Anomaly(
                anomaly_type=AnomalyType.METADATA,
                severity=AnomalySeverity.LOW,
                description="Metadata is empty",
                score=0.4,
                suggestion="Add document metadata"
            ))

    def _detect_turkish_legal_anomalies(
        self,
        data: Any,
        report: AnomalyReport,
        strict: bool
    ) -> None:
        """Detect Turkish legal convention violations"""

        if not isinstance(data, dict):
            return

        text = self._extract_text(data)
        doc_type = self._detect_document_type(data)

        # Check for mixed Turkish/English in legal terms
        madde_count = len(re.findall(r'\bmadde\b', text, re.IGNORECASE))
        article_count = len(re.findall(r'\barticle\b', text, re.IGNORECASE))

        if madde_count > 0 and article_count > 0:
            ratio = min(madde_count, article_count) / max(madde_count, article_count)
            if ratio > 0.1:  # Significant mixing
                report.add_anomaly(Anomaly(
                    anomaly_type=AnomalyType.TURKISH_LEGAL,
                    severity=AnomalySeverity.LOW,
                    description=f"Mixed Turkish/English article references (madde: {madde_count}, article: {article_count})",
                    score=ratio,
                    suggestion="Standardize to Turkish 'madde' for Turkish legal documents"
                ))

        # Check article numbering in Turkish legal format
        if 'articles' in data and isinstance(data['articles'], list):
            articles = data['articles']

            # Check for gaps in numbering
            numbers = []
            for article in articles:
                if isinstance(article, dict):
                    num = article.get('number', article.get('article_number'))
                    if num is not None:
                        try:
                            numbers.append(int(num))
                        except (ValueError, TypeError):
                            pass

            if numbers:
                numbers.sort()
                gaps = []
                for i in range(len(numbers) - 1):
                    if numbers[i+1] - numbers[i] > 1:
                        gaps.append((numbers[i], numbers[i+1]))

                if gaps and len(gaps) > len(numbers) * 0.1:  # More than 10% have gaps
                    report.add_anomaly(Anomaly(
                        anomaly_type=AnomalyType.TURKISH_LEGAL,
                        severity=AnomalySeverity.LOW,
                        description=f"Found {len(gaps)} gaps in article numbering",
                        score=len(gaps) / len(numbers),
                        suggestion="Verify article numbering includes temporary/additional articles",
                        metadata={'gaps': gaps[:5]}  # First 5 gaps
                    ))

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

    def _get_required_fields(self, doc_type: str) -> List[str]:
        """Get required fields for document type"""
        required = {
            'law': ['law_number', 'title', 'articles'],
            'regulation': ['regulation_number', 'title', 'articles'],
            'decision': ['decision_number', 'court', 'date'],
        }
        return required.get(doc_type, [])

    def _extract_text(self, data: Any) -> str:
        """Extract all text from document"""
        text_parts = []

        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            for field in ['content', 'text', 'title', 'decision_text', 'description']:
                if field in data and isinstance(data[field], str):
                    text_parts.append(data[field])

            if 'articles' in data and isinstance(data['articles'], list):
                for article in data['articles']:
                    if isinstance(article, dict):
                        if 'content' in article:
                            text_parts.append(str(article['content']))
                        if 'title' in article:
                            text_parts.append(str(article['title']))

        return ' '.join(text_parts)

    def _has_encoding_issues(self, text: str) -> bool:
        """Check for encoding issues"""
        # Turkish characters
        turkish_chars = set('ğĞıİöÖüÜşŞçÇ')

        # Check for common encoding issue patterns
        bad_patterns = [
            'Ä±', 'Ã§', 'Ã¶', 'Ã¼', 'ÄŸ', 'ÅŸ',  # UTF-8 as Latin-1
            '�',  # Replacement character
        ]

        for pattern in bad_patterns:
            if pattern in text:
                return True

        return False

    def set_baseline(self, baseline_stats: Dict[str, Any]) -> None:
        """Set baseline statistics

        Args:
            baseline_stats: Baseline statistics dictionary
        """
        self.baseline_stats = baseline_stats
        logger.info("Baseline statistics updated")

    def calculate_baseline(self, documents: List[Any]) -> Dict[str, Any]:
        """Calculate baseline statistics from document set

        Args:
            documents: List of documents

        Returns:
            Baseline statistics dictionary
        """
        article_counts = []
        text_lengths = []

        for doc in documents:
            if isinstance(doc, dict):
                if 'articles' in doc and isinstance(doc['articles'], list):
                    article_counts.append(len(doc['articles']))

                text = self._extract_text(doc)
                if text:
                    text_lengths.append(len(text))

        baseline = {}

        if article_counts:
            baseline['article_count_mean'] = statistics.mean(article_counts)
            if len(article_counts) > 1:
                baseline['article_count_std'] = statistics.stdev(article_counts)
            baseline['article_count_min'] = min(article_counts)
            baseline['article_count_max'] = max(article_counts)

        if text_lengths:
            baseline['text_length_mean'] = statistics.mean(text_lengths)
            if len(text_lengths) > 1:
                baseline['text_length_std'] = statistics.stdev(text_lengths)
            baseline['text_length_min'] = min(text_lengths)
            baseline['text_length_max'] = max(text_lengths)

        baseline['sample_size'] = len(documents)

        logger.info(f"Calculated baseline from {len(documents)} documents")
        return baseline

    def _update_stats(self, report: AnomalyReport) -> None:
        """Update statistics"""
        self.stats['total_detections'] += 1
        self.stats['total_anomalies_found'] += report.anomaly_count
        self.stats['detection_time'] += report.detection_time

        # Update anomaly type counts
        for anomaly in report.anomalies:
            self.stats['anomaly_type_counts'][anomaly.anomaly_type.value] += 1

        # Update anomaly rate
        self.stats['anomaly_rate'] = (
            self.stats['total_anomalies_found'] / self.stats['total_detections']
            if self.stats['total_detections'] > 0 else 0.0
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_detections': 0,
            'total_anomalies_found': 0,
            'anomaly_rate': 0.0,
            'detection_time': 0.0,
            'anomaly_type_counts': defaultdict(int),
        }
        logger.info("Statistics reset")


__all__ = [
    'AnomalyDetector',
    'AnomalyReport',
    'Anomaly',
    'AnomalyType',
    'AnomalySeverity'
]
