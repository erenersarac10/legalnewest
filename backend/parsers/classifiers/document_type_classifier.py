"""Document Type Classifier - Harvey/Legora CTO-Level Production-Grade
Classifies Turkish legal documents into types

Production Features:
- Multiple document type classification (Law, Regulation, Decision, etc.)
- Pattern-based classification
- ML-based classification support
- Confidence scoring
- Turkish legal document structure detection
- Hybrid classification (pattern + ML)
- Multi-label classification
- Feature extraction
- Validation and normalization
- Performance metrics
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from collections import Counter, defaultdict
import time

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Turkish legal document types"""
    LAW = "LAW"  # Kanun
    REGULATION = "REGULATION"  # Yönetmelik
    DECREE = "DECREE"  # Kararname
    DECISION = "DECISION"  # Karar
    CIRCULAR = "CIRCULAR"  # Genelge
    COMMUNIQUE = "COMMUNIQUE"  # Tebliğ
    DIRECTIVE = "DIRECTIVE"  # Yönerge
    INSTRUCTION = "INSTRUCTION"  # Talimat
    BYLAW = "BYLAW"  # Tüzük
    ANNOUNCEMENT = "ANNOUNCEMENT"  # Duyuru
    OPINION = "OPINION"  # Görüş
    CASE_LAW = "CASE_LAW"  # İçtihat
    UNKNOWN = "UNKNOWN"  # Belirlenemedi


class ConfidenceLevel(Enum):
    """Classification confidence levels"""
    VERY_HIGH = "VERY_HIGH"  # >95%
    HIGH = "HIGH"  # 80-95%
    MEDIUM = "MEDIUM"  # 60-80%
    LOW = "LOW"  # 40-60%
    VERY_LOW = "VERY_LOW"  # <40%


@dataclass
class ClassificationResult:
    """Document type classification result"""
    document_type: DocumentType
    confidence: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel

    # Evidence
    matched_patterns: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)

    # Alternative classifications
    alternatives: List[Tuple[DocumentType, float]] = field(default_factory=list)

    # Metadata
    classification_method: str = "hybrid"  # pattern, ml, hybrid
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.document_type.value} ({self.confidence:.2%} - {self.confidence_level.value})"


class DocumentTypeClassifier:
    """Document Type Classifier for Turkish Legal Documents

    Classifies Turkish legal documents into types using:
    - Pattern-based classification (structural indicators)
    - Content-based classification (keywords and phrases)
    - Format-based classification (document structure)
    - ML-based classification (if model available)
    - Hybrid classification (combination)

    Supported Types:
    - Law (Kanun) - 5237 sayılı Türk Ceza Kanunu
    - Regulation (Yönetmelik) - KVKK Yönetmeliği
    - Decree (Kararname) - Cumhurbaşkanlığı Kararnamesi
    - Decision (Karar) - Yargıtay Kararı
    - Circular (Genelge) - Maliye Genelgesi
    - Communique (Tebliğ) - Vergi Tebliği
    - And more...
    """

    # Pattern definitions for each document type
    LAW_PATTERNS = [
        r'\b(?:\d+)\s+sayılı\s+(?:kanun|KANUN)\b',  # "6698 sayılı Kanun"
        r'\bKanun\s+Numarası\s*:\s*\d+\b',  # "Kanun Numarası: 5237"
        r'\bTürk\s+\w+\s+Kanunu\b',  # "Türk Ceza Kanunu"
        r'\b(?:TCK|TMK|TTK|TBK|HMK|CMK|İYUK|HUMK)\b',  # Law abbreviations
        r'\bKabul\s+[Tt]arihi\s*:',  # "Kabul Tarihi:"
        r'\bRESMİ\s+GAZETE\b.*?\bKanun',  # Official Gazette + Law
    ]

    REGULATION_PATTERNS = [
        r'\bYönetmelik\b',  # "Yönetmelik"
        r'\b\w+\s+Yönetmeliği\b',  # "KVKK Yönetmeliği"
        r'\bKurul\s+Kararı\s*:\s*\d+',  # Regulatory board decisions
        r'\bBakanlık.*?Yönetmelik',  # Ministry regulations
        r'\bAmaç\s+ve\s+[Kk]apsam',  # Purpose and scope section
        r'\b(?:BDDK|SPK|EPDK|BTK|KVKK)\s+Yönetmelik',  # Regulatory bodies
    ]

    DECREE_PATTERNS = [
        r'\bCumhurbaşkanlığı\s+Kararnamesi\b',  # Presidential decree
        r'\bKanun\s+Hükmünde\s+Kararname\b',  # Decree law (KHK)
        r'\bKHK\b',  # KHK abbreviation
        r'\b\d+\s+sayılı.*?Kararname',  # Numbered decree
    ]

    DECISION_PATTERNS = [
        r'\b(?:Yargıtay|YARGITAY)\b',  # Supreme Court
        r'\b(?:Danıştay|DANIŞTAY)\b',  # Council of State
        r'\bAnayasa\s+Mahkemesi\b',  # Constitutional Court
        r'\bE\.\s*\d+/\d+',  # Case number format (E. 2020/123)
        r'\bK\.\s*\d+/\d+',  # Decision number format (K. 2021/456)
        r'\b\d+\.\s+Hukuk\s+Dairesi\b',  # Chamber number
        r'\b(?:Karar|KARAR)\s+[Tt]arihi',  # Decision date
        r'\bDosya\s+No\s*:',  # File number
    ]

    CIRCULAR_PATTERNS = [
        r'\bGenelge\b',  # Circular
        r'\b\d+\s+sayılı\s+Genelge\b',  # Numbered circular
        r'\bMaliye\s+Bakanlığı\s+Genelge',  # Ministry of Finance circular
    ]

    COMMUNIQUE_PATTERNS = [
        r'\bTebliğ\b',  # Communique
        r'\b\w+\s+Tebliği\b',  # Named communique
        r'\bSeri\s+No\s*:\s*\w+',  # Series number
    ]

    DIRECTIVE_PATTERNS = [
        r'\bYönerge\b',  # Directive
    ]

    INSTRUCTION_PATTERNS = [
        r'\bTalimat\b',  # Instruction
    ]

    BYLAW_PATTERNS = [
        r'\bTüzük\b',  # Bylaw
    ]

    OPINION_PATTERNS = [
        r'\bGörüş\b',  # Opinion
        r'\bDanışma\s+Kurulu\s+Görüş',  # Advisory board opinion
    ]

    def __init__(self, ml_model: Optional[Any] = None):
        """Initialize Document Type Classifier

        Args:
            ml_model: Optional ML model for classification
        """
        self.ml_model = ml_model

        # Compile patterns for performance
        self.compiled_patterns = {
            DocumentType.LAW: [re.compile(p, re.IGNORECASE) for p in self.LAW_PATTERNS],
            DocumentType.REGULATION: [re.compile(p, re.IGNORECASE) for p in self.REGULATION_PATTERNS],
            DocumentType.DECREE: [re.compile(p, re.IGNORECASE) for p in self.DECREE_PATTERNS],
            DocumentType.DECISION: [re.compile(p, re.IGNORECASE) for p in self.DECISION_PATTERNS],
            DocumentType.CIRCULAR: [re.compile(p, re.IGNORECASE) for p in self.CIRCULAR_PATTERNS],
            DocumentType.COMMUNIQUE: [re.compile(p, re.IGNORECASE) for p in self.COMMUNIQUE_PATTERNS],
            DocumentType.DIRECTIVE: [re.compile(p, re.IGNORECASE) for p in self.DIRECTIVE_PATTERNS],
            DocumentType.INSTRUCTION: [re.compile(p, re.IGNORECASE) for p in self.INSTRUCTION_PATTERNS],
            DocumentType.BYLAW: [re.compile(p, re.IGNORECASE) for p in self.BYLAW_PATTERNS],
            DocumentType.OPINION: [re.compile(p, re.IGNORECASE) for p in self.OPINION_PATTERNS],
        }

        # Statistics
        self.stats = {
            'total_classifications': 0,
            'pattern_classifications': 0,
            'ml_classifications': 0,
            'hybrid_classifications': 0,
            'type_counts': defaultdict(int)
        }

        logger.info("Initialized DocumentTypeClassifier")

    def classify(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        method: str = "hybrid"
    ) -> ClassificationResult:
        """Classify document type

        Args:
            text: Document text (first 5000 chars usually sufficient)
            metadata: Optional metadata (title, source, etc.)
            method: Classification method ('pattern', 'ml', 'hybrid')

        Returns:
            ClassificationResult with type and confidence
        """
        start_time = time.time()

        # Normalize method
        method = method.lower()
        if method not in ['pattern', 'ml', 'hybrid']:
            method = 'hybrid'

        # Choose classification method
        if method == 'pattern' or (method == 'hybrid' and not self.ml_model):
            result = self._pattern_classify(text, metadata)
            result.classification_method = 'pattern'
        elif method == 'ml' and self.ml_model:
            result = self._ml_classify(text, metadata)
            result.classification_method = 'ml'
        elif method == 'hybrid' and self.ml_model:
            result = self._hybrid_classify(text, metadata)
            result.classification_method = 'hybrid'
        else:
            # Fallback to pattern
            result = self._pattern_classify(text, metadata)
            result.classification_method = 'pattern'

        # Set processing time
        result.processing_time = time.time() - start_time

        # Update stats
        self.stats['total_classifications'] += 1
        self.stats[f'{result.classification_method}_classifications'] += 1
        self.stats['type_counts'][result.document_type.value] += 1

        logger.info(f"Classified as {result}")

        return result

    def _pattern_classify(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> ClassificationResult:
        """Pattern-based classification

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            ClassificationResult
        """
        # Limit text for performance (first 5000 chars usually enough)
        text_sample = text[:5000]

        # Count pattern matches for each type
        type_scores = defaultdict(lambda: {'count': 0, 'patterns': []})

        for doc_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text_sample)
                if matches:
                    type_scores[doc_type]['count'] += len(matches)
                    type_scores[doc_type]['patterns'].append(pattern.pattern)

        # Check metadata for hints
        if metadata:
            self._apply_metadata_hints(metadata, type_scores)

        # Find best match
        if not type_scores:
            return ClassificationResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                matched_patterns=[],
                features={'method': 'pattern', 'matches': 0}
            )

        # Sort by count
        sorted_types = sorted(
            type_scores.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        best_type, best_info = sorted_types[0]
        total_matches = sum(info['count'] for _, info in type_scores.items())

        # Calculate confidence
        confidence = min(best_info['count'] / max(total_matches, 1), 1.0)

        # Boost confidence for strong indicators
        if best_info['count'] >= 3:
            confidence = min(confidence * 1.2, 1.0)

        # Build alternatives
        alternatives = [
            (doc_type, info['count'] / max(total_matches, 1))
            for doc_type, info in sorted_types[1:4]
        ]

        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            matched_patterns=best_info['patterns'],
            features={
                'method': 'pattern',
                'total_matches': total_matches,
                'type_matches': best_info['count']
            },
            alternatives=alternatives
        )

    def _ml_classify(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> ClassificationResult:
        """ML-based classification

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            ClassificationResult
        """
        if not self.ml_model:
            logger.warning("ML model not available, falling back to pattern classification")
            return self._pattern_classify(text, metadata)

        # Extract features
        features = self._extract_ml_features(text, metadata)

        # Predict with model
        try:
            predictions = self.ml_model.predict_proba([features])[0]
            classes = self.ml_model.classes_

            # Find best prediction
            best_idx = predictions.argmax()
            best_class = classes[best_idx]
            confidence = predictions[best_idx]

            # Map class to DocumentType
            try:
                doc_type = DocumentType[best_class]
            except KeyError:
                doc_type = DocumentType.UNKNOWN

            # Build alternatives
            sorted_indices = predictions.argsort()[::-1][1:4]
            alternatives = [
                (DocumentType[classes[idx]], predictions[idx])
                for idx in sorted_indices
                if predictions[idx] > 0.1
            ]

            return ClassificationResult(
                document_type=doc_type,
                confidence=float(confidence),
                confidence_level=self._get_confidence_level(confidence),
                features={'method': 'ml', 'model': str(type(self.ml_model).__name__)},
                alternatives=alternatives
            )

        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return self._pattern_classify(text, metadata)

    def _hybrid_classify(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> ClassificationResult:
        """Hybrid classification (pattern + ML)

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            ClassificationResult
        """
        # Get both classifications
        pattern_result = self._pattern_classify(text, metadata)
        ml_result = self._ml_classify(text, metadata)

        # Combine results
        if pattern_result.document_type == ml_result.document_type:
            # Agreement - boost confidence
            combined_confidence = min(
                (pattern_result.confidence + ml_result.confidence) / 1.5,
                1.0
            )

            return ClassificationResult(
                document_type=pattern_result.document_type,
                confidence=combined_confidence,
                confidence_level=self._get_confidence_level(combined_confidence),
                matched_patterns=pattern_result.matched_patterns,
                features={
                    'method': 'hybrid',
                    'pattern_confidence': pattern_result.confidence,
                    'ml_confidence': ml_result.confidence,
                    'agreement': True
                },
                alternatives=pattern_result.alternatives
            )
        else:
            # Disagreement - use higher confidence
            if pattern_result.confidence > ml_result.confidence:
                result = pattern_result
                result.features['disagreement'] = True
                result.features['ml_alternative'] = ml_result.document_type.value
                return result
            else:
                result = ml_result
                result.features['disagreement'] = True
                result.features['pattern_alternative'] = pattern_result.document_type.value
                return result

    def _apply_metadata_hints(
        self,
        metadata: Dict[str, Any],
        type_scores: Dict[DocumentType, Dict[str, Any]]
    ) -> None:
        """Apply metadata hints to classification scores

        Args:
            metadata: Document metadata
            type_scores: Current type scores
        """
        # Check title
        title = metadata.get('title', '').lower()

        if 'kanun' in title:
            type_scores[DocumentType.LAW]['count'] += 2
        if 'yönetmelik' in title:
            type_scores[DocumentType.REGULATION]['count'] += 2
        if 'karar' in title:
            type_scores[DocumentType.DECISION]['count'] += 1
        if 'genelge' in title:
            type_scores[DocumentType.CIRCULAR]['count'] += 2
        if 'tebliğ' in title:
            type_scores[DocumentType.COMMUNIQUE]['count'] += 2

        # Check source
        source = metadata.get('source', '').lower()

        if 'yargıtay' in source or 'danıştay' in source:
            type_scores[DocumentType.DECISION]['count'] += 2
        if 'cumhurbaşkanlığı' in source:
            type_scores[DocumentType.DECREE]['count'] += 1

    def _extract_ml_features(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features for ML classification

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            Feature dictionary
        """
        features = {}

        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Pattern presence features
        for doc_type, patterns in self.compiled_patterns.items():
            feature_name = f'has_{doc_type.value.lower()}_patterns'
            features[feature_name] = any(p.search(text[:5000]) for p in patterns)

        # Metadata features
        if metadata:
            features['has_title'] = bool(metadata.get('title'))
            features['has_source'] = bool(metadata.get('source'))
            features['has_date'] = bool(metadata.get('date'))

        return features

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to level

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            ConfidenceLevel
        """
        if confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def classify_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        method: str = "hybrid"
    ) -> List[ClassificationResult]:
        """Classify multiple documents

        Args:
            texts: List of document texts
            metadata_list: Optional list of metadata dicts
            method: Classification method

        Returns:
            List of ClassificationResults
        """
        if metadata_list is None:
            metadata_list = [None] * len(texts)

        results = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if i < len(metadata_list) else None
            result = self.classify(text, metadata, method)
            results.append(result)

        logger.info(f"Batch classified {len(texts)} documents")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics

        Returns:
            Statistics dictionary
        """
        return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_classifications': 0,
            'pattern_classifications': 0,
            'ml_classifications': 0,
            'hybrid_classifications': 0,
            'type_counts': defaultdict(int)
        }
        logger.info("Stats reset")


__all__ = ['DocumentTypeClassifier', 'DocumentType', 'ClassificationResult', 'ConfidenceLevel']
