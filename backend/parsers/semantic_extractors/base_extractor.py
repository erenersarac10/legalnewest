"""Base Extractor Interface - Harvey/Legora CTO-Level Production-Grade
Foundation for all semantic extractors in Turkish legal AI system

Production Features:
- Abstract base class for all extractors
- Standardized extraction interface
- Result validation framework
- Confidence scoring system
- Caching and performance optimization
- Error handling and logging
- Metadata tracking
- Batch processing support
"""
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for extraction results"""
    VERY_HIGH = "VERY_HIGH"  # >95% confidence
    HIGH = "HIGH"  # 80-95% confidence
    MEDIUM = "MEDIUM"  # 60-80% confidence
    LOW = "LOW"  # 40-60% confidence
    VERY_LOW = "VERY_LOW"  # <40% confidence


class ExtractionMethod(Enum):
    """Methods used for extraction"""
    REGEX_PATTERN = "REGEX_PATTERN"
    NLP_MODEL = "NLP_MODEL"
    RULE_BASED = "RULE_BASED"
    HYBRID = "HYBRID"
    DICTIONARY_LOOKUP = "DICTIONARY_LOOKUP"
    MACHINE_LEARNING = "MACHINE_LEARNING"


@dataclass
class ExtractionResult:
    """Standardized result from semantic extraction

    Attributes:
        value: Extracted value (string, dict, list, etc.)
        confidence: Confidence score (0.0 to 1.0)
        confidence_level: Categorical confidence level
        start_pos: Start position in source text (character index)
        end_pos: End position in source text (character index)
        context: Surrounding text context
        method: Method used for extraction
        metadata: Additional extraction metadata
        source_text: Original source text snippet
        validated: Whether result has been validated
        validation_errors: Any validation errors encountered
    """
    value: Any
    confidence: float
    confidence_level: ConfidenceLevel
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    context: Optional[str] = None
    method: ExtractionMethod = ExtractionMethod.REGEX_PATTERN
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_text: Optional[str] = None
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize confidence scores"""
        # Ensure confidence is in valid range
        if not 0.0 <= self.confidence <= 1.0:
            logger.warning(f"Confidence {self.confidence} out of range, clamping to [0, 1]")
            self.confidence = max(0.0, min(1.0, self.confidence))

        # Auto-set confidence level if not provided
        if isinstance(self.confidence_level, str):
            self.confidence_level = ConfidenceLevel(self.confidence_level)


@dataclass
class BatchExtractionResult:
    """Results from batch extraction operation

    Attributes:
        results: List of individual extraction results
        total_count: Total number of extractions
        success_count: Number of successful extractions
        failure_count: Number of failed extractions
        avg_confidence: Average confidence across all results
        processing_time: Total processing time in seconds
        metadata: Batch-level metadata
    """
    results: List[ExtractionResult]
    total_count: int
    success_count: int
    failure_count: int
    avg_confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base class for all semantic extractors

    Provides common functionality for:
    - Text extraction and normalization
    - Result validation and scoring
    - Caching and performance optimization
    - Error handling and logging
    - Batch processing
    - Metadata tracking

    Features:
    - Standardized extraction interface
    - Confidence scoring framework
    - Multiple extraction methods support
    - Context extraction
    - Result validation
    - Performance metrics tracking
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize base extractor

        Args:
            name: Extractor name
            version: Extractor version
        """
        self.name = name
        self.version = version
        self.cache = {}
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.info(f"Initialized {self.name} v{self.version}")

    @abstractmethod
    def extract(self, text: str, **kwargs) -> List[ExtractionResult]:
        """Extract semantic information from text

        Args:
            text: Input text to extract from
            **kwargs: Additional extraction parameters

        Returns:
            List of extraction results

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement extract() method")

    def extract_single(self, text: str, **kwargs) -> Optional[ExtractionResult]:
        """Extract single result with highest confidence

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            Highest confidence result or None
        """
        results = self.extract(text, **kwargs)
        if not results:
            return None

        # Return result with highest confidence
        return max(results, key=lambda r: r.confidence)

    def extract_batch(self, texts: List[str], **kwargs) -> BatchExtractionResult:
        """Extract from multiple texts in batch

        Args:
            texts: List of input texts
            **kwargs: Additional parameters

        Returns:
            Batch extraction results
        """
        start_time = datetime.now()
        all_results = []
        success_count = 0
        failure_count = 0

        for text in texts:
            try:
                results = self.extract(text, **kwargs)
                all_results.extend(results)
                if results:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                logger.error(f"Batch extraction failed for text: {e}")
                failure_count += 1

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Calculate average confidence
        avg_confidence = (
            sum(r.confidence for r in all_results) / len(all_results)
            if all_results else 0.0
        )

        return BatchExtractionResult(
            results=all_results,
            total_count=len(texts),
            success_count=success_count,
            failure_count=failure_count,
            avg_confidence=avg_confidence,
            processing_time=processing_time,
            metadata={
                'extractor': self.name,
                'version': self.version
            }
        )

    def validate_result(self, result: ExtractionResult) -> bool:
        """Validate an extraction result

        Args:
            result: Result to validate

        Returns:
            True if valid, False otherwise
        """
        errors = []

        # Check required fields
        if result.value is None:
            errors.append("Value is None")

        # Check confidence range
        if not 0.0 <= result.confidence <= 1.0:
            errors.append(f"Confidence {result.confidence} out of range [0, 1]")

        # Check position consistency
        if result.start_pos is not None and result.end_pos is not None:
            if result.start_pos > result.end_pos:
                errors.append(f"Start position {result.start_pos} > end position {result.end_pos}")

        result.validation_errors = errors
        result.validated = True

        if errors:
            logger.warning(f"Validation errors for {self.name}: {errors}")
            return False

        return True

    def calculate_confidence(
        self,
        match_quality: float,
        context_quality: float = 1.0,
        pattern_specificity: float = 1.0,
        **kwargs
    ) -> float:
        """Calculate confidence score for extraction

        Args:
            match_quality: Quality of the match (0-1)
            context_quality: Quality of surrounding context (0-1)
            pattern_specificity: Specificity of pattern used (0-1)
            **kwargs: Additional factors

        Returns:
            Combined confidence score (0-1)
        """
        # Weighted average of factors
        weights = {
            'match_quality': 0.5,
            'context_quality': 0.3,
            'pattern_specificity': 0.2
        }

        confidence = (
            weights['match_quality'] * match_quality +
            weights['context_quality'] * context_quality +
            weights['pattern_specificity'] * pattern_specificity
        )

        # Apply additional factors from kwargs
        for key, value in kwargs.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                confidence *= value

        return min(1.0, max(0.0, confidence))

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level

        Args:
            confidence: Numeric confidence (0-1)

        Returns:
            Categorical confidence level
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

    def extract_context(
        self,
        text: str,
        start_pos: int,
        end_pos: int,
        context_chars: int = 50
    ) -> str:
        """Extract surrounding context for a match

        Args:
            text: Full text
            start_pos: Match start position
            end_pos: Match end position
            context_chars: Characters to include before/after

        Returns:
            Context string with match highlighted
        """
        # Calculate context boundaries
        context_start = max(0, start_pos - context_chars)
        context_end = min(len(text), end_pos + context_chars)

        # Extract context
        before = text[context_start:start_pos]
        match = text[start_pos:end_pos]
        after = text[end_pos:context_end]

        # Format context with markers
        context = f"...{before}>>>{match}<<<{after}..."

        return context.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize text for extraction

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Normalize Turkish characters (optional, preserve by default)
        # text = text.replace('İ', 'I').replace('ı', 'i')

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def update_stats(self, success: bool):
        """Update extraction statistics

        Args:
            success: Whether extraction succeeded
        """
        self.stats['total_extractions'] += 1
        if success:
            self.stats['successful_extractions'] += 1
        else:
            self.stats['failed_extractions'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics

        Returns:
            Statistics dictionary
        """
        total = self.stats['total_extractions']
        success_rate = (
            self.stats['successful_extractions'] / total
            if total > 0 else 0.0
        )

        return {
            **self.stats,
            'success_rate': success_rate,
            'extractor': self.name,
            'version': self.version
        }

    def clear_cache(self):
        """Clear extraction cache"""
        self.cache.clear()
        logger.debug(f"Cleared cache for {self.name}")

    def reset_stats(self):
        """Reset extraction statistics"""
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.debug(f"Reset stats for {self.name}")

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"


class RegexExtractor(BaseExtractor):
    """Base class for regex-based extractors

    Provides common regex extraction functionality with:
    - Multiple pattern support
    - Pattern priority/ordering
    - Match filtering
    - Confidence scoring based on pattern specificity
    """

    def __init__(self, name: str, patterns: List[str], version: str = "1.0.0"):
        """Initialize regex extractor

        Args:
            name: Extractor name
            patterns: List of regex patterns (ordered by priority)
            version: Version string
        """
        super().__init__(name, version)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        logger.info(f"Initialized {name} with {len(patterns)} patterns")

    def extract_with_pattern(
        self,
        text: str,
        pattern_index: int = None,
        **kwargs
    ) -> List[ExtractionResult]:
        """Extract using specific pattern or all patterns

        Args:
            text: Input text
            pattern_index: Specific pattern index (None for all)
            **kwargs: Additional parameters

        Returns:
            List of extraction results
        """
        results = []
        patterns_to_use = (
            [self.patterns[pattern_index]] if pattern_index is not None
            else self.patterns
        )

        for idx, pattern in enumerate(patterns_to_use):
            actual_idx = pattern_index if pattern_index is not None else idx
            matches = pattern.finditer(text)

            for match in matches:
                # Calculate confidence based on pattern priority
                pattern_priority = 1.0 - (actual_idx / max(len(self.patterns), 1))
                match_quality = self._assess_match_quality(match, text)
                confidence = self.calculate_confidence(
                    match_quality=match_quality,
                    pattern_specificity=pattern_priority
                )

                # Extract context
                context = self.extract_context(
                    text,
                    match.start(),
                    match.end(),
                    context_chars=kwargs.get('context_chars', 50)
                )

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context,
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={
                        'pattern_index': actual_idx,
                        'groups': match.groups()
                    }
                )

                if self.validate_result(result):
                    results.append(result)

        self.update_stats(success=len(results) > 0)
        return results

    def _assess_match_quality(self, match: re.Match, text: str) -> float:
        """Assess quality of regex match

        Args:
            match: Regex match object
            text: Full text

        Returns:
            Quality score (0-1)
        """
        # Default: 1.0 for any match
        # Subclasses can override for more sophisticated assessment
        return 1.0


__all__ = [
    'BaseExtractor',
    'RegexExtractor',
    'ExtractionResult',
    'BatchExtractionResult',
    'ConfidenceLevel',
    'ExtractionMethod'
]
