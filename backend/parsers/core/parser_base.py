"""
Base Parser Interface - Harvey/Legora CTO-Level

Abstract base class for all legal document parsers.
Defines the contract that all parsers must implement.

Parser Types:
    - SourceAdapter: Fetch documents from external sources (Resmi Gazete, Yargıtay, etc.)
    - StructuralParser: Extract document structure (clauses, sections, hierarchy)
    - SemanticExtractor: Extract semantic information (citations, entities, dates)
    - Validator: Validate parsed documents

Design Patterns:
    - Template Method: parse() orchestrates parsing steps
    - Strategy: Different parsing strategies per document type
    - Chain of Responsibility: Validators chain together

Author: Legal AI Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
import logging
from time import time

from .canonical_schema import (
    LegalDocument,
    Statute,
    CourtDecision,
    ParsingResult,
    DocumentType,
)
from .exceptions import (
    ParserError,
    ParsingError,
    ValidationError,
    wrap_exception,
)


logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    Abstract base parser.
    All parsers inherit from this class.
    """

    def __init__(
        self,
        parser_name: str,
        parser_version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize parser.

        Args:
            parser_name: Name of this parser
            parser_version: Version string
            config: Optional configuration dictionary
        """
        self.parser_name = parser_name
        self.parser_version = parser_version
        self.config = config or {}

        # Parsing statistics
        self.stats = {
            "total_parsed": 0,
            "total_failed": 0,
            "total_duration_ms": 0.0,
        }

        logger.info(f"Initialized parser: {parser_name} v{parser_version}")

    # ========================================================================
    # TEMPLATE METHOD - Main parsing workflow
    # ========================================================================

    def parse(
        self,
        source: Any,
        document_type: Optional[DocumentType] = None,
        **kwargs
    ) -> ParsingResult:
        """
        Parse a legal document (Template Method).

        This method orchestrates the parsing workflow:
        1. Preprocess input
        2. Extract raw data
        3. Transform to canonical format
        4. Validate
        5. Post-process

        Args:
            source: Input source (URL, file path, raw text, etc.)
            document_type: Expected document type
            **kwargs: Additional parser-specific arguments

        Returns:
            ParsingResult with parsed document or errors
        """
        start_time = time()
        errors: List[str] = []
        warnings: List[str] = []

        try:
            logger.info(f"[{self.parser_name}] Starting parse: {source}")

            # Step 1: Preprocess
            preprocessed = self._preprocess(source, **kwargs)

            # Step 2: Extract raw data
            raw_data = self._extract_raw_data(preprocessed, **kwargs)

            # Step 3: Transform to canonical
            document = self._transform_to_canonical(raw_data, document_type, **kwargs)

            # Step 4: Validate
            validation_errors, validation_warnings = self._validate(document)
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)

            # Step 5: Post-process
            document = self._postprocess(document, **kwargs)

            # Calculate metrics
            duration_ms = (time() - start_time) * 1000
            confidence = self._calculate_confidence(document, errors, warnings)
            completeness = self._calculate_completeness(document)

            # Update stats
            self.stats["total_parsed"] += 1
            self.stats["total_duration_ms"] += duration_ms

            logger.info(
                f"[{self.parser_name}] Parse complete: {duration_ms:.2f}ms, "
                f"confidence={confidence:.2f}, completeness={completeness:.2f}"
            )

            return ParsingResult(
                success=True,
                document=document,
                parser_name=self.parser_name,
                parsing_duration_ms=duration_ms,
                confidence_score=confidence,
                completeness_score=completeness,
                errors=errors,
                warnings=warnings,
                source_url=self._get_source_url(source),
            )

        except ParserError as e:
            # Known parser error
            self.stats["total_failed"] += 1
            duration_ms = (time() - start_time) * 1000

            logger.error(f"[{self.parser_name}] Parse failed: {e.message}")

            return ParsingResult(
                success=False,
                document=None,
                parser_name=self.parser_name,
                parsing_duration_ms=duration_ms,
                confidence_score=0.0,
                completeness_score=0.0,
                errors=[e.message] + errors,
                warnings=warnings,
                source_url=self._get_source_url(source),
            )

        except Exception as e:
            # Unexpected error
            self.stats["total_failed"] += 1
            duration_ms = (time() - start_time) * 1000

            wrapped = wrap_exception(e, {"parser": self.parser_name, "source": str(source)})
            logger.exception(f"[{self.parser_name}] Unexpected error: {wrapped.message}")

            return ParsingResult(
                success=False,
                document=None,
                parser_name=self.parser_name,
                parsing_duration_ms=duration_ms,
                confidence_score=0.0,
                completeness_score=0.0,
                errors=[wrapped.message],
                warnings=warnings,
                source_url=self._get_source_url(source),
            )

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        """
        Extract raw data from preprocessed input.

        Args:
            preprocessed: Preprocessed input
            **kwargs: Additional arguments

        Returns:
            Dictionary of extracted raw data

        Raises:
            ParsingError: If extraction fails
        """
        pass

    @abstractmethod
    def _transform_to_canonical(
        self,
        raw_data: Dict[str, Any],
        document_type: Optional[DocumentType],
        **kwargs
    ) -> LegalDocument:
        """
        Transform raw data to canonical LegalDocument format.

        Args:
            raw_data: Raw extracted data
            document_type: Expected document type
            **kwargs: Additional arguments

        Returns:
            LegalDocument (or subclass like Statute, CourtDecision)

        Raises:
            TransformationError: If transformation fails
        """
        pass

    # ========================================================================
    # HOOK METHODS - Can be overridden by subclasses
    # ========================================================================

    def _preprocess(self, source: Any, **kwargs) -> Any:
        """
        Preprocess input before extraction.
        Default: return source as-is.

        Args:
            source: Raw input
            **kwargs: Additional arguments

        Returns:
            Preprocessed input
        """
        return source

    def _postprocess(self, document: LegalDocument, **kwargs) -> LegalDocument:
        """
        Post-process document after validation.
        Default: return document as-is.

        Args:
            document: Parsed document
            **kwargs: Additional arguments

        Returns:
            Post-processed document
        """
        return document

    def _validate(self, document: LegalDocument) -> tuple[List[str], List[str]]:
        """
        Validate parsed document.
        Default: basic validation.

        Args:
            document: Document to validate

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check required fields
        if not document.title or not document.title.strip():
            errors.append("Document title is empty")

        if not document.full_text or not document.full_text.strip():
            errors.append("Document text is empty")

        if not document.metadata:
            errors.append("Document metadata is missing")

        # Collect validation errors from document itself
        if document.validation_errors:
            errors.extend(document.validation_errors)

        if document.extraction_warnings:
            warnings.extend(document.extraction_warnings)

        return errors, warnings

    def _calculate_confidence(
        self,
        document: LegalDocument,
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """
        Calculate parsing confidence score.

        Args:
            document: Parsed document
            errors: List of errors
            warnings: List of warnings

        Returns:
            Confidence score 0.0-1.0
        """
        if errors:
            return 0.0

        # Start with document's own confidence
        confidence = document.metadata.confidence_score

        # Penalize for warnings
        if warnings:
            penalty = min(len(warnings) * 0.05, 0.3)
            confidence -= penalty

        return max(0.0, min(1.0, confidence))

    def _calculate_completeness(self, document: LegalDocument) -> float:
        """
        Calculate document completeness score.

        Args:
            document: Parsed document

        Returns:
            Completeness score 0.0-1.0
        """
        total_fields = 0
        filled_fields = 0

        # Check core fields
        core_fields = ["title", "full_text"]
        for field in core_fields:
            total_fields += 1
            if getattr(document, field, None):
                filled_fields += 1

        # Check metadata fields
        metadata_fields = ["resmi_gazete_number", "resmi_gazete_date", "effectivity_date"]
        for field in metadata_fields:
            total_fields += 1
            if getattr(document.metadata, field, None):
                filled_fields += 1

        # Check optional but valuable fields
        if document.clauses:
            filled_fields += 1
        total_fields += 1

        if document.citations:
            filled_fields += 1
        total_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

    def _get_source_url(self, source: Any) -> Optional[str]:
        """
        Extract source URL from source input.

        Args:
            source: Source input

        Returns:
            Source URL if available
        """
        if isinstance(source, str) and source.startswith("http"):
            return source
        elif isinstance(source, dict) and "url" in source:
            return source["url"]
        return None

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get parser statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "avg_duration_ms": (
                self.stats["total_duration_ms"] / self.stats["total_parsed"]
                if self.stats["total_parsed"] > 0 else 0.0
            ),
            "success_rate": (
                (self.stats["total_parsed"] - self.stats["total_failed"]) / self.stats["total_parsed"]
                if self.stats["total_parsed"] > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset parser statistics."""
        self.stats = {
            "total_parsed": 0,
            "total_failed": 0,
            "total_duration_ms": 0.0,
        }

    def supports_document_type(self, document_type: DocumentType) -> bool:
        """
        Check if parser supports a document type.
        Default: support all types. Override for specific parsers.

        Args:
            document_type: Document type to check

        Returns:
            True if supported
        """
        return True


class SourceAdapter(BaseParser):
    """
    Base class for source adapters.
    Source adapters fetch documents from external sources (web scraping, APIs).

    Examples: YargitayAdapter, ResmiGazeteAdapter, MevzuatAdapter
    """

    @abstractmethod
    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch a document from the source.

        Args:
            document_id: Document identifier in source system
            **kwargs: Additional fetch parameters

        Returns:
            Raw document data

        Raises:
            NetworkError: If fetch fails
        """
        pass

    @abstractmethod
    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents in the source.

        Args:
            query: Search query
            **kwargs: Search parameters (filters, pagination, etc.)

        Returns:
            List of matching documents

        Raises:
            NetworkError: If search fails
        """
        pass


class StructuralParser(BaseParser):
    """
    Base class for structural parsers.
    Structural parsers extract document structure (clauses, sections, hierarchy).

    Examples: LawStructParser, RegulationStructParser, DecisionStructParser
    """

    @abstractmethod
    def extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clauses (madde, fıkra, bent) from text.

        Args:
            text: Document text

        Returns:
            List of clause dictionaries
        """
        pass

    @abstractmethod
    def build_hierarchy(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build clause hierarchy.

        Args:
            clauses: Flat list of clauses

        Returns:
            Hierarchical clause structure
        """
        pass


class SemanticExtractor(BaseParser):
    """
    Base class for semantic extractors.
    Semantic extractors extract semantic information (citations, entities, dates).

    Examples: CitationExtractor, EntityNER, DateExtractor
    """

    @abstractmethod
    def extract(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract semantic information from text.

        Args:
            text: Input text
            **kwargs: Extraction parameters

        Returns:
            List of extracted items
        """
        pass


class Validator(BaseParser):
    """
    Base class for validators.
    Validators check document quality and correctness.

    Examples: CitationValidator, SchemaValidator, CompletenessChecker
    """

    @abstractmethod
    def validate_document(self, document: LegalDocument) -> tuple[List[str], List[str]]:
        """
        Validate a document.

        Args:
            document: Document to validate

        Returns:
            Tuple of (errors, warnings)
        """
        pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BaseParser",
    "SourceAdapter",
    "StructuralParser",
    "SemanticExtractor",
    "Validator",
]
