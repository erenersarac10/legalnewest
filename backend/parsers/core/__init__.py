"""
Parser Core Module - Harvey/Legora CTO-Level

Core parsing infrastructure for Turkish legal documents.

This module provides:
    - Canonical schema (Pydantic models)
    - Exception hierarchy
    - Base parser classes
    - Parsing pipeline
    - Source registry

Quick Start:
    >>> from backend.parsers.core import get_registry, ParsingPipeline
    >>>
    >>> # Get source registry
    >>> registry = get_registry()
    >>> yargitay = registry.get("yargitay")
    >>>
    >>> # Create parsing pipeline
    >>> pipeline = ParsingPipeline("MyPipeline")
    >>> result = await pipeline.process(document_text)

Author: Legal AI Team
Version: 1.0.0
"""

# Schema
from .canonical_schema import (
    # Enums
    DocumentType,
    LegalHierarchy,
    JurisdictionType,
    EffectivityStatus,
    SourceType as SchemaSourceType,

    # Models
    Citation,
    LegalClause,
    Metadata,
    LegalDocument,
    Statute,
    CourtDecision,
    ParsingResult,
)

# Exceptions
from .exceptions import (
    # Base
    ParserError,

    # Configuration
    ConfigurationError,
    InvalidSourceConfigError,
    MissingAdapterError,
    SchemaValidationError,

    # Parsing
    ParsingError,
    DocumentStructureError,
    EncodingError,
    FormatDetectionError,
    ExtractionError,

    # Validation
    ValidationError,
    CitationValidationError,
    TemporalValidationError,
    CompletenessValidationError,
    ConsistencyValidationError,

    # Network
    NetworkError,
    SourceUnavailableError,
    RateLimitError,
    TimeoutError,

    # Transformation
    TransformationError,
    CanonicalConversionError,
    NormalizationError,
    SerializationError,

    # Utilities
    wrap_exception,
    is_retryable,
)

# Parser Base
from .parser_base import (
    BaseParser,
    SourceAdapter,
    StructuralParser,
    SemanticExtractor,
    Validator,
)

# Pipeline
from .pipeline import (
    ParsingPipeline,
    create_statute_pipeline,
    create_decision_pipeline,
)

# Source Registry
from .source_registry import (
    SourceStatus,
    SourceType,
    SourceMetadata,
    SourceRegistry,
    get_registry,
)


__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",

    # Schema - Enums
    "DocumentType",
    "LegalHierarchy",
    "JurisdictionType",
    "EffectivityStatus",
    "SchemaSourceType",

    # Schema - Models
    "Citation",
    "LegalClause",
    "Metadata",
    "LegalDocument",
    "Statute",
    "CourtDecision",
    "ParsingResult",

    # Exceptions - Base
    "ParserError",

    # Exceptions - Configuration
    "ConfigurationError",
    "InvalidSourceConfigError",
    "MissingAdapterError",
    "SchemaValidationError",

    # Exceptions - Parsing
    "ParsingError",
    "DocumentStructureError",
    "EncodingError",
    "FormatDetectionError",
    "ExtractionError",

    # Exceptions - Validation
    "ValidationError",
    "CitationValidationError",
    "TemporalValidationError",
    "CompletenessValidationError",
    "ConsistencyValidationError",

    # Exceptions - Network
    "NetworkError",
    "SourceUnavailableError",
    "RateLimitError",
    "TimeoutError",

    # Exceptions - Transformation
    "TransformationError",
    "CanonicalConversionError",
    "NormalizationError",
    "SerializationError",

    # Exception utilities
    "wrap_exception",
    "is_retryable",

    # Parser Base
    "BaseParser",
    "SourceAdapter",
    "StructuralParser",
    "SemanticExtractor",
    "Validator",

    # Pipeline
    "ParsingPipeline",
    "create_statute_pipeline",
    "create_decision_pipeline",

    # Source Registry
    "SourceStatus",
    "SourceType",
    "SourceMetadata",
    "SourceRegistry",
    "get_registry",
]
