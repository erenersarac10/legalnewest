"""
Parser Exception Hierarchy - Harvey/Legora CTO-Level

Comprehensive exception handling for Turkish legal document parsing.
Provides granular error types for precise error handling and debugging.

Architecture:
    ParserError (Base)
    ├── ConfigurationError
    │   ├── InvalidSourceConfigError
    │   ├── MissingAdapterError
    │   └── SchemaValidationError
    ├── ParsingError
    │   ├── DocumentStructureError
    │   ├── EncodingError
    │   ├── FormatDetectionError
    │   └── ExtractionError
    ├── ValidationError
    │   ├── CitationValidationError
    │   ├── TemporalValidationError
    │   ├── CompletenessValidationError
    │   └── ConsistencyValidationError
    ├── NetworkError
    │   ├── SourceUnavailableError
    │   ├── RateLimitError
    │   └── TimeoutError
    └── TransformationError
        ├── CanonicalConversionError
        ├── NormalizationError
        └── SerializationError

Author: Legal AI Team
Version: 1.0.0
"""

from typing import Any, Dict, Optional, List
from datetime import datetime


class ParserError(Exception):
    """
    Base exception for all parser-related errors.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        context: Additional context about the error
        timestamp: When the error occurred
        recoverable: Whether the error is recoverable
    """

    def __init__(
        self,
        message: str,
        error_code: str = "PARSER_ERROR",
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()

        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "original_exception": (
                str(self.original_exception) if self.original_exception else None
            )
        }


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(ParserError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            **kwargs
        )


class InvalidSourceConfigError(ConfigurationError):
    """Invalid source configuration."""

    def __init__(self, source_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid configuration for source '{source_name}': {reason}",
            error_code="INVALID_SOURCE_CONFIG",
            context={"source_name": source_name, "reason": reason},
            **kwargs
        )


class MissingAdapterError(ConfigurationError):
    """Required adapter not found."""

    def __init__(self, adapter_name: str, available_adapters: List[str], **kwargs):
        super().__init__(
            message=f"Adapter '{adapter_name}' not found. Available: {', '.join(available_adapters)}",
            error_code="MISSING_ADAPTER",
            context={"adapter_name": adapter_name, "available_adapters": available_adapters},
            **kwargs
        )


class SchemaValidationError(ConfigurationError):
    """Schema validation failed."""

    def __init__(self, schema_name: str, validation_errors: List[str], **kwargs):
        super().__init__(
            message=f"Schema '{schema_name}' validation failed: {'; '.join(validation_errors)}",
            error_code="SCHEMA_VALIDATION_ERROR",
            context={"schema_name": schema_name, "validation_errors": validation_errors},
            **kwargs
        )


# ============================================================================
# PARSING ERRORS
# ============================================================================

class ParsingError(ParserError):
    """Errors during document parsing."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="PARSING_ERROR",
            recoverable=True,  # Often recoverable with different strategy
            **kwargs
        )


class DocumentStructureError(ParsingError):
    """Document structure is invalid or unexpected."""

    def __init__(
        self,
        document_id: str,
        expected_structure: str,
        actual_structure: str,
        **kwargs
    ):
        super().__init__(
            message=f"Document {document_id} has unexpected structure. Expected: {expected_structure}, Got: {actual_structure}",
            error_code="DOCUMENT_STRUCTURE_ERROR",
            context={
                "document_id": document_id,
                "expected_structure": expected_structure,
                "actual_structure": actual_structure
            },
            **kwargs
        )


class EncodingError(ParsingError):
    """Character encoding issues."""

    def __init__(
        self,
        document_id: str,
        detected_encoding: Optional[str],
        **kwargs
    ):
        super().__init__(
            message=f"Encoding error in document {document_id}. Detected: {detected_encoding}",
            error_code="ENCODING_ERROR",
            context={"document_id": document_id, "detected_encoding": detected_encoding},
            **kwargs
        )


class FormatDetectionError(ParsingError):
    """Unable to detect document format."""

    def __init__(self, document_id: str, attempted_formats: List[str], **kwargs):
        super().__init__(
            message=f"Could not detect format for {document_id}. Tried: {', '.join(attempted_formats)}",
            error_code="FORMAT_DETECTION_ERROR",
            context={"document_id": document_id, "attempted_formats": attempted_formats},
            **kwargs
        )


class ExtractionError(ParsingError):
    """Error extracting specific field/section."""

    def __init__(
        self,
        field_name: str,
        document_id: str,
        reason: str,
        **kwargs
    ):
        super().__init__(
            message=f"Failed to extract '{field_name}' from {document_id}: {reason}",
            error_code="EXTRACTION_ERROR",
            context={"field_name": field_name, "document_id": document_id, "reason": reason},
            **kwargs
        )


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(ParserError):
    """Validation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            **kwargs
        )


class CitationValidationError(ValidationError):
    """Citation validation failed."""

    def __init__(
        self,
        citation: str,
        reason: str,
        document_id: str,
        **kwargs
    ):
        super().__init__(
            message=f"Invalid citation '{citation}' in {document_id}: {reason}",
            error_code="CITATION_VALIDATION_ERROR",
            context={"citation": citation, "reason": reason, "document_id": document_id},
            **kwargs
        )


class TemporalValidationError(ValidationError):
    """Temporal/date validation failed."""

    def __init__(
        self,
        field_name: str,
        invalid_value: str,
        reason: str,
        **kwargs
    ):
        super().__init__(
            message=f"Temporal validation failed for '{field_name}' ({invalid_value}): {reason}",
            error_code="TEMPORAL_VALIDATION_ERROR",
            context={"field_name": field_name, "invalid_value": invalid_value, "reason": reason},
            **kwargs
        )


class CompletenessValidationError(ValidationError):
    """Required fields missing."""

    def __init__(self, missing_fields: List[str], document_id: str, **kwargs):
        super().__init__(
            message=f"Missing required fields in {document_id}: {', '.join(missing_fields)}",
            error_code="COMPLETENESS_VALIDATION_ERROR",
            context={"missing_fields": missing_fields, "document_id": document_id},
            **kwargs
        )


class ConsistencyValidationError(ValidationError):
    """Data inconsistency detected."""

    def __init__(
        self,
        inconsistency_type: str,
        details: str,
        document_id: str,
        **kwargs
    ):
        super().__init__(
            message=f"Consistency error in {document_id} ({inconsistency_type}): {details}",
            error_code="CONSISTENCY_VALIDATION_ERROR",
            context={
                "inconsistency_type": inconsistency_type,
                "details": details,
                "document_id": document_id
            },
            **kwargs
        )


# ============================================================================
# NETWORK ERRORS
# ============================================================================

class NetworkError(ParserError):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            recoverable=True,  # Network errors often retryable
            **kwargs
        )


class SourceUnavailableError(NetworkError):
    """External source is unavailable."""

    def __init__(self, source_url: str, status_code: Optional[int], **kwargs):
        super().__init__(
            message=f"Source unavailable: {source_url} (status: {status_code})",
            error_code="SOURCE_UNAVAILABLE",
            context={"source_url": source_url, "status_code": status_code},
            **kwargs
        )


class RateLimitError(NetworkError):
    """Rate limit exceeded."""

    def __init__(
        self,
        source_name: str,
        retry_after: Optional[int],
        **kwargs
    ):
        super().__init__(
            message=f"Rate limit exceeded for {source_name}. Retry after {retry_after}s",
            error_code="RATE_LIMIT_ERROR",
            context={"source_name": source_name, "retry_after": retry_after},
            **kwargs
        )


class TimeoutError(NetworkError):
    """Request timeout."""

    def __init__(self, source_url: str, timeout_seconds: int, **kwargs):
        super().__init__(
            message=f"Timeout accessing {source_url} after {timeout_seconds}s",
            error_code="TIMEOUT_ERROR",
            context={"source_url": source_url, "timeout_seconds": timeout_seconds},
            **kwargs
        )


# ============================================================================
# TRANSFORMATION ERRORS
# ============================================================================

class TransformationError(ParserError):
    """Data transformation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="TRANSFORMATION_ERROR",
            **kwargs
        )


class CanonicalConversionError(TransformationError):
    """Error converting to canonical format."""

    def __init__(
        self,
        source_format: str,
        target_format: str,
        reason: str,
        **kwargs
    ):
        super().__init__(
            message=f"Failed to convert from {source_format} to {target_format}: {reason}",
            error_code="CANONICAL_CONVERSION_ERROR",
            context={
                "source_format": source_format,
                "target_format": target_format,
                "reason": reason
            },
            **kwargs
        )


class NormalizationError(TransformationError):
    """Error during text normalization."""

    def __init__(self, field_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Normalization failed for '{field_name}': {reason}",
            error_code="NORMALIZATION_ERROR",
            context={"field_name": field_name, "reason": reason},
            **kwargs
        )


class SerializationError(TransformationError):
    """Error serializing data."""

    def __init__(
        self,
        target_format: str,
        reason: str,
        **kwargs
    ):
        super().__init__(
            message=f"Serialization to {target_format} failed: {reason}",
            error_code="SERIALIZATION_ERROR",
            context={"target_format": target_format, "reason": reason},
            **kwargs
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wrap_exception(original: Exception, parser_context: Dict[str, Any]) -> ParserError:
    """
    Wrap a generic exception in a ParserError with context.

    Args:
        original: The original exception
        parser_context: Context about what was being parsed

    Returns:
        ParserError with wrapped context
    """
    if isinstance(original, ParserError):
        # Already a ParserError, just add context
        original.context.update(parser_context)
        return original

    # Wrap generic exception
    return ParserError(
        message=str(original),
        error_code="WRAPPED_EXCEPTION",
        context=parser_context,
        original_exception=original
    )


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The error to check

    Returns:
        True if error is retryable
    """
    if isinstance(error, ParserError):
        return error.recoverable

    # Network errors and timeouts are generally retryable
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    return False
