"""Base Validator - Harvey/Legora CTO-Level Production-Grade
Abstract base class for all validators

Production Features:
- Abstract validator interface
- Validation result standardization
- Severity levels (ERROR, WARNING, INFO)
- Validation rules management
- Statistics tracking
- Batch validation support
- Error aggregation
- Performance metrics
- Extensible architecture
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "ERROR"  # Critical issues that must be fixed
    WARNING = "WARNING"  # Issues that should be reviewed
    INFO = "INFO"  # Informational messages
    SUCCESS = "SUCCESS"  # Validation passed


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: ValidationSeverity
    code: str  # Issue code (e.g., "MISSING_ARTICLE", "INVALID_DATE")
    message: str  # Human-readable message
    location: Optional[str] = None  # Where in the document
    context: Optional[str] = None  # Surrounding context
    suggestion: Optional[str] = None  # How to fix
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value}] {self.code}: {self.message}{loc}"


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    # Statistics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings_count: int = 0
    errors_count: int = 0

    # Performance
    validation_time: float = 0.0  # Seconds

    # Metadata
    validator_name: Optional[str] = None
    validated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue"""
        self.issues.append(issue)

        if issue.severity == ValidationSeverity.ERROR:
            self.errors_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error issues"""
        return self.get_issues_by_severity(ValidationSeverity.ERROR)

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues"""
        return self.get_issues_by_severity(ValidationSeverity.WARNING)

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Validation: {'PASSED' if self.is_valid else 'FAILED'}")
        lines.append(f"Checks: {self.passed_checks}/{self.total_checks} passed")
        lines.append(f"Errors: {self.errors_count}")
        lines.append(f"Warnings: {self.warnings_count}")
        lines.append(f"Time: {self.validation_time:.3f}s")

        if self.errors_count > 0:
            lines.append(f"\nErrors:")
            for error in self.get_errors():
                lines.append(f"  - {error}")

        if self.warnings_count > 0:
            lines.append(f"\nWarnings:")
            for warning in self.get_warnings():
                lines.append(f"  - {warning}")

        return '\n'.join(lines)


class BaseValidator(ABC):
    """Abstract base class for all validators

    Provides common validation infrastructure:
    - Validation result standardization
    - Statistics tracking
    - Batch validation
    - Error handling
    - Performance metrics

    Subclasses must implement:
    - validate(): Main validation logic
    """

    def __init__(self, name: str):
        """Initialize validator

        Args:
            name: Validator name
        """
        self.name = name
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_issues': 0,
            'total_errors': 0,
            'total_warnings': 0
        }
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate data

        Args:
            data: Data to validate
            **kwargs: Additional options

        Returns:
            ValidationResult with issues found
        """
        pass

    def validate_batch(self, data_list: List[Any], **kwargs) -> List[ValidationResult]:
        """Validate multiple items

        Args:
            data_list: List of items to validate
            **kwargs: Additional options

        Returns:
            List of ValidationResults
        """
        results = []

        for i, data in enumerate(data_list):
            try:
                result = self.validate(data, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation failed for item {i}: {e}")
                # Create error result
                result = ValidationResult(is_valid=False)
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="VALIDATION_EXCEPTION",
                    message=f"Validation failed with exception: {str(e)}",
                    location=f"Item {i}"
                ))
                results.append(result)

        logger.info(f"Batch validation complete: {len(results)} items")
        return results

    def create_result(self) -> ValidationResult:
        """Create new validation result"""
        return ValidationResult(
            is_valid=True,
            validator_name=self.name
        )

    def add_error(
        self,
        result: ValidationResult,
        code: str,
        message: str,
        **kwargs
    ) -> None:
        """Add error to validation result

        Args:
            result: ValidationResult to add to
            code: Error code
            message: Error message
            **kwargs: Additional fields (location, context, suggestion)
        """
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=code,
            message=message,
            location=kwargs.get('location'),
            context=kwargs.get('context'),
            suggestion=kwargs.get('suggestion'),
            metadata=kwargs.get('metadata', {})
        )
        result.add_issue(issue)
        logger.debug(f"Error added: {code} - {message}")

    def add_warning(
        self,
        result: ValidationResult,
        code: str,
        message: str,
        **kwargs
    ) -> None:
        """Add warning to validation result

        Args:
            result: ValidationResult to add to
            code: Warning code
            message: Warning message
            **kwargs: Additional fields (location, context, suggestion)
        """
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code=code,
            message=message,
            location=kwargs.get('location'),
            context=kwargs.get('context'),
            suggestion=kwargs.get('suggestion'),
            metadata=kwargs.get('metadata', {})
        )
        result.add_issue(issue)
        logger.debug(f"Warning added: {code} - {message}")

    def add_info(
        self,
        result: ValidationResult,
        code: str,
        message: str,
        **kwargs
    ) -> None:
        """Add info message to validation result

        Args:
            result: ValidationResult to add to
            code: Info code
            message: Info message
            **kwargs: Additional fields (location, context)
        """
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            code=code,
            message=message,
            location=kwargs.get('location'),
            context=kwargs.get('context'),
            metadata=kwargs.get('metadata', {})
        )
        result.add_issue(issue)
        logger.debug(f"Info added: {code} - {message}")

    def finalize_result(self, result: ValidationResult, start_time: float) -> ValidationResult:
        """Finalize validation result

        Args:
            result: ValidationResult to finalize
            start_time: Start time from time.time()

        Returns:
            Finalized ValidationResult
        """
        # Set validation time
        result.validation_time = time.time() - start_time

        # Update stats
        self.stats['total_validations'] += 1
        if result.is_valid:
            self.stats['successful_validations'] += 1
        else:
            self.stats['failed_validations'] += 1

        self.stats['total_issues'] += len(result.issues)
        self.stats['total_errors'] += result.errors_count
        self.stats['total_warnings'] += result.warnings_count

        return result

    def update_check_stats(self, result: ValidationResult, passed: bool) -> None:
        """Update check statistics

        Args:
            result: ValidationResult to update
            passed: Whether check passed
        """
        result.total_checks += 1
        if passed:
            result.passed_checks += 1
        else:
            result.failed_checks += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics

        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_issues': 0,
            'total_errors': 0,
            'total_warnings': 0
        }
        logger.info(f"Stats reset for {self.name}")


__all__ = [
    'BaseValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity'
]
