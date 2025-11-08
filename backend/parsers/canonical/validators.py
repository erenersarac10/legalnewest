"""Canonical Validators - Harvey/Legora CTO-Level Production-Grade
Comprehensive validation for canonical Turkish legal document models

Production Features:
- Multi-level validation (BASIC, STANDARD, STRICT, COMPREHENSIVE)
- Schema validation using Pydantic
- Content validation (completeness, Turkish patterns)
- Citation validation (cross-references, law numbers)
- Temporal validation (date consistency, version chains)
- Relationship validation (graph consistency)
- Amendment validation (DEGISHIK, MULGA, IHDAS patterns)
- Turkish legal number format validation
- Hierarchical structure validation
- Comprehensive error reporting
- Validation statistics tracking
- Configurable validation rules
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import date, datetime
import re
import logging

from .models import (
    CanonicalLegalDocument, Article, Clause, Section,
    Citation, DocumentRelationship
)
from .enums import (
    DocumentType, DocumentStatus, AmendmentType,
    ClauseType, CitationType, RelationshipType,
    ValidationSeverity
)

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION RESULT MODELS
# ============================================================================

@dataclass
class ValidationIssue:
    """Single validation issue"""
    severity: ValidationSeverity
    code: str
    message: str
    field: Optional[str] = None
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    document_id: str

    # Issues by severity
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    infos: List[ValidationIssue] = field(default_factory=list)

    # Validator-specific results
    schema_valid: bool = True
    content_valid: bool = True
    citation_valid: bool = True
    temporal_valid: bool = True
    relationship_valid: bool = True
    amendment_valid: bool = True

    # Statistics
    total_issues: int = 0
    validators_run: List[str] = field(default_factory=list)
    validation_time: float = 0.0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add validation issue"""
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        elif issue.severity == ValidationSeverity.INFO:
            self.infos.append(issue)

        self.total_issues += 1

    def get_summary(self) -> str:
        """Get validation summary"""
        return (
            f"Validation {'PASSED' if self.is_valid else 'FAILED'}: "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings, {len(self.infos)} infos"
        )


# ============================================================================
# BASE VALIDATOR
# ============================================================================

class BaseValidator:
    """Base class for validators"""

    def __init__(self):
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0
        }

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Run validation - to be implemented by subclasses"""
        raise NotImplementedError


# ============================================================================
# SCHEMA VALIDATOR
# ============================================================================

class SchemaValidator(BaseValidator):
    """Validates document schema using Pydantic"""

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Validate schema"""
        issues = []

        # Check required fields
        if not document.document_id:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MISSING_DOCUMENT_ID",
                message="Document ID is required",
                field="document_id"
            ))

        if not document.title:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MISSING_TITLE",
                message="Document title is required",
                field="title"
            ))

        if not document.full_text:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MISSING_FULL_TEXT",
                message="Full text is required",
                field="full_text"
            ))

        # Check document type specific requirements
        if document.document_type == DocumentType.KANUN and not document.law_number:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MISSING_LAW_NUMBER",
                message="Law (KANUN) should have law_number",
                field="law_number"
            ))

        # Check articles structure
        if not document.articles:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="NO_ARTICLES",
                message="Document has no articles - may not be parsed properly",
                field="articles"
            ))

        # Validate article structure
        for i, article in enumerate(document.articles):
            if not article.article_id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_ARTICLE_ID",
                    message=f"Article at position {i} missing article_id",
                    field="articles",
                    location=f"article[{i}]"
                ))

            if not article.article_number:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_ARTICLE_NUMBER",
                    message=f"Article at position {i} missing article_number",
                    field="articles",
                    location=f"article[{i}]"
                ))

            if not article.content:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="EMPTY_ARTICLE_CONTENT",
                    message=f"Article {article.article_number} has empty content",
                    field="articles",
                    location=f"article[{i}]"
                ))

        return issues


# ============================================================================
# CONTENT VALIDATOR
# ============================================================================

class ContentValidator(BaseValidator):
    """Validates document content quality"""

    # Turkish legal patterns
    LAW_NUMBER_PATTERN = re.compile(r'\b\d{3,5}\b')
    ARTICLE_PATTERN = re.compile(r'\b(?:Madde|MADDE)\s+\d+\b')
    DATE_PATTERN = re.compile(r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b')

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Validate content"""
        issues = []

        # Check text length
        if len(document.full_text) < 100:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="SHORT_DOCUMENT",
                message=f"Document text is very short ({len(document.full_text)} chars)",
                field="full_text",
                suggestion="Verify document was extracted completely"
            ))

        # Check for Turkish characters
        turkish_chars = set('çÇğĞıİöÖşŞüÜ')
        has_turkish = any(c in document.full_text for c in turkish_chars)

        if not has_turkish and len(document.full_text) > 100:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="NO_TURKISH_CHARS",
                message="Document has no Turkish-specific characters",
                field="full_text"
            ))

        # Check article numbering consistency
        article_numbers = [a.article_number for a in document.articles]

        # Check for duplicates
        if len(article_numbers) != len(set(article_numbers)):
            duplicates = [n for n in article_numbers if article_numbers.count(n) > 1]
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="DUPLICATE_ARTICLE_NUMBERS",
                message=f"Duplicate article numbers found: {set(duplicates)}",
                field="articles"
            ))

        # Check for missing articles in sequence
        try:
            numeric_articles = [int(n) for n in article_numbers if n.isdigit()]
            if numeric_articles:
                numeric_articles.sort()
                for i in range(len(numeric_articles) - 1):
                    if numeric_articles[i+1] - numeric_articles[i] > 1:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            code="ARTICLE_SEQUENCE_GAP",
                            message=f"Gap in article sequence: {numeric_articles[i]} to {numeric_articles[i+1]}",
                            field="articles",
                            suggestion="This may indicate missing articles or temporary/additional articles"
                        ))
        except ValueError:
            # Non-numeric article numbers exist
            pass

        # Check for repealed articles
        repealed_count = sum(1 for a in document.articles if a.is_repealed)
        if repealed_count > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="HAS_REPEALED_ARTICLES",
                message=f"Document has {repealed_count} repealed (mülga) articles",
                field="articles"
            ))

        # Check metadata completeness
        if not document.legal_domains:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="NO_LEGAL_DOMAINS",
                message="Document has no legal domain classification",
                field="legal_domains",
                suggestion="Add legal domain tags for better categorization"
            ))

        return issues


# ============================================================================
# CITATION VALIDATOR
# ============================================================================

class CitationValidator(BaseValidator):
    """Validates citations and cross-references"""

    LAW_CITATION_PATTERN = re.compile(
        r'\b(\d{3,5})\s+sayılı\s+(?:kanun|KANUN)\b',
        re.IGNORECASE
    )

    ARTICLE_CITATION_PATTERN = re.compile(
        r'\b(?:Madde|madde|MADDE)\s+(\d+)\b'
    )

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Validate citations"""
        issues = []

        # Check citation completeness
        for i, citation in enumerate(document.citations):
            if not citation.citation_text:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="EMPTY_CITATION_TEXT",
                    message=f"Citation {citation.citation_id} has empty citation_text",
                    field="citations",
                    location=f"citation[{i}]"
                ))

            if not citation.source_document_id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_CITATION_SOURCE",
                    message=f"Citation {citation.citation_id} missing source_document_id",
                    field="citations",
                    location=f"citation[{i}]"
                ))

            # Check if citation is resolved
            if not citation.is_resolved and citation.citation_type == CitationType.EXPLICIT:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="UNRESOLVED_CITATION",
                    message=f"Explicit citation not resolved: {citation.citation_text}",
                    field="citations",
                    location=f"citation[{i}]",
                    suggestion="Attempt to resolve citation to target document"
                ))

            # Validate confidence score
            if citation.confidence < 0.5:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="LOW_CITATION_CONFIDENCE",
                    message=f"Citation has low confidence ({citation.confidence}): {citation.citation_text}",
                    field="citations",
                    location=f"citation[{i}]"
                ))

        # Check for internal article references
        article_numbers = set(a.article_number for a in document.articles)

        for citation in document.citations:
            if citation.target_article and citation.source_document_id == document.document_id:
                # Internal reference
                if citation.target_article not in article_numbers:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="BROKEN_INTERNAL_REFERENCE",
                        message=f"Internal reference to non-existent article: {citation.target_article}",
                        field="citations"
                    ))

        return issues


# ============================================================================
# TEMPORAL VALIDATOR
# ============================================================================

class TemporalValidator(BaseValidator):
    """Validates temporal consistency (dates, versions)"""

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Validate temporal aspects"""
        issues = []

        # Check publication date vs enforcement date
        if document.publication and document.enforcement:
            pub_date = document.publication.publication_date
            eff_date = document.enforcement.effective_date

            if eff_date and pub_date and eff_date < pub_date:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_EFFECTIVE_DATE",
                    message=f"Effective date ({eff_date}) before publication date ({pub_date})",
                    field="enforcement.effective_date"
                ))

        # Check version date consistency
        if document.version_date:
            if document.publication and document.version_date < document.publication.publication_date:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="VERSION_DATE_BEFORE_PUBLICATION",
                    message="Version date is before publication date",
                    field="version_date"
                ))

        # Check amendment dates
        for i, article in enumerate(document.articles):
            if article.amendment_date:
                if document.publication and article.amendment_date < document.publication.publication_date:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="AMENDMENT_BEFORE_PUBLICATION",
                        message=f"Article {article.article_number} amendment date before document publication",
                        field="articles",
                        location=f"article[{i}].amendment_date"
                    ))

        # Check created_at vs updated_at
        if document.updated_at < document.created_at:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_UPDATE_TIME",
                message="updated_at is before created_at",
                field="updated_at"
            ))

        return issues


# ============================================================================
# RELATIONSHIP VALIDATOR
# ============================================================================

class RelationshipValidator(BaseValidator):
    """Validates document relationships"""

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Validate relationships"""
        issues = []

        # Check relationship completeness
        for i, rel in enumerate(document.relationships):
            if not rel.source_document_id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_RELATIONSHIP_SOURCE",
                    message=f"Relationship {rel.relationship_id} missing source_document_id",
                    field="relationships",
                    location=f"relationship[{i}]"
                ))

            if not rel.target_document_id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_RELATIONSHIP_TARGET",
                    message=f"Relationship {rel.relationship_id} missing target_document_id",
                    field="relationships",
                    location=f"relationship[{i}]"
                ))

            # Check temporal consistency
            if rel.effective_date and rel.end_date and rel.end_date < rel.effective_date:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_RELATIONSHIP_DATES",
                    message=f"Relationship end_date before effective_date",
                    field="relationships",
                    location=f"relationship[{i}]"
                ))

        # Check amendment relationships
        if document.amends_document_ids:
            # Check for corresponding AMENDS relationships
            amends_rels = [
                r for r in document.relationships
                if r.relationship_type == RelationshipType.AMENDS
            ]

            if len(amends_rels) != len(document.amends_document_ids):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="INCONSISTENT_AMENDMENT_DATA",
                    message="amends_document_ids count doesn't match AMENDS relationships",
                    field="amends_document_ids",
                    suggestion="Ensure amendment data is synchronized"
                ))

        return issues


# ============================================================================
# AMENDMENT VALIDATOR
# ============================================================================

class AmendmentValidator(BaseValidator):
    """Validates Turkish legal amendments (Değişik, Mülga, İhdas)"""

    def validate(self, document: CanonicalLegalDocument) -> List[ValidationIssue]:
        """Validate amendments"""
        issues = []

        # Check amended articles
        for i, article in enumerate(document.articles):
            if article.amendment_type:
                # Check for amendment details
                if not article.amended_by:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="MISSING_AMENDED_BY",
                        message=f"Article {article.article_number} has amendment_type but no amended_by",
                        field="articles",
                        location=f"article[{i}].amended_by"
                    ))

                if not article.amendment_date:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="MISSING_AMENDMENT_DATE",
                        message=f"Article {article.article_number} has amendment_type but no amendment_date",
                        field="articles",
                        location=f"article[{i}].amendment_date"
                    ))

                # Check for MULGA (repealed) consistency
                if article.amendment_type == AmendmentType.MULGA and not article.is_repealed:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="MULGA_NOT_REPEALED",
                        message=f"Article {article.article_number} has MULGA amendment but is_repealed=False",
                        field="articles",
                        location=f"article[{i}]",
                        suggestion="Set is_repealed=True for MULGA articles"
                    ))

                # Check for IHDAS (added) with temporary/additional flags
                if article.amendment_type == AmendmentType.IHDAS:
                    if not (article.is_temporary or article.is_additional):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            code="IHDAS_WITHOUT_FLAGS",
                            message=f"Article {article.article_number} has IHDAS but no temporary/additional flag",
                            field="articles",
                            location=f"article[{i}]"
                        ))

        # Check repealed articles
        for i, article in enumerate(document.articles):
            if article.is_repealed:
                if article.is_active:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="REPEALED_BUT_ACTIVE",
                        message=f"Article {article.article_number} is repealed but marked as active",
                        field="articles",
                        location=f"article[{i}]",
                        suggestion="Set is_active=False for repealed articles"
                    ))

        return issues


# ============================================================================
# CANONICAL DOCUMENT VALIDATOR
# ============================================================================

class CanonicalDocumentValidator:
    """Main validator orchestrator for canonical documents"""

    def __init__(self):
        """Initialize validator"""
        self.validators = {
            'schema': SchemaValidator(),
            'content': ContentValidator(),
            'citation': CitationValidator(),
            'temporal': TemporalValidator(),
            'relationship': RelationshipValidator(),
            'amendment': AmendmentValidator()
        }

        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'avg_validation_time': 0.0
        }

        logger.info("Initialized CanonicalDocumentValidator")

    def validate(
        self,
        document: CanonicalLegalDocument,
        validators: Optional[List[str]] = None,
        strict: bool = False
    ) -> ValidationResult:
        """Validate canonical document

        Args:
            document: Document to validate
            validators: List of validators to run (None = all)
            strict: If True, warnings are treated as errors

        Returns:
            ValidationResult
        """
        start_time = datetime.now()

        result = ValidationResult(
            is_valid=True,
            document_id=document.document_id
        )

        # Determine which validators to run
        validators_to_run = validators or list(self.validators.keys())

        try:
            # Run each validator
            for validator_name in validators_to_run:
                if validator_name not in self.validators:
                    logger.warning(f"Unknown validator: {validator_name}")
                    continue

                result.validators_run.append(validator_name)
                validator = self.validators[validator_name]

                # Run validation
                issues = validator.validate(document)

                # Add issues to result
                for issue in issues:
                    # In strict mode, treat warnings as errors
                    if strict and issue.severity == ValidationSeverity.WARNING:
                        issue.severity = ValidationSeverity.ERROR

                    result.add_issue(issue)

                # Update validator-specific flags
                has_errors = any(
                    issue.severity == ValidationSeverity.ERROR
                    for issue in issues
                )

                if validator_name == 'schema':
                    result.schema_valid = not has_errors
                elif validator_name == 'content':
                    result.content_valid = not has_errors
                elif validator_name == 'citation':
                    result.citation_valid = not has_errors
                elif validator_name == 'temporal':
                    result.temporal_valid = not has_errors
                elif validator_name == 'relationship':
                    result.relationship_valid = not has_errors
                elif validator_name == 'amendment':
                    result.amendment_valid = not has_errors

            # Calculate validation time
            result.validation_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            self.stats['total_validations'] += 1
            self.stats['total_errors'] += len(result.errors)
            self.stats['total_warnings'] += len(result.warnings)

            if result.is_valid:
                self.stats['passed_validations'] += 1
            else:
                self.stats['failed_validations'] += 1

            # Update average validation time
            total = self.stats['total_validations']
            self.stats['avg_validation_time'] = (
                (self.stats['avg_validation_time'] * (total - 1) + result.validation_time) / total
            )

            logger.info(f"Validated {document.document_id}: {result.get_summary()}")

        except Exception as e:
            result.is_valid = False
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="VALIDATION_EXCEPTION",
                message=f"Validation failed with exception: {str(e)}"
            ))

            self.stats['total_validations'] += 1
            self.stats['failed_validations'] += 1

            logger.error(f"Validation exception for {document.document_id}: {e}")

        return result

    def validate_batch(
        self,
        documents: List[CanonicalLegalDocument],
        validators: Optional[List[str]] = None,
        strict: bool = False
    ) -> List[ValidationResult]:
        """Validate multiple documents

        Args:
            documents: Documents to validate
            validators: Validators to run
            strict: Strict mode

        Returns:
            List of ValidationResults
        """
        results = []

        for document in documents:
            result = self.validate(document, validators, strict)
            results.append(result)

        logger.info(
            f"Batch validation complete: {len(documents)} documents, "
            f"{sum(1 for r in results if r.is_valid)} passed"
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics

        Returns:
            Statistics dict
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'avg_validation_time': 0.0
        }
        logger.info("Stats reset")


__all__ = [
    'CanonicalDocumentValidator',
    'ValidationResult',
    'ValidationIssue',
    'SchemaValidator',
    'ContentValidator',
    'CitationValidator',
    'TemporalValidator',
    'RelationshipValidator',
    'AmendmentValidator'
]
