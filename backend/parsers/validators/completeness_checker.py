"""Completeness Checker - Harvey/Legora CTO-Level Production-Grade
Validates completeness of Turkish legal documents

Production Features:
- Document structure completeness validation
- Article numbering sequence validation (no gaps)
- Required sections validation (preamble, articles, signature)
- Metadata completeness checks
- Nested structure completeness (paragraphs, subparagraphs)
- Turkish legal document conventions
- Gap detection in article sequences
- Missing section detection
- Production-grade error messages with suggestions
- Comprehensive logging
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import time
import re
from dataclasses import dataclass
from enum import Enum

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class DocumentSection(Enum):
    """Required sections in Turkish legal documents"""
    PREAMBLE = "preamble"  # Önsöz/Dibace
    ENACTMENT_FORMULA = "enactment_formula"  # Kanunlaşma formülü
    ARTICLES = "articles"  # Maddeler
    TEMPORARY_ARTICLES = "temporary_articles"  # Geçici maddeler (optional)
    ADDITIONAL_ARTICLES = "additional_articles"  # Ek maddeler (optional)
    SIGNATURE = "signature"  # İmza
    PUBLICATION_INFO = "publication_info"  # Yayım bilgisi
    METADATA = "metadata"  # Metadata


class DocumentType(Enum):
    """Turkish legal document types"""
    LAW = "law"  # Kanun
    REGULATION = "regulation"  # Yönetmelik
    DECISION = "decision"  # Karar
    CIRCULAR = "circular"  # Genelge
    COMMUNIQUE = "communique"  # Tebliğ


@dataclass
class ArticleSequence:
    """Represents article numbering sequence"""
    article_numbers: List[int]
    gaps: List[int]  # Missing article numbers
    duplicates: List[int]  # Duplicate article numbers
    invalid: List[str]  # Invalid article numbers (non-numeric)
    is_continuous: bool

    @property
    def has_issues(self) -> bool:
        """Check if sequence has any issues"""
        return len(self.gaps) > 0 or len(self.duplicates) > 0 or len(self.invalid) > 0


@dataclass
class SectionCompleteness:
    """Completeness status of a document section"""
    section: DocumentSection
    is_present: bool
    is_required: bool
    is_complete: bool
    issues: List[str]

    @property
    def has_issues(self) -> bool:
        """Check if section has issues"""
        return not self.is_complete or len(self.issues) > 0


class CompletenessChecker(BaseValidator):
    """Completeness Checker for Turkish Legal Documents

    Validates document completeness:
    - Required sections presence
    - Article numbering continuity
    - Metadata completeness
    - Nested structure completeness
    - Turkish legal conventions

    Features:
    - Gap detection in article sequences
    - Missing section identification
    - Nested structure validation (paragraphs, subparagraphs)
    - Document type-specific requirements
    - Production-grade error messages
    """

    # Required sections by document type
    REQUIRED_SECTIONS = {
        DocumentType.LAW: [
            DocumentSection.ARTICLES,
            DocumentSection.PUBLICATION_INFO,
            DocumentSection.METADATA
        ],
        DocumentType.REGULATION: [
            DocumentSection.ARTICLES,
            DocumentSection.PUBLICATION_INFO,
            DocumentSection.METADATA
        ],
        DocumentType.DECISION: [
            DocumentSection.METADATA
        ],
    }

    # Recommended sections by document type
    RECOMMENDED_SECTIONS = {
        DocumentType.LAW: [
            DocumentSection.PREAMBLE,
            DocumentSection.ENACTMENT_FORMULA,
            DocumentSection.SIGNATURE
        ],
        DocumentType.REGULATION: [
            DocumentSection.SIGNATURE
        ],
        DocumentType.DECISION: [
            DocumentSection.SIGNATURE
        ],
    }

    # Metadata required fields by document type
    REQUIRED_METADATA = {
        DocumentType.LAW: [
            'law_number',
            'title',
            'publication_date',
            'document_type'
        ],
        DocumentType.REGULATION: [
            'regulation_number',
            'title',
            'authority',
            'publication_date',
            'document_type'
        ],
        DocumentType.DECISION: [
            'decision_number',
            'court',
            'date',
            'document_type'
        ],
    }

    def __init__(self):
        """Initialize Completeness Checker"""
        super().__init__(name="Completeness Checker")

    def validate(self, data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate document completeness

        Args:
            data: Document data dictionary
            **kwargs: Options
                - document_type: DocumentType or string
                - strict: Fail on missing recommended sections (default: False)
                - check_nested: Check nested structure completeness (default: True)

        Returns:
            ValidationResult with completeness validation issues
        """
        start_time = time.time()
        result = self.create_result()

        # Detect document type
        doc_type = kwargs.get('document_type')
        if isinstance(doc_type, str):
            doc_type = self._parse_document_type(doc_type)

        if not doc_type:
            doc_type = self._detect_document_type(data)

        if not doc_type:
            self.add_error(
                result,
                "UNKNOWN_DOCUMENT_TYPE",
                "Cannot determine document type for completeness validation",
                suggestion="Specify document_type parameter or include type indicators in data"
            )
            return self.finalize_result(result, start_time)

        logger.info(f"Validating completeness for {doc_type.value} document")

        # Validate required sections
        self._validate_required_sections(data, doc_type, result, kwargs.get('strict', False))

        # Validate article numbering
        self._validate_article_numbering(data, result)

        # Validate metadata completeness
        self._validate_metadata_completeness(data, doc_type, result)

        # Validate nested structure completeness
        if kwargs.get('check_nested', True):
            self._validate_nested_completeness(data, result)

        # Validate Turkish legal conventions
        self._validate_turkish_completeness_conventions(data, doc_type, result)

        # Check overall document completeness
        self._validate_overall_completeness(data, doc_type, result)

        return self.finalize_result(result, start_time)

    def _detect_document_type(self, data: Dict[str, Any]) -> Optional[DocumentType]:
        """Detect document type from data

        Args:
            data: Document data

        Returns:
            DocumentType or None
        """
        # Check metadata first
        if 'metadata' in data and isinstance(data['metadata'], dict):
            doc_type_str = data['metadata'].get('document_type', '').lower()
            if 'kanun' in doc_type_str or 'law' in doc_type_str:
                return DocumentType.LAW
            elif 'yönetmelik' in doc_type_str or 'regulation' in doc_type_str:
                return DocumentType.REGULATION
            elif 'karar' in doc_type_str or 'decision' in doc_type_str:
                return DocumentType.DECISION
            elif 'genelge' in doc_type_str or 'circular' in doc_type_str:
                return DocumentType.CIRCULAR
            elif 'tebliğ' in doc_type_str or 'communique' in doc_type_str:
                return DocumentType.COMMUNIQUE

        # Check for type-specific fields
        if 'law_number' in data or 'kanun_numarası' in data:
            return DocumentType.LAW
        elif 'regulation_number' in data or 'yönetmelik_numarası' in data:
            return DocumentType.REGULATION
        elif 'decision_number' in data or 'karar_numarası' in data or 'court' in data:
            return DocumentType.DECISION

        return None

    def _parse_document_type(self, doc_type_str: str) -> Optional[DocumentType]:
        """Parse document type from string

        Args:
            doc_type_str: Document type string

        Returns:
            DocumentType or None
        """
        doc_type_str = doc_type_str.lower()

        type_mapping = {
            'law': DocumentType.LAW,
            'kanun': DocumentType.LAW,
            'regulation': DocumentType.REGULATION,
            'yönetmelik': DocumentType.REGULATION,
            'decision': DocumentType.DECISION,
            'karar': DocumentType.DECISION,
            'circular': DocumentType.CIRCULAR,
            'genelge': DocumentType.CIRCULAR,
            'communique': DocumentType.COMMUNIQUE,
            'tebliğ': DocumentType.COMMUNIQUE,
        }

        return type_mapping.get(doc_type_str)

    def _validate_required_sections(
        self,
        data: Dict[str, Any],
        doc_type: DocumentType,
        result: ValidationResult,
        strict: bool
    ) -> None:
        """Validate required sections are present

        Args:
            data: Document data
            doc_type: Document type
            result: ValidationResult
            strict: Whether to treat recommended sections as required
        """
        # Check required sections
        required_sections = self.REQUIRED_SECTIONS.get(doc_type, [])

        for section in required_sections:
            section_key = section.value
            passed = section_key in data and data[section_key] is not None

            self.update_check_stats(result, passed)

            if not passed:
                self.add_error(
                    result,
                    "MISSING_REQUIRED_SECTION",
                    f"Required section '{section_key}' is missing",
                    location=section_key,
                    suggestion=f"Add '{section_key}' section to document",
                    metadata={'section': section.name, 'document_type': doc_type.value}
                )

                # Check for Turkish variants
                turkish_variant = self._get_turkish_section_name(section_key)
                if turkish_variant and turkish_variant in data:
                    self.add_info(
                        result,
                        "TURKISH_VARIANT_FOUND",
                        f"Found Turkish variant '{turkish_variant}' for '{section_key}'",
                        suggestion="Consider standardizing section names to English"
                    )

        # Check recommended sections (warnings if strict mode)
        recommended_sections = self.RECOMMENDED_SECTIONS.get(doc_type, [])

        for section in recommended_sections:
            section_key = section.value
            is_present = section_key in data and data[section_key] is not None

            if not is_present:
                if strict:
                    self.add_error(
                        result,
                        "MISSING_RECOMMENDED_SECTION",
                        f"Recommended section '{section_key}' is missing (strict mode)",
                        location=section_key,
                        suggestion=f"Add '{section_key}' section for complete document"
                    )
                else:
                    self.add_warning(
                        result,
                        "MISSING_RECOMMENDED_SECTION",
                        f"Recommended section '{section_key}' is missing",
                        location=section_key,
                        suggestion=f"Consider adding '{section_key}' section"
                    )

    def _validate_article_numbering(
        self,
        data: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate article numbering is complete and continuous

        Args:
            data: Document data
            result: ValidationResult
        """
        # Check if articles exist
        articles = data.get('articles', [])

        if not articles:
            # No articles to validate (might be OK for some document types)
            return

        if not isinstance(articles, list):
            self.add_error(
                result,
                "INVALID_ARTICLES_TYPE",
                "Articles must be a list",
                location="articles",
                suggestion="Convert articles to list format"
            )
            return

        # Extract article numbers
        sequence = self._extract_article_sequence(articles)

        # Validate sequence
        self._validate_sequence_continuity(sequence, result, "articles")

        # Check for empty articles
        self._validate_article_content(articles, result)

        # Validate temporary articles if present
        if 'temporary_articles' in data:
            temp_articles = data['temporary_articles']
            if isinstance(temp_articles, list) and temp_articles:
                temp_sequence = self._extract_article_sequence(temp_articles)
                self._validate_sequence_continuity(temp_sequence, result, "temporary_articles")

        # Validate additional articles if present
        if 'additional_articles' in data:
            add_articles = data['additional_articles']
            if isinstance(add_articles, list) and add_articles:
                add_sequence = self._extract_article_sequence(add_articles)
                self._validate_sequence_continuity(add_sequence, result, "additional_articles")

    def _extract_article_sequence(self, articles: List[Dict[str, Any]]) -> ArticleSequence:
        """Extract and analyze article numbering sequence

        Args:
            articles: List of article dictionaries

        Returns:
            ArticleSequence with analysis
        """
        article_numbers = []
        invalid = []

        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                continue

            # Try to get article number
            number = article.get('number') or article.get('numara') or article.get('madde_no')

            if number is None:
                invalid.append(f"Article {i} (no number)")
                continue

            # Parse number
            try:
                if isinstance(number, int):
                    article_numbers.append(number)
                elif isinstance(number, str):
                    # Extract numeric part (e.g., "Madde 5" -> 5)
                    match = re.search(r'\d+', number)
                    if match:
                        article_numbers.append(int(match.group()))
                    else:
                        invalid.append(f"Article {i} ('{number}')")
                else:
                    invalid.append(f"Article {i} (type: {type(number).__name__})")
            except (ValueError, TypeError):
                invalid.append(f"Article {i} ('{number}')")

        # Find gaps and duplicates
        gaps = []
        duplicates = []

        if article_numbers:
            # Check for duplicates
            seen = set()
            for num in article_numbers:
                if num in seen:
                    if num not in duplicates:
                        duplicates.append(num)
                seen.add(num)

            # Check for gaps
            if article_numbers:
                sorted_numbers = sorted(set(article_numbers))
                for i in range(len(sorted_numbers) - 1):
                    current = sorted_numbers[i]
                    next_num = sorted_numbers[i + 1]

                    # Check if there's a gap
                    if next_num - current > 1:
                        gaps.extend(range(current + 1, next_num))

        is_continuous = len(gaps) == 0 and len(duplicates) == 0 and len(invalid) == 0

        return ArticleSequence(
            article_numbers=article_numbers,
            gaps=gaps,
            duplicates=duplicates,
            invalid=invalid,
            is_continuous=is_continuous
        )

    def _validate_sequence_continuity(
        self,
        sequence: ArticleSequence,
        result: ValidationResult,
        location: str
    ) -> None:
        """Validate article sequence continuity

        Args:
            sequence: ArticleSequence to validate
            result: ValidationResult
            location: Location of articles (e.g., "articles", "temporary_articles")
        """
        passed = sequence.is_continuous
        self.update_check_stats(result, passed)

        # Report gaps
        if sequence.gaps:
            gap_list = ', '.join(str(g) for g in sorted(sequence.gaps))
            self.add_error(
                result,
                "ARTICLE_NUMBERING_GAP",
                f"Article numbering has gaps in {location}: missing numbers {gap_list}",
                location=location,
                suggestion=f"Add missing articles or renumber sequence to be continuous",
                metadata={'gaps': sequence.gaps, 'location': location}
            )

        # Report duplicates
        if sequence.duplicates:
            dup_list = ', '.join(str(d) for d in sorted(sequence.duplicates))
            self.add_error(
                result,
                "DUPLICATE_ARTICLE_NUMBERS",
                f"Duplicate article numbers found in {location}: {dup_list}",
                location=location,
                suggestion="Remove duplicate articles or renumber them",
                metadata={'duplicates': sequence.duplicates, 'location': location}
            )

        # Report invalid numbers
        if sequence.invalid:
            invalid_list = ', '.join(sequence.invalid[:5])  # Limit to first 5
            if len(sequence.invalid) > 5:
                invalid_list += f" (and {len(sequence.invalid) - 5} more)"
            self.add_error(
                result,
                "INVALID_ARTICLE_NUMBERS",
                f"Invalid article numbers in {location}: {invalid_list}",
                location=location,
                suggestion="Ensure all articles have valid numeric numbers",
                metadata={'invalid_count': len(sequence.invalid), 'location': location}
            )

    def _validate_article_content(
        self,
        articles: List[Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate article content is not empty

        Args:
            articles: List of article dictionaries
            result: ValidationResult
        """
        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                continue

            # Check if article has content
            content = article.get('content') or article.get('metin') or article.get('icerik')

            if not content or (isinstance(content, str) and not content.strip()):
                article_num = article.get('number', i + 1)
                self.add_error(
                    result,
                    "EMPTY_ARTICLE_CONTENT",
                    f"Article {article_num} has no content",
                    location=f"articles[{i}]",
                    suggestion="Add content to article or remove empty article"
                )

    def _validate_metadata_completeness(
        self,
        data: Dict[str, Any],
        doc_type: DocumentType,
        result: ValidationResult
    ) -> None:
        """Validate metadata completeness

        Args:
            data: Document data
            doc_type: Document type
            result: ValidationResult
        """
        metadata = data.get('metadata', {})

        if not metadata:
            self.add_error(
                result,
                "MISSING_METADATA",
                "Document metadata is missing",
                location="metadata",
                suggestion="Add metadata section with required fields"
            )
            return

        if not isinstance(metadata, dict):
            self.add_error(
                result,
                "INVALID_METADATA_TYPE",
                "Metadata must be a dictionary",
                location="metadata",
                suggestion="Convert metadata to dictionary format"
            )
            return

        # Check required metadata fields
        required_fields = self.REQUIRED_METADATA.get(doc_type, [])

        for field in required_fields:
            # Check both metadata and root level
            in_metadata = field in metadata and metadata[field] is not None
            in_root = field in data and data[field] is not None

            passed = in_metadata or in_root
            self.update_check_stats(result, passed)

            if not passed:
                self.add_error(
                    result,
                    "MISSING_REQUIRED_METADATA",
                    f"Required metadata field '{field}' is missing",
                    location=f"metadata.{field}",
                    suggestion=f"Add '{field}' to metadata or root level",
                    metadata={'field': field, 'document_type': doc_type.value}
                )

        # Check for empty metadata values
        empty_fields = [k for k, v in metadata.items() if v is None or (isinstance(v, str) and not v.strip())]
        if empty_fields:
            fields_str = ', '.join(empty_fields[:3])
            if len(empty_fields) > 3:
                fields_str += f" (and {len(empty_fields) - 3} more)"
            self.add_warning(
                result,
                "EMPTY_METADATA_FIELDS",
                f"Metadata has empty fields: {fields_str}",
                location="metadata",
                suggestion="Fill in empty metadata fields or remove them"
            )

    def _validate_nested_completeness(
        self,
        data: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate nested structure completeness (paragraphs, subparagraphs)

        Args:
            data: Document data
            result: ValidationResult
        """
        articles = data.get('articles', [])

        if not isinstance(articles, list):
            return

        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                continue

            article_num = article.get('number', i + 1)
            location = f"articles[{i}]"

            # Check paragraphs
            if 'paragraphs' in article:
                paragraphs = article['paragraphs']
                if isinstance(paragraphs, list) and paragraphs:
                    self._validate_paragraph_completeness(paragraphs, article_num, location, result)

            # Check for incomplete nested structures
            if 'subparagraphs' in article:
                subparagraphs = article['subparagraphs']
                if not subparagraphs or (isinstance(subparagraphs, list) and len(subparagraphs) == 0):
                    self.add_warning(
                        result,
                        "EMPTY_NESTED_STRUCTURE",
                        f"Article {article_num} has empty subparagraphs list",
                        location=location,
                        suggestion="Remove empty subparagraphs field or add content"
                    )

    def _validate_paragraph_completeness(
        self,
        paragraphs: List[Any],
        article_num: Any,
        location: str,
        result: ValidationResult
    ) -> None:
        """Validate paragraph completeness

        Args:
            paragraphs: List of paragraphs
            article_num: Article number
            location: Location string
            result: ValidationResult
        """
        for i, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, dict):
                # Check paragraph content
                content = paragraph.get('content') or paragraph.get('metin')
                if not content or (isinstance(content, str) and not content.strip()):
                    self.add_warning(
                        result,
                        "EMPTY_PARAGRAPH",
                        f"Article {article_num}, paragraph {i + 1} is empty",
                        location=f"{location}.paragraphs[{i}]",
                        suggestion="Add content to paragraph or remove it"
                    )

                # Check subparagraphs if present
                if 'subparagraphs' in paragraph:
                    subparas = paragraph['subparagraphs']
                    if isinstance(subparas, list):
                        for j, subpara in enumerate(subparas):
                            if isinstance(subpara, dict):
                                sub_content = subpara.get('content') or subpara.get('metin')
                                if not sub_content or (isinstance(sub_content, str) and not sub_content.strip()):
                                    self.add_warning(
                                        result,
                                        "EMPTY_SUBPARAGRAPH",
                                        f"Article {article_num}, paragraph {i + 1}, subparagraph {j + 1} is empty",
                                        location=f"{location}.paragraphs[{i}].subparagraphs[{j}]",
                                        suggestion="Add content to subparagraph or remove it"
                                    )

    def _validate_turkish_completeness_conventions(
        self,
        data: Dict[str, Any],
        doc_type: DocumentType,
        result: ValidationResult
    ) -> None:
        """Validate Turkish legal document completeness conventions

        Args:
            data: Document data
            doc_type: Document type
            result: ValidationResult
        """
        # Law-specific completeness checks
        if doc_type == DocumentType.LAW:
            self._validate_law_completeness(data, result)

        # Regulation-specific completeness checks
        elif doc_type == DocumentType.REGULATION:
            self._validate_regulation_completeness(data, result)

        # Decision-specific completeness checks
        elif doc_type == DocumentType.DECISION:
            self._validate_decision_completeness(data, result)

    def _validate_law_completeness(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate law-specific completeness

        Args:
            data: Document data
            result: ValidationResult
        """
        # Laws should have publication date
        if not data.get('publication_date') and not data.get('yayim_tarihi'):
            self.add_warning(
                result,
                "MISSING_PUBLICATION_DATE",
                "Law is missing publication date",
                location="publication_date",
                suggestion="Add publication_date to document"
            )

        # Laws should have enactment formula
        if not data.get('enactment_formula') and not data.get('kanunlaşma_formülü'):
            self.add_warning(
                result,
                "MISSING_ENACTMENT_FORMULA",
                "Law is missing enactment formula (kanunlaşma formülü)",
                location="enactment_formula",
                suggestion="Add enactment formula as per Turkish legal conventions"
            )

    def _validate_regulation_completeness(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate regulation-specific completeness

        Args:
            data: Document data
            result: ValidationResult
        """
        # Regulations should have purpose section
        if not data.get('purpose') and not data.get('amaç'):
            self.add_warning(
                result,
                "MISSING_PURPOSE_SECTION",
                "Regulation is missing purpose section (Amaç)",
                location="purpose",
                suggestion="Add purpose section as per regulation conventions"
            )

        # Regulations should have scope section
        if not data.get('scope') and not data.get('kapsam'):
            self.add_warning(
                result,
                "MISSING_SCOPE_SECTION",
                "Regulation is missing scope section (Kapsam)",
                location="scope",
                suggestion="Add scope section as per regulation conventions"
            )

        # Regulations should have definitions section
        if not data.get('definitions') and not data.get('tanımlar'):
            self.add_info(
                result,
                "MISSING_DEFINITIONS_SECTION",
                "Regulation is missing definitions section (Tanımlar)",
                location="definitions",
                metadata={'suggestion': "Consider adding definitions section if needed"}
            )

    def _validate_decision_completeness(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate decision-specific completeness

        Args:
            data: Document data
            result: ValidationResult
        """
        # Decisions should have decision text
        if not data.get('decision_text') and not data.get('karar_metni'):
            self.add_error(
                result,
                "MISSING_DECISION_TEXT",
                "Decision is missing decision text",
                location="decision_text",
                suggestion="Add decision_text with the court's decision"
            )

        # Decisions should have reasoning
        if not data.get('reasoning') and not data.get('gerekçe'):
            self.add_warning(
                result,
                "MISSING_REASONING",
                "Decision is missing reasoning section (Gerekçe)",
                location="reasoning",
                suggestion="Add reasoning section explaining the decision"
            )

    def _validate_overall_completeness(
        self,
        data: Dict[str, Any],
        doc_type: DocumentType,
        result: ValidationResult
    ) -> None:
        """Validate overall document completeness

        Args:
            data: Document data
            doc_type: Document type
            result: ValidationResult
        """
        # Calculate completeness score
        total_sections = len(self.REQUIRED_SECTIONS.get(doc_type, []))
        total_sections += len(self.RECOMMENDED_SECTIONS.get(doc_type, []))

        present_sections = 0

        for section in self.REQUIRED_SECTIONS.get(doc_type, []):
            if section.value in data and data[section.value]:
                present_sections += 1

        for section in self.RECOMMENDED_SECTIONS.get(doc_type, []):
            if section.value in data and data[section.value]:
                present_sections += 1

        if total_sections > 0:
            completeness_percentage = (present_sections / total_sections) * 100

            if completeness_percentage < 50:
                self.add_error(
                    result,
                    "LOW_COMPLETENESS",
                    f"Document completeness is only {completeness_percentage:.1f}% ({present_sections}/{total_sections} sections)",
                    suggestion="Add missing required and recommended sections",
                    metadata={'completeness': completeness_percentage}
                )
            elif completeness_percentage < 80:
                self.add_warning(
                    result,
                    "MODERATE_COMPLETENESS",
                    f"Document completeness is {completeness_percentage:.1f}% ({present_sections}/{total_sections} sections)",
                    suggestion="Consider adding missing recommended sections",
                    metadata={'completeness': completeness_percentage}
                )
            else:
                self.add_info(
                    result,
                    "GOOD_COMPLETENESS",
                    f"Document completeness is {completeness_percentage:.1f}% ({present_sections}/{total_sections} sections)",
                    metadata={'completeness': completeness_percentage}
                )

    def _get_turkish_section_name(self, english_name: str) -> Optional[str]:
        """Get Turkish variant of section name

        Args:
            english_name: English section name

        Returns:
            Turkish section name or None
        """
        turkish_names = {
            'preamble': 'önsöz',
            'enactment_formula': 'kanunlaşma_formülü',
            'articles': 'maddeler',
            'temporary_articles': 'geçici_maddeler',
            'additional_articles': 'ek_maddeler',
            'signature': 'imza',
            'publication_info': 'yayim_bilgisi',
            'metadata': 'üstveri',
            'purpose': 'amaç',
            'scope': 'kapsam',
            'definitions': 'tanımlar',
        }
        return turkish_names.get(english_name)


__all__ = ['CompletenessChecker', 'DocumentSection', 'DocumentType', 'ArticleSequence', 'SectionCompleteness']
