"""Citation Validator - Harvey/Legora CTO-Level Production-Grade
Validates Turkish legal citations and references

Production Features:
- Turkish legal citation format validation
- Citation structure validation
- Reference existence checking
- Completeness validation (law number, article, paragraph)
- Multiple citation format support
- Citation target validation
- Known Turkish laws database
- Citation parsing and extraction
- Production-grade error messages with suggestions
- Statistics tracking
"""
from typing import Dict, List, Any, Optional, Set, Tuple, Pattern
import logging
import time
import re
from dataclasses import dataclass, field
from enum import Enum

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Types of legal citations"""
    LAW = "LAW"  # Kanun atfı
    ARTICLE = "ARTICLE"  # Madde atfı
    PARAGRAPH = "PARAGRAPH"  # Fıkra atfı
    REGULATION = "REGULATION"  # Yönetmelik atfı
    DECISION = "DECISION"  # Karar atfı
    DIRECTIVE = "DIRECTIVE"  # Yönerge atfı
    COMMUNIQUE = "COMMUNIQUE"  # Tebliğ atfı


@dataclass
class Citation:
    """Represents a parsed citation"""
    citation_type: CitationType
    raw_text: str  # Original citation text
    law_number: Optional[str] = None  # Kanun numarası
    law_name: Optional[str] = None  # Kanun adı (TCK, KVKK, etc.)
    article_number: Optional[str] = None  # Madde numarası
    paragraph_number: Optional[str] = None  # Fıkra numarası
    regulation_name: Optional[str] = None  # Yönetmelik adı
    decision_number: Optional[str] = None  # Karar numarası
    location: Optional[str] = None  # Where in document
    is_complete: bool = True  # Has all necessary components
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.citation_type.value]
        if self.law_number:
            parts.append(f"Law {self.law_number}")
        if self.article_number:
            parts.append(f"Article {self.article_number}")
        if self.paragraph_number:
            parts.append(f"Para {self.paragraph_number}")
        return f"Citation({', '.join(parts)})"


class CitationValidator(BaseValidator):
    """Citation Validator for Turkish Legal Documents

    Validates legal citations and references:
    - Turkish legal citation formats
    - Citation structure and completeness
    - Reference existence
    - Citation targets validity
    - Multiple citation formats
    - Known laws database

    Features:
    - "X sayılı Kanun" format validation
    - "Madde X" reference validation
    - "X. Fıkra" paragraph validation
    - Law number validation
    - Article/paragraph completeness
    - Turkish legal conventions
    - Citation extraction and parsing
    """

    # Known Turkish laws database
    KNOWN_LAWS = {
        '5237': {'name': 'TCK', 'full_name': 'Türk Ceza Kanunu'},
        '6698': {'name': 'KVKK', 'full_name': 'Kişisel Verilerin Korunması Kanunu'},
        '4721': {'name': 'TMK', 'full_name': 'Türk Medeni Kanunu'},
        '6102': {'name': 'TTK', 'full_name': 'Türk Ticaret Kanunu'},
        '6098': {'name': 'TBK', 'full_name': 'Türk Borçlar Kanunu'},
        '5271': {'name': 'CMK', 'full_name': 'Ceza Muhakemesi Kanunu'},
        '6100': {'name': 'HMK', 'full_name': 'Hukuk Muhakemeleri Kanunu'},
        '2709': {'name': 'Anayasa', 'full_name': 'Türkiye Cumhuriyeti Anayasası'},
        '4857': {'name': 'İş Kanunu', 'full_name': 'İş Kanunu'},
        '213': {'name': 'VUK', 'full_name': 'Vergi Usul Kanunu'},
        '4054': {'name': 'Rekabet Kanunu', 'full_name': 'Rekabetin Korunması Hakkında Kanun'},
        '5403': {'name': 'Toprak Koruma Kanunu', 'full_name': 'Toprak Koruma ve Arazi Kullanımı Kanunu'},
        '2863': {'name': 'Kültür Varlıkları Kanunu', 'full_name': 'Kültür ve Tabiat Varlıklarını Koruma Kanunu'},
        '3194': {'name': 'İmar Kanunu', 'full_name': 'İmar Kanunu'},
    }

    # Citation patterns (Turkish legal citation formats)
    CITATION_PATTERNS = {
        # "5237 sayılı Türk Ceza Kanunu" or "5237 sayılı Kanun"
        'law_full': re.compile(
            r'(\d{3,4})\s+sayılı\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:Kanunu|Kanun))',
            re.IGNORECASE
        ),
        # "5237 sayılı TCK" or "5237 sayılı KVKK"
        'law_abbreviated': re.compile(
            r'(\d{3,4})\s+sayılı\s+([A-ZÇĞİÖŞÜ]{2,})',
            re.IGNORECASE
        ),
        # "TCK" or "KVKK" (standalone abbreviations)
        'law_abbr_only': re.compile(
            r'\b(TCK|KVKK|TMK|TTK|TBK|CMK|HMK|VUK)\b'
        ),
        # "Madde 10" or "10. madde" or "10'uncu madde"
        'article': re.compile(
            r'(?:(?:madde|md\.|m\.)\s*(\d+)|(\d+)\.?\s*(?:uncu|ıncı|nci|ncı)?\s*(?:madde|md\.|m\.))',
            re.IGNORECASE
        ),
        # "1. fıkra" or "birinci fıkra" or "fıkra 1"
        'paragraph': re.compile(
            r'(?:(?:fıkra|f\.)\s*(\d+)|(\d+)\.?\s*(?:uncu|ıncı|nci|ncı)?\s*(?:fıkra|f\.))',
            re.IGNORECASE
        ),
        # "bent (a)" or "(a) bendi"
        'subparagraph': re.compile(
            r'(?:bent\s*\(([a-zğüşıöç])\)|\(([a-zğüşıöç])\)\s*bendi)',
            re.IGNORECASE
        ),
        # Combined: "TCK'nın 10. maddesi" or "5237 sayılı Kanun'un 10. maddesi"
        'law_article': re.compile(
            r'(?:(\d{3,4})\s+sayılı\s+(?:Kanun|[A-ZÇĞİÖŞÜ]{2,})|([A-ZÇĞİÖŞÜ]{2,}))(?:\'nın|\'nin|\'nun|\'nün|\'un|\'ün)?\s+(\d+)\.?\s*(?:uncu|ıncı|nci|ncı)?\s*(?:madde|md\.)',
            re.IGNORECASE
        ),
    }

    def __init__(self):
        """Initialize Citation Validator"""
        super().__init__(name="Citation Validator")

        # Statistics specific to citations
        self.citation_stats = {
            'total_citations': 0,
            'valid_citations': 0,
            'invalid_citations': 0,
            'incomplete_citations': 0,
            'unknown_law_references': 0,
            'by_type': {ct.value: 0 for ct in CitationType},
        }

    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate citations in document

        Args:
            data: Document data (dict with text/content) or text string
            **kwargs: Options
                - extract_citations: Extract and validate citations (default: True)
                - check_references: Check if referenced laws exist (default: True)
                - strict: Fail on incomplete citations (default: False)
                - known_laws: Additional known laws dict (default: None)

        Returns:
            ValidationResult with citation validation issues
        """
        start_time = time.time()
        result = self.create_result()

        # Extract text content
        text = self._extract_text(data)
        if not text:
            self.add_error(
                result,
                "NO_TEXT_CONTENT",
                "No text content found for citation validation",
                suggestion="Provide text content in 'content', 'text', or 'decision_text' field"
            )
            return self.finalize_result(result, start_time)

        # Options
        extract_citations = kwargs.get('extract_citations', True)
        check_references = kwargs.get('check_references', True)
        strict = kwargs.get('strict', False)
        additional_laws = kwargs.get('known_laws', {})

        # Merge known laws
        known_laws = {**self.KNOWN_LAWS, **additional_laws}

        logger.info(f"Validating citations in text ({len(text)} chars)")

        # Extract citations from text
        if extract_citations:
            citations = self._extract_citations(text)
            self.citation_stats['total_citations'] += len(citations)

            if len(citations) == 0:
                self.add_info(
                    result,
                    "NO_CITATIONS_FOUND",
                    "No citations found in document text",
                    suggestion="Verify document contains legal citations or check citation format"
                )
                self.update_check_stats(result, True)
            else:
                logger.info(f"Found {len(citations)} citations")

                # Validate each citation
                for citation in citations:
                    self._validate_citation(
                        citation,
                        known_laws,
                        result,
                        strict=strict,
                        check_references=check_references
                    )

        # Validate citation format consistency
        self._validate_citation_consistency(text, result)

        # Validate Turkish legal citation conventions
        self._validate_turkish_citation_conventions(text, result)

        return self.finalize_result(result, start_time)

    def _extract_text(self, data: Any) -> Optional[str]:
        """Extract text content from data"""
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # Try common text fields
            text_fields = ['content', 'text', 'decision_text', 'article_text', 'body']
            for field in text_fields:
                if field in data and data[field]:
                    return str(data[field])

            # Try to concatenate articles if present
            if 'articles' in data:
                articles = data['articles']
                if isinstance(articles, list):
                    texts = []
                    for article in articles:
                        if isinstance(article, dict) and 'content' in article:
                            texts.append(article['content'])
                    if texts:
                        return ' '.join(texts)

        return None

    def _extract_citations(self, text: str) -> List[Citation]:
        """Extract all citations from text"""
        citations = []

        # Extract law citations
        citations.extend(self._extract_law_citations(text))

        # Extract article citations
        citations.extend(self._extract_article_citations(text))

        # Extract law+article combined citations
        citations.extend(self._extract_combined_citations(text))

        # Remove duplicates based on raw text and location
        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation.raw_text, citation.location)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)

        return unique_citations

    def _extract_law_citations(self, text: str) -> List[Citation]:
        """Extract law citations (5237 sayılı TCK, etc.)"""
        citations = []

        # Full law citations: "5237 sayılı Türk Ceza Kanunu"
        for match in self.CITATION_PATTERNS['law_full'].finditer(text):
            law_number = match.group(1)
            law_name = match.group(2)
            citation = Citation(
                citation_type=CitationType.LAW,
                raw_text=match.group(0),
                law_number=law_number,
                law_name=law_name,
                location=f"char {match.start()}-{match.end()}"
            )
            citations.append(citation)

        # Abbreviated law citations: "5237 sayılı TCK"
        for match in self.CITATION_PATTERNS['law_abbreviated'].finditer(text):
            law_number = match.group(1)
            law_abbr = match.group(2)
            citation = Citation(
                citation_type=CitationType.LAW,
                raw_text=match.group(0),
                law_number=law_number,
                law_name=law_abbr,
                location=f"char {match.start()}-{match.end()}"
            )
            citations.append(citation)

        return citations

    def _extract_article_citations(self, text: str) -> List[Citation]:
        """Extract article citations (Madde 10, etc.)"""
        citations = []

        for match in self.CITATION_PATTERNS['article'].finditer(text):
            article_num = match.group(1) or match.group(2)
            citation = Citation(
                citation_type=CitationType.ARTICLE,
                raw_text=match.group(0),
                article_number=article_num,
                is_complete=False,  # Article without law reference is incomplete
                location=f"char {match.start()}-{match.end()}"
            )
            citations.append(citation)

        return citations

    def _extract_combined_citations(self, text: str) -> List[Citation]:
        """Extract combined law+article citations (TCK'nın 10. maddesi)"""
        citations = []

        for match in self.CITATION_PATTERNS['law_article'].finditer(text):
            law_number = match.group(1)
            law_abbr = match.group(2)
            article_num = match.group(3)

            citation = Citation(
                citation_type=CitationType.ARTICLE,
                raw_text=match.group(0),
                law_number=law_number,
                law_name=law_abbr,
                article_number=article_num,
                is_complete=True,
                location=f"char {match.start()}-{match.end()}"
            )
            citations.append(citation)

        return citations

    def _validate_citation(
        self,
        citation: Citation,
        known_laws: Dict[str, Dict[str, str]],
        result: ValidationResult,
        strict: bool = False,
        check_references: bool = True
    ) -> None:
        """Validate a single citation"""

        # Track citation type
        self.citation_stats['by_type'][citation.citation_type.value] += 1

        # Validate law reference
        if citation.law_number:
            self._validate_law_reference(citation, known_laws, result, check_references)

        # Validate completeness
        if not citation.is_complete:
            self.citation_stats['incomplete_citations'] += 1

            if strict:
                self.add_error(
                    result,
                    "INCOMPLETE_CITATION",
                    f"Citation '{citation.raw_text}' is incomplete (missing law reference)",
                    location=citation.location,
                    suggestion="Add law reference (e.g., '5237 sayılı TCK'nın 10. maddesi')"
                )
                self.update_check_stats(result, False)
            else:
                self.add_warning(
                    result,
                    "INCOMPLETE_CITATION",
                    f"Citation '{citation.raw_text}' lacks law reference",
                    location=citation.location,
                    suggestion="Consider adding law reference for clarity"
                )
                self.update_check_stats(result, False)
        else:
            self.update_check_stats(result, True)

        # Validate citation format
        self._validate_citation_format(citation, result)

        # Update valid/invalid counts
        if result.is_valid or (not strict and result.errors_count == 0):
            self.citation_stats['valid_citations'] += 1
        else:
            self.citation_stats['invalid_citations'] += 1

    def _validate_law_reference(
        self,
        citation: Citation,
        known_laws: Dict[str, Dict[str, str]],
        result: ValidationResult,
        check_references: bool = True
    ) -> None:
        """Validate law reference exists and is correct"""
        if not citation.law_number:
            return

        # Check if law is known
        if check_references and citation.law_number not in known_laws:
            self.citation_stats['unknown_law_references'] += 1
            self.add_warning(
                result,
                "UNKNOWN_LAW_REFERENCE",
                f"Law number '{citation.law_number}' is not in known laws database",
                location=citation.location,
                suggestion="Verify law number is correct",
                metadata={'law_number': citation.law_number}
            )
            self.update_check_stats(result, False)
        else:
            self.update_check_stats(result, True)

            # Validate law name/abbreviation if present
            if citation.law_name and citation.law_number in known_laws:
                known_law = known_laws[citation.law_number]
                expected_names = [
                    known_law.get('name', ''),
                    known_law.get('full_name', '')
                ]

                # Check if citation name matches known names
                name_match = any(
                    expected_name and expected_name.lower() in citation.law_name.lower()
                    for expected_name in expected_names
                )

                if not name_match:
                    self.add_warning(
                        result,
                        "LAW_NAME_MISMATCH",
                        f"Law name '{citation.law_name}' doesn't match known name for law {citation.law_number}",
                        location=citation.location,
                        suggestion=f"Expected '{known_law.get('name')}' or '{known_law.get('full_name')}'",
                        metadata={
                            'cited_name': citation.law_name,
                            'expected_name': known_law.get('name'),
                            'expected_full_name': known_law.get('full_name')
                        }
                    )

    def _validate_citation_format(self, citation: Citation, result: ValidationResult) -> None:
        """Validate citation format follows Turkish legal conventions"""

        # Check law number format (should be 3-4 digits)
        if citation.law_number:
            if not citation.law_number.isdigit():
                self.add_error(
                    result,
                    "INVALID_LAW_NUMBER_FORMAT",
                    f"Law number '{citation.law_number}' should be numeric",
                    location=citation.location,
                    suggestion="Use numeric law number (e.g., '5237')"
                )
            elif len(citation.law_number) < 3 or len(citation.law_number) > 4:
                self.add_warning(
                    result,
                    "UNUSUAL_LAW_NUMBER_LENGTH",
                    f"Law number '{citation.law_number}' has unusual length (expected 3-4 digits)",
                    location=citation.location,
                    suggestion="Verify law number is correct"
                )

        # Check article number format
        if citation.article_number:
            if not citation.article_number.isdigit():
                self.add_error(
                    result,
                    "INVALID_ARTICLE_NUMBER_FORMAT",
                    f"Article number '{citation.article_number}' should be numeric",
                    location=citation.location,
                    suggestion="Use numeric article number (e.g., '10')"
                )

        # Check paragraph number format
        if citation.paragraph_number:
            if not citation.paragraph_number.isdigit():
                self.add_error(
                    result,
                    "INVALID_PARAGRAPH_NUMBER_FORMAT",
                    f"Paragraph number '{citation.paragraph_number}' should be numeric",
                    location=citation.location,
                    suggestion="Use numeric paragraph number (e.g., '1')"
                )

    def _validate_citation_consistency(self, text: str, result: ValidationResult) -> None:
        """Validate citation format consistency across document"""

        # Check for consistent use of "sayılı" vs other formats
        sayili_count = len(re.findall(r'\d+\s+sayılı', text, re.IGNORECASE))

        # Check for consistent article reference format
        madde_before = len(re.findall(r'madde\s+\d+', text, re.IGNORECASE))
        madde_after = len(re.findall(r'\d+\.\s*madde', text, re.IGNORECASE))

        if madde_before > 0 and madde_after > 0:
            self.add_info(
                result,
                "INCONSISTENT_ARTICLE_FORMAT",
                f"Document uses both 'Madde X' ({madde_before}) and 'X. madde' ({madde_after}) formats",
                suggestion="Consider standardizing article reference format throughout document"
            )

    def _validate_turkish_citation_conventions(self, text: str, result: ValidationResult) -> None:
        """Validate Turkish legal citation conventions"""

        # Check for proper possessive suffix usage ('nın, 'nin, etc.)
        # Turkish law citations should use proper possessive forms
        improper_possessive = re.findall(
            r'([A-ZÇĞİÖŞÜ]{2,})\s+(\d+)\.\s*madde',  # Missing possessive
            text
        )

        if improper_possessive:
            self.add_info(
                result,
                "MISSING_POSSESSIVE_SUFFIX",
                "Some citations may be missing possessive suffixes",
                suggestion="Use possessive form (e.g., 'TCK'nın 10. maddesi' instead of 'TCK 10. madde')"
            )

        # Check for proper abbreviation capitalization
        # Turkish law abbreviations are typically all caps
        mixed_case_abbr = re.findall(
            r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü][A-ZÇĞİÖŞÜa-zçğıöşü]+)\s+sayılı',
            text
        )

        if mixed_case_abbr:
            for abbr in set(mixed_case_abbr):
                if abbr not in ['Kanun', 'Kanunu']:  # These are valid
                    self.add_info(
                        result,
                        "MIXED_CASE_ABBREVIATION",
                        f"Law abbreviation '{abbr}' uses mixed case",
                        suggestion="Turkish law abbreviations are typically all uppercase (e.g., 'TCK', 'KVKK')"
                    )

    def validate_citation_targets(
        self,
        citations: List[Citation],
        available_laws: Set[str],
        available_articles: Dict[str, Set[str]]
    ) -> ValidationResult:
        """Validate that citation targets exist

        Args:
            citations: List of citations to validate
            available_laws: Set of available law numbers
            available_articles: Dict mapping law numbers to sets of article numbers

        Returns:
            ValidationResult with target validation issues
        """
        start_time = time.time()
        result = self.create_result()

        for citation in citations:
            # Check law target
            if citation.law_number:
                if citation.law_number not in available_laws:
                    self.add_error(
                        result,
                        "CITATION_TARGET_NOT_FOUND",
                        f"Referenced law {citation.law_number} not found in available laws",
                        location=citation.location,
                        suggestion=f"Verify law {citation.law_number} is included in dataset"
                    )
                    self.update_check_stats(result, False)
                else:
                    self.update_check_stats(result, True)

                    # Check article target
                    if citation.article_number:
                        law_articles = available_articles.get(citation.law_number, set())
                        if citation.article_number not in law_articles:
                            self.add_error(
                                result,
                                "ARTICLE_TARGET_NOT_FOUND",
                                f"Article {citation.article_number} not found in law {citation.law_number}",
                                location=citation.location,
                                suggestion=f"Verify article {citation.article_number} exists in law {citation.law_number}"
                            )
                            self.update_check_stats(result, False)
                        else:
                            self.update_check_stats(result, True)

        return self.finalize_result(result, start_time)

    def get_citation_stats(self) -> Dict[str, Any]:
        """Get citation-specific statistics

        Returns:
            Dictionary of citation statistics
        """
        stats = {
            **self.citation_stats,
            'validator_stats': self.get_stats()
        }
        return stats

    def reset_citation_stats(self) -> None:
        """Reset citation statistics"""
        self.citation_stats = {
            'total_citations': 0,
            'valid_citations': 0,
            'invalid_citations': 0,
            'incomplete_citations': 0,
            'unknown_law_references': 0,
            'by_type': {ct.value: 0 for ct in CitationType},
        }
        logger.info("Citation stats reset")


__all__ = ['CitationValidator', 'Citation', 'CitationType']
