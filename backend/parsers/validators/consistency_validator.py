"""Consistency Validator - Harvey/Legora CTO-Level Production-Grade
Validates consistency across Turkish legal documents

Production Features:
- Terminology consistency validation
- Numbering consistency (articles, paragraphs)
- Formatting consistency
- Reference consistency
- Turkish legal terminology standards
- Capitalization consistency
- Date format consistency
- Production-grade error messages with suggestions
- Context-aware validation
- Multi-level consistency checks
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import time
import re
from collections import defaultdict, Counter
from dataclasses import dataclass

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class TermUsage:
    """Tracks usage of a term throughout document"""
    term: str
    count: int
    locations: List[str]
    variants: Set[str]  # Different spellings/capitalizations


class ConsistencyValidator(BaseValidator):
    """Consistency Validator for Turkish Legal Documents

    Validates consistency across documents:
    - Terminology consistency (same terms used throughout)
    - Numbering consistency (articles, paragraphs)
    - Formatting consistency
    - Reference consistency (references to same article use same format)
    - Turkish legal terminology consistency
    - Capitalization consistency
    - Date format consistency

    Features:
    - Multi-level consistency checks
    - Turkish legal conventions
    - Context-aware validation
    - Variant detection
    - Production-grade error messages
    """

    # Turkish legal terms that should be consistent
    LEGAL_TERMS = {
        'madde': ['md', 'md.', 'madde', 'Mad.', 'Mad'],
        'fıkra': ['f', 'f.', 'fıkra', 'Fıkra'],
        'bent': ['b', 'b.', 'bent', 'Bent'],
        'kanun': ['kanun', 'Kanun', 'KANUN'],
        'yönetmelik': ['yönetmelik', 'Yönetmelik', 'YÖNETMELIK'],
        'tebliğ': ['tebliğ', 'Tebliğ', 'TEBLİĞ'],
        'karar': ['karar', 'Karar', 'KARAR'],
        'cumhurbaşkanı': ['cumhurbaşkanı', 'Cumhurbaşkanı', 'CUMHURBAŞKANI'],
        'bakanlık': ['bakanlık', 'Bakanlık', 'BAKANLIK'],
        'kurul': ['kurul', 'Kurul', 'KURUL'],
        'başkanlık': ['başkanlık', 'Başkanlık', 'BAŞKANLIK'],
    }

    # Date formats commonly used in Turkish legal documents
    DATE_FORMATS = [
        r'\d{1,2}\.\d{1,2}\.\d{4}',  # 01.01.2024
        r'\d{1,2}/\d{1,2}/\d{4}',    # 01/01/2024
        r'\d{4}-\d{2}-\d{2}',        # 2024-01-01
        r'\d{1,2}\s+\w+\s+\d{4}',    # 1 Ocak 2024
    ]

    # Article reference patterns
    ARTICLE_REFERENCE_PATTERNS = [
        r'(?:madde|md\.?)\s*(\d+)',
        r'(\d+)\.\s*madde',
        r'(\d+)\s*nci\s*madde',
        r'(\d+)\s*inci\s*madde',
        r'(\d+)\s*üncü\s*madde',
        r'(\d+)\s*uncu\s*madde',
    ]

    def __init__(self):
        """Initialize Consistency Validator"""
        super().__init__(name="Consistency Validator")

        # Tracking structures
        self.term_usage: Dict[str, TermUsage] = {}
        self.date_formats_found: List[str] = []
        self.reference_formats: Dict[int, Set[str]] = defaultdict(set)
        self.capitalization_patterns: Dict[str, Set[str]] = defaultdict(set)

    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate document consistency

        Args:
            data: Document data (dict, list, or str)
            **kwargs: Options
                - strict: Fail on warnings (default: False)
                - check_terminology: Check terminology consistency (default: True)
                - check_numbering: Check numbering consistency (default: True)
                - check_formatting: Check formatting consistency (default: True)
                - check_references: Check reference consistency (default: True)
                - check_dates: Check date format consistency (default: True)
                - check_capitalization: Check capitalization consistency (default: True)

        Returns:
            ValidationResult with consistency issues
        """
        start_time = time.time()
        result = self.create_result()

        # Reset tracking structures
        self._reset_tracking()

        # Extract options
        check_terminology = kwargs.get('check_terminology', True)
        check_numbering = kwargs.get('check_numbering', True)
        check_formatting = kwargs.get('check_formatting', True)
        check_references = kwargs.get('check_references', True)
        check_dates = kwargs.get('check_dates', True)
        check_capitalization = kwargs.get('check_capitalization', True)

        logger.info("Starting consistency validation")

        # Extract text content for analysis
        text_content = self._extract_text_content(data)

        if not text_content:
            self.add_warning(
                result,
                "NO_TEXT_CONTENT",
                "No text content found to validate consistency",
                suggestion="Ensure document contains text content"
            )
            return self.finalize_result(result, start_time)

        # Collect all terms and patterns
        self._collect_terms(text_content, data)

        # Perform consistency checks
        if check_terminology:
            self._validate_terminology_consistency(result)

        if check_numbering:
            self._validate_numbering_consistency(data, result)

        if check_formatting:
            self._validate_formatting_consistency(text_content, result)

        if check_references:
            self._validate_reference_consistency(text_content, result)

        if check_dates:
            self._validate_date_consistency(text_content, result)

        if check_capitalization:
            self._validate_capitalization_consistency(result)

        # Validate Turkish legal conventions
        self._validate_turkish_legal_consistency(text_content, result)

        logger.info(f"Consistency validation complete: {result.errors_count} errors, {result.warnings_count} warnings")

        return self.finalize_result(result, start_time)

    def _reset_tracking(self) -> None:
        """Reset tracking structures"""
        self.term_usage = {}
        self.date_formats_found = []
        self.reference_formats = defaultdict(set)
        self.capitalization_patterns = defaultdict(set)

    def _extract_text_content(self, data: Any) -> str:
        """Extract text content from data

        Args:
            data: Document data

        Returns:
            Concatenated text content
        """
        text_parts = []

        if isinstance(data, str):
            return data

        elif isinstance(data, dict):
            # Extract from common fields
            for field in ['content', 'text', 'title', 'description', 'decision_text']:
                if field in data and isinstance(data[field], str):
                    text_parts.append(data[field])

            # Extract from articles
            if 'articles' in data:
                articles = data['articles']
                if isinstance(articles, list):
                    for article in articles:
                        if isinstance(article, dict):
                            if 'content' in article:
                                text_parts.append(str(article['content']))
                            if 'title' in article:
                                text_parts.append(str(article['title']))

            # Extract from nested content
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(self._extract_text_content(value))

        elif isinstance(data, list):
            for item in data:
                text_parts.append(self._extract_text_content(item))

        return ' '.join(text_parts)

    def _collect_terms(self, text: str, data: Any) -> None:
        """Collect terms and their usage patterns

        Args:
            text: Text content
            data: Original data structure
        """
        # Collect legal terms and their variants
        for canonical_term, variants in self.LEGAL_TERMS.items():
            for variant in variants:
                # Case-insensitive search
                pattern = r'\b' + re.escape(variant) + r'\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    matched_text = match.group()
                    location = f"Position {match.start()}-{match.end()}"

                    # Track capitalization patterns
                    self.capitalization_patterns[canonical_term].add(matched_text)

                    # Track term usage
                    if canonical_term not in self.term_usage:
                        self.term_usage[canonical_term] = TermUsage(
                            term=canonical_term,
                            count=0,
                            locations=[],
                            variants=set()
                        )

                    self.term_usage[canonical_term].count += 1
                    self.term_usage[canonical_term].locations.append(location)
                    self.term_usage[canonical_term].variants.add(matched_text)

        # Collect article references
        for pattern in self.ARTICLE_REFERENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                full_reference = match.group()
                article_num = int(match.group(1))
                self.reference_formats[article_num].add(full_reference)

        # Collect date formats
        for date_pattern in self.DATE_FORMATS:
            matches = re.finditer(date_pattern, text)
            for match in matches:
                self.date_formats_found.append(match.group())

    def _validate_terminology_consistency(self, result: ValidationResult) -> None:
        """Validate terminology consistency

        Args:
            result: ValidationResult to update
        """
        for term, usage in self.term_usage.items():
            # Check if term has multiple variants
            if len(usage.variants) > 1:
                passed = False
                self.update_check_stats(result, passed)

                # Find most common variant
                variants_list = list(usage.variants)
                most_common = max(variants_list, key=lambda v: usage.variants)

                self.add_warning(
                    result,
                    "INCONSISTENT_TERMINOLOGY",
                    f"Term '{term}' has {len(usage.variants)} different variants: {', '.join(sorted(usage.variants))}",
                    context=f"Used {usage.count} times across document",
                    suggestion=f"Standardize to one variant, preferably '{most_common}'"
                )
            else:
                self.update_check_stats(result, True)

    def _validate_numbering_consistency(self, data: Any, result: ValidationResult) -> None:
        """Validate numbering consistency

        Args:
            data: Document data
            result: ValidationResult to update
        """
        if not isinstance(data, dict):
            return

        # Validate article numbering
        if 'articles' in data and isinstance(data['articles'], list):
            articles = data['articles']
            self._validate_article_numbering(articles, result)

        # Validate paragraph numbering within articles
        if 'articles' in data and isinstance(data['articles'], list):
            for i, article in enumerate(data['articles']):
                if isinstance(article, dict) and 'paragraphs' in article:
                    self._validate_paragraph_numbering(
                        article['paragraphs'],
                        result,
                        f"Article {i+1}"
                    )

    def _validate_article_numbering(self, articles: List[Dict], result: ValidationResult) -> None:
        """Validate article numbering sequence

        Args:
            articles: List of articles
            result: ValidationResult to update
        """
        if not articles:
            return

        expected_num = 1
        seen_numbers = set()

        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                continue

            # Get article number
            article_num = article.get('number', article.get('article_number'))

            if article_num is None:
                passed = False
                self.update_check_stats(result, passed)

                self.add_error(
                    result,
                    "MISSING_ARTICLE_NUMBER",
                    f"Article at index {i} is missing a number",
                    location=f"articles[{i}]",
                    suggestion="Add 'number' or 'article_number' field"
                )
                continue

            # Convert to int for comparison
            try:
                article_num = int(article_num)
            except (ValueError, TypeError):
                passed = False
                self.update_check_stats(result, passed)

                self.add_error(
                    result,
                    "INVALID_ARTICLE_NUMBER",
                    f"Article number '{article_num}' is not a valid integer",
                    location=f"articles[{i}]",
                    suggestion="Use integer article numbers"
                )
                continue

            # Check for duplicates
            if article_num in seen_numbers:
                passed = False
                self.update_check_stats(result, passed)

                self.add_error(
                    result,
                    "DUPLICATE_ARTICLE_NUMBER",
                    f"Article number {article_num} appears multiple times",
                    location=f"articles[{i}]",
                    suggestion="Ensure each article has a unique number"
                )

            seen_numbers.add(article_num)

            # Check sequential numbering (with gap detection)
            if article_num != expected_num:
                # Allow for temporary and additional articles (ek madde, geçici madde)
                is_special = any(
                    keyword in str(article.get('title', '')).lower()
                    for keyword in ['geçici', 'ek', 'temporary', 'additional']
                )

                if not is_special:
                    passed = False
                    self.update_check_stats(result, passed)

                    if article_num > expected_num:
                        gap_size = article_num - expected_num
                        self.add_warning(
                            result,
                            "ARTICLE_NUMBER_GAP",
                            f"Gap in article numbering: expected {expected_num}, found {article_num} (gap of {gap_size})",
                            location=f"articles[{i}]",
                            suggestion="Verify article sequence or add missing articles"
                        )
                    else:
                        self.add_warning(
                            result,
                            "ARTICLE_NUMBER_OUT_OF_SEQUENCE",
                            f"Article number {article_num} is out of sequence (expected {expected_num})",
                            location=f"articles[{i}]",
                            suggestion="Reorder articles or fix numbering"
                        )
                else:
                    self.update_check_stats(result, True)

            expected_num = article_num + 1

    def _validate_paragraph_numbering(
        self,
        paragraphs: List[Any],
        result: ValidationResult,
        context: str
    ) -> None:
        """Validate paragraph numbering

        Args:
            paragraphs: List of paragraphs
            result: ValidationResult to update
            context: Context (e.g., "Article 5")
        """
        if not isinstance(paragraphs, list) or not paragraphs:
            return

        expected_num = 1

        for i, paragraph in enumerate(paragraphs):
            # Extract paragraph number if it's a dict
            if isinstance(paragraph, dict):
                para_num = paragraph.get('number', paragraph.get('paragraph_number'))

                if para_num is None:
                    continue  # Unnumbered paragraphs are acceptable

                try:
                    para_num = int(para_num)
                except (ValueError, TypeError):
                    passed = False
                    self.update_check_stats(result, passed)

                    self.add_warning(
                        result,
                        "INVALID_PARAGRAPH_NUMBER",
                        f"Invalid paragraph number in {context}",
                        location=f"{context}.paragraphs[{i}]",
                        suggestion="Use integer paragraph numbers"
                    )
                    continue

                if para_num != expected_num:
                    passed = False
                    self.update_check_stats(result, passed)

                    self.add_warning(
                        result,
                        "PARAGRAPH_NUMBER_SEQUENCE",
                        f"Paragraph number {para_num} in {context} is out of sequence (expected {expected_num})",
                        location=f"{context}.paragraphs[{i}]",
                        suggestion="Check paragraph numbering sequence"
                    )

                expected_num = para_num + 1

    def _validate_formatting_consistency(self, text: str, result: ValidationResult) -> None:
        """Validate formatting consistency

        Args:
            text: Text content
            result: ValidationResult to update
        """
        # Check for mixed quotation marks
        single_quotes = text.count("'")
        double_quotes = text.count('"')
        turkish_quotes_open = text.count('"')
        turkish_quotes_close = text.count('"')

        if single_quotes > 0 and double_quotes > 0:
            passed = False
            self.update_check_stats(result, passed)

            self.add_warning(
                result,
                "MIXED_QUOTATION_MARKS",
                f"Document uses both single (') and double (\") quotation marks",
                context=f"Single: {single_quotes}, Double: {double_quotes}",
                suggestion="Standardize to one quotation mark style"
            )

        # Check for mixed dash styles
        hyphen_count = text.count('-')
        en_dash_count = text.count('–')
        em_dash_count = text.count('—')

        if sum([hyphen_count > 10, en_dash_count > 10, em_dash_count > 10]) > 1:
            passed = False
            self.update_check_stats(result, passed)

            self.add_warning(
                result,
                "MIXED_DASH_STYLES",
                "Document uses multiple dash styles",
                context=f"Hyphen: {hyphen_count}, En-dash: {en_dash_count}, Em-dash: {em_dash_count}",
                suggestion="Standardize to one dash style (typically hyphen for Turkish legal documents)"
            )

        # Check for inconsistent spacing around punctuation
        space_before_comma = len(re.findall(r'\s,', text))
        space_before_period = len(re.findall(r'\s\.', text))

        if space_before_comma > 0 or space_before_period > 0:
            passed = False
            self.update_check_stats(result, passed)

            self.add_warning(
                result,
                "INCONSISTENT_PUNCTUATION_SPACING",
                "Found spaces before punctuation marks",
                context=f"Spaces before comma: {space_before_comma}, before period: {space_before_period}",
                suggestion="Remove spaces before punctuation marks"
            )

    def _validate_reference_consistency(self, text: str, result: ValidationResult) -> None:
        """Validate reference consistency

        Args:
            text: Text content
            result: ValidationResult to update
        """
        # Check if same article is referenced with different formats
        for article_num, formats in self.reference_formats.items():
            if len(formats) > 1:
                passed = False
                self.update_check_stats(result, passed)

                self.add_warning(
                    result,
                    "INCONSISTENT_ARTICLE_REFERENCE",
                    f"Article {article_num} is referenced using {len(formats)} different formats: {', '.join(sorted(formats))}",
                    context=f"Article {article_num} references",
                    suggestion=f"Standardize references to Article {article_num} to one format"
                )
            else:
                self.update_check_stats(result, True)

    def _validate_date_consistency(self, text: str, result: ValidationResult) -> None:
        """Validate date format consistency

        Args:
            text: Text content
            result: ValidationResult to update
        """
        if not self.date_formats_found:
            return  # No dates found

        # Categorize dates by format
        format_categories = defaultdict(list)

        for date_str in self.date_formats_found:
            # Determine format category
            if re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', date_str):
                format_categories['dot_format'].append(date_str)
            elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                format_categories['slash_format'].append(date_str)
            elif re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                format_categories['iso_format'].append(date_str)
            elif re.match(r'\d{1,2}\s+\w+\s+\d{4}', date_str):
                format_categories['written_format'].append(date_str)

        # Check if multiple formats are used
        formats_used = [k for k, v in format_categories.items() if v]

        if len(formats_used) > 1:
            passed = False
            self.update_check_stats(result, passed)

            format_examples = {k: v[0] for k, v in format_categories.items() if v}

            self.add_warning(
                result,
                "INCONSISTENT_DATE_FORMAT",
                f"Document uses {len(formats_used)} different date formats",
                context=f"Formats found: {', '.join(f'{k} (e.g., {v})' for k, v in format_examples.items())}",
                suggestion="Standardize to one date format (typically DD.MM.YYYY for Turkish legal documents)"
            )
        elif formats_used:
            self.update_check_stats(result, True)

    def _validate_capitalization_consistency(self, result: ValidationResult) -> None:
        """Validate capitalization consistency

        Args:
            result: ValidationResult to update
        """
        for term, patterns in self.capitalization_patterns.items():
            if len(patterns) > 1:
                # Check if variations are acceptable (e.g., start of sentence)
                # For now, flag if more than 2 variants exist
                if len(patterns) > 2:
                    passed = False
                    self.update_check_stats(result, passed)

                    self.add_warning(
                        result,
                        "INCONSISTENT_CAPITALIZATION",
                        f"Term '{term}' appears with {len(patterns)} different capitalizations: {', '.join(sorted(patterns))}",
                        suggestion=f"Standardize capitalization of '{term}'"
                    )
                else:
                    # Check if one is all caps and one is lowercase (might be acceptable)
                    patterns_list = list(patterns)
                    if not (patterns_list[0].isupper() or patterns_list[1].isupper()):
                        passed = False
                        self.update_check_stats(result, passed)

                        self.add_info(
                            result,
                            "CAPITALIZATION_VARIATION",
                            f"Term '{term}' appears with different capitalizations: {', '.join(sorted(patterns))}",
                            suggestion="Verify capitalization is intentional"
                        )

    def _validate_turkish_legal_consistency(self, text: str, result: ValidationResult) -> None:
        """Validate Turkish legal document consistency conventions

        Args:
            text: Text content
            result: ValidationResult to update
        """
        # Check for consistent use of "Madde" vs "md."
        madde_full = len(re.findall(r'\bmadde\b', text, re.IGNORECASE))
        madde_abbrev = len(re.findall(r'\bmd\.?\b', text, re.IGNORECASE))

        if madde_full > 0 and madde_abbrev > 0:
            # Both forms used - check if one is heavily preferred
            total = madde_full + madde_abbrev
            if madde_full / total > 0.3 and madde_abbrev / total > 0.3:
                passed = False
                self.update_check_stats(result, passed)

                self.add_info(
                    result,
                    "MIXED_ARTICLE_NOTATION",
                    f"Document uses both 'madde' ({madde_full} times) and 'md.' ({madde_abbrev} times)",
                    suggestion="Consider standardizing to one form ('madde' is preferred in formal legal documents)"
                )

        # Check for consistent use of "fıkra" vs "f."
        fikra_full = len(re.findall(r'\bfıkra\b', text, re.IGNORECASE))
        fikra_abbrev = len(re.findall(r'\bf\.?\b', text, re.IGNORECASE))

        if fikra_full > 0 and fikra_abbrev > 0:
            passed = False
            self.update_check_stats(result, passed)

            self.add_info(
                result,
                "MIXED_PARAGRAPH_NOTATION",
                f"Document uses both 'fıkra' ({fikra_full} times) and 'f.' ({fikra_abbrev} times)",
                suggestion="Consider standardizing to one form"
            )

        # Check for Turkish legal document structural consistency
        self._validate_turkish_ordinal_consistency(text, result)

    def _validate_turkish_ordinal_consistency(self, text: str, result: ValidationResult) -> None:
        """Validate consistency of Turkish ordinal numbers

        Args:
            text: Text content
            result: ValidationResult to update
        """
        # Turkish ordinal suffixes: ıncı, inci, uncu, üncü, nci, ncı, ncu, ncü
        # Should be used consistently

        ordinal_patterns = {
            'inci': len(re.findall(r'\d+\s*inci\b', text, re.IGNORECASE)),
            'ıncı': len(re.findall(r'\d+\s*ıncı\b', text, re.IGNORECASE)),
            'uncu': len(re.findall(r'\d+\s*uncu\b', text, re.IGNORECASE)),
            'üncu': len(re.findall(r'\d+\s*üncu\b', text, re.IGNORECASE)),
            'nci': len(re.findall(r'\d+\s*nci\b', text, re.IGNORECASE)),
            'ncı': len(re.findall(r'\d+\s*ncı\b', text, re.IGNORECASE)),
            'ncu': len(re.findall(r'\d+\s*ncu\b', text, re.IGNORECASE)),
            'ncü': len(re.findall(r'\d+\s*ncü\b', text, re.IGNORECASE)),
        }

        # Filter out unused patterns
        used_patterns = {k: v for k, v in ordinal_patterns.items() if v > 0}

        # Check for incorrect ordinal usage
        # In formal Turkish legal documents, ordinals should follow vowel harmony
        # This is a complex check, so we'll just flag if there's unusual variety

        if len(used_patterns) > 4:
            passed = False
            self.update_check_stats(result, passed)

            self.add_info(
                result,
                "VARIED_ORDINAL_USAGE",
                f"Document uses {len(used_patterns)} different ordinal suffixes",
                context=f"Patterns: {', '.join(f'{k} ({v} times)' for k, v in sorted(used_patterns.items(), key=lambda x: -x[1])[:3])}",
                suggestion="Verify ordinal suffix usage follows Turkish vowel harmony rules"
            )


__all__ = ['ConsistencyValidator']
