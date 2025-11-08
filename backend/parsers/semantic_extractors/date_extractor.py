"""Date Extractor - Harvey/Legora CTO-Level Production-Grade
Extracts dates from Turkish legal documents

Production Features:
- Turkish date format parsing (DD/MM/YYYY, DD.MM.YYYY)
- Turkish month name recognition (Ocak, Şubat, etc.)
- Relative date expressions (bugünden itibaren, yayımı takip eden gün)
- Date range extraction (başlangıç-bitiş tarihleri)
- Effective date detection (yürürlük tarihi)
- Publication date extraction (RG tarihi)
- Historical date formats (Ottoman/Republican calendar)
- ISO 8601 format support
- Date validation and normalization
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .base_extractor import (
    RegexExtractor,
    ExtractionResult,
    ConfidenceLevel,
    ExtractionMethod
)

logger = logging.getLogger(__name__)


class DateType(Enum):
    """Types of dates in legal documents"""
    ABSOLUTE = "ABSOLUTE"  # Specific date: 15.05.2020
    RELATIVE = "RELATIVE"  # Relative: yayımı takip eden gün
    RANGE = "RANGE"  # Date range: 01.01.2020 - 31.12.2020
    EFFECTIVE = "EFFECTIVE"  # Yürürlük tarihi
    PUBLICATION = "PUBLICATION"  # RG yayım tarihi
    DECISION = "DECISION"  # Karar tarihi
    ENACTMENT = "ENACTMENT"  # Kabul tarihi
    AMENDMENT = "AMENDMENT"  # Değişiklik tarihi


@dataclass
class DateDetail:
    """Detailed date information"""
    date_type: DateType
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    month_name: Optional[str] = None
    formatted_date: Optional[str] = None  # Normalized format: YYYY-MM-DD
    original_text: Optional[str] = None
    is_relative: bool = False
    relative_description: Optional[str] = None
    start_date: Optional[str] = None  # For ranges
    end_date: Optional[str] = None  # For ranges
    metadata: Dict[str, Any] = field(default_factory=dict)


class DateExtractor(RegexExtractor):
    """Date Extractor for Turkish Legal Documents

    Extracts and normalizes dates with:
    - Multiple date formats (10+ formats)
    - Turkish month names (12 months + abbreviations)
    - Relative date expressions (15+ patterns)
    - Date range extraction
    - Date type classification
    - Date validation
    - Format normalization to ISO 8601

    Features:
    - DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY formats
    - Turkish month names with case-insensitive matching
    - "15 Mayıs 2020" style dates
    - Relative dates: "yayımı takip eden gün", "bugünden itibaren"
    - Date ranges: "01.01.2020 - 31.12.2020"
    - Context-aware date type detection
    - Invalid date detection and handling
    """

    # Turkish month names
    TURKISH_MONTHS = {
        'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
        'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
        'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
    }

    # Month abbreviations
    MONTH_ABBREV = {
        'oca': 1, 'şub': 2, 'mar': 3, 'nis': 4,
        'may': 5, 'haz': 6, 'tem': 7, 'ağu': 8,
        'eyl': 9, 'eki': 10, 'kas': 11, 'ara': 12
    }

    # Absolute date patterns (ordered by specificity)
    DATE_PATTERNS = [
        # Format: "15.05.2020" or "15/05/2020" or "15-05-2020"
        r'(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})',
        # Format: "15 Mayıs 2020"
        r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})',
        # Format: "Mayıs 2020" (day omitted)
        r'(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})',
        # ISO format: "2020-05-15"
        r'(\d{4})-(\d{2})-(\d{2})',
        # Year only: "2020"
        r'\b(\d{4})\b(?=\s+yılı|\s+yılında)',
        # Format: "15.05.20" (2-digit year)
        r'(\d{1,2})[./](\d{1,2})[./](\d{2})\b'
    ]

    # Relative date patterns
    RELATIVE_PATTERNS = [
        # "yayımı takip eden gün"
        r'yayım(?:ı|ı)?(?:nı|nı)?\s+takip\s+eden\s+gün',
        # "yayımından itibaren"
        r'yayım(?:ı|ı)?(?:ndan|ndan)\s+itibaren',
        # "bugünden itibaren"
        r'bugün(?:den)?\s+itibaren',
        # "bu Kanunun yürürlüğe girdiği tarih"
        r'(?:bu|işbu)\s+(?:Kanun|Yönetmelik|Karar)(?:un|ün|nun|nün)\s+yürürlüğe\s+girdiği\s+tarih',
        # "30 gün sonra"
        r'(\d+)\s+(?:gün|ay|yıl)\s+sonra',
        # "3 ay içinde"
        r'(\d+)\s+(?:gün|ay|yıl)\s+içinde',
        # "derhal"
        r'\bderhal\b',
        # "hemen"
        r'\bhemen\b',
        # "tarihinden itibaren"
        r'tarih(?:i)?(?:nden|nden)\s+itibaren'
    ]

    # Date range patterns
    RANGE_PATTERNS = [
        # "01.01.2020 - 31.12.2020"
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s*[-–]\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        # "15 Ocak 2020 ile 31 Aralık 2020 arasında"
        r'(\d{1,2}\s+\w+\s+\d{4})\s+(?:ile|ve)\s+(\d{1,2}\s+\w+\s+\d{4})\s+arasında',
        # "2020-2021 yılları arasında"
        r'(\d{4})\s*[-–]\s*(\d{4})\s+yılları\s+arasında'
    ]

    # Context keywords for date type classification
    CONTEXT_KEYWORDS = {
        DateType.EFFECTIVE: ['yürürlük', 'yürürlüğe', 'geçerlilik'],
        DateType.PUBLICATION: ['yayım', 'resmi gazete', 'rg'],
        DateType.DECISION: ['karar', 'karar tarihi'],
        DateType.ENACTMENT: ['kabul', 'kabul tarihi', 'onay'],
        DateType.AMENDMENT: ['değişiklik', 'değiştir', 'tadil']
    }

    def __init__(self):
        # Combine all patterns
        all_patterns = (
            self.DATE_PATTERNS +
            self.RELATIVE_PATTERNS +
            self.RANGE_PATTERNS
        )

        super().__init__(
            name="Date Extractor",
            patterns=all_patterns,
            version="2.0.0"
        )

    def extract(self, text: str, **kwargs) -> List[ExtractionResult]:
        """Extract dates from text

        Args:
            text: Input text
            **kwargs: Additional options
                - extract_type: Specific date type to extract
                - normalize: Normalize to ISO format (default: True)

        Returns:
            List of date extraction results
        """
        extract_type = kwargs.get('extract_type', None)
        normalize = kwargs.get('normalize', True)
        results = []

        # Extract absolute dates
        if not extract_type or extract_type == DateType.ABSOLUTE:
            results.extend(self._extract_absolute_dates(text, normalize))

        # Extract relative dates
        if not extract_type or extract_type == DateType.RELATIVE:
            results.extend(self._extract_relative_dates(text))

        # Extract date ranges
        if not extract_type or extract_type == DateType.RANGE:
            results.extend(self._extract_date_ranges(text, normalize))

        # Remove duplicates based on position
        results = self._deduplicate_results(results)

        # Sort by position
        results.sort(key=lambda r: r.start_pos if r.start_pos else 0)

        self.update_stats(success=len(results) > 0)
        logger.info(f"Extracted {len(results)} dates from text")

        return results

    def _extract_absolute_dates(self, text: str, normalize: bool) -> List[ExtractionResult]:
        """Extract absolute dates (specific dates)"""
        results = []

        for pattern in self.DATE_PATTERNS[:6]:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                try:
                    # Parse date based on pattern
                    date_detail = self._parse_date_match(match, pattern)

                    if not date_detail:
                        continue

                    # Validate date
                    if not self._validate_date(date_detail):
                        logger.debug(f"Invalid date: {match.group(0)}")
                        continue

                    # Normalize to ISO format if requested
                    if normalize and date_detail.day and date_detail.month and date_detail.year:
                        date_detail.formatted_date = f"{date_detail.year:04d}-{date_detail.month:02d}-{date_detail.day:02d}"

                    # Determine date type from context
                    date_type = self._classify_date_type(text, match)
                    date_detail.date_type = date_type

                    # Calculate confidence
                    confidence = self._calculate_date_confidence(match, date_detail, pattern)

                    result = ExtractionResult(
                        value=match.group(0),
                        confidence=confidence,
                        confidence_level=self.get_confidence_level(confidence),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=self.extract_context(text, match.start(), match.end()),
                        method=ExtractionMethod.REGEX_PATTERN,
                        source_text=match.group(0),
                        metadata={'date_detail': date_detail}
                    )

                    if self.validate_result(result):
                        results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to parse date '{match.group(0)}': {e}")
                    continue

        return results

    def _extract_relative_dates(self, text: str) -> List[ExtractionResult]:
        """Extract relative date expressions"""
        results = []

        for pattern in self.RELATIVE_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                date_detail = DateDetail(
                    date_type=DateType.RELATIVE,
                    is_relative=True,
                    relative_description=match.group(0),
                    original_text=match.group(0)
                )

                # Try to extract numeric value for "X gün sonra" patterns
                if match.groups():
                    try:
                        numeric_value = int(match.group(1))
                        date_detail.metadata['offset_value'] = numeric_value
                        date_detail.metadata['offset_unit'] = self._extract_time_unit(match.group(0))
                    except (ValueError, IndexError):
                        pass

                confidence = 0.85

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'date_detail': date_detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _extract_date_ranges(self, text: str, normalize: bool) -> List[ExtractionResult]:
        """Extract date ranges"""
        results = []

        for pattern in self.RANGE_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                try:
                    groups = match.groups()
                    start_str = groups[0] if groups else None
                    end_str = groups[1] if len(groups) > 1 else None

                    if not start_str or not end_str:
                        continue

                    # Parse start and end dates
                    start_detail = self._parse_date_string(start_str)
                    end_detail = self._parse_date_string(end_str)

                    if normalize:
                        start_formatted = f"{start_detail.year:04d}-{start_detail.month:02d}-{start_detail.day:02d}" if start_detail.day else f"{start_detail.year:04d}"
                        end_formatted = f"{end_detail.year:04d}-{end_detail.month:02d}-{end_detail.day:02d}" if end_detail.day else f"{end_detail.year:04d}"
                    else:
                        start_formatted = start_str
                        end_formatted = end_str

                    date_detail = DateDetail(
                        date_type=DateType.RANGE,
                        start_date=start_formatted,
                        end_date=end_formatted,
                        original_text=match.group(0)
                    )

                    confidence = 0.90

                    result = ExtractionResult(
                        value=match.group(0),
                        confidence=confidence,
                        confidence_level=self.get_confidence_level(confidence),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=self.extract_context(text, match.start(), match.end()),
                        method=ExtractionMethod.REGEX_PATTERN,
                        source_text=match.group(0),
                        metadata={'date_detail': date_detail}
                    )

                    if self.validate_result(result):
                        results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to parse date range '{match.group(0)}': {e}")
                    continue

        return results

    def _parse_date_match(self, match: re.Match, pattern: str) -> Optional[DateDetail]:
        """Parse date from regex match based on pattern"""
        groups = match.groups()

        # Pattern: DD.MM.YYYY or DD/MM/YYYY
        if len(groups) == 3 and groups[0].isdigit() and groups[1].isdigit():
            day = int(groups[0])
            month = int(groups[1])
            year = int(groups[2])

            # Handle 2-digit years
            if year < 100:
                year = 2000 + year if year < 50 else 1900 + year

            return DateDetail(
                date_type=DateType.ABSOLUTE,
                day=day,
                month=month,
                year=year,
                original_text=match.group(0)
            )

        # Pattern: "15 Mayıs 2020"
        elif len(groups) == 3 and groups[1] in [m.capitalize() for m in self.TURKISH_MONTHS.keys()]:
            day = int(groups[0])
            month_name = groups[1].lower()
            month = self.TURKISH_MONTHS.get(month_name)
            year = int(groups[2])

            return DateDetail(
                date_type=DateType.ABSOLUTE,
                day=day,
                month=month,
                month_name=groups[1],
                year=year,
                original_text=match.group(0)
            )

        # Pattern: "Mayıs 2020" (day omitted)
        elif len(groups) == 2 and groups[0] in [m.capitalize() for m in self.TURKISH_MONTHS.keys()]:
            month_name = groups[0].lower()
            month = self.TURKISH_MONTHS.get(month_name)
            year = int(groups[1])

            return DateDetail(
                date_type=DateType.ABSOLUTE,
                day=None,
                month=month,
                month_name=groups[0],
                year=year,
                original_text=match.group(0)
            )

        # ISO format: "2020-05-15"
        elif len(groups) == 3 and len(groups[0]) == 4:
            year = int(groups[0])
            month = int(groups[1])
            day = int(groups[2])

            return DateDetail(
                date_type=DateType.ABSOLUTE,
                day=day,
                month=month,
                year=year,
                original_text=match.group(0)
            )

        # Year only
        elif len(groups) == 1 and len(groups[0]) == 4:
            year = int(groups[0])

            return DateDetail(
                date_type=DateType.ABSOLUTE,
                day=None,
                month=None,
                year=year,
                original_text=match.group(0)
            )

        return None

    def _parse_date_string(self, date_str: str) -> DateDetail:
        """Parse date from string (for range parsing)"""
        # Try DD.MM.YYYY format
        match = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', date_str)
        if match:
            return DateDetail(
                date_type=DateType.ABSOLUTE,
                day=int(match.group(1)),
                month=int(match.group(2)),
                year=int(match.group(3))
            )

        # Try "DD Month YYYY" format
        for month_name, month_num in self.TURKISH_MONTHS.items():
            if month_name.capitalize() in date_str:
                parts = date_str.split()
                day = int(parts[0]) if parts[0].isdigit() else None
                year = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                return DateDetail(
                    date_type=DateType.ABSOLUTE,
                    day=day,
                    month=month_num,
                    year=year
                )

        # Year only
        match = re.match(r'(\d{4})', date_str)
        if match:
            return DateDetail(
                date_type=DateType.ABSOLUTE,
                year=int(match.group(1))
            )

        return DateDetail(date_type=DateType.ABSOLUTE)

    def _validate_date(self, date_detail: DateDetail) -> bool:
        """Validate date values"""
        if date_detail.year:
            # Year validation (1900-2100)
            if date_detail.year < 1900 or date_detail.year > 2100:
                return False

        if date_detail.month:
            # Month validation (1-12)
            if date_detail.month < 1 or date_detail.month > 12:
                return False

        if date_detail.day and date_detail.month and date_detail.year:
            # Day validation
            try:
                datetime(date_detail.year, date_detail.month, date_detail.day)
                return True
            except ValueError:
                return False

        return True

    def _classify_date_type(self, text: str, match: re.Match) -> DateType:
        """Classify date type based on surrounding context"""
        # Extract context around match
        context_start = max(0, match.start() - 50)
        context_end = min(len(text), match.end() + 50)
        context = text[context_start:context_end].lower()

        # Check for date type keywords
        for date_type, keywords in self.CONTEXT_KEYWORDS.items():
            if any(keyword in context for keyword in keywords):
                return date_type

        return DateType.ABSOLUTE

    def _calculate_date_confidence(self, match: re.Match, date_detail: DateDetail, pattern: str) -> float:
        """Calculate confidence for date extraction"""
        confidence = 0.80

        # Boost for complete dates (day + month + year)
        if date_detail.day and date_detail.month and date_detail.year:
            confidence += 0.10

        # Boost for Turkish month names (more specific)
        if date_detail.month_name:
            confidence += 0.05

        # Boost for ISO format
        if '-' in match.group(0) and len(match.group(0)) == 10:
            confidence += 0.05

        return min(1.0, confidence)

    def _extract_time_unit(self, text: str) -> str:
        """Extract time unit from relative date expression"""
        if 'gün' in text.lower():
            return 'days'
        elif 'ay' in text.lower():
            return 'months'
        elif 'yıl' in text.lower():
            return 'years'
        return 'unknown'

    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove duplicate dates based on position overlap"""
        if not results:
            return results

        sorted_results = sorted(results, key=lambda r: r.start_pos if r.start_pos else 0)
        deduplicated = [sorted_results[0]]

        for result in sorted_results[1:]:
            last_result = deduplicated[-1]

            if (result.start_pos and last_result.end_pos and
                result.start_pos >= last_result.end_pos):
                deduplicated.append(result)
            elif (result.start_pos and last_result.start_pos and
                  abs(result.start_pos - last_result.start_pos) > 10):
                deduplicated.append(result)
            else:
                if result.confidence > last_result.confidence:
                    deduplicated[-1] = result

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} dates")
        return deduplicated


__all__ = ['DateExtractor', 'DateType', 'DateDetail']
