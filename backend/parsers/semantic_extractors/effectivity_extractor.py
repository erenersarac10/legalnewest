"""Effectivity Extractor - Harvey/Legora CTO-Level Production-Grade
Extracts effectivity dates (yürürlük tarihi) from Turkish legal documents

Production Features:
- Publication date extraction (Yayım tarihi)
- Effective date extraction (Yürürlük tarihi)
- Expiration date extraction (Son geçerlilik tarihi)
- Conditional effectivity detection (Cumhurbaşkanı kararıyla)
- Relative date calculation (30 gün sonra, 6 ay içinde)
- Validity period extraction (6 ay süreyle)
- Multiple effectivity clause support
- Date normalization and validation
- Context-aware extraction
- Confidence scoring
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


class EffectivityType(Enum):
    """Types of effectivity dates"""
    PUBLICATION = "PUBLICATION"  # Yayım tarihi
    EFFECTIVE = "EFFECTIVE"  # Yürürlük tarihi
    EXPIRATION = "EXPIRATION"  # Son geçerlilik tarihi
    CONDITIONAL = "CONDITIONAL"  # Koşullu yürürlük
    IMMEDIATE = "IMMEDIATE"  # Derhal yürürlük
    DEFERRED = "DEFERRED"  # Ertelenmiş yürürlük
    PARTIAL = "PARTIAL"  # Kısmi yürürlük


class EffectivityCondition(Enum):
    """Conditions for effectivity"""
    UNCONDITIONAL = "UNCONDITIONAL"  # Koşulsuz
    PRESIDENTIAL_DECREE = "PRESIDENTIAL_DECREE"  # Cumhurbaşkanı kararı
    MINISTERIAL_DECREE = "MINISTERIAL_DECREE"  # Bakanlar Kurulu kararı
    REGULATION = "REGULATION"  # Yönetmelik çıkması
    OTHER = "OTHER"  # Diğer koşullar


@dataclass
class EffectivityDetail:
    """Detailed effectivity information"""
    effectivity_type: EffectivityType
    text: str
    date: Optional[str] = None  # ISO 8601 format or relative expression
    relative_days: Optional[int] = None  # Days from publication
    relative_months: Optional[int] = None  # Months from publication
    relative_years: Optional[int] = None  # Years from publication
    condition: EffectivityCondition = EffectivityCondition.UNCONDITIONAL
    condition_text: Optional[str] = None
    validity_period_days: Optional[int] = None  # Validity period in days
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


class EffectivityExtractor(RegexExtractor):
    """Effectivity Date Extractor for Turkish Legal Documents

    Extracts effectivity information with:
    - Publication dates (yayımı tarihinde)
    - Effective dates (yürürlüğe girer, mer'i olur)
    - Expiration dates (yürürlükten kalkar)
    - Conditional effectivity (Cumhurbaşkanı kararıyla)
    - Relative dates (30 gün sonra, 6 ay içinde)
    - Validity periods (6 ay süreyle geçerlidir)
    - Immediate effectivity (derhal, hemen)
    - Deferred effectivity (ertelenmiştir)
    - Partial effectivity (kısmen yürürlüğe girer)

    Features:
    - 20+ effectivity patterns
    - Relative date calculation
    - Conditional clause detection
    - Validity period extraction
    - Date normalization to ISO 8601
    - Context-aware classification
    - Turkish legal terminology
    """

    # Immediate effectivity patterns (derhal, hemen)
    IMMEDIATE_PATTERNS = [
        r"(?:yayım(?:ı|ından)?|yayın(?:ı|ından)?)\s+(?:tarihinde|günü)\s+(?:yürürlüğe girer|mer['']?i olur)",
        r'derhal\s+yürürlüğe\s+girer',
        r'hemen\s+yürürlüğe\s+girer',
        r'yayım(?:ı|ından)?\s+tarihinde\s+geçerli(?:dir)?',
    ]

    # Deferred effectivity patterns (30 gün sonra, etc.)
    DEFERRED_PATTERNS = [
        # X days/months/years after publication
        r'yayım(?:ı|ından)?\s+tarihinden?\s+itibaren\s+(\d+)\s+(gün|ay|yıl)\s+sonra\s+yürürlüğe\s+girer',
        r'yayım(?:ı|ından)?\s+tarihini?\s+izleyen\s+(\d+)\s*(?:ıncı|inci|üncü|ncı|nci|uncu)?\s+(gün|ay|yıl)(?:dan|den)?\s+(?:itibaren\s+)?yürürlüğe\s+girer',
        r'yayım(?:ı|ından)?\s+tarihinden?\s+(\d+)\s+(gün|ay|yıl)\s+sonra',
        # Next day after publication
        r'yayım(?:ı|ından)?\s+(?:tarihini?\s+)?takip\s+eden\s+(?:gün|günden\s+itibaren)',
        r'yayım(?:ı|ından)?\s+(?:tarihini?\s+)?izleyen\s+(?:gün|günden\s+itibaren)',
        # X days/months before specific date
        r'(\d+)\s+(gün|ay|yıl)\s+(?:önce|evvel)',
    ]

    # Specific date patterns (01.01.2024 tarihinde yürürlüğe girer)
    SPECIFIC_DATE_PATTERNS = [
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihinde\s+yürürlüğe\s+girer',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihinden\s+itibaren\s+(?:yürürlüğe\s+girer|geçerlidir)',
        r'(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})\s+tarihinde\s+yürürlüğe\s+girer',
        r'yürürlük\s+tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
    ]

    # Conditional effectivity patterns
    CONDITIONAL_PATTERNS = [
        r'Cumhurbaşkan(?:ı|lığı)(?:nca|nın)?\s+(?:belirle(?:yeceği|diği)|karar(?:ı|ıyla|laştır(?:acağı|dığı)))\s+tarihte?\s+yürürlüğe\s+girer',
        r'Cumhurbaşkan(?:ı|lığı)\s+Karar(?:ı|ıyla|name(?:si)?yle)\s+yürürlüğe\s+(?:girer|konulur)',
        r'Bakanlar\s+Kurulu(?:nca|nun)?\s+(?:karar(?:ı|ıyla)|belirle(?:yeceği|diği))\s+tarihte?\s+yürürlüğe\s+girer',
        r'(?:ilgili\s+)?yönetmeliğ(?:in|i)\s+(?:yayım(?:ı|lanması)|çıkarılması)(?:ndan|yla)\s+(?:sonra\s+)?yürürlüğe\s+girer',
    ]

    # Expiration/termination patterns
    EXPIRATION_PATTERNS = [
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihinde\s+yürürlükten\s+kalkar',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihine\s+kadar\s+geçerlidir',
        r'yayım(?:ı|ından)?\s+tarihinden\s+itibaren\s+(\d+)\s+(gün|ay|yıl)\s+süre(?:yle|ile)\s+geçerlidir',
        r'(\d+)\s+(gün|ay|yıl)\s+süre(?:yle|ile)\s+yürürlükte\s+kalır',
    ]

    # Partial effectivity patterns
    PARTIAL_PATTERNS = [
        r'(?:madde(?:si|nin|leri)?|fıkra(?:sı|nın|ları)?|bent(?:i|leri)?)\s+(\d+(?:/\d+)?(?:-[a-z])?)\s+yürürlüğe\s+girer',
        r'kısmen\s+yürürlüğe\s+girer',
        r'aşamalı\s+(?:olarak\s+)?yürürlüğe\s+girer',
    ]

    # Turkish months for date parsing
    TURKISH_MONTHS = {
        'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
        'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
        'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
    }

    # Time unit to days conversion
    TIME_UNIT_TO_DAYS = {
        'gün': 1,
        'ay': 30,  # Approximate
        'yıl': 365,  # Approximate
    }

    def __init__(self):
        # Build comprehensive pattern list
        patterns = self._build_patterns()

        super().__init__(
            name="Effectivity Extractor",
            patterns=patterns,
            version="2.0.0"
        )

    def _build_patterns(self) -> List[str]:
        """Build comprehensive effectivity pattern list"""
        patterns = []

        # Add all pattern categories
        patterns.extend(self.IMMEDIATE_PATTERNS)
        patterns.extend(self.DEFERRED_PATTERNS)
        patterns.extend(self.SPECIFIC_DATE_PATTERNS)
        patterns.extend(self.CONDITIONAL_PATTERNS)
        patterns.extend(self.EXPIRATION_PATTERNS)
        patterns.extend(self.PARTIAL_PATTERNS)

        # Generic effectivity patterns
        patterns.extend([
            r'yürürlük(?:te|e)\s+(?:gir(?:er|miştir)|konulmuştur)',
            r"mer['']?i(?:yet)?\s+(?:olur|bulur)",
            r'geçerli(?:dir|lik)',
        ])

        return patterns

    def extract(self, text: str, **kwargs) -> List[ExtractionResult]:
        """Extract effectivity information from text

        Args:
            text: Input text
            **kwargs: Additional options
                - effectivity_type: Specific effectivity type to extract

        Returns:
            List of effectivity extraction results
        """
        effectivity_type_filter = kwargs.get('effectivity_type', None)
        results = []

        # Extract using patterns
        base_results = self.extract_with_pattern(text, **kwargs)

        # Classify and enrich each result
        for result in base_results:
            effectivity_detail = self._classify_effectivity(result.value, text, result)

            if effectivity_detail:
                # Filter by type if specified
                if effectivity_type_filter and effectivity_detail.effectivity_type != effectivity_type_filter:
                    continue

                result.metadata['effectivity_detail'] = effectivity_detail
                result.confidence = effectivity_detail.confidence
                results.append(result)

        # Remove duplicates
        results = self._deduplicate_results(results)

        # Sort by position
        results.sort(key=lambda r: r.start_pos if r.start_pos else 0)

        self.update_stats(success=len(results) > 0)
        logger.info(f"Extracted {len(results)} effectivity clauses from text")

        return results

    def _classify_effectivity(self, text: str, full_text: str, result: ExtractionResult) -> Optional[EffectivityDetail]:
        """Classify effectivity type and extract details"""
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Check IMMEDIATE effectivity
        for pattern in self.IMMEDIATE_PATTERNS:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return EffectivityDetail(
                    effectivity_type=EffectivityType.IMMEDIATE,
                    text=text_clean,
                    date='PUBLICATION_DATE',
                    relative_days=0,
                    confidence=0.95
                )

        # Check CONDITIONAL effectivity
        condition_match = self._check_conditional(text_clean)
        if condition_match:
            condition_type, condition_text = condition_match
            return EffectivityDetail(
                effectivity_type=EffectivityType.CONDITIONAL,
                text=text_clean,
                condition=condition_type,
                condition_text=condition_text,
                confidence=0.90
            )

        # Check SPECIFIC DATE effectivity
        specific_date = self._extract_specific_date(text_clean)
        if specific_date:
            return EffectivityDetail(
                effectivity_type=EffectivityType.EFFECTIVE,
                text=text_clean,
                date=specific_date,
                confidence=0.95
            )

        # Check DEFERRED effectivity (relative dates)
        deferred_info = self._extract_deferred_info(text_clean)
        if deferred_info:
            days, months, years, date_expr = deferred_info
            return EffectivityDetail(
                effectivity_type=EffectivityType.DEFERRED,
                text=text_clean,
                date=date_expr,
                relative_days=days,
                relative_months=months,
                relative_years=years,
                confidence=0.90
            )

        # Check EXPIRATION effectivity
        expiration_info = self._extract_expiration_info(text_clean)
        if expiration_info:
            date, validity_days = expiration_info
            return EffectivityDetail(
                effectivity_type=EffectivityType.EXPIRATION,
                text=text_clean,
                date=date,
                validity_period_days=validity_days,
                confidence=0.85
            )

        # Check PARTIAL effectivity
        if self._is_partial_effectivity(text_clean):
            return EffectivityDetail(
                effectivity_type=EffectivityType.PARTIAL,
                text=text_clean,
                confidence=0.80,
                attributes={'partial': True}
            )

        # Default: EFFECTIVE (generic)
        return EffectivityDetail(
            effectivity_type=EffectivityType.EFFECTIVE,
            text=text_clean,
            confidence=0.70
        )

    def _check_conditional(self, text: str) -> Optional[Tuple[EffectivityCondition, str]]:
        """Check if effectivity is conditional"""
        text_lower = text.lower()

        if 'cumhurbaşkan' in text_lower:
            match = re.search(
                r'cumhurbaşkan(?:ı|lığı)(?:nca|nın|ı)?\s+(?:belirle(?:yeceği|diği)|karar(?:ı|ıyla|laştır(?:acağı|dığı)))',
                text, re.IGNORECASE
            )
            if match:
                return (EffectivityCondition.PRESIDENTIAL_DECREE, match.group(0))

        if 'bakanlar kurulu' in text_lower:
            match = re.search(
                r'bakanlar\s+kurulu(?:nca|nun)?\s+(?:karar(?:ı|ıyla)|belirle(?:yeceği|diği))',
                text, re.IGNORECASE
            )
            if match:
                return (EffectivityCondition.MINISTERIAL_DECREE, match.group(0))

        if 'yönetmelik' in text_lower and ('yayım' in text_lower or 'çıkarılma' in text_lower):
            match = re.search(
                r'yönetmeliğ(?:in|i)\s+(?:yayım(?:ı|lanması)|çıkarılması)',
                text, re.IGNORECASE
            )
            if match:
                return (EffectivityCondition.REGULATION, match.group(0))

        return None

    def _extract_specific_date(self, text: str) -> Optional[str]:
        """Extract specific date from effectivity clause"""
        # DD.MM.YYYY or DD/MM/YYYY format
        match = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', text)
        if match:
            day, month, year = match.groups()
            try:
                # Validate date
                datetime(int(year), int(month), int(day))
                # Return ISO format
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except ValueError:
                logger.warning(f"Invalid date: {day}/{month}/{year}")
                return None

        # Turkish month name format (15 Mayıs 2020)
        match = re.search(
            r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})',
            text, re.IGNORECASE
        )
        if match:
            day, month_name, year = match.groups()
            month = self.TURKISH_MONTHS.get(month_name.lower())
            if month:
                try:
                    # Validate date
                    datetime(int(year), month, int(day))
                    # Return ISO format
                    return f"{year}-{str(month).zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    logger.warning(f"Invalid date: {day} {month_name} {year}")
                    return None

        return None

    def _extract_deferred_info(self, text: str) -> Optional[Tuple[Optional[int], Optional[int], Optional[int], str]]:
        """Extract deferred effectivity information (relative dates)"""
        # X days/months/years after publication
        match = re.search(
            r'yayım(?:ı|ından)?\s+tarihinden?\s+itibaren\s+(\d+)\s+(gün|ay|yıl)\s+sonra',
            text, re.IGNORECASE
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()

            days = None
            months = None
            years = None

            if unit == 'gün':
                days = amount
            elif unit == 'ay':
                months = amount
            elif unit == 'yıl':
                years = amount

            date_expr = f"+{amount} {unit}"
            return (days, months, years, date_expr)

        # Next day after publication
        if re.search(r'yayım(?:ı|ından)?\s+(?:tarihini?\s+)?(?:takip|izleyen)\s+(?:gün|günden)', text, re.IGNORECASE):
            return (1, None, None, "+1 gün")

        # X days/months before specific date
        match = re.search(r'(\d+)\s+(gün|ay|yıl)\s+(?:önce|evvel)', text, re.IGNORECASE)
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()

            days = None
            months = None
            years = None

            if unit == 'gün':
                days = -amount
            elif unit == 'ay':
                months = -amount
            elif unit == 'yıl':
                years = -amount

            date_expr = f"-{amount} {unit}"
            return (days, months, years, date_expr)

        return None

    def _extract_expiration_info(self, text: str) -> Optional[Tuple[Optional[str], Optional[int]]]:
        """Extract expiration/termination information"""
        # Specific expiration date
        match = re.search(
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihinde\s+yürürlükten\s+kalkar',
            text, re.IGNORECASE
        )
        if match:
            date_str = match.group(1)
            # Parse date
            parts = re.split(r'[./]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                iso_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                return (iso_date, None)

        # Validity period (X days/months/years)
        match = re.search(
            r'(?:yayım(?:ı|ından)?\s+tarihinden\s+itibaren\s+)?(\d+)\s+(gün|ay|yıl)\s+süre(?:yle|ile)\s+(?:geçerlidir|yürürlükte)',
            text, re.IGNORECASE
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()

            # Convert to days
            if unit == 'gün':
                validity_days = amount
            elif unit == 'ay':
                validity_days = amount * 30  # Approximate
            elif unit == 'yıl':
                validity_days = amount * 365  # Approximate
            else:
                validity_days = None

            return (None, validity_days)

        # Valid until specific date
        match = re.search(
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihine\s+kadar\s+geçerlidir',
            text, re.IGNORECASE
        )
        if match:
            date_str = match.group(1)
            # Parse date
            parts = re.split(r'[./]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                iso_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                return (iso_date, None)

        return None

    def _is_partial_effectivity(self, text: str) -> bool:
        """Check if effectivity is partial"""
        keywords = ['kısmen', 'aşamalı', 'madde', 'fıkra', 'bent']
        text_lower = text.lower()

        # Must contain "yürürlük" and at least one partial keyword
        if 'yürürlük' in text_lower:
            return any(kw in text_lower for kw in keywords)

        return False

    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove duplicate effectivity clauses"""
        if not results:
            return results

        sorted_results = sorted(results, key=lambda r: r.start_pos if r.start_pos else 0)
        deduplicated = [sorted_results[0]]

        for result in sorted_results[1:]:
            last_result = deduplicated[-1]

            # Check for overlap
            if (result.start_pos and last_result.end_pos and
                result.start_pos < last_result.end_pos):
                # Overlapping clauses - keep higher confidence
                if result.confidence > last_result.confidence:
                    deduplicated[-1] = result
            else:
                # No overlap - add new clause
                deduplicated.append(result)

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} effectivity clauses")
        return deduplicated


__all__ = ['EffectivityExtractor', 'EffectivityType', 'EffectivityCondition', 'EffectivityDetail']
