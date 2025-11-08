"""Temporal Validator - Harvey/Legora CTO-Level Production-Grade
Validates temporal aspects of Turkish legal documents

Production Features:
- Date format validation (DD.MM.YYYY, Turkish formats)
- Turkish month name parsing
- Date sequence validation (publication < effectivity)
- Temporal relationship validation
- Date consistency checks
- Effectivity date validation
- Temporal anomaly detection (future dates, very old dates)
- Turkish date terminology parsing
- Relative date expressions ("30 gün sonra", "yayımı tarihinde")
- Historical date range validation
- Production-grade error messages
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class TemporalValidator(BaseValidator):
    """Temporal Validator for Turkish Legal Documents

    Validates all temporal aspects of legal documents:
    - Date format validation
    - Turkish date parsing
    - Date sequence logic
    - Temporal relationships
    - Effectivity rules
    - Anomaly detection

    Features:
    - Multiple date formats (DD.MM.YYYY, Turkish month names)
    - Turkish legal terminology ("yayımı tarihinde yürürlüğe girer")
    - Relative date expressions ("30 gün sonra")
    - Historical date validation (1920-2050)
    - Date consistency across document
    - Publication/effectivity relationship rules
    """

    # Turkish month names mapping
    TURKISH_MONTHS = {
        'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
        'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
        'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
    }

    # Date format patterns
    DATE_PATTERNS = {
        'DD.MM.YYYY': r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$',
        'DD/MM/YYYY': r'^(\d{1,2})/(\d{1,2})/(\d{4})$',
        'DD-MM-YYYY': r'^(\d{1,2})-(\d{1,2})-(\d{4})$',
        'YYYY-MM-DD': r'^(\d{4})-(\d{1,2})-(\d{1,2})$',
        'DD Month YYYY': r'^(\d{1,2})\s+(\w+)\s+(\d{4})$',
    }

    # Turkish effectivity phrases
    EFFECTIVITY_PHRASES = {
        'yayımı tarihinde yürürlüğe girer': 0,  # Same day as publication
        'yayımı tarihinde': 0,
        'yayımından itibaren': 0,
        'yayımından bir ay sonra': 30,
        'yayımından 30 gün sonra': 30,
        'yayımından 60 gün sonra': 60,
        'yayımından 90 gün sonra': 90,
        'yayımından altı ay sonra': 180,
        'yayımından bir yıl sonra': 365,
    }

    # Valid date range for Turkish legal system
    MIN_YEAR = 1920  # Turkish Republic founding
    MAX_YEAR = 2050  # Reasonable future limit

    def __init__(self):
        """Initialize Temporal Validator"""
        super().__init__(name="Temporal Validator")

        # Current date for reference
        self.current_date = datetime.now()

        # Statistics
        self.date_stats = {
            'total_dates_validated': 0,
            'invalid_formats': 0,
            'temporal_violations': 0,
            'anomalies_detected': 0,
        }

    def validate(self, data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate temporal aspects of document

        Args:
            data: Document data dictionary
            **kwargs: Options
                - strict: Fail on warnings (default: False)
                - allow_future_dates: Allow future dates (default: False)
                - max_future_years: Maximum years in future (default: 5)

        Returns:
            ValidationResult with temporal validation issues
        """
        start_time = time.time()
        result = self.create_result()

        strict = kwargs.get('strict', False)
        allow_future = kwargs.get('allow_future_dates', False)
        max_future_years = kwargs.get('max_future_years', 5)

        logger.info(f"Validating temporal aspects (strict={strict})")

        # Extract dates from document
        dates = self._extract_dates(data, result)

        # Validate date formats
        self._validate_date_formats(dates, result)

        # Validate date ranges (historical validity)
        self._validate_date_ranges(dates, result, allow_future, max_future_years)

        # Validate date sequences (publication before effectivity)
        self._validate_date_sequences(dates, result)

        # Validate temporal relationships
        self._validate_temporal_relationships(data, dates, result)

        # Validate effectivity rules
        self._validate_effectivity_rules(data, dates, result)

        # Detect temporal anomalies
        self._detect_temporal_anomalies(dates, result)

        # Validate date consistency
        self._validate_date_consistency(data, dates, result)

        # Parse Turkish date expressions
        self._validate_turkish_date_expressions(data, result)

        return self.finalize_result(result, start_time)

    def _extract_dates(
        self,
        data: Dict[str, Any],
        result: ValidationResult
    ) -> Dict[str, Optional[datetime]]:
        """Extract and parse all dates from document

        Returns:
            Dictionary of date field names to parsed datetime objects
        """
        dates = {}

        # Common date fields
        date_fields = [
            'publication_date', 'yayım_tarihi',
            'effectivity_date', 'yürürlük_tarihi',
            'signature_date', 'imza_tarihi',
            'date', 'tarih',
            'decision_date', 'karar_tarihi',
            'enactment_date', 'kabul_tarihi',
        ]

        for field in date_fields:
            if field in data:
                date_str = data[field]
                if date_str:
                    parsed_date = self._parse_date(date_str, field, result)
                    dates[field] = parsed_date
                    if parsed_date:
                        self.date_stats['total_dates_validated'] += 1

        # Check metadata for dates
        if 'metadata' in data and isinstance(data['metadata'], dict):
            metadata = data['metadata']
            for field in date_fields:
                if field in metadata:
                    date_str = metadata[field]
                    if date_str:
                        parsed_date = self._parse_date(date_str, f"metadata.{field}", result)
                        dates[f"metadata.{field}"] = parsed_date
                        if parsed_date:
                            self.date_stats['total_dates_validated'] += 1

        return dates

    def _parse_date(
        self,
        date_str: str,
        field_name: str,
        result: ValidationResult
    ) -> Optional[datetime]:
        """Parse date string to datetime object

        Args:
            date_str: Date string
            field_name: Field name for error reporting
            result: ValidationResult to add issues to

        Returns:
            Parsed datetime or None if parsing failed
        """
        if not isinstance(date_str, str):
            return None

        date_str = date_str.strip()

        # Try DD.MM.YYYY format (most common in Turkish documents)
        match = re.match(self.DATE_PATTERNS['DD.MM.YYYY'], date_str)
        if match:
            day, month, year = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError as e:
                self.add_error(
                    result,
                    "INVALID_DATE_VALUES",
                    f"Invalid date values in '{field_name}': {date_str}",
                    location=field_name,
                    context=str(e),
                    suggestion="Check day/month values are valid"
                )
                return None

        # Try DD/MM/YYYY format
        match = re.match(self.DATE_PATTERNS['DD/MM/YYYY'], date_str)
        if match:
            day, month, year = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError as e:
                self.add_error(
                    result,
                    "INVALID_DATE_VALUES",
                    f"Invalid date values in '{field_name}': {date_str}",
                    location=field_name,
                    suggestion="Check day/month values are valid"
                )
                return None

        # Try YYYY-MM-DD format (ISO)
        match = re.match(self.DATE_PATTERNS['YYYY-MM-DD'], date_str)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError:
                return None

        # Try Turkish month names (e.g., "15 Ocak 2023")
        match = re.match(self.DATE_PATTERNS['DD Month YYYY'], date_str)
        if match:
            day_str, month_name, year_str = match.groups()
            month_name_lower = month_name.lower()

            if month_name_lower in self.TURKISH_MONTHS:
                day = int(day_str)
                month = self.TURKISH_MONTHS[month_name_lower]
                year = int(year_str)
                try:
                    return datetime(year, month, day)
                except ValueError:
                    return None

        # If no pattern matched
        return None

    def _validate_date_formats(
        self,
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Validate date formats"""
        for field_name, date_obj in dates.items():
            passed = date_obj is not None

            self.update_check_stats(result, passed)

            if not passed:
                self.add_error(
                    result,
                    "INVALID_DATE_FORMAT",
                    f"Cannot parse date in field '{field_name}'",
                    location=field_name,
                    suggestion="Use format DD.MM.YYYY or Turkish month names (e.g., '15 Ocak 2023')"
                )
                self.date_stats['invalid_formats'] += 1

    def _validate_date_ranges(
        self,
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult,
        allow_future: bool,
        max_future_years: int
    ) -> None:
        """Validate dates are within acceptable historical range"""
        min_date = datetime(self.MIN_YEAR, 1, 1)
        max_date = datetime(self.MAX_YEAR, 12, 31)

        if not allow_future:
            max_date = self.current_date
        else:
            future_limit = self.current_date + timedelta(days=365 * max_future_years)
            max_date = min(max_date, future_limit)

        for field_name, date_obj in dates.items():
            if date_obj is None:
                continue

            # Check minimum date (Turkish Republic founding)
            if date_obj < min_date:
                self.add_error(
                    result,
                    "DATE_TOO_OLD",
                    f"Date in '{field_name}' ({date_obj.strftime('%d.%m.%Y')}) is before {self.MIN_YEAR}",
                    location=field_name,
                    context=f"Turkish legal system began in {self.MIN_YEAR}",
                    suggestion="Verify the date is correct"
                )
                self.date_stats['anomalies_detected'] += 1

            # Check maximum date
            if date_obj > max_date:
                if not allow_future and date_obj > self.current_date:
                    self.add_error(
                        result,
                        "FUTURE_DATE",
                        f"Date in '{field_name}' ({date_obj.strftime('%d.%m.%Y')}) is in the future",
                        location=field_name,
                        suggestion="Future dates not allowed for this document type"
                    )
                    self.date_stats['anomalies_detected'] += 1
                elif date_obj > datetime(self.MAX_YEAR, 12, 31):
                    self.add_warning(
                        result,
                        "DATE_TOO_FAR_FUTURE",
                        f"Date in '{field_name}' ({date_obj.strftime('%d.%m.%Y')}) is beyond {self.MAX_YEAR}",
                        location=field_name,
                        suggestion="Verify this date is realistic"
                    )

    def _validate_date_sequences(
        self,
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Validate logical date sequences"""

        # Publication date should be before or equal to effectivity date
        pub_date = dates.get('publication_date') or dates.get('yayım_tarihi')
        eff_date = dates.get('effectivity_date') or dates.get('yürürlük_tarihi')

        if pub_date and eff_date:
            passed = pub_date <= eff_date

            self.update_check_stats(result, passed)

            if not passed:
                self.add_error(
                    result,
                    "INVALID_DATE_SEQUENCE",
                    f"Publication date ({pub_date.strftime('%d.%m.%Y')}) is after effectivity date ({eff_date.strftime('%d.%m.%Y')})",
                    location="publication_date, effectivity_date",
                    context="Legal documents become effective on or after publication",
                    suggestion="Effectivity date must be >= publication date"
                )
                self.date_stats['temporal_violations'] += 1

        # Enactment date should be before publication date
        enact_date = dates.get('enactment_date') or dates.get('kabul_tarihi')

        if enact_date and pub_date:
            # Allow same day or enactment before publication
            if enact_date > pub_date:
                self.add_warning(
                    result,
                    "UNUSUAL_DATE_SEQUENCE",
                    f"Enactment date ({enact_date.strftime('%d.%m.%Y')}) is after publication date ({pub_date.strftime('%d.%m.%Y')})",
                    location="enactment_date, publication_date",
                    suggestion="Typically documents are enacted before publication"
                )

        # Signature date should be before or equal to publication date
        sig_date = dates.get('signature_date') or dates.get('imza_tarihi')

        if sig_date and pub_date:
            if sig_date > pub_date:
                self.add_warning(
                    result,
                    "UNUSUAL_DATE_SEQUENCE",
                    f"Signature date ({sig_date.strftime('%d.%m.%Y')}) is after publication date ({pub_date.strftime('%d.%m.%Y')})",
                    location="signature_date, publication_date",
                    suggestion="Typically documents are signed before publication"
                )

    def _validate_temporal_relationships(
        self,
        data: Dict[str, Any],
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Validate temporal relationships in document"""

        # Check if effectivity is immediate (yayımı tarihinde yürürlüğe girer)
        pub_date = dates.get('publication_date') or dates.get('yayım_tarihi')
        eff_date = dates.get('effectivity_date') or dates.get('yürürlük_tarihi')

        if pub_date and eff_date:
            days_diff = (eff_date - pub_date).days

            # Check for common delay periods
            if days_diff == 0:
                self.add_info(
                    result,
                    "IMMEDIATE_EFFECTIVITY",
                    "Document becomes effective on publication date",
                    location="effectivity_date",
                    metadata={'delay_days': 0}
                )
            elif days_diff in [30, 60, 90, 180, 365]:
                self.add_info(
                    result,
                    "STANDARD_DELAY_PERIOD",
                    f"Document becomes effective {days_diff} days after publication (standard delay)",
                    location="effectivity_date",
                    metadata={'delay_days': days_diff}
                )
            elif days_diff > 0:
                self.add_info(
                    result,
                    "CUSTOM_DELAY_PERIOD",
                    f"Document becomes effective {days_diff} days after publication",
                    location="effectivity_date",
                    metadata={'delay_days': days_diff}
                )

        # Check for amendment effective dates
        if 'amendments' in data and isinstance(data['amendments'], list):
            for i, amendment in enumerate(data['amendments']):
                if isinstance(amendment, dict):
                    self._validate_amendment_dates(amendment, pub_date, result, f"amendments[{i}]")

    def _validate_amendment_dates(
        self,
        amendment: Dict[str, Any],
        base_pub_date: Optional[datetime],
        result: ValidationResult,
        location: str
    ) -> None:
        """Validate amendment temporal data"""

        if 'date' in amendment:
            amend_date_str = amendment['date']
            amend_date = self._parse_date(amend_date_str, f"{location}.date", result)

            if amend_date and base_pub_date:
                # Amendment should be after original publication
                if amend_date < base_pub_date:
                    self.add_warning(
                        result,
                        "AMENDMENT_BEFORE_ORIGINAL",
                        f"Amendment date ({amend_date.strftime('%d.%m.%Y')}) is before original publication ({base_pub_date.strftime('%d.%m.%Y')})",
                        location=location,
                        suggestion="Amendments typically occur after original publication"
                    )

    def _validate_effectivity_rules(
        self,
        data: Dict[str, Any],
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Validate Turkish legal effectivity rules"""

        # Check for effectivity clause in text
        effectivity_text = None

        # Check common locations for effectivity clause
        for field in ['effectivity_clause', 'yürürlük_maddesi', 'final_article', 'son_madde']:
            if field in data:
                effectivity_text = data[field]
                break

        # Check last article
        if not effectivity_text and 'articles' in data:
            articles = data['articles']
            if isinstance(articles, list) and len(articles) > 0:
                last_article = articles[-1]
                if isinstance(last_article, dict) and 'content' in last_article:
                    content = last_article['content'].lower()
                    if 'yürürlük' in content:
                        effectivity_text = last_article['content']

        if effectivity_text:
            self._parse_effectivity_clause(effectivity_text, dates, result)

    def _parse_effectivity_clause(
        self,
        clause_text: str,
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Parse effectivity clause and validate consistency"""

        clause_lower = clause_text.lower()

        pub_date = dates.get('publication_date') or dates.get('yayım_tarihi')
        eff_date = dates.get('effectivity_date') or dates.get('yürürlük_tarihi')

        # Check for standard effectivity phrases
        for phrase, delay_days in self.EFFECTIVITY_PHRASES.items():
            if phrase in clause_lower:
                if pub_date and eff_date:
                    actual_delay = (eff_date - pub_date).days

                    passed = actual_delay == delay_days

                    self.update_check_stats(result, passed)

                    if not passed:
                        self.add_warning(
                            result,
                            "EFFECTIVITY_CLAUSE_MISMATCH",
                            f"Effectivity clause says '{phrase}' ({delay_days} days) but actual delay is {actual_delay} days",
                            location="effectivity_clause",
                            context=f"Publication: {pub_date.strftime('%d.%m.%Y')}, Effectivity: {eff_date.strftime('%d.%m.%Y')}",
                            suggestion="Ensure effectivity date matches the clause"
                        )

                return  # Found a match, stop checking

        # Check for numeric delay expressions (e.g., "30 gün sonra")
        delay_match = re.search(r'(\d+)\s*(gün|ay|yıl)\s*sonra', clause_lower)
        if delay_match:
            number = int(delay_match.group(1))
            unit = delay_match.group(2)

            if unit == 'gün':
                expected_delay = number
            elif unit == 'ay':
                expected_delay = number * 30  # Approximate
            elif unit == 'yıl':
                expected_delay = number * 365  # Approximate

            if pub_date and eff_date:
                actual_delay = (eff_date - pub_date).days

                # Allow some tolerance for month/year approximations
                tolerance = 5 if unit != 'gün' else 0
                passed = abs(actual_delay - expected_delay) <= tolerance

                self.update_check_stats(result, passed)

                if not passed:
                    self.add_warning(
                        result,
                        "EFFECTIVITY_CLAUSE_MISMATCH",
                        f"Effectivity clause says '{number} {unit} sonra' (~{expected_delay} days) but actual delay is {actual_delay} days",
                        location="effectivity_clause",
                        suggestion="Ensure effectivity date matches the clause"
                    )

    def _detect_temporal_anomalies(
        self,
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Detect temporal anomalies"""

        pub_date = dates.get('publication_date') or dates.get('yayım_tarihi')
        eff_date = dates.get('effectivity_date') or dates.get('yürürlük_tarihi')

        # Extremely long delay (>2 years)
        if pub_date and eff_date:
            delay_days = (eff_date - pub_date).days

            if delay_days > 730:  # 2 years
                self.add_warning(
                    result,
                    "UNUSUALLY_LONG_DELAY",
                    f"Effectivity date is {delay_days} days ({delay_days//365} years) after publication",
                    location="effectivity_date",
                    context="Most laws become effective within 1 year",
                    suggestion="Verify this long delay is intentional"
                )
                self.date_stats['anomalies_detected'] += 1

        # Very old publication date (>50 years)
        if pub_date:
            years_old = (self.current_date - pub_date).days / 365

            if years_old > 50:
                self.add_info(
                    result,
                    "HISTORICAL_DOCUMENT",
                    f"Document was published {int(years_old)} years ago ({pub_date.strftime('%d.%m.%Y')})",
                    location="publication_date",
                    metadata={'years_old': int(years_old)}
                )

        # Weekend publication (unusual for Official Gazette)
        if pub_date:
            # 5 = Saturday, 6 = Sunday
            if pub_date.weekday() in [5, 6]:
                self.add_info(
                    result,
                    "WEEKEND_PUBLICATION",
                    f"Publication date ({pub_date.strftime('%d.%m.%Y')}) is on a weekend",
                    location="publication_date",
                    context="Official Gazette typically publishes on weekdays"
                )

    def _validate_date_consistency(
        self,
        data: Dict[str, Any],
        dates: Dict[str, Optional[datetime]],
        result: ValidationResult
    ) -> None:
        """Validate date consistency across document"""

        # Check if dates appear in multiple formats
        date_formats_found = set()

        for field_name in dates.keys():
            if field_name in data:
                date_str = str(data[field_name])

                for format_name, pattern in self.DATE_PATTERNS.items():
                    if re.match(pattern, date_str):
                        date_formats_found.add(format_name)

        if len(date_formats_found) > 1:
            self.add_warning(
                result,
                "INCONSISTENT_DATE_FORMATS",
                f"Document uses multiple date formats: {', '.join(date_formats_found)}",
                suggestion="Use consistent date format throughout document (prefer DD.MM.YYYY)"
            )

    def _validate_turkish_date_expressions(
        self,
        data: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate Turkish date terminology and expressions"""

        # Check for temporal keywords in document
        temporal_keywords = [
            'yürürlük', 'yayım', 'tarih', 'gün', 'ay', 'yıl',
            'sonra', 'önce', 'itibaren', 'kadar', 'başlamak'
        ]

        # Convert data to searchable text
        text_content = self._extract_text_content(data)

        if text_content:
            text_lower = text_content.lower()

            # Look for effectivity expressions
            if 'yürürlük' in text_lower:
                # Check for missing effectivity date
                if not ('effectivity_date' in data or 'yürürlük_tarihi' in data):
                    self.add_warning(
                        result,
                        "MISSING_EFFECTIVITY_DATE",
                        "Document mentions 'yürürlük' but no effectivity_date field found",
                        suggestion="Add effectivity_date field to document data"
                    )

            # Check for relative date expressions without base date
            if re.search(r'\d+\s*(gün|ay|yıl)\s*sonra', text_lower):
                pub_date = data.get('publication_date') or data.get('yayım_tarihi')
                if not pub_date:
                    self.add_warning(
                        result,
                        "RELATIVE_DATE_WITHOUT_BASE",
                        "Document contains relative date expression but publication_date is missing",
                        suggestion="Add publication_date to calculate effectivity date"
                    )

    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        """Extract all text content from document for analysis"""
        text_parts = []

        # Extract from common text fields
        text_fields = ['title', 'content', 'text', 'preamble', 'effectivity_clause']

        for field in text_fields:
            if field in data and isinstance(data[field], str):
                text_parts.append(data[field])

        # Extract from articles
        if 'articles' in data and isinstance(data['articles'], list):
            for article in data['articles']:
                if isinstance(article, dict):
                    if 'content' in article:
                        text_parts.append(str(article['content']))
                    if 'title' in article:
                        text_parts.append(str(article['title']))

        return ' '.join(text_parts)

    def validate_date_string(self, date_str: str) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """Validate a single date string

        Args:
            date_str: Date string to validate

        Returns:
            Tuple of (is_valid, parsed_date, error_message)
        """
        result = self.create_result()
        parsed_date = self._parse_date(date_str, "date_string", result)

        if parsed_date:
            return (True, parsed_date, None)
        else:
            return (False, None, "Invalid date format")

    def calculate_effectivity_date(
        self,
        publication_date: datetime,
        delay_days: int = 0
    ) -> datetime:
        """Calculate effectivity date from publication date

        Args:
            publication_date: Publication date
            delay_days: Number of days to delay (default: 0)

        Returns:
            Calculated effectivity date
        """
        return publication_date + timedelta(days=delay_days)

    def parse_turkish_delay_expression(self, expression: str) -> Optional[int]:
        """Parse Turkish delay expression to days

        Args:
            expression: Turkish expression (e.g., "30 gün sonra")

        Returns:
            Number of days or None if not recognized
        """
        expression_lower = expression.lower().strip()

        # Check known phrases
        if expression_lower in self.EFFECTIVITY_PHRASES:
            return self.EFFECTIVITY_PHRASES[expression_lower]

        # Parse numeric expressions
        delay_match = re.search(r'(\d+)\s*(gün|ay|yıl)\s*sonra', expression_lower)
        if delay_match:
            number = int(delay_match.group(1))
            unit = delay_match.group(2)

            if unit == 'gün':
                return number
            elif unit == 'ay':
                return number * 30  # Approximate
            elif unit == 'yıl':
                return number * 365  # Approximate

        return None

    def get_date_stats(self) -> Dict[str, Any]:
        """Get date validation statistics

        Returns:
            Dictionary of date statistics
        """
        return {
            **self.date_stats,
            **self.get_stats()
        }


__all__ = ['TemporalValidator']
