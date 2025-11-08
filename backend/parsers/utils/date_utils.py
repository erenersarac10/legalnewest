"""
Date Utilities - Harvey/Legora CTO-Level

Turkish date parsing and formatting for legal documents.
Handles various Turkish date formats from legal documents.

Formats Supported:
    - "26 Eylül 2004" (full Turkish month)
    - "26.09.2004" (dot notation)
    - "26/09/2004" (slash notation)
    - "26-09-2004" (dash notation)
    - "Eylül 2004" (month year only)
    - "2004" (year only)

Author: Legal AI Team
Version: 1.0.0
"""

import re
from datetime import datetime, date
from typing import Optional, List, Tuple
from dateutil.parser import parse as dateutil_parse
import calendar


# ============================================================================
# TURKISH MONTH NAMES
# ============================================================================

TURKISH_MONTHS = {
    'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
    'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
    'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
}

TURKISH_MONTHS_ABBR = {
    'oca': 1, 'şub': 2, 'mar': 3, 'nis': 4,
    'may': 5, 'haz': 6, 'tem': 7, 'ağu': 8,
    'eyl': 9, 'eki': 10, 'kas': 11, 'ara': 12
}

MONTH_NAMES = {v: k.capitalize() for k, v in TURKISH_MONTHS.items()}


# ============================================================================
# DATE PARSING
# ============================================================================

def parse_turkish_date(date_str: str) -> Optional[date]:
    """
    Parse Turkish date string to date object.

    Handles various formats:
        - "26 Eylül 2004"
        - "26.09.2004"
        - "26/09/2004"
        - "26-09-2004"

    Args:
        date_str: Date string in Turkish format

    Returns:
        date object or None if parsing fails
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Try Turkish text format first: "26 Eylül 2004"
    result = _parse_turkish_text_date(date_str)
    if result:
        return result

    # Try numeric formats: "26.09.2004", "26/09/2004"
    result = _parse_numeric_date(date_str)
    if result:
        return result

    # Try ISO format
    try:
        return datetime.fromisoformat(date_str).date()
    except:
        pass

    # Last resort: dateutil
    try:
        return dateutil_parse(date_str, dayfirst=True).date()
    except:
        return None


def _parse_turkish_text_date(date_str: str) -> Optional[date]:
    """Parse Turkish text date like '26 Eylül 2004'."""
    # Pattern: "26 Eylül 2004" or "26 Eyl 2004"
    pattern = r'(\d{1,2})\s+([A-Za-zçğıöşüÇĞİÖŞÜ]+)\s+(\d{4})'
    match = re.match(pattern, date_str, re.IGNORECASE)

    if match:
        day = int(match.group(1))
        month_str = match.group(2).lower()
        year = int(match.group(3))

        # Check full month names
        month = TURKISH_MONTHS.get(month_str)

        # Check abbreviated month names
        if not month:
            month = TURKISH_MONTHS_ABBR.get(month_str[:3])

        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None

    return None


def _parse_numeric_date(date_str: str) -> Optional[date]:
    """Parse numeric date like '26.09.2004' or '26/09/2004'."""
    # Replace separators with dots
    normalized = date_str.replace('/', '.').replace('-', '.')

    # Pattern: DD.MM.YYYY or DD.MM.YY
    pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})'
    match = re.match(pattern, normalized)

    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        # Handle 2-digit years
        if year < 100:
            year += 2000 if year < 50 else 1900

        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def extract_dates_from_text(text: str) -> List[Tuple[str, date]]:
    """
    Extract all dates from text.

    Args:
        text: Input text

    Returns:
        List of (original_string, parsed_date) tuples
    """
    dates = []

    # Pattern 1: Turkish text dates
    pattern1 = r'\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}'
    for match in re.finditer(pattern1, text, re.IGNORECASE):
        date_str = match.group()
        parsed = parse_turkish_date(date_str)
        if parsed:
            dates.append((date_str, parsed))

    # Pattern 2: Numeric dates
    pattern2 = r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}'
    for match in re.finditer(pattern2, text):
        date_str = match.group()
        parsed = parse_turkish_date(date_str)
        if parsed:
            dates.append((date_str, parsed))

    return dates


# ============================================================================
# DATE FORMATTING
# ============================================================================

def format_date_turkish(d: date, include_day_name: bool = False) -> str:
    """
    Format date in Turkish format.

    Args:
        d: Date to format
        include_day_name: Whether to include day name

    Returns:
        Formatted date string like "26 Eylül 2004"
    """
    month_name = MONTH_NAMES.get(d.month, "")

    if include_day_name:
        day_names = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        day_name = day_names[d.weekday()]
        return f"{d.day} {month_name} {d.year} {day_name}"

    return f"{d.day} {month_name} {d.year}"


def format_date_resmi_gazete(d: date) -> str:
    """
    Format date in Resmi Gazete style.

    Args:
        d: Date to format

    Returns:
        Formatted string like "26.09.2004"
    """
    return d.strftime("%d.%m.%Y")


# ============================================================================
# DATE RANGE PARSING
# ============================================================================

def parse_date_range(range_str: str) -> Optional[Tuple[date, date]]:
    """
    Parse date range string.

    Formats:
        - "26 Eylül 2004 - 30 Eylül 2004"
        - "26.09.2004-30.09.2004"

    Args:
        range_str: Date range string

    Returns:
        Tuple of (start_date, end_date) or None
    """
    # Split on dash or hyphen
    parts = re.split(r'\s*[-–—]\s*', range_str)

    if len(parts) == 2:
        start = parse_turkish_date(parts[0])
        end = parse_turkish_date(parts[1])

        if start and end:
            return (start, end)

    return None


# ============================================================================
# DATE VALIDATION
# ============================================================================

def is_valid_turkish_date(date_str: str) -> bool:
    """
    Check if string is a valid Turkish date.

    Args:
        date_str: Date string to validate

    Returns:
        True if valid
    """
    return parse_turkish_date(date_str) is not None


def is_future_date(d: date) -> bool:
    """Check if date is in the future."""
    return d > date.today()


def is_legal_document_date(d: date) -> bool:
    """
    Check if date is reasonable for a legal document.

    Legal documents shouldn't be from too far in the past or future.

    Args:
        d: Date to check

    Returns:
        True if reasonable
    """
    # Turkish Republic founded in 1923
    min_year = 1923
    max_year = date.today().year + 5  # Allow 5 years in future

    return min_year <= d.year <= max_year


# ============================================================================
# RELATIVE DATES
# ============================================================================

def parse_relative_date(text: str, reference_date: Optional[date] = None) -> Optional[date]:
    """
    Parse relative date expressions.

    Supports:
        - "bugün" (today)
        - "dün" (yesterday)
        - "yarın" (tomorrow)
        - "geçen hafta" (last week)
        - "gelecek ay" (next month)

    Args:
        text: Relative date text
        reference_date: Reference date (defaults to today)

    Returns:
        Parsed date or None
    """
    if reference_date is None:
        reference_date = date.today()

    text = text.lower().strip()

    from datetime import timedelta

    if text == 'bugün':
        return reference_date
    elif text == 'dün':
        return reference_date - timedelta(days=1)
    elif text == 'yarın':
        return reference_date + timedelta(days=1)
    elif 'geçen hafta' in text:
        return reference_date - timedelta(weeks=1)
    elif 'gelecek hafta' in text:
        return reference_date + timedelta(weeks=1)
    elif 'geçen ay' in text:
        # Subtract one month
        month = reference_date.month - 1
        year = reference_date.year
        if month < 1:
            month = 12
            year -= 1
        return date(year, month, reference_date.day)
    elif 'gelecek ay' in text:
        # Add one month
        month = reference_date.month + 1
        year = reference_date.year
        if month > 12:
            month = 1
            year += 1
        return date(year, month, reference_date.day)

    return None


# ============================================================================
# DATE ARITHMETIC
# ============================================================================

def add_business_days(start_date: date, days: int) -> date:
    """
    Add business days (excluding weekends).

    Args:
        start_date: Starting date
        days: Number of business days to add

    Returns:
        Resulting date
    """
    from datetime import timedelta

    current = start_date
    added = 0

    while added < days:
        current += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            added += 1

    return current


def calculate_deadline(start_date: date, duration_days: int, include_weekends: bool = False) -> date:
    """
    Calculate legal deadline.

    Args:
        start_date: Start date
        duration_days: Duration in days
        include_weekends: Whether to count weekends

    Returns:
        Deadline date
    """
    from datetime import timedelta

    if include_weekends:
        return start_date + timedelta(days=duration_days)
    else:
        return add_business_days(start_date, duration_days)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Parsing
    'parse_turkish_date',
    'extract_dates_from_text',
    'parse_date_range',
    'parse_relative_date',

    # Formatting
    'format_date_turkish',
    'format_date_resmi_gazete',

    # Validation
    'is_valid_turkish_date',
    'is_future_date',
    'is_legal_document_date',

    # Arithmetic
    'add_business_days',
    'calculate_deadline',

    # Constants
    'TURKISH_MONTHS',
    'MONTH_NAMES',
]
