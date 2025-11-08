"""
Regex Utilities - Harvey/Legora CTO-Level

Compiled regex patterns with caching for Turkish legal documents.
Provides common patterns for citations, dates, article numbers, etc.

Performance: Compiled regexes are cached for reuse.

Author: Legal AI Team
Version: 1.0.0
"""

import re
from typing import Dict, Pattern, List, Optional
from functools import lru_cache


# ============================================================================
# REGEX CACHE
# ============================================================================

_regex_cache: Dict[str, Pattern] = {}


@lru_cache(maxsize=128)
def compile_cached(pattern: str, flags: int = 0) -> Pattern:
    """
    Compile regex with caching.

    Args:
        pattern: Regex pattern
        flags: Regex flags

    Returns:
        Compiled pattern
    """
    cache_key = f"{pattern}_{flags}"

    if cache_key not in _regex_cache:
        _regex_cache[cache_key] = re.compile(pattern, flags)

    return _regex_cache[cache_key]


# ============================================================================
# COMMON PATTERNS
# ============================================================================

# Law number patterns
LAW_NUMBER_PATTERN = r'\d{3,5}\s+(?:[Ss]ayılı|SAYILI)'
KANUN_PATTERN = r'(?:\d{3,5}\s+[Ss]ayılı\s+)?(?:[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+){0,3}Kanun(?:u|unun)?'

# Article patterns
MADDE_PATTERN = r'(?:Madde|MADDE|madde)\s+\d+'
FIKRA_PATTERN = r'(?:Fıkra|FIKRA|fıkra)\s+\d+'
BENT_PATTERN = r'(?:Bent|BENT|bent)\s+[a-z]\)'

# Date patterns
TURKISH_DATE_PATTERN = r'\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}'
NUMERIC_DATE_PATTERN = r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}'

# Resmi Gazete patterns
RG_NUMBER_PATTERN = r'(?:RG|R\.G\.|Resmi Gazete)\s*(?:No|Sayı|:)?\s*:?\s*(\d+)'
RG_DATE_PATTERN = r'(?:RG|R\.G\.|Resmi Gazete).*?(\d{1,2}[./]\d{1,2}[./]\d{4})'

# Court patterns
YARGITAY_PATTERN = r'Yargıtay\s+(?:\d+\.?\s*)?(?:Hukuk|Ceza)\s+Daire(?:si)?'
DANISTAY_PATTERN = r'Danıştay\s+\d+\.?\s*Daire(?:si)?'
AYM_PATTERN = r'Anayasa Mahkemesi|AYM|T\.C\.\s*Anayasa\s*Mahkemesi'

# Case number patterns
ESAS_NO_PATTERN = r'(?:Esas|E\.)\s*(?:No|:)?\s*:?\s*(\d{4}/\d+)'
KARAR_NO_PATTERN = r'(?:Karar|K\.)\s*(?:No|:)?\s*:?\s*(\d{4}/\d+)'


# ============================================================================
# PRECOMPILED PATTERNS
# ============================================================================

class CompiledPatterns:
    """Namespace for precompiled regex patterns."""

    LAW_NUMBER = compile_cached(LAW_NUMBER_PATTERN, re.IGNORECASE)
    KANUN = compile_cached(KANUN_PATTERN, re.IGNORECASE)

    MADDE = compile_cached(MADDE_PATTERN, re.IGNORECASE)
    FIKRA = compile_cached(FIKRA_PATTERN, re.IGNORECASE)
    BENT = compile_cached(BENT_PATTERN, re.IGNORECASE)

    TURKISH_DATE = compile_cached(TURKISH_DATE_PATTERN, re.IGNORECASE)
    NUMERIC_DATE = compile_cached(NUMERIC_DATE_PATTERN)

    RG_NUMBER = compile_cached(RG_NUMBER_PATTERN, re.IGNORECASE)
    RG_DATE = compile_cached(RG_DATE_PATTERN, re.IGNORECASE)

    YARGITAY = compile_cached(YARGITAY_PATTERN, re.IGNORECASE)
    DANISTAY = compile_cached(DANISTAY_PATTERN, re.IGNORECASE)
    AYM = compile_cached(AYM_PATTERN, re.IGNORECASE)

    ESAS_NO = compile_cached(ESAS_NO_PATTERN, re.IGNORECASE)
    KARAR_NO = compile_cached(KARAR_NO_PATTERN, re.IGNORECASE)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_law_numbers(text: str) -> List[str]:
    """
    Extract law numbers from text.

    Args:
        text: Input text

    Returns:
        List of law numbers (e.g., ["5237 Sayılı", "6098 sayılı"])
    """
    return CompiledPatterns.LAW_NUMBER.findall(text)


def extract_article_numbers(text: str) -> List[str]:
    """
    Extract article numbers from text.

    Args:
        text: Input text

    Returns:
        List of article numbers (e.g., ["Madde 141", "madde 52"])
    """
    return CompiledPatterns.MADDE.findall(text)


def extract_rg_numbers(text: str) -> List[str]:
    """
    Extract Resmi Gazete numbers from text.

    Args:
        text: Input text

    Returns:
        List of RG numbers
    """
    matches = CompiledPatterns.RG_NUMBER.findall(text)
    return [m if isinstance(m, str) else m[0] for m in matches]


def extract_case_numbers(text: str) -> List[Dict[str, str]]:
    """
    Extract case numbers (Esas/Karar) from text.

    Args:
        text: Input text

    Returns:
        List of dicts with 'esas' and/or 'karar' keys
    """
    results = []

    esas_matches = CompiledPatterns.ESAS_NO.findall(text)
    karar_matches = CompiledPatterns.KARAR_NO.findall(text)

    for i, esas in enumerate(esas_matches):
        result = {'esas': esas}
        if i < len(karar_matches):
            result['karar'] = karar_matches[i]
        results.append(result)

    return results


def extract_court_names(text: str) -> List[str]:
    """
    Extract court names from text.

    Args:
        text: Input text

    Returns:
        List of court names
    """
    courts = []

    yargitay = CompiledPatterns.YARGITAY.findall(text)
    danistay = CompiledPatterns.DANISTAY.findall(text)
    aym = CompiledPatterns.AYM.findall(text)

    courts.extend(yargitay)
    courts.extend(danistay)
    courts.extend(aym)

    return courts


# ============================================================================
# PATTERN MATCHING
# ============================================================================

def is_law_citation(text: str) -> bool:
    """
    Check if text is a law citation.

    Args:
        text: Text to check

    Returns:
        True if matches law citation pattern
    """
    return bool(CompiledPatterns.LAW_NUMBER.search(text))


def is_article_reference(text: str) -> bool:
    """
    Check if text is an article reference.

    Args:
        text: Text to check

    Returns:
        True if matches article pattern
    """
    return bool(CompiledPatterns.MADDE.search(text))


def is_court_decision(text: str) -> bool:
    """
    Check if text mentions a court decision.

    Args:
        text: Text to check

    Returns:
        True if mentions court
    """
    return bool(
        CompiledPatterns.YARGITAY.search(text) or
        CompiledPatterns.DANISTAY.search(text) or
        CompiledPatterns.AYM.search(text)
    )


# ============================================================================
# TEXT SPLITTING
# ============================================================================

def split_by_articles(text: str) -> List[str]:
    """
    Split text by article markers (MADDE).

    Args:
        text: Input text

    Returns:
        List of text chunks (one per article)
    """
    # Split on "MADDE" or "Madde"
    pattern = r'(?=(?:MADDE|Madde)\s+\d+)'
    parts = re.split(pattern, text)

    # Remove empty parts
    return [p.strip() for p in parts if p.strip()]


def split_by_sections(text: str) -> List[str]:
    """
    Split text by major sections (BÖLÜM, KISIM).

    Args:
        text: Input text

    Returns:
        List of text chunks
    """
    pattern = r'(?=(?:BÖLÜM|KISIM|BİRİNCİ|İKİNCİ|ÜÇÜNCÜ))'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    return [p.strip() for p in parts if p.strip()]


# ============================================================================
# CLEANING
# ============================================================================

def remove_line_numbers(text: str) -> str:
    """
    Remove line numbers from text.

    Args:
        text: Input text

    Returns:
        Text with line numbers removed
    """
    # Remove standalone numbers at line start
    pattern = r'^\s*\d+\s+'
    return re.sub(pattern, '', text, flags=re.MULTILINE)


def remove_page_numbers(text: str) -> str:
    """
    Remove page numbers from text.

    Args:
        text: Input text

    Returns:
        Text with page numbers removed
    """
    # Remove "Sayfa X" or "Page X"
    pattern = r'(?:Sayfa|Page)\s+\d+\s*'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)


def normalize_spaces(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized spaces
    """
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)

    # Replace multiple newlines with max 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================================================
# VALIDATION
# ============================================================================

def is_valid_law_number(number_str: str) -> bool:
    """
    Validate law number format.

    Args:
        number_str: Number string to validate

    Returns:
        True if valid format
    """
    # Should be 3-5 digits followed by "Sayılı" or "sayılı"
    pattern = r'^\d{3,5}\s+[Ss]ayılı$'
    return bool(re.match(pattern, number_str))


def is_valid_rg_number(rg_str: str) -> bool:
    """
    Validate Resmi Gazete number format.

    Args:
        rg_str: RG string to validate

    Returns:
        True if valid format
    """
    # Should be 5 digits
    return bool(re.match(r'^\d{5}$', rg_str))


def is_valid_case_number(case_str: str) -> bool:
    """
    Validate case number format (YYYY/NNNN).

    Args:
        case_str: Case number to validate

    Returns:
        True if valid format
    """
    return bool(re.match(r'^\d{4}/\d+$', case_str))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Caching
    'compile_cached',

    # Patterns
    'CompiledPatterns',

    # Extraction
    'extract_law_numbers',
    'extract_article_numbers',
    'extract_rg_numbers',
    'extract_case_numbers',
    'extract_court_names',

    # Matching
    'is_law_citation',
    'is_article_reference',
    'is_court_decision',

    # Splitting
    'split_by_articles',
    'split_by_sections',

    # Cleaning
    'remove_line_numbers',
    'remove_page_numbers',
    'normalize_spaces',

    # Validation
    'is_valid_law_number',
    'is_valid_rg_number',
    'is_valid_case_number',
]
