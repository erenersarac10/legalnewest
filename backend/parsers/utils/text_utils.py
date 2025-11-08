"""
Text Processing Utilities - Harvey/Legora CTO-Level

Advanced text processing for Turkish legal documents.
Handles Turkish character normalization, cleaning, tokenization, and analysis.

Features:
    - Turkish character normalization (İ→I, Ş→S, Ğ→G, etc.)
    - Legal text cleaning (headers, footers, page numbers)
    - Smart tokenization (preserves legal citations)
    - Text similarity and comparison
    - Layout analysis and structure detection

Author: Legal AI Team
Version: 1.0.0
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Dict, Set
from difflib import SequenceMatcher


# ============================================================================
# TURKISH CHARACTER MAPS
# ============================================================================

TURKISH_TO_LATIN = {
    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
}

LATIN_TO_TURKISH = {v: k for k, v in TURKISH_TO_LATIN.items()}


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_turkish_text(text: str, preserve_case: bool = True) -> str:
    """
    Normalize Turkish text for processing.

    Args:
        text: Input text
        preserve_case: Whether to preserve case

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Normalize Unicode (NFC)
    text = unicodedata.normalize('NFC', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')

    if not preserve_case:
        text = text.lower()

    return text


def turkishify(text: str) -> str:
    """
    Convert Latin characters back to Turkish equivalents.

    Args:
        text: Text with Latin characters

    Returns:
        Text with Turkish characters
    """
    for latin, turkish in LATIN_TO_TURKISH.items():
        text = text.replace(latin, turkish)
    return text


def latinize(text: str) -> str:
    """
    Convert Turkish characters to Latin equivalents for search/comparison.

    Args:
        text: Text with Turkish characters

    Returns:
        Text with Latin characters
    """
    for turkish, latin in TURKISH_TO_LATIN.items():
        text = text.replace(turkish, latin)
    return text


# ============================================================================
# CLEANING
# ============================================================================

def clean_legal_text(text: str) -> str:
    """
    Clean legal document text (remove headers, footers, page numbers).

    Args:
        text: Raw legal document text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip page numbers (standalone numbers)
        if re.match(r'^\d+$', line):
            continue

        # Skip common headers/footers
        if any(keyword in line.lower() for keyword in [
            'sayfa', 'page', 'resmi gazete', 'www.', 'http://', 'https://'
        ]):
            continue

        # Skip very short lines (likely artifacts)
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_extra_whitespace(text: str) -> str:
    """
    Remove excessive whitespace while preserving structure.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Remove multiple newlines (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove spaces at line start/end
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def extract_clean_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """
    Extract clean paragraphs from text.

    Args:
        text: Input text
        min_length: Minimum paragraph length

    Returns:
        List of paragraphs
    """
    text = clean_legal_text(text)
    text = remove_extra_whitespace(text)

    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_preserving_citations(text: str) -> List[str]:
    """
    Tokenize text while preserving legal citations as single tokens.

    Legal citations like "5237 Sayılı TCK Madde 141" stay together.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Patterns to preserve
    citation_pattern = r'\d+\s+(?:Sayılı|sayılı)\s+[A-ZÇĞİÖŞÜ]+(?:\s+Madde\s+\d+)?'
    article_pattern = r'Madde\s+\d+'
    date_pattern = r'\d{1,2}[./]\d{1,2}[./]\d{2,4}'

    # Replace patterns with placeholders
    placeholders = {}
    counter = 0

    for pattern in [citation_pattern, article_pattern, date_pattern]:
        for match in re.finditer(pattern, text):
            placeholder = f'__PRESERVE_{counter}__'
            placeholders[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)
            counter += 1

    # Tokenize
    tokens = text.split()

    # Restore placeholders
    restored_tokens = []
    for token in tokens:
        if token in placeholders:
            restored_tokens.append(placeholders[token])
        else:
            restored_tokens.append(token)

    return restored_tokens


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences (Turkish-aware).

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Turkish sentence endings
    sentence_enders = r'[.!?]'

    # Split on sentence enders followed by space and capital letter
    sentences = re.split(r'([.!?])\s+(?=[A-ZÇĞIÖŞÜ])', text)

    # Reconstruct sentences
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])

    return [s.strip() for s in result if s.strip()]


# ============================================================================
# SIMILARITY & COMPARISON
# ============================================================================

def text_similarity(text1: str, text2: str, normalize: bool = True) -> float:
    """
    Calculate similarity between two texts (0.0 to 1.0).

    Args:
        text1: First text
        text2: Second text
        normalize: Whether to normalize before comparison

    Returns:
        Similarity score
    """
    if normalize:
        text1 = normalize_turkish_text(text1, preserve_case=False)
        text2 = normalize_turkish_text(text2, preserve_case=False)

    return SequenceMatcher(None, text1, text2).ratio()


def find_text_differences(text1: str, text2: str) -> List[Tuple[str, str, str]]:
    """
    Find differences between two texts.

    Args:
        text1: Original text
        text2: Modified text

    Returns:
        List of (operation, original, modified) tuples
    """
    import difflib

    matcher = difflib.SequenceMatcher(None, text1, text2)
    differences = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            differences.append(('replace', text1[i1:i2], text2[j1:j2]))
        elif tag == 'delete':
            differences.append(('delete', text1[i1:i2], ''))
        elif tag == 'insert':
            differences.append(('insert', '', text2[j1:j2]))

    return differences


# ============================================================================
# STRUCTURE DETECTION
# ============================================================================

def detect_document_sections(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect major sections in a legal document.

    Args:
        text: Document text

    Returns:
        List of (section_title, start_pos, end_pos)
    """
    sections = []

    # Common section headers in Turkish legal documents
    section_patterns = [
        r'^(?:BÖLÜM|BİRİNCİ BÖLÜM|İKİNCİ BÖLÜM|KISIM|BİRİNCİ KISIM)',
        r'^(?:MADDE|Madde)\s+\d+',
        r'^(?:GEÇİCİ MADDE|Geçici Madde)',
        r'^(?:EK MADDE|Ek Madde)',
    ]

    lines = text.split('\n')
    current_pos = 0

    for i, line in enumerate(lines):
        line = line.strip()

        for pattern in section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                sections.append((line, current_pos, -1))

        current_pos += len(line) + 1  # +1 for newline

    # Set end positions
    for i in range(len(sections) - 1):
        sections[i] = (sections[i][0], sections[i][1], sections[i + 1][1])

    if sections:
        sections[-1] = (sections[-1][0], sections[-1][1], len(text))

    return sections


def is_legal_heading(text: str) -> bool:
    """
    Check if text is a legal document heading.

    Args:
        text: Text to check

    Returns:
        True if heading
    """
    text = text.strip()

    # All caps
    if text.isupper() and len(text) > 3:
        return True

    # Starts with MADDE/BÖLÜM/etc.
    heading_keywords = ['MADDE', 'BÖLÜM', 'KISIM', 'EK', 'GEÇİCİ']
    if any(text.upper().startswith(kw) for kw in heading_keywords):
        return True

    return False


# ============================================================================
# WORD ANALYSIS
# ============================================================================

def count_words(text: str) -> int:
    """Count words in text."""
    return len(tokenize_preserving_citations(text))


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    return len(split_sentences(text))


def calculate_readability_score(text: str) -> float:
    """
    Calculate simple readability score (higher = easier).

    Based on average word length and sentence length.

    Args:
        text: Input text

    Returns:
        Readability score (0-100, higher = easier)
    """
    words = tokenize_preserving_citations(text)
    sentences = split_sentences(text)

    if not words or not sentences:
        return 0.0

    avg_word_length = sum(len(w) for w in words) / len(words)
    avg_sentence_length = len(words) / len(sentences)

    # Simple formula (lower = easier)
    difficulty = (avg_word_length * 1.5) + (avg_sentence_length * 0.5)

    # Invert and scale to 0-100
    readability = max(0, min(100, 100 - (difficulty * 2)))

    return readability


def extract_keywords(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extract top keywords from text.

    Args:
        text: Input text
        top_n: Number of keywords to return

    Returns:
        List of (keyword, frequency) tuples
    """
    # Turkish stopwords
    stopwords = {
        've', 'veya', 'ile', 'bir', 'bu', 'şu', 'o', 'da', 'de',
        'ki', 'mi', 'mu', 'mı', 'mü', 'için', 'gibi', 'kadar',
        'daha', 'çok', 'az', 'her', 'hiç', 'bazı', 'bütün'
    }

    words = tokenize_preserving_citations(text)
    words = [w.lower() for w in words if len(w) > 3]
    words = [w for w in words if w not in stopwords]

    # Count frequencies
    freq: Dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:top_n]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Normalization
    'normalize_turkish_text',
    'turkishify',
    'latinize',

    # Cleaning
    'clean_legal_text',
    'remove_extra_whitespace',
    'extract_clean_paragraphs',

    # Tokenization
    'tokenize_preserving_citations',
    'split_sentences',

    # Similarity
    'text_similarity',
    'find_text_differences',

    # Structure
    'detect_document_sections',
    'is_legal_heading',

    # Analysis
    'count_words',
    'count_sentences',
    'calculate_readability_score',
    'extract_keywords',
]
