"""Parser Presets Module"""
from .citation_patterns import PATTERNS as CITATION_PATTERNS
from .court_patterns import COURTS, COMPILED as COURT_PATTERNS
from .clause_patterns import PATTERNS as CLAUSE_PATTERNS
from .regex_patterns import COMPILED as REGEX_PATTERNS
from .keyword_lexicon import ALL_KEYWORDS, DOCUMENT_TYPES, LEGAL_ACTIONS
from .source_mappings import SOURCE_URLS, API_ENDPOINTS

__all__ = [
    'CITATION_PATTERNS', 'COURT_PATTERNS', 'CLAUSE_PATTERNS', 'REGEX_PATTERNS',
    'ALL_KEYWORDS', 'DOCUMENT_TYPES', 'SOURCE_URLS', 'API_ENDPOINTS'
]
