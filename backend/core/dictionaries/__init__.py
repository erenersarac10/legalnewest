"""
Turkish Legal Dictionaries Module.

Harvey/Legora %100: Domain-specific terminology management.

This module provides:
- Synonym management for Turkish legal terms
- Multi-language support (TR, EN, AR, ZH)
- Elasticsearch synonyms integration
- RAG query expansion
- Version control and A/B testing

Usage:
    >>> from backend.core.dictionaries import get_synonym_manager
    >>>
    >>> manager = await get_synonym_manager()
    >>> synonyms = manager.get_synonyms("mahkeme")
    >>> # ['divan', 'heyet', 'kurul', ...]
"""

from backend.core.dictionaries.synonym_manager import (
    Synonym Manager,
    SynonymEntry,
    get_synonym_manager,
)

__all__ = [
    "SynonymManager",
    "SynonymEntry",
    "get_synonym_manager",
]
