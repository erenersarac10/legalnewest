"""Law Structural Parser - Harvey/Legora CTO-Level
Parses Turkish laws (Kanun) with full hierarchy: KISIM → BÖLÜM → MADDE → FIKRA → BENT
"""
from typing import Dict, List, Any, Optional
from .base_structural_parser import BaseStructuralParser
from .clause_hierarchy_builder import ClauseHierarchyBuilder
from ..core import LegalDocument

class LawStructuralParser(BaseStructuralParser):
    """
    Structural parser for Turkish Laws (Kanun).

    Turkish laws typically have this structure:
    - Title (e.g., "5237 Sayılı Türk Ceza Kanunu")
    - Preamble (optional - introduction text)
    - KISIM (Parts) - BİRİNCİ KISIM, İKİNCİ KISIM...
    - BÖLÜM (Chapters) - BİRİNCİ BÖLÜM, İKİNCİ BÖLÜM...
    - MADDE (Articles) - Madde 1, Madde 2...
    - GEÇİCİ MADDE (Temporary Articles)
    - EK MADDE (Additional Articles)

    Each MADDE can contain:
    - FIKRA (Paragraphs) - (1), (2), (3)...
    - BENT (Clauses) - a), b), c)...
    - ALT BENT (Subclauses) - i), ii), iii)...
    """

    def __init__(self):
        super().__init__("Law Structural Parser", "1.0.0")
        self.hierarchy_builder = ClauseHierarchyBuilder()

    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract complete law structure including all hierarchies.

        Returns:
            Dict with: 'preamble', 'parts', 'chapters', 'articles', 'temporary_articles',
                      'additional_articles', 'annexes', 'has_annexes'
        """
        text = preprocessed.full_text

        # Extract preamble
        preamble = self._extract_preamble(text)

        # Extract structural elements
        parts = self._extract_parts(text)
        chapters = self._extract_chapters(text)
        articles = self._extract_articles(text)

        # Separate temporary and additional articles
        regular_articles = [art for art in articles if not art.get('is_temporary')]
        temporary_articles = [art for art in articles if art.get('is_temporary')]

        # Build clause hierarchies for all articles
        for article in regular_articles:
            hierarchy = self.hierarchy_builder.build_hierarchy(
                article.get('full_text', ''),
                article.get('number')
            )
            article['hierarchy'] = hierarchy
            article['clauses'] = self.hierarchy_builder.to_legal_clauses(hierarchy)

        # Check for annexes (usually mentioned at end)
        has_annexes = self._check_for_annexes(text)

        return {
            'preamble': preamble,
            'parts': parts,
            'chapters': chapters,
            'articles': regular_articles,
            'temporary_articles': temporary_articles,
            'has_annexes': has_annexes,
            'document_type': 'kanun'
        }

    def _check_for_annexes(self, text: str) -> bool:
        """Check if document has annexes (EK) mentioned"""
        import re
        annex_pattern = re.compile(r'(?:EK|Ek)\s+(?:\d+|[IVXLCDM]+)', re.IGNORECASE)
        return bool(annex_pattern.search(text))

    def _extract_law_number(self, title: str) -> Optional[str]:
        """Extract law number from title (e.g., '5237' from '5237 Sayılı Türk Ceza Kanunu')"""
        import re
        match = re.search(r'(\d{3,5})\s+[Ss]ayılı', title)
        return match.group(1) if match else None

__all__ = ['LawStructuralParser']
