"""Regulation Structural Parser - Harvey/Legora CTO-Level
Parses Turkish regulations (Yönetmelik) with BÖLÜM → MADDE hierarchy
"""
from typing import Dict, List, Any, Optional
from .base_structural_parser import BaseStructuralParser
from .clause_hierarchy_builder import ClauseHierarchyBuilder
from ..core import LegalDocument

class RegulationStructuralParser(BaseStructuralParser):
    """
    Structural parser for Turkish Regulations (Yönetmelik).

    Regulations typically have simpler structure than laws:
    - Title (e.g., "Ticaret Sicili Yönetmeliği")
    - Amaç ve Kapsam (Purpose and Scope) - usually Madde 1-2
    - Tanımlar (Definitions) - usually Madde 3
    - BÖLÜM (Chapters)
    - MADDE (Articles)
    - Yürürlük (Enforcement) - final article
    - Yürütme (Execution) - final article

    Unlike laws, regulations rarely have KISIM (Parts) structure.
    """

    def __init__(self):
        super().__init__("Regulation Structural Parser", "1.0.0")
        self.hierarchy_builder = ClauseHierarchyBuilder()

    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract regulation structure.

        Returns:
            Dict with: 'preamble', 'chapters', 'articles', 'purpose', 'definitions',
                      'enforcement', 'execution'
        """
        text = preprocessed.full_text

        # Extract preamble (usually before first BÖLÜM or MADDE)
        preamble = self._extract_preamble(text)

        # Extract chapters (BÖLÜM)
        chapters = self._extract_chapters(text)

        # Extract all articles
        articles = self._extract_articles(text)

        # Identify special articles
        purpose_article = self._find_purpose_article(articles)
        definitions_article = self._find_definitions_article(articles)
        enforcement_article = self._find_enforcement_article(articles)
        execution_article = self._find_execution_article(articles)

        # Build clause hierarchies
        for article in articles:
            hierarchy = self.hierarchy_builder.build_hierarchy(
                article.get('full_text', ''),
                article.get('number')
            )
            article['hierarchy'] = hierarchy
            article['clauses'] = self.hierarchy_builder.to_legal_clauses(hierarchy)

        # Associate articles with chapters
        articles_with_chapters = self._associate_articles_with_chapters(articles, chapters)

        return {
            'preamble': preamble,
            'chapters': chapters,
            'articles': articles_with_chapters,
            'purpose': purpose_article,
            'definitions': definitions_article,
            'enforcement': enforcement_article,
            'execution': execution_article,
            'document_type': 'yonetmelik',
            'has_annexes': self._check_for_annexes(text)
        }

    def _find_purpose_article(self, articles: List[Dict]) -> Optional[Dict]:
        """Find 'Amaç' or 'Amaç ve Kapsam' article (usually Madde 1)"""
        for article in articles:
            title = article.get('title', '').lower()
            if 'amaç' in title or 'kapsam' in title:
                return article
        return None

    def _find_definitions_article(self, articles: List[Dict]) -> Optional[Dict]:
        """Find 'Tanımlar' article (usually Madde 3)"""
        for article in articles:
            title = article.get('title', '').lower()
            if 'tanım' in title:
                return article
        return None

    def _find_enforcement_article(self, articles: List[Dict]) -> Optional[Dict]:
        """Find 'Yürürlük' article (usually last or second-to-last)"""
        for article in reversed(articles):
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            if 'yürürlük' in title or 'yürürlüğe' in content[:100]:
                return article
        return None

    def _find_execution_article(self, articles: List[Dict]) -> Optional[Dict]:
        """Find 'Yürütme' article (usually last article)"""
        for article in reversed(articles):
            title = article.get('title', '').lower()
            if 'yürütme' in title:
                return article
        return None

    def _associate_articles_with_chapters(self, articles: List[Dict], chapters: List[Dict]) -> List[Dict]:
        """Associate each article with its containing chapter based on position"""
        if not chapters:
            return articles

        for article in articles:
            article_pos = article.get('start_pos', 0)

            # Find which chapter this article belongs to
            for i, chapter in enumerate(chapters):
                chapter_start = chapter.get('start_pos', 0)
                chapter_end = chapters[i + 1].get('start_pos', float('inf')) if i + 1 < len(chapters) else float('inf')

                if chapter_start <= article_pos < chapter_end:
                    article['chapter_number'] = chapter.get('number')
                    article['chapter_title'] = chapter.get('title')
                    break

        return articles

    def _check_for_annexes(self, text: str) -> bool:
        """Check if regulation has annexes (EK)"""
        import re
        annex_pattern = re.compile(r'(?:EK|Ek)\s+(?:\d+|[IVXLCDM]+)', re.IGNORECASE)
        return bool(annex_pattern.search(text))

__all__ = ['RegulationStructuralParser']
