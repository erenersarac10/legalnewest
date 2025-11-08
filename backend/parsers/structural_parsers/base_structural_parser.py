"""Base Structural Parser - Harvey/Legora CTO-Level
Abstract base class for all document structure parsers (hierarchy extraction)
"""
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import re
from ..core import StructuralParser, ParsingResult, LegalDocument, DocumentStructure, LegalClause
from ..utils import normalize_turkish_text, detect_document_sections
from ..presets import CLAUSE_PATTERNS

class BaseStructuralParser(StructuralParser):
    """
    Base class for structural parsers that extract hierarchical document structure.

    Turkish legal documents follow a strict hierarchy:
    - KISIM (Part) → BÖLÜM (Chapter) → MADDE (Article) → FIKRA (Paragraph) → BENT (Clause)

    This parser extracts this hierarchy and builds a tree structure.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.patterns = CLAUSE_PATTERNS

    def _preprocess(self, source: LegalDocument, **kwargs) -> LegalDocument:
        """Normalize text and prepare for structural parsing"""
        if not isinstance(source, LegalDocument):
            raise ValueError("BaseStructuralParser requires LegalDocument as input")

        # Normalize Turkish text
        normalized_text = normalize_turkish_text(source.full_text, preserve_case=True)

        # Create normalized document
        return LegalDocument(
            metadata=source.metadata,
            title=source.title,
            full_text=normalized_text,
            clauses=source.clauses.copy(),
            citations=source.citations.copy()
        )

    @abstractmethod
    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract raw structural data from document.

        Returns:
            Dict with keys: 'preamble', 'parts', 'chapters', 'articles', 'sections'
        """
        pass

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[Any], **kwargs) -> DocumentStructure:
        """Transform raw structural data to canonical DocumentStructure"""
        from ..core.canonical_schema import DocumentStructure, StructuralElement

        elements = []

        # Add preamble if present
        if raw_data.get('preamble'):
            elements.append(StructuralElement(
                element_type='preamble',
                content=raw_data['preamble'],
                level=0,
                position=0
            ))

        # Add parts (KISIM)
        for idx, part in enumerate(raw_data.get('parts', [])):
            elements.append(StructuralElement(
                element_type='part',
                content=part.get('content', ''),
                level=1,
                position=idx,
                title=part.get('title', ''),
                number=part.get('number')
            ))

        # Add chapters (BÖLÜM)
        for idx, chapter in enumerate(raw_data.get('chapters', [])):
            elements.append(StructuralElement(
                element_type='chapter',
                content=chapter.get('content', ''),
                level=2,
                position=idx,
                title=chapter.get('title', ''),
                number=chapter.get('number'),
                parent_number=chapter.get('part_number')
            ))

        # Add articles (MADDE)
        for idx, article in enumerate(raw_data.get('articles', [])):
            elements.append(StructuralElement(
                element_type='article',
                content=article.get('content', ''),
                level=3,
                position=idx,
                title=article.get('title', ''),
                number=article.get('number'),
                parent_number=article.get('chapter_number'),
                is_temporary=article.get('is_temporary', False)
            ))

        return DocumentStructure(
            elements=elements,
            total_articles=len(raw_data.get('articles', [])),
            has_annexes=raw_data.get('has_annexes', False)
        )

    def _extract_preamble(self, text: str) -> Optional[str]:
        """Extract preamble (başlangıç metni) before first MADDE"""
        match = re.search(r'^(.*?)(?=(?:MADDE|Madde|GEÇİCİ MADDE)\s+\d+)', text, re.DOTALL | re.MULTILINE)
        if match:
            preamble = match.group(1).strip()
            # Must be substantial (> 50 chars) to be considered preamble
            if len(preamble) > 50:
                return preamble
        return None

    def _extract_parts(self, text: str) -> List[Dict[str, Any]]:
        """Extract KISIM (Part) sections"""
        parts = []
        pattern = re.compile(r'(?:^|\n)((?:KISIM|Kısım|BİRİNCİ KISIM|İKİNCİ KISIM)\s*:?\s*(.+?))\n', re.MULTILINE)

        for idx, match in enumerate(pattern.finditer(text)):
            full_title = match.group(1).strip()
            title = match.group(2).strip() if match.group(2) else ''

            parts.append({
                'number': idx + 1,
                'title': title,
                'full_title': full_title,
                'start_pos': match.start(),
                'content': ''  # Will be filled by chapter/article extraction
            })

        return parts

    def _extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Extract BÖLÜM (Chapter) sections"""
        chapters = []
        pattern = re.compile(r'(?:^|\n)((?:BÖLÜM|Bölüm|BİRİNCİ BÖLÜM|İKİNCİ BÖLÜM)\s*:?\s*(.+?))\n', re.MULTILINE)

        for idx, match in enumerate(pattern.finditer(text)):
            full_title = match.group(1).strip()
            title = match.group(2).strip() if match.group(2) else ''

            chapters.append({
                'number': idx + 1,
                'title': title,
                'full_title': full_title,
                'start_pos': match.start(),
                'content': ''
            })

        return chapters

    def _extract_articles(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract MADDE (Article) sections with full content.

        Returns list of articles with number, title, and full content (including fıkra/bent)
        """
        articles = []

        # Pattern for both regular and temporary articles
        article_pattern = re.compile(
            r'(?:^|\n)((?:GEÇİCİ\s+)?(?:MADDE|Madde)\s+(\d+))\s*[-–—]?\s*(.*?)(?=(?:\n(?:GEÇİCİ\s+)?(?:MADDE|Madde)\s+\d+|\Z))',
            re.DOTALL | re.MULTILINE
        )

        for match in article_pattern.finditer(text):
            header = match.group(1).strip()
            number = int(match.group(2))
            title_and_content = match.group(3).strip()

            # Extract title (usually first line)
            lines = title_and_content.split('\n', 1)
            title = lines[0].strip() if lines else ''
            content = lines[1].strip() if len(lines) > 1 else ''

            is_temporary = 'GEÇİCİ' in header.upper()

            articles.append({
                'number': number,
                'title': title,
                'content': content,
                'full_text': title_and_content,
                'is_temporary': is_temporary,
                'header': header,
                'start_pos': match.start()
            })

        return articles

    def _validate(self, document: DocumentStructure, **kwargs) -> Tuple[List[str], List[str]]:
        """Validate extracted document structure"""
        errors = []
        warnings = []

        # Check for articles
        if document.total_articles == 0:
            errors.append("No articles (MADDE) found in document")

        # Check for article number continuity
        article_numbers = [elem.number for elem in document.elements if elem.element_type == 'article' and elem.number is not None]
        if article_numbers:
            expected = list(range(1, max(article_numbers) + 1))
            missing = set(expected) - set(article_numbers)
            if missing:
                warnings.append(f"Missing article numbers: {sorted(missing)}")

        # Check for proper hierarchy
        has_parts = any(elem.element_type == 'part' for elem in document.elements)
        has_chapters = any(elem.element_type == 'chapter' for elem in document.elements)

        if has_chapters and not has_parts:
            warnings.append("Document has chapters (BÖLÜM) but no parts (KISIM) - unusual structure")

        return errors, warnings

__all__ = ['BaseStructuralParser']
