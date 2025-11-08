"""CBK (Presidential Decree) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses Cumhurbaşkanlığı Kararnamesi (Presidential Decrees) structure

Production Features:
- Hierarchical structure parsing (Kısım > Bölüm > Madde)
- Article extraction with numbering validation
- Paragraph and subparagraph detection
- Annex and attachment handling
- Cross-reference extraction
- Comprehensive error handling and validation
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag

from ..core import StructuralParser, ParsedElement, DocumentStructure
from ..errors import ParsingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class CBKArticle:
    """Represents a CBK article (Madde)"""
    number: int
    title: Optional[str]
    content: str
    paragraphs: List[str]
    subparagraphs: List[Tuple[str, str]]  # (letter, content)
    cross_references: List[str]


@dataclass
class CBKChapter:
    """Represents a CBK chapter (Bölüm)"""
    number: int
    title: str
    articles: List[CBKArticle]


@dataclass
class CBKPart:
    """Represents a CBK part (Kısım)"""
    number: int
    title: str
    chapters: List[CBKChapter]


class CBKStructuralParser(StructuralParser):
    """Presidential Decree (Cumhurbaşkanlığı Kararnamesi) Structural Parser

    Parses the hierarchical structure of Turkish Presidential Decrees:
    - CBK Number and Title
    - Preamble (Başlangıç)
    - Parts (Kısım) - optional
    - Chapters (Bölüm) - optional
    - Articles (Madde) - required
    - Paragraphs and subparagraphs
    - Annexes (Ek)

    Features:
    - Validates CBK number format (e.g., "1 Sayılı CBK", "Cumhurbaşkanlığı Kararnamesi No: 5")
    - Extracts hierarchical structure
    - Handles numbered and lettered subdivisions
    - Detects cross-references
    - Parses annexes and attachments
    """

    # CBK number patterns
    CBK_NUMBER_PATTERNS = [
        r'(\d+)\s+[Ss]ayılı\s+(?:Cumhurbaşkanlığı\s+)?[Kk]ararname',
        r'Cumhurbaşkanlığı\s+Kararnamesi\s+(?:No|Sayı)\s*:?\s*(\d+)',
        r'CBK\s+(?:No|Sayı)\s*:?\s*(\d+)',
        r'Karar(?:name)?\s+(?:No|Sayı)\s*:?\s*(\d+)'
    ]

    # Structural markers
    PART_PATTERN = r'(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s+KISIM'
    CHAPTER_PATTERN = r'(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU|ONBİRİNCİ|ONİKİNCİ)\s+BÖLÜM'
    ARTICLE_PATTERN = r'(?:MADDE|Madde)\s+(\d+)\s*[–-]\s*(.+)'

    # Turkish ordinal numbers
    ORDINALS = {
        'BİRİNCİ': 1, 'İKİNCİ': 2, 'ÜÇÜNCÜ': 3, 'DÖRDÜNCÜ': 4, 'BEŞİNCİ': 5,
        'ALTINCI': 6, 'YEDİNCİ': 7, 'SEKİZİNCİ': 8, 'DOKUZUNCU': 9, 'ONUNCU': 10,
        'ONBİRİNCİ': 11, 'ONİKİNCİ': 12, 'ONÜÇüNCÜ': 13, 'ONDÖRDÜNCÜ': 14, 'ONBEŞİNCİ': 15
    }

    def __init__(self):
        super().__init__("CBK Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse CBK structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode, validate_numbering)

        Returns:
            DocumentStructure with parsed hierarchy

        Raises:
            ParsingError: If parsing fails
            ValidationError: If structure is invalid
        """
        try:
            html_mode = kwargs.get('html_mode', False)

            if html_mode:
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
            else:
                text = content

            # Extract CBK number
            cbk_number = self._extract_cbk_number(text)
            if not cbk_number:
                logger.warning("Could not extract CBK number")

            # Extract title
            title = self._extract_title(text)

            # Extract preamble if exists
            preamble = self._extract_preamble(text)

            # Parse hierarchical structure
            parts = self._parse_parts(text)

            # If no parts, parse chapters directly
            if not parts:
                chapters = self._parse_chapters(text)

                # If no chapters, parse articles directly
                if not chapters:
                    articles = self._parse_articles(text)
                    structure_type = 'SIMPLE'
                else:
                    articles = []
                    for chapter in chapters:
                        articles.extend(chapter.articles)
                    structure_type = 'CHAPTER_BASED'
            else:
                articles = []
                for part in parts:
                    for chapter in part.chapters:
                        articles.extend(chapter.articles)
                structure_type = 'FULL_HIERARCHY'

            # Validate article numbering
            if kwargs.get('validate_numbering', True):
                self._validate_article_numbering(articles)

            # Extract annexes
            annexes = self._extract_annexes(text)

            logger.info(f"Successfully parsed CBK: {cbk_number}, {len(articles)} articles, "
                       f"{len(parts)} parts, {len(annexes)} annexes")

            return DocumentStructure(
                document_type='CBK',
                cbk_number=cbk_number,
                title=title,
                preamble=preamble,
                parts=parts,
                articles=articles,
                annexes=annexes,
                structure_type=structure_type,
                metadata={
                    'total_articles': len(articles),
                    'total_parts': len(parts),
                    'total_annexes': len(annexes)
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse CBK structure: {e}")
            raise ParsingError(f"CBK parsing failed: {str(e)}") from e

    def _extract_cbk_number(self, text: str) -> Optional[str]:
        """Extract CBK number from text"""
        for pattern in self.CBK_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                logger.debug(f"Extracted CBK number: {number}")
                return number

        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract CBK title (usually first line or after number)"""
        lines = text.split('\n')

        # Look for title pattern in first few lines
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and len(line) < 200:
                # Skip if it's the number line
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.CBK_NUMBER_PATTERNS):
                    continue

                # Common title patterns
                if any(keyword in line.upper() for keyword in ['HAKKINDA', 'İLE İLGİLİ', 'DAİR', 'KARARNAMESĐ']):
                    logger.debug(f"Extracted title: {line}")
                    return line

        return None

    def _extract_preamble(self, text: str) -> Optional[str]:
        """Extract preamble (Başlangıç) section"""
        # Preamble usually starts after title and before first KISIM/BÖLÜM/MADDE
        preamble_pattern = r'(?:Amaç|Kapsam|Dayanak|Tanımlar).*?(?=(?:BİRİNCİ\s+(?:KISIM|BÖLÜM)|MADDE\s+1))'
        match = re.search(preamble_pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            preamble = match.group(0).strip()
            logger.debug(f"Extracted preamble ({len(preamble)} chars)")
            return preamble

        return None

    def _parse_parts(self, text: str) -> List[CBKPart]:
        """Parse parts (Kısım) from text"""
        parts = []

        # Find all part markers
        part_matches = list(re.finditer(self.PART_PATTERN, text, re.IGNORECASE))

        if not part_matches:
            return parts

        for i, match in enumerate(part_matches):
            part_start = match.start()
            part_end = part_matches[i + 1].start() if i + 1 < len(part_matches) else len(text)
            part_text = text[part_start:part_end]

            # Extract part number from ordinal
            part_header = match.group(0)
            ordinal = part_header.split()[0]
            part_number = self.ORDINALS.get(ordinal, i + 1)

            # Extract part title (line after KISIM marker)
            lines = part_text.split('\n')
            part_title = lines[1].strip() if len(lines) > 1 else ''

            # Parse chapters within this part
            chapters = self._parse_chapters(part_text)

            parts.append(CBKPart(
                number=part_number,
                title=part_title,
                chapters=chapters
            ))

            logger.debug(f"Parsed Part {part_number}: {len(chapters)} chapters")

        return parts

    def _parse_chapters(self, text: str) -> List[CBKChapter]:
        """Parse chapters (Bölüm) from text"""
        chapters = []

        # Find all chapter markers
        chapter_matches = list(re.finditer(self.CHAPTER_PATTERN, text, re.IGNORECASE))

        if not chapter_matches:
            return chapters

        for i, match in enumerate(chapter_matches):
            chapter_start = match.start()
            chapter_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_text = text[chapter_start:chapter_end]

            # Extract chapter number from ordinal
            chapter_header = match.group(0)
            ordinal = chapter_header.split()[0]
            chapter_number = self.ORDINALS.get(ordinal, i + 1)

            # Extract chapter title
            lines = chapter_text.split('\n')
            chapter_title = lines[1].strip() if len(lines) > 1 else ''

            # Parse articles within this chapter
            articles = self._parse_articles(chapter_text)

            chapters.append(CBKChapter(
                number=chapter_number,
                title=chapter_title,
                articles=articles
            ))

            logger.debug(f"Parsed Chapter {chapter_number}: {len(articles)} articles")

        return chapters

    def _parse_articles(self, text: str) -> List[CBKArticle]:
        """Parse articles (Madde) from text"""
        articles = []

        # Find all article markers
        article_matches = list(re.finditer(self.ARTICLE_PATTERN, text, re.IGNORECASE))

        for i, match in enumerate(article_matches):
            article_number = int(match.group(1))
            article_title = match.group(2).strip()

            # Extract article content
            article_start = match.end()
            article_end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
            article_text = text[article_start:article_end].strip()

            # Parse paragraphs
            paragraphs = self._parse_paragraphs(article_text)

            # Parse subparagraphs (lettered items)
            subparagraphs = self._parse_subparagraphs(article_text)

            # Extract cross-references
            cross_refs = self._extract_cross_references(article_text)

            articles.append(CBKArticle(
                number=article_number,
                title=article_title,
                content=article_text,
                paragraphs=paragraphs,
                subparagraphs=subparagraphs,
                cross_references=cross_refs
            ))

            logger.debug(f"Parsed Article {article_number}: {len(paragraphs)} paragraphs, "
                        f"{len(subparagraphs)} subparagraphs")

        return articles

    def _parse_paragraphs(self, text: str) -> List[str]:
        """Parse numbered paragraphs from article text"""
        paragraphs = []

        # Pattern: (1), (2), etc.
        paragraph_pattern = r'\((\d+)\)\s*([^(]+?)(?=\(\d+\)|$)'
        matches = re.finditer(paragraph_pattern, text, re.DOTALL)

        for match in matches:
            para_number = match.group(1)
            para_content = match.group(2).strip()
            paragraphs.append(para_content)

        return paragraphs

    def _parse_subparagraphs(self, text: str) -> List[Tuple[str, str]]:
        """Parse lettered subparagraphs from article text"""
        subparagraphs = []

        # Pattern: a), b), c) or a-, b-, c-
        subpara_pattern = r'([a-zğüşöçı])[)\-]\s*([^a-zğüşöçı][^)]+?)(?=[a-zğüşöçı][)\-]|$)'
        matches = re.finditer(subpara_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            letter = match.group(1)
            content = match.group(2).strip()
            subparagraphs.append((letter, content))

        return subparagraphs

    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references to other articles/laws"""
        references = []

        # Pattern: "bu Kararname", "bu madde", "X sayılı madde", etc.
        patterns = [
            r'(\d+)\s+sayılı\s+(?:madde|Madde)',
            r'bu\s+(?:Kararname|madde)',
            r'(\d+)\s+(?:inci|nci|üncü|uncu)\s+madde'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append(match.group(0))

        return list(set(references))  # Remove duplicates

    def _extract_annexes(self, text: str) -> List[Dict[str, Any]]:
        """Extract annexes (Ek) from text"""
        annexes = []

        # Pattern: "EK-1", "EK: 1", "EKLER"
        annex_patterns = [
            r'EK[:\-\s]+(\d+)',
            r'EK\s+([A-ZÇĞİÖŞÜ])',
            r'EKLER\s*:?'
        ]

        for pattern in annex_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                annex_marker = match.group(0)
                annex_start = match.start()

                # Extract annex content (next 500 chars or until next annex)
                annex_content = text[annex_start:annex_start + 500].strip()

                annexes.append({
                    'marker': annex_marker,
                    'content_preview': annex_content[:200]
                })

        return annexes

    def _validate_article_numbering(self, articles: List[CBKArticle]):
        """Validate that article numbering is sequential"""
        if not articles:
            return

        for i, article in enumerate(articles):
            expected_number = i + 1
            if article.number != expected_number:
                logger.warning(f"Article numbering gap: expected {expected_number}, "
                             f"found {article.number}")
                raise ValidationError(f"Invalid article numbering at position {i + 1}")

        logger.debug(f"Article numbering validated: 1-{len(articles)} sequential")

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed CBK structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not structure.articles or len(structure.articles) == 0:
            raise ValidationError("CBK must contain at least one article")

        if structure.cbk_number is None:
            logger.warning("CBK number is missing")

        # Validate article numbering
        self._validate_article_numbering(structure.articles)

        logger.info("CBK structure validation passed")
        return True


__all__ = ['CBKStructuralParser', 'CBKArticle', 'CBKChapter', 'CBKPart']
