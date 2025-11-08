"""Law Structural Parser - Harvey/Legora CTO-Level
Parses Turkish laws (Kanun) with full hierarchy: KISIM → BÖLÜM → MADDE → FIKRA → BENT → ALT BENT

Production-grade implementation with:
- Complete hierarchy extraction (KISIM/BÖLÜM/MADDE/FIKRA/BENT/ALT BENT)
- Temporary and additional articles (GEÇİCİ MADDE, EK MADDE)
- Amendment detection and tracking
- Cross-reference extraction
- Table and annex handling
- Official Gazette metadata
- Numbering validation
- Comprehensive error handling
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import logging

from .base_structural_parser import BaseStructuralParser
from .clause_hierarchy_builder import ClauseHierarchyBuilder
from ..core import LegalDocument
from ..core.exceptions import ParsingError, ValidationError
from ..utils.text_utils import normalize_turkish_text
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


class LawStructuralParser(BaseStructuralParser):
    """
    Production-grade structural parser for Turkish Laws (Kanun).

    Turkish Legal Hierarchy:
    1. KISIM (Part) - BİRİNCİ KISIM, İKİNCİ KISIM, ÜÇÜNCÜ KISIM...
    2. BÖLÜM (Chapter) - BİRİNCİ BÖLÜM, İKİNCİ BÖLÜM, ÜÇÜNCÜ BÖLÜM...
    3. MADDE (Article) - Madde 1, Madde 2, Madde 3...
    4. FIKRA (Paragraph) - (1), (2), (3)...
    5. BENT (Clause) - a), b), c)...
    6. ALT BENT (Subclause) - i), ii), iii)...

    Special Article Types:
    - GEÇİCİ MADDE (Temporary Article): Time-limited provisions
    - EK MADDE (Additional Article): Added by amendments
    - MÜLGA MADDE (Repealed Article): Explicitly repealed

    Features:
    - Full hierarchy extraction with parent-child relationships
    - Amendment detection and version tracking
    - Cross-reference extraction (e.g., "5. maddede belirtilen...")
    - Table extraction within articles
    - Annex detection and parsing
    - Official Gazette metadata extraction
    - Numbering validation and gap detection
    - Turkish ordinal number parsing (BİRİNCİ, İKİNCİ, ÜÇÜNCÜ...)
    """

    # Turkish ordinal numbers for KISIM and BÖLÜM
    ORDINAL_NUMBERS = {
        'BİRİNCİ': 1, 'İKİNCİ': 2, 'ÜÇÜNCÜ': 3, 'DÖRDÜNCÜ': 4, 'BEŞİNCİ': 5,
        'ALTINCI': 6, 'YEDİNCİ': 7, 'SEKİZİNCİ': 8, 'DOKUZUNCU': 9, 'ONUNCU': 10,
        'ONBİRİNCİ': 11, 'ONİKİNCİ': 12, 'ONÜÇÜNCÜ': 13, 'ONDÖRDÜNCÜ': 14,
        'ONBEŞİNCİ': 15, 'ONALTINCI': 16, 'ONYEDİNCİ': 17, 'ONSEKİZİNCİ': 18,
        'ONDOKUZUNCU': 19, 'YİRMİNCİ': 20
    }

    # Article type patterns
    ARTICLE_TYPES = {
        'regular': r'(?:MADDE|Madde)\s+(\d+)',
        'temporary': r'(?:GEÇİCİ|Geçici)\s+(?:MADDE|Madde)\s+(\d+)',
        'additional': r'(?:EK|Ek)\s+(?:MADDE|Madde)\s+(\d+)',
        'repealed': r'(?:MÜLGA|Mülga)\s+(?:MADDE|Madde)\s+(\d+)'
    }

    def __init__(self):
        super().__init__("Law Structural Parser", "2.0.0")
        self.hierarchy_builder = ClauseHierarchyBuilder()
        logger.info(f"Initialized {self.name} v{self.version}")

    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract complete law structure including all hierarchies.

        Returns:
            Dict with: 'law_number', 'enactment_date', 'official_gazette', 'preamble',
                      'parts', 'chapters', 'articles', 'temporary_articles',
                      'additional_articles', 'repealed_articles', 'annexes',
                      'cross_references', 'amendments', 'tables'
        """
        text = preprocessed.full_text
        title = preprocessed.title if hasattr(preprocessed, 'title') else ''

        try:
            # Extract metadata
            law_number = self._extract_law_number(title, text)
            enactment_date = self._extract_enactment_date(text)
            official_gazette = self._extract_official_gazette(text)

            # Extract preamble
            preamble = self._extract_preamble(text)

            # Extract structural hierarchy
            parts = self._extract_parts(text)
            chapters = self._extract_chapters(text)
            articles = self._extract_articles(text)

            # Classify articles by type
            regular_articles = [art for art in articles if art.get('type') == 'regular']
            temporary_articles = [art for art in articles if art.get('type') == 'temporary']
            additional_articles = [art for art in articles if art.get('type') == 'additional']
            repealed_articles = [art for art in articles if art.get('type') == 'repealed']

            # Build clause hierarchies for regular articles
            for article in regular_articles:
                try:
                    hierarchy = self.hierarchy_builder.build_hierarchy(
                        article.get('full_text', ''),
                        article.get('number')
                    )
                    article['hierarchy'] = hierarchy
                    article['clauses'] = self.hierarchy_builder.to_legal_clauses(hierarchy)

                    # Extract cross-references within article
                    article['cross_references'] = self._extract_cross_references(
                        article.get('full_text', '')
                    )

                    # Extract tables within article
                    article['tables'] = self._extract_tables_in_article(
                        article.get('full_text', '')
                    )

                except Exception as e:
                    logger.warning(f"Error building hierarchy for article {article.get('number')}: {e}")
                    article['hierarchy'] = None
                    article['clauses'] = []

            # Associate chapters with parts
            self._associate_chapters_with_parts(parts, chapters)

            # Associate articles with chapters/parts
            self._associate_articles_with_structure(articles, chapters, parts)

            # Validate article numbering
            numbering_issues = self._validate_article_numbering(regular_articles)

            # Extract annexes
            annexes = self._extract_annexes(text)

            # Extract amendments mentioned in text
            amendments = self._extract_amendments(text)

            # Extract all cross-references
            cross_references = self._extract_all_cross_references(text)

            logger.info(
                f"Extracted law structure: {len(parts)} parts, {len(chapters)} chapters, "
                f"{len(regular_articles)} articles, {len(temporary_articles)} temporary, "
                f"{len(additional_articles)} additional"
            )

            return {
                'law_number': law_number,
                'enactment_date': enactment_date,
                'official_gazette': official_gazette,
                'preamble': preamble,
                'parts': parts,
                'chapters': chapters,
                'articles': regular_articles,
                'temporary_articles': temporary_articles,
                'additional_articles': additional_articles,
                'repealed_articles': repealed_articles,
                'annexes': annexes,
                'cross_references': cross_references,
                'amendments': amendments,
                'numbering_issues': numbering_issues,
                'document_type': 'kanun'
            }

        except Exception as e:
            logger.error(f"Error extracting law structure: {e}")
            raise ParsingError(f"Failed to extract law structure: {e}")

    def _extract_law_number(self, title: str, text: str) -> Optional[str]:
        """
        Extract law number from title or text.

        Patterns:
        - "5237 Sayılı Türk Ceza Kanunu" → "5237"
        - "6698 sayılı Kişisel Verilerin Korunması Kanunu" → "6698"
        """
        # Try title first
        patterns = [
            r'(\d{3,5})\s+[Ss]ayılı',
            r'Kanun\s+(?:No|Numarası)\s*:?\s*(\d{3,5})',
            r'(\d{3,5})\s+[Nn]o\'?lu'
        ]

        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                logger.debug(f"Extracted law number from title: {match.group(1)}")
                return match.group(1)

        # Try text if not found in title
        for pattern in patterns:
            match = re.search(pattern, text[:500])  # Check first 500 chars
            if match:
                logger.debug(f"Extracted law number from text: {match.group(1)}")
                return match.group(1)

        return None

    def _extract_enactment_date(self, text: str) -> Optional[str]:
        """
        Extract enactment date (Kabul Tarihi).

        Patterns:
        - "Kabul Tarihi: 26/9/2004"
        - "26.9.2004 tarihinde kabul edilmiştir"
        """
        patterns = [
            r'Kabul\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihinde\s+kabul',
            r'Kanunun\s+kabul\s+tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                logger.debug(f"Extracted enactment date: {match.group(1)}")
                return match.group(1)

        return None

    def _extract_official_gazette(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract Official Gazette (Resmi Gazete) publication info.

        Returns:
            Dict with 'number' and 'date'
        """
        # Pattern: "Resmi Gazete Tarihi: 12.10.2004  Sayı: 25611"
        patterns = [
            r'Resmi\s+Gazete.*?Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4}).*?Sayı\s*:?\s*(\d+)',
            r'Resmi\s+Gazete\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4}).*?(\d{4,6})',
            r'R\.?G\.?\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})\s*-\s*(\d{4,6})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE | re.DOTALL)
            if match:
                gazette_info = {
                    'date': match.group(1),
                    'number': match.group(2)
                }
                logger.debug(f"Extracted Official Gazette: {gazette_info}")
                return gazette_info

        return None

    def _extract_preamble(self, text: str) -> Optional[str]:
        """
        Extract preamble (introduction text before first structural element).

        The preamble is text between title/metadata and the first KISIM/BÖLÜM/MADDE.
        """
        # Find first structural element
        structural_patterns = [
            r'(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ)\s+KISIM',
            r'(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ)\s+BÖLÜM',
            r'(?:MADDE|Madde)\s+1\s*(?:–|—|-)',
            r'^MADDE\s+1$'
        ]

        first_pos = len(text)  # Default to end of text
        for pattern in structural_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match and match.start() < first_pos:
                first_pos = match.start()

        # If no structural element found, no preamble
        if first_pos == len(text):
            return None

        # Extract text before first structural element
        preamble_text = text[:first_pos].strip()

        # Skip if too short (likely just title/metadata)
        if len(preamble_text) < 50:
            return None

        # Skip common metadata lines
        lines = preamble_text.split('\n')
        filtered_lines = []
        for line in lines:
            line_upper = line.upper()
            # Skip metadata lines
            if any(kw in line_upper for kw in ['SAYILI', 'RESMİ GAZETE', 'KABUL TARİHİ', 'YAYIM TARİHİ']):
                continue
            filtered_lines.append(line)

        preamble = '\n'.join(filtered_lines).strip()
        return preamble if len(preamble) > 30 else None

    def _extract_parts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract KISIM (Parts) from law text.

        Pattern: "BİRİNCİ KISIM\nGenel Hükümler"
        """
        parts = []

        # Pattern: Ordinal + KISIM + optional title on next line
        pattern = r'(' + '|'.join(self.ORDINAL_NUMBERS.keys()) + r')\s+KISIM\s*\n?([^\n]*)'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            ordinal = match.group(1).upper()
            title = match.group(2).strip() if match.group(2) else None

            number = self.ORDINAL_NUMBERS.get(ordinal)
            if number:
                parts.append({
                    'number': number,
                    'ordinal': ordinal,
                    'title': title,
                    'start_pos': match.start(),
                    'type': 'KISIM'
                })
                logger.debug(f"Extracted KISIM {number}: {title}")

        # Sort by position
        parts.sort(key=lambda x: x['start_pos'])

        return parts

    def _extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract BÖLÜM (Chapters) from law text.

        Pattern: "BİRİNCİ BÖLÜM\nAmaç, Kapsam ve Tanımlar"
        """
        chapters = []

        # Pattern: Ordinal + BÖLÜM + optional title
        pattern = r'(' + '|'.join(self.ORDINAL_NUMBERS.keys()) + r')\s+BÖLÜM\s*\n?([^\n]*)'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            ordinal = match.group(1).upper()
            title = match.group(2).strip() if match.group(2) else None

            number = self.ORDINAL_NUMBERS.get(ordinal)
            if number:
                chapters.append({
                    'number': number,
                    'ordinal': ordinal,
                    'title': title,
                    'start_pos': match.start(),
                    'type': 'BÖLÜM',
                    'part_number': None  # Will be associated later
                })
                logger.debug(f"Extracted BÖLÜM {number}: {title}")

        # Sort by position
        chapters.sort(key=lambda x: x['start_pos'])

        return chapters

    def _extract_articles(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all articles (MADDE, GEÇİCİ MADDE, EK MADDE, MÜLGA MADDE).

        Returns list of articles with type, number, title, and full_text.
        """
        articles = []

        # Extract each article type
        for art_type, pattern in self.ARTICLE_TYPES.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number = int(match.group(1))
                start_pos = match.start()

                # Find article title (text after "MADDE X –")
                title_match = re.search(
                    rf'{re.escape(match.group(0))}\s*(?:–|—|-)\s*([^\n]+)',
                    text[start_pos:start_pos+200],
                    re.IGNORECASE
                )
                title = title_match.group(1).strip() if title_match else None

                # Extract full article text (until next article or structural element)
                full_text = self._extract_article_text(text, start_pos)

                articles.append({
                    'type': art_type,
                    'number': number,
                    'title': title,
                    'full_text': full_text,
                    'start_pos': start_pos,
                    'chapter_number': None,  # Will be associated later
                    'part_number': None  # Will be associated later
                })

                logger.debug(f"Extracted {art_type} MADDE {number}: {title}")

        # Sort by position (maintains document order)
        articles.sort(key=lambda x: x['start_pos'])

        return articles

    def _extract_article_text(self, text: str, start_pos: int) -> str:
        """
        Extract full text of an article from start position to next article/structural element.
        """
        # Find next article or structural element
        next_patterns = [
            r'\n(?:MADDE|Madde)\s+\d+\s*(?:–|—|-)',
            r'\n(?:GEÇİCİ|Geçici)\s+(?:MADDE|Madde)',
            r'\n(?:EK|Ek)\s+(?:MADDE|Madde)',
            r'\n(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ)\s+(?:KISIM|BÖLÜM)'
        ]

        end_pos = len(text)  # Default to end of text
        for pattern in next_patterns:
            match = re.search(pattern, text[start_pos+10:], re.MULTILINE)
            if match and (start_pos + 10 + match.start()) < end_pos:
                end_pos = start_pos + 10 + match.start()

        article_text = text[start_pos:end_pos].strip()
        return article_text

    def _associate_chapters_with_parts(self, parts: List[Dict], chapters: List[Dict]):
        """
        Associate each BÖLÜM with its parent KISIM based on position.
        """
        if not parts:
            return

        for chapter in chapters:
            # Find the last KISIM that appears before this BÖLÜM
            chapter_pos = chapter['start_pos']
            parent_part = None

            for part in reversed(parts):
                if part['start_pos'] < chapter_pos:
                    parent_part = part
                    break

            if parent_part:
                chapter['part_number'] = parent_part['number']
                logger.debug(f"Associated BÖLÜM {chapter['number']} with KISIM {parent_part['number']}")

    def _associate_articles_with_structure(
        self,
        articles: List[Dict],
        chapters: List[Dict],
        parts: List[Dict]
    ):
        """
        Associate each article with its parent BÖLÜM and KISIM based on position.
        """
        for article in articles:
            article_pos = article['start_pos']

            # Find parent chapter
            parent_chapter = None
            for chapter in reversed(chapters):
                if chapter['start_pos'] < article_pos:
                    parent_chapter = chapter
                    break

            if parent_chapter:
                article['chapter_number'] = parent_chapter['number']
                article['part_number'] = parent_chapter.get('part_number')
            else:
                # No chapter, try to find parent part directly
                parent_part = None
                for part in reversed(parts):
                    if part['start_pos'] < article_pos:
                        parent_part = part
                        break

                if parent_part:
                    article['part_number'] = parent_part['number']

    def _validate_article_numbering(self, articles: List[Dict]) -> List[str]:
        """
        Validate that article numbers are sequential and identify gaps.

        Returns list of issues found.
        """
        issues = []

        if not articles:
            return issues

        # Sort by number
        sorted_articles = sorted(articles, key=lambda x: x['number'])

        # Check for gaps
        for i in range(len(sorted_articles) - 1):
            current = sorted_articles[i]['number']
            next_num = sorted_articles[i + 1]['number']

            if next_num - current > 1:
                issues.append(f"Gap in numbering: Article {current} → {next_num} (missing {next_num - current - 1} articles)")
                logger.warning(issues[-1])

        # Check for duplicates
        numbers = [art['number'] for art in articles]
        duplicates = [num for num in set(numbers) if numbers.count(num) > 1]
        if duplicates:
            issues.append(f"Duplicate article numbers found: {duplicates}")
            logger.warning(issues[-1])

        return issues

    def _extract_annexes(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract annexes (EK) mentioned in the law.

        Pattern: "EK-1", "Ek 2", "EK: Liste"
        """
        annexes = []

        patterns = [
            r'EK[-\s](\d+|[IVXLCDM]+)\s*:?\s*([^\n]+)',
            r'Ek\s+(\d+)\s*:?\s*([^\n]+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annex_id = match.group(1)
                title = match.group(2).strip()

                annexes.append({
                    'id': annex_id,
                    'title': title,
                    'start_pos': match.start()
                })
                logger.debug(f"Extracted annex: EK {annex_id} - {title}")

        return annexes

    def _extract_amendments(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mentions of amendments to this law.

        Pattern: "...5237 sayılı Kanunun 15 inci maddesinin değişik (1) numaralı fıkrası..."
        """
        amendments = []

        patterns = [
            r'(\d{3,5})\s+sayılı\s+Kanun(?:un|\'un)?\s+(\d+)\s*(?:inci|nci|üncü|ncü)?\s+maddesi(?:nin|nde|ne)?\s+(değiş\w+|eklen\w+|mülga|yürürlük)',
            r'Bu\s+(?:Kanun|madde).+?(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarih.+?(\d{3,5})\s+sayılı\s+Kanun'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amendments.append({
                    'matched_text': match.group(0),
                    'position': match.start()
                })

        logger.debug(f"Extracted {len(amendments)} amendment references")
        return amendments

    def _extract_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract cross-references to other articles within this article.

        Patterns:
        - "5. maddede belirtilen"
        - "birinci fıkrasında"
        - "a bendinde"
        """
        references = []

        patterns = [
            r'(\d+)\s*(?:inci|nci|üncü|ncü)?\s+maddede',
            r'(\d+)\s*(?:inci|nci|üncü|ncü)?\s+maddenin',
            r'bu\s+maddenin\s+(\d+)\s*(?:inci|nci|üncü|ncü)?\s+fıkrası',
            r'([a-z])\s+bendinde',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                references.append({
                    'text': match.group(0),
                    'reference': match.group(1) if match.group(1) else None,
                    'position': match.start()
                })

        return references

    def _extract_all_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract all cross-references in the entire law"""
        return self._extract_cross_references(text)

    def _extract_tables_in_article(self, text: str) -> List[str]:
        """
        Extract tables within an article (if any).

        Tables are often indicated by structured data or specific keywords.
        """
        tables = []

        # Simple heuristic: look for multiple lines with | or tab separators
        lines = text.split('\n')
        table_lines = []
        in_table = False

        for line in lines:
            if '|' in line or line.count('\t') >= 2:
                table_lines.append(line)
                in_table = True
            elif in_table and line.strip():
                # Continue if looks like continuation
                table_lines.append(line)
            elif in_table:
                # End of table
                if len(table_lines) >= 2:
                    tables.append('\n'.join(table_lines))
                table_lines = []
                in_table = False

        # Catch table at end
        if len(table_lines) >= 2:
            tables.append('\n'.join(table_lines))

        return tables


__all__ = ['LawStructuralParser']
