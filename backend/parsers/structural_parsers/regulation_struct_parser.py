"""Regulation Structural Parser - Harvey/Legora CTO-Level
Parses Turkish regulations (Yönetmelik) with BÖLÜM → MADDE hierarchy

Production-grade implementation with:
- Complete regulation metadata extraction (authority, base law, Resmi Gazete)
- Chapter and article hierarchy (BÖLÜM → MADDE → FIKRA → BENT)
- Special article identification (Amaç, Kapsam, Tanımlar, Yürürlük, Yürütme)
- Annex extraction and parsing
- Cross-reference detection
- Amendment tracking
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


class RegulationStructuralParser(BaseStructuralParser):
    """
    Production-grade structural parser for Turkish Regulations (Yönetmelik).

    Turkish Regulation Structure:
    - Simpler than laws: No KISIM (Parts), mainly BÖLÜM (Chapters)
    - Standard articles: Amaç, Kapsam, Tanımlar, ..., Yürürlük, Yürütme
    - Issued by executive authority under a base law (Ana Kanun)

    Hierarchy:
    1. BÖLÜM (Chapter) - BİRİNCİ BÖLÜM, İKİNCİ BÖLÜM...
    2. MADDE (Article) - Madde 1, Madde 2, Madde 3...
    3. FIKRA (Paragraph) - (1), (2), (3)...
    4. BENT (Clause) - a), b), c)...
    5. ALT BENT (Subclause) - i), ii), iii)...

    Standard Article Pattern:
    - Madde 1: Amaç (Purpose)
    - Madde 2: Kapsam (Scope)
    - Madde 3: Tanımlar (Definitions)
    - ... substantive articles ...
    - Madde N-1: Yürürlük (Enforcement date)
    - Madde N: Yürütme (Execution authority)

    Features:
    - Issuing authority identification (Ministry, President, etc.)
    - Base law reference extraction
    - Official Gazette metadata
    - Special article identification and classification
    - Chapter-article association
    - Cross-reference extraction (to base law and other regulations)
    - Annex extraction (EK-1, EK-2...)
    - Amendment detection
    - Numbering validation
    """

    # Turkish ordinal numbers for BÖLÜM
    ORDINAL_NUMBERS = {
        'BİRİNCİ': 1, 'İKİNCİ': 2, 'ÜÇÜNCÜ': 3, 'DÖRDÜNCÜ': 4, 'BEŞİNCİ': 5,
        'ALTINCI': 6, 'YEDİNCİ': 7, 'SEKİZİNCİ': 8, 'DOKUZUNCU': 9, 'ONUNCU': 10,
        'ONBİRİNCİ': 11, 'ONİKİNCİ': 12, 'ONÜÇÜNCÜ': 13, 'ONDÖRDÜNCÜ': 14,
        'ONBEŞİNCİ': 15, 'ONALTINCI': 16, 'ONYEDİNCİ': 17, 'ONSEKİZİNCİ': 18,
        'ONDOKUZUNCU': 19, 'YİRMİNCİ': 20
    }

    # Issuing authorities
    AUTHORITIES = {
        'cumhurbaşkanı': 'Cumhurbaşkanlığı',
        'bakanlar kurulu': 'Bakanlar Kurulu',
        'maliye bakanlığı': 'Maliye Bakanlığı',
        'adalet bakanlığı': 'Adalet Bakanlığı',
        'içişleri bakanlığı': 'İçişleri Bakanlığı',
        'dışişleri bakanlığı': 'Dışişleri Bakanlığı',
        'milli eğitim bakanlığı': 'Milli Eğitim Bakanlığı',
        'sağlık bakanlığı': 'Sağlık Bakanlığı',
        'çalışma ve sosyal güvenlik bakanlığı': 'Çalışma ve Sosyal Güvenlik Bakanlığı',
        'ticaret bakanlığı': 'Ticaret Bakanlığı',
        'sanayi ve teknoloji bakanlığı': 'Sanayi ve Teknoloji Bakanlığı',
    }

    # Special article types
    SPECIAL_ARTICLE_TYPES = {
        'amac': 'Amaç',
        'kapsam': 'Kapsam',
        'tanimlar': 'Tanımlar',
        'yururluk': 'Yürürlük',
        'yurutme': 'Yürütme',
        'ilkeler': 'Temel İlkeler',
        'esaslar': 'Genel Esaslar'
    }

    def __init__(self):
        super().__init__("Regulation Structural Parser", "2.0.0")
        self.hierarchy_builder = ClauseHierarchyBuilder()
        logger.info(f"Initialized {self.name} v{self.version}")

    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract complete regulation structure.

        Returns:
            Dict with: 'authority', 'base_law', 'official_gazette', 'preamble',
                      'chapters', 'articles', 'special_articles', 'annexes',
                      'cross_references', 'amendments'
        """
        text = preprocessed.full_text
        title = preprocessed.title if hasattr(preprocessed, 'title') else ''

        try:
            # Extract metadata
            authority = self._extract_authority(title, text)
            base_law = self._extract_base_law(text)
            official_gazette = self._extract_official_gazette(text)
            enactment_date = self._extract_enactment_date(text)

            # Extract preamble
            preamble = self._extract_preamble(text)

            # Extract structural elements
            chapters = self._extract_chapters(text)
            articles = self._extract_articles(text)

            # Build clause hierarchies for articles
            for article in articles:
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

            # Identify special articles
            special_articles = self._identify_special_articles(articles)

            # Associate articles with chapters
            self._associate_articles_with_chapters(articles, chapters)

            # Validate article numbering
            numbering_issues = self._validate_article_numbering(articles)

            # Extract annexes
            annexes = self._extract_annexes(text)

            # Extract cross-references to base law
            cross_references = self._extract_all_cross_references(text)

            # Extract amendments
            amendments = self._extract_amendments(text)

            logger.info(
                f"Extracted regulation structure: {len(chapters)} chapters, "
                f"{len(articles)} articles, {len(special_articles)} special articles"
            )

            return {
                'authority': authority,
                'base_law': base_law,
                'official_gazette': official_gazette,
                'enactment_date': enactment_date,
                'preamble': preamble,
                'chapters': chapters,
                'articles': articles,
                'special_articles': special_articles,
                'annexes': annexes,
                'cross_references': cross_references,
                'amendments': amendments,
                'numbering_issues': numbering_issues,
                'document_type': 'yonetmelik'
            }

        except Exception as e:
            logger.error(f"Error extracting regulation structure: {e}")
            raise ParsingError(f"Failed to extract regulation structure: {e}")

    def _extract_authority(self, title: str, text: str) -> Optional[str]:
        """
        Extract issuing authority from title or preamble.

        Patterns:
        - "... Bakanlığı Yönetmeliği"
        - "Cumhurbaşkanı Kararı ile..."
        - "Bakanlar Kurulu Kararı ile..."
        """
        # Check title first
        text_to_check = (title + '\n' + text[:500]).lower()

        for keyword, authority in self.AUTHORITIES.items():
            if keyword in text_to_check:
                logger.debug(f"Identified authority: {authority}")
                return authority

        # Fallback patterns
        patterns = [
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Bakanlığı)',
            r'(Cumhurbaşkanlığı)',
            r'(Bakanlar\s+Kurulu)'
        ]

        for pattern in patterns:
            match = re.search(pattern, title + '\n' + text[:500])
            if match:
                logger.debug(f"Extracted authority: {match.group(1)}")
                return match.group(1)

        return None

    def _extract_base_law(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract base law reference (regulations are issued under a law).

        Pattern: "...5237 sayılı Türk Ceza Kanunu'nun 25. maddesine dayanarak..."
        """
        patterns = [
            r'(\d{3,5})\s+sayılı\s+([^\'\"]+?)(?:Kanun|kanun)(?:un|\'un|\'nun)\s+(\d+)\s*(?:inci|nci|üncü|ncü)?\s+madde(?:si|sine)',
            r'(\d{3,5})\s+sayılı\s+([^\'\"]+?)(?:Kanun|kanun)\s*(?:hükümlerine|uyarınca|gereğince)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1500], re.IGNORECASE)
            if match:
                base_law = {
                    'number': match.group(1),
                    'name': match.group(2).strip(),
                }
                if len(match.groups()) >= 3:
                    base_law['article'] = match.group(3)

                logger.debug(f"Extracted base law: {base_law}")
                return base_law

        return None

    def _extract_official_gazette(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract Official Gazette (Resmi Gazete) publication info.

        Returns:
            Dict with 'number' and 'date'
        """
        patterns = [
            r'Resmi\s+Gazete.*?Tarihi?\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4}).*?Sayı\s*:?\s*(\d+)',
            r'R\.?G\.?\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})\s*[/-]\s*(\d{4,6})',
            r'Resmi\s+Gazete\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4}).*?(\d{4,6})'
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

    def _extract_enactment_date(self, text: str) -> Optional[str]:
        """
        Extract enactment/publication date.

        Pattern: "...tarihi itibariyle yürürlüğe girer"
        """
        patterns = [
            r'Yayım(?:ı|ı)?\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihinde\s+yayım',
            r'yürürlük\s+tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1500], re.IGNORECASE)
            if match:
                logger.debug(f"Extracted enactment date: {match.group(1)}")
                return match.group(1)

        return None

    def _extract_preamble(self, text: str) -> Optional[str]:
        """
        Extract preamble (introduction text before first BÖLÜM or MADDE).

        Regulations often have a preamble explaining authority and purpose.
        """
        # Find first structural element
        structural_patterns = [
            r'(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ)\s+BÖLÜM',
            r'(?:MADDE|Madde)\s+1\s*(?:–|—|-)',
            r'^MADDE\s+1$'
        ]

        first_pos = len(text)
        for pattern in structural_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match and match.start() < first_pos:
                first_pos = match.start()

        if first_pos == len(text):
            return None

        preamble_text = text[:first_pos].strip()

        # Skip if too short
        if len(preamble_text) < 50:
            return None

        # Filter out metadata lines
        lines = preamble_text.split('\n')
        filtered_lines = []
        for line in lines:
            line_upper = line.upper()
            # Skip metadata
            if any(kw in line_upper for kw in ['RESMİ GAZETE', 'SAYISI', 'TARİHİ', 'YÖNETMELİK']):
                continue
            filtered_lines.append(line)

        preamble = '\n'.join(filtered_lines).strip()
        return preamble if len(preamble) > 30 else None

    def _extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract BÖLÜM (Chapters) from regulation text.

        Pattern: "BİRİNCİ BÖLÜM\nGenel Hükümler"
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
                    'type': 'BÖLÜM'
                })
                logger.debug(f"Extracted BÖLÜM {number}: {title}")

        # Sort by position
        chapters.sort(key=lambda x: x['start_pos'])

        return chapters

    def _extract_articles(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all articles (MADDE) from regulation.

        Returns list of articles with number, title, and full_text.
        """
        articles = []

        # Pattern: MADDE X – Title
        pattern = r'(?:MADDE|Madde)\s+(\d+)'

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

            # Extract full article text
            full_text = self._extract_article_text(text, start_pos)

            articles.append({
                'number': number,
                'title': title,
                'full_text': full_text,
                'start_pos': start_pos,
                'chapter_number': None  # Will be associated later
            })

            logger.debug(f"Extracted MADDE {number}: {title}")

        # Sort by number
        articles.sort(key=lambda x: x['number'])

        return articles

    def _extract_article_text(self, text: str, start_pos: int) -> str:
        """
        Extract full text of an article from start position to next article.
        """
        # Find next article or chapter
        next_patterns = [
            r'\n(?:MADDE|Madde)\s+\d+\s*(?:–|—|-)',
            r'\n(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ)\s+BÖLÜM'
        ]

        end_pos = len(text)
        for pattern in next_patterns:
            match = re.search(pattern, text[start_pos+10:], re.MULTILINE)
            if match and (start_pos + 10 + match.start()) < end_pos:
                end_pos = start_pos + 10 + match.start()

        article_text = text[start_pos:end_pos].strip()
        return article_text

    def _identify_special_articles(self, articles: List[Dict]) -> Dict[str, Dict]:
        """
        Identify special articles (Amaç, Kapsam, Tanımlar, Yürürlük, Yürütme).

        Returns dict with article type as key and article data as value.
        """
        special = {}

        for article in articles:
            title = article.get('title', '').lower()

            # Check for each special type
            if 'amaç' in title:
                special['purpose'] = article
                logger.debug(f"Identified purpose article: Madde {article['number']}")

            if 'kapsam' in title:
                special['scope'] = article
                logger.debug(f"Identified scope article: Madde {article['number']}")

            if 'tanım' in title:
                special['definitions'] = article
                logger.debug(f"Identified definitions article: Madde {article['number']}")

            if 'yürürlük' in title:
                special['enforcement'] = article
                logger.debug(f"Identified enforcement article: Madde {article['number']}")

            if 'yürütme' in title:
                special['execution'] = article
                logger.debug(f"Identified execution article: Madde {article['number']}")

            if 'ilke' in title:
                special['principles'] = article

            if 'esas' in title and 'genel' in title:
                special['general_provisions'] = article

        return special

    def _associate_articles_with_chapters(self, articles: List[Dict], chapters: List[Dict]):
        """
        Associate each article with its containing chapter based on position.
        """
        if not chapters:
            return

        for article in articles:
            article_pos = article['start_pos']

            # Find which chapter this article belongs to
            for i, chapter in enumerate(chapters):
                chapter_start = chapter['start_pos']
                # Chapter ends where next chapter starts (or at document end)
                chapter_end = chapters[i + 1]['start_pos'] if i + 1 < len(chapters) else float('inf')

                if chapter_start <= article_pos < chapter_end:
                    article['chapter_number'] = chapter['number']
                    article['chapter_title'] = chapter.get('title')
                    logger.debug(f"Associated MADDE {article['number']} with BÖLÜM {chapter['number']}")
                    break

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
                issues.append(f"Gap in numbering: Article {current} → {next_num}")
                logger.warning(issues[-1])

        # Check for duplicates
        numbers = [art['number'] for art in articles]
        duplicates = [num for num in set(numbers) if numbers.count(num) > 1]
        if duplicates:
            issues.append(f"Duplicate article numbers: {duplicates}")
            logger.warning(issues[-1])

        return issues

    def _extract_annexes(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract annexes (EK) from regulation.

        Pattern: "EK-1", "Ek 2", "EK: Formlar Listesi"
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

    def _extract_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract cross-references to other articles or laws.

        Patterns:
        - "5. maddede belirtilen"
        - "6698 sayılı Kanun"
        - "bu maddenin (1) numaralı fıkrası"
        """
        references = []

        patterns = [
            # Article references
            r'(\d+)\s*(?:inci|nci|üncü|ncü)?\s+maddede',
            r'(\d+)\s*(?:inci|nci|üncü|ncü)?\s+maddenin',
            # Law references
            r'(\d{3,5})\s+sayılı\s+(?:Kanun|kanun)',
            # Paragraph references
            r'\((\d+)\)\s*(?:numaralı)?\s+fıkra',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                references.append({
                    'text': match.group(0),
                    'reference': match.group(1),
                    'position': match.start()
                })

        return references

    def _extract_all_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract all cross-references in entire regulation"""
        return self._extract_cross_references(text)

    def _extract_amendments(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract amendment references.

        Pattern: "...değişik madde...", "...eklenen fıkra..."
        """
        amendments = []

        patterns = [
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarih.+?(\d{4,6})\s+sayılı.+?(değiş\w+|eklen\w+|mülga)',
            r'Bu\s+(?:Yönetmelik|madde).+?(değişik|ek|mülga)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amendments.append({
                    'matched_text': match.group(0),
                    'position': match.start()
                })

        logger.debug(f"Extracted {len(amendments)} amendment references")
        return amendments

    def _extract_tables_in_article(self, text: str) -> List[str]:
        """
        Extract tables within an article.

        Tables often use | or tab separators.
        """
        tables = []

        lines = text.split('\n')
        table_lines = []
        in_table = False

        for line in lines:
            if '|' in line or line.count('\t') >= 2:
                table_lines.append(line)
                in_table = True
            elif in_table and line.strip():
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


__all__ = ['RegulationStructuralParser']
