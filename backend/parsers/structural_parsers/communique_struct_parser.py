"""Communique (Tebliğ) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses Tebliğ (Communique) document structure

Production Features:
- Communique number and date extraction (RG publication)
- Authority and base law identification
- Article-based structure parsing
- Special provisions extraction
- Effective date handling
"""
from typing import Dict, List, Any, Optional
import re
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup

from ..core import StructuralParser, ParsedElement, DocumentStructure
from ..errors import ParsingError, ValidationError
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


@dataclass
class CommuniqueArticle:
    """Represents a communique article"""
    number: int
    title: Optional[str]
    content: str
    paragraphs: List[str]


class CommuniqueStructuralParser(StructuralParser):
    """Communique (Tebliğ) Structural Parser

    Parses official communique documents with:
    - Communique number and series (Sıra No)
    - Publication date (RG date and number)
    - Issuing authority
    - Base law/regulation reference
    - Articles with paragraphs
    - Effective date
    - Special provisions

    Features:
    - RG (Official Gazette) metadata extraction
    - Authority identification
    - Article-based structure parsing
    - Effective date extraction
    - Reference validation
    """

    # Communique number patterns
    COMMUNIQUE_NUMBER_PATTERNS = [
        r'(?:Tebliğ|Teblig)\s+(?:No|Sayı)\s*:?\s*(\d+)',
        r'(?:Sıra\s+No|Seri\s+No)\s*:?\s*(\d+)',
        r'(\d+)\s+[Ss]ayılı\s+[Tt]ebliğ'
    ]

    # RG (Resmi Gazete) patterns
    RG_PATTERNS = [
        r'(?:Resmi Gazete|RG)\s*:?\s*Tarih\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})\s*Sayı\s*:?\s*(\d+)',
        r'(?:Resmi Gazete|RG)\s+Sayı\s*:?\s*(\d+)',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarih(?:li)?\s+ve\s+(\d+)\s+sayılı\s+(?:Resmi Gazete|RG)'
    ]

    # Authority patterns
    AUTHORITY_PATTERNS = [
        r'([A-ZÇĞİÖŞÜ][^\n]{5,80}?(?:Bakanlığı|Başkanlığı|Kurulu|Kurumu))',
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+(?:Bakanlığı|Başkanlığı))'
    ]

    # Base law patterns
    BASE_LAW_PATTERNS = [
        r'(\d{4})\s+sayılı\s+([^\n]{10,100}?Kanun)',
        r'([^\n]{10,100}?Yönetmeliği)(?:nin|nın)\s+\d+',
        r'(\d{4})\s+sayılı\s+([^\n]{10,100}?Kanun)(?:un|ün)\s+\d+'
    ]

    # Article pattern
    ARTICLE_PATTERN = r'(?:MADDE|Madde)\s+(\d+)\s*[–-]\s*(.+?)(?=(?:MADDE|Madde)\s+\d+|Yürürlük|$)'

    def __init__(self):
        super().__init__("Communique Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse communique structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode)

        Returns:
            DocumentStructure with parsed communique

        Raises:
            ParsingError: If parsing fails
        """
        try:
            html_mode = kwargs.get('html_mode', False)

            if html_mode:
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
            else:
                text = content

            # Extract communique metadata
            communique_number = self._extract_communique_number(text)
            rg_info = self._extract_rg_info(text)
            authority = self._extract_authority(text)
            base_law = self._extract_base_law(text)

            # Parse articles
            articles = self._parse_articles(text)

            # Extract effective date
            effective_date = self._extract_effective_date(text)

            # Extract special provisions
            special_provisions = self._extract_special_provisions(text)

            logger.info(f"Successfully parsed communique: {communique_number}, "
                       f"{len(articles)} articles")

            return DocumentStructure(
                document_type='COMMUNIQUE',
                communique_number=communique_number,
                rg_info=rg_info,
                authority=authority,
                base_law=base_law,
                articles=articles,
                effective_date=effective_date,
                special_provisions=special_provisions,
                metadata={
                    'total_articles': len(articles),
                    'has_rg_info': rg_info is not None,
                    'has_base_law': base_law is not None
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse communique structure: {e}")
            raise ParsingError(f"Communique parsing failed: {str(e)}") from e

    def _extract_communique_number(self, text: str) -> Optional[str]:
        """Extract communique number"""
        for pattern in self.COMMUNIQUE_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                logger.debug(f"Extracted communique number: {number}")
                return number

        return None

    def _extract_rg_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract Resmi Gazete (Official Gazette) information"""
        for pattern in self.RG_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    rg_info = {
                        'date': groups[0] if '/' in groups[0] or '.' in groups[0] else None,
                        'number': groups[1] if groups[1].isdigit() else groups[0]
                    }
                elif len(groups) == 1:
                    rg_info = {'number': groups[0]}
                else:
                    continue

                logger.debug(f"Extracted RG info: {rg_info}")
                return rg_info

        return None

    def _extract_authority(self, text: str) -> Optional[str]:
        """Extract issuing authority"""
        for pattern in self.AUTHORITY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                authority = match.group(1).strip()
                logger.debug(f"Extracted authority: {authority}")
                return authority

        return None

    def _extract_base_law(self, text: str) -> Optional[str]:
        """Extract base law/regulation reference"""
        for pattern in self.BASE_LAW_PATTERNS:
            match = re.search(pattern, text)
            if match:
                base_law = match.group(0).strip()
                logger.debug(f"Extracted base law: {base_law[:50]}...")
                return base_law

        return None

    def _parse_articles(self, text: str) -> List[CommuniqueArticle]:
        """Parse articles from text"""
        articles = []

        # Find all article matches
        matches = list(re.finditer(self.ARTICLE_PATTERN, text, re.DOTALL | re.IGNORECASE))

        for i, match in enumerate(matches):
            article_number = int(match.group(1))
            article_title = match.group(2).strip() if match.group(2) else None

            # Extract article content
            article_start = match.end()
            article_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[article_start:article_end].strip()

            # Parse paragraphs
            paragraphs = self._parse_paragraphs(article_text)

            articles.append(CommuniqueArticle(
                number=article_number,
                title=article_title,
                content=article_text,
                paragraphs=paragraphs
            ))

            logger.debug(f"Parsed Article {article_number}: {len(paragraphs)} paragraphs")

        return articles

    def _parse_paragraphs(self, text: str) -> List[str]:
        """Parse numbered paragraphs from article text"""
        paragraphs = []

        # Pattern: (1), (2), etc.
        paragraph_pattern = r'\((\d+)\)\s*([^(]+?)(?=\(\d+\)|$)'
        matches = re.finditer(paragraph_pattern, text, re.DOTALL)

        for match in matches:
            para_content = match.group(2).strip()
            paragraphs.append(para_content)

        return paragraphs

    def _extract_effective_date(self, text: str) -> Optional[str]:
        """Extract effective date (Yürürlük Tarihi)"""
        patterns = [
            r'Yürürlük\s*:?\s*([^\n]+)',
            r'yürürlüğe\s+girer.*?(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'yayım(?:ı|ı)(?:nı|nı)?\s+takip\s+eden'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                effective_text = match.group(1) if len(match.groups()) > 0 else match.group(0)
                logger.debug(f"Extracted effective date: {effective_text}")
                return effective_text.strip()

        return None

    def _extract_special_provisions(self, text: str) -> List[str]:
        """Extract special provisions (transitional, final provisions)"""
        provisions = []

        # Common special provision markers
        markers = [
            r'Geçici\s+Madde\s+\d+',
            r'Ek\s+Madde\s+\d+',
            r'Değiştirilen\s+Madde'
        ]

        for marker in markers:
            matches = re.finditer(marker, text, re.IGNORECASE)
            for match in matches:
                # Extract some context
                start = match.start()
                end = min(start + 200, len(text))
                provision = text[start:end].strip()
                provisions.append(provision)

        logger.debug(f"Extracted {len(provisions)} special provisions")
        return provisions

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed communique structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not structure.articles or len(structure.articles) == 0:
            raise ValidationError("Communique must contain at least one article")

        if structure.communique_number is None:
            logger.warning("Communique number is missing")

        if structure.authority is None:
            logger.warning("Issuing authority is missing")

        logger.info("Communique structure validation passed")
        return True


__all__ = ['CommuniqueStructuralParser', 'CommuniqueArticle']
