"""RG Issue (Resmi Gazete Sayısı) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses Turkish Official Gazette (Resmi Gazete) issue structure

Production Features:
- Issue number and date extraction
- Issue type classification (Normal, Mükerrer)
- Document listing and indexing
- Page number tracking
- Document type classification
- Table of contents parsing
- Section organization
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from bs4 import BeautifulSoup

from ..core import StructuralParser, ParsedElement, DocumentStructure
from ..errors import ParsingError, ValidationError
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


@dataclass
class RGDocument:
    """Represents a document published in RG issue"""
    doc_number: Optional[int]
    doc_type: str  # KANUN, YÖNETMELİK, TEBLİĞ, KARAR, etc.
    title: str
    issuing_authority: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    reference_number: Optional[str]  # e.g., "6698 sayılı Kanun"


@dataclass
class RGSection:
    """Represents a section within RG issue"""
    section_name: str  # YÜRÜTME VE İDARE, YARGI, İLAN, etc.
    documents: List[RGDocument]


@dataclass
class RGIssue:
    """Represents complete RG issue structure"""
    issue_number: int
    issue_date: str
    issue_type: str  # NORMAL, MÜKERRER, MÜKERRER_2, etc.
    sections: List[RGSection]
    total_pages: Optional[int]
    total_documents: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RGIssueStructuralParser(StructuralParser):
    """RG Issue (Resmi Gazete Sayısı) Structural Parser

    Parses Official Gazette issues with:
    - Issue number extraction (5-digit format)
    - Publication date parsing
    - Issue type classification (Normal, Mükerrer variants)
    - Document listing and indexing
    - Page number tracking
    - Document type classification (8 types)
    - Section organization (6 sections)
    - Table of contents extraction

    Features:
    - Multiple issue number formats
    - Turkish date parsing
    - Document type keyword matching
    - Automatic page range detection
    - Section-based organization
    - Reference number extraction
    """

    # Issue number patterns
    ISSUE_NUMBER_PATTERNS = [
        r'Resmi\s+Gazete\s+Sayı\s*:?\s*(\d{4,5})',
        r'RG\s+Sayı\s*:?\s*(\d{4,5})',
        r'Sayı\s*:?\s*(\d{4,5})',
        r'(\d{4,5})\s+sayılı\s+(?:Resmi Gazete|RG)'
    ]

    # Issue date patterns
    ISSUE_DATE_PATTERNS = [
        r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})',
        r'Tarih\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})',
    ]

    # Issue type patterns
    ISSUE_TYPE_PATTERNS = [
        (r'Mükerrer\s*[-–]\s*2', 'MÜKERRER_2'),
        (r'Mükerrer\s*[-–]\s*3', 'MÜKERRER_3'),
        (r'Mükerrer', 'MÜKERRER'),
        (r'Normal', 'NORMAL')
    ]

    # Document type classification
    DOCUMENT_TYPES = {
        'KANUN': ['kanun', 'kanunu'],
        'KHK': ['khk', 'kanun hükmünde kararname'],
        'CBK': ['cumhurbaşkanlığı kararnamesi', 'cbk'],
        'YÖNETMELİK': ['yönetmelik', 'yönetmeliği'],
        'TEBLİĞ': ['tebliğ', 'tebliği'],
        'KARAR': ['karar', 'kararı', 'kurul kararı'],
        'GENELGE': ['genelge', 'genelgesi'],
        'İLAN': ['ilan', 'ilanı', 'duyuru']
    }

    # RG Section names
    RG_SECTIONS = [
        'YÜRÜTME VE İDARE BÖLÜMÜ',
        'DÜZENLEYICI VE DENETLEYİCİ KURUMLAR',
        'YARGI BÖLÜMÜ',
        'İLAN BÖLÜMÜ',
        'KOMİSYON RAPORLARI',
        'DİĞER İLANLAR'
    ]

    # Page number patterns
    PAGE_PATTERNS = [
        r'Sayfa\s*:?\s*(\d+)',
        r'(?:s|S)\.\s*(\d+)',
        r'\(Sayfa:\s*(\d+)\)',
        r'(\d+)-(\d+)'  # Page range
    ]

    # Reference number patterns
    REFERENCE_PATTERNS = [
        r'(\d{4})\s+sayılı\s+([A-ZÇĞİÖŞÜ][^\n]{5,80})',
        r'Karar\s+Sayısı\s*:?\s*([\d/\-]+)',
        r'Tebliğ\s+No\s*:?\s*([\d/\-]+)'
    ]

    def __init__(self):
        super().__init__("RG Issue Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse RG issue structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode)

        Returns:
            DocumentStructure with parsed RG issue

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

            # Extract issue metadata
            issue_number = self._extract_issue_number(text)
            issue_date = self._extract_issue_date(text)
            issue_type = self._classify_issue_type(text)

            # Parse sections and documents
            sections = self._parse_sections(text)

            # Calculate total pages and documents
            total_pages = self._extract_total_pages(text)
            total_documents = sum(len(section.documents) for section in sections)

            # Create RG issue object
            rg_issue = RGIssue(
                issue_number=issue_number or 0,
                issue_date=issue_date or 'Unknown',
                issue_type=issue_type,
                sections=sections,
                total_pages=total_pages,
                total_documents=total_documents,
                metadata={
                    'section_count': len(sections),
                    'document_count': total_documents,
                    'has_page_numbers': total_pages is not None
                }
            )

            logger.info(f"Successfully parsed RG issue: {issue_number}, "
                       f"Date: {issue_date}, Type: {issue_type}, "
                       f"Documents: {total_documents}")

            return DocumentStructure(
                document_type='RG_ISSUE',
                rg_issue=rg_issue,
                metadata=rg_issue.metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse RG issue structure: {e}")
            raise ParsingError(f"RG issue parsing failed: {str(e)}") from e

    def _extract_issue_number(self, text: str) -> Optional[int]:
        """Extract RG issue number"""
        for pattern in self.ISSUE_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    number = int(match.group(1))
                    logger.debug(f"Extracted issue number: {number}")
                    return number
                except (ValueError, IndexError):
                    continue

        logger.warning("RG issue number not found")
        return None

    def _extract_issue_date(self, text: str) -> Optional[str]:
        """Extract RG issue date"""
        for pattern in self.ISSUE_DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(0)
                logger.debug(f"Extracted issue date: {date_str}")
                return date_str

        logger.warning("RG issue date not found")
        return None

    def _classify_issue_type(self, text: str) -> str:
        """Classify RG issue type"""
        for pattern, issue_type in self.ISSUE_TYPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug(f"Classified issue type: {issue_type}")
                return issue_type

        logger.debug("Classified issue type: NORMAL (default)")
        return 'NORMAL'

    def _extract_total_pages(self, text: str) -> Optional[int]:
        """Extract total page count"""
        # Look for highest page number
        max_page = 0
        for pattern in self.PAGE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) == 2:  # Page range
                        page_end = int(match.group(2))
                        max_page = max(max_page, page_end)
                    else:
                        page_num = int(match.group(1))
                        max_page = max(max_page, page_num)
                except (ValueError, IndexError):
                    continue

        if max_page > 0:
            logger.debug(f"Extracted total pages: {max_page}")
            return max_page

        return None

    def _parse_sections(self, text: str) -> List[RGSection]:
        """Parse RG sections and their documents"""
        sections = []

        # Try to split by section headers
        section_splits = self._split_by_sections(text)

        if section_splits:
            for section_name, section_text in section_splits:
                documents = self._parse_documents(section_text)
                if documents:
                    sections.append(RGSection(
                        section_name=section_name,
                        documents=documents
                    ))
                    logger.debug(f"Parsed section '{section_name}': {len(documents)} documents")
        else:
            # No clear sections, parse all documents into a general section
            documents = self._parse_documents(text)
            if documents:
                sections.append(RGSection(
                    section_name='GENEL',
                    documents=documents
                ))

        return sections

    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text by RG section headers"""
        splits = []

        # Find section headers
        section_positions = []
        for section_name in self.RG_SECTIONS:
            match = re.search(re.escape(section_name), text, re.IGNORECASE)
            if match:
                section_positions.append((match.start(), section_name))

        if not section_positions:
            return []

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # Extract text for each section
        for i, (pos, name) in enumerate(section_positions):
            start = pos
            end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
            section_text = text[start:end]
            splits.append((name, section_text))

        return splits

    def _parse_documents(self, text: str) -> List[RGDocument]:
        """Parse documents from section text"""
        documents = []
        lines = text.split('\n')

        current_doc = None
        doc_number = 1

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 10:
                continue

            # Check if this line looks like a document title
            doc_type = self._classify_document_type(line_stripped)

            if doc_type != 'UNKNOWN':
                # This looks like a document entry
                title = line_stripped
                authority = self._extract_authority(line_stripped)
                reference = self._extract_reference(line_stripped)
                page_info = self._extract_page_info(line_stripped)

                doc = RGDocument(
                    doc_number=doc_number,
                    doc_type=doc_type,
                    title=title[:200],  # Limit title length
                    issuing_authority=authority,
                    page_start=page_info[0] if page_info else None,
                    page_end=page_info[1] if page_info and len(page_info) > 1 else None,
                    reference_number=reference
                )

                documents.append(doc)
                doc_number += 1
                logger.debug(f"Parsed document: {doc_type} - {title[:50]}...")

        return documents

    def _classify_document_type(self, text: str) -> str:
        """Classify document type from text"""
        text_lower = text.lower()

        for doc_type, keywords in self.DOCUMENT_TYPES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return doc_type

        return 'UNKNOWN'

    def _extract_authority(self, text: str) -> Optional[str]:
        """Extract issuing authority from document title"""
        # Common authority patterns
        authority_patterns = [
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+(?:Bakanlığı|Başkanlığı|Kurulu|Kurumu))',
            r'(T\.C\.\s+[A-ZÇĞİÖŞÜ][^\n]+?(?:Bakanlığı|Başkanlığı))'
        ]

        for pattern in authority_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _extract_reference(self, text: str) -> Optional[str]:
        """Extract reference number from document title"""
        for pattern in self.REFERENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return None

    def _extract_page_info(self, text: str) -> Optional[Tuple[int, ...]]:
        """Extract page number or range from text"""
        for pattern in self.PAGE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2:  # Page range
                        page_start = int(match.group(1))
                        page_end = int(match.group(2))
                        return (page_start, page_end)
                    else:
                        page_num = int(match.group(1))
                        return (page_num,)
                except (ValueError, IndexError):
                    continue

        return None

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed RG issue structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not hasattr(structure, 'rg_issue') or structure.rg_issue is None:
            raise ValidationError("RG issue structure is missing")

        rg_issue = structure.rg_issue

        if rg_issue.issue_number == 0:
            logger.warning("RG issue number is missing or invalid")

        if rg_issue.issue_date == 'Unknown':
            logger.warning("RG issue date is missing")

        if not rg_issue.sections or len(rg_issue.sections) == 0:
            logger.warning("RG issue has no sections")

        if rg_issue.total_documents == 0:
            logger.warning("RG issue has no documents")

        # Validate sections
        for section in rg_issue.sections:
            if not section.section_name:
                logger.warning("Section missing name")

            if not section.documents:
                logger.warning(f"Section '{section.section_name}' has no documents")

        logger.info("RG issue structure validation passed")
        return True


__all__ = ['RGIssueStructuralParser', 'RGIssue', 'RGSection', 'RGDocument']
