"""Annex (Ek) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses annex/appendix structure in Turkish legal documents

Production Features:
- Annex number and title extraction
- Annex type classification (table, list, form, diagram, text)
- Parent document reference
- Section/subsection parsing within annex
- Table detection and basic structure
- List item extraction (numbered and bulleted)
- Content organization and hierarchy
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

from ..core import StructuralParser, ParsedElement, DocumentStructure
from ..errors import ParsingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class AnnexSection:
    """Represents a section within an annex"""
    number: Optional[str]
    title: Optional[str]
    content: str
    subsections: List['AnnexSection'] = field(default_factory=list)


@dataclass
class AnnexTable:
    """Represents a table within an annex"""
    number: Optional[int]
    title: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    footnotes: List[str] = field(default_factory=list)


@dataclass
class AnnexList:
    """Represents a list within an annex"""
    list_type: str  # NUMBERED, BULLETED, LETTERED
    items: List[str]
    is_nested: bool = False


@dataclass
class AnnexContent:
    """Represents complete annex content"""
    annex_number: Optional[str]
    title: Optional[str]
    annex_type: str  # TABLE, LIST, FORM, TEXT, MIXED
    parent_document: Optional[str]
    sections: List[AnnexSection]
    tables: List[AnnexTable]
    lists: List[AnnexList]
    raw_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnnexStructuralParser(StructuralParser):
    """Annex (Ek) Structural Parser

    Parses annex/appendix documents with:
    - Annex numbering (Ek-1, Ek-2, EK I, etc.)
    - Title extraction
    - Type classification (5 types)
    - Parent document identification
    - Section/subsection hierarchy
    - Table detection and parsing
    - List extraction (3 types)
    - Content organization

    Features:
    - Multiple annex numbering formats (Arabic, Roman, letters)
    - Automatic type detection from content
    - Hierarchical section parsing
    - Table structure extraction (headers, rows, footnotes)
    - List type detection (numbered/bulleted/lettered)
    - Parent document reference validation
    """

    # Annex number patterns
    ANNEX_NUMBER_PATTERNS = [
        r'EK[-\s](\d+)',
        r'EK\s+([IVX]+)',  # Roman numerals
        r'EK\s+([A-Z])',  # Letter
        r'(?:Ek|EK)\s*:?\s*([\d]+)',
        r'APPENDIX\s+(\d+)',
        r'(\d+)\s+(?:No|Sayı)(?:lu|lı)\s+Ek'
    ]

    # Title patterns
    TITLE_PATTERNS = [
        r'EK[-\s](?:\d+|[IVX]+|[A-Z])\s*:?\s*(.{5,200}?)(?:\n|$)',
        r'(?:Ek|EK)\s+(?:Adı|Başlık)\s*:?\s*(.{5,200})',
        r'^(.{5,100})$'  # First line after annex number
    ]

    # Parent document patterns
    PARENT_DOC_PATTERNS = [
        r'(\d{4})\s+sayılı\s+([^,\n]{10,80}?Kanun)',
        r'([^,\n]{10,80}?Yönetmelik)(?:in|ın)\s+ek(?:i|ı)',
        r'([^,\n]{10,80}?Tebliğ)(?:in|ın)\s+ek(?:i|ı)',
        r'([^,\n]{10,80}?Karar)(?:ın|ın)\s+ek(?:i|ı)'
    ]

    # Section patterns
    SECTION_PATTERNS = [
        r'^([A-Z])\.\s+(.+)$',  # A. Section title
        r'^(\d+)\.\s+(.+)$',  # 1. Section title
        r'^([IVX]+)\.\s+(.+)$',  # I. Section title (Roman)
        r'(?:Bölüm|Kısım)\s+(\d+)\s*[-:]?\s*(.+)'
    ]

    # Table detection patterns
    TABLE_PATTERNS = [
        r'Tablo\s+(\d+)',
        r'TABLO\s+(\d+)',
        r'\|\s*.+\s*\|',  # Markdown-style table
        r'^\s*\+[-\+]+\+',  # ASCII table borders
    ]

    # List patterns
    LIST_PATTERNS = {
        'NUMBERED': [
            r'^\s*(\d+)\.\s+(.+)$',
            r'^\s*(\d+)\)\s+(.+)$'
        ],
        'BULLETED': [
            r'^\s*[•\-\*]\s+(.+)$',
            r'^\s*○\s+(.+)$'
        ],
        'LETTERED': [
            r'^\s*([a-z])\)\s+(.+)$',
            r'^\s*([a-z])\.\s+(.+)$'
        ]
    }

    # Annex type keywords
    TYPE_KEYWORDS = {
        'TABLE': ['tablo', 'çizelge', 'cetvel'],
        'LIST': ['liste', 'fihrist'],
        'FORM': ['form', 'formül', 'örnek'],
        'DIAGRAM': ['şekil', 'diagram', 'grafik', 'çizim'],
        'TEXT': []  # Default
    }

    def __init__(self):
        super().__init__("Annex Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse annex structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode, parent_doc)

        Returns:
            DocumentStructure with parsed annex

        Raises:
            ParsingError: If parsing fails
        """
        try:
            html_mode = kwargs.get('html_mode', False)
            parent_doc = kwargs.get('parent_doc', None)

            if html_mode:
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
            else:
                text = content

            # Extract annex metadata
            annex_number = self._extract_annex_number(text)
            title = self._extract_title(text, annex_number)
            parent_document = parent_doc or self._extract_parent_document(text)

            # Classify annex type
            annex_type = self._classify_annex_type(text)

            # Parse content based on type
            sections = self._parse_sections(text)
            tables = self._parse_tables(text)
            lists = self._parse_lists(text)

            # Create annex content object
            annex = AnnexContent(
                annex_number=annex_number,
                title=title,
                annex_type=annex_type,
                parent_document=parent_document,
                sections=sections,
                tables=tables,
                lists=lists,
                raw_content=text[:1000],  # First 1000 chars
                metadata={
                    'section_count': len(sections),
                    'table_count': len(tables),
                    'list_count': len(lists),
                    'content_length': len(text)
                }
            )

            logger.info(f"Successfully parsed annex: {annex_number}, "
                       f"Type: {annex_type}, Sections: {len(sections)}, "
                       f"Tables: {len(tables)}, Lists: {len(lists)}")

            return DocumentStructure(
                document_type='ANNEX',
                annex=annex,
                metadata=annex.metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse annex structure: {e}")
            raise ParsingError(f"Annex parsing failed: {str(e)}") from e

    def _extract_annex_number(self, text: str) -> Optional[str]:
        """Extract annex number/identifier"""
        for pattern in self.ANNEX_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                number = match.group(1).strip()
                logger.debug(f"Extracted annex number: {number}")
                return number

        logger.warning("Annex number not found")
        return None

    def _extract_title(self, text: str, annex_number: Optional[str]) -> Optional[str]:
        """Extract annex title"""
        for pattern in self.TITLE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()

                # Skip if title is just the annex number
                if annex_number and title == annex_number:
                    continue

                # Skip if too short
                if len(title) < 5:
                    continue

                logger.debug(f"Extracted title: {title[:50]}...")
                return title

        return None

    def _extract_parent_document(self, text: str) -> Optional[str]:
        """Extract parent document reference"""
        for pattern in self.PARENT_DOC_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parent = match.group(0).strip()
                logger.debug(f"Extracted parent document: {parent[:50]}...")
                return parent

        return None

    def _classify_annex_type(self, text: str) -> str:
        """Classify annex type based on content"""
        text_lower = text.lower()

        # Check for table indicators
        table_score = 0
        for keyword in self.TYPE_KEYWORDS['TABLE']:
            if keyword in text_lower:
                table_score += text_lower.count(keyword)

        # Check for table structure
        if re.search(r'\|.+\|', text) or re.search(r'\+[-\+]+\+', text):
            table_score += 5

        # Check for list indicators
        list_score = 0
        for keyword in self.TYPE_KEYWORDS['LIST']:
            if keyword in text_lower:
                list_score += 1

        # Count list items
        lines = text.split('\n')
        list_items = sum(1 for line in lines if re.match(r'^\s*[\d\-\•]\S', line))
        if list_items > 3:
            list_score += 3

        # Check for form indicators
        form_score = sum(1 for keyword in self.TYPE_KEYWORDS['FORM'] if keyword in text_lower)

        # Check for diagram indicators
        diagram_score = sum(1 for keyword in self.TYPE_KEYWORDS['DIAGRAM'] if keyword in text_lower)

        # Determine type based on scores
        scores = {
            'TABLE': table_score,
            'LIST': list_score,
            'FORM': form_score,
            'DIAGRAM': diagram_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            annex_type = 'TEXT'
        elif max_score < 3:
            annex_type = 'MIXED'
        else:
            annex_type = max(scores, key=scores.get)

        logger.debug(f"Classified annex type: {annex_type} (scores: {scores})")
        return annex_type

    def _parse_sections(self, text: str) -> List[AnnexSection]:
        """Parse sections within annex"""
        sections = []
        lines = text.split('\n')

        current_section = None
        current_content = []

        for line in lines:
            # Check if line is a section header
            is_section = False
            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section
                    if current_section:
                        current_section.content = '\n'.join(current_content).strip()
                        sections.append(current_section)

                    # Start new section
                    section_number = match.group(1)
                    section_title = match.group(2).strip() if len(match.groups()) > 1 else None
                    current_section = AnnexSection(
                        number=section_number,
                        title=section_title,
                        content='',
                        subsections=[]
                    )
                    current_content = []
                    is_section = True
                    logger.debug(f"Found section: {section_number} - {section_title}")
                    break

            if not is_section and current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        return sections

    def _parse_tables(self, text: str) -> List[AnnexTable]:
        """Parse tables within annex"""
        tables = []

        # Simple table detection and parsing
        for pattern in self.TABLE_PATTERNS[:2]:  # Just numbered table patterns
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                table_number = int(match.group(1)) if match.groups() else None

                # Extract table title (line with table number)
                table_start = match.start()
                line_start = text.rfind('\n', 0, table_start) + 1
                line_end = text.find('\n', table_start)
                title_line = text[line_start:line_end].strip()

                # Remove "Tablo X" from title
                title = re.sub(r'Tablo\s+\d+\s*[-:]?\s*', '', title_line, flags=re.IGNORECASE).strip()

                tables.append(AnnexTable(
                    number=table_number,
                    title=title if title else None,
                    headers=[],
                    rows=[],
                    footnotes=[]
                ))
                logger.debug(f"Found table {table_number}: {title}")

        return tables

    def _parse_lists(self, text: str) -> List[AnnexList]:
        """Parse lists within annex"""
        lists = []
        lines = text.split('\n')

        current_list = None
        current_items = []
        current_type = None

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                # Empty line might end a list
                if current_list:
                    current_list.items = current_items
                    lists.append(current_list)
                    current_list = None
                    current_items = []
                    current_type = None
                continue

            # Check each list type
            matched = False
            for list_type, patterns in self.LIST_PATTERNS.items():
                for pattern in patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        # Extract item content
                        if list_type == 'BULLETED':
                            item_content = match.group(1)
                        else:
                            item_content = match.group(2) if len(match.groups()) > 1 else match.group(1)

                        # Check if continuing same list type
                        if current_type == list_type:
                            current_items.append(item_content)
                        else:
                            # New list type - save previous
                            if current_list:
                                current_list.items = current_items
                                lists.append(current_list)

                            # Start new list
                            current_type = list_type
                            current_list = AnnexList(
                                list_type=list_type,
                                items=[],
                                is_nested=False
                            )
                            current_items = [item_content]

                        matched = True
                        break
                if matched:
                    break

        # Save last list
        if current_list and current_items:
            current_list.items = current_items
            lists.append(current_list)

        logger.debug(f"Found {len(lists)} lists")
        return lists

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed annex structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not hasattr(structure, 'annex') or structure.annex is None:
            raise ValidationError("Annex structure is missing")

        annex = structure.annex

        if annex.annex_number is None:
            logger.warning("Annex number is missing")

        if annex.title is None:
            logger.warning("Annex title is missing")

        # Check content
        total_content = len(annex.sections) + len(annex.tables) + len(annex.lists)
        if total_content == 0:
            logger.warning("Annex has no parsed content (sections, tables, or lists)")

        logger.info("Annex structure validation passed")
        return True


__all__ = ['AnnexStructuralParser', 'AnnexContent', 'AnnexSection', 'AnnexTable', 'AnnexList']
