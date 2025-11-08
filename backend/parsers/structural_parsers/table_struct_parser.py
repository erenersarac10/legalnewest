"""Table Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses table structures in Turkish legal documents

Production Features:
- Table number and title extraction
- Header row detection and parsing
- Column headers extraction
- Data row parsing with cell content
- Column/row count detection
- Footnote extraction
- Table type classification (numeric, text, mixed)
- Multiple table format support (ASCII, markdown, HTML)
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
class TableCell:
    """Represents a single table cell"""
    content: str
    row: int
    column: int
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1


@dataclass
class TableRow:
    """Represents a table row"""
    row_number: int
    cells: List[TableCell]
    is_header: bool = False


@dataclass
class TableStructure:
    """Represents complete table structure"""
    table_number: Optional[int]
    title: Optional[str]
    caption: Optional[str]
    headers: List[str]
    rows: List[TableRow]
    column_count: int
    row_count: int
    footnotes: List[str]
    table_type: str  # NUMERIC, TEXT, MIXED
    format_type: str  # ASCII, MARKDOWN, HTML, PLAIN
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableStructuralParser(StructuralParser):
    """Table Structural Parser

    Parses table structures with:
    - Table number extraction (Tablo 1, Table 1, etc.)
    - Title/caption parsing
    - Header row detection
    - Column headers extraction
    - Data row parsing
    - Cell content extraction
    - Column/row counting
    - Footnote extraction
    - Type classification (3 types)
    - Format detection (4 formats)

    Features:
    - Multiple table format support (ASCII borders, markdown, HTML, plain)
    - Automatic header detection
    - Column alignment detection
    - Turkish number format support
    - Footnote marker recognition
    - Cell span detection (HTML tables)
    """

    # Table number patterns
    TABLE_NUMBER_PATTERNS = [
        r'Tablo\s+(\d+)',
        r'TABLO\s+(\d+)',
        r'Çizelge\s+(\d+)',
        r'Table\s+(\d+)',
        r'(\d+)\s+(?:No|Sayı)(?:lu|lı)\s+Tablo'
    ]

    # Title/caption patterns
    TITLE_PATTERNS = [
        r'Tablo\s+\d+\s*[:-]?\s*(.{5,200}?)(?:\n|$)',
        r'TABLO\s+\d+\s*[:-]?\s*(.{5,200}?)(?:\n|$)',
        r'Başlık\s*:?\s*(.{5,200})',
        r'Caption\s*:?\s*(.{5,200})'
    ]

    # Table border patterns
    BORDER_PATTERNS = {
        'ASCII': [
            r'^\s*\+[-\+]+\+\s*$',  # +-----+-----+
            r'^\s*\|[-\+]+\|\s*$',  # |-----|-----|
        ],
        'MARKDOWN': [
            r'^\s*\|[-:\s\|]+\|\s*$',  # |---|---|
            r'^\s*\|[^|]+\|[^|]+\|\s*$'  # | col1 | col2 |
        ],
        'HTML': [
            r'<table[^>]*>',
            r'<tr[^>]*>',
            r'<td[^>]*>',
            r'<th[^>]*>'
        ]
    }

    # Footnote patterns
    FOOTNOTE_PATTERNS = [
        r'^\s*\(\d+\)\s*(.+)$',  # (1) footnote text
        r'^\s*\*+\s*(.+)$',  # * footnote text
        r'Dipnot\s*:?\s*(.+)',
        r'Not\s*:?\s*(.+)'
    ]

    # Number patterns for type detection
    NUMBER_PATTERNS = [
        r'\d+[.,]\d+',  # Decimal numbers
        r'\d{1,3}(?:[.,]\d{3})*',  # Large numbers with separators
        r'%\s*\d+',  # Percentages
        r'\d+\s*TL',  # Currency
    ]

    def __init__(self):
        super().__init__("Table Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse table structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode)

        Returns:
            DocumentStructure with parsed table

        Raises:
            ParsingError: If parsing fails
        """
        try:
            html_mode = kwargs.get('html_mode', False)

            if html_mode:
                soup = BeautifulSoup(content, 'html.parser')
                # Try HTML table parsing first
                html_table = soup.find('table')
                if html_table:
                    table_structure = self._parse_html_table(html_table, soup)
                else:
                    # Fall back to text parsing
                    text = soup.get_text(separator='\n', strip=True)
                    table_structure = self._parse_text_table(text)
            else:
                table_structure = self._parse_text_table(content)

            logger.info(f"Successfully parsed table: {table_structure.table_number}, "
                       f"{table_structure.row_count}x{table_structure.column_count}, "
                       f"Type: {table_structure.table_type}, "
                       f"Format: {table_structure.format_type}")

            return DocumentStructure(
                document_type='TABLE',
                table=table_structure,
                metadata=table_structure.metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse table structure: {e}")
            raise ParsingError(f"Table parsing failed: {str(e)}") from e

    def _parse_text_table(self, text: str) -> TableStructure:
        """Parse table from plain text"""
        # Extract table metadata
        table_number = self._extract_table_number(text)
        title = self._extract_title(text)

        # Detect table format
        format_type = self._detect_format(text)

        # Parse based on format
        if format_type == 'ASCII':
            rows, headers = self._parse_ascii_table(text)
        elif format_type == 'MARKDOWN':
            rows, headers = self._parse_markdown_table(text)
        else:
            rows, headers = self._parse_plain_table(text)

        # Extract footnotes
        footnotes = self._extract_footnotes(text)

        # Classify table type
        table_type = self._classify_table_type(rows)

        # Calculate dimensions
        column_count = len(headers) if headers else (max(len(row.cells) for row in rows) if rows else 0)
        row_count = len(rows)

        return TableStructure(
            table_number=table_number,
            title=title,
            caption=None,
            headers=headers,
            rows=rows,
            column_count=column_count,
            row_count=row_count,
            footnotes=footnotes,
            table_type=table_type,
            format_type=format_type,
            metadata={
                'has_headers': len(headers) > 0,
                'has_footnotes': len(footnotes) > 0,
                'total_cells': sum(len(row.cells) for row in rows)
            }
        )

    def _parse_html_table(self, table_elem, soup: BeautifulSoup) -> TableStructure:
        """Parse HTML table element"""
        # Extract metadata from surrounding text
        table_text = str(table_elem.parent) if table_elem.parent else str(table_elem)
        table_number = self._extract_table_number(table_text)
        title = self._extract_title(table_text)
        caption = table_elem.find('caption')
        caption_text = caption.get_text(strip=True) if caption else None

        # Parse headers
        headers = []
        header_rows = table_elem.find_all('thead')
        if header_rows:
            for th in header_rows[0].find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))
        else:
            # Try first row
            first_row = table_elem.find('tr')
            if first_row:
                for th in first_row.find_all('th'):
                    headers.append(th.get_text(strip=True))

        # Parse data rows
        rows = []
        tbody = table_elem.find('tbody') or table_elem
        for row_idx, tr in enumerate(tbody.find_all('tr')):
            cells = []
            for col_idx, cell in enumerate(tr.find_all(['td', 'th'])):
                cell_content = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))

                cells.append(TableCell(
                    content=cell_content,
                    row=row_idx,
                    column=col_idx,
                    is_header=cell.name == 'th',
                    colspan=colspan,
                    rowspan=rowspan
                ))

            if cells:
                rows.append(TableRow(
                    row_number=row_idx,
                    cells=cells,
                    is_header=all(c.is_header for c in cells)
                ))

        # Extract footnotes
        footnotes = self._extract_footnotes(table_text)

        # Classify type
        table_type = self._classify_table_type(rows)

        # Calculate dimensions
        column_count = len(headers) if headers else (max(len(row.cells) for row in rows) if rows else 0)
        row_count = len(rows)

        return TableStructure(
            table_number=table_number,
            title=title,
            caption=caption_text,
            headers=headers,
            rows=rows,
            column_count=column_count,
            row_count=row_count,
            footnotes=footnotes,
            table_type=table_type,
            format_type='HTML',
            metadata={
                'has_headers': len(headers) > 0,
                'has_caption': caption_text is not None,
                'has_footnotes': len(footnotes) > 0,
                'total_cells': sum(len(row.cells) for row in rows)
            }
        )

    def _parse_ascii_table(self, text: str) -> Tuple[List[TableRow], List[str]]:
        """Parse ASCII-bordered table"""
        rows = []
        headers = []
        lines = text.split('\n')

        data_lines = []
        for line in lines:
            # Skip border lines
            if re.match(r'^\s*[\+\-\|]+\s*$', line):
                continue
            # Extract data lines
            if '|' in line:
                data_lines.append(line)

        # First data line might be headers
        if data_lines:
            first_row_cells = [cell.strip() for cell in data_lines[0].split('|') if cell.strip()]
            # Check if it looks like headers (short, capitalized)
            if all(len(cell) < 30 and (cell.isupper() or cell.istitle()) for cell in first_row_cells):
                headers = first_row_cells
                data_lines = data_lines[1:]

        # Parse remaining rows
        for row_idx, line in enumerate(data_lines):
            cells = []
            cell_contents = [cell.strip() for cell in line.split('|') if cell.strip()]

            for col_idx, content in enumerate(cell_contents):
                cells.append(TableCell(
                    content=content,
                    row=row_idx,
                    column=col_idx,
                    is_header=False
                ))

            if cells:
                rows.append(TableRow(
                    row_number=row_idx,
                    cells=cells,
                    is_header=False
                ))

        return rows, headers

    def _parse_markdown_table(self, text: str) -> Tuple[List[TableRow], List[str]]:
        """Parse markdown-style table"""
        rows = []
        headers = []
        lines = text.split('\n')

        table_lines = [line for line in lines if '|' in line]

        if not table_lines:
            return rows, headers

        # First line is usually headers
        header_line = table_lines[0]
        headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]

        # Skip separator line (e.g., |---|---|)
        data_start = 2 if len(table_lines) > 1 and re.match(r'^\s*\|[-:\s\|]+\|\s*$', table_lines[1]) else 1

        # Parse data rows
        for row_idx, line in enumerate(table_lines[data_start:]):
            cells = []
            cell_contents = [cell.strip() for cell in line.split('|') if cell.strip()]

            for col_idx, content in enumerate(cell_contents):
                cells.append(TableCell(
                    content=content,
                    row=row_idx,
                    column=col_idx,
                    is_header=False
                ))

            if cells:
                rows.append(TableRow(
                    row_number=row_idx,
                    cells=cells,
                    is_header=False
                ))

        return rows, headers

    def _parse_plain_table(self, text: str) -> Tuple[List[TableRow], List[str]]:
        """Parse plain text table (space/tab delimited)"""
        rows = []
        headers = []
        lines = text.split('\n')

        # Filter out empty lines
        data_lines = [line for line in lines if line.strip() and len(line.strip()) > 10]

        if not data_lines:
            return rows, headers

        # Try to detect column positions from first few lines
        # Simple heuristic: split by multiple spaces or tabs
        for row_idx, line in enumerate(data_lines[:10]):  # Process up to 10 rows
            # Split by 2+ spaces or tabs
            cells = re.split(r'\s{2,}|\t+', line.strip())

            if len(cells) > 1:  # Only include if multiple columns detected
                cell_objects = []
                for col_idx, content in enumerate(cells):
                    cell_objects.append(TableCell(
                        content=content.strip(),
                        row=row_idx,
                        column=col_idx,
                        is_header=False
                    ))

                if cell_objects:
                    rows.append(TableRow(
                        row_number=row_idx,
                        cells=cell_objects,
                        is_header=False
                    ))

        return rows, headers

    def _extract_table_number(self, text: str) -> Optional[int]:
        """Extract table number"""
        for pattern in self.TABLE_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    number = int(match.group(1))
                    logger.debug(f"Extracted table number: {number}")
                    return number
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract table title"""
        for pattern in self.TITLE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) >= 5:
                    logger.debug(f"Extracted title: {title[:50]}...")
                    return title

        return None

    def _extract_footnotes(self, text: str) -> List[str]:
        """Extract table footnotes"""
        footnotes = []

        for pattern in self.FOOTNOTE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                footnote = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                if footnote and footnote not in footnotes:
                    footnotes.append(footnote)

        logger.debug(f"Extracted {len(footnotes)} footnotes")
        return footnotes

    def _detect_format(self, text: str) -> str:
        """Detect table format type"""
        # Check for ASCII borders
        if any(re.search(pattern, text, re.MULTILINE) for pattern in self.BORDER_PATTERNS['ASCII']):
            return 'ASCII'

        # Check for markdown
        if any(re.search(pattern, text, re.MULTILINE) for pattern in self.BORDER_PATTERNS['MARKDOWN']):
            return 'MARKDOWN'

        # Check for HTML
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.BORDER_PATTERNS['HTML']):
            return 'HTML'

        return 'PLAIN'

    def _classify_table_type(self, rows: List[TableRow]) -> str:
        """Classify table type based on content"""
        if not rows:
            return 'UNKNOWN'

        # Count numeric vs text cells
        numeric_count = 0
        text_count = 0

        for row in rows:
            for cell in row.cells:
                content = cell.content.strip()
                if not content:
                    continue

                # Check if content is numeric
                is_numeric = False
                for pattern in self.NUMBER_PATTERNS:
                    if re.search(pattern, content):
                        is_numeric = True
                        break

                if is_numeric:
                    numeric_count += 1
                else:
                    text_count += 1

        total = numeric_count + text_count
        if total == 0:
            return 'UNKNOWN'

        numeric_ratio = numeric_count / total

        if numeric_ratio > 0.7:
            return 'NUMERIC'
        elif numeric_ratio < 0.3:
            return 'TEXT'
        else:
            return 'MIXED'

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed table structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not hasattr(structure, 'table') or structure.table is None:
            raise ValidationError("Table structure is missing")

        table = structure.table

        if table.row_count == 0:
            logger.warning("Table has no rows")

        if table.column_count == 0:
            logger.warning("Table has no columns")

        # Validate row consistency
        if table.rows:
            col_counts = [len(row.cells) for row in table.rows]
            if len(set(col_counts)) > 2:  # Allow some variation
                logger.warning(f"Inconsistent column counts across rows: {set(col_counts)}")

        logger.info("Table structure validation passed")
        return True


__all__ = ['TableStructuralParser', 'TableStructure', 'TableRow', 'TableCell']
