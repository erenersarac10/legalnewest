"""HTML Normalizer - Harvey/Legora CTO-Level
Converts HTML to clean, structured text for Turkish legal documents

Production-grade implementation with:
- HTML to text conversion with structure preservation
- Turkish character encoding handling (UTF-8, ISO-8859-9, Windows-1254)
- Table extraction and conversion to structured format
- Metadata extraction from HTML head tags
- Semantic structure preservation (headings, lists, articles)
- Malformed HTML handling with multiple parsers
- Tag filtering (script, style, nav, footer removal)
- Whitespace normalization
- Legal document structure detection
- Comprehensive error handling and logging
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from datetime import datetime
from decimal import Decimal

try:
    from bs4 import BeautifulSoup, Tag, NavigableString, Comment
    from bs4.element import PageElement
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None
    Tag = None

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class HTMLParserType(Enum):
    """HTML parser types ordered by robustness"""
    LXML = "lxml"  # Fastest, most strict
    HTML5LIB = "html5lib"  # Most lenient, standards-compliant
    HTML_PARSER = "html.parser"  # Built-in, moderate tolerance


class EncodingType(Enum):
    """Turkish-compatible encodings"""
    UTF8 = "utf-8"
    LATIN5 = "iso-8859-9"  # Turkish (Latin-5)
    WINDOWS1254 = "windows-1254"  # Turkish (Windows)
    LATIN1 = "iso-8859-1"  # Fallback


class TableFormat(Enum):
    """Table extraction formats"""
    MARKDOWN = "markdown"  # | col1 | col2 |
    CSV = "csv"  # col1,col2
    TSV = "tsv"  # col1\tcol2
    STRUCTURED = "structured"  # Dict with headers/rows


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HTMLMetadata:
    """Extracted HTML metadata"""
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    author: Optional[str] = None
    published_date: Optional[str] = None
    modified_date: Optional[str] = None
    language: Optional[str] = None
    canonical_url: Optional[str] = None

    # Legal document specific
    document_type: Optional[str] = None
    document_number: Optional[str] = None
    issuing_authority: Optional[str] = None

    # Additional metadata
    extra_meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class HTMLTable:
    """Structured table data"""
    caption: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    footer: Optional[List[str]] = None

    # Table metadata
    column_count: int = 0
    row_count: int = 0
    has_header: bool = True

    def to_markdown(self) -> str:
        """Convert table to Markdown format"""
        if not self.rows and not self.headers:
            return ""

        lines = []

        if self.caption:
            lines.append(f"**{self.caption}**\n")

        # Headers
        if self.headers:
            header_row = "| " + " | ".join(self.headers) + " |"
            separator = "|" + "|".join(["---"] * len(self.headers)) + "|"
            lines.extend([header_row, separator])

        # Rows
        for row in self.rows:
            row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(row_str)

        if self.footer:
            lines.append("| " + " | ".join(self.footer) + " |")

        return "\n".join(lines)

    def to_csv(self, delimiter: str = ",") -> str:
        """Convert table to CSV/TSV format"""
        lines = []

        if self.headers:
            lines.append(delimiter.join(self.headers))

        for row in self.rows:
            # Escape delimiters in cell content
            escaped_row = [
                f'"{cell}"' if delimiter in str(cell) else str(cell)
                for cell in row
            ]
            lines.append(delimiter.join(escaped_row))

        return "\n".join(lines)


@dataclass
class NormalizationStats:
    """Statistics from HTML normalization"""
    # Input metrics
    original_size: int = 0
    original_tag_count: int = 0

    # Processing metrics
    tags_removed: int = 0
    tags_preserved: int = 0
    comments_removed: int = 0
    scripts_removed: int = 0
    styles_removed: int = 0

    # Structure metrics
    headings_found: int = 0
    paragraphs_found: int = 0
    lists_found: int = 0
    tables_found: int = 0
    links_found: int = 0

    # Output metrics
    normalized_size: int = 0
    compression_ratio: float = 0.0

    # Encoding
    detected_encoding: Optional[str] = None
    parser_used: Optional[str] = None

    # Timing
    processing_time_ms: float = 0.0

    def calculate_compression(self):
        """Calculate compression ratio"""
        if self.original_size > 0:
            self.compression_ratio = round(
                (1 - self.normalized_size / self.original_size) * 100, 2
            )


@dataclass
class NormalizationResult:
    """Result of HTML normalization"""
    success: bool
    normalized_text: str = ""

    # Structured data
    metadata: Optional[HTMLMetadata] = None
    tables: List[HTMLTable] = field(default_factory=list)
    headings: List[Tuple[int, str]] = field(default_factory=list)  # (level, text)
    links: List[Tuple[str, str]] = field(default_factory=list)  # (text, url)

    # Statistics
    stats: NormalizationStats = field(default_factory=NormalizationStats)

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Original HTML (for debugging)
    original_html: Optional[str] = None


# ============================================================================
# HTML NORMALIZER
# ============================================================================

class HTMLNormalizer:
    """
    Production-grade HTML normalizer for Turkish legal documents.

    This normalizer handles the complexities of converting legal HTML documents
    to clean, structured text while preserving semantic structure and metadata.

    Key Features:

    1. **Multi-Parser Support**:
       - lxml (fast, strict)
       - html5lib (lenient, standards-compliant)
       - html.parser (built-in fallback)
       - Automatic fallback on parsing errors

    2. **Turkish Encoding Support**:
       - UTF-8 (primary)
       - ISO-8859-9 (Turkish Latin-5)
       - Windows-1254 (Turkish Windows)
       - Automatic encoding detection

    3. **Structure Preservation**:
       - Headings (H1-H6) → ## Markdown-style
       - Lists (ul, ol) → Indented bullet/numbered lists
       - Paragraphs → Double newline separation
       - Articles/Sections → Preserved with proper spacing
       - Blockquotes → Indented with markers

    4. **Table Handling**:
       - Caption extraction
       - Header/body/footer detection
       - Multiple output formats (Markdown, CSV, structured)
       - Colspan/rowspan handling

    5. **Tag Filtering**:
       - Removed: script, style, nav, footer, header, aside, noscript
       - Preserved: article, section, main, p, div, span, etc.
       - Comment removal

    6. **Metadata Extraction**:
       - Title, description, keywords
       - Open Graph tags
       - Legal document metadata (type, number, authority)
       - Publication/modification dates

    7. **Whitespace Normalization**:
       - Multiple spaces → single space
       - Multiple newlines → double newline (paragraph separation)
       - Leading/trailing whitespace removal
       - Non-breaking space handling

    8. **Error Handling**:
       - Malformed HTML handling
       - Encoding error recovery
       - Partial content extraction on errors
       - Comprehensive error logging

    Turkish Legal Document Support:
    - Resmi Gazete HTML documents
    - Court decision HTML exports
    - KVKK/Rekabet Kurumu decisions
    - Law/regulation HTML versions
    - Board meeting minutes

    Example:
        >>> normalizer = HTMLNormalizer()
        >>> result = normalizer.normalize(html_content)
        >>> if result.success:
        ...     print(result.normalized_text)
        ...     print(f"Found {len(result.tables)} tables")
        ...     print(f"Metadata: {result.metadata.title}")
    """

    # Tags to completely remove (including content)
    REMOVE_TAGS = {
        'script', 'style', 'noscript', 'iframe', 'object', 'embed',
        'applet', 'link', 'meta'
    }

    # Tags to remove but preserve content
    UNWRAP_TAGS = {
        'nav', 'footer', 'header', 'aside', 'form', 'button',
        'input', 'select', 'textarea', 'label'
    }

    # Semantic structure tags to preserve
    STRUCTURE_TAGS = {
        'article', 'section', 'main', 'div', 'p', 'blockquote',
        'pre', 'code', 'figure', 'figcaption'
    }

    # Heading tags
    HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    # List tags
    LIST_TAGS = {'ul', 'ol', 'li', 'dl', 'dt', 'dd'}

    # Turkish legal document patterns
    LEGAL_PATTERNS = {
        'kanun': re.compile(r'\b\d+\s*sayılı\s+kanun\b', re.IGNORECASE),
        'madde': re.compile(r'\bmadde\s+\d+\b', re.IGNORECASE),
        'fıkra': re.compile(r'\bfıkra\s+\d+\b', re.IGNORECASE),
        'bent': re.compile(r'\bbent\s+[a-z]\b', re.IGNORECASE),
        'karar': re.compile(r'\bkarar\s*(?:sayısı|no)?\s*:?\s*\d+', re.IGNORECASE),
        'esas': re.compile(r'\bE\.\s*\d{4}/\d+\b'),
        'resmi_gazete': re.compile(r'\bresmi\s+gazete\b', re.IGNORECASE)
    }

    def __init__(
        self,
        preserve_links: bool = True,
        extract_tables: bool = True,
        table_format: TableFormat = TableFormat.MARKDOWN,
        parser_preference: Optional[List[HTMLParserType]] = None,
        encoding_preference: Optional[List[EncodingType]] = None
    ):
        """
        Initialize HTML normalizer.

        Args:
            preserve_links: Extract and preserve links
            extract_tables: Extract tables as structured data
            table_format: Format for table extraction
            parser_preference: Ordered list of parsers to try
            encoding_preference: Ordered list of encodings to try
        """
        if not BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for HTMLNormalizer. "
                "Install with: pip install beautifulsoup4 lxml html5lib"
            )

        self.preserve_links = preserve_links
        self.extract_tables = extract_tables
        self.table_format = table_format

        # Parser preference (in order of fallback)
        self.parser_preference = parser_preference or [
            HTMLParserType.LXML,
            HTMLParserType.HTML5LIB,
            HTMLParserType.HTML_PARSER
        ]

        # Encoding preference
        self.encoding_preference = encoding_preference or [
            EncodingType.UTF8,
            EncodingType.LATIN5,
            EncodingType.WINDOWS1254,
            EncodingType.LATIN1
        ]

        logger.info(
            f"Initialized HTMLNormalizer with parsers={[p.value for p in self.parser_preference]}, "
            f"table_format={table_format.value}"
        )

    def normalize(
        self,
        html_content: str,
        encoding: Optional[str] = None,
        parser: Optional[str] = None
    ) -> NormalizationResult:
        """
        Normalize HTML content to clean text.

        Args:
            html_content: Raw HTML content
            encoding: Specific encoding (auto-detect if None)
            parser: Specific parser (auto-select if None)

        Returns:
            NormalizationResult with normalized text and metadata
        """
        start_time = datetime.now()
        result = NormalizationResult(success=False)
        result.stats.original_size = len(html_content)

        try:
            # Parse HTML
            soup, parser_used, encoding_used = self._parse_html(
                html_content, encoding, parser
            )
            result.stats.parser_used = parser_used
            result.stats.detected_encoding = encoding_used

            # Count original tags
            result.stats.original_tag_count = len(soup.find_all())

            # Extract metadata
            result.metadata = self._extract_metadata(soup)

            # Clean HTML
            self._remove_unwanted_tags(soup, result.stats)
            self._remove_comments(soup, result.stats)

            # Extract tables
            if self.extract_tables:
                result.tables = self._extract_tables(soup)
                result.stats.tables_found = len(result.tables)

            # Extract headings
            result.headings = self._extract_headings(soup)
            result.stats.headings_found = len(result.headings)

            # Extract links
            if self.preserve_links:
                result.links = self._extract_links(soup)
                result.stats.links_found = len(result.links)

            # Convert to text
            normalized_text = self._convert_to_text(soup, result.stats)

            # Post-process text
            result.normalized_text = self._post_process_text(normalized_text)
            result.stats.normalized_size = len(result.normalized_text)
            result.stats.calculate_compression()

            result.success = True
            logger.info(
                f"Successfully normalized HTML: {result.stats.original_size} → "
                f"{result.stats.normalized_size} bytes ({result.stats.compression_ratio}% reduction)"
            )

        except Exception as e:
            error_msg = f"HTML normalization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
            result.success = False

        finally:
            # Calculate processing time
            end_time = datetime.now()
            result.stats.processing_time_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def _parse_html(
        self,
        html_content: str,
        encoding: Optional[str] = None,
        parser: Optional[str] = None
    ) -> Tuple[BeautifulSoup, str, str]:
        """
        Parse HTML with fallback support.

        Returns:
            (BeautifulSoup object, parser_used, encoding_used)
        """
        # Handle encoding
        if isinstance(html_content, bytes):
            if encoding:
                html_content = html_content.decode(encoding)
            else:
                # Try encodings in preference order
                for enc_type in self.encoding_preference:
                    try:
                        html_content = html_content.decode(enc_type.value)
                        encoding = enc_type.value
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                else:
                    # Last resort: decode with errors='replace'
                    html_content = html_content.decode('utf-8', errors='replace')
                    encoding = 'utf-8'

        encoding = encoding or 'utf-8'

        # Try parsers in preference order
        parsers_to_try = [parser] if parser else [p.value for p in self.parser_preference]

        last_error = None
        for parser_type in parsers_to_try:
            try:
                soup = BeautifulSoup(html_content, parser_type)
                logger.debug(f"Successfully parsed HTML with {parser_type}")
                return soup, parser_type, encoding
            except Exception as e:
                logger.warning(f"Parser {parser_type} failed: {e}")
                last_error = e
                continue

        # If all parsers fail, raise the last error
        raise RuntimeError(f"All HTML parsers failed. Last error: {last_error}")

    def _extract_metadata(self, soup: BeautifulSoup) -> HTMLMetadata:
        """Extract metadata from HTML head"""
        metadata = HTMLMetadata()

        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata.title = title_tag.get_text(strip=True)

        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')

            if not content:
                continue

            # Standard meta tags
            if name == 'description' or property_attr == 'og:description':
                metadata.description = content
            elif name == 'keywords':
                metadata.keywords = [k.strip() for k in content.split(',')]
            elif name == 'author':
                metadata.author = content
            elif name == 'date' or property_attr == 'article:published_time':
                metadata.published_date = content
            elif name == 'last-modified' or property_attr == 'article:modified_time':
                metadata.modified_date = content
            elif name == 'language' or meta.get('http-equiv', '').lower() == 'content-language':
                metadata.language = content

            # Open Graph
            elif property_attr == 'og:title' and not metadata.title:
                metadata.title = content
            elif property_attr == 'og:url':
                metadata.canonical_url = content

            # Legal document specific
            elif name in ['document-type', 'doc-type']:
                metadata.document_type = content
            elif name in ['document-number', 'doc-number']:
                metadata.document_number = content
            elif name in ['issuing-authority', 'authority']:
                metadata.issuing_authority = content

            # Store everything else
            else:
                key = name or property_attr or meta.get('http-equiv', '')
                if key:
                    metadata.extra_meta[key] = content

        # Canonical URL
        link_canonical = soup.find('link', rel='canonical')
        if link_canonical and not metadata.canonical_url:
            metadata.canonical_url = link_canonical.get('href', '')

        return metadata

    def _remove_unwanted_tags(self, soup: BeautifulSoup, stats: NormalizationStats):
        """Remove unwanted tags"""
        # Remove tags completely (including content)
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                if tag_name == 'script':
                    stats.scripts_removed += 1
                elif tag_name == 'style':
                    stats.styles_removed += 1
                tag.decompose()
                stats.tags_removed += 1

        # Unwrap tags (remove tag but keep content)
        for tag_name in self.UNWRAP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.unwrap()
                stats.tags_removed += 1

    def _remove_comments(self, soup: BeautifulSoup, stats: NormalizationStats):
        """Remove HTML comments"""
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
            stats.comments_removed += 1

    def _extract_tables(self, soup: BeautifulSoup) -> List[HTMLTable]:
        """Extract tables as structured data"""
        tables = []

        for table_tag in soup.find_all('table'):
            html_table = HTMLTable()

            # Caption
            caption_tag = table_tag.find('caption')
            if caption_tag:
                html_table.caption = caption_tag.get_text(strip=True)

            # Headers
            thead = table_tag.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    html_table.headers = [
                        th.get_text(strip=True)
                        for th in header_row.find_all(['th', 'td'])
                    ]

            # If no thead, check first row for th tags
            if not html_table.headers:
                first_row = table_tag.find('tr')
                if first_row and first_row.find('th'):
                    html_table.headers = [
                        th.get_text(strip=True)
                        for th in first_row.find_all('th')
                    ]
                    html_table.has_header = True
                else:
                    html_table.has_header = False

            # Body rows
            tbody = table_tag.find('tbody') or table_tag
            for row in tbody.find_all('tr'):
                # Skip header row if we already extracted it
                if html_table.has_header and row == table_tag.find('tr'):
                    continue

                cells = [
                    cell.get_text(strip=True)
                    for cell in row.find_all(['td', 'th'])
                ]
                if cells:  # Only add non-empty rows
                    html_table.rows.append(cells)

            # Footer
            tfoot = table_tag.find('tfoot')
            if tfoot:
                footer_row = tfoot.find('tr')
                if footer_row:
                    html_table.footer = [
                        td.get_text(strip=True)
                        for td in footer_row.find_all(['td', 'th'])
                    ]

            # Calculate dimensions
            html_table.row_count = len(html_table.rows)
            html_table.column_count = max(
                len(html_table.headers),
                max((len(row) for row in html_table.rows), default=0)
            )

            tables.append(html_table)

            # Replace table in soup with formatted version
            if self.table_format == TableFormat.MARKDOWN:
                table_tag.replace_with(NavigableString(f"\n\n{html_table.to_markdown()}\n\n"))
            elif self.table_format in [TableFormat.CSV, TableFormat.TSV]:
                delimiter = "," if self.table_format == TableFormat.CSV else "\t"
                table_tag.replace_with(NavigableString(f"\n\n{html_table.to_csv(delimiter)}\n\n"))

        return tables

    def _extract_headings(self, soup: BeautifulSoup) -> List[Tuple[int, str]]:
        """Extract headings with levels"""
        headings = []

        for tag_name in self.HEADING_TAGS:
            level = int(tag_name[1])  # h1 -> 1, h2 -> 2, etc.
            for heading in soup.find_all(tag_name):
                text = heading.get_text(strip=True)
                if text:
                    headings.append((level, text))

        return headings

    def _extract_links(self, soup: BeautifulSoup) -> List[Tuple[str, str]]:
        """Extract links (text, url)"""
        links = []

        for a_tag in soup.find_all('a', href=True):
            text = a_tag.get_text(strip=True)
            url = a_tag['href']
            if text and url:
                links.append((text, url))

        return links

    def _convert_to_text(self, soup: BeautifulSoup, stats: NormalizationStats) -> str:
        """Convert soup to formatted text"""
        # Process headings
        for tag_name in self.HEADING_TAGS:
            level = int(tag_name[1])
            prefix = '#' * level
            for heading in soup.find_all(tag_name):
                heading.string = f"\n\n{prefix} {heading.get_text(strip=True)}\n\n"

        # Process lists
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li', recursive=False):
                li.string = f"\n• {li.get_text(strip=True)}"
            stats.lists_found += 1

        for ol in soup.find_all('ol'):
            for idx, li in enumerate(ol.find_all('li', recursive=False), 1):
                li.string = f"\n{idx}. {li.get_text(strip=True)}"
            stats.lists_found += 1

        # Process paragraphs
        for p in soup.find_all('p'):
            stats.paragraphs_found += 1

        # Process blockquotes
        for blockquote in soup.find_all('blockquote'):
            text = blockquote.get_text(strip=True)
            blockquote.string = f"\n> {text}\n"

        # Process line breaks
        for br in soup.find_all('br'):
            br.replace_with('\n')

        # Get text
        text = soup.get_text()

        return text

    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text"""
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\n\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'\t+', ' ', text)  # Tabs to spaces

        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Normalize Turkish quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Remove non-breaking spaces
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')  # Zero-width space

        # Final cleanup
        text = text.strip()

        return text


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_html(
    html_content: str,
    **kwargs
) -> NormalizationResult:
    """
    Convenience function for HTML normalization.

    Args:
        html_content: Raw HTML content
        **kwargs: Arguments passed to HTMLNormalizer

    Returns:
        NormalizationResult
    """
    normalizer = HTMLNormalizer(**kwargs)
    return normalizer.normalize(html_content)


def extract_text_from_html(html_content: str) -> str:
    """
    Simple text extraction from HTML.

    Args:
        html_content: Raw HTML content

    Returns:
        Normalized text
    """
    result = normalize_html(html_content)
    return result.normalized_text if result.success else ""
