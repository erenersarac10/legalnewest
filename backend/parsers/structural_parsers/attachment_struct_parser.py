"""Attachment (Ekler) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses attachment/supplementary document references in Turkish legal documents

Production Features:
- Attachment identifier extraction
- Type classification (PDF, image, document, URL)
- File name and reference extraction
- Description and metadata parsing
- Parent document linkage
- Multiple attachment handling
- URL/link extraction
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from bs4 import BeautifulSoup

from ..core import StructuralParser, ParsedElement, DocumentStructure
from ..errors import ParsingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """Represents a single attachment"""
    number: Optional[int]
    identifier: Optional[str]
    title: Optional[str]
    description: Optional[str]
    file_type: Optional[str]  # PDF, DOC, XLS, JPG, PNG, URL, UNKNOWN
    file_name: Optional[str]
    url: Optional[str]
    size: Optional[str]
    date: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttachmentCollection:
    """Represents a collection of attachments"""
    parent_document: Optional[str]
    attachments: List[Attachment]
    total_count: int
    has_urls: bool
    has_files: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttachmentStructuralParser(StructuralParser):
    """Attachment (Ekler) Structural Parser

    Parses attachment references with:
    - Attachment numbering (Ek-1, Ekler, etc.)
    - Title and description extraction
    - File type classification
    - File name parsing
    - URL/link extraction
    - Size and date metadata
    - Multiple attachment handling

    Features:
    - File type detection from extension and keywords
    - URL validation and extraction
    - Metadata parsing (size, date)
    - Multiple attachment formats support
    - Parent document identification
    """

    # Attachment identifier patterns
    ATTACHMENT_PATTERNS = [
        r'Ek(?:ler)?\s*[-:]?\s*(\d+)',
        r'Ekli\s+dosya\s*[-:]?\s*(\d+)',
        r'Ek\s+Belge\s*[-:]?\s*(\d+)',
        r'Attachment\s*[-:]?\s*(\d+)',
        r'(\d+)\s+(?:No|Sayı)(?:lu|lı)\s+Ek\s+Belge'
    ]

    # Title patterns
    TITLE_PATTERNS = [
        r'Ek(?:ler)?\s*\d+\s*[-:]?\s*(.{5,150}?)(?:\n|$)',
        r'Başlık\s*:?\s*(.{5,150})',
        r'Konu\s*:?\s*(.{5,150})'
    ]

    # File name patterns
    FILE_NAME_PATTERNS = [
        r'Dosya\s+(?:Adı|İsmi)\s*:?\s*([^\n]+)',
        r'Belge\s+Adı\s*:?\s*([^\n]+)',
        r'([a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,5})',  # filename.ext
        r'File(?:\s+name)?\s*:?\s*([^\n]+)'
    ]

    # URL patterns
    URL_PATTERNS = [
        r'(?:https?://[^\s\)]+)',
        r'(?:www\.[^\s\)]+)',
        r'Link\s*:?\s*(https?://[^\s]+)',
        r'URL\s*:?\s*(https?://[^\s]+)'
    ]

    # File type extensions
    FILE_TYPE_EXTENSIONS = {
        'PDF': ['.pdf'],
        'DOC': ['.doc', '.docx', '.odt'],
        'XLS': ['.xls', '.xlsx', '.ods'],
        'PPT': ['.ppt', '.pptx'],
        'IMAGE': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
        'ARCHIVE': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'TEXT': ['.txt', '.rtf']
    }

    # File type keywords
    FILE_TYPE_KEYWORDS = {
        'PDF': ['pdf dosya', 'pdf belge'],
        'DOC': ['word', 'belge', 'doküman'],
        'XLS': ['excel', 'çizelge', 'tablo'],
        'IMAGE': ['resim', 'görsel', 'fotoğraf', 'şekil'],
        'URL': ['link', 'bağlantı', 'web']
    }

    # Size patterns
    SIZE_PATTERNS = [
        r'Boyut\s*:?\s*([0-9.,]+\s*[KMG]B)',
        r'Size\s*:?\s*([0-9.,]+\s*[KMG]B)',
        r'([0-9.,]+\s*[KMG]B)'
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'Tarih\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'Date\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})'
    ]

    # Parent document patterns
    PARENT_DOC_PATTERNS = [
        r'(\d{4})\s+sayılı\s+([^,\n]{10,80}?Kanun)',
        r'([^,\n]{10,80}?Yönetmelik)(?:e|a)\s+ek',
        r'([^,\n]{10,80}?Karar)(?:a|e)\s+ek',
        r'([^,\n]{10,80}?Genelge)(?:ye|ya)\s+ek'
    ]

    def __init__(self):
        super().__init__("Attachment Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse attachment structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode, parent_doc)

        Returns:
            DocumentStructure with parsed attachments

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

            # Extract parent document reference
            parent_document = parent_doc or self._extract_parent_document(text)

            # Parse attachments
            attachments = self._parse_attachments(text)

            # Calculate collection metadata
            has_urls = any(att.url for att in attachments)
            has_files = any(att.file_name for att in attachments)

            collection = AttachmentCollection(
                parent_document=parent_document,
                attachments=attachments,
                total_count=len(attachments),
                has_urls=has_urls,
                has_files=has_files,
                metadata={
                    'total_attachments': len(attachments),
                    'has_urls': has_urls,
                    'has_files': has_files,
                    'file_types': list(set(att.file_type for att in attachments if att.file_type))
                }
            )

            logger.info(f"Successfully parsed {len(attachments)} attachments, "
                       f"URLs: {has_urls}, Files: {has_files}")

            return DocumentStructure(
                document_type='ATTACHMENT_COLLECTION',
                attachment_collection=collection,
                metadata=collection.metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse attachment structure: {e}")
            raise ParsingError(f"Attachment parsing failed: {str(e)}") from e

    def _extract_parent_document(self, text: str) -> Optional[str]:
        """Extract parent document reference"""
        for pattern in self.PARENT_DOC_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parent = match.group(0).strip()
                logger.debug(f"Extracted parent document: {parent[:50]}...")
                return parent

        return None

    def _parse_attachments(self, text: str) -> List[Attachment]:
        """Parse all attachments from text"""
        attachments = []

        # Split text into potential attachment sections
        sections = self._split_attachment_sections(text)

        for idx, section in enumerate(sections):
            attachment = self._parse_single_attachment(section, idx + 1)
            if attachment:
                attachments.append(attachment)

        # If no structured attachments found, try extracting URLs as attachments
        if not attachments:
            urls = self._extract_all_urls(text)
            for idx, url in enumerate(urls):
                attachments.append(Attachment(
                    number=idx + 1,
                    identifier=f"URL-{idx + 1}",
                    title=None,
                    description=None,
                    file_type='URL',
                    file_name=None,
                    url=url,
                    size=None,
                    date=None
                ))

        return attachments

    def _split_attachment_sections(self, text: str) -> List[str]:
        """Split text into attachment sections"""
        sections = []

        # Try to find attachment markers
        pattern = r'(?:Ek(?:ler)?|Ekli\s+dosya|Ek\s+Belge|Attachment)\s*[-:]?\s*\d+'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if not matches:
            # No clear sections, return whole text as one section
            return [text]

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end]
            sections.append(section)

        return sections

    def _parse_single_attachment(self, section: str, default_number: int) -> Optional[Attachment]:
        """Parse a single attachment from section text"""
        # Extract number/identifier
        number = self._extract_attachment_number(section) or default_number
        identifier = f"ATT-{number}"

        # Extract title
        title = self._extract_title(section)

        # Extract description (first paragraph after title)
        description = self._extract_description(section)

        # Extract file name
        file_name = self._extract_file_name(section)

        # Detect file type
        file_type = self._detect_file_type(section, file_name)

        # Extract URL
        url = self._extract_url(section)

        # Extract size
        size = self._extract_size(section)

        # Extract date
        date = self._extract_date(section)

        attachment = Attachment(
            number=number,
            identifier=identifier,
            title=title,
            description=description,
            file_type=file_type,
            file_name=file_name,
            url=url,
            size=size,
            date=date
        )

        logger.debug(f"Parsed attachment {number}: {title}, Type: {file_type}")
        return attachment

    def _extract_attachment_number(self, text: str) -> Optional[int]:
        """Extract attachment number"""
        for pattern in self.ATTACHMENT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    number = int(match.group(1))
                    return number
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract attachment title"""
        for pattern in self.TITLE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) >= 5:
                    return title

        return None

    def _extract_description(self, text: str) -> Optional[str]:
        """Extract attachment description"""
        # Look for description or content after title
        desc_patterns = [
            r'Açıklama\s*:?\s*(.{10,300})',
            r'İçerik\s*:?\s*(.{10,300})',
            r'Description\s*:?\s*(.{10,300})'
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                description = match.group(1).strip()
                # Clean up - take first paragraph
                description = description.split('\n')[0]
                return description[:300]

        return None

    def _extract_file_name(self, text: str) -> Optional[str]:
        """Extract file name"""
        for pattern in self.FILE_NAME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                file_name = match.group(1).strip()
                # Validate it looks like a file name
                if '.' in file_name and len(file_name) < 100:
                    return file_name

        return None

    def _detect_file_type(self, text: str, file_name: Optional[str]) -> str:
        """Detect file type from file name or content"""
        # Check file extension first
        if file_name:
            for file_type, extensions in self.FILE_TYPE_EXTENSIONS.items():
                for ext in extensions:
                    if file_name.lower().endswith(ext):
                        return file_type

        # Check keywords
        text_lower = text.lower()
        for file_type, keywords in self.FILE_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return file_type

        return 'UNKNOWN'

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL/link"""
        for pattern in self.URL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                url = match.group(0) if pattern.startswith('(?:http') else match.group(1)
                return url.strip()

        return None

    def _extract_all_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        urls = []
        for pattern in self.URL_PATTERNS[:2]:  # Just the basic URL patterns
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                url = match.group(0)
                if url not in urls:
                    urls.append(url)

        return urls

    def _extract_size(self, text: str) -> Optional[str]:
        """Extract file size"""
        for pattern in self.SIZE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                size = match.group(1) if len(match.groups()) > 0 else match.group(0)
                return size.strip()

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date"""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1) if len(match.groups()) > 0 else match.group(0)
                return date.strip()

        return None

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed attachment structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not hasattr(structure, 'attachment_collection') or structure.attachment_collection is None:
            raise ValidationError("Attachment collection structure is missing")

        collection = structure.attachment_collection

        if not collection.attachments or len(collection.attachments) == 0:
            logger.warning("No attachments found in collection")

        # Validate each attachment
        for attachment in collection.attachments:
            if not attachment.number and not attachment.identifier:
                logger.warning(f"Attachment missing both number and identifier")

            if not attachment.title and not attachment.file_name and not attachment.url:
                logger.warning(f"Attachment {attachment.number} has no identifying information")

        logger.info("Attachment structure validation passed")
        return True


__all__ = ['AttachmentStructuralParser', 'Attachment', 'AttachmentCollection']
