"""Circular (Genelge) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses Genelge (Circular) document structure

Production Features:
- Circular number and date extraction
- Subject and addressee parsing
- Numbered and unnumbered paragraph extraction
- Signature block detection
- Annex handling
- Reference extraction
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
class CircularParagraph:
    """Represents a circular paragraph"""
    number: Optional[int]
    content: str
    is_numbered: bool


@dataclass
class CircularSignature:
    """Represents signature block"""
    name: Optional[str]
    title: Optional[str]
    organization: Optional[str]


class CircularStructuralParser(StructuralParser):
    """Circular (Genelge) Structural Parser

    Parses administrative circular documents with:
    - Circular number and date
    - Subject (Konu)
    - Addressee (Gereği/Muhatap)
    - Body paragraphs (numbered or unnumbered)
    - Signature block (İmza)
    - Annexes (Ek)

    Features:
    - Flexible circular number format detection
    - Turkish date parsing
    - Multi-addressee support
    - Paragraph numbering detection
    - Reference extraction
    """

    # Circular number patterns
    CIRCULAR_NUMBER_PATTERNS = [
        r'(?:Genelge|Sirküler)\s+(?:No|Sayı)\s*:?\s*(\d{4}/\d{1,4})',
        r'(\d{4}/\d{1,4})\s+[Ss]ayılı\s+[Gg]enelge',
        r'Sayı\s*:?\s*(\d{4}/\d{1,4})',
        r'No\s*:?\s*(\d{1,5})'
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'Tarih\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarih',
        r'(\d{1,2}\s+\w+\s+\d{4})'  # "15 Mayıs 2023" format
    ]

    # Subject patterns
    SUBJECT_PATTERNS = [
        r'Konu\s*:?\s*([^\n]+)',
        r'Konu\s+Başlığı\s*:?\s*([^\n]+)',
        r'İlgili\s*:?\s*([^\n]+)'
    ]

    # Addressee patterns
    ADDRESSEE_PATTERNS = [
        r'Gereği\s*:?\s*([^\n]+)',
        r'İlgi\s*:?\s*([^\n]+)',
        r'İlgililere\s*:?\s*([^\n]+)'
    ]

    def __init__(self):
        super().__init__("Circular Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse circular structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode)

        Returns:
            DocumentStructure with parsed circular

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

            # Extract circular metadata
            circular_number = self._extract_circular_number(text)
            date = self._extract_date(text)
            subject = self._extract_subject(text)
            addressees = self._extract_addressees(text)

            # Parse body paragraphs
            paragraphs = self._parse_paragraphs(text)

            # Extract signature block
            signature = self._extract_signature(text)

            # Extract references
            references = self._extract_references(text)

            # Extract annexes
            annexes = self._extract_annexes(text)

            logger.info(f"Successfully parsed circular: {circular_number}, "
                       f"{len(paragraphs)} paragraphs, {len(annexes)} annexes")

            return DocumentStructure(
                document_type='CIRCULAR',
                circular_number=circular_number,
                date=date,
                subject=subject,
                addressees=addressees,
                paragraphs=paragraphs,
                signature=signature,
                references=references,
                annexes=annexes,
                metadata={
                    'total_paragraphs': len(paragraphs),
                    'numbered_paragraphs': sum(1 for p in paragraphs if p.is_numbered),
                    'total_annexes': len(annexes)
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse circular structure: {e}")
            raise ParsingError(f"Circular parsing failed: {str(e)}") from e

    def _extract_circular_number(self, text: str) -> Optional[str]:
        """Extract circular number"""
        for pattern in self.CIRCULAR_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                logger.debug(f"Extracted circular number: {number}")
                return number

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract circular date"""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Try to parse Turkish date
                try:
                    parsed_date = parse_turkish_date(date_str)
                    logger.debug(f"Extracted date: {parsed_date}")
                    return parsed_date
                except:
                    logger.debug(f"Extracted date (unparsed): {date_str}")
                    return date_str

        return None

    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract subject (Konu)"""
        for pattern in self.SUBJECT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                logger.debug(f"Extracted subject: {subject[:50]}...")
                return subject

        return None

    def _extract_addressees(self, text: str) -> List[str]:
        """Extract addressees (recipients)"""
        addressees = []

        for pattern in self.ADDRESSEE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                addressee_text = match.group(1).strip()
                # Split by common delimiters
                parts = re.split(r'[,;]', addressee_text)
                addressees.extend([p.strip() for p in parts if p.strip()])

        logger.debug(f"Extracted {len(addressees)} addressees")
        return addressees

    def _parse_paragraphs(self, text: str) -> List[CircularParagraph]:
        """Parse body paragraphs (numbered or unnumbered)"""
        paragraphs = []

        # Try to find numbered paragraphs first
        numbered_pattern = r'(?:^|\n)\s*(\d+)[.)\-]\s+([^\n]+(?:\n(?!\s*\d+[.)\-])[^\n]+)*)'
        matches = list(re.finditer(numbered_pattern, text, re.MULTILINE))

        if matches:
            # Document has numbered paragraphs
            for match in matches:
                number = int(match.group(1))
                content = match.group(2).strip()
                paragraphs.append(CircularParagraph(
                    number=number,
                    content=content,
                    is_numbered=True
                ))
            logger.debug(f"Parsed {len(paragraphs)} numbered paragraphs")
        else:
            # Split into unnumbered paragraphs by line breaks
            # Extract main content (after metadata, before signature)
            content_start = self._find_content_start(text)
            content_end = self._find_content_end(text)

            if content_start >= 0:
                main_content = text[content_start:content_end] if content_end > 0 else text[content_start:]

                # Split by double line break or paragraph markers
                para_texts = re.split(r'\n\s*\n', main_content)

                for para_text in para_texts:
                    para_text = para_text.strip()
                    if len(para_text) > 20:  # Filter out too short segments
                        paragraphs.append(CircularParagraph(
                            number=None,
                            content=para_text,
                            is_numbered=False
                        ))

            logger.debug(f"Parsed {len(paragraphs)} unnumbered paragraphs")

        return paragraphs

    def _find_content_start(self, text: str) -> int:
        """Find where main content starts (after metadata)"""
        # Content usually starts after "Konu:" or "Gereği:"
        for pattern in [r'Konu\s*:.*?\n', r'Gereği\s*:.*?\n', r'İlgi\s*:.*?\n']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.end()

        return 0

    def _find_content_end(self, text: str) -> int:
        """Find where main content ends (before signature)"""
        # Content usually ends before signature phrases
        signature_markers = [
            r'Saygılarımla',
            r'Saygılarımızla',
            r'Bilgilerinize sunarım',
            r'Gereğini rica ederim'
        ]

        for marker in signature_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                return match.start()

        return -1

    def _extract_signature(self, text: str) -> Optional[CircularSignature]:
        """Extract signature block"""
        # Look for signature section at the end
        signature_pattern = r'(?:Saygılarımla|Saygılarımızla).*?(?:$|\n\n)'
        match = re.search(signature_pattern, text, re.DOTALL | re.IGNORECASE)

        if not match:
            return None

        signature_text = match.group(0)

        # Extract name (usually capitalized words)
        name_pattern = r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)'
        name_match = re.search(name_pattern, signature_text)
        name = name_match.group(1) if name_match else None

        # Extract title (common titles)
        title_pattern = r'((?:Genel Müdür|Başkan|Müdür|Daire Başkanı|Bakan)(?:\s+\w+)?)'
        title_match = re.search(title_pattern, signature_text, re.IGNORECASE)
        title = title_match.group(1) if title_match else None

        # Extract organization
        org_pattern = r'([A-ZÇĞİÖŞÜ][^\n]+?(?:Bakanlığı|Başkanlığı|Müdürlüğü|Kurumu))'
        org_match = re.search(org_pattern, signature_text)
        organization = org_match.group(1) if org_match else None

        logger.debug(f"Extracted signature: {name}, {title}")

        return CircularSignature(
            name=name,
            title=title,
            organization=organization
        )

    def _extract_references(self, text: str) -> List[str]:
        """Extract references to laws, regulations, other circulars"""
        references = []

        # Law references
        law_pattern = r'(\d{4})\s+sayılı\s+(?:Kanun|Yasa)'
        matches = re.finditer(law_pattern, text)
        for match in matches:
            references.append(match.group(0))

        # Regulation references
        reg_pattern = r'([^\n]+?)\s+Yönetmeliği'
        matches = re.finditer(reg_pattern, text)
        for match in matches:
            ref = match.group(0)
            if len(ref) < 100:  # Avoid too long matches
                references.append(ref)

        # Other circular references
        circ_pattern = r'(\d{4}/\d+)\s+sayılı\s+[Gg]enelge'
        matches = re.finditer(circ_pattern, text)
        for match in matches:
            references.append(match.group(0))

        # Remove duplicates
        references = list(set(references))

        logger.debug(f"Extracted {len(references)} references")
        return references[:10]  # Limit to top 10

    def _extract_annexes(self, text: str) -> List[Dict[str, Any]]:
        """Extract annexes"""
        annexes = []

        # Pattern: "EK-1", "EK: A", "EKLER"
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

                # Extract annex content preview
                annex_content = text[annex_start:annex_start + 300].strip()

                annexes.append({
                    'marker': annex_marker,
                    'content_preview': annex_content[:150]
                })

        logger.debug(f"Extracted {len(annexes)} annexes")
        return annexes

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed circular structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not structure.paragraphs or len(structure.paragraphs) == 0:
            raise ValidationError("Circular must contain at least one paragraph")

        if structure.circular_number is None:
            logger.warning("Circular number is missing")

        logger.info("Circular structure validation passed")
        return True


__all__ = ['CircularStructuralParser', 'CircularParagraph', 'CircularSignature']
