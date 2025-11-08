"""Board Decision (Kurul Kararı) Structural Parser - Harvey/Legora CTO-Level Production-Grade
Parses Turkish regulatory board/council decision structure

Production Features:
- Decision number and date extraction
- Board/council identification
- Decision type classification (approval, rejection, amendment, etc.)
- Voting results parsing
- Subject and topic extraction
- Annexes and attachments tracking
- Effective date handling
- Base law/regulation references
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from bs4 import BeautifulSoup

from ..core import StructuralParser, ParsedElement, DocumentStructure
from ..errors import ParsingError, ValidationError
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


@dataclass
class BoardMember:
    """Represents a board member"""
    name: Optional[str]
    title: Optional[str]
    vote: Optional[str]  # KABUL, RED, ÇEKİMSER


@dataclass
class VotingResult:
    """Represents voting results"""
    kabul: int  # Votes in favor
    red: int  # Votes against
    cekimser: int  # Abstentions
    toplam: int  # Total votes
    sonuc: str  # Result: KABUL/RED


@dataclass
class DecisionAnnex:
    """Represents a decision annex"""
    number: int
    title: Optional[str]
    description: Optional[str]


@dataclass
class BoardDecision:
    """Represents a complete board decision"""
    decision_number: Optional[str]
    decision_date: Optional[str]
    board_name: Optional[str]
    decision_type: Optional[str]
    subject: Optional[str]
    content: str
    voting_result: Optional[VotingResult]
    board_members: List[BoardMember]
    annexes: List[DecisionAnnex]
    effective_date: Optional[str]
    base_references: List[str]


class BoardDecisionStructuralParser(StructuralParser):
    """Board Decision (Kurul Kararı) Structural Parser

    Parses regulatory board and council decisions with:
    - Decision numbering (Karar No, Karar Sayısı)
    - Decision date and meeting date
    - Board/council identification
    - Decision type classification
    - Voting results (if present)
    - Subject and topic extraction
    - Annexes and attachments
    - Effective date
    - Base law/regulation references

    Features:
    - Multiple board/council support (SPK, BDDK, RTÜK, etc.)
    - Decision type classification (9 types)
    - Voting results extraction
    - Turkish number/date parsing
    - Annex tracking
    - Reference validation
    """

    # Board/Council patterns
    BOARD_PATTERNS = [
        r'(Sermaye Piyasası Kurulu)',
        r'(Bankacılık Düzenleme ve Denetleme Kurumu)',
        r'(Radyo ve Televizyon Üst Kurulu)',
        r'(Rekabet Kurulu)',
        r'(Bilgi Teknolojileri ve İletişim Kurumu)',
        r'(Enerji Piyasası Düzenleme Kurumu)',
        r'(Kişisel Verileri Koruma Kurulu)',
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Kurulu)',
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Kurumu)'
    ]

    # Decision number patterns
    DECISION_NUMBER_PATTERNS = [
        r'Karar\s+(?:No|Sayısı)\s*:?\s*([\d/\-]+)',
        r'(?:Sayı|No)\s*:?\s*([\d/\-]+)',
        r'([\d/\-]+)\s+sayılı\s+karar',
        r'Karar\s*:?\s*([\d/\-]+)'
    ]

    # Decision date patterns
    DECISION_DATE_PATTERNS = [
        r'Karar\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'Toplantı\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihli\s+(?:toplantı|karar)',
        r'Tarih\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})'
    ]

    # Decision types
    DECISION_TYPES = {
        'ONAY': ['onay', 'onaylan', 'uygun görül', 'kabul'],
        'RET': ['ret', 'reddedil', 'uygun görülmedi'],
        'DEĞİŞİKLİK': ['değişiklik', 'tadil', 'revize'],
        'İPTAL': ['iptal', 'kaldır'],
        'DÜZENLEME': ['düzenlem', 'tebliğ', 'yönetmelik'],
        'İZİN': ['izin', 'ruhsat', 'lisans'],
        'CEZA': ['ceza', 'para ceza', 'yaptırım', 'idari para'],
        'SORUŞTURMA': ['soruşturma', 'inceleme', 'araştırma'],
        'DİĞER': []  # Default fallback
    }

    # Voting patterns
    VOTING_PATTERNS = [
        r'Kabul\s*:?\s*(\d+)',
        r'Red\s*:?\s*(\d+)',
        r'Çekimser\s*:?\s*(\d+)',
        r'(\d+)\s+kabul.*?(\d+)\s+red',
        r'oybirliği',
        r'oy\s+(?:birliği|çokluğu)'
    ]

    # Subject patterns
    SUBJECT_PATTERNS = [
        r'Konu\s*:?\s*([^\n]{10,200})',
        r'Karar\s+Konusu\s*:?\s*([^\n]{10,200})',
        r'İlgili\s*:?\s*([^\n]{10,200})'
    ]

    # Annex patterns
    ANNEX_PATTERNS = [
        r'Ek[-\s](\d+)\s*:?\s*([^\n]{5,100})',
        r'EK\s*[-:]?\s*(\d+)',
        r'(?:Ekte|Ekli).*?(\d+)\s+adet'
    ]

    # Effective date patterns
    EFFECTIVE_DATE_PATTERNS = [
        r'Yürürlük\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'yürürlüğe\s+girer.*?(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'yayım(?:ı)?(?:nı)?\s+takip\s+eden\s+gün(?:den)?',
        r'derhal\s+yürürlüğe'
    ]

    # Base law reference patterns
    BASE_LAW_PATTERNS = [
        r'(\d{4})\s+sayılı\s+([^,\n]{10,80}?Kanun)',
        r'(\d{4})\s+sayılı\s+([^,\n]{10,80}?Yönetmelik)',
        r'([^,\n]{10,80}?Tebliğ)(?:in|ın)\s+\d+'
    ]

    def __init__(self):
        super().__init__("Board Decision Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def parse(self, content: str, **kwargs) -> DocumentStructure:
        """Parse board decision structure from text content

        Args:
            content: Raw text or HTML content
            **kwargs: Additional options (html_mode)

        Returns:
            DocumentStructure with parsed board decision

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

            # Extract decision metadata
            decision_number = self._extract_decision_number(text)
            decision_date = self._extract_decision_date(text)
            board_name = self._extract_board_name(text)
            decision_type = self._classify_decision_type(text)
            subject = self._extract_subject(text)

            # Extract voting results
            voting_result = self._extract_voting_results(text)

            # Extract board members (if listed)
            board_members = self._extract_board_members(text)

            # Extract annexes
            annexes = self._extract_annexes(text)

            # Extract effective date
            effective_date = self._extract_effective_date(text)

            # Extract base law references
            base_references = self._extract_base_references(text)

            # Create decision object
            decision = BoardDecision(
                decision_number=decision_number,
                decision_date=decision_date,
                board_name=board_name,
                decision_type=decision_type,
                subject=subject,
                content=text[:500],  # First 500 chars for preview
                voting_result=voting_result,
                board_members=board_members,
                annexes=annexes,
                effective_date=effective_date,
                base_references=base_references
            )

            logger.info(f"Successfully parsed board decision: {decision_number}, "
                       f"Board: {board_name}, Type: {decision_type}")

            return DocumentStructure(
                document_type='BOARD_DECISION',
                decision=decision,
                metadata={
                    'decision_number': decision_number,
                    'board_name': board_name,
                    'decision_type': decision_type,
                    'has_voting_results': voting_result is not None,
                    'annex_count': len(annexes),
                    'member_count': len(board_members)
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse board decision structure: {e}")
            raise ParsingError(f"Board decision parsing failed: {str(e)}") from e

    def _extract_decision_number(self, text: str) -> Optional[str]:
        """Extract decision number"""
        for pattern in self.DECISION_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1).strip()
                logger.debug(f"Extracted decision number: {number}")
                return number

        logger.warning("Decision number not found")
        return None

    def _extract_decision_date(self, text: str) -> Optional[str]:
        """Extract decision date"""
        for pattern in self.DECISION_DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
                logger.debug(f"Extracted decision date: {date_str}")
                return date_str

        logger.warning("Decision date not found")
        return None

    def _extract_board_name(self, text: str) -> Optional[str]:
        """Extract board/council name"""
        for pattern in self.BOARD_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                board_name = match.group(1).strip()
                logger.debug(f"Extracted board name: {board_name}")
                return board_name

        logger.warning("Board name not found")
        return None

    def _classify_decision_type(self, text: str) -> str:
        """Classify decision type based on content"""
        text_lower = text.lower()

        for decision_type, keywords in self.DECISION_TYPES.items():
            if decision_type == 'DİĞER':
                continue
            for keyword in keywords:
                if keyword in text_lower:
                    logger.debug(f"Classified decision type: {decision_type}")
                    return decision_type

        logger.debug("Decision type classified as: DİĞER")
        return 'DİĞER'

    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract decision subject/topic"""
        for pattern in self.SUBJECT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                logger.debug(f"Extracted subject: {subject[:50]}...")
                return subject

        # Fallback: extract from first paragraph
        lines = text.split('\n')
        for line in lines[1:5]:  # Check first few lines after title
            if len(line.strip()) > 20:
                logger.debug(f"Extracted subject from first paragraph: {line[:50]}...")
                return line.strip()[:200]

        return None

    def _extract_voting_results(self, text: str) -> Optional[VotingResult]:
        """Extract voting results if present"""
        # Check for unanimous vote
        if re.search(r'oybirliği', text, re.IGNORECASE):
            logger.debug("Detected unanimous vote (oybirliği)")
            return VotingResult(
                kabul=-1,  # -1 indicates unanimous
                red=0,
                cekimser=0,
                toplam=-1,
                sonuc='KABUL'
            )

        # Extract individual vote counts
        kabul = 0
        red = 0
        cekimser = 0

        for pattern in self.VOTING_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'kabul' in pattern.lower():
                    kabul = int(match.group(1))
                elif 'red' in pattern.lower():
                    red = int(match.group(1))
                elif 'çekimser' in pattern.lower():
                    cekimser = int(match.group(1))

        if kabul > 0 or red > 0:
            toplam = kabul + red + cekimser
            sonuc = 'KABUL' if kabul > red else 'RED'

            result = VotingResult(
                kabul=kabul,
                red=red,
                cekimser=cekimser,
                toplam=toplam,
                sonuc=sonuc
            )
            logger.debug(f"Extracted voting results: {kabul} kabul, {red} red, {cekimser} çekimser")
            return result

        return None

    def _extract_board_members(self, text: str) -> List[BoardMember]:
        """Extract board members and their votes if listed"""
        members = []

        # Pattern: Name - Title (Vote)
        member_pattern = r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\s*[-–]\s*([^(\n]+)(?:\s*\(([^)]+)\))?'
        matches = re.finditer(member_pattern, text)

        for match in matches:
            name = match.group(1).strip()
            title = match.group(2).strip() if match.group(2) else None
            vote = match.group(3).strip() if match.group(3) else None

            # Validate that this looks like a board member entry
            if title and any(keyword in title.lower() for keyword in ['başkan', 'üye', 'member']):
                members.append(BoardMember(
                    name=name,
                    title=title,
                    vote=vote
                ))
                logger.debug(f"Extracted board member: {name} - {title}")

        return members

    def _extract_annexes(self, text: str) -> List[DecisionAnnex]:
        """Extract annexes/attachments"""
        annexes = []
        seen_numbers = set()

        for pattern in self.ANNEX_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    annex_number = int(match.group(1))
                    if annex_number in seen_numbers:
                        continue

                    annex_title = match.group(2).strip() if len(match.groups()) > 1 else None

                    annexes.append(DecisionAnnex(
                        number=annex_number,
                        title=annex_title,
                        description=None
                    ))
                    seen_numbers.add(annex_number)
                    logger.debug(f"Extracted annex {annex_number}: {annex_title}")

                except (ValueError, IndexError):
                    continue

        return sorted(annexes, key=lambda x: x.number)

    def _extract_effective_date(self, text: str) -> Optional[str]:
        """Extract effective date"""
        for pattern in self.EFFECTIVE_DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) > 0:
                    effective_text = match.group(1)
                else:
                    effective_text = match.group(0)

                logger.debug(f"Extracted effective date: {effective_text}")
                return effective_text.strip()

        return None

    def _extract_base_references(self, text: str) -> List[str]:
        """Extract base law/regulation references"""
        references = []
        seen_refs = set()

        for pattern in self.BASE_LAW_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(0).strip()

                # Avoid duplicates
                if ref_text in seen_refs:
                    continue

                references.append(ref_text)
                seen_refs.add(ref_text)
                logger.debug(f"Extracted base reference: {ref_text[:50]}...")

        return references

    def validate_structure(self, structure: DocumentStructure) -> bool:
        """Validate parsed board decision structure

        Args:
            structure: Parsed document structure

        Returns:
            True if valid

        Raises:
            ValidationError: If structure is invalid
        """
        if not hasattr(structure, 'decision') or structure.decision is None:
            raise ValidationError("Board decision structure is missing")

        decision = structure.decision

        if decision.decision_number is None:
            logger.warning("Decision number is missing")

        if decision.board_name is None:
            logger.warning("Board name is missing")

        if decision.decision_date is None:
            logger.warning("Decision date is missing")

        # Validate voting results if present
        if decision.voting_result:
            result = decision.voting_result
            if result.toplam > 0:  # -1 means unanimous
                if result.kabul + result.red + result.cekimser != result.toplam:
                    logger.warning("Voting result totals don't match")

        logger.info("Board decision structure validation passed")
        return True


__all__ = ['BoardDecisionStructuralParser', 'BoardDecision', 'VotingResult', 'BoardMember', 'DecisionAnnex']
