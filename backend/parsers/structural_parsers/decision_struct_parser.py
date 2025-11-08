"""Court Decision Structural Parser - Harvey/Legora CTO-Level
Parses Turkish court decisions with E/K/T pattern and judgment structure
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from .base_structural_parser import BaseStructuralParser
from ..core import LegalDocument

class DecisionStructuralParser(BaseStructuralParser):
    """
    Structural parser for Turkish Court Decisions (Mahkeme Kararı).

    Court decisions have unique structure:
    - Header: Court name, chamber, E/K/T numbers, date
    - Parties: Davacı (Plaintiff), Davalı (Defendant), Müdahil (Intervener)
    - Subject: Dava konusu
    - Background: Olayın özeti
    - Arguments: Tarafların iddiaları
    - Court Analysis: Mahkemenin değerlendirmesi
    - Judgment: Hüküm
    - Legal Basis: Yasal dayanak (law articles cited)
    - Dissenting Opinion: Karşı oy (if any)

    E/K/T Pattern:
    - E: Esas (Case Number)
    - K: Karar (Decision Number)
    - T: Tarih (Date)
    """

    def __init__(self):
        super().__init__("Decision Structural Parser", "1.0.0")

    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract court decision structure.

        Returns:
            Dict with: 'header', 'esas_no', 'karar_no', 'date', 'parties',
                      'subject', 'background', 'analysis', 'judgment', 'legal_basis', 'dissent'
        """
        text = preprocessed.full_text

        # Extract header information
        header = self._extract_header(text)

        # Extract E/K/T numbers
        esas_no = self._extract_esas_number(text)
        karar_no = self._extract_karar_number(text)
        decision_date = self._extract_decision_date(text)

        # Extract parties
        parties = self._extract_parties(text)

        # Extract sections
        subject = self._extract_section(text, ['KONU', 'DAVA KONUSU'])
        background = self._extract_section(text, ['OLAY', 'OLAYLARIN ÖZETİ', 'DAVA SÜRECİ'])
        analysis = self._extract_section(text, ['DEĞERLENDİRME', 'TÜRK MİLLETİ ADINA', 'GEREKÇE'])
        judgment = self._extract_judgment(text)

        # Extract legal basis (cited laws/articles)
        legal_basis = self._extract_legal_citations(text)

        # Check for dissenting opinion
        dissent = self._extract_dissent(text)

        return {
            'header': header,
            'esas_no': esas_no,
            'karar_no': karar_no,
            'decision_date': decision_date,
            'parties': parties,
            'subject': subject,
            'background': background,
            'analysis': analysis,
            'judgment': judgment,
            'legal_basis': legal_basis,
            'dissent': dissent,
            'document_type': 'mahkeme_karari'
        }

    def _extract_header(self, text: str) -> Dict[str, Any]:
        """Extract court header (name, chamber, etc.)"""
        lines = text.split('\n')[:20]  # Header usually in first 20 lines

        court_name = None
        chamber = None

        for line in lines:
            line_upper = line.upper()

            # Yargıtay patterns
            if 'YARGITAY' in line_upper:
                match = re.search(r'YARGITAY\s+(\d+)\.\s*(HU KUK|CEZA)\s+DAİRESİ?', line_upper)
                if match:
                    chamber_num = match.group(1)
                    chamber_type = match.group(2).replace(' ', '')
                    court_name = 'Yargıtay'
                    chamber = f"{chamber_num}. {chamber_type} Dairesi"

            # Danıştay patterns
            elif 'DANIŞTAY' in line_upper or 'DANISTAY' in line_upper:
                match = re.search(r'DANIŞ?TAY\s+(\d+)\.\s*DAİRESİ?', line_upper)
                if match:
                    court_name = 'Danıştay'
                    chamber = f"{match.group(1)}. Daire"

            # Anayasa Mahkemesi
            elif 'ANAYASA MAHKEMESİ' in line_upper:
                court_name = 'Anayasa Mahkemesi'

        return {
            'court_name': court_name,
            'chamber': chamber,
            'raw_header': '\n'.join(lines[:10])
        }

    def _extract_esas_number(self, text: str) -> Optional[str]:
        """Extract E (Esas) number"""
        patterns = [
            r'E\.?\s*:?\s*(\d{4}/\d+)',
            r'ESAS\s*:?\s*(\d{4}/\d+)',
            r'Esas\s+No\s*:?\s*(\d{4}/\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_karar_number(self, text: str) -> Optional[str]:
        """Extract K (Karar) number"""
        patterns = [
            r'K\.?\s*:?\s*(\d{4}/\d+)',
            r'KARAR\s*:?\s*(\d{4}/\d+)',
            r'Karar\s+No\s*:?\s*(\d{4}/\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_decision_date(self, text: str) -> Optional[str]:
        """Extract T (Tarih) decision date"""
        from ..utils import parse_turkish_date

        patterns = [
            r'T\.?\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'TARİH\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'Karar\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_parties(self, text: str) -> Dict[str, List[str]]:
        """Extract parties (Davacı, Davalı, Müdahil)"""
        parties = {
            'plaintiff': [],  # Davacı
            'defendant': [],  # Davalı
            'intervener': []  # Müdahil
        }

        # Extract plaintiff
        plaintiff_match = re.search(r'DAVACI\s*:?\s*(.+?)(?=DAVALI|MÜDAH İL|KONU|$)', text, re.IGNORECASE | re.DOTALL)
        if plaintiff_match:
            parties['plaintiff'] = [p.strip() for p in plaintiff_match.group(1).split('\n') if p.strip()]

        # Extract defendant
        defendant_match = re.search(r'DAVALI\s*:?\s*(.+?)(?=MÜDAH İL|KONU|DAVA|$)', text, re.IGNORECASE | re.DOTALL)
        if defendant_match:
            parties['defendant'] = [d.strip() for d in defendant_match.group(1).split('\n') if d.strip()]

        # Extract intervener
        intervener_match = re.search(r'MÜDAHİL\s*:?\s*(.+?)(?=KONU|DAVA|$)', text, re.IGNORECASE | re.DOTALL)
        if intervener_match:
            parties['intervener'] = [i.strip() for i in intervener_match.group(1).split('\n') if i.strip()]

        return parties

    def _extract_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract a section by keywords"""
        for keyword in keywords:
            pattern = rf'{keyword}\s*:?\s*(.+?)(?=\n(?:[A-ZÇĞİÖŞÜ]{{3,}}|HÜKÜM|KARAR|$))'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _extract_judgment(self, text: str) -> Optional[str]:
        """Extract final judgment (HÜKÜM)"""
        patterns = [
            r'HÜKÜM\s*:?\s*(.+?)(?=\nKARŞI OY|$)',
            r'KARAR\s*:?\s*(.+?)(?=\nKARŞI OY|$)',
            r'(?:TÜRK MİLLETİ ADINA).+?((?:KABULÜNE|REDDİNE|BERAAT|MAHKÛM İYET).+?)(?=\nKARŞI OY|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _extract_legal_citations(self, text: str) -> List[str]:
        """Extract cited laws and articles"""
        from ..presets import CITATION_PATTERNS
        citations = []

        # Find all law citations
        law_pattern = re.compile(CITATION_PATTERNS['LAW_FULL'], re.IGNORECASE)
        for match in law_pattern.finditer(text):
            citations.append(match.group(0))

        # Find article references
        article_pattern = re.compile(r'(?:\d{3,5})\s+[Ss]ayılı.+?Madde\s+\d+', re.IGNORECASE)
        for match in article_pattern.finditer(text):
            citations.append(match.group(0))

        return list(set(citations))  # Remove duplicates

    def _extract_dissent(self, text: str) -> Optional[str]:
        """Extract dissenting opinion (Karşı Oy)"""
        match = re.search(r'KARŞI OY\s*:?\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

__all__ = ['DecisionStructuralParser']
