"""Court Decision Structural Parser - Harvey/Legora CTO-Level
Parses Turkish court decisions with E/K/T pattern and comprehensive judgment structure

Production-grade implementation with:
- Multi-court support (Yargıtay, Danıştay, AYM, İlk Derece Mahkemeleri)
- Complete E/K/T number extraction and validation
- Party extraction (Davacı, Davalı, Müdahil, Temyiz Eden, Karşı Taraf)
- Full judgment structure parsing (Background, Arguments, Analysis, Judgment, Dissent)
- Legal citation extraction and classification
- Court chamber identification
- Precedent detection
- Table and exhibit extraction
- Comprehensive error handling
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from decimal import Decimal
import logging

from .base_structural_parser import BaseStructuralParser
from ..core import LegalDocument
from ..core.exceptions import ParsingError, ValidationError
from ..utils.text_utils import normalize_turkish_text
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


class DecisionStructuralParser(BaseStructuralParser):
    """
    Production-grade structural parser for Turkish Court Decisions (Mahkeme Kararı).

    Turkish Court System:
    1. **Yargıtay** (Court of Cassation): Final appeals for civil/criminal cases
       - Hukuk Daireleri (Civil Chambers): 1-23
       - Ceza Daireleri (Criminal Chambers): 1-15
       - Hukuk/Ceza Genel Kurulu (General Assembly)

    2. **Danıştay** (Council of State): Administrative law court
       - İdari Daireler (Administrative Chambers): 1-15
       - İdari Dava Daireleri (Administrative Case Chambers)

    3. **Anayasa Mahkemesi** (Constitutional Court): Constitutional review
       - Bireysel Başvuru (Individual Application)
       - İptal Davası (Annulment Case)

    4. **İlk Derece Mahkemeleri** (First Instance Courts):
       - Asliye Hukuk Mahkemesi (Civil Court)
       - Asliye Ceza Mahkemesi (Criminal Court)
       - İş Mahkemesi (Labor Court)
       - Ticaret Mahkemesi (Commercial Court)
       - İdare Mahkemesi (Administrative Court)

    E/K/T Pattern:
    - **E** (Esas): Case number - when case was opened (e.g., "2023/1234")
    - **K** (Karar): Decision number - when decision was made (e.g., "2024/567")
    - **T** (Tarih): Decision date (e.g., "15.03.2024")

    Decision Structure:
    1. **Header**: Court name, chamber, E/K/T numbers, date, judges
    2. **Parties**: Davacı (Plaintiff), Davalı (Defendant), Müdahil (Intervener)
    3. **Subject**: Dava konusu (Case subject)
    4. **Background**: Olayın özeti (Facts)
    5. **Arguments**: Tarafların iddiaları (Parties' arguments)
    6. **Court Analysis**: Mahkemenin değerlendirmesi (Court's reasoning)
    7. **Judgment**: Hüküm (Final decision - KABUL, RED, BOZMA, etc.)
    8. **Legal Basis**: Yasal dayanak (Law articles cited)
    9. **Dissent**: Karşı oy (Dissenting opinion if any)
    10. **Exhibits**: Deliller, belgeler (Evidence, documents)

    Features:
    - Multi-court detection and classification
    - Chamber identification (number and type)
    - E/K/T extraction with validation
    - Party role extraction (davacı, davalı, müdahil, temyiz eden)
    - Judge name extraction
    - Legal citation extraction (laws, regulations, precedents)
    - Judgment type classification (KABUL, RED, BOZMA, ONAMA, etc.)
    - Dissenting opinion detection
    - Cross-reference to prior decisions
    - Monetary amount extraction (tazminat, ceza, harç)
    - Comprehensive error handling and logging
    """

    # Court types and patterns
    COURT_TYPES = {
        'YARGITAY': {
            'name': 'Yargıtay',
            'type': 'cassation',
            'chambers': {
                'HUKUK': {'range': (1, 23), 'type': 'civil'},
                'CEZA': {'range': (1, 15), 'type': 'criminal'},
                'HUKUK GENEL KURULU': {'type': 'civil_general'},
                'CEZA GENEL KURULU': {'type': 'criminal_general'}
            }
        },
        'DANIŞTAY': {
            'name': 'Danıştay',
            'type': 'administrative',
            'chambers': {
                'DAİRE': {'range': (1, 15), 'type': 'administrative'}
            }
        },
        'ANAYASA MAHKEMESİ': {
            'name': 'Anayasa Mahkemesi',
            'type': 'constitutional',
            'case_types': ['Bireysel Başvuru', 'İptal Davası', 'Uyuşmazlık']
        },
        'ASLİYE': {
            'name': 'İlk Derece Mahkemesi',
            'type': 'first_instance',
            'subtypes': ['HUKUK', 'CEZA', 'TİCARET', 'İŞ', 'İDARE', 'AİLE', 'VERGİ']
        }
    }

    # Judgment types
    JUDGMENT_TYPES = {
        'KABUL': 'Accepted/Granted',
        'RED': 'Rejected/Denied',
        'BOZMA': 'Reversed (Yargıtay)',
        'ONAMA': 'Affirmed/Upheld',
        'DÜZELTİLMİŞ KARAR': 'Corrected Decision',
        'BERAAT': 'Acquittal (Criminal)',
        'MAHKÛMIYET': 'Conviction (Criminal)',
        'HÜKMÜN AÇIKLANMASININ GERİ BIRAKILMASI': 'HAGB - Deferred Announcement',
        'CEZA VERİLMESİNE YER OLMADIĞINA': 'No Grounds for Penalty',
        'TEDBİR': 'Provisional Measure',
        'YÜRÜTMENİN DURDURULMASI': 'Stay of Execution (Administrative)'
    }

    # Party roles
    PARTY_ROLES = {
        'davacı': 'Plaintiff',
        'davalı': 'Defendant',
        'müdahil': 'Intervener',
        'temyiz eden': 'Appellant (Cassation)',
        'karşı taraf': 'Opposing Party',
        'sanık': 'Defendant (Criminal)',
        'mağdur': 'Victim',
        'katılan': 'Intervenor (Criminal)',
        'başvurucu': 'Applicant (Constitutional Court)'
    }

    def __init__(self):
        super().__init__("Decision Structural Parser", "2.0.0")
        logger.info(f"Initialized {self.name} v{self.version}")

    def _extract_raw_data(self, preprocessed: LegalDocument, **kwargs) -> Dict[str, Any]:
        """
        Extract complete court decision structure.

        Returns:
            Dict with: 'court_info', 'esas_no', 'karar_no', 'decision_date', 'parties',
                      'subject', 'background', 'arguments', 'analysis', 'judgment',
                      'judgment_type', 'legal_basis', 'dissent', 'judges', 'exhibits',
                      'amounts', 'precedents', 'cross_references'
        """
        text = preprocessed.full_text
        title = preprocessed.title if hasattr(preprocessed, 'title') else ''

        try:
            # Extract court information
            court_info = self._extract_court_info(text, title)

            # Extract E/K/T numbers
            esas_no = self._extract_esas_number(text)
            karar_no = self._extract_karar_number(text)
            decision_date = self._extract_decision_date(text)

            # Validate E/K/T pattern
            ekt_valid = self._validate_ekt_pattern(esas_no, karar_no, decision_date)

            # Extract parties
            parties = self._extract_parties(text, court_info.get('type'))

            # Extract judges/panel
            judges = self._extract_judges(text)

            # Extract case subject
            subject = self._extract_section(text, ['KONU', 'DAVA KONUSU', 'İSTEM'])

            # Extract background/facts
            background = self._extract_section(text, [
                'OLAY', 'OLAYLARIN ÖZETİ', 'DAVA SÜRECİ', 'OLAYLAR'
            ])

            # Extract arguments
            arguments = self._extract_arguments(text)

            # Extract court's analysis/reasoning
            analysis = self._extract_section(text, [
                'DEĞERLENDİRME', 'TÜRK MİLLETİ ADINA', 'GEREKÇE',
                'KARARIN GEREKÇESİ', 'MAHKEMENİN DEĞERLENDİRMESİ'
            ])

            # Extract judgment/verdict
            judgment = self._extract_judgment(text)
            judgment_type = self._classify_judgment(judgment, text) if judgment else None

            # Extract legal basis (cited laws/articles/precedents)
            legal_basis = self._extract_legal_citations(text)

            # Extract precedent references
            precedents = self._extract_precedent_references(text)

            # Extract dissenting opinion
            dissent = self._extract_dissent(text)

            # Extract exhibits/evidence
            exhibits = self._extract_exhibits(text)

            # Extract monetary amounts
            amounts = self._extract_amounts(text)

            # Extract cross-references
            cross_references = self._extract_cross_references(text)

            # Extract tables
            tables = self._extract_tables(text)

            logger.info(
                f"Extracted decision: {court_info.get('name')} - E:{esas_no} K:{karar_no} - "
                f"{len(parties)} parties, {len(legal_basis)} citations, "
                f"judgment={judgment_type}"
            )

            return {
                'court_info': court_info,
                'esas_no': esas_no,
                'karar_no': karar_no,
                'decision_date': decision_date,
                'ekt_valid': ekt_valid,
                'parties': parties,
                'judges': judges,
                'subject': subject,
                'background': background,
                'arguments': arguments,
                'analysis': analysis,
                'judgment': judgment,
                'judgment_type': judgment_type,
                'legal_basis': legal_basis,
                'precedents': precedents,
                'dissent': dissent,
                'exhibits': exhibits,
                'amounts': amounts,
                'cross_references': cross_references,
                'tables': tables,
                'document_type': 'mahkeme_karari'
            }

        except Exception as e:
            logger.error(f"Error extracting decision structure: {e}")
            raise ParsingError(f"Failed to extract decision structure: {e}")

    def _extract_court_info(self, text: str, title: str) -> Dict[str, Any]:
        """
        Extract court name, chamber, type from header.

        Returns:
            Dict with 'name', 'type', 'chamber', 'chamber_type', 'raw_header'
        """
        lines = (title + '\n' + text).split('\n')[:30]  # Check first 30 lines
        header_text = '\n'.join(lines).upper()

        court_info = {
            'name': None,
            'type': None,
            'chamber': None,
            'chamber_number': None,
            'chamber_type': None,
            'raw_header': '\n'.join(lines[:15])
        }

        # Yargıtay patterns
        if 'YARGITAY' in header_text:
            court_info['name'] = 'Yargıtay'
            court_info['type'] = 'cassation'

            # Extract chamber
            chamber_patterns = [
                r'YARGITAY\s+(\d+)\.\s*(HUKUK|CEZA)\s+DAİRESİ',
                r'YARGITAY\s+(HUKUK|CEZA)\s+GENEL\s+KURULU'
            ]

            for pattern in chamber_patterns:
                match = re.search(pattern, header_text)
                if match:
                    if match.group(1).isdigit():
                        court_info['chamber_number'] = int(match.group(1))
                        court_info['chamber_type'] = match.group(2).lower()
                        court_info['chamber'] = f"{match.group(1)}. {match.group(2).title()} Dairesi"
                    else:
                        court_info['chamber_type'] = match.group(1).lower() + '_genel_kurulu'
                        court_info['chamber'] = f"{match.group(1).title()} Genel Kurulu"
                    break

        # Danıştay patterns
        elif 'DANIŞTAY' in header_text or 'DANISTAY' in header_text:
            court_info['name'] = 'Danıştay'
            court_info['type'] = 'administrative'

            match = re.search(r'DANIŞ?TAY\s+(\d+)\.\s*DAİRE', header_text)
            if match:
                court_info['chamber_number'] = int(match.group(1))
                court_info['chamber'] = f"{match.group(1)}. Daire"
                court_info['chamber_type'] = 'administrative'

        # Anayasa Mahkemesi
        elif 'ANAYASA MAHKEMESİ' in header_text or 'ANAYASA MK' in header_text:
            court_info['name'] = 'Anayasa Mahkemesi'
            court_info['type'] = 'constitutional'

            # Check case type
            if 'BİREYSEL BAŞVURU' in header_text:
                court_info['case_type'] = 'Bireysel Başvuru'
            elif 'İPTAL' in header_text:
                court_info['case_type'] = 'İptal Davası'

        # İlk Derece Mahkemeleri (First Instance Courts)
        elif any(kw in header_text for kw in ['MAHKEMESİ', 'MAHK.']):
            court_info['type'] = 'first_instance'

            # Determine subtype
            if 'ASLİYE HUKUK' in header_text:
                court_info['name'] = 'Asliye Hukuk Mahkemesi'
                court_info['subtype'] = 'civil'
            elif 'ASLİYE CEZA' in header_text:
                court_info['name'] = 'Asliye Ceza Mahkemesi'
                court_info['subtype'] = 'criminal'
            elif 'İŞ MAHKEMESİ' in header_text:
                court_info['name'] = 'İş Mahkemesi'
                court_info['subtype'] = 'labor'
            elif 'TİCARET MAHKEMESİ' in header_text:
                court_info['name'] = 'Ticaret Mahkemesi'
                court_info['subtype'] = 'commercial'
            elif 'İDARE MAHKEMESİ' in header_text:
                court_info['name'] = 'İdare Mahkemesi'
                court_info['subtype'] = 'administrative'
            elif 'AİLE MAHKEMESİ' in header_text:
                court_info['name'] = 'Aile Mahkemesi'
                court_info['subtype'] = 'family'
            elif 'VERGİ MAHKEMESİ' in header_text:
                court_info['name'] = 'Vergi Mahkemesi'
                court_info['subtype'] = 'tax'
            else:
                court_info['name'] = 'İlk Derece Mahkemesi'

        logger.debug(f"Extracted court info: {court_info['name']} - {court_info.get('chamber')}")
        return court_info

    def _extract_esas_number(self, text: str) -> Optional[str]:
        """
        Extract E (Esas) number - case file number.

        Patterns: "E. 2023/1234", "ESAS: 2023/1234", "Esas No: 2023/1234"
        """
        patterns = [
            r'E\.?\s*:?\s*(\d{4}/\d+)',
            r'ESAS\s*:?\s*(\d{4}/\d+)',
            r'Esas\s+(?:No|Sayısı)\s*:?\s*(\d{4}/\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                logger.debug(f"Extracted Esas number: {match.group(1)}")
                return match.group(1)
        return None

    def _extract_karar_number(self, text: str) -> Optional[str]:
        """
        Extract K (Karar) number - decision number.

        Patterns: "K. 2024/567", "KARAR: 2024/567", "Karar No: 2024/567"
        """
        patterns = [
            r'K\.?\s*:?\s*(\d{4}/\d+)',
            r'KARAR\s*:?\s*(\d{4}/\d+)',
            r'Karar\s+(?:No|Sayısı)\s*:?\s*(\d{4}/\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                logger.debug(f"Extracted Karar number: {match.group(1)}")
                return match.group(1)
        return None

    def _extract_decision_date(self, text: str) -> Optional[str]:
        """
        Extract T (Tarih) decision date.

        Patterns: "T. 15.03.2024", "TARİH: 15/03/2024", "Karar Tarihi: 15.03.2024"
        """
        patterns = [
            r'T\.?\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'TARİH\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'Karar\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                logger.debug(f"Extracted decision date: {match.group(1)}")
                return match.group(1)
        return None

    def _validate_ekt_pattern(self, esas: Optional[str], karar: Optional[str], tarih: Optional[str]) -> bool:
        """
        Validate E/K/T pattern consistency.

        E year should be <= K year (case opened before/same year as decision)
        """
        if not (esas and karar):
            return False

        try:
            esas_year = int(esas.split('/')[0])
            karar_year = int(karar.split('/')[0])

            if esas_year > karar_year:
                logger.warning(f"Invalid E/K pattern: E={esas} K={karar} (Esas year > Karar year)")
                return False

            return True
        except:
            return False

    def _extract_parties(self, text: str, court_type: Optional[str]) -> Dict[str, List[str]]:
        """
        Extract parties based on court type.

        Civil: Davacı, Davalı, Müdahil
        Criminal: Sanık, Mağdur, Katılan
        Cassation: Temyiz Eden, Karşı Taraf
        Constitutional: Başvurucu
        """
        parties = {}

        # Civil/Commercial parties
        plaintiff_match = re.search(
            r'DAVACI\s*:?\s*(.+?)(?=DAVALI|MÜDAH|KONU|DAVA|İSTEM|$)',
            text[:2000], re.IGNORECASE | re.DOTALL
        )
        if plaintiff_match:
            parties['plaintiff'] = self._parse_party_list(plaintiff_match.group(1))

        defendant_match = re.search(
            r'DAVALI\s*:?\s*(.+?)(?=MÜDAH|KONU|DAVA|İSTEM|$)',
            text[:2000], re.IGNORECASE | re.DOTALL
        )
        if defendant_match:
            parties['defendant'] = self._parse_party_list(defendant_match.group(1))

        intervener_match = re.search(
            r'MÜDAHİL\s*:?\s*(.+?)(?=KONU|DAVA|İSTEM|$)',
            text[:2000], re.IGNORECASE | re.DOTALL
        )
        if intervener_match:
            parties['intervener'] = self._parse_party_list(intervener_match.group(1))

        # Cassation parties
        appellant_match = re.search(
            r'TEMYİZ\s+EDEN\s*:?\s*(.+?)(?=KARŞI|DAVA|$)',
            text[:2000], re.IGNORECASE | re.DOTALL
        )
        if appellant_match:
            parties['appellant'] = self._parse_party_list(appellant_match.group(1))

        opposing_match = re.search(
            r'KARŞI\s+TARAF\s*:?\s*(.+?)(?=DAVA|KONU|$)',
            text[:2000], re.IGNORECASE | re.DOTALL
        )
        if opposing_match:
            parties['opposing'] = self._parse_party_list(opposing_match.group(1))

        # Criminal parties
        if court_type in ['criminal', 'cassation']:
            defendant_crim_match = re.search(
                r'SANIK\s*:?\s*(.+?)(?=MAĞDUR|KATILAN|SUÇ|$)',
                text[:2000], re.IGNORECASE | re.DOTALL
            )
            if defendant_crim_match:
                parties['defendant_criminal'] = self._parse_party_list(defendant_crim_match.group(1))

            victim_match = re.search(
                r'MAĞDUR\s*:?\s*(.+?)(?=KATILAN|SUÇ|$)',
                text[:2000], re.IGNORECASE | re.DOTALL
            )
            if victim_match:
                parties['victim'] = self._parse_party_list(victim_match.group(1))

        # Constitutional Court
        if court_type == 'constitutional':
            applicant_match = re.search(
                r'BAŞVURUCU\s*:?\s*(.+?)(?=KONU|İSTEM|$)',
                text[:2000], re.IGNORECASE | re.DOTALL
            )
            if applicant_match:
                parties['applicant'] = self._parse_party_list(applicant_match.group(1))

        logger.debug(f"Extracted {len(parties)} party types with {sum(len(v) for v in parties.values())} total parties")
        return parties

    def _parse_party_list(self, text: str) -> List[str]:
        """Parse a party list (may be multiline, comma-separated)"""
        # Split by newlines and commas
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            # Skip metadata lines
            if any(kw in line.upper() for kw in ['KONU', 'DAVA', 'İSTEM', 'OLAYLAR']):
                break
            items.append(line)

        return items[:10]  # Limit to 10 parties to avoid runaway extraction

    def _extract_judges(self, text: str) -> List[str]:
        """
        Extract judge names from panel/header.

        Pattern: "Üye: Name SURNAME, Name2 SURNAME2"
        """
        judges = []

        patterns = [
            r'Üye\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ]+)',
            r'Başkan\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ]+)',
            r'Raportör\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ]+)'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text[:1500]):
                judge_name = match.group(1).strip()
                if judge_name not in judges:
                    judges.append(judge_name)

        return judges

    def _extract_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract a section by keywords"""
        for keyword in keywords:
            pattern = rf'{keyword}\s*:?\s*(.+?)(?=\n(?:[A-ZÇĞİÖŞÜ]{{3,}}|HÜKÜM|KARAR|GEREKÇE|$))'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if len(content) > 20:  # Meaningful content
                    return content
        return None

    def _extract_arguments(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract arguments from each party.

        Returns dict with 'plaintiff_arg', 'defendant_arg'
        """
        arguments = {}

        # Plaintiff/Appellant arguments
        plaintiff_arg = self._extract_section(text, [
            'DAVACININ İDDİASI', 'BAŞVURUCUNUN İDDİALARI',
            'TEMYİZ EDENIN İDDİALARI', 'İDDİA'
        ])
        if plaintiff_arg:
            arguments['plaintiff'] = plaintiff_arg

        # Defendant arguments
        defendant_arg = self._extract_section(text, [
            'DAVALININ SAVUNMASI', 'KARŞI TARAFIN SAVUNMASI', 'SAVUNMA'
        ])
        if defendant_arg:
            arguments['defendant'] = defendant_arg

        return arguments

    def _extract_judgment(self, text: str) -> Optional[str]:
        """Extract final judgment (HÜKÜM)"""
        patterns = [
            r'HÜKÜM\s*:?\s*(.+?)(?=\nKARŞI OY|$)',
            r'(?:TÜRK MİLLETİ ADINA).+?((?:KABULÜNE|REDDİNE|BOZULMASINA|ONANMASINA|BERAAT|MAHKÛM).+?)(?=\nKARŞI OY|İMZA|$)',
            r'KARAR\s*:?\s*(.+?)(?=\nKARŞI OY|İMZA|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                judgment = match.group(1).strip()
                if len(judgment) > 10:
                    return judgment
        return None

    def _classify_judgment(self, judgment_text: str, full_text: str) -> Optional[str]:
        """
        Classify judgment type based on text.

        Returns: KABUL, RED, BOZMA, ONAMA, etc.
        """
        text_upper = (judgment_text + ' ' + full_text[:3000]).upper()

        for judgment_type in self.JUDGMENT_TYPES.keys():
            if judgment_type in text_upper:
                logger.debug(f"Classified judgment as: {judgment_type}")
                return judgment_type

        return None

    def _extract_legal_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract cited laws, regulations, and articles.

        Returns list of citation dicts with 'type', 'law_number', 'article', 'text'
        """
        citations = []

        # Law citations with article
        law_pattern = r'(\d{3,5})\s+sayılı\s+([^\'\"]+?)(?:Kanun|kanun)(?:un|\'un)?\s+(\d+)\s*(?:inci|nci|üncü|ncü)?\s+madde'
        for match in re.finditer(law_pattern, text, re.IGNORECASE):
            citations.append({
                'type': 'law',
                'law_number': match.group(1),
                'law_name': match.group(2).strip(),
                'article': match.group(3),
                'text': match.group(0)
            })

        # General law references
        gen_law_pattern = r'(\d{3,5})\s+sayılı\s+([^\'\"]+?)(?:Kanun|kanun)'
        for match in re.finditer(gen_law_pattern, text, re.IGNORECASE):
            if not any(c['text'] == match.group(0) for c in citations):
                citations.append({
                    'type': 'law',
                    'law_number': match.group(1),
                    'law_name': match.group(2).strip(),
                    'text': match.group(0)
                })

        logger.debug(f"Extracted {len(citations)} legal citations")
        return citations

    def _extract_precedent_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract references to prior court decisions (precedents).

        Pattern: "Yargıtay 5. Hukuk Dairesi'nin 2020/1234 E., 2021/567 K. sayılı kararı"
        """
        precedents = []

        patterns = [
            r'Yargıtay\s+(\d+)\.\s+(Hukuk|Ceza)\s+Dairesi(?:\'nin|nin)?\s+(\d{4}/\d+)\s+E\.?,\s+(\d{4}/\d+)\s+K\.',
            r'Danıştay\s+(\d+)\.\s+Daire(?:\'nin|nin)?\s+(\d{4}/\d+)\s+E\.?,\s+(\d{4}/\d+)\s+K\.',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                precedents.append({
                    'court': 'Yargıtay' if 'Yargıtay' in match.group(0) else 'Danıştay',
                    'chamber': match.group(1),
                    'esas': match.group(3) if 'Yargıtay' in match.group(0) else match.group(2),
                    'karar': match.group(4) if 'Yargıtay' in match.group(0) else match.group(3),
                    'text': match.group(0)
                })

        return precedents

    def _extract_dissent(self, text: str) -> Optional[str]:
        """Extract dissenting opinion (Karşı Oy)"""
        match = re.search(r'KARŞI OY\s*:?(.+?)(?=İMZA|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            dissent = match.group(1).strip()
            return dissent if len(dissent) > 20 else None
        return None

    def _extract_exhibits(self, text: str) -> List[str]:
        """
        Extract exhibits/evidence mentioned.

        Pattern: "Ek-1", "Delil 1", "Bilirkişi Raporu"
        """
        exhibits = []

        patterns = [
            r'Ek[-\s](\d+|[A-Z])',
            r'Delil\s+(\d+)',
            r'(Bilirkişi\s+Raporu)',
            r'(Keşif\s+Raporu)',
            r'(Fotoğraf)',
            r'(Belge\s+\d+)'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                exhibit = match.group(0)
                if exhibit not in exhibits:
                    exhibits.append(exhibit)

        return exhibits[:20]  # Limit to 20

    def _extract_amounts(self, text: str) -> Dict[str, Any]:
        """
        Extract monetary amounts (tazminat, ceza, harç).

        Turkish format: 1.234.567,89 TL
        """
        amounts = {}

        # Tazminat (compensation)
        tazminat_pattern = r'tazminat.*?([\d.]+,\d{2})\s*(?:TL|₺)'
        match = re.search(tazminat_pattern, text, re.IGNORECASE)
        if match:
            amounts['compensation'] = self._parse_turkish_decimal(match.group(1))

        # Para cezası (fine)
        ceza_pattern = r'para\s+cezası.*?([\d.]+,\d{2})\s*(?:TL|₺)'
        match = re.search(ceza_pattern, text, re.IGNORECASE)
        if match:
            amounts['fine'] = self._parse_turkish_decimal(match.group(1))

        # Vekalet ücreti (attorney fees)
        vekalet_pattern = r'vekalet\s+ücreti.*?([\d.]+,\d{2})\s*(?:TL|₺)'
        match = re.search(vekalet_pattern, text, re.IGNORECASE)
        if match:
            amounts['attorney_fees'] = self._parse_turkish_decimal(match.group(1))

        return amounts

    def _parse_turkish_decimal(self, amount_str: str) -> Optional[Decimal]:
        """Convert Turkish format number to Decimal: 1.234.567,89 → 1234567.89"""
        try:
            # Remove thousand separators, replace decimal comma
            clean = amount_str.replace('.', '').replace(',', '.')
            return Decimal(clean)
        except:
            return None

    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references to other articles/paragraphs"""
        references = []

        patterns = [
            r'(\d+)\s*(?:inci|nci|üncü|ncü)?\s+maddede',
            r'(\d+)\s*(?:inci|nci|üncü|ncü)?\s+fıkra',
            r'([a-z])\s+bendinde'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref = match.group(0)
                if ref not in references:
                    references.append(ref)

        return references[:30]  # Limit

    def _extract_tables(self, text: str) -> List[str]:
        """Extract tables from decision text"""
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
                if len(table_lines) >= 2:
                    tables.append('\n'.join(table_lines))
                table_lines = []
                in_table = False

        if len(table_lines) >= 2:
            tables.append('\n'.join(table_lines))

        return tables


__all__ = ['DecisionStructuralParser']
