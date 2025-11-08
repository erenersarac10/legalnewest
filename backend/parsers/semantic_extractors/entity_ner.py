"""Entity NER (Named Entity Recognition) - Harvey/Legora CTO-Level Production-Grade
Extracts named entities from Turkish legal documents

Production Features:
- 10+ entity types (PERSON, ORGANIZATION, LOCATION, COURT, LAW, MONEY, etc.)
- Turkish name pattern recognition (titles, names, surnames)
- Company type recognition (A.Ş., Ltd. Şti., Koop., etc.)
- Turkish location patterns (İl, İlçe, Mahalle, Sokak)
- Legal title recognition (Av., Dr., Prof., Doç., etc.)
- Court and authority entity extraction
- Money and percentage extraction with normalization
- Date and time entity extraction
- Entity relationship hints (person-organization, court-location)
- Context-aware entity classification
- Confidence scoring and validation
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import re
import logging
from dataclasses import dataclass, field
from enum import Enum

from .base_extractor import (
    RegexExtractor,
    ExtractionResult,
    ConfidenceLevel,
    ExtractionMethod
)

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Named entity types for Turkish legal documents"""
    PERSON = "PERSON"  # Kişiler (Av. Mehmet Yılmaz)
    ORGANIZATION = "ORGANIZATION"  # Kurumlar (ABC A.Ş.)
    LOCATION = "LOCATION"  # Yerler (İstanbul, Beyoğlu İlçesi)
    COURT = "COURT"  # Mahkemeler (Ankara 2. Asliye Ceza Mahkemesi)
    LAW = "LAW"  # Kanunlar (5237 sayılı TCK)
    MONEY = "MONEY"  # Para (10.000 TL)
    PERCENT = "PERCENT"  # Yüzde (%15)
    DATE = "DATE"  # Tarihler (15.05.2020)
    TIME = "TIME"  # Zaman (14:30)
    ARTICLE = "ARTICLE"  # Maddeler (madde 15)
    CASE_NUMBER = "CASE_NUMBER"  # Dosya numaraları (2020/123 E.)


@dataclass
class EntityDetail:
    """Detailed entity information"""
    entity_type: EntityType
    text: str
    normalized_value: Optional[str] = None
    subtype: Optional[str] = None  # e.g., "COMPANY" for ORGANIZATION
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)  # Related entities


class EntityNER(RegexExtractor):
    """Entity Named Entity Recognition for Turkish Legal Documents

    Extracts and classifies named entities with:
    - Person names (with titles: Av., Dr., Prof.)
    - Organizations (A.Ş., Ltd. Şti., Koop., Dernek)
    - Locations (İl, İlçe, Mahalle, Sokak, Cadde)
    - Courts (Asliye, Bölge, Yargıtay, Danıştay)
    - Laws (6698 sayılı KVKK)
    - Money amounts (10.000 TL, 5,000 USD)
    - Percentages (%15, yüzde 20)
    - Dates and times
    - Article references
    - Case numbers

    Features:
    - Turkish character support
    - Context-aware classification
    - Entity normalization
    - Relationship extraction
    - Multi-pattern matching
    - Confidence scoring
    """

    # Turkish legal and academic titles
    TITLES = [
        'Av.',  # Avukat
        'Dr.',  # Doktor
        'Prof.',  # Profesör
        'Doç.',  # Doçent
        'Yrd. Doç.',  # Yardımcı Doçent
        'Uzm.',  # Uzman
        'Müh.',  # Mühendis
        'Mim.',  # Mimar
        'Ek.',  # Eczacı
        'Hk.',  # Hakim
        'Sav.',  # Savcı
        'Başsav.',  # Başsavcı
        'T.C.',  # Türkiye Cumhuriyeti
    ]

    # Turkish company suffixes
    COMPANY_SUFFIXES = [
        'A\\.Ş\\.',  # Anonim Şirket
        'A\\.S\\.',
        'Ltd\\. Şti\\.',  # Limited Şirket
        'LTD\\. ŞTİ\\.',
        'Ltd\\.',
        'LTD\\.',
        'Koop\\.',  # Kooperatif
        'KOOP\\.',
        'Dernek',  # Association
        'DERNEK',
        'Vakıf',  # Foundation
        'VAKIF',
        'San\\. ve Tic\\. A\\.Ş\\.',  # Sanayi ve Ticaret
        'Tic\\. A\\.Ş\\.',
        'A\\.O\\.',  # Açık Ortaklık
    ]

    # Location keywords
    LOCATION_KEYWORDS = [
        'İli?',
        'İlçesi?',
        'Mahalle(?:si)?',
        'Sokak',
        'Sokağı',
        'Cadde(?:si)?',
        'Bulvarı?',
        'Meydanı?',
        'Köyü?',
        'Kasaba(?:sı)?',
        'Belde(?:si)?',
    ]

    # Court types
    COURT_TYPES = [
        'Asliye Ceza',
        'Asliye Hukuk',
        'Asliye Ticaret',
        'Asliye İş',
        'Sulh Ceza',
        'Sulh Hukuk',
        'Ağır Ceza',
        'Bölge Adliye',
        'İdare',
        'Vergi',
        'İcra',
        'Kadastro',
    ]

    # Currency codes
    CURRENCIES = [
        'TL',
        'TRY',
        'USD',
        'EUR',
        'GBP',
        'JPY',
        '₺',
        '\\$',
        '€',
        '£',
    ]

    def __init__(self):
        # Build comprehensive pattern list
        patterns = self._build_patterns()

        super().__init__(
            name="Entity NER",
            patterns=patterns,
            version="2.0.0"
        )

    def _build_patterns(self) -> List[str]:
        """Build comprehensive NER pattern list"""
        patterns = []

        # Person patterns (with titles)
        title_pattern = '|'.join(re.escape(t) for t in self.TITLES)
        patterns.extend([
            # Title + Name + Surname (Av. Mehmet Yılmaz)
            f'(?:{title_pattern})\\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){{1,3}})',
            # Name + Surname (capital letters, 2-4 words)
            r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){1,3})\b',
        ])

        # Organization patterns
        company_suffix = '|'.join(self.COMPANY_SUFFIXES)
        patterns.extend([
            # Company name + suffix (ABC Teknoloji A.Ş.)
            f'([A-ZÇĞİÖŞÜ][A-Za-zçğıöşüÇĞİÖŞÜ0-9\\s&-]{{2,50}}(?:{company_suffix}))',
            # T.C. + Ministry/Institution
            r'T\.C\.\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:Bakanlığı|Başkanlığı|Kurumu|Kurulu))',
        ])

        # Location patterns
        location_kw = '|'.join(self.LOCATION_KEYWORDS)
        patterns.extend([
            # City/District + İl/İlçe (İstanbul İli, Beyoğlu İlçesi)
            f'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?\\s+(?:{location_kw}))',
            # Street/Avenue (Atatürk Caddesi, İstiklal Sokağı)
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)*\s+(?:Caddesi|Sokağı|Bulvarı|Meydanı))',
        ])

        # Court patterns
        court_types = '|'.join(self.COURT_TYPES)
        patterns.extend([
            # City + Number + Court Type + Mahkemesi
            f'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\\s+\\d+\\.)?\\s+(?:{court_types})\\s+Mahkeme(?:si)?)',
            # Yargıtay/Danıştay + Chamber
            r'((?:Yargıtay|Danıştay)\s+\d+\.\s+(?:Ceza|Hukuk)?\s*Daire(?:si)?)',
        ])

        # Law patterns
        patterns.extend([
            # Law number + name (6698 sayılı KVKK)
            r'(\d{4})\s+sayılı\s+([A-ZÇĞİÖŞÜ][A-Za-zçğıöşüÇĞİÖŞÜ\s]{2,50})',
            # Abbreviation (TCK, KVKK, TTK)
            r'\b([A-ZÇĞIÖŞÜ]{2,6}K)\b',
        ])

        # Money patterns
        currencies = '|'.join(self.CURRENCIES)
        patterns.extend([
            # Amount + Currency (10.000 TL, 5,000 USD)
            f'(\\d{{1,3}}(?:[.,]\\d{{3}})*(?:[.,]\\d{{1,2}})?\\s*(?:{currencies}))',
            # Currency + Amount ($ 5,000)
            f'((?:{currencies})\\s*\\d{{1,3}}(?:[.,]\\d{{3}})*(?:[.,]\\d{{1,2}})?)',
        ])

        # Percentage patterns
        patterns.extend([
            r'(%\s*\d+(?:[.,]\d+)?)',  # %15, %12.5
            r'(yüzde\s+\d+(?:[.,]\d+)?)',  # yüzde 15
        ])

        # Date patterns
        patterns.extend([
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})',  # 15.05.2020
            r'(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})',  # 15 Mayıs 2020
        ])

        # Time patterns
        patterns.extend([
            r'(\d{1,2}:\d{2}(?::\d{2})?)',  # 14:30:00
        ])

        # Article patterns
        patterns.extend([
            r'([Mm]adde\s+\d+(?:/\d+)?(?:-[a-z])?)',  # madde 15/3-a
        ])

        # Case number patterns
        patterns.extend([
            r'(\d{4}/\d+\s+E\.)',  # 2020/123 E.
            r'(\d{4}/\d+\s+K\.)',  # 2020/456 K.
            r'(Esas\s+No\s*:?\s*\d{4}/\d+)',  # Esas No: 2020/123
        ])

        return patterns

    def extract(self, text: str, **kwargs) -> List[ExtractionResult]:
        """Extract named entities from text

        Args:
            text: Input text
            **kwargs: Additional options
                - entity_type: Specific entity type to extract

        Returns:
            List of entity extraction results
        """
        entity_type_filter = kwargs.get('entity_type', None)
        results = []

        # Extract using patterns
        base_results = self.extract_with_pattern(text, **kwargs)

        # Classify and enrich each result
        for result in base_results:
            entity_detail = self._classify_entity(result.value, text, result)

            if entity_detail:
                # Filter by type if specified
                if entity_type_filter and entity_detail.entity_type != entity_type_filter:
                    continue

                result.metadata['entity_detail'] = entity_detail
                result.confidence = entity_detail.confidence
                results.append(result)

        # Remove duplicates
        results = self._deduplicate_results(results)

        # Extract relationships
        results = self._extract_relationships(results, text)

        # Sort by position
        results.sort(key=lambda r: r.start_pos if r.start_pos else 0)

        self.update_stats(success=len(results) > 0)
        logger.info(f"Extracted {len(results)} entities from text")

        return results

    def _classify_entity(self, text: str, full_text: str, result: ExtractionResult) -> Optional[EntityDetail]:
        """Classify entity type and extract details"""
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Check MONEY
        if self._is_money(text_clean):
            normalized = self._normalize_money(text_clean)
            return EntityDetail(
                entity_type=EntityType.MONEY,
                text=text_clean,
                normalized_value=normalized,
                confidence=0.95
            )

        # Check PERCENT
        if self._is_percent(text_clean):
            normalized = self._normalize_percent(text_clean)
            return EntityDetail(
                entity_type=EntityType.PERCENT,
                text=text_clean,
                normalized_value=normalized,
                confidence=0.95
            )

        # Check CASE_NUMBER
        if self._is_case_number(text_clean):
            return EntityDetail(
                entity_type=EntityType.CASE_NUMBER,
                text=text_clean,
                normalized_value=text_clean,
                confidence=0.90
            )

        # Check ARTICLE
        if 'madde' in text_lower:
            return EntityDetail(
                entity_type=EntityType.ARTICLE,
                text=text_clean,
                confidence=0.90
            )

        # Check DATE
        if self._is_date(text_clean):
            return EntityDetail(
                entity_type=EntityType.DATE,
                text=text_clean,
                confidence=0.85
            )

        # Check TIME
        if self._is_time(text_clean):
            return EntityDetail(
                entity_type=EntityType.TIME,
                text=text_clean,
                confidence=0.90
            )

        # Check COURT
        if self._is_court(text_clean):
            return EntityDetail(
                entity_type=EntityType.COURT,
                text=text_clean,
                subtype='COURT',
                confidence=0.90
            )

        # Check LAW
        if self._is_law(text_clean):
            return EntityDetail(
                entity_type=EntityType.LAW,
                text=text_clean,
                confidence=0.85
            )

        # Check LOCATION
        if self._is_location(text_clean):
            return EntityDetail(
                entity_type=EntityType.LOCATION,
                text=text_clean,
                confidence=0.75
            )

        # Check ORGANIZATION
        if self._is_organization(text_clean):
            subtype = self._get_organization_subtype(text_clean)
            return EntityDetail(
                entity_type=EntityType.ORGANIZATION,
                text=text_clean,
                subtype=subtype,
                confidence=0.80
            )

        # Check PERSON
        if self._is_person(text_clean, full_text):
            confidence = 0.75
            # Increase confidence if has title
            if any(title in text_clean for title in self.TITLES):
                confidence = 0.90
            return EntityDetail(
                entity_type=EntityType.PERSON,
                text=text_clean,
                confidence=confidence
            )

        # Default: classify as ORGANIZATION (low confidence)
        return EntityDetail(
            entity_type=EntityType.ORGANIZATION,
            text=text_clean,
            confidence=0.50
        )

    def _is_money(self, text: str) -> bool:
        """Check if text represents money amount"""
        return any(curr in text for curr in ['TL', 'USD', 'EUR', 'GBP', '₺', '$', '€', '£'])

    def _normalize_money(self, text: str) -> str:
        """Normalize money amount (e.g., '10.000 TL' -> '10000.00 TL')"""
        # Extract number and currency
        match = re.search(r'([\d.,]+)\s*([A-Z₺$€£]+)', text)
        if not match:
            return text

        amount_str = match.group(1)
        currency = match.group(2)

        # Remove thousand separators (. or ,) and normalize decimal separator
        # Turkish: 10.000,50 TL -> 10000.50 TL
        # English: 10,000.50 USD -> 10000.50 USD

        # Determine if comma is decimal separator (Turkish) or thousand (English)
        if ',' in amount_str and '.' in amount_str:
            # Both present: . is thousand, , is decimal (Turkish)
            amount_str = amount_str.replace('.', '').replace(',', '.')
        elif ',' in amount_str:
            # Only comma: check if it's decimal (Turkish) or thousand (English)
            # If there are more than 2 digits after comma, it's thousand separator
            parts = amount_str.split(',')
            if len(parts[-1]) > 2:
                amount_str = amount_str.replace(',', '')
            else:
                amount_str = amount_str.replace(',', '.')
        else:
            # Only dots: thousand separator
            amount_str = amount_str.replace('.', '')

        try:
            amount = float(amount_str)
            return f"{amount:.2f} {currency}"
        except ValueError:
            return text

    def _is_percent(self, text: str) -> bool:
        """Check if text represents percentage"""
        return '%' in text or 'yüzde' in text.lower()

    def _normalize_percent(self, text: str) -> str:
        """Normalize percentage (e.g., 'yüzde 15' -> '15%')"""
        match = re.search(r'(\d+(?:[.,]\d+)?)', text)
        if match:
            value = match.group(1).replace(',', '.')
            return f"{value}%"
        return text

    def _is_case_number(self, text: str) -> bool:
        """Check if text is a case number"""
        return bool(re.search(r'\d{4}/\d+\s+[EK]\.', text) or 'Esas No' in text)

    def _is_date(self, text: str) -> bool:
        """Check if text is a date"""
        turkish_months = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                         'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
        return (bool(re.search(r'\d{1,2}[./]\d{1,2}[./]\d{4}', text)) or
                any(month in text for month in turkish_months))

    def _is_time(self, text: str) -> bool:
        """Check if text is a time"""
        return bool(re.search(r'\d{1,2}:\d{2}', text))

    def _is_court(self, text: str) -> bool:
        """Check if text is a court name"""
        court_keywords = ['Mahkeme', 'Yargıtay', 'Danıştay', 'Daire']
        return any(kw in text for kw in court_keywords)

    def _is_law(self, text: str) -> bool:
        """Check if text is a law reference"""
        return (bool(re.search(r'\d{4}\s+sayılı', text)) or
                bool(re.search(r'\b[A-ZÇĞIÖŞÜ]{2,6}K\b', text)))

    def _is_location(self, text: str) -> bool:
        """Check if text is a location"""
        location_keywords = ['İli', 'İlçesi', 'Mahalle', 'Sokak', 'Cadde',
                            'Bulvar', 'Meydan', 'Köy']
        return any(kw in text for kw in location_keywords)

    def _is_organization(self, text: str) -> bool:
        """Check if text is an organization"""
        org_keywords = ['A.Ş.', 'Ltd.', 'Koop.', 'Dernek', 'Vakıf',
                       'Bakanlığı', 'Başkanlığı', 'Kurumu', 'Kurulu']
        return any(kw in text for kw in org_keywords)

    def _get_organization_subtype(self, text: str) -> str:
        """Get organization subtype"""
        if 'A.Ş.' in text or 'Ltd.' in text:
            return 'COMPANY'
        elif 'Bakanlığı' in text or 'Başkanlığı' in text or 'Kurumu' in text:
            return 'GOVERNMENT'
        elif 'Dernek' in text:
            return 'ASSOCIATION'
        elif 'Vakıf' in text:
            return 'FOUNDATION'
        elif 'Koop.' in text:
            return 'COOPERATIVE'
        else:
            return 'OTHER'

    def _is_person(self, text: str, full_text: str) -> bool:
        """Check if text is a person name"""
        # Must have at least 2 words (name + surname)
        words = text.split()
        if len(words) < 2:
            return False

        # Check if starts with title
        if any(text.startswith(title) for title in self.TITLES):
            return True

        # Check if all words start with capital letter (Turkish names)
        if all(word[0].isupper() for word in words if len(word) > 0):
            # Exclude if it's an organization keyword
            if self._is_organization(text):
                return False
            # Exclude if it's a location
            if self._is_location(text):
                return False
            # Likely a person name
            return True

        return False

    def _extract_relationships(self, results: List[ExtractionResult], text: str) -> List[ExtractionResult]:
        """Extract relationships between entities"""
        # Build entity index
        entities_by_type = {}
        for result in results:
            if 'entity_detail' not in result.metadata:
                continue
            entity_detail = result.metadata['entity_detail']
            entity_type = entity_detail.entity_type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(result)

        # Extract relationships
        for result in results:
            if 'entity_detail' not in result.metadata:
                continue
            entity_detail = result.metadata['entity_detail']

            # Person-Organization relationship
            if entity_detail.entity_type == EntityType.PERSON:
                # Find nearby organizations (within 100 characters)
                start = result.start_pos or 0
                nearby_orgs = [
                    org_result.metadata['entity_detail'].text
                    for org_result in entities_by_type.get(EntityType.ORGANIZATION, [])
                    if (org_result.start_pos or 0) > start - 100 and
                       (org_result.start_pos or 0) < start + 100
                ]
                if nearby_orgs:
                    entity_detail.relationships.extend(f"works_at:{org}" for org in nearby_orgs[:2])

            # Court-Location relationship
            if entity_detail.entity_type == EntityType.COURT:
                # Extract location from court name
                court_name = entity_detail.text
                # City is usually the first word
                words = court_name.split()
                if words:
                    city = words[0]
                    entity_detail.attributes['city'] = city

        return results

    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove duplicate entities"""
        if not results:
            return results

        sorted_results = sorted(results, key=lambda r: r.start_pos if r.start_pos else 0)
        deduplicated = [sorted_results[0]]

        for result in sorted_results[1:]:
            last_result = deduplicated[-1]

            # Check for overlap
            if (result.start_pos and last_result.end_pos and
                result.start_pos < last_result.end_pos):
                # Overlapping entities - keep higher confidence
                if result.confidence > last_result.confidence:
                    deduplicated[-1] = result
            else:
                # No overlap - add new entity
                deduplicated.append(result)

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} entities")
        return deduplicated


__all__ = ['EntityNER', 'EntityType', 'EntityDetail']
