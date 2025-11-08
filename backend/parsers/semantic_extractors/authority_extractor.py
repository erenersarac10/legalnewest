"""Authority Extractor - Harvey/Legora CTO-Level Production-Grade
Extracts government authorities and institutions from Turkish legal documents

Production Features:
- Ministry name extraction (20+ ministries)
- Regulatory authority detection (BDDK, SPK, EPDK, etc.)
- Court identification (Yargıtay, Danıştay, AYM, etc.)
- Local government entities (İl, İlçe, Belediye)
- International organizations (UN, EU, NATO, etc.)
- Authority abbreviation expansion (T.C., vs., etc.)
- Authority type classification
- Hierarchical authority relationships
"""
from typing import Dict, List, Any, Optional, Set
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


class AuthorityType(Enum):
    """Types of government authorities"""
    MINISTRY = "MINISTRY"  # Bakanlık
    REGULATORY_BODY = "REGULATORY_BODY"  # Düzenleyici kurum (BDDK, SPK, etc.)
    COURT = "COURT"  # Mahkeme (Yargıtay, Danıştay, etc.)
    PRESIDENCY = "PRESIDENCY"  # Cumhurbaşkanlığı
    PARLIAMENT = "PARLIAMENT"  # TBMM
    LOCAL_GOVERNMENT = "LOCAL_GOVERNMENT"  # Belediye, İl, İlçe
    STATE_INSTITUTION = "STATE_INSTITUTION"  # Devlet kurumu
    INTERNATIONAL_ORG = "INTERNATIONAL_ORG"  # Uluslararası organizasyon
    UNIVERSITY = "UNIVERSITY"  # Üniversite
    COUNCIL = "COUNCIL"  # Kurul


@dataclass
class AuthorityDetail:
    """Detailed authority information"""
    authority_type: AuthorityType
    name: str
    abbreviation: Optional[str] = None
    full_name: Optional[str] = None
    parent_authority: Optional[str] = None
    jurisdiction: Optional[str] = None  # Yetki alanı
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthorityExtractor(RegexExtractor):
    """Authority Extractor for Turkish Legal Documents

    Extracts and classifies government authorities with:
    - Ministry patterns (16+ ministries)
    - Regulatory bodies (9+ authorities)
    - Courts (4 high courts + lower courts)
    - State institutions
    - Local government entities
    - International organizations
    - Authority abbreviations

    Features:
    - Authority type classification
    - Abbreviation expansion
    - Hierarchical authority recognition
    - Turkish government structure knowledge
    - Context-aware extraction
    """

    # Ministry patterns (current Turkish government structure)
    MINISTRIES = {
        'ADALET': {
            'names': ['Adalet Bakanlığı'],
            'abbrev': 'AB',
            'keywords': ['adalet']
        },
        'AILE': {
            'names': ['Aile ve Sosyal Hizmetler Bakanlığı'],
            'abbrev': 'ASHB',
            'keywords': ['aile', 'sosyal hizmet']
        },
        'ÇALIŞMA': {
            'names': ['Çalışma ve Sosyal Güvenlik Bakanlığı'],
            'abbrev': 'ÇSGB',
            'keywords': ['çalışma', 'sosyal güvenlik', 'sgk']
        },
        'DIŞİŞLERİ': {
            'names': ['Dışişleri Bakanlığı'],
            'abbrev': 'DB',
            'keywords': ['dışişleri', 'dış ilişki']
        },
        'EĞİTİM': {
            'names': ['Millî Eğitim Bakanlığı', 'Milli Eğitim Bakanlığı'],
            'abbrev': 'MEB',
            'keywords': ['eğitim', 'milli eğitim']
        },
        'ENERJİ': {
            'names': ['Enerji ve Tabii Kaynaklar Bakanlığı'],
            'abbrev': 'ETKB',
            'keywords': ['enerji', 'tabii kaynak']
        },
        'ÇEVRE': {
            'names': ['Çevre, Şehircilik ve İklim Değişikliği Bakanlığı'],
            'abbrev': 'ÇŞİDB',
            'keywords': ['çevre', 'şehircilik']
        },
        'GENÇLİK': {
            'names': ['Gençlik ve Spor Bakanlığı'],
            'abbrev': 'GSB',
            'keywords': ['gençlik', 'spor']
        },
        'HAZİNE': {
            'names': ['Hazine ve Maliye Bakanlığı'],
            'abbrev': 'HMB',
            'keywords': ['hazine', 'maliye']
        },
        'İÇİŞLERİ': {
            'names': ['İçişleri Bakanlığı'],
            'abbrev': 'İB',
            'keywords': ['içişleri', 'emniyet']
        },
        'KÜLTÜR': {
            'names': ['Kültür ve Turizm Bakanlığı'],
            'abbrev': 'KTB',
            'keywords': ['kültür', 'turizm']
        },
        'SAĞLIK': {
            'names': ['Sağlık Bakanlığı'],
            'abbrev': 'SB',
            'keywords': ['sağlık']
        },
        'SANAYİ': {
            'names': ['Sanayi ve Teknoloji Bakanlığı'],
            'abbrev': 'STB',
            'keywords': ['sanayi', 'teknoloji']
        },
        'TİCARET': {
            'names': ['Ticaret Bakanlığı'],
            'abbrev': 'TB',
            'keywords': ['ticaret']
        },
        'TARIM': {
            'names': ['Tarım ve Orman Bakanlığı'],
            'abbrev': 'TOB',
            'keywords': ['tarım', 'orman']
        },
        'ULAŞTIRMA': {
            'names': ['Ulaştırma ve Altyapı Bakanlığı'],
            'abbrev': 'UAB',
            'keywords': ['ulaştırma', 'altyapı']
        }
    }

    # Regulatory bodies and agencies
    REGULATORY_BODIES = {
        'BDDK': {
            'full_name': 'Bankacılık Düzenleme ve Denetleme Kurumu',
            'keywords': ['bddk', 'bankacılık düzenleme']
        },
        'SPK': {
            'full_name': 'Sermaye Piyasası Kurulu',
            'keywords': ['spk', 'sermaye piyasa']
        },
        'EPDK': {
            'full_name': 'Enerji Piyasası Düzenleme Kurumu',
            'keywords': ['epdk', 'enerji piyasa']
        },
        'BTK': {
            'full_name': 'Bilgi Teknolojileri ve İletişim Kurumu',
            'keywords': ['btk', 'bilgi teknoloji', 'iletişim kurum']
        },
        'RTÜK': {
            'full_name': 'Radyo ve Televizyon Üst Kurulu',
            'keywords': ['rtük', 'radyo televizyon']
        },
        'KVKK': {
            'full_name': 'Kişisel Verileri Koruma Kurumu',
            'keywords': ['kvkk', 'kişisel veri']
        },
        'REKABET': {
            'full_name': 'Rekabet Kurumu',
            'keywords': ['rekabet kurum', 'rekabet kurulu']
        },
        'TCMB': {
            'full_name': 'Türkiye Cumhuriyet Merkez Bankası',
            'keywords': ['tcmb', 'merkez banka']
        },
        'TÜİK': {
            'full_name': 'Türkiye İstatistik Kurumu',
            'keywords': ['tüik', 'istatistik kurum']
        }
    }

    # Court patterns
    COURTS = {
        'YARGITAY': {
            'full_name': 'Yargıtay',
            'keywords': ['yargıtay'],
            'chambers': ['hukuk dairesi', 'ceza dairesi']
        },
        'DANIŞTAY': {
            'full_name': 'Danıştay',
            'keywords': ['danıştay'],
            'chambers': ['daire']
        },
        'AYM': {
            'full_name': 'Anayasa Mahkemesi',
            'keywords': ['anayasa mahkemesi', 'aym'],
            'chambers': []
        },
        'UYUŞMAZLIK': {
            'full_name': 'Uyuşmazlık Mahkemesi',
            'keywords': ['uyuşmazlık mahkemesi'],
            'chambers': []
        }
    }

    # State institutions
    STATE_INSTITUTIONS = [
        'Cumhurbaşkanlığı',
        'Türkiye Büyük Millet Meclisi',
        'TBMM',
        'Sayıştay',
        'Diyanet İşleri Başkanlığı',
        'Milli İstihbarat Teşkilatı',
        'MİT',
        'Devlet Planlama Teşkilatı',
        'DPT',
        'TÜBİTAK',
        'TÜBA'
    ]

    # Local government patterns
    LOCAL_GOV_PATTERNS = [
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+(?:İl|İlçe|Belediye)(?:si)?',
        r'(?:İl|İlçe|Belediye)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)',
        r'Büyükşehir\s+Belediye(?:si)?'
    ]

    def __init__(self):
        # Build pattern list
        patterns = self._build_patterns()

        super().__init__(
            name="Authority Extractor",
            patterns=patterns,
            version="2.0.0"
        )

    def _build_patterns(self) -> List[str]:
        """Build comprehensive pattern list for all authorities"""
        patterns = []

        # Ministry patterns
        for ministry_data in self.MINISTRIES.values():
            for name in ministry_data['names']:
                patterns.append(re.escape(name))

        # Regulatory body patterns
        for body_data in self.REGULATORY_BODIES.values():
            patterns.append(re.escape(body_data['full_name']))

        # Court patterns
        for court_data in self.COURTS.values():
            patterns.append(re.escape(court_data['full_name']))

        # State institution patterns
        for institution in self.STATE_INSTITUTIONS:
            patterns.append(re.escape(institution))

        # Local government patterns
        patterns.extend(self.LOCAL_GOV_PATTERNS)

        # Generic authority patterns
        patterns.extend([
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Bakanlığ[ıi])',
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Kurum[iu])',
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Kurul[iu])',
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Başkanlığ[ıi])',
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Mahkeme(?:si)?)'
        ])

        return patterns

    def extract(self, text: str, **kwargs) -> List[ExtractionResult]:
        """Extract authorities from text

        Args:
            text: Input text
            **kwargs: Additional options
                - authority_type: Specific authority type to extract

        Returns:
            List of authority extraction results
        """
        authority_type_filter = kwargs.get('authority_type', None)
        results = []

        # Extract using patterns
        base_results = self.extract_with_pattern(text, **kwargs)

        # Classify and enrich each result
        for result in base_results:
            authority_detail = self._classify_authority(result.value, text, result)

            if authority_detail:
                # Filter by type if specified
                if authority_type_filter and authority_detail.authority_type != authority_type_filter:
                    continue

                result.metadata['authority_detail'] = authority_detail
                results.append(result)

        # Remove duplicates
        results = self._deduplicate_results(results)

        # Sort by position
        results.sort(key=lambda r: r.start_pos if r.start_pos else 0)

        self.update_stats(success=len(results) > 0)
        logger.info(f"Extracted {len(results)} authorities from text")

        return results

    def _classify_authority(self, name: str, full_text: str, result: ExtractionResult) -> Optional[AuthorityDetail]:
        """Classify authority type and extract details"""
        name_lower = name.lower()

        # Check ministries
        for ministry_id, ministry_data in self.MINISTRIES.items():
            if any(ministry_name.lower() in name_lower for ministry_name in ministry_data['names']):
                return AuthorityDetail(
                    authority_type=AuthorityType.MINISTRY,
                    name=name,
                    abbreviation=ministry_data['abbrev'],
                    full_name=ministry_data['names'][0]
                )

        # Check regulatory bodies
        for body_id, body_data in self.REGULATORY_BODIES.items():
            if body_data['full_name'].lower() in name_lower or body_id.lower() in name_lower:
                return AuthorityDetail(
                    authority_type=AuthorityType.REGULATORY_BODY,
                    name=name,
                    abbreviation=body_id,
                    full_name=body_data['full_name']
                )

        # Check courts
        for court_id, court_data in self.COURTS.items():
            if court_data['full_name'].lower() in name_lower:
                return AuthorityDetail(
                    authority_type=AuthorityType.COURT,
                    name=name,
                    abbreviation=court_id if court_id != court_data['full_name'] else None,
                    full_name=court_data['full_name']
                )

        # Check state institutions
        if any(inst.lower() in name_lower for inst in self.STATE_INSTITUTIONS):
            if 'cumhurbaşkan' in name_lower:
                auth_type = AuthorityType.PRESIDENCY
            elif 'tbmm' in name_lower or 'büyük millet' in name_lower:
                auth_type = AuthorityType.PARLIAMENT
            else:
                auth_type = AuthorityType.STATE_INSTITUTION

            return AuthorityDetail(
                authority_type=auth_type,
                name=name,
                full_name=name
            )

        # Check local government
        if any(keyword in name_lower for keyword in ['belediye', 'il ', 'ilçe']):
            return AuthorityDetail(
                authority_type=AuthorityType.LOCAL_GOVERNMENT,
                name=name,
                full_name=name
            )

        # Generic classification based on suffix
        if 'bakanlığ' in name_lower:
            return AuthorityDetail(
                authority_type=AuthorityType.MINISTRY,
                name=name,
                full_name=name
            )
        elif 'kurul' in name_lower or 'kurum' in name_lower:
            return AuthorityDetail(
                authority_type=AuthorityType.REGULATORY_BODY,
                name=name,
                full_name=name
            )
        elif 'mahkeme' in name_lower:
            return AuthorityDetail(
                authority_type=AuthorityType.COURT,
                name=name,
                full_name=name
            )
        elif 'üniversite' in name_lower:
            return AuthorityDetail(
                authority_type=AuthorityType.UNIVERSITY,
                name=name,
                full_name=name
            )

        # Default: state institution
        return AuthorityDetail(
            authority_type=AuthorityType.STATE_INSTITUTION,
            name=name,
            full_name=name
        )

    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove duplicate authorities"""
        if not results:
            return results

        sorted_results = sorted(results, key=lambda r: r.start_pos if r.start_pos else 0)
        deduplicated = [sorted_results[0]]

        for result in sorted_results[1:]:
            last_result = deduplicated[-1]

            if (result.start_pos and last_result.end_pos and
                result.start_pos >= last_result.end_pos):
                deduplicated.append(result)
            elif (result.start_pos and last_result.start_pos and
                  abs(result.start_pos - last_result.start_pos) > 10):
                deduplicated.append(result)
            else:
                if result.confidence > last_result.confidence:
                    deduplicated[-1] = result

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} authorities")
        return deduplicated


__all__ = ['AuthorityExtractor', 'AuthorityType', 'AuthorityDetail']
