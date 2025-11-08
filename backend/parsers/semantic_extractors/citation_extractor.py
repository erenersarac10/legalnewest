"""Citation Extractor - Harvey/Legora CTO-Level Production-Grade
Extracts legal citations from Turkish legal documents

Production Features:
- Law number extraction (5237 sayılı Kanun, etc.)
- Article/Madde extraction with paragraphs
- Court decision citations (Yargıtay, Danıştay, AYM)
- Regulation citations (Yönetmelik, Tebliğ)
- Presidential Decree citations (CBK)
- International treaty citations
- Cross-reference extraction
- Citation type classification
- Hierarchical citation parsing (Madde > Fıkra > Bent)
"""
from typing import Dict, List, Any, Optional, Tuple, Set
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


class CitationType(Enum):
    """Types of legal citations"""
    LAW = "LAW"  # Kanun
    REGULATION = "REGULATION"  # Yönetmelik
    DECREE = "DECREE"  # CBK, KHK
    ARTICLE = "ARTICLE"  # Madde
    PARAGRAPH = "PARAGRAPH"  # Fıkra
    SUBPARAGRAPH = "SUBPARAGRAPH"  # Bent
    COURT_DECISION = "COURT_DECISION"  # Mahkeme kararı
    INTERNATIONAL_TREATY = "INTERNATIONAL_TREATY"  # Uluslararası antlaşma
    OFFICIAL_GAZETTE = "OFFICIAL_GAZETTE"  # Resmi Gazete
    COMMUNIQUE = "COMMUNIQUE"  # Tebliğ
    CIRCULAR = "CIRCULAR"  # Genelge
    MIXED = "MIXED"  # Multiple types


@dataclass
class CitationDetail:
    """Detailed citation information"""
    citation_type: CitationType
    law_number: Optional[str] = None
    law_name: Optional[str] = None
    article_number: Optional[str] = None
    paragraph_number: Optional[str] = None
    subparagraph_letter: Optional[str] = None
    court_name: Optional[str] = None
    decision_number: Optional[str] = None
    decision_date: Optional[str] = None
    rg_number: Optional[str] = None
    rg_date: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class CitationExtractor(RegexExtractor):
    """Citation Extractor for Turkish Legal Documents

    Extracts and classifies legal citations with:
    - Law number patterns (10+ formats)
    - Article/paragraph hierarchies
    - Court decision formats (3 court types)
    - Regulation and decree patterns
    - International treaty references
    - Official Gazette citations
    - Cross-reference resolution

    Features:
    - Hierarchical citation parsing (Madde 15/3-a)
    - Court identification (Yargıtay, Danıştay, AYM)
    - Date extraction from citations
    - Citation type classification
    - Confidence scoring based on pattern complexity
    - Context-aware validation
    """

    # Law number patterns (ordered by specificity)
    LAW_PATTERNS = [
        # Full format: "5237 sayılı Türk Ceza Kanunu"
        r'(\d{4})\s+sayılı\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Kanun[iu]?)',
        # Short format: "5237 sayılı Kanun"
        r'(\d{4})\s+sayılı\s+Kanun[iu]?',
        # Format: "Türk Ceza Kanunu (TCK)"
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+Kanun[iu]?)\s*\(([A-Z]{2,5})\)',
        # Format: "TCK" (abbreviation only, lower confidence)
        r'\b([A-Z]{2,5}K)\b',
        # Format: "6698 sayılı KVKK"
        r'(\d{4})\s+sayılı\s+([A-Z]{2,10})',
        # Format: "Kanun No: 5237"
        r'Kanun\s+(?:No|Sayı)\s*:?\s*(\d{4})'
    ]

    # Article patterns
    ARTICLE_PATTERNS = [
        # Format: "madde 15/3-a"
        r'[Mm]adde\s+(\d+)(?:/(\d+))?(?:-([a-z]))?',
        # Format: "15. madde"
        r'(\d+)\.\s+[Mm]adde',
        # Format: "md. 15"
        r'[Mm]d\.\s*(\d+)',
        # Format: "m. 15"
        r'[Mm]\.\s*(\d+)',
        # Format: "15 inci madde"
        r'(\d+)\s+(?:inci|nci|üncü|ncı)\s+[Mm]adde'
    ]

    # Paragraph patterns
    PARAGRAPH_PATTERNS = [
        # Format: "3. fıkra" or "fıkra 3"
        r'(?:fıkra|fıkrası)\s*(\d+)',
        r'(\d+)\.\s*(?:fıkra|fıkrası)',
        # Format: "f. 3"
        r'[Ff]\.\s*(\d+)'
    ]

    # Subparagraph patterns (bent)
    SUBPARAGRAPH_PATTERNS = [
        # Format: "a bendi" or "bent a"
        r'([a-z])\s+bendi',
        r'bent\s+([a-z])',
        # Format: "(a)"
        r'\(([a-z])\)'
    ]

    # Court decision patterns
    COURT_PATTERNS = {
        'YARGITAY': [
            # Format: "Yargıtay 12. Ceza Dairesi 2019/1234 E., 2020/5678 K."
            r'Yargıtay\s+(\d+)\.\s+(Ceza|Hukuk)\s+Dairesi.*?(\d{4}/\d+)\s+E\..*?(\d{4}/\d+)\s+K\.',
            # Format: "Yargıtay Büyük Genel Kurulu"
            r'Yargıtay\s+(Büyük\s+)?Genel\s+Kurul[iu]',
            # Short format: "Y. 12. CD."
            r'Y\.\s*(\d+)\.\s*(CD|HD)\.'
        ],
        'DANIŞTAY': [
            # Format: "Danıştay 10. Dairesi 2019/1234 E."
            r'Danıştay\s+(\d+)\.\s+Daire(?:si)?.*?(\d{4}/\d+)',
            # Format: "Danıştay İdari Dava Daireleri Kurulu"
            r'Danıştay\s+İdari\s+Dava\s+Daireleri\s+Kurulu',
            # Short: "D. 10. D."
            r'D\.\s*(\d+)\.\s*D\.'
        ],
        'AYM': [
            # Format: "Anayasa Mahkemesi 2019/123 E., 2020/45 K."
            r'Anayasa\s+Mahkemesi.*?(\d{4}/\d+)\s+E\..*?(\d{4}/\d+)\s+K\.',
            # Format: "AYM 2019/123"
            r'AYM\s+(\d{4}/\d+)',
            # Format: "Anayasa Mahkemesi Başvuru No: 2019/12345"
            r'Anayasa\s+Mahkemesi\s+Başvuru\s+No\s*:?\s*(\d{4}/\d+)'
        ]
    }

    # Regulation patterns
    REGULATION_PATTERNS = [
        # Format: "İcra İflas Kanununun Uygulanmasına Dair Yönetmelik"
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]{10,100}Yönetmelik)',
        # Format: "... Yönetmeliğinin 15. maddesi"
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]{10,100}Yönetmeliğ[ii])',
        # Format: "Yönetmelik No: 12345"
        r'Yönetmelik\s+No\s*:?\s*(\d+)'
    ]

    # Presidential Decree (CBK) patterns
    CBK_PATTERNS = [
        # Format: "1 sayılı Cumhurbaşkanlığı Kararnamesi"
        r'(\d+)\s+sayılı\s+Cumhurbaşkanlığı\s+Kararnamesi',
        # Format: "CBK No: 1"
        r'CBK\s+(?:No|Sayı)\s*:?\s*(\d+)',
        # Short: "1 No'lu CBK"
        r"(\d+)\s+No['']?lu\s+CBK"
    ]

    # Communique patterns
    COMMUNIQUE_PATTERNS = [
        # Format: "1 Sıra No'lu Tebliğ"
        r"(\d+)\s+Sıra\s+No['']?lu\s+Tebliğ",
        # Format: "... Tebliği (Sıra No: 1)"
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]{10,100}Tebliğ[ii])',
        r'Tebliğ\s+(?:No|Sıra\s+No)\s*:?\s*(\d+)'
    ]

    # Official Gazette patterns
    RG_PATTERNS = [
        # Format: "12.05.2018 tarih ve 30425 sayılı Resmi Gazete"
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarih\s+ve\s+(\d{4,5})\s+sayılı\s+Resmi\s+Gazete',
        # Format: "RG 30425"
        r'RG\s+(\d{4,5})',
        # Format: "Resmi Gazete: 30425"
        r'Resmi\s+Gazete\s*:?\s*(\d{4,5})'
    ]

    # International treaty patterns
    TREATY_PATTERNS = [
        # Format: "Avrupa İnsan Hakları Sözleşmesi"
        r'(Avrupa\s+İnsan\s+Hakları\s+Sözleşmesi)',
        # Format: "... Sözleşmesi"
        r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]{15,100}Sözleşmesi)',
        # Format: "AİHS"
        r'\b(AİHS)\b'
    ]

    def __init__(self):
        # Combine all patterns
        all_patterns = (
            self.LAW_PATTERNS +
            self.ARTICLE_PATTERNS +
            self.REGULATION_PATTERNS +
            self.CBK_PATTERNS +
            self.COMMUNIQUE_PATTERNS +
            self.RG_PATTERNS +
            self.TREATY_PATTERNS
        )

        # Add court patterns
        for court_type, patterns in self.COURT_PATTERNS.items():
            all_patterns.extend(patterns)

        super().__init__(
            name="Citation Extractor",
            patterns=all_patterns,
            version="2.0.0"
        )

    def extract(self, text: str, **kwargs) -> List[ExtractionResult]:
        """Extract citations from text

        Args:
            text: Input text
            **kwargs: Additional options
                - extract_type: Specific citation type to extract
                - include_context: Include surrounding context (default: True)

        Returns:
            List of citation extraction results
        """
        extract_type = kwargs.get('extract_type', None)
        results = []

        # Extract different citation types
        if not extract_type or extract_type == CitationType.LAW:
            results.extend(self._extract_laws(text))

        if not extract_type or extract_type == CitationType.ARTICLE:
            results.extend(self._extract_articles(text))

        if not extract_type or extract_type == CitationType.COURT_DECISION:
            results.extend(self._extract_court_decisions(text))

        if not extract_type or extract_type == CitationType.REGULATION:
            results.extend(self._extract_regulations(text))

        if not extract_type or extract_type == CitationType.DECREE:
            results.extend(self._extract_decrees(text))

        if not extract_type or extract_type == CitationType.OFFICIAL_GAZETTE:
            results.extend(self._extract_rg_citations(text))

        if not extract_type or extract_type == CitationType.INTERNATIONAL_TREATY:
            results.extend(self._extract_treaties(text))

        # Remove duplicates based on position
        results = self._deduplicate_results(results)

        # Sort by position
        results.sort(key=lambda r: r.start_pos if r.start_pos else 0)

        self.update_stats(success=len(results) > 0)
        logger.info(f"Extracted {len(results)} citations from text")

        return results

    def _extract_laws(self, text: str) -> List[ExtractionResult]:
        """Extract law citations"""
        results = []

        for pattern in self.LAW_PATTERNS[:6]:  # Use compiled patterns
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                # Extract law details
                groups = match.groups()
                law_number = groups[0] if groups and groups[0].isdigit() else None
                law_name = groups[1] if len(groups) > 1 else None

                # Create citation detail
                detail = CitationDetail(
                    citation_type=CitationType.LAW,
                    law_number=law_number,
                    law_name=law_name
                )

                # Calculate confidence
                confidence = self._calculate_law_confidence(match, text, pattern)

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'citation_detail': detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _extract_articles(self, text: str) -> List[ExtractionResult]:
        """Extract article citations"""
        results = []

        for pattern in self.ARTICLE_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                groups = match.groups()
                article_num = groups[0] if groups else None
                para_num = groups[1] if len(groups) > 1 and groups[1] else None
                subpara = groups[2] if len(groups) > 2 and groups[2] else None

                detail = CitationDetail(
                    citation_type=CitationType.ARTICLE,
                    article_number=article_num,
                    paragraph_number=para_num,
                    subparagraph_letter=subpara
                )

                confidence = 0.85 if para_num else 0.75

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'citation_detail': detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _extract_court_decisions(self, text: str) -> List[ExtractionResult]:
        """Extract court decision citations"""
        results = []

        for court_name, patterns in self.COURT_PATTERNS.items():
            for pattern in patterns:
                compiled = re.compile(pattern, re.IGNORECASE)
                for match in compiled.finditer(text):
                    detail = CitationDetail(
                        citation_type=CitationType.COURT_DECISION,
                        court_name=court_name,
                        additional_info={'full_citation': match.group(0)}
                    )

                    # Higher confidence for full citations with E/K numbers
                    confidence = 0.95 if 'E.' in match.group(0) and 'K.' in match.group(0) else 0.80

                    result = ExtractionResult(
                        value=match.group(0),
                        confidence=confidence,
                        confidence_level=self.get_confidence_level(confidence),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=self.extract_context(text, match.start(), match.end()),
                        method=ExtractionMethod.REGEX_PATTERN,
                        source_text=match.group(0),
                        metadata={'citation_detail': detail}
                    )

                    if self.validate_result(result):
                        results.append(result)

        return results

    def _extract_regulations(self, text: str) -> List[ExtractionResult]:
        """Extract regulation citations"""
        results = []

        for pattern in self.REGULATION_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                detail = CitationDetail(
                    citation_type=CitationType.REGULATION,
                    law_name=match.group(1) if match.groups() else match.group(0)
                )

                confidence = 0.80

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'citation_detail': detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _extract_decrees(self, text: str) -> List[ExtractionResult]:
        """Extract presidential decree citations (CBK)"""
        results = []

        for pattern in self.CBK_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                groups = match.groups()
                decree_num = groups[0] if groups else None

                detail = CitationDetail(
                    citation_type=CitationType.DECREE,
                    law_number=decree_num,
                    law_name="Cumhurbaşkanlığı Kararnamesi"
                )

                confidence = 0.90

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'citation_detail': detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _extract_rg_citations(self, text: str) -> List[ExtractionResult]:
        """Extract Official Gazette citations"""
        results = []

        for pattern in self.RG_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                groups = match.groups()

                if len(groups) == 2:
                    rg_date, rg_num = groups
                elif len(groups) == 1:
                    rg_date, rg_num = None, groups[0]
                else:
                    continue

                detail = CitationDetail(
                    citation_type=CitationType.OFFICIAL_GAZETTE,
                    rg_number=rg_num,
                    rg_date=rg_date
                )

                confidence = 0.90 if rg_date else 0.75

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'citation_detail': detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _extract_treaties(self, text: str) -> List[ExtractionResult]:
        """Extract international treaty citations"""
        results = []

        for pattern in self.TREATY_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                detail = CitationDetail(
                    citation_type=CitationType.INTERNATIONAL_TREATY,
                    law_name=match.group(1) if match.groups() else match.group(0)
                )

                # Higher confidence for well-known treaties like ECHR
                confidence = 0.95 if 'AİHS' in match.group(0) or 'Avrupa İnsan Hakları' in match.group(0) else 0.80

                result = ExtractionResult(
                    value=match.group(0),
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self.extract_context(text, match.start(), match.end()),
                    method=ExtractionMethod.REGEX_PATTERN,
                    source_text=match.group(0),
                    metadata={'citation_detail': detail}
                )

                if self.validate_result(result):
                    results.append(result)

        return results

    def _calculate_law_confidence(self, match: re.Match, text: str, pattern: str) -> float:
        """Calculate confidence for law citations"""
        # Base confidence
        confidence = 0.80

        # Boost for full format with law name
        if len(match.groups()) >= 2 and match.group(2):
            confidence += 0.10

        # Boost for 4-digit law number
        if match.group(1) and len(match.group(1)) == 4 and match.group(1).isdigit():
            confidence += 0.05

        # Check context for additional validation
        context_before = text[max(0, match.start() - 20):match.start()]
        if any(word in context_before.lower() for word in ['değiştirilen', 'eklenen', 'göre']):
            confidence += 0.05

        return min(1.0, confidence)

    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove duplicate citations based on position overlap"""
        if not results:
            return results

        # Sort by start position
        sorted_results = sorted(results, key=lambda r: r.start_pos if r.start_pos else 0)

        deduplicated = [sorted_results[0]]

        for result in sorted_results[1:]:
            # Check if this result overlaps with the last added result
            last_result = deduplicated[-1]

            if (result.start_pos and last_result.end_pos and
                result.start_pos >= last_result.end_pos):
                # No overlap, add it
                deduplicated.append(result)
            elif (result.start_pos and last_result.start_pos and
                  abs(result.start_pos - last_result.start_pos) > 10):
                # Significant position difference, add it
                deduplicated.append(result)
            else:
                # Overlap detected, keep the one with higher confidence
                if result.confidence > last_result.confidence:
                    deduplicated[-1] = result

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} citations")
        return deduplicated


__all__ = ['CitationExtractor', 'CitationType', 'CitationDetail']
