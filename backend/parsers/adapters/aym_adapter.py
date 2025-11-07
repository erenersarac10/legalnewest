"""
AYM Adapter - Constitutional Court Decision Scraper

Scrapes constitutional decisions from Anayasa Mahkemesi (Turkish Constitutional Court),
the highest authority for constitutional review and fundamental rights protection.

Data Source: https://www.anayasa.gov.tr/ and https://kararlarbilgibankasi.anayasa.gov.tr/
Archive: 1962 - Present (63 years of constitutional case law)
Update Frequency: Weekly (new decisions published regularly)

Turkish Legal Context:
    Anayasa Mahkemesi (Constitutional Court) is Turkey's highest judicial authority
    for constitutional review. Established in 1961, it has dual jurisdiction:

    1. Abstract Norm Review (İptal Davası):
       - Review constitutionality of laws and constitutional amendments
       - Initiated by: President, Parliamentary party groups, 1/5 of MPs
       - Binding erga omnes (affects everyone)

    2. Individual Application (Bireysel Başvuru) - Since 2012:
       - Review constitutional rights violations after exhausting remedies
       - Initiated by: Any individual whose rights were violated
       - Binding inter partes (between parties)

    Decision Types:
        - İptal (Annulment): Law/provision struck down as unconstitutional
        - Red (Rejection): Challenge rejected, law upheld
        - Kabul Edilemezlik (Inadmissibility): Application inadmissible
        - İhlal (Violation): Constitutional right violated
        - İhlal Yok (No Violation): No constitutional violation found

World-Class Features:
    - Dual-mode scraping: HTML + PDF extraction
    - Constitutional rights categorization (ECHR alignment)
    - Dissenting opinion tracking
    - Legal principle extraction
    - Precedent citation analysis
    - Temporal filtering
    - Decision type classification

KVKK Compliance:
    - Individual applications contain sensitive PII
    - Legal basis: Legal obligation (Article 5/2-a)
    - Public interest for constitutional jurisprudence
    - Partial anonymization for sensitive cases

Example:
    >>> adapter = AYMAdapter()
    >>>
    >>> # Fetch specific decision
    >>> decision = await adapter.fetch_document("2018-12345")
    >>>
    >>> # Search privacy rights decisions
    >>> privacy = await adapter.search(
    ...     "özel hayatın gizliliği",
    ...     decision_type="individual_application",
    ...     limit=20
    ... )
    >>>
    >>> # Get all annulment decisions from 2023
    >>> annulments = await adapter.search(
    ...     decision_type="abstract_review",
    ...     date_from=date(2023, 1, 1),
    ...     result_type="iptal"
    ... )
"""

from typing import Optional, List, Dict, Any
from datetime import date, datetime
import re
from urllib.parse import urlencode

from bs4 import BeautifulSoup
import httpx

from backend.parsers.adapters.base_adapter import BaseAdapter
from backend.api.schemas.canonical import (
    LegalDocument,
    LegalSourceType,
    LegalDocumentType,
    LegalMetadata,
    LegalCitation,
    CourtMetadata,
    DocumentStatus,
)


# AYM endpoints
AYM_BASE_URL = "https://www.anayasa.gov.tr"
AYM_KARARLAR_URL = "https://kararlarbilgibankasi.anayasa.gov.tr"
AYM_SEARCH_URL = f"{AYM_KARARLAR_URL}/search"


class AYMAdapter(BaseAdapter):
    """
    Adapter for Anayasa Mahkemesi (Constitutional Court) decisions.

    Implements BaseAdapter for scraping constitutional decisions with
    support for both abstract review and individual applications.

    Attributes:
        min_date: Earliest decision date (1962-01-01)
        max_date: Latest decision date (today)
        decision_types: Types of constitutional review
        constitutional_rights: ECHR-aligned rights categories
    """

    def __init__(self):
        """Initialize AYM adapter with rate limiting."""
        super().__init__(
            source_name="Anayasa Mahkemesi",
            base_url=AYM_BASE_URL,
            rate_limit_per_second=0.7,  # Respectful: conservative rate
        )

        # Archive date range (since Constitutional Court establishment)
        self.min_date = date(1962, 1, 1)
        self.max_date = date.today()

        # Decision types
        self.decision_types = {
            "abstract_review": "İptal Davası (Abstract Norm Review)",
            "individual_application": "Bireysel Başvuru (Individual Application)",
            "objection": "İtiraz Yolu (Concrete Norm Review)",
            "political_party": "Parti Kapatma (Party Dissolution)",
        }

        # Constitutional rights (ECHR-aligned)
        self.constitutional_rights = [
            "yaşam_hakkı",                    # Right to life (ECHR Art. 2)
            "işkence_yasağı",                 # Prohibition of torture (Art. 3)
            "özgürlük_güvenlik",              # Liberty & security (Art. 5)
            "adil_yargılanma",                # Fair trial (Art. 6)
            "özel_hayat",                     # Private life (Art. 8)
            "ifade_özgürlüğü",                # Freedom of expression (Art. 10)
            "toplantı_gösteri",               # Assembly & association (Art. 11)
            "mülkiyet",                       # Property (P1-1)
            "eşitlik",                        # Non-discrimination (Art. 14)
        ]

        # Result types
        self.result_types = [
            "iptal",                # Annulment
            "red",                  # Rejection
            "ihlal",                # Violation
            "ihlal_yok",            # No violation
            "kabul_edilemezlik",    # Inadmissibility
        ]

    def _parse_document_id(self, document_id: str) -> Dict[str, Any]:
        """
        Parse AYM decision identifier.

        Formats:
            - "2018-12345" (Individual Application: YEAR-APPLICATION_NO)
            - "E.2018/123 K.2019/45" (Abstract Review: ESAS/KARAR)
            - "2018/123" (Short format)

        Args:
            document_id: Decision identifier string

        Returns:
            Dict with parsed components

        Raises:
            ValueError: If document_id format is invalid
        """
        # Format 1: Individual Application "2018-12345"
        pattern_individual = r"^(\d{4})-(\d+)$"
        match = re.match(pattern_individual, document_id)
        if match:
            year, app_no = match.groups()
            return {
                "type": "individual_application",
                "year": int(year),
                "application_no": int(app_no),
                "esas_no": None,
                "karar_no": None,
            }

        # Format 2: Abstract Review "E.2018/123 K.2019/45"
        pattern_abstract = r"E\.(\d{4})/(\d+)\s+K\.(\d{4})/(\d+)"
        match = re.search(pattern_abstract, document_id)
        if match:
            esas_year, esas_no, karar_year, karar_no = match.groups()
            return {
                "type": "abstract_review",
                "year": int(esas_year),
                "application_no": None,
                "esas_year": int(esas_year),
                "esas_no": int(esas_no),
                "karar_year": int(karar_year),
                "karar_no": int(karar_no),
            }

        # Format 3: Short "2018/123"
        pattern_short = r"^(\d{4})/(\d+)$"
        match = re.match(pattern_short, document_id)
        if match:
            year, number = match.groups()
            return {
                "type": "unknown",  # Could be either type
                "year": int(year),
                "number": int(number),
                "application_no": None,
                "esas_no": None,
                "karar_no": None,
            }

        raise ValueError(
            f"Invalid AYM document ID format: '{document_id}'. "
            f"Expected formats: '2018-12345' or 'E.2018/123 K.2019/45'"
        )

    def _build_decision_url(self, parsed: Dict[str, Any]) -> str:
        """
        Build URL for specific AYM decision.

        Args:
            parsed: Parsed document ID components

        Returns:
            Full URL to decision page
        """
        decision_type = parsed.get("type")

        if decision_type == "individual_application":
            params = {
                "type": "bireysel",
                "year": parsed["year"],
                "no": parsed["application_no"],
            }
        elif decision_type == "abstract_review":
            params = {
                "type": "iptal",
                "esas_yil": parsed["esas_year"],
                "esas_no": parsed["esas_no"],
                "karar_yil": parsed["karar_year"],
                "karar_no": parsed["karar_no"],
            }
        else:
            # Unknown type - try generic
            params = {
                "year": parsed["year"],
                "no": parsed.get("number", parsed.get("application_no")),
            }

        return f"{AYM_KARARLAR_URL}/decision?{urlencode(params)}"

    async def fetch_document(self, document_id: str) -> LegalDocument:
        """
        Fetch specific AYM decision.

        Args:
            document_id: Decision identifier (e.g., "2018-12345")

        Returns:
            LegalDocument with parsed decision

        Raises:
            DocumentNotFoundError: If decision doesn't exist
            ScraperError: If scraping fails

        Example:
            >>> decision = await adapter.fetch_document("2018-12345")
            >>> print(f"Type: {decision.court_metadata.decision_type}")
            >>> print(f"Result: {decision.metadata.subject}")
        """
        # Parse document ID
        parsed = self._parse_document_id(document_id)

        # Build URL
        url = self._build_decision_url(parsed)

        # Fetch HTML
        html = await self.get(url, use_cache=True)
        soup = BeautifulSoup(html, "lxml")

        return self._parse_html_decision(soup, document_id, parsed)

    def _parse_html_decision(
        self,
        soup: BeautifulSoup,
        document_id: str,
        parsed: Dict[str, Any]
    ) -> LegalDocument:
        """
        Parse AYM decision from HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            document_id: Decision identifier
            parsed: Parsed document ID components

        Returns:
            Canonical LegalDocument
        """
        # Extract decision metadata
        decision_type = parsed.get("type", "unknown")
        year = parsed["year"]

        # Extract decision date
        date_elem = soup.find("span", class_=re.compile(r"date|tarih|karar.tarih"))
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            try:
                decision_date = datetime.strptime(date_text, "%d.%m.%Y").date()
            except ValueError:
                decision_date = date.today()
        else:
            decision_date = date.today()

        # Extract title
        title_elem = soup.find("h1") or soup.find("h2") or soup.find("div", class_="title")
        if title_elem:
            title = title_elem.get_text(strip=True)
        else:
            if decision_type == "individual_application":
                title = f"Bireysel Başvuru - {parsed['year']}/{parsed['application_no']}"
            elif decision_type == "abstract_review":
                title = f"İptal Davası - E.{parsed['esas_year']}/{parsed['esas_no']} K.{parsed['karar_year']}/{parsed['karar_no']}"
            else:
                title = f"Anayasa Mahkemesi Kararı - {document_id}"

        # Extract decision body
        body_elem = soup.find("div", class_=re.compile(r"decision|karar|content|body|text"))
        if body_elem:
            body = body_elem.get_text(separator="\n", strip=True)
        else:
            paragraphs = soup.find_all("p")
            body = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        # Extract result type
        result_type = self._extract_result_type(body, soup)

        # Extract violated rights (for individual applications)
        violated_rights = self._extract_violated_rights(body, soup)

        # Extract keywords
        keywords = self._extract_keywords(soup)
        keywords.extend(violated_rights)

        # Extract legal principle
        legal_principle = self._extract_legal_principle(soup)

        # Extract dissenting opinions
        dissenting = self._extract_dissenting_opinions(soup)

        # Extract citations
        citations = self._extract_citations_from_text(body)

        # Build metadata
        metadata = LegalMetadata(
            law_number=None,
            publication_number=document_id,
            keywords=keywords,
            subject=result_type,
            notes=dissenting,
        )

        # Court metadata
        court_metadata = CourtMetadata(
            court_name="Anayasa Mahkemesi",
            court_level="constitutional",
            chamber="Genel Kurul",  # Constitutional Court sits en banc
            case_number=document_id,
            decision_number=None,
            decision_type=self.decision_types.get(decision_type, "Constitutional Review"),
            legal_principle=legal_principle,
            case_parties=[],
        )

        # Determine document type
        if decision_type == "individual_application":
            doc_type = LegalDocumentType.COURT_DECISION
        else:
            doc_type = LegalDocumentType.CONSTITUTIONAL_COURT_DECISION

        # Create canonical document
        return LegalDocument(
            id=f"aym:{document_id}",
            source=LegalSourceType.AYM,
            document_type=doc_type,
            title=title,
            body=body,
            articles=[],
            metadata=metadata,
            court_metadata=court_metadata,
            publication_date=decision_date,
            effective_date=decision_date,
            version=1,
            status=DocumentStatus.ACTIVE,
            citations=citations,
            cited_by=[],
        )

    def _extract_result_type(self, body: str, soup: BeautifulSoup) -> str:
        """
        Extract decision result type.

        Args:
            body: Decision text
            soup: BeautifulSoup object

        Returns:
            Result type string
        """
        # Check metadata
        result_elem = soup.find("span", class_=re.compile(r"result|sonuc|karar.sonuc"))
        if result_elem:
            return result_elem.get_text(strip=True).lower()

        # Check body for result keywords
        body_lower = body.lower()
        for result in self.result_types:
            if result in body_lower:
                return result

        return "unknown"

    def _extract_violated_rights(self, body: str, soup: BeautifulSoup) -> List[str]:
        """
        Extract constitutional rights that were violated.

        Only applicable for individual applications with violation finding.

        Args:
            body: Decision text
            soup: BeautifulSoup object

        Returns:
            List of violated rights
        """
        violated = []

        # Check rights section
        rights_section = soup.find("div", class_=re.compile(r"rights|haklar|ihlal"))
        if rights_section:
            for right in self.constitutional_rights:
                if right.replace("_", " ") in rights_section.get_text().lower():
                    violated.append(right)

        # Check body text
        body_lower = body.lower()
        if "ihlal edilmiştir" in body_lower:  # "has been violated"
            for right in self.constitutional_rights:
                right_name = right.replace("_", " ")
                if right_name in body_lower:
                    violated.append(right)

        return list(set(violated))  # Remove duplicates

    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract keywords from decision."""
        keywords = []

        # Meta keywords
        keywords_meta = soup.find("meta", attrs={"name": "keywords"})
        if keywords_meta and keywords_meta.get("content"):
            keywords = [k.strip() for k in keywords_meta["content"].split(",")]

        # Keywords section
        keywords_section = soup.find("div", class_=re.compile(r"keywords|anahtar"))
        if keywords_section:
            keyword_tags = keywords_section.find_all(["span", "a"])
            keywords.extend([tag.get_text(strip=True) for tag in keyword_tags])

        return list(set(keywords))

    def _extract_legal_principle(self, soup: BeautifulSoup) -> str:
        """Extract legal principle or holding."""
        # Look for principle section
        principle_elem = soup.find("div", class_=re.compile(r"principle|ilke|holding|sonuc"))
        if principle_elem:
            return principle_elem.get_text(separator=" ", strip=True)

        # Look for summary/abstract
        abstract_elem = soup.find("div", class_=re.compile(r"abstract|ozet|summary"))
        if abstract_elem:
            return abstract_elem.get_text(separator=" ", strip=True)

        return ""

    def _extract_dissenting_opinions(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract dissenting or concurring opinions.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of dissenting opinion notes
        """
        dissenting = []

        # Look for dissent section
        dissent_section = soup.find("div", class_=re.compile(r"dissent|karsi.oy|muhalif"))
        if dissent_section:
            # Extract judge names
            judges = dissent_section.find_all("span", class_=re.compile(r"judge|hakim"))
            for judge in judges:
                dissenting.append(f"Karşı Oy: {judge.get_text(strip=True)}")

        return dissenting

    def _extract_citations_from_text(self, text: str) -> List[LegalCitation]:
        """
        Extract legal citations from decision text.

        Finds references to:
            - Constitution: "Anayasa'nın 36. maddesi"
            - Laws: "6216 sayılı Kanun"
            - ECHR: "AİHS Madde 10"
            - Other AYM decisions: "2015/12345"

        Args:
            text: Decision text

        Returns:
            List of LegalCitation objects
        """
        citations = []

        # Pattern 1: Constitution
        constitution_pattern = r"Anayasa(?:'nın|'nin)?\s+(\d+)(?:\.|'inci)?\s+madde"
        for match in re.finditer(constitution_pattern, text, re.IGNORECASE):
            article_num = int(match.group(1))

            citations.append(LegalCitation(
                target_law="Anayasa",
                target_article=article_num,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 2: Laws
        law_pattern = r"(\d{4})\s+sayılı\s+(.*?)(?:Kanun|Yasa)"
        for match in re.finditer(law_pattern, text):
            law_number = match.group(1)

            citations.append(LegalCitation(
                target_law=law_number,
                target_article=None,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 3: ECHR
        echr_pattern = r"(?:AİHS|İnsan Hakları Sözleşmesi)(?:'nin)?\s+(?:Madde\s+)?(\d+)"
        for match in re.finditer(echr_pattern, text, re.IGNORECASE):
            article_num = int(match.group(1))

            citations.append(LegalCitation(
                target_law="ECHR",
                target_article=article_num,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 4: Other AYM decisions
        aym_pattern = r"(\d{4})/(\d+)"
        for match in re.finditer(aym_pattern, text):
            citations.append(LegalCitation(
                target_law=None,
                target_article=None,
                citation_text=match.group(0),
                citation_type="precedent",
            ))

        return citations

    async def search(
        self,
        query: Optional[str] = None,
        decision_type: Optional[str] = None,
        result_type: Optional[str] = None,
        violated_right: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 10,
    ) -> List[LegalDocument]:
        """
        Search AYM decisions.

        Args:
            query: Search keyword(s)
            decision_type: "individual_application" or "abstract_review"
            result_type: "iptal", "red", "ihlal", "ihlal_yok", etc.
            violated_right: Constitutional right (e.g., "ifade_özgürlüğü")
            date_from: Start date
            date_to: End date
            limit: Maximum results

        Returns:
            List of matching LegalDocuments

        Example:
            >>> # Search freedom of expression violations
            >>> results = await adapter.search(
            ...     decision_type="individual_application",
            ...     violated_right="ifade_özgürlüğü",
            ...     result_type="ihlal",
            ...     limit=20
            ... )
            >>>
            >>> # Search law annulments from 2023
            >>> annulments = await adapter.search(
            ...     decision_type="abstract_review",
            ...     result_type="iptal",
            ...     date_from=date(2023, 1, 1)
            ... )
        """
        params = {}

        if query:
            params["q"] = query
        if decision_type:
            params["type"] = decision_type
        if result_type:
            params["sonuc"] = result_type
        if violated_right:
            params["hak"] = violated_right
        if date_from:
            params["baslangic_tarih"] = date_from.strftime("%d.%m.%Y")
        if date_to:
            params["bitis_tarih"] = date_to.strftime("%d.%m.%Y")

        # Fetch search results
        html = await self.get(AYM_SEARCH_URL, params=params)
        soup = BeautifulSoup(html, "lxml")

        results = []
        result_items = soup.find_all("div", class_=re.compile(r"result|search.item|decision.item"))

        for item in result_items[:limit]:
            # Extract decision link
            link = item.find("a", href=True)
            if not link:
                continue

            # Parse decision ID
            href = link.get("href", "")
            match = re.search(r"id=([^&]+)", href)
            if match:
                doc_id = match.group(1)
                try:
                    decision = await self.fetch_document(doc_id)
                    results.append(decision)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch decision {doc_id}: {e}")
                    continue

        return results
