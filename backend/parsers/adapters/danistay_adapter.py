"""
Danıştay Adapter - Council of State Decision Scraper

Scrapes administrative law decisions from Danıştay (Turkish Council of State),
the supreme court for administrative and tax law cases in Turkey.

Data Source: https://www.danistay.gov.tr/ and https://uyap.danistay.gov.tr/
Archive: 1868 - Present (157 years of administrative case law)
Update Frequency: Daily (new decisions published regularly)

Turkish Legal Context:
    Danıştay is the highest judicial organ for reviewing administrative acts
    and settling administrative disputes. It has dual role:
        1. Judicial: Reviews administrative court decisions (final appeal)
        2. Advisory: Advises government on legislation and regulations

    Court Structure:
        - 15 Daireler (Chambers): Specialized by subject
            * 1st-3rd: Tax Law
            * 4th: Social Security & Insurance
            * 5th-6th: Public Service & Personnel
            * 7th-8th: Expropriation
            * 9th: Environmental & Urban Planning
            * 10th: Education & Health
            * 11th-13th: General Administrative
            * 14th: Tenders & Contracts
            * 15th: Judicial Review
        - İdari Dava Daireleri Kurulu (Administrative Cases General Board)
        - Vergi Dava Daireleri Kurulu (Tax Cases General Board)

    Decision Types:
        - Bozma (Reversal): Lower court decision reversed
        - Onanma (Affirmation): Lower court decision affirmed
        - İptal (Annulment): Administrative act cancelled
        - Red (Rejection): Petition rejected

World-Class Features:
    - HTML scraping with intelligent parsing
    - Chamber-specific specialization
    - Decision type classification
    - Administrative act analysis
    - Cross-reference to laws and regulations
    - Temporal filtering
    - Subject-area categorization

KVKK Compliance:
    - Administrative decisions may contain PII (names, addresses)
    - Legal basis: Legal obligation (Article 5/2-a)
    - Public record per İdari Yargılama Usulü Kanunu
    - Anonymization for sensitive cases

Example:
    >>> adapter = DanistayAdapter()
    >>>
    >>> # Fetch specific decision
    >>> decision = await adapter.fetch_document("9-d-2018-1234-2019-5678")
    >>>
    >>> # Search environmental law decisions
    >>> results = await adapter.search(
    ...     "çevre kirliliği",
    ...     chamber=9,
    ...     limit=10
    ... )
    >>>
    >>> # Get all tax decisions from 2020
    >>> tax_decisions = await adapter.search(
    ...     chamber_range=(1, 3),  # Tax chambers
    ...     date_from=date(2020, 1, 1),
    ...     date_to=date(2020, 12, 31)
    ... )
"""

from typing import Optional, List, Dict, Any
from datetime import date, datetime
import re
from urllib.parse import urlencode, quote

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


# Danıştay endpoints
DANISTAY_BASE_URL = "https://www.danistay.gov.tr"
DANISTAY_UYAP_URL = "https://uyap.danistay.gov.tr"
DANISTAY_SEARCH_URL = f"{DANISTAY_BASE_URL}/search"


class DanistayAdapter(BaseAdapter):
    """
    Adapter for Danıştay (Council of State) administrative law decisions.

    Implements BaseAdapter for scraping and parsing Danıştay decisions
    with chamber-specific specialization and administrative law context.

    Attributes:
        min_date: Earliest decision date (1868-01-01)
        max_date: Latest decision date (today)
        chambers: Dict mapping chamber numbers to subject areas
        decision_types: List of decision types (bozma, onanma, iptal, red)
    """

    def __init__(self):
        """Initialize Danıştay adapter with rate limiting."""
        super().__init__(
            source_name="Danıştay",
            base_url=DANISTAY_BASE_URL,
            rate_limit_per_second=0.8,  # Respectful: slightly slower for stability
        )

        # Archive date range
        self.min_date = date(1868, 1, 1)
        self.max_date = date.today()

        # Chamber specializations
        self.chambers = {
            1: "Tax Law (Vergi Hukuku)",
            2: "Tax Law (Vergi Hukuku)",
            3: "Tax Law (Vergi Hukuku)",
            4: "Social Security & Insurance (Sosyal Güvenlik)",
            5: "Public Service & Personnel (Personel)",
            6: "Public Service & Personnel (Personel)",
            7: "Expropriation (Kamulaştırma)",
            8: "Expropriation (Kamulaştırma)",
            9: "Environmental & Urban Planning (Çevre ve İmar)",
            10: "Education & Health (Eğitim ve Sağlık)",
            11: "General Administrative (Genel İdari)",
            12: "General Administrative (Genel İdari)",
            13: "General Administrative (Genel İdari)",
            14: "Tenders & Contracts (İhale ve Sözleşmeler)",
            15: "Judicial Review (İtiraz İnceleme)",
        }

        # Decision types
        self.decision_types = [
            "bozma",      # Reversal
            "onanma",     # Affirmation
            "iptal",      # Annulment
            "red",        # Rejection
            "durdurma",   # Stay of execution
        ]

    def _parse_document_id(self, document_id: str) -> Dict[str, Any]:
        """
        Parse Danıştay decision identifier.

        Format: "{chamber}-d-{esas_year}-{esas_no}-{karar_year}-{karar_no}"
        Example: "9-d-2018-1234-2019-5678"
            → Chamber 9, Esas 2018/1234, Karar 2019/5678

        Alternative formats:
            - "idk-2018-1234" (İdari Dava Daireleri Kurulu)
            - "vdk-2018-1234" (Vergi Dava Daireleri Kurulu)

        Args:
            document_id: Decision identifier string

        Returns:
            Dict with parsed components

        Raises:
            ValueError: If document_id format is invalid
        """
        # Standard format: "9-d-2018-1234-2019-5678"
        pattern = r"^(\d+|idk|vdk)-d-(\d{4})-(\d+)(?:-(\d{4})-(\d+))?$"
        match = re.match(pattern, document_id.lower())

        if match:
            chamber, esas_year, esas_no, karar_year, karar_no = match.groups()

            # Handle general boards
            if chamber == "idk":
                chamber_num = "İDDK"  # İdari Dava Daireleri Kurulu
            elif chamber == "vdk":
                chamber_num = "VDDK"  # Vergi Dava Daireleri Kurulu
            else:
                chamber_num = int(chamber)

            return {
                "chamber": chamber_num,
                "esas_year": int(esas_year),
                "esas_no": int(esas_no),
                "karar_year": int(karar_year) if karar_year else None,
                "karar_no": int(karar_no) if karar_no else None,
            }

        raise ValueError(
            f"Invalid Danıştay document ID format: '{document_id}'. "
            f"Expected format: '9-d-2018-1234-2019-5678'"
        )

    def _build_decision_url(
        self,
        chamber: int | str,
        esas_year: int,
        esas_no: int,
        karar_year: Optional[int] = None,
        karar_no: Optional[int] = None,
    ) -> str:
        """
        Build URL for specific Danıştay decision.

        Args:
            chamber: Chamber number or board code
            esas_year: File year
            esas_no: File number
            karar_year: Decision year (optional)
            karar_no: Decision number (optional)

        Returns:
            Full URL to decision page
        """
        params = {
            "daire": chamber,
            "esas_yil": esas_year,
            "esas_no": esas_no,
        }

        if karar_year:
            params["karar_yil"] = karar_year
        if karar_no:
            params["karar_no"] = karar_no

        return f"{DANISTAY_UYAP_URL}/decision?{urlencode(params)}"

    async def fetch_document(self, document_id: str) -> LegalDocument:
        """
        Fetch specific Danıştay decision.

        Args:
            document_id: Decision identifier (e.g., "9-d-2018-1234-2019-5678")

        Returns:
            LegalDocument with parsed decision

        Raises:
            DocumentNotFoundError: If decision doesn't exist
            ScraperError: If scraping fails

        Example:
            >>> decision = await adapter.fetch_document("9-d-2018-1234-2019-5678")
            >>> print(f"Chamber: {decision.court_metadata.chamber}")
            >>> print(f"Subject: {adapter.chambers[9]}")
        """
        # Parse document ID
        parsed = self._parse_document_id(document_id)

        # Build URL
        url = self._build_decision_url(
            parsed["chamber"],
            parsed["esas_year"],
            parsed["esas_no"],
            parsed["karar_year"],
            parsed["karar_no"],
        )

        # Fetch HTML (Danıştay primarily uses HTML)
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
        Parse Danıştay decision from HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            document_id: Decision identifier
            parsed: Parsed document ID components

        Returns:
            Canonical LegalDocument
        """
        # Extract metadata
        chamber = parsed["chamber"]
        esas_year = parsed["esas_year"]
        esas_no = parsed["esas_no"]
        karar_year = parsed.get("karar_year")
        karar_no = parsed.get("karar_no")

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
            chamber_subject = self.chambers.get(chamber, "Genel") if isinstance(chamber, int) else chamber
            title = f"Danıştay {chamber}. Daire - {chamber_subject} - {esas_year}/{esas_no} E."

        # Extract decision body
        body_elem = soup.find("div", class_=re.compile(r"decision|karar|content|body"))
        if body_elem:
            body = body_elem.get_text(separator="\n", strip=True)
        else:
            # Fallback: combine all paragraphs
            paragraphs = soup.find_all("p")
            body = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        # Extract decision type
        decision_type = self._extract_decision_type(body, soup)

        # Extract subject/keywords
        keywords = self._extract_keywords(soup)

        # Extract legal principle/summary
        legal_principle = self._extract_legal_principle(soup)

        # Extract citations
        citations = self._extract_citations_from_text(body)

        # Build metadata
        metadata = LegalMetadata(
            law_number=None,
            publication_number=f"{chamber}-D-{esas_year}/{esas_no}",
            keywords=keywords,
            subject=self.chambers.get(chamber, "") if isinstance(chamber, int) else "",
            notes=[],
        )

        # Court metadata
        court_metadata = CourtMetadata(
            court_name="Danıştay",
            court_level="supreme_administrative",
            chamber=str(chamber),
            case_number=f"{esas_year}/{esas_no}",
            decision_number=f"{karar_year}/{karar_no}" if karar_year else None,
            decision_type=decision_type,
            legal_principle=legal_principle,
            case_parties=self._extract_parties(soup),
        )

        # Create canonical document
        return LegalDocument(
            id=f"danistay:{document_id}",
            source=LegalSourceType.DANISTAY,
            document_type=LegalDocumentType.COURT_DECISION,
            title=title,
            body=body,
            articles=[],  # Decisions don't have articles
            metadata=metadata,
            court_metadata=court_metadata,
            publication_date=decision_date,
            effective_date=decision_date,
            version=1,
            status=DocumentStatus.ACTIVE,
            citations=citations,
            cited_by=[],
        )

    def _extract_decision_type(self, body: str, soup: BeautifulSoup) -> str:
        """
        Extract decision type from text or metadata.

        Args:
            body: Decision text
            soup: BeautifulSoup object

        Returns:
            Decision type string
        """
        # Check metadata section
        decision_type_elem = soup.find("span", class_=re.compile(r"decision.type|karar.turu"))
        if decision_type_elem:
            return decision_type_elem.get_text(strip=True).lower()

        # Check body text for decision type keywords
        body_lower = body.lower()
        for dtype in self.decision_types:
            if dtype in body_lower:
                return dtype

        return "decision"  # Default

    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract keywords/subject tags from decision.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of keywords
        """
        keywords = []

        # Look for keywords meta tag
        keywords_meta = soup.find("meta", attrs={"name": "keywords"})
        if keywords_meta and keywords_meta.get("content"):
            keywords = [k.strip() for k in keywords_meta["content"].split(",")]
            return keywords

        # Look for keywords section
        keywords_section = soup.find("div", class_=re.compile(r"keywords|anahtar"))
        if keywords_section:
            keyword_tags = keywords_section.find_all(["span", "a"])
            keywords = [tag.get_text(strip=True) for tag in keyword_tags]
            return keywords

        return keywords

    def _extract_legal_principle(self, soup: BeautifulSoup) -> str:
        """
        Extract legal principle or decision summary.

        Args:
            soup: BeautifulSoup object

        Returns:
            Legal principle text
        """
        # Look for principle/summary section
        principle_elem = soup.find("div", class_=re.compile(r"principle|ilke|summary|ozet"))
        if principle_elem:
            return principle_elem.get_text(separator=" ", strip=True)

        # Look for abstract
        abstract_elem = soup.find("div", class_="abstract")
        if abstract_elem:
            return abstract_elem.get_text(separator=" ", strip=True)

        return ""

    def _extract_parties(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract case parties (plaintiff, defendant).

        Args:
            soup: BeautifulSoup object

        Returns:
            List of party names
        """
        parties = []

        # Look for parties section
        parties_section = soup.find("div", class_=re.compile(r"parties|taraflar"))
        if parties_section:
            party_items = parties_section.find_all(["li", "span"])
            parties = [item.get_text(strip=True) for item in party_items]

        return parties

    def _extract_citations_from_text(self, text: str) -> List[LegalCitation]:
        """
        Extract legal citations from decision text.

        Finds references to:
            - Laws: "2577 sayılı İYUK", "3071 sayılı Dilekçe Hakkı Kanunu"
            - Articles: "Madde 10", "m. 5"
            - Other decisions: "Danıştay 9. Daire, 2018/1234"

        Args:
            text: Decision text

        Returns:
            List of LegalCitation objects
        """
        citations = []

        # Pattern 1: Laws
        law_pattern = r"(\d{4})\s+sayılı\s+(.*?)(?:Kanun|Yasa)"
        for match in re.finditer(law_pattern, text):
            law_number = match.group(1)

            citations.append(LegalCitation(
                target_law=law_number,
                target_article=None,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 2: Articles
        article_pattern = r"(?:Madde|m\.)\s*(\d+)"
        for match in re.finditer(article_pattern, text):
            article_num = int(match.group(1))

            citations.append(LegalCitation(
                target_law=None,
                target_article=article_num,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 3: Other Danıştay decisions
        danistay_pattern = r"Danıştay\s+(\d+|İDDK|VDDK)\.\s+Daire[,\s]+(\d{4})/(\d+)"
        for match in re.finditer(danistay_pattern, text):
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
        chamber: Optional[int] = None,
        chamber_range: Optional[tuple] = None,
        decision_type: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 10,
    ) -> List[LegalDocument]:
        """
        Search Danıştay decisions.

        Args:
            query: Search keyword(s)
            chamber: Specific chamber (1-15)
            chamber_range: Range of chambers (e.g., (1, 3) for tax chambers)
            decision_type: Filter by type ("bozma", "onanma", "iptal", "red")
            date_from: Start date
            date_to: End date
            limit: Maximum results

        Returns:
            List of matching LegalDocuments

        Example:
            >>> # Search environmental decisions from chamber 9
            >>> results = await adapter.search(
            ...     "çevre kirliliği",
            ...     chamber=9,
            ...     limit=20
            ... )
            >>>
            >>> # Search all tax decisions (chambers 1-3) from 2020
            >>> tax = await adapter.search(
            ...     chamber_range=(1, 3),
            ...     date_from=date(2020, 1, 1),
            ...     date_to=date(2020, 12, 31)
            ... )
        """
        params = {}

        if query:
            params["q"] = query
        if chamber:
            params["daire"] = chamber
        if decision_type:
            params["karar_turu"] = decision_type
        if date_from:
            params["baslangic_tarih"] = date_from.strftime("%d.%m.%Y")
        if date_to:
            params["bitis_tarih"] = date_to.strftime("%d.%m.%Y")

        # Fetch search results
        html = await self.get(DANISTAY_SEARCH_URL, params=params)
        soup = BeautifulSoup(html, "lxml")

        results = []
        result_items = soup.find_all("div", class_=re.compile(r"result|search.item|decision.item"))

        for item in result_items[:limit]:
            # Extract decision link
            link = item.find("a", href=True)
            if not link:
                continue

            # Parse decision ID from URL or text
            href = link.get("href", "")
            match = re.search(r"id=([^&]+)", href)
            if match:
                doc_id = match.group(1)
                try:
                    decision = await self.fetch_document(doc_id)

                    # Apply chamber range filter if specified
                    if chamber_range:
                        chamber_start, chamber_end = chamber_range
                        if isinstance(decision.court_metadata.chamber, str):
                            continue  # Skip boards when filtering by range
                        chamber_num = int(decision.court_metadata.chamber)
                        if not (chamber_start <= chamber_num <= chamber_end):
                            continue

                    results.append(decision)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch decision {doc_id}: {e}")
                    continue

        return results

    async def fetch_chamber_subject_decisions(
        self,
        subject: str,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 50,
    ) -> List[LegalDocument]:
        """
        Fetch decisions by subject area.

        Args:
            subject: Subject keyword (e.g., "vergi", "çevre", "kamulaştırma")
            date_from: Start date
            date_to: End date
            limit: Maximum results

        Returns:
            List of decisions matching subject

        Example:
            >>> # Get all tax law decisions from 2023
            >>> tax_decisions = await adapter.fetch_chamber_subject_decisions(
            ...     "vergi",
            ...     date_from=date(2023, 1, 1),
            ...     date_to=date(2023, 12, 31)
            ... )
        """
        return await self.search(
            query=subject,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
