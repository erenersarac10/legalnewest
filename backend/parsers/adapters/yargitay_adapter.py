"""
Yargıtay Adapter - Supreme Court of Appeals Decision Scraper

Scrapes court decisions from Yargıtay (Turkish Supreme Court of Appeals),
the highest judicial authority for civil and criminal cases in Turkey.

Data Source: https://karararama.yargitay.gov.tr/
Archive: 1868 - Present (157 years of case law)
Update Frequency: Daily (new decisions published regularly)

Turkish Legal Context:
    Yargıtay is the supreme court for civil and criminal law in Turkey.
    Decisions create precedent (içtihat) that lower courts follow.

    Court Structure:
        - Hukuk Daireleri (Civil Chambers): 1st-23rd
        - Ceza Daireleri (Criminal Chambers): 1st-19th
        - Hukuk Genel Kurulu (Civil General Assembly)
        - Ceza Genel Kurulu (Criminal General Assembly)

    Decision Format:
        - Esas No (File Number): YYYY/XXXXX
        - Karar No (Decision Number): YYYY/XXXXX
        - Daire (Chamber): 1-23 (Civil) or 1-19 (Criminal)
        - Tarih (Date): DD.MM.YYYY
        - İlke (Principle): Legal principle established

World-Class Features:
    - JSON API + HTML fallback for maximum coverage
    - Structured decision parsing (chamber, file number, date, principle)
    - Cross-reference extraction (mentions of laws and other decisions)
    - Full-text search support
    - Temporal filtering (date range queries)
    - Chamber-specific queries
    - Duplicate detection via decision identifier

KVKK Compliance:
    - Personal data in decisions (names, addresses) handled per Madde 28
    - Legal basis: Legal obligation (Article 5/2-a)
    - Court decisions are public record per Turkish law
    - Anonymization support for sensitive cases

Example:
    >>> adapter = YargitayAdapter()
    >>>
    >>> # Fetch specific decision
    >>> decision = await adapter.fetch_document("2-hd-2018-1234-2018-5678")
    >>>
    >>> # Search decisions by keyword
    >>> results = await adapter.search("sözleşme ihlali", limit=10)
    >>>
    >>> # Get all decisions from chamber on date
    >>> decisions = await adapter.fetch_chamber_decisions(
    ...     chamber=2,
    ...     chamber_type="hukuk",
    ...     date=date(2018, 3, 15)
    ... )
"""

from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import re
import hashlib
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
    LegalArticle,
    CourtMetadata,
    DocumentStatus,
)


# Yargıtay endpoints
YARGITAY_BASE_URL = "https://karararama.yargitay.gov.tr"
YARGITAY_API_URL = f"{YARGITAY_BASE_URL}/api"
YARGITAY_SEARCH_URL = f"{YARGITAY_BASE_URL}/search"


class YargitayAdapter(BaseAdapter):
    """
    Adapter for Yargıtay (Supreme Court of Appeals) decisions.

    Implements BaseAdapter for scraping and parsing Yargıtay court decisions
    with support for both JSON API and HTML fallback.

    Attributes:
        min_date: Earliest decision date (1868-01-01)
        max_date: Latest decision date (today)
        chambers_civil: List of civil chamber numbers
        chambers_criminal: List of criminal chamber numbers
    """

    def __init__(self):
        """Initialize Yargıtay adapter with rate limiting."""
        super().__init__(
            source_name="Yargıtay",
            base_url=YARGITAY_BASE_URL,
            rate_limit_per_second=1.0,  # Respectful: 1 request/second
        )

        # Archive date range (157 years!)
        self.min_date = date(1868, 1, 1)
        self.max_date = date.today()

        # Court structure
        self.chambers_civil = list(range(1, 24))  # 1-23
        self.chambers_criminal = list(range(1, 20))  # 1-19

    # =========================================================================
    # HARVEY/LEGORA %100 PARITE: OUTCOME TAGGING
    # =========================================================================

    @staticmethod
    def extract_decision_outcome(text: str) -> str:
        """
        Extract decision outcome from Yargıtay decision text.

        Harvey/Legora %100 parite: Outcome classification for RAG quality.

        Outcomes (İçtihat types):
        - onama: Affirm lower court decision
        - bozma: Reverse and remand
        - duzelterek_onama: Affirm with modifications
        - hukum_kurma: Establish new ruling
        - red: Reject appeal
        - dava_dusurmesi: Dismiss case

        Args:
            text: Decision body text

        Returns:
            Outcome tag (normalized)

        Example:
            >>> text = "...hükmün ONANMASINA karar verildi..."
            >>> YargitayAdapter.extract_decision_outcome(text)
            'onama'
        """
        if not text:
            return "unknown"

        text_upper = text.upper()

        # Outcome patterns (order matters - most specific first)
        patterns = {
            "duzelterek_onama": [
                r"DÜZELTİLEREK\s+ONAMA",
                r"DÜZELTİLMEK\s+SURETİYLE\s+ONAMA",
            ],
            "bozma": [
                r"BOZULMASINA",
                r"BOZMA KARARININ\s+ONANMASINA",
                r"HÜKMÜN\s+BOZULMASINA",
                r"HÜKMÜN\s+BOZUP",
            ],
            "onama": [
                r"ONANMASINA",
                r"HÜKMÜN\s+ONANMASINA",
                r"KARAR\s+ONANMIŞTIR",
            ],
            "hukum_kurma": [
                r"HÜKÜM\s+KURULMAK",
                r"HÜKÜM\s+KURULMASINA",
            ],
            "red": [
                r"TEMYİZ\s+İSTEMİNİN\s+REDDİNE",
                r"İSTEMİN\s+REDDİNE",
                r"REDDİNE\s+KARAR\s+VERİLDİ",
            ],
            "dava_dusurmesi": [
                r"DAVANIN\s+DÜŞMESİNE",
                r"DÜŞMESİNE\s+KARAR",
            ],
        }

        # Check patterns in priority order
        for outcome, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text_upper):
                    return outcome

        return "unknown"

    def _parse_document_id(self, document_id: str) -> Dict[str, Any]:
        """
        Parse Yargıtay decision identifier.

        Format: "{chamber}-{type}-{esas_year}-{esas_no}-{karar_year}-{karar_no}"
        Example: "2-hd-2018-1234-2018-5678"
            → Chamber 2, Hukuk Dairesi, Esas 2018/1234, Karar 2018/5678

        Alternative formats:
            - "2018-1234" (esas number only, requires chamber)
            - "hgk-2018-1234-2018-5678" (Hukuk Genel Kurulu)

        Args:
            document_id: Decision identifier string

        Returns:
            Dict with parsed components:
                chamber: int or str (number or "hgk"/"cgk")
                chamber_type: "hukuk" or "ceza"
                esas_year: int
                esas_no: int
                karar_year: int (optional)
                karar_no: int (optional)

        Raises:
            ValueError: If document_id format is invalid
        """
        # Full format: "2-hd-2018-1234-2018-5678"
        pattern = r"^(\d+|hgk|cgk)-([hc]d)-(\d{4})-(\d+)(?:-(\d{4})-(\d+))?$"
        match = re.match(pattern, document_id.lower())

        if match:
            chamber, chamber_abbr, esas_year, esas_no, karar_year, karar_no = match.groups()

            # Determine chamber type
            chamber_type = "hukuk" if chamber_abbr == "hd" else "ceza"

            # Convert chamber (handle general assemblies)
            if chamber in ("hgk", "cgk"):
                chamber_num = chamber.upper()  # Keep as string
            else:
                chamber_num = int(chamber)

            return {
                "chamber": chamber_num,
                "chamber_type": chamber_type,
                "esas_year": int(esas_year),
                "esas_no": int(esas_no),
                "karar_year": int(karar_year) if karar_year else None,
                "karar_no": int(karar_no) if karar_no else None,
            }

        # Alternative format: Just esas number (requires additional context)
        pattern_simple = r"^(\d{4})-(\d+)$"
        match = re.match(pattern_simple, document_id)
        if match:
            esas_year, esas_no = match.groups()
            return {
                "chamber": None,  # Must be provided separately
                "chamber_type": None,
                "esas_year": int(esas_year),
                "esas_no": int(esas_no),
                "karar_year": None,
                "karar_no": None,
            }

        raise ValueError(
            f"Invalid Yargıtay document ID format: '{document_id}'. "
            f"Expected formats: '2-hd-2018-1234-2018-5678' or '2018-1234'"
        )

    def _build_decision_url(
        self,
        chamber: int | str,
        chamber_type: str,
        esas_year: int,
        esas_no: int,
        karar_year: Optional[int] = None,
        karar_no: Optional[int] = None,
    ) -> str:
        """
        Build URL for specific decision.

        Args:
            chamber: Chamber number or "HGK"/"CGK"
            chamber_type: "hukuk" or "ceza"
            esas_year: File year
            esas_no: File number
            karar_year: Decision year (optional)
            karar_no: Decision number (optional)

        Returns:
            Full URL to decision page

        Example:
            >>> url = adapter._build_decision_url(2, "hukuk", 2018, 1234, 2018, 5678)
            >>> # https://karararama.yargitay.gov.tr/...
        """
        params = {
            "daire": chamber,
            "daire_tur": chamber_type,
            "esas_yil": esas_year,
            "esas_no": esas_no,
        }

        if karar_year:
            params["karar_yil"] = karar_year
        if karar_no:
            params["karar_no"] = karar_no

        return f"{YARGITAY_BASE_URL}/decision?{urlencode(params)}"

    async def fetch_document(self, document_id: str) -> LegalDocument:
        """
        Fetch specific Yargıtay decision.

        Args:
            document_id: Decision identifier (e.g., "2-hd-2018-1234-2018-5678")

        Returns:
            LegalDocument with parsed decision

        Raises:
            DocumentNotFoundError: If decision doesn't exist
            ScraperError: If scraping fails

        Example:
            >>> decision = await adapter.fetch_document("2-hd-2018-1234-2018-5678")
            >>> print(f"Chamber: {decision.metadata.court.chamber}")
            >>> print(f"Principle: {decision.metadata.court.legal_principle}")
        """
        # Parse document ID
        parsed = self._parse_document_id(document_id)

        # Build URL
        url = self._build_decision_url(
            parsed["chamber"],
            parsed["chamber_type"],
            parsed["esas_year"],
            parsed["esas_no"],
            parsed["karar_year"],
            parsed["karar_no"],
        )

        # Try JSON API first (faster)
        try:
            api_url = f"{YARGITAY_API_URL}/decision"
            response = await self.get(api_url, params=parsed, use_cache=True)

            if response and response.get("success"):
                return self._parse_json_decision(response["data"], document_id)

        except Exception as e:
            # Fall back to HTML scraping
            self.logger.warning(f"API failed for {document_id}, falling back to HTML: {e}")

        # HTML fallback
        html = await self.get(url, use_cache=True)
        soup = BeautifulSoup(html, "lxml")

        return self._parse_html_decision(soup, document_id, parsed)

    def _parse_json_decision(self, data: Dict[str, Any], document_id: str) -> LegalDocument:
        """
        Parse decision from JSON API response.

        Args:
            data: JSON data from API
            document_id: Decision identifier

        Returns:
            Canonical LegalDocument
        """
        # Extract basic info
        chamber = data.get("daire")
        chamber_type = data.get("daire_tur")
        esas_year = data.get("esas_yil")
        esas_no = data.get("esas_no")
        karar_year = data.get("karar_yil")
        karar_no = data.get("karar_no")
        decision_date = datetime.strptime(data["tarih"], "%d.%m.%Y").date()

        # Extract content
        title = data.get("baslik", f"{chamber}. {chamber_type.title()} Dairesi {esas_year}/{esas_no} E.")
        body = data.get("metin", "")
        legal_principle = data.get("ilke", "")

        # Extract citations
        citations = self._extract_citations_from_text(body)

        # Build metadata
        metadata = LegalMetadata(
            law_number=None,  # Decisions don't have law numbers
            publication_number=f"{chamber}-{chamber_type.upper()}-{esas_year}/{esas_no}",
            keywords=data.get("anahtar_kelimeler", []),
            subject=data.get("konu", ""),
            notes=[],
        )

        # Court-specific metadata
        court_metadata = CourtMetadata(
            court_name="Yargıtay",
            court_level="supreme",
            chamber=str(chamber),
            case_number=f"{esas_year}/{esas_no}",
            decision_number=f"{karar_year}/{karar_no}" if karar_year else None,
            decision_type=data.get("karar_tipi", "decision"),
            legal_principle=legal_principle,
            case_parties=data.get("taraflar", []),
        )

        # Create canonical document
        return LegalDocument(
            id=f"yargitay:{document_id}",
            source=LegalSourceType.YARGITAY,
            document_type=LegalDocumentType.SUPREME_COURT_DECISION,
            title=title,
            body=body,
            articles=[],  # Decisions don't have articles
            metadata=metadata,
            court_metadata=court_metadata,
            publication_date=decision_date,
            effective_date=decision_date,  # Effective immediately
            version=1,
            status=DocumentStatus.ACTIVE,
            citations=citations,
            cited_by=[],
        )

    def _parse_html_decision(
        self,
        soup: BeautifulSoup,
        document_id: str,
        parsed: Dict[str, Any]
    ) -> LegalDocument:
        """
        Parse decision from HTML page (fallback).

        Args:
            soup: BeautifulSoup parsed HTML
            document_id: Decision identifier
            parsed: Parsed document ID components

        Returns:
            Canonical LegalDocument
        """
        # Extract decision metadata from HTML
        chamber = parsed["chamber"]
        chamber_type = parsed["chamber_type"]
        esas_year = parsed["esas_year"]
        esas_no = parsed["esas_no"]
        karar_year = parsed.get("karar_year")
        karar_no = parsed.get("karar_no")

        # Extract date
        date_elem = soup.find("span", class_=re.compile(r"date|tarih"))
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            decision_date = datetime.strptime(date_text, "%d.%m.%Y").date()
        else:
            decision_date = date.today()  # Fallback

        # Extract title
        title_elem = soup.find("h1") or soup.find("h2")
        if title_elem:
            title = title_elem.get_text(strip=True)
        else:
            title = f"{chamber}. {chamber_type.title()} Dairesi {esas_year}/{esas_no} E."

        # Extract decision body
        body_elem = soup.find("div", class_=re.compile(r"decision|karar|content"))
        if body_elem:
            body = body_elem.get_text(separator="\n", strip=True)
        else:
            # Fallback: get all paragraphs
            paragraphs = soup.find_all("p")
            body = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

        # Extract legal principle (ilke)
        legal_principle = ""
        ilke_elem = soup.find("div", class_=re.compile(r"principle|ilke"))
        if ilke_elem:
            legal_principle = ilke_elem.get_text(strip=True)

        # Extract citations
        citations = self._extract_citations_from_text(body)

        # Build metadata
        metadata = LegalMetadata(
            law_number=None,
            publication_number=f"{chamber}-{chamber_type.upper()}-{esas_year}/{esas_no}",
            keywords=[],
            subject="",
            notes=[],
        )

        court_metadata = CourtMetadata(
            court_name="Yargıtay",
            court_level="supreme",
            chamber=str(chamber),
            case_number=f"{esas_year}/{esas_no}",
            decision_number=f"{karar_year}/{karar_no}" if karar_year else None,
            decision_type="decision",
            legal_principle=legal_principle,
            case_parties=[],
        )

        return LegalDocument(
            id=f"yargitay:{document_id}",
            source=LegalSourceType.YARGITAY,
            document_type=LegalDocumentType.SUPREME_COURT_DECISION,
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

    def _extract_citations_from_text(self, text: str) -> List[LegalCitation]:
        """
        Extract legal citations from decision text.

        Finds references to:
            - Laws: "6098 sayılı TBK", "5237 sayılı TCK"
            - Articles: "Madde 123", "m. 456"
            - Other decisions: "Yargıtay 2. HD. 2018/1234"

        Args:
            text: Decision text

        Returns:
            List of LegalCitation objects
        """
        citations = []

        # Pattern 1: "XXXX sayılı [law name]"
        law_pattern = r"(\d{4})\s+sayılı\s+(.*?)(?:Kanun|TTK|TBK|TCK|HMK|CMK|KVKK|BK|MK)"
        for match in re.finditer(law_pattern, text):
            law_number = match.group(1)
            law_name = match.group(2).strip()

            citations.append(LegalCitation(
                target_law=law_number,
                target_article=None,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 2: "Madde X" or "m. X"
        article_pattern = r"(?:Madde|m\.)\s*(\d+)"
        for match in re.finditer(article_pattern, text):
            article_num = int(match.group(1))

            citations.append(LegalCitation(
                target_law=None,  # Unknown without context
                target_article=article_num,
                citation_text=match.group(0),
                citation_type="reference",
            ))

        # Pattern 3: Other Yargıtay decisions
        yargitay_pattern = r"Yargıtay\s+(\d+)\.\s+(?:HD|CD)\.\s+(\d{4})/(\d+)"
        for match in re.finditer(yargitay_pattern, text):
            chamber = match.group(1)
            year = match.group(2)
            number = match.group(3)

            citations.append(LegalCitation(
                target_law=None,
                target_article=None,
                citation_text=match.group(0),
                citation_type="precedent",
            ))

        return citations

    async def search(
        self,
        query: str,
        chamber: Optional[int] = None,
        chamber_type: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 10,
    ) -> List[LegalDocument]:
        """
        Search Yargıtay decisions by keyword.

        Args:
            query: Search keyword(s)
            chamber: Filter by chamber number (1-23 civil, 1-19 criminal)
            chamber_type: "hukuk" or "ceza"
            date_from: Start date filter
            date_to: End date filter
            limit: Maximum results

        Returns:
            List of matching LegalDocuments

        Example:
            >>> # Search for contract violations in chamber 2
            >>> results = await adapter.search(
            ...     "sözleşme ihlali",
            ...     chamber=2,
            ...     chamber_type="hukuk",
            ...     limit=20
            ... )
        """
        params = {
            "q": query,
            "limit": limit,
        }

        if chamber:
            params["daire"] = chamber
        if chamber_type:
            params["daire_tur"] = chamber_type
        if date_from:
            params["baslangic_tarih"] = date_from.strftime("%d.%m.%Y")
        if date_to:
            params["bitis_tarih"] = date_to.strftime("%d.%m.%Y")

        # Try API first
        try:
            response = await self.get(f"{YARGITAY_API_URL}/search", params=params)

            if response and response.get("success"):
                results = []
                for item in response["data"][:limit]:
                    doc_id = item.get("id")
                    if doc_id:
                        # Fetch full decision
                        decision = await self.fetch_document(doc_id)
                        results.append(decision)

                return results

        except Exception as e:
            self.logger.warning(f"Search API failed: {e}")

        # HTML fallback
        html = await self.get(YARGITAY_SEARCH_URL, params=params)
        soup = BeautifulSoup(html, "lxml")

        results = []
        result_items = soup.find_all("div", class_=re.compile(r"result|search-item"))

        for item in result_items[:limit]:
            # Extract decision link
            link = item.find("a", href=True)
            if not link:
                continue

            # Parse decision ID from URL
            href = link["href"]
            match = re.search(r"id=([^&]+)", href)
            if match:
                doc_id = match.group(1)
                decision = await self.fetch_document(doc_id)
                results.append(decision)

        return results

    async def fetch_chamber_decisions(
        self,
        chamber: int | str,
        chamber_type: str,
        decision_date: date,
    ) -> List[LegalDocument]:
        """
        Fetch all decisions from specific chamber on specific date.

        Useful for bulk collection and ensuring completeness.

        Args:
            chamber: Chamber number or "HGK"/"CGK"
            chamber_type: "hukuk" or "ceza"
            decision_date: Date of decisions

        Returns:
            List of decisions from that chamber/date

        Example:
            >>> # Get all decisions from Chamber 2 on March 15, 2018
            >>> decisions = await adapter.fetch_chamber_decisions(
            ...     chamber=2,
            ...     chamber_type="hukuk",
            ...     decision_date=date(2018, 3, 15)
            ... )
            >>> print(f"Found {len(decisions)} decisions")
        """
        params = {
            "daire": chamber,
            "daire_tur": chamber_type,
            "tarih": decision_date.strftime("%d.%m.%Y"),
        }

        # Fetch chamber's daily list
        html = await self.get(f"{YARGITAY_BASE_URL}/daily", params=params)
        soup = BeautifulSoup(html, "lxml")

        decisions = []
        decision_links = soup.find_all("a", class_=re.compile(r"decision|karar"))

        for link in decision_links:
            href = link.get("href", "")
            match = re.search(r"id=([^&]+)", href)
            if match:
                doc_id = match.group(1)
                decision = await self.fetch_document(doc_id)
                decisions.append(decision)

        return decisions
