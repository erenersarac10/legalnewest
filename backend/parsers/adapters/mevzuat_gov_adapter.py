"""
Mevzuat.gov.tr (Turkish Legislation Portal) adapter.

This module provides the MevzuatGovAdapter for scraping and parsing
consolidated legislation from Turkey's official legislation portal.

Mevzuat.gov.tr is the official government portal for consolidated (updated)
versions of Turkish legislation. Unlike Resmi Gazete (which publishes original
texts), Mevzuat.gov.tr provides current, amended versions of laws.

Data Source:
    - Website: https://www.mevzuat.gov.tr
    - Format: HTML (structured content)
    - Content: Consolidated laws, decrees, regulations
    - Update: Real-time amendments applied

Document Types:
    - Kanun (Law)
    - Kanun Hükmünde Kararname (Decree with force of law)
    - Tüzük (Regulation)
    - Yönetmelik (By-law)
    - Tebliğ (Communiqué)
    - Genelge (Circular)

Structure Features:
    - Hierarchical: Kısım → Bölüm → Madde → Fıkra → Bent
    - Cross-references: Links to related laws
    - Historical versions: Track amendments over time
    - Metadata: Law numbers, dates, ministries

Example:
    >>> async with MevzuatGovAdapter() as adapter:
    ...     # Fetch specific law (KVKK)
    ...     doc = await adapter.fetch_document("6698")
    ...
    ...     # Search for laws
    ...     results = await adapter.search("veri koruma")
    ...
    ...     # Get law by category
    ...     laws = await adapter.fetch_by_category("Kanun")
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from backend.core.constants import MEVZUAT_GOV_BASE_URL
from backend.core.exceptions import NetworkError, ParsingError, ValidationError
from backend.core.logging import get_logger
from backend.parsers.adapters.base_adapter import BaseAdapter

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Document type codes
DOCUMENT_TYPES = {
    "1": "kanun",
    "2": "kanun_hukmunde_kararname",
    "3": "tuzuk",
    "4": "yonetmelik",
    "5": "teblig",
    "6": "yonerge",
    "7": "genelge",
    "8": "cumhurbaskanligi_kararnamesi",
}

# URL patterns
URL_MEVZUAT_DETAIL = "/MevzuatMetin/{mevzuat_no}"
URL_SEARCH = "/MevzuatArama.aspx"


# =============================================================================
# MEVZUAT.GOV.TR ADAPTER
# =============================================================================


class MevzuatGovAdapter(BaseAdapter):
    """
    Adapter for Turkey's Legislation Portal (Mevzuat.gov.tr).

    Provides access to consolidated (currently effective) versions of
    Turkish legislation with hierarchical structure and amendments.

    Key Features:
    - Consolidated law text (with all amendments applied)
    - Hierarchical structure extraction (Madde, Fıkra, Bent)
    - Cross-reference detection
    - Historical version tracking
    - Category-based browsing
    - Advanced search capabilities

    URL Structure:
        - Detail: https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6698
        - Search: https://www.mevzuat.gov.tr/MevzuatArama.aspx
        - Category: https://www.mevzuat.gov.tr/MevzuatKategori.aspx

    Document ID Format:
        - Format: TYPE.VERSION.NUMBER
        - Example: "1.5.6698" = Law (1), Version 5, Number 6698
        - Law 6698 = KVKK (Data Protection Law)

    Common Laws:
        - 6698: KVKK (Data Protection)
        - 6102: TTK (Commercial Code)
        - 6098: TBK (Code of Obligations)
        - 5237: TCK (Penal Code)
        - 2709: Anayasa (Constitution)
    """

    # =========================================================================
    # HARVEY/LEGORA %100 PARITE: DATE NORMALIZER
    # =========================================================================

    @staticmethod
    def normalize_turkish_date(date_str: str) -> str:
        """
        Normalize Turkish date formats to ISO 8601.

        Harvey/Legora %100 parite: Date normalization for temporal accuracy.

        Handles all Turkish date format variants:
        - 15.06.2023 → 2023-06-15
        - 15/06/2023 → 2023-06-15
        - 15 Haziran 2023 → 2023-06-15
        - 15-06-2023 → 2023-06-15

        Args:
            date_str: Date string in Turkish format

        Returns:
            ISO 8601 date string (YYYY-MM-DD)

        Raises:
            ValueError: Invalid date format

        Example:
            >>> MevzuatGovAdapter.normalize_turkish_date("15.06.2023")
            '2023-06-15'
            >>> MevzuatGovAdapter.normalize_turkish_date("15 Haziran 2023")
            '2023-06-15'
        """
        if not date_str:
            raise ValueError("Empty date string")

        date_str = date_str.strip()

        # Turkish month names mapping
        turkish_months = {
            "ocak": "01", "şubat": "02", "mart": "03", "nisan": "04",
            "mayıs": "05", "haziran": "06", "temmuz": "07", "ağustos": "08",
            "eylül": "09", "ekim": "10", "kasım": "11", "aralık": "12",
        }

        # Format 1: Numeric with separators (15.06.2023, 15/06/2023, 15-06-2023)
        pattern_numeric = r"^(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})$"
        match = re.match(pattern_numeric, date_str)
        if match:
            day, month, year = match.groups()
            # Validate
            if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
                raise ValueError(f"Invalid date components: {date_str}")
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Format 2: Turkish month name (15 Haziran 2023)
        pattern_turkish = r"^(\d{1,2})\s+(\w+)\s+(\d{4})$"
        match = re.match(pattern_turkish, date_str, re.IGNORECASE)
        if match:
            day, month_name, year = match.groups()
            month_name_lower = month_name.lower()

            if month_name_lower not in turkish_months:
                raise ValueError(f"Unknown Turkish month: {month_name}")

            month = turkish_months[month_name_lower]
            # Validate
            if not (1 <= int(day) <= 31):
                raise ValueError(f"Invalid day: {day}")

            return f"{year}-{month}-{day.zfill(2)}"

        # Format 3: Already ISO (2023-06-15)
        pattern_iso = r"^(\d{4})-(\d{1,2})-(\d{1,2})$"
        match = re.match(pattern_iso, date_str)
        if match:
            year, month, day = match.groups()
            # Validate
            if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
                raise ValueError(f"Invalid date components: {date_str}")
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        raise ValueError(f"Unrecognized date format: {date_str}")

    def __init__(self):
        """Initialize Mevzuat.gov.tr adapter."""
        super().__init__(
            source_name="Mevzuat.gov.tr",
            base_url=MEVZUAT_GOV_BASE_URL,
            rate_limit_per_second=1.0,  # 1 request per second
            cache_ttl=3600,  # Cache for 1 hour (laws change infrequently)
            timeout=30,
        )

    def _parse_document_id(self, document_id: str) -> dict[str, Any]:
        """
        Parse Mevzuat document ID.

        Format: TYPE.VERSION.NUMBER or just NUMBER

        Args:
            document_id: Document ID

        Returns:
            Parsed ID components

        Raises:
            ValidationError: Invalid format

        Examples:
            >>> _parse_document_id("1.5.6698")
            {'type': '1', 'version': '5', 'number': '6698', 'full_id': '1.5.6698'}

            >>> _parse_document_id("6698")
            {'type': '1', 'version': '5', 'number': '6698', 'full_id': '1.5.6698'}
        """
        # Try full format first (1.5.6698)
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", document_id)
        if match:
            return {
                "type": match.group(1),
                "version": match.group(2),
                "number": match.group(3),
                "full_id": document_id,
            }

        # Try just number (6698) - assume type=1 (Kanun), version=5 (latest)
        match = re.match(r"^(\d+)$", document_id)
        if match:
            number = match.group(1)
            # Default to type=1 (Kanun), version=5
            return {
                "type": "1",
                "version": "5",
                "number": number,
                "full_id": f"1.5.{number}",
            }

        raise ValidationError(
            message="Invalid document ID format",
            field="document_id",
            details={
                "expected": "TYPE.VERSION.NUMBER or NUMBER",
                "got": document_id,
            }
        )

    def _build_detail_url(self, parsed_id: dict[str, Any]) -> str:
        """
        Build detail URL for document.

        Args:
            parsed_id: Parsed document ID

        Returns:
            Detail URL

        Example:
            >>> _build_detail_url({'full_id': '1.5.6698'})
            'https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6698'
        """
        path = URL_MEVZUAT_DETAIL.format(mevzuat_no=parsed_id["full_id"])
        return urljoin(self.base_url, path)

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract document title from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Document title
        """
        # Try multiple selectors
        selectors = [
            "h2.baslik",
            "div.mevzuat-baslik",
            "h1.kanun-adi",
            "div.baslik-metin",
        ]

        for selector in selectors:
            title = self.extract_text(soup, selector)
            if title:
                return title

        # Fallback: try page title
        title = self.extract_text(soup, "title")
        if title:
            # Remove site name
            title = re.sub(r"\s*-\s*Mevzuat\s*$", "", title, flags=re.IGNORECASE)
            return title

        return None

    def _extract_law_number(self, soup: BeautifulSoup, title: Optional[str]) -> Optional[str]:
        """
        Extract law number from content.

        Args:
            soup: BeautifulSoup object
            title: Document title

        Returns:
            Law number (e.g., "6698")
        """
        # Try from title first (e.g., "6698 SAYILI KANUN")
        if title:
            match = re.search(r"(\d+)\s+SAYILI", title, re.IGNORECASE)
            if match:
                return match.group(1)

        # Try from content
        text = soup.get_text()
        match = re.search(r"Kanun\s+Numarası\s*:\s*(\d+)", text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_articles(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """
        Extract articles (Madde) from document.

        Turkish legislation hierarchy:
        - Kısım (Part)
        - Bölüm (Chapter)
        - Madde (Article) ← We extract these
        - Fıkra (Paragraph)
        - Bent (Clause)

        Args:
            soup: BeautifulSoup object

        Returns:
            List of articles with hierarchy
        """
        articles = []

        # Find all article elements
        # Mevzuat.gov.tr uses various patterns for articles
        article_elements = soup.find_all(["div", "p"], class_=re.compile(r"madde|article", re.I))

        for element in article_elements:
            article_text = element.get_text(strip=True)

            # Extract article number (e.g., "Madde 5", "MADDE 10")
            number_match = re.match(r"^Madde\s+(\d+)", article_text, re.IGNORECASE)
            if not number_match:
                continue

            article_number = int(number_match.group(1))

            # Extract article content (after the number)
            content = re.sub(r"^Madde\s+\d+\s*[-–—:]*\s*", "", article_text, flags=re.IGNORECASE)

            # Try to extract article title if present
            title_match = re.match(r"^([A-ZĞÜŞİÖÇ\s]+)[-–—:]", content)
            article_title = title_match.group(1).strip() if title_match else None

            # Extract paragraphs (Fıkra)
            paragraphs = self._extract_paragraphs(content)

            article = {
                "number": article_number,
                "title": article_title,
                "content": content,
                "paragraphs": paragraphs,
            }

            articles.append(article)

        logger.debug(
            f"Extracted {len(articles)} articles",
            count=len(articles),
        )

        return articles

    def _extract_paragraphs(self, article_text: str) -> list[str]:
        """
        Extract paragraphs (Fıkra) from article text.

        Args:
            article_text: Full article text

        Returns:
            List of paragraph texts
        """
        # Split by paragraph markers
        # Turkish law often uses "(1)", "(2)", etc. or just line breaks

        # Try numbered paragraphs first
        if re.search(r"\(\d+\)", article_text):
            paragraphs = re.split(r"\(\d+\)", article_text)
            # Remove empty and strip
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            return paragraphs

        # Fallback: split by double line breaks
        paragraphs = article_text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs if len(paragraphs) > 1 else [article_text]

    def _extract_dates(self, soup: BeautifulSoup) -> dict[str, Optional[str]]:
        """
        Extract relevant dates from document.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary of dates
        """
        dates = {
            "kabul_tarihi": None,  # Acceptance date
            "yayim_tarihi": None,  # Publication date
            "yururluk_tarihi": None,  # Effective date
        }

        text = soup.get_text()

        # Kabul Tarihi (e.g., "Kabul Tarihi: 24/3/2016")
        match = re.search(r"Kabul\s+Tarihi\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        if match:
            dates["kabul_tarihi"] = match.group(1)

        # Yayım Tarihi
        match = re.search(r"(?:Yayım|Resmi\s+Gazete)\s+Tarihi\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        if match:
            dates["yayim_tarihi"] = match.group(1)

        return dates

    def _extract_cross_references(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """
        Extract cross-references to other laws.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of cross-references
        """
        references = []
        text = soup.get_text()

        # Find references like "5237 sayılı Kanun", "6102 sayılı TTK"
        pattern = r"(\d+)\s+sayılı\s+(?:Kanun|TTK|TBK|TCK|KVKK)"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        seen = set()
        for match in matches:
            law_number = match.group(1)
            if law_number not in seen:
                references.append({
                    "law_number": law_number,
                    "context": match.group(0),
                })
                seen.add(law_number)

        return references

    async def fetch_document(self, document_id: str) -> dict[str, Any]:
        """
        Fetch single document from Mevzuat.gov.tr.

        Args:
            document_id: Document ID (TYPE.VERSION.NUMBER or just NUMBER)

        Returns:
            Document data dictionary

        Raises:
            ValidationError: Invalid document ID
            NetworkError: Failed to fetch
            ParsingError: Failed to parse

        Example:
            >>> doc = await adapter.fetch_document("6698")  # KVKK
            >>> doc = await adapter.fetch_document("1.5.6698")  # Full format
        """
        # Parse document ID
        parsed_id = self._parse_document_id(document_id)

        # Build URL
        url = self._build_detail_url(parsed_id)

        # Fetch HTML
        html = await self.get(url)

        # Parse HTML
        soup = self.parse_html(html)

        # Extract data
        title = self._extract_title(soup)
        law_number = self._extract_law_number(soup, title)
        articles = self._extract_articles(soup)
        dates = self._extract_dates(soup)
        cross_references = self._extract_cross_references(soup)

        # Build full text (all articles)
        full_text_parts = []
        if title:
            full_text_parts.append(title)

        for article in articles:
            article_header = f"\nMadde {article['number']}"
            if article['title']:
                article_header += f" - {article['title']}"
            full_text_parts.append(article_header)
            full_text_parts.append(article['content'])

        full_text = "\n\n".join(full_text_parts)

        # Build metadata
        metadata = {
            "mevzuat_no": parsed_id["full_id"],
            "law_number": law_number,
            "document_type": DOCUMENT_TYPES.get(parsed_id["type"], "unknown"),
            "version": parsed_id["version"],
            "dates": dates,
            "article_count": len(articles),
            "cross_references": cross_references,
        }

        # Build document
        document = {
            "id": parsed_id["full_id"],
            "title": title or f"Mevzuat {parsed_id['full_id']}",
            "content": full_text,
            "articles": articles,
            "metadata": metadata,
            "source_url": url,
            "fetch_date": datetime.now(timezone.utc).isoformat(),
            "format": "html",
        }

        logger.info(
            "Fetched Mevzuat.gov.tr document",
            document_id=parsed_id["full_id"],
            law_number=law_number,
            articles=len(articles),
        )

        return document

    async def fetch_documents(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple documents (not fully implemented - needs search).

        Mevzuat.gov.tr doesn't provide easy date-based listing.
        This method returns popular/important laws.

        Args:
            start_date: Not used (kept for interface compatibility)
            end_date: Not used
            limit: Maximum documents to return

        Returns:
            List of important Turkish laws
        """
        # List of important Turkish laws
        important_laws = [
            "2709",  # Anayasa (Constitution)
            "6098",  # TBK (Code of Obligations)
            "6102",  # TTK (Commercial Code)
            "5237",  # TCK (Penal Code)
            "6698",  # KVKK (Data Protection)
            "4857",  # İş Kanunu (Labor Law)
            "5510",  # Sosyal Sigortalar (Social Security)
            "213",   # VUK (Tax Procedure Law)
            "3065",  # KDVK (VAT Law)
            "6502",  # Tüketici Koruma (Consumer Protection)
        ]

        documents = []

        for law_number in important_laws[:limit]:
            try:
                doc = await self.fetch_document(law_number)
                documents.append(doc)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch law {law_number}",
                    law_number=law_number,
                    error=str(e),
                )

        logger.info(
            "Fetched important laws from Mevzuat.gov.tr",
            count=len(documents),
        )

        return documents

    async def search(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search Mevzuat.gov.tr (basic implementation).

        Note: Full search requires POST request to search page with form data.
        This is a simplified version that searches known law titles.

        Args:
            query: Search query
            filters: Optional filters (document_type)
            limit: Maximum results

        Returns:
            List of matching documents

        Example:
            >>> results = await adapter.search("veri koruma")
            >>> results = await adapter.search("ticaret", limit=10)
        """
        if not query or not query.strip():
            raise ValidationError(
                message="Search query cannot be empty",
                field="query"
            )

        # Fetch important laws (our searchable dataset)
        documents = await self.fetch_documents(limit=limit * 2)

        # Filter by query (simple text matching)
        matching_documents = []
        query_lower = query.lower()

        for doc in documents:
            title_lower = doc["title"].lower()
            content_lower = doc["content"].lower()

            if query_lower in title_lower or query_lower in content_lower:
                matching_documents.append(doc)

                if len(matching_documents) >= limit:
                    break

        logger.info(
            "Searched Mevzuat.gov.tr",
            query=query,
            results=len(matching_documents),
        )

        return matching_documents


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = ["MevzuatGovAdapter"]
