"""GİB (Revenue Administration) Adapter - Harvey/Legora CTO-Level
Production-grade adapter for fetching tax regulations, circulars, and rulings from gib.gov.tr

This adapter handles:
- Genel Tebliğ (General Communiqués): Official tax regulations
- Sirküler (Circulars): Procedural guidelines for tax offices
- Özelge (Tax Rulings): Binding decisions on specific tax questions
- VUK Tebliğleri (Tax Procedural Law Communiqués)
- Rate limiting, caching, retry logic
- Multiple document format handling (HTML, PDF metadata)
"""
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import asyncio
import re
import logging
from datetime import datetime, date
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, quote

from ..core import SourceAdapter, ParsingResult, DocumentType
from ..core.exceptions import (
    NetworkError, ParsingError, ValidationError,
    RateLimitError, DocumentNotFoundError
)
from ..utils import normalize_turkish_text, parse_turkish_date, TTLCache, retry
from ..presets import SOURCE_URLS

logger = logging.getLogger(__name__)


class GIBAdapter(SourceAdapter):
    """
    Production-grade adapter for GİB (Gelir İdaresi Başkanlığı - Revenue Administration).

    GİB is Turkey's tax authority, publishing three main document types:

    1. **Genel Tebliğ** (General Communiqués):
       - Numbered series for each tax law (e.g., "123 Seri No.lu VUK Genel Tebliği")
       - Contains detailed procedures and interpretations
       - Binding on all taxpayers and tax offices

    2. **Sirküler** (Circulars):
       - Internal instructions for tax offices
       - Procedural guidance for tax auditors
       - Not always published publicly

    3. **Özelge** (Tax Rulings):
       - Binding decisions on specific tax questions
       - Numbered format: "B.07.1.GİB.4.34.16.01-XXX-123"
       - Can be used as precedent for similar cases

    Architecture:
    - Rate-limited requests (60/minute for GİB)
    - TTL cache for document metadata
    - PDF metadata extraction (many GİB docs are PDF)
    - Series number parsing ("123 Seri No.lu")
    - Tax law classification (GVK, KVK, VUK, KDVK, etc.)
    """

    # GİB document endpoints
    ENDPOINTS = {
        'genel_teblig': '/Yardim-ve-Kaynaklar/Mevzuat-Rehberi/Genel-Tebligler',
        'sirkular': '/Yardim-ve-Kaynaklar/Mevzuat-Rehberi/Sirkulerler',
        'ozelge': '/Ozelge-Sistemi',
        'search': '/Arama',
        'detail': '/node/{document_id}'
    }

    # Fallback CSS selectors
    SELECTORS = {
        'title': [
            'h1.page-title',
            'div.field-name-title h1',
            'h1',
            'div.baslik'
        ],
        'content': [
            'div.field-name-body',
            'div.field-item',
            'div.content',
            'article.node',
            'div.icerik'
        ],
        'metadata': [
            'div.field-name-field-metadata',
            'table.meta-table',
            'div.document-info'
        ],
        'date': [
            'span.date-display-single',
            'time',
            'span.tarih',
            'div.publish-date'
        ],
        'pdf_link': [
            'a.pdf-download',
            'a[href$=".pdf"]',
            'div.attachment a'
        ],
        'series': [
            'span.seri-no',
            'div.series-number'
        ]
    }

    # Tax law abbreviations for classification
    TAX_LAWS = {
        'GVK': 'Gelir Vergisi Kanunu',
        'KVK': 'Kurumlar Vergisi Kanunu',
        'VUK': 'Vergi Usul Kanunu',
        'KDVK': 'Katma Değer Vergisi Kanunu',
        'ÖTV': 'Özel Tüketim Vergisi',
        'MTV': 'Motorlu Taşıtlar Vergisi',
        'BSMV': 'Banka ve Sigorta Muameleleri Vergisi',
        'AATUHK': 'Amme Alacaklarının Tahsil Usulü Hakkında Kanun',
        'VDMK': 'Vergi Dairesi Müdürleri Kanunu'
    }

    def __init__(self, cache_ttl: int = 3600, rate_limit: int = 60):
        """
        Initialize GİB adapter.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            rate_limit: Maximum requests per minute (default: 60)
        """
        super().__init__("GİB Adapter", "2.0.0")
        self.base_url = SOURCE_URLS.get('gib', 'https://www.gib.gov.tr')

        # Caching
        self.cache = TTLCache(default_ttl=cache_ttl)

        # Rate limiting
        self.rate_limit = rate_limit
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # Session config
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'tr-TR,tr;q=0.9',
            'Connection': 'keep-alive'
        }

        logger.info(f"GİBAdapter initialized (cache_ttl={cache_ttl}s, rate_limit={rate_limit}/min)")

    async def _enforce_rate_limit(self):
        """Enforce rate limiting (60 requests/minute for GİB)."""
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()
            self.request_times = [t for t in self.request_times if now - t < 60]

            if len(self.request_times) >= self.rate_limit:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    now = asyncio.get_event_loop().time()
                    self.request_times = [t for t in self.request_times if now - t < 60]

            self.request_times.append(now)

    @retry(max_attempts=3, backoff_factor=2.0)
    async def _fetch_html(self, url: str, **kwargs) -> str:
        """Fetch HTML with retry logic and caching."""
        await self._enforce_rate_limit()

        cache_key = f"html:{url}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {url}")
            return cached

        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                async with session.get(url, **kwargs) as response:
                    if response.status == 404:
                        raise DocumentNotFoundError(f"Document not found: {url}")
                    elif response.status == 429:
                        raise RateLimitError("Rate limited by GİB server")
                    elif response.status >= 400:
                        raise NetworkError(
                            f"HTTP {response.status} error for {url}",
                            error_code=f"HTTP_{response.status}",
                            recoverable=(response.status < 500)
                        )

                    html = await response.text()

                    if len(html) < 100:
                        raise ValidationError(f"Suspiciously short response ({len(html)} chars)")

                    self.cache.set(cache_key, html)
                    logger.debug(f"Fetched and cached {url} ({len(html)} chars)")
                    return html

        except asyncio.TimeoutError as e:
            raise NetworkError("Timeout fetching URL", error_code="TIMEOUT", recoverable=True) from e
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error: {str(e)}", error_code="CLIENT_ERROR", recoverable=True) from e

    def _find_element_with_fallback(self, soup: BeautifulSoup, selector_key: str) -> Optional[Tag]:
        """Try multiple CSS selectors as fallback."""
        for selector in self.SELECTORS.get(selector_key, []):
            try:
                if '.' in selector or '#' in selector:
                    element = soup.select_one(selector)
                else:
                    element = soup.find(selector.split('.')[0])

                if element:
                    logger.debug(f"Found {selector_key} with selector: {selector}")
                    return element
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue

        logger.warning(f"No element found for {selector_key}")
        return None

    def _extract_series_number(self, text: str) -> Optional[str]:
        """
        Extract series number from GİB document.

        Formats:
        - "123 Seri No.lu Gelir Vergisi Genel Tebliği"
        - "Seri No: 456"
        - "VUK-789"

        Args:
            text: Text to search

        Returns:
            Series number or None
        """
        patterns = [
            r'(\d+)\s+[Ss]eri\s+[Nn]o',           # 123 Seri No
            r'[Ss]eri\s+[Nn]o\s*:?\s*(\d+)',      # Seri No: 123
            r'([A-Z]+)-(\d+)',                     # VUK-123
            r'No\s*:?\s*(\d+)',                    # No: 123
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return first capturing group that's a number
                groups = match.groups()
                for group in groups:
                    if group and group.isdigit():
                        return group

        return None

    def _extract_ozelge_number(self, text: str) -> Optional[str]:
        """
        Extract Özelge (tax ruling) number.

        Format: "B.07.1.GİB.4.34.16.01-XXX-123"

        Args:
            text: Text to search

        Returns:
            Özelge number or None
        """
        # GİB Özelge pattern
        pattern = r'B\.\d+\.\d+\.G[İI]B\.\d+\.\d+\.\d+\.\d+-[A-Z0-9]+-\d+'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

        # Alternative simpler pattern
        pattern2 = r'Özelge\s+[Nn]o\s*:?\s*([A-Z0-9\.-]+)'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            return match2.group(1)

        return None

    def _classify_tax_law(self, title: str, content: str) -> Optional[str]:
        """
        Classify which tax law the document relates to.

        Args:
            title: Document title
            content: Document content (first 1000 chars)

        Returns:
            Tax law abbreviation (GVK, KVK, etc.) or None
        """
        text = (title + ' ' + content[:1000]).upper()

        # Check each tax law
        for abbr, full_name in self.TAX_LAWS.items():
            if abbr in text or full_name.upper() in text:
                return abbr

        return None

    def _classify_document_type(self, title: str, url: str, series_number: Optional[str], ozelge_number: Optional[str]) -> str:
        """
        Classify GİB document type.

        Args:
            title: Document title
            url: Document URL
            series_number: Extracted series number
            ozelge_number: Extracted özelge number

        Returns:
            'genel_teblig', 'sirkular', or 'ozelge'
        """
        title_lower = title.lower()
        url_lower = url.lower()

        # Özelge (tax ruling)
        if ozelge_number or 'özelge' in title_lower or 'ozelge' in url_lower:
            return 'ozelge'

        # Sirküler (circular)
        if 'sirküler' in title_lower or 'sirkular' in title_lower or 'sirkular' in url_lower:
            return 'sirkular'

        # Genel Tebliğ (general communiqué)
        if series_number or 'genel tebliğ' in title_lower or 'teblig' in url_lower:
            return 'genel_teblig'

        # Default to genel tebliğ
        return 'genel_teblig'

    def _extract_publish_date(self, soup: BeautifulSoup, text: str) -> Optional[date]:
        """Extract publication date from document."""
        # Try dedicated date element
        date_elem = self._find_element_with_fallback(soup, 'date')
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            parsed = parse_turkish_date(date_text)
            if parsed:
                return parsed

        # Try text patterns
        # "26 Eylül 2004"
        turkish_pattern = r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})'
        match = re.search(turkish_pattern, text, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            year = int(match.group(3))

            month_map = {
                'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
                'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
                'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
            }
            month = month_map.get(month_name)
            if month:
                try:
                    return date(year, month, day)
                except ValueError:
                    pass

        # "26.09.2004" or "26/09/2004"
        numeric_pattern = r'(\d{1,2})[./](\d{1,2})[./](\d{4})'
        match = re.search(numeric_pattern, text)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                return date(year, month, day)
            except ValueError:
                pass

        return None

    def _extract_pdf_link(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract PDF download link if available."""
        pdf_elem = self._find_element_with_fallback(soup, 'pdf_link')
        if pdf_elem:
            href = pdf_elem.get('href', '')
            if href:
                return urljoin(self.base_url, href)

        return None

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch GİB document with comprehensive metadata extraction.

        Args:
            document_id: GİB document ID or node number
            **kwargs: Additional parameters

        Returns:
            Dict with extracted document data:
                - title: Document title
                - content: Full text content
                - url: Source URL
                - series_number: Series number (for Genel Tebliğ)
                - ozelge_number: Özelge number (for tax rulings)
                - document_type: genel_teblig/sirkular/ozelge
                - tax_law: Related tax law (GVK, KVK, etc.)
                - publish_date: Publication date
                - pdf_url: PDF download link
                - metadata: Additional metadata

        Raises:
            DocumentNotFoundError: If document doesn't exist
            ParsingError: If parsing fails
        """
        url = f"{self.base_url}{self.ENDPOINTS['detail'].format(document_id=document_id)}"

        try:
            html = await self._fetch_html(url)
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title_elem = self._find_element_with_fallback(soup, 'title')
            title = title_elem.get_text(strip=True) if title_elem else ''

            if not title:
                raise ParsingError("Could not extract document title", error_code="NO_TITLE")

            # Extract content
            content_elem = self._find_element_with_fallback(soup, 'content')
            if not content_elem:
                raise ParsingError("Could not extract document content", error_code="NO_CONTENT")

            content = content_elem.get_text(separator='\n', strip=True)
            content = normalize_turkish_text(content, preserve_case=True)

            # Extract series number
            series_number = self._extract_series_number(title) or self._extract_series_number(content[:500])

            # Extract özelge number
            ozelge_number = self._extract_ozelge_number(title) or self._extract_ozelge_number(content[:500])

            # Classify document type
            doc_type = self._classify_document_type(title, url, series_number, ozelge_number)

            # Classify related tax law
            tax_law = self._classify_tax_law(title, content)

            # Extract publish date
            publish_date = self._extract_publish_date(soup, content)

            # Extract PDF link
            pdf_url = self._extract_pdf_link(soup)

            # Extract metadata
            metadata = {}
            meta_elem = self._find_element_with_fallback(soup, 'metadata')
            if meta_elem:
                for row in meta_elem.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        metadata[key] = value

            result = {
                'title': title,
                'content': content,
                'url': url,
                'series_number': series_number,
                'ozelge_number': ozelge_number,
                'document_type': doc_type,
                'tax_law': tax_law,
                'publish_date': publish_date.isoformat() if publish_date else None,
                'pdf_url': pdf_url,
                'metadata': metadata,
                'source': 'GİB',
                'fetched_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Successfully fetched GİB document: {series_number or ozelge_number or document_id} ({doc_type})")
            return result

        except (DocumentNotFoundError, ParsingError):
            raise
        except Exception as e:
            raise ParsingError(
                f"Error fetching GİB document {document_id}: {str(e)}",
                error_code="FETCH_ERROR",
                context={'document_id': document_id, 'url': url}
            ) from e

    async def search_documents(self, query: str, doc_type: Optional[str] = None, page: int = 1, limit: int = 20, **kwargs) -> List[Dict[str, Any]]:
        """
        Search GİB documents with optional type filtering.

        Args:
            query: Search query
            doc_type: Filter by type ('genel_teblig', 'sirkular', 'ozelge')
            page: Page number
            limit: Results per page
            **kwargs: Additional parameters

        Returns:
            List of search result dicts
        """
        # Build search URL with type filter
        search_url = f"{self.base_url}{self.ENDPOINTS['search']}?q={quote(query)}&page={page}"

        if doc_type:
            search_url += f"&type={doc_type}"

        try:
            html = await self._fetch_html(search_url)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Try multiple selectors for search results
            result_selectors = [
                'div.search-result',
                'div.view-content div.views-row',
                'li.search-result',
                'article.node-teaser'
            ]

            items = []
            for selector in result_selectors:
                items = soup.select(selector)
                if items:
                    break

            for item in items[:limit]:
                try:
                    link = item.find('a')
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    href = link.get('href', '')
                    full_url = urljoin(self.base_url, href)

                    snippet = ''
                    snippet_elem = item.find(['p', 'div'], class_=re.compile('snippet|teaser|summary'))
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)

                    # Extract numbers
                    series_num = self._extract_series_number(title + ' ' + snippet)
                    ozelge_num = self._extract_ozelge_number(title + ' ' + snippet)

                    # Classify type
                    result_type = self._classify_document_type(title, full_url, series_num, ozelge_num)

                    results.append({
                        'title': title,
                        'url': full_url,
                        'snippet': snippet,
                        'type': result_type,
                        'series_number': series_num,
                        'ozelge_number': ozelge_num
                    })

                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue

            logger.info(f"Search '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        """Template method: pass through"""
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        """Transform GİB data to canonical LegalDocument format."""
        from ..core.canonical_schema import (
            LegalDocument, Metadata, SourceType,
            JurisdictionType, LegalHierarchy, EffectivityStatus
        )

        # Map GİB types to canonical types
        type_mapping = {
            'genel_teblig': DocumentType.TEBLIG,
            'sirkular': DocumentType.GENELGE,
            'ozelge': DocumentType.OZELGE
        }

        doc_type = type_mapping.get(raw_data.get('document_type'), DocumentType.TEBLIG)

        # Create metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.TEBLIG,
            source=SourceType.GIB,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            document_number=raw_data.get('series_number') or raw_data.get('ozelge_number'),
            publish_date=raw_data.get('publish_date'),
            fetch_timestamp=raw_data.get('fetched_at'),
            related_law=raw_data.get('tax_law')
        )

        # Create document
        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', ''),
            pdf_url=raw_data.get('pdf_url')
        )


__all__ = ['GIBAdapter']
