"""SPK (Capital Markets Board) Adapter - Harvey/Legora CTO-Level
Production-grade adapter for fetching regulations, communiqués, and board decisions from spk.gov.tr

This adapter handles:
- Multiple SPK document types (Tebliğ, Kurul Kararı, Duyuru, Rehber)
- Rate limiting (30 requests/minute)
- Retry logic with exponential backoff
- Multiple fallback CSS selectors
- Pagination for search results
- Comprehensive metadata extraction
- Table and attachment detection
- Cache integration for performance
- Robust error handling and logging
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
from ..utils import (
    normalize_turkish_text, parse_turkish_date,
    TTLCache, retry
)
from ..presets import SOURCE_URLS, RATE_LIMITS

logger = logging.getLogger(__name__)


class SPKAdapter(SourceAdapter):
    """
    Production-grade adapter for SPK (Sermaye Piyasası Kurulu - Capital Markets Board).

    SPK is the regulatory authority for Turkish capital markets, publishing:
    - Tebliğ (Communiqués): Detailed regulations for market participants
    - Kurul Kararı (Board Decisions): Regulatory decisions on specific cases
    - Duyuru (Announcements): Public notices and updates
    - Rehber (Guidelines): Best practice guides for market participants

    Architecture:
    - Rate-limited requests (30/minute per SPK policy)
    - TTL cache (1 hour default) for repeated requests
    - Multiple fallback selectors for robust parsing
    - Comprehensive metadata extraction
    - Async/await for performance

    Examples:
        >>> adapter = SPKAdapter()
        >>> doc = await adapter.fetch_document("VII-128-1")  # Tebliğ number
        >>> search_results = await adapter.search_documents("halka arz")
        >>> decisions = await adapter.fetch_recent_decisions(limit=10)
    """

    # SPK endpoints for different document types
    ENDPOINTS = {
        'teblig': '/Sayfa/Dosya/Tebligler',
        'karar': '/Sayfa/Dosya/Kararlar',
        'duyuru': '/Sayfa/Dosya/Duyurular',
        'rehber': '/Sayfa/Dosya/Rehberler',
        'search': '/Arama',
        'detail': '/Sayfa/Dosya/{document_id}'
    }

    # Multiple fallback CSS selectors (websites change their HTML)
    SELECTORS = {
        'title': [
            'h1.document-title',
            'div.baslik h1',
            'h1',
            'div.page-title',
            'div.title h1'
        ],
        'content': [
            'div.document-content',
            'div.icerik-detay',
            'div.icerik',
            'div.content',
            'article.document',
            'div.main-content'
        ],
        'metadata': [
            'div.document-meta',
            'div.bilgi',
            'table.info',
            'div.metadata'
        ],
        'date': [
            'span.publish-date',
            'span.tarih',
            'div.date',
            'time'
        ],
        'attachments': [
            'div.attachments a',
            'div.ekler a',
            'ul.attachment-list a'
        ],
        'tables': [
            'table.data-table',
            'table',
            'div.table-wrapper table'
        ]
    }

    # Turkish month mapping for date parsing
    TURKISH_MONTHS = {
        'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4,
        'mayıs': 5, 'haziran': 6, 'temmuz': 7, 'ağustos': 8,
        'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
    }

    def __init__(self, cache_ttl: int = 3600, rate_limit: int = 30):
        """
        Initialize SPK adapter with caching and rate limiting.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            rate_limit: Maximum requests per minute (default: 30)
        """
        super().__init__("SPK Adapter", "2.0.0")
        self.base_url = SOURCE_URLS.get('spk', 'https://www.spk.gov.tr')

        # Initialize cache for performance
        self.cache = TTLCache(default_ttl=cache_ttl)

        # Rate limiting setup
        self.rate_limit = rate_limit
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # Session configuration
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

        logger.info(f"SPKAdapter initialized (cache_ttl={cache_ttl}s, rate_limit={rate_limit}/min)")

    async def _enforce_rate_limit(self):
        """
        Enforce rate limiting (30 requests/minute for SPK).

        Implements sliding window rate limiting to prevent 429 errors.
        """
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()

            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]

            # Check if we've hit the limit
            if len(self.request_times) >= self.rate_limit:
                # Calculate wait time until oldest request expires
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Recheck after waiting
                    now = asyncio.get_event_loop().time()
                    self.request_times = [t for t in self.request_times if now - t < 60]

            # Record this request
            self.request_times.append(now)

    @retry(max_attempts=3, backoff_factor=2.0)
    async def _fetch_html(self, url: str, **kwargs) -> str:
        """
        Fetch HTML from URL with retry logic and rate limiting.

        Args:
            url: Full URL to fetch
            **kwargs: Additional aiohttp parameters

        Returns:
            HTML content as string

        Raises:
            NetworkError: If all retry attempts fail
            RateLimitError: If rate limited by server
        """
        # Enforce rate limiting before request
        await self._enforce_rate_limit()

        # Check cache first
        cache_key = f"html:{url}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {url}")
            return cached

        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                async with session.get(url, **kwargs) as response:
                    # Handle different status codes
                    if response.status == 404:
                        raise DocumentNotFoundError(f"Document not found: {url}")
                    elif response.status == 429:
                        raise RateLimitError("Rate limited by SPK server")
                    elif response.status >= 400:
                        raise NetworkError(
                            f"HTTP {response.status} error for {url}",
                            error_code=f"HTTP_{response.status}",
                            recoverable=(response.status < 500)
                        )

                    html = await response.text()

                    # Validate response
                    if len(html) < 100:
                        raise ValidationError(f"Suspiciously short response ({len(html)} chars)")

                    # Cache successful response
                    self.cache.set(cache_key, html)
                    logger.debug(f"Fetched and cached {url} ({len(html)} chars)")

                    return html

        except asyncio.TimeoutError as e:
            raise NetworkError(
                f"Timeout fetching {url}",
                error_code="TIMEOUT",
                recoverable=True
            ) from e
        except aiohttp.ClientError as e:
            raise NetworkError(
                f"Network error fetching {url}: {str(e)}",
                error_code="CLIENT_ERROR",
                recoverable=True
            ) from e

    def _find_element_with_fallback(self, soup: BeautifulSoup, selector_key: str) -> Optional[Tag]:
        """
        Try multiple CSS selectors as fallback for robust parsing.

        Args:
            soup: BeautifulSoup object
            selector_key: Key in self.SELECTORS dict

        Returns:
            First matching Tag or None
        """
        selectors = self.SELECTORS.get(selector_key, [])

        for selector in selectors:
            try:
                # Try as CSS class/id first
                if '.' in selector or '#' in selector:
                    element = soup.select_one(selector)
                else:
                    # Try as tag name
                    element = soup.find(selector.split('.')[0])

                if element:
                    logger.debug(f"Found {selector_key} with selector: {selector}")
                    return element
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue

        logger.warning(f"No element found for {selector_key} with any selector")
        return None

    def _extract_spk_number(self, text: str) -> Optional[str]:
        """
        Extract SPK document number from text.

        SPK numbers have formats like:
        - VII-128.1 (Tebliğ)
        - 45/1234 (Karar)
        - 2023/DUYURU-156 (Duyuru)

        Args:
            text: Text to search

        Returns:
            SPK number or None
        """
        patterns = [
            r'([IVX]+-\d+(?:\.\d+)?)',  # VII-128.1
            r'(\d+/\d+)',                 # 45/1234
            r'(\d{4}/[A-Z]+-\d+)',       # 2023/DUYURU-156
            r'Sayı\s*:?\s*([^\s]+)',     # Sayı: XXX
            r'No\s*:?\s*([^\s]+)'        # No: XXX
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_publish_date(self, soup: BeautifulSoup, text: str) -> Optional[date]:
        """
        Extract publication date from document.

        Tries multiple strategies:
        1. Dedicated date elements (span.date, time, etc.)
        2. Text patterns ("26 Eylül 2004", "26.09.2004")
        3. Metadata tables

        Args:
            soup: BeautifulSoup object
            text: Full document text

        Returns:
            date object or None
        """
        # Strategy 1: Find date element
        date_elem = self._find_element_with_fallback(soup, 'date')
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            parsed = parse_turkish_date(date_text)
            if parsed:
                return parsed

        # Strategy 2: Search for date patterns in text
        # Turkish date: "26 Eylül 2004"
        turkish_pattern = r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})'
        match = re.search(turkish_pattern, text, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month = self.TURKISH_MONTHS.get(match.group(2).lower())
            year = int(match.group(3))
            if month:
                try:
                    return date(year, month, day)
                except ValueError:
                    pass

        # Numeric date: "26.09.2004" or "26/09/2004"
        numeric_pattern = r'(\d{1,2})[./](\d{1,2})[./](\d{4})'
        match = re.search(numeric_pattern, text)
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            try:
                return date(year, month, day)
            except ValueError:
                pass

        return None

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract tables from document with headers and data.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of table dicts with 'headers' and 'rows'
        """
        tables = []

        for selector in self.SELECTORS['tables']:
            found_tables = soup.select(selector)

            for idx, table in enumerate(found_tables):
                try:
                    # Extract headers
                    headers = []
                    thead = table.find('thead')
                    if thead:
                        for th in thead.find_all('th'):
                            headers.append(th.get_text(strip=True))
                    else:
                        # Try first row as headers
                        first_row = table.find('tr')
                        if first_row:
                            for th in first_row.find_all(['th', 'td']):
                                headers.append(th.get_text(strip=True))

                    # Extract rows
                    rows = []
                    tbody = table.find('tbody') or table
                    for tr in tbody.find_all('tr')[1 if not thead else 0:]:
                        row_data = []
                        for td in tr.find_all('td'):
                            row_data.append(td.get_text(strip=True))
                        if row_data:  # Skip empty rows
                            rows.append(row_data)

                    if rows:  # Only add tables with content
                        tables.append({
                            'index': idx,
                            'headers': headers,
                            'rows': rows,
                            'row_count': len(rows),
                            'column_count': len(headers) or (len(rows[0]) if rows else 0)
                        })

                except Exception as e:
                    logger.warning(f"Error extracting table {idx}: {e}")
                    continue

        logger.debug(f"Extracted {len(tables)} tables")
        return tables

    def _extract_attachments(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract attachment links (PDF, DOC, XLS, etc.).

        Args:
            soup: BeautifulSoup object

        Returns:
            List of attachment dicts with 'name', 'url', 'type'
        """
        attachments = []

        # Find all attachment links
        for selector in self.SELECTORS['attachments']:
            links = soup.select(selector)

            for link in links:
                href = link.get('href', '')
                if not href:
                    continue

                # Full URL
                full_url = urljoin(self.base_url, href)

                # Detect file type
                file_ext = None
                if href.lower().endswith('.pdf'):
                    file_ext = 'pdf'
                elif href.lower().endswith(('.doc', '.docx')):
                    file_ext = 'doc'
                elif href.lower().endswith(('.xls', '.xlsx')):
                    file_ext = 'xls'

                if file_ext:
                    attachments.append({
                        'name': link.get_text(strip=True),
                        'url': full_url,
                        'type': file_ext
                    })

        logger.debug(f"Extracted {len(attachments)} attachments")
        return attachments

    def _classify_document_type(self, title: str, content: str, url: str) -> str:
        """
        Classify SPK document type based on title, content, and URL.

        Args:
            title: Document title
            content: Document content
            url: Document URL

        Returns:
            Document type: 'teblig', 'karar', 'duyuru', or 'rehber'
        """
        title_lower = title.lower()
        content_lower = content.lower()
        url_lower = url.lower()

        # Check URL first (most reliable)
        if 'teblig' in url_lower:
            return 'teblig'
        elif 'karar' in url_lower:
            return 'karar'
        elif 'duyuru' in url_lower:
            return 'duyuru'
        elif 'rehber' in url_lower:
            return 'rehber'

        # Check title
        if 'tebliğ' in title_lower:
            return 'teblig'
        elif 'karar' in title_lower or 'kurul kararı' in title_lower:
            return 'karar'
        elif 'duyuru' in title_lower:
            return 'duyuru'
        elif 'rehber' in title_lower or 'kılavuz' in title_lower:
            return 'rehber'

        # Check for SPK number patterns
        if re.search(r'[IVX]+-\d+', title):
            return 'teblig'
        elif re.search(r'\d+/\d+', title):
            return 'karar'

        # Default to teblig
        return 'teblig'

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch SPK document by ID with comprehensive metadata extraction.

        Args:
            document_id: SPK document identifier (e.g., "VII-128-1", "45/1234")
            **kwargs: Additional parameters

        Returns:
            Dict with extracted document data:
                - title: Document title
                - content: Full text content
                - url: Source URL
                - spk_number: SPK document number
                - document_type: teblig/karar/duyuru/rehber
                - publish_date: Publication date
                - tables: Extracted tables
                - attachments: PDF/DOC/XLS attachments
                - metadata: Additional metadata

        Raises:
            DocumentNotFoundError: If document doesn't exist
            ParsingError: If parsing fails
        """
        # Build URL
        url = f"{self.base_url}{self.ENDPOINTS['detail'].format(document_id=document_id)}"

        try:
            # Fetch HTML
            html = await self._fetch_html(url)
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title with fallback selectors
            title_elem = self._find_element_with_fallback(soup, 'title')
            title = title_elem.get_text(strip=True) if title_elem else ''

            if not title:
                raise ParsingError("Could not extract document title", error_code="NO_TITLE")

            # Extract content with fallback selectors
            content_elem = self._find_element_with_fallback(soup, 'content')
            if not content_elem:
                raise ParsingError("Could not extract document content", error_code="NO_CONTENT")

            # Get full text content
            content = content_elem.get_text(separator='\n', strip=True)
            content = normalize_turkish_text(content, preserve_case=True)

            # Extract SPK document number
            spk_number = self._extract_spk_number(title) or self._extract_spk_number(content[:500])

            # Classify document type
            doc_type = self._classify_document_type(title, content, url)

            # Extract publication date
            publish_date = self._extract_publish_date(soup, content)

            # Extract tables
            tables = self._extract_tables(soup)

            # Extract attachments
            attachments = self._extract_attachments(soup)

            # Extract additional metadata
            metadata = {}
            meta_elem = self._find_element_with_fallback(soup, 'metadata')
            if meta_elem:
                # Try to extract key-value pairs from metadata section
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
                'spk_number': spk_number,
                'document_type': doc_type,
                'publish_date': publish_date.isoformat() if publish_date else None,
                'tables': tables,
                'attachments': attachments,
                'metadata': metadata,
                'source': 'SPK',
                'fetched_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Successfully fetched SPK document: {spk_number or document_id} ({doc_type})")
            return result

        except (DocumentNotFoundError, ParsingError):
            raise
        except Exception as e:
            raise ParsingError(
                f"Error fetching SPK document {document_id}: {str(e)}",
                error_code="FETCH_ERROR",
                context={'document_id': document_id, 'url': url}
            ) from e

    async def search_documents(self, query: str, page: int = 1, limit: int = 20, **kwargs) -> List[Dict[str, Any]]:
        """
        Search SPK documents with pagination support.

        Args:
            query: Search query (Turkish)
            page: Page number (1-indexed)
            limit: Results per page
            **kwargs: Additional search parameters

        Returns:
            List of search result dicts with 'title', 'url', 'snippet', 'type'
        """
        search_url = f"{self.base_url}{self.ENDPOINTS['search']}?q={quote(query)}&page={page}"

        try:
            html = await self._fetch_html(search_url)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Try multiple search result selectors
            result_selectors = [
                'div.search-result',
                'div.arama-sonuc',
                'li.result',
                'div.item'
            ]

            items = []
            for selector in result_selectors:
                items = soup.select(selector)
                if items:
                    break

            for item in items[:limit]:
                try:
                    # Extract title and link
                    link = item.find('a')
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    href = link.get('href', '')
                    full_url = urljoin(self.base_url, href)

                    # Extract snippet
                    snippet = ''
                    snippet_elem = item.find(['p', 'div'], class_=re.compile('snippet|ozet|description'))
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)

                    # Classify type from URL or title
                    doc_type = self._classify_document_type(title, snippet, full_url)

                    results.append({
                        'title': title,
                        'url': full_url,
                        'snippet': snippet,
                        'type': doc_type
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
        """Template method: pass through preprocessed data"""
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        """
        Transform raw SPK data to canonical LegalDocument format.

        Args:
            raw_data: Dict from fetch_document()
            document_type: Optional document type override
            **kwargs: Additional parameters

        Returns:
            LegalDocument with full metadata
        """
        from ..core.canonical_schema import (
            LegalDocument, Metadata, SourceType,
            JurisdictionType, LegalHierarchy, EffectivityStatus
        )

        # Map SPK document types to canonical types
        type_mapping = {
            'teblig': DocumentType.TEBLIG,
            'karar': DocumentType.KURUL_KARARI,
            'duyuru': DocumentType.DUYURU,
            'rehber': DocumentType.REHBER
        }

        doc_type = type_mapping.get(raw_data.get('document_type'), DocumentType.TEBLIG)

        # Create metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.TEBLIG,
            source=SourceType.SPK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            decision_number=raw_data.get('spk_number'),
            publish_date=raw_data.get('publish_date'),
            fetch_timestamp=raw_data.get('fetched_at')
        )

        # Create document
        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', ''),
            tables=raw_data.get('tables', []),
            attachments=raw_data.get('attachments', [])
        )


__all__ = ['SPKAdapter']
