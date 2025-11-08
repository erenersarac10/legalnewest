"""KVKK (Personal Data Protection Authority) Adapter - Harvey/Legora CTO-Level
Production-grade adapter for fetching decisions, guidelines, and breach notifications from kvkk.gov.tr

This adapter handles:
- Kurul Kararları (Board Decisions): Administrative fine decisions, warnings
- Rehberler (Guidelines): Best practice guides for data controllers
- Tebliğler (Communiqués): Regulatory communications
- Veri İhlali Bildirimleri (Data Breach Notifications): Public breach notices
- İdari Para Cezası (Administrative Fines): Penalty decisions
- Robust error handling, caching, retry logic
"""
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import asyncio
import re
import logging
from datetime import datetime, date
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, quote
from decimal import Decimal

from ..core import SourceAdapter, ParsingResult, DocumentType
from ..core.exceptions import (
    NetworkError, ParsingError, ValidationError,
    RateLimitError, DocumentNotFoundError
)
from ..utils import normalize_turkish_text, parse_turkish_date, TTLCache, retry
from ..presets import SOURCE_URLS

logger = logging.getLogger(__name__)


class KVKKAdapter(SourceAdapter):
    """
    Production-grade adapter for KVKK (Kişisel Verilerin Korunması Kurumu - Personal Data Protection Authority).

    KVKK is Turkey's data protection authority, responsible for:
    - Enforcing KVKK Law No. 6698 (Turkey's GDPR equivalent)
    - Issuing administrative fines for data protection violations
    - Publishing guidelines for data controllers and processors
    - Maintaining public registry of data breach notifications
    - Issuing opinions on data protection matters

    Document Types:
    1. **Kurul Kararı** (Board Decision):
       - E/K numbering: "2023/123" (E: Esas/Case), "2023/456" (K: Karar/Decision)
       - Contains violation analysis, legal reasoning, sanctions
       - Administrative fines can reach millions of TL

    2. **Rehber** (Guideline):
       - Sector-specific guidance (healthcare, finance, e-commerce, etc.)
       - Technical and organizational measures
       - Compliance checklists

    3. **Veri İhlali Bildirimi** (Data Breach Notification):
       - Public notifications of significant breaches
       - Number of affected individuals
       - Breach type and response measures

    Architecture:
    - Rate limiting (30/min for KVKK)
    - TTL cache for decisions
    - E/K number extraction
    - Fine amount parsing (Turkish number format: "1.234.567,89 TL")
    - Violation type classification
    - KVKK article citation extraction (Law 6698)
    """

    # KVKK endpoints
    ENDPOINTS = {
        'karar': '/Icerik/6742/Kararlar',
        'rehber': '/Icerik/6699/Rehberler',
        'teblig': '/Icerik/6700/Tebligler',
        'veri_ihlali': '/Icerik/6879/Veri-Ihlal-Bildirimleri',
        'search': '/Arama',
        'detail': '/Icerik/{document_id}'
    }

    # Fallback selectors
    SELECTORS = {
        'title': [
            'h1.page-header',
            'div.icerik-baslik h1',
            'h1',
            'div.title'
        ],
        'content': [
            'div.icerik-detay',
            'div.content',
            'div.document-content',
            'article.node'
        ],
        'metadata': [
            'div.karar-bilgi',
            'table.meta-info',
            'div.document-meta'
        ],
        'date': [
            'span.karar-tarihi',
            'span.date',
            'time'
        ],
        'decision_info': [
            'div.karar-no',
            'span.esas-karar',
            'table.decision-numbers'
        ],
        'fine_amount': [
            'span.ceza-miktari',
            'div.idari-para-cezasi',
            'td.fine'
        ]
    }

    # KVKK Law 6698 articles (for citation detection)
    KVKK_ARTICLES = {
        '4': 'İlgili Kişinin Hakları',
        '5': 'Veri Güvenliği',
        '6': 'Veri İşleme Şartları',
        '7': 'Özel Nitelikli Kişisel Veriler',
        '10': 'Veri Sorumlusu',
        '12': 'Veri İşleyen',
        '15': 'İdari Yaptırımlar'
    }

    # Violation types
    VIOLATION_TYPES = {
        'aydinlatma': 'Aydınlatma Yükümlülüğü İhlali',
        'guvenlik': 'Veri Güvenliği İhlali',
        'riza': 'Açık Rıza Alınmaması',
        'silme': 'Silme/İmha Yükümlülüğü İhlali',
        'aktarim': 'Yurtdışı Aktarım İhlali',
        'kayit': 'VERBİS Kayıt Yükümlülüğü İhlali'
    }

    def __init__(self, cache_ttl: int = 3600, rate_limit: int = 30):
        """
        Initialize KVKK adapter.

        Args:
            cache_ttl: Cache time-to-live in seconds
            rate_limit: Max requests per minute
        """
        super().__init__("KVKK Adapter", "2.0.0")
        self.base_url = SOURCE_URLS.get('kvkk', 'https://www.kvkk.gov.tr')

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

        logger.info(f"KVKKAdapter initialized (cache_ttl={cache_ttl}s, rate_limit={rate_limit}/min)")

    async def _enforce_rate_limit(self):
        """Enforce rate limiting (30/min for KVKK)."""
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
        """Fetch HTML with retry and caching."""
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
                        raise RateLimitError("Rate limited by KVKK server")
                    elif response.status >= 400:
                        raise NetworkError(
                            f"HTTP {response.status} for {url}",
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
            raise NetworkError("Timeout", error_code="TIMEOUT", recoverable=True) from e
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
                    logger.debug(f"Found {selector_key} with: {selector}")
                    return element
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue

        logger.warning(f"No element found for {selector_key}")
        return None

    def _extract_esas_karar_numbers(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract E (Esas) and K (Karar) numbers from KVKK decision.

        Format: "Esas No: 2023/123" and "Karar No: 2023/456"

        Args:
            text: Text to search

        Returns:
            Tuple of (esas_no, karar_no)
        """
        esas_no = None
        karar_no = None

        # Esas (Case number)
        esas_patterns = [
            r'Esas\s+(?:No|Sayısı)\s*:?\s*(\d{4}/\d+)',
            r'E\s*:?\s*(\d{4}/\d+)',
            r'Esas\s*:?\s*(\d{4}/\d+)'
        ]
        for pattern in esas_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                esas_no = match.group(1)
                break

        # Karar (Decision number)
        karar_patterns = [
            r'Karar\s+(?:No|Sayısı)\s*:?\s*(\d{4}/\d+)',
            r'K\s*:?\s*(\d{4}/\d+)',
            r'Karar\s*:?\s*(\d{4}/\d+)'
        ]
        for pattern in karar_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                karar_no = match.group(1)
                break

        return esas_no, karar_no

    def _extract_fine_amount(self, soup: BeautifulSoup, text: str) -> Optional[Decimal]:
        """
        Extract administrative fine amount in Turkish Lira.

        Turkish number format: "1.234.567,89 TL"

        Args:
            soup: BeautifulSoup object
            text: Full text

        Returns:
            Decimal amount or None
        """
        # Try dedicated fine element
        fine_elem = self._find_element_with_fallback(soup, 'fine_amount')
        if fine_elem:
            fine_text = fine_elem.get_text(strip=True)
        else:
            # Search in text
            fine_pattern = r'(?:İdari\s+Para\s+Cezası|Ceza\s+Miktarı)\s*:?\s*([\d.,]+)\s*TL'
            match = re.search(fine_pattern, text, re.IGNORECASE)
            if not match:
                return None
            fine_text = match.group(1)

        # Parse Turkish number format: "1.234.567,89" → 1234567.89
        try:
            # Remove thousand separators (.)
            fine_text = fine_text.replace('.', '')
            # Replace decimal comma with dot
            fine_text = fine_text.replace(',', '.')
            return Decimal(fine_text)
        except Exception as e:
            logger.warning(f"Could not parse fine amount '{fine_text}': {e}")
            return None

    def _classify_violation_type(self, title: str, content: str) -> List[str]:
        """
        Classify KVKK violation types from decision text.

        Args:
            title: Decision title
            content: Decision content (first 2000 chars)

        Returns:
            List of violation type keys
        """
        text = (title + ' ' + content[:2000]).lower()
        violations = []

        for key, description in self.VIOLATION_TYPES.items():
            if key in text or description.lower() in text:
                violations.append(key)

        return violations

    def _extract_kvkk_citations(self, text: str) -> List[str]:
        """
        Extract KVKK Law 6698 article citations.

        Args:
            text: Text to search

        Returns:
            List of cited articles
        """
        citations = []

        # Pattern: "6698 sayılı Kanun'un 15 inci maddesi"
        pattern = r'6698\s+sayılı\s+Kanun(?:\'un|un)?\s+(\d+)\s*(?:inci|nci|üncü|uncu)?\s*madde'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            article_num = match.group(1)
            if article_num not in citations:
                citations.append(article_num)

        # Alternative: "Madde 15"
        pattern2 = r'Madde\s+(\d+)'
        for match in re.finditer(pattern2, text):
            article_num = match.group(1)
            if article_num in self.KVKK_ARTICLES and article_num not in citations:
                citations.append(article_num)

        return citations

    def _classify_document_type(self, title: str, url: str, esas_no: Optional[str], karar_no: Optional[str]) -> str:
        """
        Classify KVKK document type.

        Args:
            title: Document title
            url: Document URL
            esas_no: Esas number
            karar_no: Karar number

        Returns:
            'karar', 'rehber', 'teblig', or 'veri_ihlali'
        """
        title_lower = title.lower()
        url_lower = url.lower()

        # Kurul Kararı (has E/K numbers)
        if esas_no or karar_no or 'kurul kararı' in title_lower or 'karar' in url_lower:
            return 'karar'

        # Veri İhlali Bildirimi
        if 'veri ihlal' in title_lower or 'veri-ihlal' in url_lower or 'breach' in title_lower:
            return 'veri_ihlali'

        # Rehber
        if 'rehber' in title_lower or 'kılavuz' in title_lower or 'rehber' in url_lower:
            return 'rehber'

        # Tebliğ
        if 'tebliğ' in title_lower or 'teblig' in url_lower:
            return 'teblig'

        # Default to karar
        return 'karar'

    def _extract_publish_date(self, soup: BeautifulSoup, text: str) -> Optional[date]:
        """Extract publication/decision date."""
        # Try dedicated date element
        date_elem = self._find_element_with_fallback(soup, 'date')
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            parsed = parse_turkish_date(date_text)
            if parsed:
                return parsed

        # Search in text
        # "26 Eylül 2023"
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

        return None

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch KVKK document with comprehensive metadata extraction.

        Args:
            document_id: KVKK document ID
            **kwargs: Additional parameters

        Returns:
            Dict with extracted data:
                - title: Document title
                - content: Full text
                - url: Source URL
                - esas_no: Esas number (for decisions)
                - karar_no: Karar number (for decisions)
                - document_type: karar/rehber/teblig/veri_ihlali
                - fine_amount: Administrative fine (TL)
                - violation_types: List of violation classifications
                - kvkk_citations: Cited KVKK articles
                - publish_date: Publication date

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
                raise ParsingError("Could not extract title", error_code="NO_TITLE")

            # Extract content
            content_elem = self._find_element_with_fallback(soup, 'content')
            if not content_elem:
                raise ParsingError("Could not extract content", error_code="NO_CONTENT")

            content = content_elem.get_text(separator='\n', strip=True)
            content = normalize_turkish_text(content, preserve_case=True)

            # Extract E/K numbers
            esas_no, karar_no = self._extract_esas_karar_numbers(title + '\n' + content[:1000])

            # Classify document type
            doc_type = self._classify_document_type(title, url, esas_no, karar_no)

            # Extract fine amount (only for decisions)
            fine_amount = None
            if doc_type == 'karar':
                fine_amount = self._extract_fine_amount(soup, content)

            # Classify violation types
            violation_types = self._classify_violation_type(title, content)

            # Extract KVKK citations
            kvkk_citations = self._extract_kvkk_citations(content)

            # Extract publish date
            publish_date = self._extract_publish_date(soup, content)

            result = {
                'title': title,
                'content': content,
                'url': url,
                'esas_no': esas_no,
                'karar_no': karar_no,
                'document_type': doc_type,
                'fine_amount': str(fine_amount) if fine_amount else None,
                'violation_types': violation_types,
                'kvkk_citations': kvkk_citations,
                'publish_date': publish_date.isoformat() if publish_date else None,
                'source': 'KVKK',
                'fetched_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Successfully fetched KVKK document: {karar_no or esas_no or document_id} ({doc_type})")
            return result

        except (DocumentNotFoundError, ParsingError):
            raise
        except Exception as e:
            raise ParsingError(
                f"Error fetching KVKK document {document_id}: {str(e)}",
                error_code="FETCH_ERROR",
                context={'document_id': document_id, 'url': url}
            ) from e

    async def search_documents(self, query: str, doc_type: Optional[str] = None, page: int = 1, limit: int = 20, **kwargs) -> List[Dict[str, Any]]:
        """
        Search KVKK documents.

        Args:
            query: Search query
            doc_type: Filter by type
            page: Page number
            limit: Results per page

        Returns:
            List of search results
        """
        search_url = f"{self.base_url}{self.ENDPOINTS['search']}?q={quote(query)}&page={page}"

        if doc_type:
            search_url += f"&type={doc_type}"

        try:
            html = await self._fetch_html(search_url)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Try multiple selectors
            result_selectors = [
                'div.search-result',
                'div.arama-sonuc',
                'li.result-item',
                'article.node'
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
                    snippet_elem = item.find(['p', 'div'], class_=re.compile('snippet|ozet'))
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)

                    # Extract E/K numbers
                    esas, karar = self._extract_esas_karar_numbers(title + ' ' + snippet)

                    # Classify type
                    result_type = self._classify_document_type(title, full_url, esas, karar)

                    results.append({
                        'title': title,
                        'url': full_url,
                        'snippet': snippet,
                        'type': result_type,
                        'esas_no': esas,
                        'karar_no': karar
                    })

                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue

            logger.info(f"Search '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        """Template method: pass through"""
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        """Transform KVKK data to canonical format."""
        from ..core.canonical_schema import (
            LegalDocument, Metadata, SourceType,
            JurisdictionType, LegalHierarchy, EffectivityStatus
        )

        # Map types
        type_mapping = {
            'karar': DocumentType.KVKK_KARARI,
            'rehber': DocumentType.REHBER,
            'teblig': DocumentType.TEBLIG,
            'veri_ihlali': DocumentType.VERI_IHLALI
        }

        doc_type = type_mapping.get(raw_data.get('document_type'), DocumentType.KVKK_KARARI)

        # Create metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_KARAR,
            source=SourceType.KVKK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            esas_no=raw_data.get('esas_no'),
            karar_no=raw_data.get('karar_no'),
            publish_date=raw_data.get('publish_date'),
            fetch_timestamp=raw_data.get('fetched_at'),
            fine_amount=raw_data.get('fine_amount'),
            violation_types=raw_data.get('violation_types'),
            kvkk_citations=raw_data.get('kvkk_citations')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['KVKKAdapter']
