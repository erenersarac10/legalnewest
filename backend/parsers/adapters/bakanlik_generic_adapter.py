"""Generic Ministry (Bakanlık) Adapter - Harvey/Legora CTO-Level Production-Grade
Flexible adapter for fetching documents from various Turkish ministries

Production Features:
- Rate limiting: 30 requests/minute sliding window
- TTL caching: 1-hour default with cache key management
- Retry logic: Exponential backoff (3 attempts, 2.0x factor)
- Multiple fallback CSS selectors for robust parsing
- Comprehensive error handling and logging
- Ministry-specific configuration support
- Turkish administrative terminology support
"""
from typing import Dict, List, Any, Optional
import aiohttp
import re
import asyncio
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote, urlparse

from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata
from ..errors import NetworkError, ParsingError, ValidationError, DocumentNotFoundError
from ..utils import retry, parse_turkish_date, normalize_turkish_text

logger = logging.getLogger(__name__)


class BakanlikGenericAdapter(SourceAdapter):
    """Generic Ministry (Bakanlık) Adapter

    Flexible adapter that can handle documents from various Turkish ministries
    with configurable URL patterns and selectors.
    """

    # Multiple fallback CSS selectors for robust parsing
    SELECTORS = {
        'title': [
            'h1.baslik',
            'h1.document-title',
            'h1.page-title',
            'div.baslik h1',
            'h1',
            'div.title h1',
            'h2.title',
            'div.content-title h1'
        ],
        'content': [
            'div.icerik',
            'div.content',
            'div.document-content',
            'div.main-content',
            'article',
            'div.document-body',
            'div.detay',
            'div.text-content'
        ],
        'metadata': [
            'div.belge-bilgi',
            'div.document-info',
            'div.meta-info',
            'table.bilgi',
            'div.metadata',
            'div.info-box'
        ],
        'date_info': [
            'div.tarih',
            'div.date',
            'span.date',
            'div.yayim-tarihi'
        ]
    }

    # Document types with Turkish/English keywords
    DOCUMENT_TYPES = {
        'GENELGE': {
            'keywords': ['genelge', 'circular', 'sirküler'],
            'type_code': 'GNG',
            'description': 'Ministry Circular'
        },
        'TEBLİĞ': {
            'keywords': ['tebliğ', 'teblig', 'communique'],
            'type_code': 'TBL',
            'description': 'Ministry Communique'
        },
        'YÖNERGE': {
            'keywords': ['yönerge', 'directive', 'direktif'],
            'type_code': 'YNG',
            'description': 'Ministry Directive'
        },
        'KARAR': {
            'keywords': ['karar', 'decision', 'kurul kararı'],
            'type_code': 'KRR',
            'description': 'Ministry Decision'
        },
        'DUYURU': {
            'keywords': ['duyuru', 'announcement', 'açıklama'],
            'type_code': 'DYR',
            'description': 'Ministry Announcement'
        },
        'YÖNETMELİK': {
            'keywords': ['yönetmelik', 'regulation', 'tüzük'],
            'type_code': 'YNT',
            'description': 'Ministry Regulation'
        }
    }

    # Turkish ministries configuration
    MINISTRIES = {
        'SAĞLIK': {
            'name': 'Sağlık Bakanlığı',
            'english_name': 'Ministry of Health',
            'base_url': 'https://www.saglik.gov.tr',
            'topics': ['sağlık', 'hastane', 'ilaç', 'health', 'medical']
        },
        'EĞİTİM': {
            'name': 'Milli Eğitim Bakanlığı',
            'english_name': 'Ministry of National Education',
            'base_url': 'https://www.meb.gov.tr',
            'topics': ['eğitim', 'okul', 'öğretmen', 'education', 'school']
        },
        'ADLİYE': {
            'name': 'Adalet Bakanlığı',
            'english_name': 'Ministry of Justice',
            'base_url': 'https://www.adalet.gov.tr',
            'topics': ['adalet', 'mahkeme', 'yargı', 'justice', 'court']
        },
        'İÇİŞLERİ': {
            'name': 'İçişleri Bakanlığı',
            'english_name': 'Ministry of Interior',
            'base_url': 'https://www.icisleri.gov.tr',
            'topics': ['güvenlik', 'polis', 'nüfus', 'security', 'police']
        },
        'DIŞİŞLERİ': {
            'name': 'Dışişleri Bakanlığı',
            'english_name': 'Ministry of Foreign Affairs',
            'base_url': 'https://www.mfa.gov.tr',
            'topics': ['dış politika', 'diplomasi', 'foreign policy', 'diplomacy']
        },
        'MALİYE': {
            'name': 'Hazine ve Maliye Bakanlığı',
            'english_name': 'Ministry of Treasury and Finance',
            'base_url': 'https://www.hmb.gov.tr',
            'topics': ['vergi', 'bütçe', 'maliye', 'tax', 'budget', 'finance']
        },
        'ÇEVRE': {
            'name': 'Çevre, Şehircilik ve İklim Değişikliği Bakanlığı',
            'english_name': 'Ministry of Environment',
            'base_url': 'https://www.csb.gov.tr',
            'topics': ['çevre', 'iklim', 'environment', 'climate']
        },
        'ULAŞTIRMA': {
            'name': 'Ulaştırma ve Altyapı Bakanlığı',
            'english_name': 'Ministry of Transportation',
            'base_url': 'https://www.uab.gov.tr',
            'topics': ['ulaştırma', 'altyapı', 'transportation', 'infrastructure']
        }
    }

    def __init__(self, ministry_code: str = None, base_url: str = None):
        """Initialize Generic Ministry Adapter

        Args:
            ministry_code: Optional ministry code (e.g., 'SAĞLIK', 'EĞİTİM')
            base_url: Optional custom base URL
        """
        super().__init__("Genel Bakanlık Adapter", "2.0.0")

        # Configure ministry
        self.ministry_code = ministry_code
        if ministry_code and ministry_code in self.MINISTRIES:
            ministry_config = self.MINISTRIES[ministry_code]
            self.ministry_name = ministry_config['name']
            self.base_url = base_url or ministry_config['base_url']
        else:
            self.ministry_name = "Generic Ministry"
            self.base_url = base_url or "https://www.gov.tr"

        # Rate limiting: 30 requests per minute (conservative for government sites)
        self.rate_limit = 30
        self.rate_window = 60  # seconds
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL caching: 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        self._cache_lock = asyncio.Lock()

        logger.info(f"Initialized {self.name} v{self.version} for {self.ministry_name} "
                   f"with rate limit {self.rate_limit}/min")

    async def _enforce_rate_limit(self):
        """Sliding window rate limiting - 30 requests per minute"""
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()
            # Remove requests outside the window
            self.request_times = [t for t in self.request_times if now - t < self.rate_window]

            if len(self.request_times) >= self.rate_limit:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = self.rate_window - (now - oldest_request)
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Recurse to recheck
                    return await self._enforce_rate_limit()

            self.request_times.append(now)
            logger.debug(f"Rate limit check passed. {len(self.request_times)}/{self.rate_limit} requests in window")

    async def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired"""
        async with self._cache_lock:
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                cached_time = cached_data.get('cached_at', 0)
                now = asyncio.get_event_loop().time()

                if now - cached_time < self.cache_ttl:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_data.get('data')
                else:
                    # Expired, remove from cache
                    logger.debug(f"Cache expired for key: {cache_key}")
                    del self.cache[cache_key]

            return None

    async def _set_cached(self, cache_key: str, data: Dict[str, Any]):
        """Set cache with current timestamp"""
        async with self._cache_lock:
            self.cache[cache_key] = {
                'data': data,
                'cached_at': asyncio.get_event_loop().time()
            }
            logger.debug(f"Cached data for key: {cache_key}")

    @retry(max_attempts=3, backoff_factor=2.0, exceptions=(NetworkError, aiohttp.ClientError))
    async def _fetch_html(self, url: str, **kwargs) -> str:
        """Fetch HTML with comprehensive error handling, caching, and retry logic

        Args:
            url: URL to fetch
            **kwargs: Additional arguments (e.g., use_cache, timeout)

        Returns:
            HTML content as string

        Raises:
            NetworkError: On network/HTTP errors
            ValidationError: On invalid response
        """
        # Check cache first (if enabled)
        use_cache = kwargs.get('use_cache', True)
        cache_key = f"html:{url}"

        if use_cache:
            cached = await self._get_cached(cache_key)
            if cached:
                return cached

        # Enforce rate limiting
        await self._enforce_rate_limit()

        # Prepare request
        timeout_total = kwargs.get('timeout', 30)
        timeout = aiohttp.ClientTimeout(total=timeout_total, connect=10)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                logger.debug(f"Fetching URL: {url}")
                async with session.get(url) as response:
                    # Check status code
                    if response.status == 404:
                        raise DocumentNotFoundError(f"Document not found: {url}")
                    elif response.status >= 400:
                        raise NetworkError(f"HTTP {response.status} for URL: {url}")

                    html = await response.text()

                    # Validate response
                    if not html or len(html) < 100:
                        raise ValidationError(f"Invalid or empty response from {url}")

                    logger.info(f"Successfully fetched {len(html)} bytes from {url}")

                    # Cache the result
                    if use_cache:
                        await self._set_cached(cache_key, html)

                    return html

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching {url}: {e}")
            raise NetworkError(f"Failed to fetch {url}: {str(e)}") from e
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout fetching {url}")
            raise NetworkError(f"Timeout fetching {url}") from e

    def _find_element(self, soup: BeautifulSoup, selector_key: str) -> Optional[Any]:
        """Find element using multiple fallback selectors

        Args:
            soup: BeautifulSoup object
            selector_key: Key in SELECTORS dict

        Returns:
            Found element or None
        """
        selectors = self.SELECTORS.get(selector_key, [])
        for selector in selectors:
            # Try class selector
            if '.' in selector and not selector.startswith('div.'):
                class_name = selector.split('.')[-1]
                elem = soup.find(class_=class_name)
                if elem:
                    logger.debug(f"Found element with selector: {selector}")
                    return elem
            # Try tag.class selector
            elif '.' in selector:
                tag, class_name = selector.split('.', 1)
                elem = soup.find(tag, class_=class_name)
                if elem:
                    logger.debug(f"Found element with selector: {selector}")
                    return elem
            # Try tag selector
            else:
                elem = soup.find(selector)
                if elem:
                    logger.debug(f"Found element with selector: {selector}")
                    return elem

        logger.warning(f"No element found for selector key: {selector_key}")
        return None

    def _classify_document_type(self, text: str) -> Optional[Dict[str, Any]]:
        """Classify ministry document type based on keywords

        Args:
            text: Text to analyze (usually title)

        Returns:
            Dict with document type info or None
        """
        text_lower = text.lower()

        for doc_name, doc_info in self.DOCUMENT_TYPES.items():
            for keyword in doc_info['keywords']:
                if keyword in text_lower:
                    logger.debug(f"Classified document type: {doc_name}")
                    return {
                        'name': doc_name,
                        'code': doc_info['type_code'],
                        'description': doc_info['description']
                    }

        return None

    def _identify_ministry(self, url: str, text: str) -> Optional[str]:
        """Identify which ministry a document belongs to

        Args:
            url: Document URL
            text: Document text

        Returns:
            Ministry code or None
        """
        # Check URL domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        for ministry_code, config in self.MINISTRIES.items():
            ministry_domain = urlparse(config['base_url']).netloc
            if ministry_domain in domain:
                return ministry_code

        # Check text for ministry topics
        text_lower = text.lower()
        for ministry_code, config in self.MINISTRIES.items():
            for topic in config['topics']:
                if topic in text_lower:
                    return ministry_code

        return None

    def _extract_document_number(self, text: str) -> Optional[str]:
        """Extract document number

        Formats:
        - Sayı: YYYY/NN
        - No: NNNN
        - Genelge No: YYYY-NN

        Args:
            text: Text containing document number

        Returns:
            Document number or None
        """
        # Pattern 1: Sayı: YYYY/NN
        pattern1 = r'Sayı\s*:?\s*(\d{4}/\d{1,4})'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: No: NNNN or Genelge No: YYYY-NN
        pattern2 = r'(?:Genelge\s+)?No\s*:?\s*(\d{4}[/-]\d{1,4}|\d{3,5})'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: Karar No
        pattern3 = r'Karar\s+No\s*:?\s*(\d{4}[/-]\d{1,4})'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_publication_date(self, soup: BeautifulSoup, text: str) -> Optional[str]:
        """Extract publication date

        Args:
            soup: BeautifulSoup object
            text: Text containing date

        Returns:
            Date string or None
        """
        # Try to find date element
        date_elem = self._find_element(soup, 'date_info')
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            # Extract date from element
            date_pattern = r'(\d{2}[./]\d{2}[./]\d{4})'
            match = re.search(date_pattern, date_text)
            if match:
                return match.group(1)

        # Pattern: "Yayım Tarihi: dd.mm.yyyy"
        date_pattern = r'(?:Yayım|Yayın|Tarih)\s*:?\s*(\d{2}[./]\d{2}[./]\d{4})'
        match = re.search(date_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Generic date pattern
        date_pattern2 = r'(\d{2}[./]\d{2}[./]\d{4})'
        match = re.search(date_pattern2, text)
        if match:
            return match.group(1)

        return None

    def _extract_author_unit(self, text: str) -> Optional[str]:
        """Extract issuing unit/department

        Args:
            text: Text containing author info

        Returns:
            Unit name or None
        """
        # Pattern: "...Genel Müdürlüğü" or "...Başkanlığı"
        unit_pattern = r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)*)\s+(?:Genel Müdürlüğü|Başkanlığı|Dairesi)'
        match = re.search(unit_pattern, text)
        if match:
            return match.group(0)

        return None

    def _extract_related_laws(self, text: str) -> List[str]:
        """Extract referenced laws

        Args:
            text: Text containing law references

        Returns:
            List of law references
        """
        laws = []

        # Pattern: "NNNN sayılı Kanun"
        law_pattern = r'(\d{4})\s+sayılı\s+(?:Kanun|Yasa)'
        matches = re.finditer(law_pattern, text)
        for match in matches:
            law_number = match.group(1)
            laws.append(f"{law_number} sayılı Kanun")

        return list(set(laws))[:5]  # Unique, max 5

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords/topics from text

        Args:
            text: Document text

        Returns:
            List of keywords
        """
        keywords = []

        # Check ministry topics
        if self.ministry_code and self.ministry_code in self.MINISTRIES:
            ministry_topics = self.MINISTRIES[self.ministry_code]['topics']
            text_lower = text.lower()
            for topic in ministry_topics:
                if topic in text_lower:
                    keywords.append(topic)

        return keywords

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches ministry documents (Genelge, Tebliğ, Yönerge, Karar)

        Args:
            document_id: Document identifier
            **kwargs: Additional options (url, ministry_code)

        Returns:
            Dict with document data and ministry metadata

        Raises:
            NetworkError: On fetch failures
            ParsingError: On parsing failures
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            # Allow custom URL
            url = kwargs.get('url') or f"{self.base_url}/Mevzuat/{document_id}"
            logger.info(f"Fetching ministry document: {document_id} from {self.ministry_name}")

            # Fetch HTML
            html = await self._fetch_html(url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title using multiple selectors
            title_elem = self._find_element(soup, 'title')
            title = title_elem.text.strip() if title_elem else ''

            if not title:
                raise ParsingError(f"Failed to extract title from document {document_id}")

            # Extract content using multiple selectors
            content_elem = self._find_element(soup, 'content')
            content = content_elem.get_text(separator='\n', strip=True) if content_elem else ''

            # Extract metadata
            metadata_elem = self._find_element(soup, 'metadata')
            metadata_text = metadata_elem.get_text(strip=True) if metadata_elem else ''

            # Combine for extraction
            combined_text = f"{title} {metadata_text} {content[:2000]}"

            # Classify document type
            doc_type_info = self._classify_document_type(title)
            doc_type = doc_type_info['name'] if doc_type_info else None
            doc_type_code = doc_type_info['code'] if doc_type_info else None

            # Identify ministry
            ministry_code = kwargs.get('ministry_code') or self._identify_ministry(url, combined_text)
            if ministry_code and ministry_code in self.MINISTRIES:
                ministry_name = self.MINISTRIES[ministry_code]['name']
            else:
                ministry_name = self.ministry_name

            # Extract document number
            document_number = self._extract_document_number(combined_text)

            # Extract publication date
            publication_date = self._extract_publication_date(soup, combined_text)

            # Extract author unit
            author_unit = self._extract_author_unit(combined_text)

            # Extract related laws
            related_laws = self._extract_related_laws(content)

            # Extract keywords
            keywords = self._extract_keywords(combined_text)

            result = {
                'title': title,
                'content': content,
                'url': url,
                'document_id': document_id,
                'source': ministry_name,
                # Document classification
                'document_type': doc_type,
                'document_type_code': doc_type_code,
                # Metadata
                'ministry_code': ministry_code,
                'ministry_name': ministry_name,
                'document_number': document_number,
                'publication_date': publication_date,
                'author_unit': author_unit,
                # Related content
                'related_laws': related_laws,
                'keywords': keywords
            }

            logger.info(f"Successfully parsed ministry document {document_id}: "
                       f"ministry={ministry_code}, type={doc_type}")

            return result

        except DocumentNotFoundError:
            raise
        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {str(e)}") from e

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search ministry documents by keyword with pagination and filtering

        Args:
            query: Search query
            **kwargs: Options (page, limit, ministry_code, doc_type, url)

        Returns:
            List of search result dicts

        Raises:
            NetworkError: On fetch failures
            ValidationError: On invalid parameters
        """
        try:
            # Normalize Turkish query
            query_normalized = normalize_turkish_text(query)

            # Pagination
            page = kwargs.get('page', 1)
            limit = kwargs.get('limit', 20)
            if limit > 100:
                raise ValidationError("Limit cannot exceed 100")

            # Build search URL
            search_url = kwargs.get('url') or f"{self.base_url}/Arama?q={quote(query_normalized)}&page={page}"

            # Add filters
            ministry_filter = kwargs.get('ministry_code')
            if ministry_filter:
                search_url += f"&ministry={ministry_filter}"

            doc_type_filter = kwargs.get('doc_type')
            if doc_type_filter:
                search_url += f"&type={doc_type_filter}"

            logger.info(f"Searching ministry documents: query='{query}', page={page}, limit={limit}")

            # Fetch search results
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Find result items (multiple selectors)
            result_selectors = [
                'div.sonuc',
                'div.search-result',
                'div.result-item',
                'div.arama-sonuc',
                'div.list-item'
            ]

            items = []
            for selector in result_selectors:
                if '.' in selector:
                    tag, class_name = selector.split('.', 1)
                    items = soup.find_all(tag, class_=class_name)
                    if items:
                        break

            for item in items[:limit]:
                try:
                    # Extract title and link
                    title_elem = item.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    href = title_elem.get('href', '')
                    doc_url = urljoin(self.base_url, href)

                    # Extract document ID from URL
                    doc_id_match = re.search(r'/(?:Mevzuat|Duyuru|Karar)/([^/]+)', doc_url)
                    doc_id = doc_id_match.group(1) if doc_id_match else None

                    # Extract summary
                    summary_elem = item.find(['div', 'p'], class_=re.compile(r'ozet|summary|excerpt'))
                    summary = summary_elem.text.strip() if summary_elem else ''

                    # Classify document type
                    doc_type_info = self._classify_document_type(title)

                    # Extract document number
                    document_number = self._extract_document_number(title)

                    # Identify ministry
                    ministry_code = self._identify_ministry(doc_url, f"{title} {summary}")

                    result_item = {
                        'title': title,
                        'url': doc_url,
                        'document_id': doc_id,
                        'summary': summary,
                        'document_type': doc_type_info['name'] if doc_type_info else None,
                        'document_number': document_number,
                        'ministry_code': ministry_code,
                        'source': self.ministry_name
                    }

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to parse search result item: {e}")
                    continue

            logger.info(f"Found {len(results)} ministry search results for query '{query}'")
            return results

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise NetworkError(f"Search failed: {str(e)}") from e

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        """Extract raw data from preprocessed input"""
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        """Transform to canonical LegalDocument format

        Args:
            raw_data: Raw extracted data
            document_type: Optional document type override
            **kwargs: Additional options

        Returns:
            LegalDocument instance
        """
        from ..core.canonical_schema import (
            LegalDocument, Metadata, SourceType, JurisdictionType,
            LegalHierarchy, EffectivityStatus
        )

        # Determine document type
        title_lower = raw_data.get('title', '').lower()
        doc_type_field = raw_data.get('document_type', '')

        if 'genelge' in title_lower or doc_type_field == 'GENELGE':
            doc_type = DocumentType.GENELGE
        elif 'tebliğ' in title_lower or doc_type_field == 'TEBLİĞ':
            doc_type = DocumentType.TEBLIG
        elif 'yönerge' in title_lower or doc_type_field == 'YÖNERGE':
            doc_type = DocumentType.YONERGE
        elif 'yönetmelik' in title_lower or doc_type_field == 'YÖNETMELİK':
            doc_type = DocumentType.YONETMELIK
        elif 'karar' in title_lower or doc_type_field == 'KARAR':
            doc_type = DocumentType.KARAR
        elif 'duyuru' in title_lower or doc_type_field == 'DUYURU':
            doc_type = DocumentType.DUYURU
        else:
            doc_type = DocumentType.GENELGE

        # Map ministry code to SourceType
        ministry_code = raw_data.get('ministry_code')
        if ministry_code:
            source_type = SourceType.BAKANLIK
        else:
            source_type = SourceType.DIGER

        # Build metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_DUZENLEME,
            source=source_type,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            # Ministry-specific metadata
            ministry_code=ministry_code,
            ministry_name=raw_data.get('ministry_name'),
            document_number=raw_data.get('document_number'),
            publication_date=raw_data.get('publication_date'),
            author_unit=raw_data.get('author_unit'),
            related_laws=raw_data.get('related_laws'),
            keywords=raw_data.get('keywords')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['BakanlikGenericAdapter']
