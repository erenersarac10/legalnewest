"""BDDK (Banking Regulation and Supervision Agency) Adapter - Harvey/Legora CTO-Level Production-Grade
Fetches banking regulations, circulars, and supervisory decisions from bddk.org.tr

Production Features:
- Rate limiting: 30 requests/minute sliding window
- TTL caching: 1-hour default with cache key management
- Retry logic: Exponential backoff (3 attempts, 2.0x factor)
- Multiple fallback CSS selectors for robust parsing
- Comprehensive error handling and logging
- Banking-specific extraction: regulations, capital ratios, licensing
- Turkish banking terminology support
"""
from typing import Dict, List, Any, Optional
import aiohttp
import re
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata
from ..errors import NetworkError, ParsingError, ValidationError, DocumentNotFoundError
from ..utils import retry, parse_turkish_date, normalize_turkish_text

logger = logging.getLogger(__name__)


class BDDKAdapter(SourceAdapter):
    """Banking Regulation and Supervision Agency (BDDK) Adapter

    Handles banking regulations, circulars, board decisions, and supervisory guidelines
    from the Turkish Banking Regulator.
    """

    # Multiple fallback CSS selectors for robust parsing
    SELECTORS = {
        'title': [
            'h1.baslik',
            'h1.document-title',
            'h1.page-title',
            'div.baslik h1',
            'h1',
            'div.title h1'
        ],
        'content': [
            'div.icerik',
            'div.content',
            'div.document-content',
            'div.main-content',
            'article',
            'div.document-body'
        ],
        'metadata': [
            'div.belge-bilgi',
            'div.document-info',
            'div.meta-info',
            'table.bilgi',
            'div.metadata'
        ],
        'regulation_details': [
            'div.duzenleme-detay',
            'div.regulation-details',
            'div.detaylar'
        ]
    }

    # Document types with Turkish/English keywords
    DOCUMENT_TYPES = {
        'YÖNETMELİK': {
            'keywords': ['yönetmelik', 'regulation', 'tüzük'],
            'type_code': 'YNT',
            'description': 'Banking Regulation'
        },
        'GENELGE': {
            'keywords': ['genelge', 'circular', 'sirküler'],
            'type_code': 'GNG',
            'description': 'Banking Circular'
        },
        'KURUL_KARARI': {
            'keywords': ['kurul kararı', 'board decision', 'karar'],
            'type_code': 'KK',
            'description': 'Board Decision'
        },
        'TEBLİĞ': {
            'keywords': ['tebliğ', 'communique', 'teblig'],
            'type_code': 'TBL',
            'description': 'Banking Communique'
        },
        'KILAVUZ': {
            'keywords': ['kılavuz', 'guideline', 'rehber'],
            'type_code': 'KLV',
            'description': 'Supervisory Guideline'
        }
    }

    # Banking regulation topics
    REGULATION_TOPICS = {
        'SERMAYE_YETERLİLİĞİ': ['sermaye yeterlilik', 'capital adequacy', 'özkaynak'],
        'RİSK_YÖNETİMİ': ['risk yönetim', 'risk management', 'risk ölçüm'],
        'İÇ_SİSTEMLER': ['iç sistem', 'internal systems', 'iç kontrol', 'internal control'],
        'KREDİ_RİSKİ': ['kredi risk', 'credit risk', 'takipteki alacak'],
        'PİYASA_RİSKİ': ['piyasa risk', 'market risk'],
        'LİKİDİTE': ['likidite', 'liquidity', 'likidite risk'],
        'OPERASYONEL_RİSK': ['operasyonel risk', 'operational risk'],
        'KURUMSAL_YÖNETİM': ['kurumsal yönetim', 'corporate governance', 'yönetim kurulu'],
        'KARA_PARA': ['kara para', 'money laundering', 'aklama', 'spk'],
        'TÜKETİCİ_HAKKI': ['tüketici', 'consumer', 'müşteri hakları'],
        'BANKACILIK_LİSANSI': ['banka lisans', 'banking license', 'kuruluş izni']
    }

    # Regulation status
    REGULATION_STATUS = {
        'YÜRÜRLÜKTE': ['yürürlükte', 'in force', 'geçerli'],
        'YÜRÜRLÜKTEN_KALDIRILDI': ['yürürlükten kaldırıl', 'repealed', 'iptal'],
        'DEĞİŞİKLİK_YAPILDI': ['değiştirilen', 'amended', 'tadil'],
        'TASLAK': ['taslak', 'draft', 'görüş']
    }

    def __init__(self):
        super().__init__("BDDK Adapter", "2.0.0")
        self.base_url = "https://www.bddk.org.tr"

        # Rate limiting: 30 requests per minute (conservative for regulator)
        self.rate_limit = 30
        self.rate_window = 60  # seconds
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL caching: 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        self._cache_lock = asyncio.Lock()

        logger.info(f"Initialized {self.name} v{self.version} with rate limit {self.rate_limit}/min")

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
        """Classify banking document type based on keywords

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

    def _extract_regulation_number(self, text: str) -> Optional[str]:
        """Extract regulation number

        Formats:
        - Sayı: YYYY/NN
        - No: NNNN
        - BDDK-YYYY-NN

        Args:
            text: Text containing regulation number

        Returns:
            Regulation number or None
        """
        # Pattern 1: Sayı: YYYY/NN
        pattern1 = r'Sayı\s*:?\s*(\d{4}/\d{1,4})'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: No: NNNN
        pattern2 = r'No\s*:?\s*(\d{3,5})'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: BDDK-YYYY-NN
        pattern3 = r'BDDK[- ](\d{4})[- ](\d{1,4})'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return f"BDDK-{match.group(1)}-{match.group(2)}"

        return None

    def _extract_publication_date(self, text: str) -> Optional[str]:
        """Extract publication date

        Args:
            text: Text containing date

        Returns:
            Date string or None
        """
        # Pattern: "Yayım Tarihi: dd.mm.yyyy"
        date_pattern = r'(?:Yayım|Yayın|Kabul|Onay)\s*(?:Tarih|Tarihi)\s*:?\s*(\d{2}[./]\d{2}[./]\d{4})'
        match = re.search(date_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Generic date pattern
        date_pattern2 = r'(\d{2}[./]\d{2}[./]\d{4})'
        match = re.search(date_pattern2, text)
        if match:
            return match.group(1)

        return None

    def _extract_regulation_topic(self, text: str) -> List[str]:
        """Extract regulation topics/subjects

        Args:
            text: Text to analyze

        Returns:
            List of identified topics
        """
        text_lower = text.lower()
        topics = []

        for topic_name, keywords in self.REGULATION_TOPICS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.append(topic_name)
                    break  # Only add once per topic

        return topics

    def _extract_regulation_status(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract regulation status

        Args:
            text: Text containing status info

        Returns:
            Dict with status info or None
        """
        text_lower = text.lower()

        for status, keywords in self.REGULATION_STATUS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'status': status,
                        'keyword': keyword
                    }

        return None

    def _extract_capital_ratio(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract capital adequacy ratio requirements

        Args:
            text: Text containing ratio info

        Returns:
            Dict with ratio info or None
        """
        ratio_info = {}

        # Pattern: %12, %8 (percentage format)
        # Standard ratio pattern
        ratio_pattern = r'(?:oran|ratio|yeterlilik)\s*:?\s*%?\s*(\d+(?:[.,]\d+)?)\s*%?'
        match = re.search(ratio_pattern, text, re.IGNORECASE)
        if match:
            try:
                ratio_str = match.group(1).replace(',', '.')
                ratio_info['ratio'] = float(ratio_str)
                ratio_info['formatted'] = f"%{ratio_str}"
            except (ValueError, InvalidOperation):
                pass

        # Minimum capital requirement
        min_capital_pattern = r'(?:asgari|minimum)\s+(?:sermaye|capital)\s*:?\s*([\d.,]+)\s*(?:TL|milyon|milyar)'
        match = re.search(min_capital_pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1)
            # Parse Turkish number format
            try:
                amount_clean = amount_str.replace('.', '').replace(',', '.')
                ratio_info['minimum_capital'] = float(amount_clean)
            except (ValueError, InvalidOperation):
                pass

        return ratio_info if ratio_info else None

    def _extract_effective_date(self, text: str) -> Optional[str]:
        """Extract effective date (Yürürlük Tarihi)

        Args:
            text: Text containing effective date

        Returns:
            Date string or None
        """
        # Pattern: "Yürürlük Tarihi: dd.mm.yyyy"
        effective_pattern = r'Yürürlük\s+(?:Tarih|Tarihi)\s*:?\s*(\d{2}[./]\d{2}[./]\d{4})'
        match = re.search(effective_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern: "yürürlüğe girer" with date
        effective_pattern2 = r'(\d{2}[./]\d{2}[./]\d{4})\s*(?:tarih|tarihinde)\s*yürürlüğe'
        match = re.search(effective_pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_amended_regulations(self, text: str) -> List[str]:
        """Extract references to amended regulations

        Args:
            text: Text containing amendments

        Returns:
            List of amended regulation numbers
        """
        amended = []

        # Pattern: "değiştiren/değiştirilen YYYY/NN"
        pattern = r'(?:değiştir|tadil|amended)\w*\s+(?:sayılı\s+)?(\d{4}/\d{1,4})'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            amended.append(match.group(1))

        return amended[:5]  # Limit to top 5

    def _extract_base_law(self, text: str) -> Optional[str]:
        """Extract base law reference

        Args:
            text: Text containing law reference

        Returns:
            Law reference or None
        """
        # Pattern: "5411 sayılı Bankacılık Kanunu"
        law_pattern = r'(\d{4})\s+sayılı\s+(?:Kanun|Yasa|Bankacılık Kanunu)'
        match = re.search(law_pattern, text)
        if match:
            law_number = match.group(1)
            # Extract full name
            full_match = re.search(rf'{law_number}\s+sayılı\s+([^.]+)', text)
            if full_match:
                return full_match.group(1).strip()
            return f"{law_number} sayılı Kanun"

        return None

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches BDDK documents (Yönetmelik, Genelge, Kurul Kararı)

        Args:
            document_id: Document identifier
            **kwargs: Additional options

        Returns:
            Dict with document data and banking metadata

        Raises:
            NetworkError: On fetch failures
            ParsingError: On parsing failures
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            url = f"{self.base_url}/Mevzuat/{document_id}"
            logger.info(f"Fetching BDDK document: {document_id}")

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

            # Combine title and metadata for extraction
            combined_text = f"{title} {metadata_text} {content[:2000]}"

            # Classify document type
            doc_type_info = self._classify_document_type(title)
            doc_type = doc_type_info['name'] if doc_type_info else None
            doc_type_code = doc_type_info['code'] if doc_type_info else None

            # Extract regulation number
            regulation_number = self._extract_regulation_number(combined_text)

            # Extract dates
            publication_date = self._extract_publication_date(combined_text)
            effective_date = self._extract_effective_date(combined_text)

            # Extract regulation topics
            topics = self._extract_regulation_topic(combined_text)

            # Extract regulation status
            status = self._extract_regulation_status(combined_text)

            # Extract capital ratio requirements
            capital_ratio = self._extract_capital_ratio(content)

            # Extract amended regulations
            amended_regulations = self._extract_amended_regulations(content)

            # Extract base law
            base_law = self._extract_base_law(combined_text)

            # Extract regulation details
            regulation_details_elem = self._find_element(soup, 'regulation_details')
            regulation_details = regulation_details_elem.get_text(strip=True) if regulation_details_elem else None

            result = {
                'title': title,
                'content': content,
                'url': url,
                'document_id': document_id,
                'source': 'BDDK',
                # Document classification
                'document_type': doc_type,
                'document_type_code': doc_type_code,
                # Regulation metadata
                'regulation_number': regulation_number,
                'publication_date': publication_date,
                'effective_date': effective_date,
                # Topics and status
                'topics': topics,
                'topics_count': len(topics),
                'status': status,
                # Banking-specific
                'capital_ratio': capital_ratio,
                'amended_regulations': amended_regulations,
                'base_law': base_law,
                'regulation_details': regulation_details
            }

            logger.info(f"Successfully parsed BDDK document {document_id}: "
                       f"type={doc_type}, topics={len(topics)}")

            return result

        except DocumentNotFoundError:
            raise
        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {str(e)}") from e

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search BDDK documents by keyword with pagination and filtering

        Args:
            query: Search query
            **kwargs: Options (page, limit, doc_type, topic)

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
            search_url = f"{self.base_url}/Mevzuat/Arama?q={quote(query_normalized)}&page={page}"

            # Add filters
            doc_type_filter = kwargs.get('doc_type')
            if doc_type_filter:
                search_url += f"&type={doc_type_filter}"

            topic_filter = kwargs.get('topic')
            if topic_filter:
                search_url += f"&topic={topic_filter}"

            logger.info(f"Searching BDDK: query='{query}', page={page}, limit={limit}")

            # Fetch search results
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Find result items (multiple selectors)
            result_selectors = [
                'div.mevzuat-item',
                'div.search-result',
                'div.result-item',
                'div.sonuc'
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
                    doc_id_match = re.search(r'/Mevzuat/([^/]+)', doc_url)
                    doc_id = doc_id_match.group(1) if doc_id_match else None

                    # Extract summary
                    summary_elem = item.find(['div', 'p'], class_=re.compile(r'ozet|summary|excerpt'))
                    summary = summary_elem.text.strip() if summary_elem else ''

                    # Classify document type
                    doc_type_info = self._classify_document_type(title)

                    # Extract regulation number from title
                    regulation_number = self._extract_regulation_number(title)

                    # Extract topics
                    topics = self._extract_regulation_topic(f"{title} {summary}")

                    result_item = {
                        'title': title,
                        'url': doc_url,
                        'document_id': doc_id,
                        'summary': summary,
                        'document_type': doc_type_info['name'] if doc_type_info else None,
                        'regulation_number': regulation_number,
                        'topics': topics,
                        'source': 'BDDK'
                    }

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to parse search result item: {e}")
                    continue

            logger.info(f"Found {len(results)} BDDK search results for query '{query}'")
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

        if 'yönetmelik' in title_lower or doc_type_field == 'YÖNETMELİK':
            doc_type = DocumentType.YONETMELIK
        elif 'genelge' in title_lower or doc_type_field == 'GENELGE':
            doc_type = DocumentType.GENELGE
        elif 'tebliğ' in title_lower or doc_type_field == 'TEBLİĞ':
            doc_type = DocumentType.TEBLIG
        elif 'kurul kararı' in title_lower or doc_type_field == 'KURUL_KARARI':
            doc_type = DocumentType.KURUL_KARARI
        elif 'kılavuz' in title_lower or doc_type_field == 'KILAVUZ':
            doc_type = DocumentType.KILAVUZ
        else:
            doc_type = DocumentType.YONETMELIK

        # Determine effectivity status
        status = raw_data.get('status', {})
        if status:
            status_code = status.get('status')
            if status_code == 'YÜRÜRLÜKTE':
                effectivity = EffectivityStatus.YURURLUKTE
            elif status_code == 'YÜRÜRLÜKTEN_KALDIRILDI':
                effectivity = EffectivityStatus.YURURLUKTEN_KALDIRILDI
            elif status_code == 'DEĞİŞİKLİK_YAPILDI':
                effectivity = EffectivityStatus.YURURLUKTE
            else:
                effectivity = EffectivityStatus.TASLAK
        else:
            effectivity = EffectivityStatus.YURURLUKTE

        # Build metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_DUZENLEME,
            source=SourceType.BDDK,
            source_url=raw_data.get('url'),
            effectivity_status=effectivity,
            # Banking-specific metadata
            regulation_number=raw_data.get('regulation_number'),
            publication_date=raw_data.get('publication_date'),
            effective_date=raw_data.get('effective_date'),
            topics=raw_data.get('topics'),
            base_law=raw_data.get('base_law'),
            amended_regulations=raw_data.get('amended_regulations')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['BDDKAdapter']
