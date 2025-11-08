"""BTK (Information and Communication Technologies Authority) Adapter - Harvey/Legora CTO-Level Production-Grade
Fetches telecommunications regulations, decisions, and directives from btk.gov.tr

Production Features:
- Rate limiting: 35 requests/minute sliding window
- TTL caching: 1-hour default with cache key management
- Retry logic: Exponential backoff (3 attempts, 2.0x factor)
- Multiple fallback CSS selectors for robust parsing
- Comprehensive error handling and logging
- Telecom-specific extraction: spectrum, licensing, tariffs
- Turkish telecommunications terminology support
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


class BTKAdapter(SourceAdapter):
    """Information and Communication Technologies Authority (BTK) Adapter

    Handles telecommunications regulations, spectrum decisions, licensing, and ICT directives
    from the Turkish ICT Regulator.
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
        'decision_details': [
            'div.karar-detay',
            'div.decision-details',
            'div.detaylar'
        ]
    }

    # Document types with Turkish/English keywords
    DOCUMENT_TYPES = {
        'YÖNETMELİK': {
            'keywords': ['yönetmelik', 'regulation', 'tüzük'],
            'type_code': 'YNT',
            'description': 'Telecommunications Regulation'
        },
        'TEBLİĞ': {
            'keywords': ['tebliğ', 'communique', 'teblig'],
            'type_code': 'TBL',
            'description': 'Telecommunications Communique'
        },
        'KURUL_KARARI': {
            'keywords': ['kurul kararı', 'board decision', 'karar'],
            'type_code': 'KK',
            'description': 'Board Decision'
        },
        'YÖNERGE': {
            'keywords': ['yönerge', 'directive', 'direktif'],
            'type_code': 'YNG',
            'description': 'Directive'
        },
        'KILAVUZ': {
            'keywords': ['kılavuz', 'guideline', 'rehber'],
            'type_code': 'KLV',
            'description': 'Technical Guideline'
        }
    }

    # Telecommunications topics
    TELECOM_TOPICS = {
        'SPEKTRUM': ['spektrum', 'spectrum', 'frekans', 'frequency'],
        'LİSANSLAMA': ['lisans', 'license', 'izin', 'permit'],
        'İNTERNET': ['internet', 'genişbant', 'broadband', 'erişim'],
        'MOBİL_İLETİŞİM': ['mobil', 'mobile', 'gsm', '3g', '4g', '5g', 'hücresel'],
        'SABIT_TELEFON': ['sabit telefon', 'fixed line', 'pstn'],
        'YAYIN': ['yayın', 'broadcasting', 'radyo', 'televizyon'],
        'TÜKETİCİ_HAKKI': ['tüketici', 'consumer', 'abone hakları'],
        'ALTYAPI': ['altyapı', 'infrastructure', 'fiber', 'kablo'],
        'SİBER_GÜVENLİK': ['siber güvenlik', 'cyber security', 'bilgi güvenliği'],
        'VERİ_KORUMA': ['veri koruma', 'data protection', 'gizlilik', 'privacy'],
        'NUMARALANDIRMA': ['numara', 'numbering', 'numaralandırma'],
        'KALİTE': ['hizmet kalite', 'quality of service', 'qos']
    }

    # Spectrum bands (frequency ranges)
    SPECTRUM_BANDS = {
        'VHF': ['vhf', '30-300 mhz'],
        'UHF': ['uhf', '300-3000 mhz'],
        '800MHz': ['800 mhz', '790-862 mhz'],
        '900MHz': ['900 mhz', '880-960 mhz'],
        '1800MHz': ['1800 mhz', '1710-1880 mhz'],
        '2100MHz': ['2100 mhz', 'umts', '1920-2170 mhz'],
        '2600MHz': ['2600 mhz', '2500-2690 mhz'],
        '3.5GHz': ['3.5 ghz', '3400-3800 mhz', '5g'],
        '26GHz': ['26 ghz', '24.25-27.5 ghz']
    }

    # Service types
    SERVICE_TYPES = {
        'SABİT_TELEFON': ['sabit telefon', 'fixed telephony', 'pstn'],
        'MOBİL_TELEFON': ['mobil telefon', 'mobile telephony', 'gsm'],
        'İNTERNET_ERİŞİM': ['internet erişim', 'internet access', 'isp'],
        'TOPTAN_HİZMET': ['toptan', 'wholesale'],
        'YAYIN_HİZMETİ': ['yayın hizmet', 'broadcasting service']
    }

    def __init__(self):
        super().__init__("BTK Adapter", "2.0.0")
        self.base_url = "https://www.btk.gov.tr"

        # Rate limiting: 35 requests per minute
        self.rate_limit = 35
        self.rate_window = 60  # seconds
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL caching: 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        self._cache_lock = asyncio.Lock()

        logger.info(f"Initialized {self.name} v{self.version} with rate limit {self.rate_limit}/min")

    async def _enforce_rate_limit(self):
        """Sliding window rate limiting - 35 requests per minute"""
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
        """Classify telecommunications document type based on keywords

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

    def _extract_decision_number(self, text: str) -> Optional[str]:
        """Extract decision/regulation number

        Formats:
        - Karar No: YYYY-NN-NNN
        - Sayı: YYYY/NN
        - BTK-YYYY-NN

        Args:
            text: Text containing decision number

        Returns:
            Decision number or None
        """
        # Pattern 1: Karar No: YYYY-NN-NNN
        pattern1 = r'Karar\s+No\s*:?\s*(\d{4}-\d{1,2}-\d{1,4})'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: Sayı: YYYY/NN
        pattern2 = r'Sayı\s*:?\s*(\d{4}/\d{1,4})'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: BTK-YYYY-NN
        pattern3 = r'BTK[- ](\d{4})[- ](\d{1,4})'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return f"BTK-{match.group(1)}-{match.group(2)}"

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

    def _extract_telecom_topics(self, text: str) -> List[str]:
        """Extract telecommunications topics

        Args:
            text: Text to analyze

        Returns:
            List of identified topics
        """
        text_lower = text.lower()
        topics = []

        for topic_name, keywords in self.TELECOM_TOPICS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.append(topic_name)
                    break  # Only add once per topic

        return topics

    def _extract_spectrum_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract spectrum/frequency information

        Args:
            text: Text containing spectrum info

        Returns:
            Dict with spectrum info or None
        """
        spectrum_info = {}

        # Identify spectrum bands
        text_lower = text.lower()
        bands = []
        for band_name, keywords in self.SPECTRUM_BANDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    bands.append(band_name)
                    break

        if bands:
            spectrum_info['bands'] = bands

        # Extract frequency values
        # Pattern: "2600 MHz" or "3.5 GHz"
        freq_pattern = r'(\d+(?:[.,]\d+)?)\s*(?:MHz|GHz|mhz|ghz)'
        matches = re.findall(freq_pattern, text)
        if matches:
            frequencies = []
            for freq_str in matches[:5]:  # Limit to 5
                freq_clean = freq_str.replace(',', '.')
                frequencies.append(f"{freq_clean}")
            spectrum_info['frequencies'] = frequencies

        # Extract bandwidth
        bandwidth_pattern = r'(\d+)\s*(?:MHz|mhz)\s*(?:bant genişliği|bandwidth)'
        match = re.search(bandwidth_pattern, text, re.IGNORECASE)
        if match:
            spectrum_info['bandwidth'] = f"{match.group(1)} MHz"

        return spectrum_info if spectrum_info else None

    def _extract_service_type(self, text: str) -> List[str]:
        """Extract service types

        Args:
            text: Text containing service info

        Returns:
            List of service types
        """
        text_lower = text.lower()
        services = []

        for service_name, keywords in self.SERVICE_TYPES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    services.append(service_name)
                    break

        return services

    def _extract_license_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract licensing information

        Args:
            text: Text containing license info

        Returns:
            Dict with license info or None
        """
        license_info = {}

        # License type
        license_types = []
        if 'genel yetki' in text.lower() or 'general authorization' in text.lower():
            license_types.append('Genel Yetki')
        if 'bireysel yetki' in text.lower() or 'individual authorization' in text.lower():
            license_types.append('Bireysel Yetki')

        if license_types:
            license_info['license_types'] = license_types

        # License period (years)
        period_pattern = r'(\d+)\s*(?:yıl|year)'
        match = re.search(period_pattern, text, re.IGNORECASE)
        if match:
            license_info['period_years'] = int(match.group(1))

        # License fee
        fee_pattern = r'(?:lisans bedel|license fee)\s*:?\s*([\d.,]+)\s*(?:TL|₺)'
        match = re.search(fee_pattern, text, re.IGNORECASE)
        if match:
            license_info['fee'] = match.group(1)

        return license_info if license_info else None

    def _extract_tariff_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tariff/pricing information

        Args:
            text: Text containing tariff info

        Returns:
            Dict with tariff info or None
        """
        tariff_info = {}

        # Price per minute (TL/dk)
        minute_pattern = r'([\d.,]+)\s*(?:TL|₺)/(?:dk|dakika|minute)'
        match = re.search(minute_pattern, text, re.IGNORECASE)
        if match:
            tariff_info['price_per_minute'] = match.group(1)

        # Monthly fee (Aylık ücret)
        monthly_pattern = r'(?:aylık|monthly)\s*(?:ücret|fee)\s*:?\s*([\d.,]+)\s*(?:TL|₺)'
        match = re.search(monthly_pattern, text, re.IGNORECASE)
        if match:
            tariff_info['monthly_fee'] = match.group(1)

        # Data pricing (TL/GB or TL/MB)
        data_pattern = r'([\d.,]+)\s*(?:TL|₺)/(?:GB|MB|gb|mb)'
        match = re.search(data_pattern, text, re.IGNORECASE)
        if match:
            tariff_info['data_price'] = match.group(0)

        return tariff_info if tariff_info else None

    def _extract_operator_info(self, text: str) -> List[str]:
        """Extract operator/carrier names

        Args:
            text: Text containing operator info

        Returns:
            List of operator names
        """
        operators = []

        # Turkish mobile operators
        operator_keywords = [
            'turkcell', 'vodafone', 'türk telekom', 'turk telekom',
            'avea', 'ttnet', 'superonline', 'millenicom'
        ]

        text_lower = text.lower()
        for operator in operator_keywords:
            if operator in text_lower:
                # Capitalize properly
                operators.append(operator.title())

        return list(set(operators))[:5]  # Unique, max 5

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches BTK documents (Yönetmelik, Tebliğ, Kurul Kararı)

        Args:
            document_id: Document identifier
            **kwargs: Additional options

        Returns:
            Dict with document data and telecommunications metadata

        Raises:
            NetworkError: On fetch failures
            ParsingError: On parsing failures
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            url = f"{self.base_url}/Mevzuat/{document_id}"
            logger.info(f"Fetching BTK document: {document_id}")

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

            # Extract decision number
            decision_number = self._extract_decision_number(combined_text)

            # Extract publication date
            publication_date = self._extract_publication_date(combined_text)

            # Extract telecommunications topics
            topics = self._extract_telecom_topics(combined_text)

            # Extract spectrum information
            spectrum_info = self._extract_spectrum_info(content)

            # Extract service types
            service_types = self._extract_service_type(combined_text)

            # Extract license information
            license_info = self._extract_license_info(content)

            # Extract tariff information
            tariff_info = self._extract_tariff_info(content)

            # Extract operators
            operators = self._extract_operator_info(combined_text)

            # Extract decision details
            decision_details_elem = self._find_element(soup, 'decision_details')
            decision_details = decision_details_elem.get_text(strip=True) if decision_details_elem else None

            result = {
                'title': title,
                'content': content,
                'url': url,
                'document_id': document_id,
                'source': 'BTK',
                # Document classification
                'document_type': doc_type,
                'document_type_code': doc_type_code,
                # Metadata
                'decision_number': decision_number,
                'publication_date': publication_date,
                # Topics and services
                'topics': topics,
                'topics_count': len(topics),
                'service_types': service_types,
                # Telecom-specific
                'spectrum_info': spectrum_info,
                'license_info': license_info,
                'tariff_info': tariff_info,
                'operators': operators,
                'decision_details': decision_details
            }

            logger.info(f"Successfully parsed BTK document {document_id}: "
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
        """Search BTK documents by keyword with pagination and filtering

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

            logger.info(f"Searching BTK: query='{query}', page={page}, limit={limit}")

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

                    # Extract decision number
                    decision_number = self._extract_decision_number(title)

                    # Extract topics
                    topics = self._extract_telecom_topics(f"{title} {summary}")

                    result_item = {
                        'title': title,
                        'url': doc_url,
                        'document_id': doc_id,
                        'summary': summary,
                        'document_type': doc_type_info['name'] if doc_type_info else None,
                        'decision_number': decision_number,
                        'topics': topics,
                        'source': 'BTK'
                    }

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to parse search result item: {e}")
                    continue

            logger.info(f"Found {len(results)} BTK search results for query '{query}'")
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
        elif 'tebliğ' in title_lower or doc_type_field == 'TEBLİĞ':
            doc_type = DocumentType.TEBLIG
        elif 'kurul kararı' in title_lower or doc_type_field == 'KURUL_KARARI':
            doc_type = DocumentType.KURUL_KARARI
        elif 'yönerge' in title_lower or doc_type_field == 'YÖNERGE':
            doc_type = DocumentType.YONERGE
        elif 'kılavuz' in title_lower or doc_type_field == 'KILAVUZ':
            doc_type = DocumentType.KILAVUZ
        else:
            doc_type = DocumentType.YONETMELIK

        # Build metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_DUZENLEME,
            source=SourceType.BTK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            # Telecommunications metadata
            decision_number=raw_data.get('decision_number'),
            publication_date=raw_data.get('publication_date'),
            topics=raw_data.get('topics'),
            spectrum_info=raw_data.get('spectrum_info'),
            service_types=raw_data.get('service_types'),
            operators=raw_data.get('operators')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['BTKAdapter']
