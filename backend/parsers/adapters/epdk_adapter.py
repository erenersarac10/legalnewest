"""EPDK (Energy Market Regulatory Authority) Adapter - Harvey/Legora CTO-Level
Fetches energy sector regulations and decisions from epdk.gov.tr

Production-grade implementation with:
- Rate limiting (35 req/min sliding window)
- TTL caching (1 hour default)
- Retry logic with exponential backoff
- Multiple fallback CSS selectors
- Comprehensive error handling
- Sector classification (Elektrik, Doğalgaz, Petrol, LPG)
- License type extraction
- Tariff decision parsing
- Price regulation extraction
"""
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup
from decimal import Decimal
import logging

from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata
from ..core.exceptions import (
    NetworkError,
    ParsingError,
    ValidationError,
    DocumentNotFoundError
)
from ..utils.cache_utils import TTLCache
from ..utils.retry_utils import retry
from ..utils.text_utils import normalize_turkish_text
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


class EPDKAdapter(SourceAdapter):
    """
    Production-grade EPDK (Energy Market Regulatory Authority) adapter.

    Handles:
    - Kurul Kararı (Board Decisions): Regulatory decisions
    - Tebliğ (Communiqués): Official announcements
    - Yönetmelik (Regulations): Energy market regulations
    - Lisans Kararları (License Decisions): Company licenses
    - Tarife Kararları (Tariff Decisions): Energy pricing decisions

    Energy Sectors:
    - Elektrik (Electricity): Generation, transmission, distribution
    - Doğalgaz (Natural Gas): Import, transmission, distribution
    - Petrol (Petroleum): Refining, distribution, storage
    - LPG (Liquefied Petroleum Gas): Distribution, autogas

    Features:
    - Rate limiting: 35 requests/minute
    - Caching: 1 hour TTL
    - Retry: 3 attempts with 2.0x backoff
    - Multiple selectors: 5-6 fallback CSS selectors per element
    - Decision number extraction (YYYY-NN format)
    - License type classification
    - Tariff amount extraction (Turkish Lira/kWh format)
    - Sector identification
    """

    # Multiple fallback CSS selectors
    SELECTORS = {
        'title': [
            'h1.icerik-baslik',
            'h1',
            'div.baslik h1',
            'div.page-title',
            'div.title'
        ],
        'content': [
            'div.icerik-detay',
            'div.content',
            'div.main-content',
            'article',
            'div.document-content'
        ],
        'date': [
            'span.tarih',
            'div.date',
            'span.publish-date',
            'div.metadata span.date'
        ],
        'decision_number': [
            'span.karar-no',
            'div.decision-number',
            'span.document-number'
        ]
    }

    # Document type classification
    DOCUMENT_TYPES = {
        'KURUL_KARARI': {
            'keywords': ['kurul kararı', 'kurul karar', 'board decision'],
            'doc_type': DocumentType.KURUL_KARARI
        },
        'TEBLIG': {
            'keywords': ['tebliğ', 'teblig', 'communique'],
            'doc_type': DocumentType.TEBLIG
        },
        'YONETMELIK': {
            'keywords': ['yönetmelik', 'yonetmelik', 'regulation'],
            'doc_type': DocumentType.YONETMELIK
        },
        'LISANS': {
            'keywords': ['lisans', 'license'],
            'doc_type': DocumentType.KURUL_KARARI
        },
        'TARIFE': {
            'keywords': ['tarife', 'fiyat', 'tariff', 'price'],
            'doc_type': DocumentType.KURUL_KARARI
        }
    }

    # Energy sectors
    ENERGY_SECTORS = {
        'ELEKTRİK': {
            'keywords': ['elektrik', 'electricity', 'enerji', 'güç'],
            'subsectors': ['üretim', 'iletim', 'dağıtım', 'perakende']
        },
        'DOĞALGAZ': {
            'keywords': ['doğalgaz', 'doğal gaz', 'natural gas', 'gaz'],
            'subsectors': ['ithalat', 'iletim', 'dağıtım', 'depolama']
        },
        'PETROL': {
            'keywords': ['petrol', 'petroleum', 'akaryakıt', 'rafineri'],
            'subsectors': ['rafineri', 'dağıtım', 'depolama', 'bayilik']
        },
        'LPG': {
            'keywords': ['lpg', 'sıvılaştırılmış petrol gazı', 'otogaz'],
            'subsectors': ['dağıtım', 'dolum', 'bayilik']
        }
    }

    # License types
    LICENSE_TYPES = {
        'ÜRETİM': 'Generation License',
        'İLETİM': 'Transmission License',
        'DAĞITIM': 'Distribution License',
        'PERAKENDE': 'Retail License',
        'İTHALAT': 'Import License',
        'İHRACAT': 'Export License',
        'DEPOLAMA': 'Storage License',
        'RAFİNERİ': 'Refinery License'
    }

    def __init__(self):
        super().__init__("EPDK Adapter", "2.0.0")
        self.base_url = "https://www.epdk.gov.tr"

        # Rate limiting: 35 requests/minute (moderate for EPDK)
        self.rate_limit = 35
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL Cache: 1 hour
        self.cache = TTLCache(ttl=3600)

        logger.info(f"Initialized {self.name} v{self.version} with rate limit {self.rate_limit} req/min")

    async def _enforce_rate_limit(self):
        """Sliding window rate limiting - 35 requests per minute"""
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

    @retry(max_attempts=3, backoff_factor=2.0, exceptions=(NetworkError, aiohttp.ClientError))
    async def _fetch_html(self, url: str, **kwargs) -> str:
        """Fetch HTML with rate limiting, caching, and retry logic"""
        # Check cache
        cache_key = f"html:{url}"
        if cached := self.cache.get(cache_key):
            logger.debug(f"Cache hit for {url}")
            return cached

        # Enforce rate limiting
        await self._enforce_rate_limit()

        # Prepare request
        timeout = aiohttp.ClientTimeout(
            total=kwargs.get('timeout', 30),
            connect=kwargs.get('connect_timeout', 10)
        )
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
            **(kwargs.get('headers', {}))
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug(f"Fetching {url}")
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status == 404:
                        raise DocumentNotFoundError(f"Document not found: {url}")
                    elif response.status >= 500:
                        raise NetworkError(f"Server error {response.status}: {url}")
                    elif response.status >= 400:
                        raise ValidationError(f"Client error {response.status}: {url}")

                    html = await response.text()

                    if len(html) < 100:
                        raise ValidationError(f"Response too short ({len(html)} bytes): {url}")

                    self.cache.set(cache_key, html)
                    logger.info(f"Successfully fetched {url} ({len(html)} bytes)")
                    return html

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching {url}: {e}")
            raise NetworkError(f"Failed to fetch {url}: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout fetching {url}: {e}")
            raise NetworkError(f"Timeout fetching {url}: {e}")

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """
        Fetches EPDK documents (Kurul Kararı, Tebliğ, Yönetmelik).

        Args:
            document_id: Document ID
            **kwargs: Additional parameters

        Returns:
            Dict with document data

        Raises:
            DocumentNotFoundError: If document doesn't exist
            ParsingError: If parsing fails
        """
        url = f"{self.base_url}/Detay/Icerik/{document_id}"

        try:
            # Fetch HTML
            html = await self._fetch_html(url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title using multiple selectors
            title = self._extract_with_selectors(soup, self.SELECTORS['title'])
            if not title:
                raise ParsingError(f"Could not extract title from {url}")

            # Extract content
            content = self._extract_with_selectors(soup, self.SELECTORS['content'])
            if not content:
                raise ParsingError(f"Could not extract content from {url}")

            # Extract decision number
            decision_number = self._extract_decision_number(title, soup)

            # Extract publication date
            publish_date = self._extract_publish_date(soup, content)

            # Classify document type
            doc_type = self._classify_document_type(title, content)

            # Identify energy sector
            sector = self._identify_sector(title, content)

            # Extract license information if applicable
            license_info = self._extract_license_info(content) if 'lisans' in content.lower() else None

            # Extract tariff information if applicable
            tariff_info = self._extract_tariff_info(content) if 'tarife' in content.lower() or 'fiyat' in content.lower() else None

            # Extract price regulations if applicable
            price_regulations = self._extract_price_regulations(content) if 'fiyat' in content.lower() else None

            logger.info(f"Successfully parsed EPDK document: {decision_number} ({doc_type}, {sector})")

            return {
                'title': title.strip(),
                'content': content.strip(),
                'url': url,
                'decision_number': decision_number,
                'publish_date': publish_date,
                'document_type': doc_type,
                'sector': sector,
                'license_info': license_info,
                'tariff_info': tariff_info,
                'price_regulations': price_regulations,
                'source': 'EPDK',
                'document_id': document_id
            }

        except (DocumentNotFoundError, ParsingError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing EPDK document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {e}")

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search EPDK documents by keyword.

        Args:
            query: Search query
            **kwargs: Additional parameters (page, limit, sector)

        Returns:
            List of search results
        """
        page = kwargs.get('page', 1)
        limit = kwargs.get('limit', 20)
        sector_filter = kwargs.get('sector')

        # Normalize query
        normalized_query = normalize_turkish_text(query)

        # Construct search URL
        search_url = f"{self.base_url}/Arama?q={normalized_query}&page={page}"

        if sector_filter:
            search_url += f"&sektor={sector_filter}"

        try:
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Try multiple selectors for result items
            result_selectors = [
                'div.search-result',
                'div.arama-sonuc',
                'div.result-item',
                'li.search-item'
            ]

            items = []
            for selector in result_selectors:
                items = soup.find_all(class_=selector.split('.')[1])
                if items:
                    logger.debug(f"Found {len(items)} results using selector: {selector}")
                    break

            if not items:
                logger.warning(f"No results found for query: {query}")
                return []

            # Parse each result
            for item in items[:limit]:
                try:
                    title_elem = item.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    href = title_elem.get('href', '')
                    url = self.base_url + href if href.startswith('/') else href

                    # Extract decision number
                    decision_number = self._extract_decision_number(title, None)

                    # Extract date
                    date_elem = item.find('span', class_='date') or item.find('span', class_='tarih')
                    date = date_elem.text.strip() if date_elem else None

                    # Extract summary
                    summary_elem = item.find('div', class_='summary') or item.find('p')
                    summary = summary_elem.text.strip() if summary_elem else ''

                    # Identify sector
                    sector = self._identify_sector(title, summary)

                    results.append({
                        'title': title,
                        'url': url,
                        'decision_number': decision_number,
                        'date': date,
                        'summary': summary,
                        'sector': sector
                    })

                except Exception as e:
                    logger.warning(f"Error parsing search result item: {e}")
                    continue

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise ParsingError(f"Search failed: {e}")

    def _extract_with_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
        """Try multiple CSS selectors and return first match"""
        for selector in selectors:
            if '.' in selector:
                tag, class_name = selector.split('.', 1)
                elem = soup.find(tag, class_=class_name)
            else:
                elem = soup.find(selector)

            if elem:
                logger.debug(f"Matched selector: {selector}")
                return elem.get_text(strip=True)

        return None

    def _extract_decision_number(self, title: str, soup: Optional[BeautifulSoup]) -> Optional[str]:
        """
        Extract EPDK decision number.

        Formats:
        - "Kurul Kararı: 1234-56" → "1234-56"
        - "Karar No: 1234-56" → "1234-56"
        """
        # Try structured extraction first
        if soup:
            for selector in self.SELECTORS['decision_number']:
                if '.' in selector:
                    tag, class_name = selector.split('.', 1)
                    elem = soup.find(tag, class_=class_name)
                else:
                    elem = soup.find(selector)

                if elem:
                    text = elem.get_text(strip=True)
                    match = re.search(r'(\d{4}-\d+)', text)
                    if match:
                        logger.debug(f"Extracted decision number: {match.group(1)}")
                        return match.group(1)

        # Fallback to title pattern matching
        patterns = [
            r'(\d{4}-\d+)',
            r'Kurul\s+Kararı\s*:?\s*(\d{4}-\d+)',
            r'Karar\s+(?:No|Sayısı)\s*:?\s*(\d{4}-\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                logger.debug(f"Extracted decision number from title: {match.group(1)}")
                return match.group(1)

        return None

    def _extract_publish_date(self, soup: BeautifulSoup, content: str) -> Optional[str]:
        """Extract publication date"""
        # Try structured extraction
        for selector in self.SELECTORS['date']:
            if '.' in selector:
                tag, class_name = selector.split('.', 1)
                elem = soup.find(tag, class_=class_name)
            else:
                elem = soup.find(selector)

            if elem:
                date_text = elem.get_text(strip=True)
                try:
                    return parse_turkish_date(date_text)
                except:
                    return date_text

        # Fallback to content pattern matching
        date_patterns = [
            r'Yayım(?:lanma)?\s+Tarihi\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'Tarih\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _classify_document_type(self, title: str, content: str) -> str:
        """Classify document type based on title and content"""
        text = (title + ' ' + content[:500]).lower()

        for doc_type, info in self.DOCUMENT_TYPES.items():
            if any(kw in text for kw in info['keywords']):
                logger.debug(f"Classified document as: {doc_type}")
                return doc_type

        return 'TEBLIG'  # Default

    def _identify_sector(self, title: str, content: str) -> Optional[str]:
        """Identify energy sector"""
        text = (title + ' ' + content[:1000]).lower()

        for sector, info in self.ENERGY_SECTORS.items():
            if any(kw in text for kw in info['keywords']):
                logger.debug(f"Identified sector: {sector}")
                return sector

        return None

    def _extract_license_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract license information"""
        license_info = {}

        # License type
        for license_type_key, license_type_name in self.LICENSE_TYPES.items():
            if license_type_key.lower() in text.lower():
                license_info['license_type'] = license_type_name
                break

        # Company name (license holder)
        company_pattern = r'(?:şirket(?:i)?|firma)\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))'
        match = re.search(company_pattern, text, re.IGNORECASE)
        if match:
            license_info['company'] = match.group(1).strip()

        # License number
        license_num_pattern = r'Lisans\s+(?:No|Numarası)\s*:?\s*([A-Z0-9/-]+)'
        match = re.search(license_num_pattern, text, re.IGNORECASE)
        if match:
            license_info['license_number'] = match.group(1)

        return license_info if license_info else None

    def _extract_tariff_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tariff information"""
        tariff_info = {}

        # Electricity tariff (kr/kWh or TL/kWh)
        electricity_pattern = r'([\d,]+)\s*(?:kr|kuruş|TL)/kWh'
        match = re.search(electricity_pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '.')
            try:
                tariff_info['electricity_tariff_kwh'] = Decimal(amount_str)
            except:
                pass

        # Natural gas tariff (TL/m³)
        gas_pattern = r'([\d.,]+)\s*TL/m[³3]'
        match = re.search(gas_pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace('.', '').replace(',', '.')
            try:
                tariff_info['gas_tariff_m3'] = Decimal(amount_str)
            except:
                pass

        return tariff_info if tariff_info else None

    def _extract_price_regulations(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract price regulation information"""
        regulations = {}

        # Maximum price (tavan fiyat)
        max_price_pattern = r'tavan\s+fiyat\s*:?\s*([\d.,]+)\s*TL'
        match = re.search(max_price_pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace('.', '').replace(',', '.')
            try:
                regulations['max_price'] = Decimal(amount_str)
            except:
                pass

        # Reference price (referans fiyat)
        ref_price_pattern = r'referans\s+fiyat\s*:?\s*([\d.,]+)\s*TL'
        match = re.search(ref_price_pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace('.', '').replace(',', '.')
            try:
                regulations['reference_price'] = Decimal(amount_str)
            except:
                pass

        return regulations if regulations else None

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from classification
        doc_type_str = raw_data.get('document_type', 'TEBLIG')

        if doc_type_str == 'YONETMELIK':
            doc_type = DocumentType.YONETMELIK
            hierarchy = LegalHierarchy.YONETMELIK
        elif doc_type_str == 'KURUL_KARARI':
            doc_type = DocumentType.KURUL_KARARI
            hierarchy = LegalHierarchy.KURUL_KARARI
        else:
            doc_type = DocumentType.TEBLIG
            hierarchy = LegalHierarchy.TEBLIG

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=hierarchy,
            source=SourceType.EPDK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            decision_number=raw_data.get('decision_number'),
            sector=raw_data.get('sector'),
            license_info=raw_data.get('license_info'),
            tariff_info=raw_data.get('tariff_info')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['EPDKAdapter']
