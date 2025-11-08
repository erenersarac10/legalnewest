"""SGK (Social Security Institution) Adapter - Harvey/Legora CTO-Level
Fetches circulars, guides, and rulings from sgk.gov.tr

Production-grade implementation with:
- Rate limiting (40 req/min sliding window)
- TTL caching (1 hour default)
- Retry logic with exponential backoff
- Multiple fallback CSS selectors
- Comprehensive error handling
- Circular/Tebliğ classification
- Premium rate extraction
- Healthcare rights parsing
- Retirement calculation references
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


class SGKAdapter(SourceAdapter):
    """
    Production-grade SGK (Social Security Institution) adapter.

    Handles:
    - Genelge (Circulars): General instructions
    - Tebliğ (Communiqués): Official announcements
    - Rehber (Guides): Procedural guides
    - Prim Oranları (Premium Rates): Contribution rates
    - Sağlık Hakları (Healthcare Rights): Health insurance coverage
    - Emeklilik (Retirement): Pension regulations

    Features:
    - Rate limiting: 40 requests/minute
    - Caching: 1 hour TTL
    - Retry: 3 attempts with 2.0x backoff
    - Multiple selectors: 5-6 fallback CSS selectors per element
    - Premium rate extraction with percentage parsing
    - Circular number extraction (YYYY/NN format)
    - Document classification by type
    - Healthcare coverage detection
    """

    # Multiple fallback CSS selectors
    SELECTORS = {
        'title': [
            'h1.document-title',
            'h1',
            'div.baslik h1',
            'div.page-title',
            'div.title'
        ],
        'content': [
            'div.icerik',
            'div.detay',
            'div.content',
            'div.main-content',
            'article'
        ],
        'date': [
            'span.tarih',
            'div.date',
            'span.publish-date',
            'div.metadata span.date'
        ],
        'circular_number': [
            'span.genelge-no',
            'div.circular-number',
            'span.document-number'
        ]
    }

    # Document type classification
    DOCUMENT_TYPES = {
        'GENELGE': {
            'keywords': ['genelge', 'circular'],
            'doc_type': DocumentType.GENELGE
        },
        'TEBLIG': {
            'keywords': ['tebliğ', 'teblig', 'communique'],
            'doc_type': DocumentType.TEBLIG
        },
        'REHBER': {
            'keywords': ['rehber', 'kılavuz', 'kilavuz', 'guide'],
            'doc_type': DocumentType.REHBER
        },
        'PRIM': {
            'keywords': ['prim', 'premium', 'katkı', 'contribution'],
            'doc_type': DocumentType.GENELGE
        }
    }

    # Premium rate categories
    PREMIUM_CATEGORIES = {
        'ÇalışanPrimiOranı': 'Employee Premium Rate',
        'İşverenPrimiOranı': 'Employer Premium Rate',
        'DevletKatkısı': 'State Contribution',
        'AsgariÜcret': 'Minimum Wage',
        'TavanÜcret': 'Maximum Wage Base'
    }

    def __init__(self):
        super().__init__("SGK Adapter", "2.0.0")
        self.base_url = "https://www.sgk.gov.tr"

        # Rate limiting: 40 requests/minute (SGK is moderate)
        self.rate_limit = 40
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL Cache: 1 hour
        self.cache = TTLCache(ttl=3600)

        logger.info(f"Initialized {self.name} v{self.version} with rate limit {self.rate_limit} req/min")

    async def _enforce_rate_limit(self):
        """Sliding window rate limiting - 40 requests per minute"""
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
        Fetches SGK documents (Genelge, Tebliğ, Rehber).

        Args:
            document_id: Document ID or circular number
            **kwargs: Additional parameters

        Returns:
            Dict with document data

        Raises:
            DocumentNotFoundError: If document doesn't exist
            ParsingError: If parsing fails
        """
        url = f"{self.base_url}/Sayfa/Detail/{document_id}"

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

            # Extract circular/document number
            circular_number = self._extract_circular_number(title, soup)

            # Extract publication date
            publish_date = self._extract_publish_date(soup, content)

            # Classify document type
            doc_type = self._classify_document_type(title, content)

            # Extract premium rates if applicable
            premium_rates = self._extract_premium_rates(content) if 'prim' in content.lower() else None

            # Extract healthcare rights if applicable
            healthcare_rights = self._extract_healthcare_rights(content) if 'sağlık' in content.lower() else None

            # Extract retirement information if applicable
            retirement_info = self._extract_retirement_info(content) if 'emeklilik' in content.lower() else None

            logger.info(f"Successfully parsed SGK document: {circular_number} ({doc_type})")

            return {
                'title': title.strip(),
                'content': content.strip(),
                'url': url,
                'circular_number': circular_number,
                'publish_date': publish_date,
                'document_type': doc_type,
                'premium_rates': premium_rates,
                'healthcare_rights': healthcare_rights,
                'retirement_info': retirement_info,
                'source': 'SGK',
                'document_id': document_id
            }

        except (DocumentNotFoundError, ParsingError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing SGK document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {e}")

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search SGK documents by keyword.

        Args:
            query: Search query
            **kwargs: Additional parameters (page, limit, doc_type)

        Returns:
            List of search results
        """
        page = kwargs.get('page', 1)
        limit = kwargs.get('limit', 20)
        doc_type_filter = kwargs.get('doc_type')

        # Normalize query
        normalized_query = normalize_turkish_text(query)

        # Construct search URL (SGK search endpoint)
        search_url = f"{self.base_url}/Arama?q={normalized_query}&page={page}"

        if doc_type_filter:
            search_url += f"&type={doc_type_filter}"

        try:
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Try multiple selectors for result items
            result_selectors = [
                'div.search-result-item',
                'div.arama-sonuc',
                'div.result-item',
                'li.search-result'
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

                    # Extract circular number from title
                    circular_number = self._extract_circular_number(title, None)

                    # Extract date if available
                    date_elem = item.find('span', class_='date') or item.find('span', class_='tarih')
                    date = date_elem.text.strip() if date_elem else None

                    # Extract summary
                    summary_elem = item.find('div', class_='summary') or item.find('p')
                    summary = summary_elem.text.strip() if summary_elem else ''

                    results.append({
                        'title': title,
                        'url': url,
                        'circular_number': circular_number,
                        'date': date,
                        'summary': summary
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

    def _extract_circular_number(self, title: str, soup: Optional[BeautifulSoup]) -> Optional[str]:
        """
        Extract SGK circular number.

        Formats:
        - "2023/15 Sayılı Genelge" → "2023/15"
        - "Genelge No: 2023/15" → "2023/15"
        """
        # Try structured extraction first
        if soup:
            for selector in self.SELECTORS['circular_number']:
                if '.' in selector:
                    tag, class_name = selector.split('.', 1)
                    elem = soup.find(tag, class_=class_name)
                else:
                    elem = soup.find(selector)

                if elem:
                    text = elem.get_text(strip=True)
                    match = re.search(r'(\d{4}/\d+)', text)
                    if match:
                        logger.debug(f"Extracted circular number: {match.group(1)}")
                        return match.group(1)

        # Fallback to title pattern matching
        patterns = [
            r'(\d{4}/\d+)\s+[Ss]ayılı',
            r'(?:Genelge|Tebliğ)\s+(?:No|Sayısı)\s*:?\s*(\d{4}/\d+)',
            r'(\d{4}/\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                logger.debug(f"Extracted circular number from title: {match.group(1)}")
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
                # Try to parse Turkish date
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

        return 'GENELGE'  # Default

    def _extract_premium_rates(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract premium rates (prim oranları).

        Handles percentage format: %14.5, %20
        """
        rates = {}

        # Employee premium rate
        employee_pattern = r'[Çç]alışan(?:\s+prim(?:i)?)?\s+oran[ıi]\s*:?\s*%?\s*([\d,]+)'
        match = re.search(employee_pattern, text, re.IGNORECASE)
        if match:
            rate_str = match.group(1).replace(',', '.')
            try:
                rates['employee_rate'] = Decimal(rate_str)
            except:
                pass

        # Employer premium rate
        employer_pattern = r'[İi]şveren(?:\s+prim(?:i)?)?\s+oran[ıi]\s*:?\s*%?\s*([\d,]+)'
        match = re.search(employer_pattern, text, re.IGNORECASE)
        if match:
            rate_str = match.group(1).replace(',', '.')
            try:
                rates['employer_rate'] = Decimal(rate_str)
            except:
                pass

        # State contribution
        state_pattern = r'[Dd]evlet\s+katkı(?:sı)?\s*:?\s*%?\s*([\d,]+)'
        match = re.search(state_pattern, text, re.IGNORECASE)
        if match:
            rate_str = match.group(1).replace(',', '.')
            try:
                rates['state_contribution'] = Decimal(rate_str)
            except:
                pass

        return rates if rates else None

    def _extract_healthcare_rights(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract healthcare rights information"""
        rights = {}

        # Check for coverage mentions
        if 'genel sağlık sigortası' in text.lower() or 'gss' in text.lower():
            rights['general_health_insurance'] = True

        if 'ilaç' in text.lower():
            rights['medication_coverage'] = True

        if 'hastane' in text.lower() or 'tedavi' in text.lower():
            rights['hospital_coverage'] = True

        return rights if rights else None

    def _extract_retirement_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract retirement information"""
        info = {}

        # Retirement age
        age_pattern = r'emeklilik\s+yaş[ıi]\s*:?\s*(\d+)'
        match = re.search(age_pattern, text, re.IGNORECASE)
        if match:
            info['retirement_age'] = int(match.group(1))

        # Premium days requirement
        days_pattern = r'(\d+)\s+gün\s+prim'
        match = re.search(days_pattern, text, re.IGNORECASE)
        if match:
            info['required_premium_days'] = int(match.group(1))

        return info if info else None

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from title
        title_lower = raw_data.get('title', '').lower()
        if 'genelge' in title_lower:
            doc_type = DocumentType.GENELGE
        elif 'tebliğ' in title_lower:
            doc_type = DocumentType.TEBLIG
        elif 'rehber' in title_lower or 'kılavuz' in title_lower:
            doc_type = DocumentType.REHBER
        else:
            doc_type = DocumentType.GENELGE

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.GENELGE,
            source=SourceType.SGK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            circular_number=raw_data.get('circular_number')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['SGKAdapter']
