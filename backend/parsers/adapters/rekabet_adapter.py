"""Rekabet Kurumu (Competition Authority) Adapter - Harvey/Legora CTO-Level
Fetches competition law decisions and merger clearances from rekabet.gov.tr

Production-grade implementation with:
- Rate limiting (30 req/min sliding window)
- TTL caching (1 hour default)
- Retry logic with exponential backoff
- Multiple fallback CSS selectors
- Comprehensive error handling
- Decision classification (Merger/Board/Investigation)
- Company name extraction
- Search with pagination
- Turkish text processing
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup
from datetime import datetime
from decimal import Decimal
import logging

from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata
from ..core.exceptions import (
    NetworkError,
    ParsingError,
    ValidationError,
    RateLimitError,
    DocumentNotFoundError
)
from ..utils.cache_utils import TTLCache
from ..utils.retry_utils import retry
from ..utils.text_utils import normalize_turkish_text
from ..utils.date_utils import parse_turkish_date

logger = logging.getLogger(__name__)


class RekabetAdapter(SourceAdapter):
    """
    Production-grade Rekabet Kurumu (Competition Authority) adapter.

    Handles three main document types:
    1. Kurul Kararı (Board Decisions) - suffix: -K
    2. Birleşme/Devralma (Merger/Acquisition) - suffix: -M
    3. Soruşturma (Investigation) - suffix: -S

    Decision Number Format: YY-AA/BBB-C
    - YY: Year (23 = 2023)
    - AA: Meeting/Order number
    - BBB: Sequential number in that meeting
    - C: Type (K=Board, M=Merger, S=Investigation)

    Features:
    - Rate limiting: 30 requests/minute (Competition Authority is conservative)
    - Caching: 1 hour TTL for decisions (they rarely change)
    - Retry: 3 attempts with 2.0x backoff for network issues
    - Multiple selectors: 5-6 fallback CSS selectors per element
    - Company extraction: Identifies parties in mergers/investigations
    - Sector classification: Determines market sector
    - Fine amount parsing: Extracts administrative fines
    """

    # Multiple fallback CSS selectors for robustness
    SELECTORS = {
        'title': [
            'h1.karar-baslik',
            'h1.decision-title',
            'div.baslik h1',
            'h1',
            'div.page-title h1',
            'div.title'
        ],
        'content': [
            'div.karar-icerik',
            'div.decision-content',
            'div.content',
            'div.main-content',
            'article.decision',
            'div.text-content'
        ],
        'summary': [
            'div.karar-ozeti',
            'div.summary',
            'div.ozet',
            'p.summary',
            'div.abstract'
        ],
        'parties': [
            'div.taraflar',
            'div.parties',
            'div.companies',
            'div.sirketler',
            'table.parties-table'
        ],
        'decision_info': [
            'div.karar-bilgileri',
            'div.decision-info',
            'div.metadata',
            'table.info-table',
            'div.details'
        ],
        'sector': [
            'span.sektor',
            'div.sector',
            'span.market',
            'div.market-info'
        ]
    }

    # Decision type classification
    DECISION_TYPES = {
        'M': {
            'code': 'MERGER',
            'name': 'Birleşme/Devralma',
            'doc_type': DocumentType.BIRLESMELER
        },
        'K': {
            'code': 'BOARD',
            'name': 'Kurul Kararı',
            'doc_type': DocumentType.KURUL_KARARI
        },
        'S': {
            'code': 'INVESTIGATION',
            'name': 'Soruşturma Kararı',
            'doc_type': DocumentType.KURUL_KARARI
        },
        'Y': {
            'code': 'EXEMPTION',
            'name': 'Muafiyet Kararı',
            'doc_type': DocumentType.KURUL_KARARI
        }
    }

    # Market sectors for classification
    SECTORS = {
        'enerji': 'Enerji',
        'telekomünikasyon': 'Telekomünikasyon',
        'finans': 'Finansal Hizmetler',
        'perakende': 'Perakende',
        'otomotiv': 'Otomotiv',
        'gida': 'Gıda',
        'saglik': 'Sağlık',
        'insaat': 'İnşaat',
        'medya': 'Medya',
        'ulastirma': 'Ulaştırma',
        'turizm': 'Turizm',
        'teknoloji': 'Teknoloji'
    }

    # Violation types (for investigation decisions)
    VIOLATION_TYPES = {
        'rekabeti_sinirlandirma': '4054 sayılı Kanun md. 4 - Rekabeti Sınırlayıcı Anlaşma',
        'hakim_durum': '4054 sayılı Kanun md. 6 - Hâkim Durumun Kötüye Kullanılması',
        'birlesmeler': '4054 sayılı Kanun md. 7 - İzinsiz Birleşme/Devralma',
        'aydinlatma': '4054 sayılı Kanun md. 15 - Aydınlatma Yükümlülüğü İhlali'
    }

    def __init__(self):
        super().__init__("Rekabet Kurumu Adapter", "2.0.0")
        self.base_url = "https://www.rekabet.gov.tr"
        self.search_url = f"{self.base_url}/KararArama"

        # Rate limiting: 30 requests/minute (sliding window)
        self.rate_limit = 30
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL Cache: 1 hour for decisions
        self.cache = TTLCache(ttl=3600)

        logger.info(f"Initialized {self.name} v{self.version} with rate limit {self.rate_limit} req/min")

    async def _enforce_rate_limit(self):
        """Sliding window rate limiting - 30 requests per minute"""
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()

            # Remove requests older than 60 seconds
            self.request_times = [t for t in self.request_times if now - t < 60]

            # Check if we've hit the limit
            if len(self.request_times) >= self.rate_limit:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    now = asyncio.get_event_loop().time()
                    self.request_times = [t for t in self.request_times if now - t < 60]

            # Add current request
            self.request_times.append(now)

    @retry(max_attempts=3, backoff_factor=2.0, exceptions=(NetworkError, aiohttp.ClientError))
    async def _fetch_html(self, url: str, **kwargs) -> str:
        """
        Fetch HTML with rate limiting, caching, and retry logic.

        Args:
            url: URL to fetch
            **kwargs: Additional parameters (timeout, headers, etc.)

        Returns:
            HTML content as string

        Raises:
            NetworkError: If all retry attempts fail
            ValidationError: If response is invalid
        """
        # Check cache first
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
                    # Check status
                    if response.status == 404:
                        raise DocumentNotFoundError(f"Document not found: {url}")
                    elif response.status >= 500:
                        raise NetworkError(f"Server error {response.status}: {url}")
                    elif response.status >= 400:
                        raise ValidationError(f"Client error {response.status}: {url}")

                    html = await response.text()

                    # Validate response
                    if len(html) < 100:
                        raise ValidationError(f"Response too short ({len(html)} bytes): {url}")

                    # Cache successful response
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
        Fetches Rekabet Kurumu decision by ID.

        Args:
            document_id: Decision ID or decision number (e.g., "23-15/345-M" or numeric ID)
            **kwargs: Additional fetch parameters

        Returns:
            Dict with document data

        Raises:
            DocumentNotFoundError: If document doesn't exist
            ParsingError: If parsing fails
        """
        # Construct URL
        url = f"{self.base_url}/Karar?kararId={document_id}"

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

            # Extract decision number and classify
            decision_number = self._extract_decision_number(title)
            decision_type = self._classify_decision(decision_number, title, content)

            # Extract decision date
            decision_date = self._extract_decision_date(title, soup)

            # Extract parties (for mergers and investigations)
            parties = self._extract_parties(soup, content, decision_type)

            # Extract sector
            sector = self._extract_sector(soup, title, content)

            # Extract summary
            summary = self._extract_with_selectors(soup, self.SELECTORS['summary'])

            # Extract fine amount (for investigation decisions)
            fine_amount = self._extract_fine_amount(content) if decision_type in ['INVESTIGATION', 'BOARD'] else None

            # Extract violation type (for investigation decisions)
            violation_type = self._extract_violation_type(content) if decision_type in ['INVESTIGATION', 'BOARD'] else None

            # Extract decision result (approval, rejection, fine, etc.)
            decision_result = self._extract_decision_result(content, decision_type)

            logger.info(f"Successfully parsed Rekabet decision: {decision_number} ({decision_type})")

            return {
                'title': title.strip(),
                'content': content.strip(),
                'url': url,
                'decision_number': decision_number,
                'decision_date': decision_date,
                'decision_type': decision_type,
                'parties': parties,
                'sector': sector,
                'summary': summary.strip() if summary else None,
                'fine_amount': fine_amount,
                'violation_type': violation_type,
                'decision_result': decision_result,
                'source': 'Rekabet Kurumu',
                'document_id': document_id
            }

        except (DocumentNotFoundError, ParsingError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing Rekabet document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {e}")

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search Rekabet Kurumu decisions by keyword.

        Args:
            query: Search query
            **kwargs: Additional search parameters (page, limit, filters)

        Returns:
            List of search results with title, url, summary, decision_number, date
        """
        page = kwargs.get('page', 1)
        limit = kwargs.get('limit', 20)

        # Normalize Turkish characters in query
        normalized_query = normalize_turkish_text(query)

        # Construct search URL
        search_url = f"{self.search_url}?q={normalized_query}&page={page}"

        # Add filters if provided
        if decision_type := kwargs.get('decision_type'):
            search_url += f"&type={decision_type}"
        if year := kwargs.get('year'):
            search_url += f"&year={year}"
        if sector := kwargs.get('sector'):
            search_url += f"&sector={sector}"

        try:
            # Fetch search results
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Try multiple selectors for result items
            result_selectors = [
                'div.karar-item',
                'div.decision-item',
                'div.search-result',
                'li.result-item',
                'article.karar'
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
                    # Extract title and URL
                    title_elem = item.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    href = title_elem.get('href', '')
                    url = self.base_url + href if href.startswith('/') else href

                    # Extract decision number from title
                    decision_number = self._extract_decision_number(title)

                    # Extract date
                    date_elem = item.find('span', class_='date') or item.find('span', class_='tarih')
                    date = date_elem.text.strip() if date_elem else None

                    # Extract summary
                    summary_elem = item.find('div', class_='ozet') or item.find('div', class_='summary') or item.find('p')
                    summary = summary_elem.text.strip() if summary_elem else ''

                    # Extract sector if available
                    sector_elem = item.find('span', class_='sektor') or item.find('span', class_='sector')
                    sector = sector_elem.text.strip() if sector_elem else None

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

    def _extract_decision_number(self, text: str) -> Optional[str]:
        """
        Extract Rekabet decision number.

        Format: YY-AA/BBB-C
        - YY: Year (23 = 2023)
        - AA: Meeting/Order number
        - BBB: Sequential number
        - C: Type (K/M/S/Y)

        Examples:
        - "23-15/345-M" (2023, 15th meeting, 345th decision, Merger)
        - "22-05/123-K" (2022, 5th meeting, 123rd decision, Board)
        """
        patterns = [
            r'(\d{2}-\d+/\d+-[KMSY])',  # Full format: 23-15/345-M
            r'(\d{2}-\d+/\d+)',          # Without type: 23-15/345
            r'Karar\s+(?:No|Sayısı)\s*:?\s*(\d{2}-\d+/\d+-[KMSY])',  # With label
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                logger.debug(f"Extracted decision number: {match.group(1)}")
                return match.group(1)

        return None

    def _classify_decision(self, decision_number: Optional[str], title: str, content: str) -> str:
        """
        Classify decision type based on decision number suffix and content.

        Returns: MERGER, BOARD, INVESTIGATION, EXEMPTION, or UNKNOWN
        """
        if decision_number:
            # Check suffix
            for suffix, info in self.DECISION_TYPES.items():
                if decision_number.endswith(f'-{suffix}'):
                    logger.debug(f"Classified as {info['code']} from suffix -{suffix}")
                    return info['code']

        # Fallback to content analysis
        text = (title + ' ' + content[:2000]).upper()

        if any(kw in text for kw in ['BİRLEŞME', 'DEVRALMA', 'MERGER', 'ACQUISITION']):
            return 'MERGER'
        elif any(kw in text for kw in ['SORUŞTURMA', 'İHLAL', 'INVESTIGATION', 'VIOLATION']):
            return 'INVESTIGATION'
        elif any(kw in text for kw in ['MUAFİYET', 'EXEMPTION']):
            return 'EXEMPTION'
        elif any(kw in text for kw in ['KURUL KARARI', 'BOARD DECISION']):
            return 'BOARD'

        return 'UNKNOWN'

    def _extract_decision_date(self, title: str, soup: BeautifulSoup) -> Optional[str]:
        """Extract decision date from title or metadata"""
        # Try title first
        date_match = re.search(r'(\d{1,2}[./]\d{1,2}[./]\d{4})', title)
        if date_match:
            return date_match.group(1)

        # Try metadata area
        for selector in self.SELECTORS['decision_info']:
            elem = soup.find(class_=selector.split('.')[1]) if '.' in selector else soup.find(selector)
            if elem:
                date_match = re.search(r'(\d{1,2}[./]\d{1,2}[./]\d{4})', elem.text)
                if date_match:
                    return date_match.group(1)

        return None

    def _extract_parties(self, soup: BeautifulSoup, content: str, decision_type: str) -> List[str]:
        """
        Extract party names (companies) involved in decision.

        For mergers: Acquirer and target companies
        For investigations: Companies under investigation
        """
        parties = []

        # Try structured extraction first
        for selector in self.SELECTORS['parties']:
            elem = soup.find(class_=selector.split('.')[1]) if '.' in selector else soup.find(selector)
            if elem:
                # Extract from table or list
                if elem.name == 'table':
                    for row in elem.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            parties.append(cells[1].get_text(strip=True))
                else:
                    # Extract from divs/lists
                    for item in elem.find_all(['div', 'li', 'p']):
                        text = item.get_text(strip=True)
                        if text and len(text) > 3:
                            parties.append(text)

        # If no structured data, try pattern matching
        if not parties:
            parties = self._extract_parties_from_text(content, decision_type)

        # Clean and deduplicate
        parties = list(set([p.strip() for p in parties if p.strip()]))
        logger.debug(f"Extracted {len(parties)} parties: {parties[:3]}...")

        return parties

    def _extract_parties_from_text(self, text: str, decision_type: str) -> List[str]:
        """Extract company names from text using patterns"""
        parties = []

        if decision_type == 'MERGER':
            # Look for merger-specific patterns
            patterns = [
                r'devralan\s+(?:şirket|firma)\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))',
                r'devrolunan\s+(?:şirket|firma)\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))',
                r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))\s+(?:ile|ve)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))',
            ]
        else:
            # Investigation patterns
            patterns = [
                r'soruşturma\s+açılan\s+(?:şirket|firma|teşebbüs)\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))',
                r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:A\.?Ş\.?|Ltd\.?\s*Şti\.?))\s+hakkında',
            ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                for i in range(1, len(match.groups()) + 1):
                    if company := match.group(i):
                        parties.append(company.strip())

        return parties

    def _extract_sector(self, soup: BeautifulSoup, title: str, content: str) -> Optional[str]:
        """Extract market sector"""
        # Try structured extraction
        for selector in self.SELECTORS['sector']:
            elem = soup.find(class_=selector.split('.')[1]) if '.' in selector else soup.find(selector)
            if elem:
                return elem.get_text(strip=True)

        # Pattern matching in content
        text = (title + ' ' + content[:1000]).lower()
        for keyword, sector in self.SECTORS.items():
            if keyword in text:
                logger.debug(f"Classified sector as: {sector}")
                return sector

        return None

    def _extract_fine_amount(self, text: str) -> Optional[Decimal]:
        """
        Extract administrative fine amount in Turkish Lira.

        Handles Turkish number format: 1.234.567,89 TL
        """
        patterns = [
            r'([\d.]+,\d{2})\s*(?:TL|₺|Türk\s+Lirası)',
            r'idari\s+para\s+cezası\s*:?\s*([\d.]+,\d{2})',
            r'ceza\s+tutarı\s*:?\s*([\d.]+,\d{2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1)
                # Convert Turkish format to Decimal
                # 1.234.567,89 → 1234567.89
                amount_str = amount_str.replace('.', '')  # Remove thousand separators
                amount_str = amount_str.replace(',', '.') # Replace decimal comma

                try:
                    amount = Decimal(amount_str)
                    logger.debug(f"Extracted fine amount: {amount} TL")
                    return amount
                except:
                    continue

        return None

    def _extract_violation_type(self, text: str) -> Optional[str]:
        """Classify violation type for investigation decisions"""
        text_upper = text.upper()

        for key, description in self.VIOLATION_TYPES.items():
            # Check for specific article references
            if 'MADDE 4' in text_upper or 'MD. 4' in text_upper or 'M. 4' in text_upper:
                if any(kw in text_upper for kw in ['ANLAŞMA', 'UYUMLU', 'KOORDİNASYON']):
                    return self.VIOLATION_TYPES['rekabeti_sinirlandirma']

            if 'MADDE 6' in text_upper or 'MD. 6' in text_upper or 'M. 6' in text_upper:
                if any(kw in text_upper for kw in ['HÂKİM', 'DOMINANT', 'KÖTÜYE']):
                    return self.VIOLATION_TYPES['hakim_durum']

            if 'MADDE 7' in text_upper or 'MD. 7' in text_upper:
                if any(kw in text_upper for kw in ['BİRLEŞME', 'DEVRALMA', 'İZİN']):
                    return self.VIOLATION_TYPES['birlesmeler']

            if 'MADDE 15' in text_upper or 'MD. 15' in text_upper:
                return self.VIOLATION_TYPES['aydinlatma']

        return None

    def _extract_decision_result(self, text: str, decision_type: str) -> Optional[str]:
        """
        Extract decision result (approval, rejection, fine, etc.)

        Returns: APPROVED, REJECTED, CONDITIONAL_APPROVAL, FINED, DISMISSED
        """
        text_upper = text.upper()

        if decision_type == 'MERGER':
            if any(kw in text_upper for kw in ['UYGUN GÖRÜLDÜ', 'ONAYLANDI', 'İZİN VERİLDİ']):
                return 'APPROVED'
            elif any(kw in text_upper for kw in ['KOŞULLU', 'ŞARTLI', 'TAAHHÜTİLE']):
                return 'CONDITIONAL_APPROVAL'
            elif any(kw in text_upper for kw in ['REDDEDİLDİ', 'UYGUN GÖRÜLMEDI']):
                return 'REJECTED'

        elif decision_type in ['INVESTIGATION', 'BOARD']:
            if any(kw in text_upper for kw in ['İHLAL TESPİT', 'PARA CEZASI', 'CEZA VERİLMESİNE']):
                return 'FINED'
            elif any(kw in text_upper for kw in ['YETERSİZ KANIT', 'DÜŞÜRÜLDÜ', 'İHLAL YOK']):
                return 'DISMISSED'

        return None

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        """Pass through preprocessed data"""
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        """
        Transform Rekabet data to canonical LegalDocument format.

        Args:
            raw_data: Raw extracted data
            document_type: Optional document type override
            **kwargs: Additional parameters

        Returns:
            LegalDocument in canonical schema
        """
        from ..core.canonical_schema import (
            LegalDocument,
            Metadata,
            SourceType,
            JurisdictionType,
            LegalHierarchy,
            EffectivityStatus
        )

        # Determine document type
        decision_type_code = raw_data.get('decision_type', 'UNKNOWN')

        if decision_type_code == 'MERGER':
            doc_type = DocumentType.BIRLESMELER
        elif decision_type_code in ['INVESTIGATION', 'BOARD', 'EXEMPTION']:
            doc_type = DocumentType.KURUL_KARARI
        else:
            doc_type = document_type or DocumentType.KURUL_KARARI

        # Create metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_KARAR,
            source=SourceType.REKABET,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            decision_number=raw_data.get('decision_number'),
            decision_date=raw_data.get('decision_date'),
            # Additional metadata
            parties=raw_data.get('parties', []),
            sector=raw_data.get('sector'),
            fine_amount=str(raw_data.get('fine_amount')) if raw_data.get('fine_amount') else None,
            violation_type=raw_data.get('violation_type'),
            decision_result=raw_data.get('decision_result')
        )

        # Create canonical document
        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', ''),
            summary=raw_data.get('summary')
        )


__all__ = ['RekabetAdapter']
