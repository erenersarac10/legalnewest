"""TBMM (Turkish Grand National Assembly) Adapter - Harvey/Legora CTO-Level Production-Grade
Fetches legislation, parliamentary minutes, and commission reports from tbmm.gov.tr

Production Features:
- Rate limiting: 40 requests/minute sliding window
- TTL caching: 1-hour default with cache key management
- Retry logic: Exponential backoff (3 attempts, 2.0x factor)
- Multiple fallback CSS selectors for robust parsing
- Comprehensive error handling and logging
- Parliamentary-specific extraction: bill tracking, voting, commissions
- Turkish legislative terminology support
"""
from typing import Dict, List, Any, Optional
import aiohttp
import re
import asyncio
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata
from ..errors import NetworkError, ParsingError, ValidationError, DocumentNotFoundError
from ..utils import retry, parse_turkish_date, normalize_turkish_text

logger = logging.getLogger(__name__)


class TBMMAdapter(SourceAdapter):
    """Turkish Grand National Assembly (TBMM) Adapter

    Handles legislative bills, commission reports, plenary minutes, and parliamentary decisions
    from the Turkish Parliament.
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
            'div#content',
            'div.main-content',
            'article',
            'div.document-body'
        ],
        'info_table': [
            'table.bilgi',
            'table.info',
            'table.document-info',
            'div.bilgi table',
            'table.metadata'
        ],
        'voting_section': [
            'div.oylama',
            'div.voting',
            'table.oylama-sonuc',
            'div.vote-results'
        ],
        'commission_info': [
            'div.komisyon',
            'div.commission',
            'div.komisyon-bilgi'
        ]
    }

    # Document types with Turkish/English keywords
    DOCUMENT_TYPES = {
        'KANUN_TEKLİFİ': {
            'keywords': ['kanun teklif', 'bill proposal', 'milletvekili teklif'],
            'type_code': 'KT',
            'description': 'Parliamentary Bill Proposal'
        },
        'KANUN_TASARISI': {
            'keywords': ['kanun tasarı', 'government bill', 'hükümet tasarı'],
            'type_code': 'KTS',
            'description': 'Government Bill Draft'
        },
        'KOMİSYON_RAPORU': {
            'keywords': ['komisyon rapor', 'commission report'],
            'type_code': 'KR',
            'description': 'Commission Report'
        },
        'TUTANAK': {
            'keywords': ['tutanak', 'plenary minutes', 'birleşim tutanak'],
            'type_code': 'TT',
            'description': 'Plenary Minutes'
        },
        'MECLİS_KARARI': {
            'keywords': ['meclis karar', 'parliamentary decision', 'tbmm karar'],
            'type_code': 'MK',
            'description': 'Parliamentary Decision'
        }
    }

    # Bill status tracking
    BILL_STATUS = {
        'KOMİSYONDA': ['komisyon', 'commission stage', 'komisyonda'],
        'GENEL_KURULDA': ['genel kurul', 'plenary', 'görüşül'],
        'KABUL_EDİLDİ': ['kabul edildi', 'adopted', 'accepted', 'onaylandı'],
        'REDDEDİLDİ': ['reddedildi', 'rejected', 'declined'],
        'GERİ_ÇEKİLDİ': ['geri çekildi', 'withdrawn', 'geri alındı'],
        'YASALASTI': ['yasalaştı', 'enacted', 'kanunlaştı']
    }

    # Parliamentary commissions
    COMMISSIONS = {
        'İÇİŞLERİ': ['içişleri komisyon', 'internal affairs'],
        'DIŞİŞLERİ': ['dışişleri komisyon', 'foreign affairs'],
        'PLAN_BÜTÇE': ['plan ve bütçe', 'plan and budget', 'bütçe komisyon'],
        'ADLİYE': ['adalet komisyon', 'justice commission', 'adliye'],
        'MİLLİ_EĞİTİM': ['milli eğitim', 'national education'],
        'SAĞLIK': ['sağlık komisyon', 'health commission'],
        'ÇEVRE': ['çevre komisyon', 'environment commission'],
        'ANAYASA': ['anayasa komisyon', 'constitution commission'],
        'TARIM': ['tarım komisyon', 'agriculture commission'],
        'SANAYİ': ['sanayi komisyon', 'industry commission'],
        'TURİZM': ['turizm komisyon', 'tourism commission'],
        'ULAŞTIRMA': ['ulaştırma komisyon', 'transportation commission']
    }

    # Legislative periods (Dönem)
    # TBMM has been through 28 legislative periods (as of 2024)
    LEGISLATIVE_PERIODS = list(range(1, 29))

    def __init__(self):
        super().__init__("TBMM Adapter", "2.0.0")
        self.base_url = "https://www.tbmm.gov.tr"

        # Rate limiting: 40 requests per minute (sliding window)
        self.rate_limit = 40
        self.rate_window = 60  # seconds
        self.request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL caching: 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        self._cache_lock = asyncio.Lock()

        logger.info(f"Initialized {self.name} v{self.version} with rate limit {self.rate_limit}/min")

    async def _enforce_rate_limit(self):
        """Sliding window rate limiting - 40 requests per minute"""
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
            # Try ID selector
            if selector.startswith('div#'):
                elem_id = selector.split('#')[-1]
                elem = soup.find(id=elem_id)
                if elem:
                    logger.debug(f"Found element with selector: {selector}")
                    return elem
            # Try class selector
            elif '.' in selector and not selector.startswith('div.'):
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
        """Classify parliamentary document type based on keywords

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

    def _extract_legislative_numbers(self, info_table: BeautifulSoup) -> Dict[str, Optional[str]]:
        """Extract legislative numbers from info table

        Args:
            info_table: BeautifulSoup table element

        Returns:
            Dict with esas_no, karar_no, sira_sayisi, donem, yasama_yili
        """
        numbers = {
            'esas_no': None,
            'karar_no': None,
            'sira_sayisi': None,
            'donem': None,
            'yasama_yili': None
        }

        if not info_table:
            return numbers

        for row in info_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                label = cells[0].text.strip().lower()
                value = cells[1].text.strip()

                if 'esas' in label:
                    numbers['esas_no'] = value
                elif 'karar' in label:
                    numbers['karar_no'] = value
                elif 'sıra sayı' in label or 'sira sayi' in label:
                    numbers['sira_sayisi'] = value
                elif 'dönem' in label or 'donem' in label:
                    # Extract period number
                    period_match = re.search(r'(\d+)', value)
                    if period_match:
                        numbers['donem'] = period_match.group(1)
                elif 'yasama yıl' in label or 'yasama yil' in label:
                    numbers['yasama_yili'] = value

        return numbers

    def _extract_bill_status(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract bill status from text

        Args:
            text: Text containing status information

        Returns:
            Dict with status info or None
        """
        text_lower = text.lower()

        for status, keywords in self.BILL_STATUS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'status': status,
                        'keyword': keyword,
                        'description': self._get_status_description(status)
                    }

        return None

    def _get_status_description(self, status: str) -> str:
        """Get human-readable status description"""
        descriptions = {
            'KOMİSYONDA': 'In Commission',
            'GENEL_KURULDA': 'In Plenary',
            'KABUL_EDİLDİ': 'Adopted',
            'REDDEDİLDİ': 'Rejected',
            'GERİ_ÇEKİLDİ': 'Withdrawn',
            'YASALASTI': 'Enacted into Law'
        }
        return descriptions.get(status, 'Unknown')

    def _extract_commission(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract commission information

        Args:
            text: Text containing commission info

        Returns:
            Dict with commission info or None
        """
        text_lower = text.lower()

        for commission, keywords in self.COMMISSIONS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'commission': commission,
                        'keyword': keyword
                    }

        return None

    def _extract_voting_results(self, soup: BeautifulSoup, full_text: str) -> Optional[Dict[str, Any]]:
        """Extract voting results

        Args:
            soup: BeautifulSoup object
            full_text: Full text content

        Returns:
            Dict with voting results or None
        """
        voting_data = {}

        # Try to find voting section
        voting_elem = self._find_element(soup, 'voting_section')
        voting_text = voting_elem.get_text() if voting_elem else full_text

        # Extract vote counts
        # Pattern: "Kabul: 250, Red: 100, Çekimser: 10"
        kabul_pattern = r'(?:kabul|evet|yes)\s*:?\s*(\d+)'
        red_pattern = r'(?:red|hayır|no)\s*:?\s*(\d+)'
        cekimser_pattern = r'(?:çekimser|abstain)\s*:?\s*(\d+)'

        kabul_match = re.search(kabul_pattern, voting_text, re.IGNORECASE)
        if kabul_match:
            voting_data['evet'] = int(kabul_match.group(1))

        red_match = re.search(red_pattern, voting_text, re.IGNORECASE)
        if red_match:
            voting_data['hayir'] = int(red_match.group(1))

        cekimser_match = re.search(cekimser_pattern, voting_text, re.IGNORECASE)
        if cekimser_match:
            voting_data['cekimser'] = int(cekimser_match.group(1))

        # Calculate totals
        if voting_data:
            voting_data['toplam'] = sum(voting_data.values())

            # Determine result
            evet = voting_data.get('evet', 0)
            hayir = voting_data.get('hayir', 0)
            if evet > hayir:
                voting_data['sonuc'] = 'KABUL'
            elif hayir > evet:
                voting_data['sonuc'] = 'RED'
            else:
                voting_data['sonuc'] = 'BERABERLİK'

            return voting_data

        return None

    def _extract_rapporteur(self, text: str) -> Optional[str]:
        """Extract rapporteur (sözcü) name

        Args:
            text: Text containing rapporteur info

        Returns:
            Rapporteur name or None
        """
        # Pattern: "Sözcü: [Name]" or "Raportör: [Name]"
        rapporteur_pattern = r'(?:sözcü|raportör)\s*:?\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)'
        match = re.search(rapporteur_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_proposers(self, text: str) -> List[str]:
        """Extract bill proposers (Milletvekili or Hükümet)

        Args:
            text: Text containing proposer info

        Returns:
            List of proposer names
        """
        proposers = []

        # Pattern: "Milletvekili [Name]"
        mv_pattern = r'Milletvekili\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)'
        matches = re.finditer(mv_pattern, text)
        for match in matches:
            proposers.append(match.group(1))

        # Check for government proposal
        if 'hükümet' in text.lower() or 'bakanlar kurulu' in text.lower():
            proposers.append('Hükümet')

        return proposers[:10]  # Limit to top 10

    def _extract_submission_date(self, text: str) -> Optional[str]:
        """Extract submission date (Verildiği Tarih)

        Args:
            text: Text containing date

        Returns:
            Date string or None
        """
        # Pattern: "dd.mm.yyyy" or "dd/mm/yyyy"
        date_pattern = r'(\d{2}[./]\d{2}[./]\d{4})'
        match = re.search(date_pattern, text)
        if match:
            return match.group(1)

        return None

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches TBMM documents (Kanun Teklifi, Komisyon Raporu, Tutanak)

        Args:
            document_id: Document identifier (format: "dönem/yasama_yılı/esas_no" or direct ID)
            **kwargs: Additional options

        Returns:
            Dict with document data and legislative metadata

        Raises:
            NetworkError: On fetch failures
            ParsingError: On parsing failures
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            # TBMM document ID format: "dönem/yasama_yılı/esas_no"
            url = f"{self.base_url}/sirasayi/{document_id}"
            logger.info(f"Fetching TBMM document: {document_id}")

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

            # Extract info table
            info_table = self._find_element(soup, 'info_table')

            # Extract legislative numbers
            legislative_numbers = self._extract_legislative_numbers(info_table)

            # Classify document type
            doc_type_info = self._classify_document_type(title)
            doc_type = doc_type_info['name'] if doc_type_info else None
            doc_type_code = doc_type_info['code'] if doc_type_info else None

            # Extract bill status
            combined_text = f"{title} {content[:2000]}"
            bill_status = self._extract_bill_status(combined_text)

            # Extract commission
            commission = self._extract_commission(combined_text)

            # Extract voting results
            voting_results = self._extract_voting_results(soup, content)

            # Extract rapporteur
            rapporteur = self._extract_rapporteur(combined_text)

            # Extract proposers
            proposers = self._extract_proposers(combined_text)

            # Extract submission date
            submission_date = self._extract_submission_date(combined_text)

            # Extract commission info element
            commission_elem = self._find_element(soup, 'commission_info')
            commission_text = commission_elem.get_text(strip=True) if commission_elem else ''

            result = {
                'title': title,
                'content': content,
                'url': url,
                'document_id': document_id,
                'source': 'TBMM',
                # Document classification
                'document_type': doc_type,
                'document_type_code': doc_type_code,
                # Legislative numbers
                'esas_no': legislative_numbers['esas_no'],
                'karar_no': legislative_numbers['karar_no'],
                'sira_sayisi': legislative_numbers['sira_sayisi'],
                'donem': legislative_numbers['donem'],
                'yasama_yili': legislative_numbers['yasama_yili'],
                # Bill tracking
                'bill_status': bill_status,
                'commission': commission,
                'commission_details': commission_text if commission_text else None,
                # Voting
                'voting_results': voting_results,
                # People
                'rapporteur': rapporteur,
                'proposers': proposers,
                'proposers_count': len(proposers),
                # Dates
                'submission_date': submission_date
            }

            logger.info(f"Successfully parsed TBMM document {document_id}: "
                       f"type={doc_type}, status={bill_status['status'] if bill_status else 'N/A'}")

            return result

        except DocumentNotFoundError:
            raise
        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {str(e)}") from e

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search TBMM documents by keyword with pagination and filtering

        Args:
            query: Search query
            **kwargs: Options (page, limit, donem, document_type)

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
            search_url = f"{self.base_url}/arama?q={quote(query_normalized)}&page={page}"

            # Add filters
            donem_filter = kwargs.get('donem')
            if donem_filter:
                search_url += f"&donem={donem_filter}"

            doc_type_filter = kwargs.get('document_type')
            if doc_type_filter:
                search_url += f"&type={doc_type_filter}"

            logger.info(f"Searching TBMM: query='{query}', page={page}, limit={limit}")

            # Fetch search results
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Find result items (multiple selectors)
            result_selectors = [
                'div.sonuc',
                'div.search-result',
                'div.result-item',
                'div.arama-sonuc'
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
                    doc_id_match = re.search(r'/sirasayi/([^/]+)', doc_url)
                    doc_id = doc_id_match.group(1) if doc_id_match else None

                    # Extract excerpt/summary
                    excerpt_elem = item.find(['p', 'div'], class_=re.compile(r'excerpt|ozet|summary'))
                    excerpt = excerpt_elem.text.strip() if excerpt_elem else ''

                    # Classify document type
                    doc_type_info = self._classify_document_type(title)

                    # Extract dönem from title or excerpt
                    donem = None
                    donem_match = re.search(r'(\d+)\.\s*(?:Dönem|dönem)', f"{title} {excerpt}")
                    if donem_match:
                        donem = donem_match.group(1)

                    result_item = {
                        'title': title,
                        'url': doc_url,
                        'document_id': doc_id,
                        'excerpt': excerpt,
                        'document_type': doc_type_info['name'] if doc_type_info else None,
                        'donem': donem,
                        'source': 'TBMM'
                    }

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to parse search result item: {e}")
                    continue

            logger.info(f"Found {len(results)} TBMM search results for query '{query}'")
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

        # Determine document type from title and document_type field
        title_lower = raw_data.get('title', '').lower()
        doc_type_field = raw_data.get('document_type', '')

        if 'kanun teklif' in title_lower or doc_type_field == 'KANUN_TEKLİFİ':
            doc_type = DocumentType.KANUN_TASARISI
        elif 'kanun tasarı' in title_lower or doc_type_field == 'KANUN_TASARISI':
            doc_type = DocumentType.KANUN_TASARISI
        elif 'komisyon rapor' in title_lower or doc_type_field == 'KOMİSYON_RAPORU':
            doc_type = DocumentType.KOMISYON_RAPORU
        elif 'tutanak' in title_lower or doc_type_field == 'TUTANAK':
            doc_type = DocumentType.TUTANAK
        elif 'karar' in title_lower or doc_type_field == 'MECLİS_KARARI':
            doc_type = DocumentType.MECLIS_KARARI
        else:
            doc_type = DocumentType.KANUN_TASARISI

        # Determine effectivity status based on bill status
        bill_status = raw_data.get('bill_status', {})
        if bill_status:
            status_code = bill_status.get('status')
            if status_code == 'YASALASTI':
                effectivity = EffectivityStatus.YURURLUKTE
            elif status_code == 'KABUL_EDİLDİ':
                effectivity = EffectivityStatus.YURURLUKTE
            elif status_code == 'REDDEDİLDİ':
                effectivity = EffectivityStatus.YURURLUKTEN_KALDIRILDI
            else:
                effectivity = EffectivityStatus.TASLAK
        else:
            effectivity = EffectivityStatus.TASLAK

        # Build metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.YASAMA,
            hierarchy_level=LegalHierarchy.KANUN,  # Legislative level
            source=SourceType.TBMM,
            source_url=raw_data.get('url'),
            effectivity_status=effectivity,
            # Legislative metadata
            esas_no=raw_data.get('esas_no'),
            karar_no=raw_data.get('karar_no'),
            sira_sayisi=raw_data.get('sira_sayisi'),
            donem=raw_data.get('donem'),
            yasama_yili=raw_data.get('yasama_yili'),
            # Bill tracking
            bill_status=bill_status.get('status') if bill_status else None,
            commission=raw_data.get('commission', {}).get('commission') if raw_data.get('commission') else None,
            voting_results=raw_data.get('voting_results'),
            rapporteur=raw_data.get('rapporteur'),
            proposers=raw_data.get('proposers')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['TBMMAdapter']
