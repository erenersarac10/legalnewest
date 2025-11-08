"""Sayıştay (Court of Accounts) Adapter - Harvey/Legora CTO-Level Production-Grade
Fetches audit reports and rulings from sayistay.gov.tr

Production Features:
- Rate limiting: 35 requests/minute sliding window
- TTL caching: 1-hour default with cache key management
- Retry logic: Exponential backoff (3 attempts, 2.0x factor)
- Multiple fallback CSS selectors for robust parsing
- Comprehensive error handling and logging
- Audit-specific extraction: findings, compliance, budget discrepancies
- Turkish audit terminology support
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


class SayistayAdapter(SourceAdapter):
    """Court of Accounts (Sayıştay) Adapter

    Handles audit reports, compliance audits, performance audits, and financial audits
    from the Turkish Court of Accounts.
    """

    # Multiple fallback CSS selectors for robust parsing
    SELECTORS = {
        'title': [
            'h1.rapor-baslik',
            'h1.document-title',
            'h1.page-title',
            'div.rapor-header h1',
            'h1',
            'div.title h1'
        ],
        'content': [
            'div.rapor-icerik',
            'div.rapor-content',
            'div.content',
            'div.main-content',
            'article.rapor',
            'div.document-body'
        ],
        'report_info': [
            'div.rapor-bilgi',
            'div.report-info',
            'div.meta-info',
            'div.document-meta',
            'div.rapor-metadata'
        ],
        'findings': [
            'div.bulgular',
            'div.tespit',
            'div.findings',
            'div.audit-findings',
            'section.bulgular'
        ],
        'recommendations': [
            'div.oneriler',
            'div.tavsiyeler',
            'div.recommendations',
            'section.oneriler'
        ]
    }

    # Audit report types with Turkish/English keywords
    REPORT_TYPES = {
        'MALİ_DENETİM': {
            'keywords': ['mali denetim', 'mali rapor', 'financial audit', 'mali kontrol'],
            'type_code': 'MD',
            'description': 'Financial Audit Report'
        },
        'UYGUNLUK_DENETİMİ': {
            'keywords': ['uygunluk denetim', 'compliance audit', 'uygunluk kontrol'],
            'type_code': 'UD',
            'description': 'Compliance Audit Report'
        },
        'PERFORMANS_DENETİMİ': {
            'keywords': ['performans denetim', 'performance audit', 'etkinlik denetim'],
            'type_code': 'PD',
            'description': 'Performance Audit Report'
        },
        'DÜZENLİLİK_DENETİMİ': {
            'keywords': ['düzenlilik denetim', 'regularity audit'],
            'type_code': 'DD',
            'description': 'Regularity Audit Report'
        }
    }

    # Compliance status indicators
    COMPLIANCE_LEVELS = {
        'UYGUN': ['uygun', 'compliant', 'uygunluk sağlan', 'düzeltilmiş'],
        'UYGUNSUZ': ['uygunsuz', 'non-compliant', 'mevzuata aykırı', 'hukuka aykırı'],
        'KISMİ_UYGUN': ['kısmen uygun', 'partially compliant', 'kısmi uygunluk'],
        'SÜREKLİ_UYGUNSUZLUK': ['sürekli uygunsuzluk', 'tekrar eden', 'ongoing non-compliance']
    }

    # Entity types that can be audited
    ENTITY_TYPES = {
        'BAKANLIK': ['bakanlığ', 'ministry', 'nezaret'],
        'KAMU_İDARESİ': ['genel müdürlük', 'başkanlık', 'kurum', 'kurul'],
        'BELEDİYE': ['belediye', 'municipality', 'şehir', 'büyükşehir'],
        'ÜNİVERSİTE': ['üniversite', 'university', 'yüksekokul'],
        'KAMU_İKTİSADİ': ['kamu iktisadi teşebbüs', 'kit', 'public economic enterprise'],
        'DÖNER_SERMAYE': ['döner sermaye', 'revolving fund']
    }

    def __init__(self):
        super().__init__("Sayıştay Adapter", "2.0.0")
        self.base_url = "https://www.sayistay.gov.tr"

        # Rate limiting: 35 requests per minute (sliding window)
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

    def _classify_report_type(self, text: str) -> Optional[Dict[str, Any]]:
        """Classify audit report type based on keywords

        Args:
            text: Text to analyze (usually title)

        Returns:
            Dict with report type info or None
        """
        text_lower = text.lower()

        for report_name, report_info in self.REPORT_TYPES.items():
            for keyword in report_info['keywords']:
                if keyword in text_lower:
                    logger.debug(f"Classified report type: {report_name}")
                    return {
                        'name': report_name,
                        'code': report_info['type_code'],
                        'description': report_info['description']
                    }

        return None

    def _extract_report_number(self, text: str) -> Optional[str]:
        """Extract Sayıştay report number

        Formats:
        - YYYY/NN (e.g., 2023/15)
        - Rapor No: YYYY-NN
        - No: NNNN

        Args:
            text: Text containing report number

        Returns:
            Report number string or None
        """
        # Pattern 1: YYYY/NN
        pattern1 = r'(\d{4})/(\d{1,3})'
        match = re.search(pattern1, text)
        if match:
            return match.group(0)

        # Pattern 2: Rapor No: YYYY-NN
        pattern2 = r'Rapor\s+No[:\s]+(\d{4}-\d{1,3})'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: No: NNNN
        pattern3 = r'No[:\s]+(\d{3,5})'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_audit_year(self, text: str) -> Optional[str]:
        """Extract audit year(s) from text

        Args:
            text: Text containing year

        Returns:
            Year string or year range (e.g., "2023" or "2022-2023")
        """
        # Year range pattern: 2022-2023
        range_pattern = r'(\d{4})\s*-\s*(\d{4})'
        match = re.search(range_pattern, text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        # Single year pattern
        year_pattern = r'(\d{4})'
        match = re.search(year_pattern, text)
        if match:
            year = int(match.group(1))
            # Validate year range (Sayıştay established in 1862)
            if 1862 <= year <= datetime.now().year + 1:
                return match.group(1)

        return None

    def _extract_audit_period(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract audit period information

        Args:
            text: Text containing period info

        Returns:
            Dict with start/end dates or None
        """
        period_info = {}

        # Pattern: "01.01.2023 - 31.12.2023"
        date_range_pattern = r'(\d{2}\.\d{2}\.\d{4})\s*-\s*(\d{2}\.\d{2}\.\d{4})'
        match = re.search(date_range_pattern, text)
        if match:
            period_info['start_date'] = match.group(1)
            period_info['end_date'] = match.group(2)
            return period_info

        # Pattern: "2023 yılı" or "2022-2023 dönemi"
        year_pattern = r'(\d{4})\s*(?:yılı|mali yılı|hesap dönemi)'
        match = re.search(year_pattern, text)
        if match:
            year = match.group(1)
            period_info['year'] = year
            period_info['description'] = f"{year} Mali Yılı"
            return period_info

        return None if not period_info else period_info

    def _extract_entity_audited(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract information about the entity being audited

        Args:
            text: Text containing entity information

        Returns:
            Dict with entity info or None
        """
        entity_info = {}

        # Classify entity type
        text_lower = text.lower()
        for entity_type, keywords in self.ENTITY_TYPES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entity_info['type'] = entity_type
                    break
            if 'type' in entity_info:
                break

        # Extract entity name (usually before "Denetim Raporu")
        name_pattern = r'(.+?)\s+(?:Denetim Raporu|Mali Raporu|Performans Raporu)'
        match = re.search(name_pattern, text, re.IGNORECASE)
        if match:
            entity_name = match.group(1).strip()
            # Clean up common prefixes
            entity_name = re.sub(r'^(T\.C\.|TC)\s+', '', entity_name)
            entity_info['name'] = entity_name

        return entity_info if entity_info else None

    def _extract_audit_findings(self, text: str) -> List[Dict[str, Any]]:
        """Extract audit findings and issues

        Args:
            text: Text containing findings

        Returns:
            List of finding dicts
        """
        findings = []

        # Common finding indicators
        finding_patterns = [
            r'tespit\s+edilmiştir?\s*:?\s*([^.]+)',
            r'bulgu\s*:?\s*([^.]+)',
            r'uygunsuzluk\s+tespit\s+edilmiş',
            r'mevzuata\s+aykırı',
            r'usulsüzlük\s+tespit'
        ]

        for pattern in finding_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                finding_text = match.group(0)
                # Extract surrounding context (up to next period)
                start_pos = match.start()
                end_pos = text.find('.', start_pos)
                if end_pos == -1:
                    end_pos = min(start_pos + 200, len(text))

                context = text[start_pos:end_pos].strip()

                findings.append({
                    'text': context,
                    'type': 'finding',
                    'severity': self._assess_severity(context)
                })

        return findings

    def _assess_severity(self, finding_text: str) -> str:
        """Assess severity of a finding

        Args:
            finding_text: Finding text

        Returns:
            Severity level: 'HIGH', 'MEDIUM', 'LOW'
        """
        text_lower = finding_text.lower()

        # High severity indicators
        high_severity = ['ağır', 'önemli', 'ciddi', 'yüksek', 'kritik', 'büyük']
        if any(keyword in text_lower for keyword in high_severity):
            return 'HIGH'

        # Low severity indicators
        low_severity = ['hafif', 'küçük', 'minor', 'düşük']
        if any(keyword in text_lower for keyword in low_severity):
            return 'LOW'

        return 'MEDIUM'

    def _extract_budget_discrepancies(self, text: str) -> List[Dict[str, Any]]:
        """Extract budget and financial discrepancies

        Args:
            text: Text containing financial information

        Returns:
            List of discrepancy dicts
        """
        discrepancies = []

        # Pattern: Amount in TL (various formats)
        # 1.234.567,89 TL or 1.234.567 TL
        amount_pattern = r'([\d.,]+)\s*(?:TL|₺|Türk Lirası)'

        # Discrepancy indicators
        discrepancy_patterns = [
            (r'fazla\s+(?:ödeme|harcama|tahakkuk)\s*:?\s*' + amount_pattern, 'OVERPAYMENT'),
            (r'eksik\s+(?:ödeme|tahakkuk|kayıt)\s*:?\s*' + amount_pattern, 'UNDERPAYMENT'),
            (r'kayıt\s+dışı\s*:?\s*' + amount_pattern, 'UNRECORDED'),
            (r'yetkisiz\s+(?:harcama|ödeme)\s*:?\s*' + amount_pattern, 'UNAUTHORIZED'),
            (r'usulsüz\s+(?:harcama|ödeme)\s*:?\s*' + amount_pattern, 'IRREGULAR')
        ]

        for pattern, discrepancy_type in discrepancy_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1)
                # Parse Turkish number format
                try:
                    # Remove thousand separators (.) and replace decimal comma
                    amount_clean = amount_str.replace('.', '').replace(',', '.')
                    amount = Decimal(amount_clean)

                    discrepancies.append({
                        'type': discrepancy_type,
                        'amount': float(amount),
                        'amount_formatted': f"{amount_str} TL",
                        'context': match.group(0)
                    })
                except (InvalidOperation, ValueError) as e:
                    logger.warning(f"Failed to parse amount: {amount_str} - {e}")
                    continue

        return discrepancies

    def _extract_compliance_status(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract compliance status from audit text

        Args:
            text: Audit report text

        Returns:
            Dict with compliance info or None
        """
        text_lower = text.lower()

        for status, keywords in self.COMPLIANCE_LEVELS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)

                    return {
                        'status': status,
                        'keyword': keyword,
                        'occurrences': count,
                        'description': self._get_compliance_description(status)
                    }

        return None

    def _get_compliance_description(self, status: str) -> str:
        """Get human-readable compliance description"""
        descriptions = {
            'UYGUN': 'Mevzuata uygun / Compliant',
            'UYGUNSUZ': 'Mevzuata aykırı / Non-compliant',
            'KISMİ_UYGUN': 'Kısmen uygun / Partially compliant',
            'SÜREKLİ_UYGUNSUZLUK': 'Sürekli uygunsuzluk / Ongoing non-compliance'
        }
        return descriptions.get(status, 'Belirsiz / Unknown')

    def _extract_recommendations(self, soup: BeautifulSoup, full_text: str) -> List[str]:
        """Extract audit recommendations

        Args:
            soup: BeautifulSoup object
            full_text: Full text content

        Returns:
            List of recommendations
        """
        recommendations = []

        # Try to find recommendations section
        rec_elem = self._find_element(soup, 'recommendations')
        if rec_elem:
            # Extract list items or paragraphs
            items = rec_elem.find_all(['li', 'p'])
            for item in items:
                text = item.get_text(strip=True)
                if len(text) > 20:  # Filter out too short items
                    recommendations.append(text)

        # Fallback: Pattern matching in full text
        if not recommendations:
            rec_pattern = r'(?:öneri|tavsiye|recommendation)\s*:?\s*([^.]+\.)'
            matches = re.finditer(rec_pattern, full_text, re.IGNORECASE)
            for match in matches:
                recommendations.append(match.group(0))

        return recommendations[:10]  # Limit to top 10

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches Sayıştay documents (Denetim Raporu, Karar, Görüş)

        Args:
            document_id: Document identifier
            **kwargs: Additional options

        Returns:
            Dict with document data and audit metadata

        Raises:
            NetworkError: On fetch failures
            ParsingError: On parsing failures
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            url = f"{self.base_url}/Rapor/{document_id}"
            logger.info(f"Fetching Sayıştay document: {document_id}")

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

            # Extract report metadata
            report_info_elem = self._find_element(soup, 'report_info')
            report_info_text = report_info_elem.get_text(strip=True) if report_info_elem else ''

            # Combine title and report info for comprehensive extraction
            combined_text = f"{title} {report_info_text}"

            # Classify report type
            report_type_info = self._classify_report_type(title)
            report_type = report_type_info['name'] if report_type_info else None
            report_type_code = report_type_info['code'] if report_type_info else None

            # Extract audit year
            audit_year = self._extract_audit_year(combined_text)

            # Extract report number
            report_number = self._extract_report_number(combined_text)

            # Extract audit period
            audit_period = self._extract_audit_period(combined_text)

            # Extract entity being audited
            entity_audited = self._extract_entity_audited(title)

            # Extract audit findings
            findings_elem = self._find_element(soup, 'findings')
            findings_text = findings_elem.get_text(strip=True) if findings_elem else content[:5000]
            findings = self._extract_audit_findings(findings_text)

            # Extract budget discrepancies
            discrepancies = self._extract_budget_discrepancies(content)

            # Extract compliance status
            compliance_status = self._extract_compliance_status(content)

            # Extract recommendations
            recommendations = self._extract_recommendations(soup, content)

            result = {
                'title': title,
                'content': content,
                'url': url,
                'document_id': document_id,
                'source': 'Sayıştay',
                # Report metadata
                'report_type': report_type,
                'report_type_code': report_type_code,
                'report_number': report_number,
                'audit_year': audit_year,
                'audit_period': audit_period,
                # Audit details
                'entity_audited': entity_audited,
                'findings': findings,
                'findings_count': len(findings),
                'budget_discrepancies': discrepancies,
                'total_discrepancy_amount': sum(d.get('amount', 0) for d in discrepancies),
                'compliance_status': compliance_status,
                'recommendations': recommendations,
                'recommendations_count': len(recommendations)
            }

            logger.info(f"Successfully parsed Sayıştay document {document_id}: "
                       f"{len(findings)} findings, {len(discrepancies)} discrepancies")

            return result

        except DocumentNotFoundError:
            raise
        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching document {document_id}: {e}")
            raise ParsingError(f"Failed to parse document {document_id}: {str(e)}") from e

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search Sayıştay documents by keyword with pagination and filtering

        Args:
            query: Search query
            **kwargs: Options (page, limit, report_type, year)

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
            report_type_filter = kwargs.get('report_type')
            if report_type_filter:
                search_url += f"&type={report_type_filter}"

            year_filter = kwargs.get('year')
            if year_filter:
                search_url += f"&year={year_filter}"

            logger.info(f"Searching Sayıştay: query='{query}', page={page}, limit={limit}")

            # Fetch search results
            html = await self._fetch_html(search_url, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')

            results = []

            # Find result items (multiple selectors)
            result_selectors = [
                'div.rapor-item',
                'div.search-result',
                'div.result-item',
                'article.rapor'
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
                    doc_id_match = re.search(r'/Rapor/(\d+)', doc_url)
                    doc_id = doc_id_match.group(1) if doc_id_match else None

                    # Extract summary/excerpt
                    summary_elem = item.find(['div', 'p'], class_=re.compile(r'ozet|summary|excerpt'))
                    summary = summary_elem.text.strip() if summary_elem else ''

                    # Extract year from title
                    year = self._extract_audit_year(title)

                    # Classify report type
                    report_type_info = self._classify_report_type(title)

                    result_item = {
                        'title': title,
                        'url': doc_url,
                        'document_id': doc_id,
                        'summary': summary,
                        'year': year,
                        'report_type': report_type_info['name'] if report_type_info else None,
                        'source': 'Sayıştay'
                    }

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to parse search result item: {e}")
                    continue

            logger.info(f"Found {len(results)} Sayıştay search results for query '{query}'")
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

        # Determine document type from title and report_type
        title_lower = raw_data.get('title', '').lower()
        report_type = raw_data.get('report_type', '')

        if 'denetim rapor' in title_lower or report_type:
            doc_type = DocumentType.DENETIM_RAPORU
        elif 'karar' in title_lower:
            doc_type = DocumentType.KURUL_KARARI
        elif 'görüş' in title_lower:
            doc_type = DocumentType.GORUS
        else:
            doc_type = DocumentType.DENETIM_RAPORU

        # Build metadata
        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_KARAR,
            source=SourceType.SAYISTAY,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            # Audit-specific metadata
            audit_year=raw_data.get('audit_year'),
            report_type=raw_data.get('report_type'),
            report_number=raw_data.get('report_number'),
            entity_audited=raw_data.get('entity_audited'),
            findings_count=raw_data.get('findings_count'),
            compliance_status=raw_data.get('compliance_status')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )


__all__ = ['SayistayAdapter']
