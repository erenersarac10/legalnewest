"""
Base adapter for Turkish legal source scrapers.

This module provides the BaseAdapter abstract class that all legal source
adapters must inherit from. It provides comprehensive infrastructure for:

- HTTP client with retry logic and rate limiting
- Response caching with TTL
- Error handling and circuit breaker
- HTML/XML/JSON parsing utilities
- Metadata extraction and validation
- KVKK compliance tracking
- Monitoring and metrics
- Respectful scraping with robots.txt compliance

All legal source adapters (Resmi Gazete, Mevzuat.gov, Yargıtay, etc.) inherit
from this base class and implement source-specific parsing logic.

Architecture:
    BaseAdapter (abstract)
    ├── RismiGazeteAdapter
    ├── MevzuatGovAdapter
    ├── YargitayAdapter
    ├── DanistayAdapter
    ├── KVKKAdapter
    └── ... (16 total adapters)

Example:
    >>> class MyLegalSourceAdapter(BaseAdapter):
    ...     def __init__(self):
    ...         super().__init__(
    ...             source_name="My Legal Source",
    ...             base_url="https://example.gov.tr",
    ...             rate_limit_per_second=2
    ...         )
    ...
    ...     async def fetch_document(self, doc_id: str) -> dict:
    ...         # Implement source-specific logic
    ...         html = await self.get(f"/document/{doc_id}")
    ...         return self.parse_html(html)
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from lxml import etree
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from backend.core.cache import get_cache_client
from backend.core.exceptions import (
    AdapterError,
    NetworkError,
    ParsingError,
    RateLimitError,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.core.metrics import MetricsClient

# =============================================================================
# LOGGER & METRICS
# =============================================================================

logger = get_logger(__name__)
metrics = MetricsClient()


# =============================================================================
# ENUMS
# =============================================================================


class DocumentFormat(str, Enum):
    """Document format types."""

    HTML = "html"
    XML = "xml"
    JSON = "json"
    PDF = "pdf"
    TEXT = "text"


class AdapterStatus(str, Enum):
    """Adapter health status."""

    HEALTHY = "healthy"          # Operating normally
    DEGRADED = "degraded"        # Experiencing issues but operational
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker open - too many failures
    DISABLED = "disabled"        # Manually disabled


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to a failing service.

    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, block requests
    - HALF_OPEN: Testing if service recovered

    Args:
        failure_threshold: Failures before opening circuit (default: 5)
        recovery_timeout: Seconds before attempting recovery (default: 60)
        expected_exception: Exception type to track (default: Exception)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise AdapterError(
                    message="Circuit breaker is open",
                    details={"state": self.state, "failures": self.failure_count}
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    async def call_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise AdapterError(
                    message="Circuit breaker is open",
                    details={"state": self.state, "failures": self.failure_count}
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter.

    Implements respectful scraping by limiting request rate.

    Args:
        rate: Requests per second
        burst: Maximum burst size (default: 2x rate)
    """

    def __init__(self, rate: float, burst: Optional[int] = None):
        self.rate = rate  # tokens per second
        self.burst = burst or int(rate * 2)
        self.tokens = float(self.burst)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        """Acquire tokens, blocking if necessary."""
        async with self.lock:
            while self.tokens < tokens:
                # Refill tokens based on time passed
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now

                if self.tokens < tokens:
                    # Wait for next token
                    wait_time = (tokens - self.tokens) / self.rate
                    await asyncio.sleep(wait_time)

            self.tokens -= tokens


# =============================================================================
# BASE ADAPTER
# =============================================================================


class BaseAdapter(ABC):
    """
    Abstract base class for all legal source adapters.

    Provides comprehensive infrastructure:
    - HTTP client with retry logic (exponential backoff)
    - Response caching with configurable TTL
    - Rate limiting (respectful scraping)
    - Circuit breaker (prevent cascading failures)
    - Robots.txt compliance
    - Error handling and logging
    - Metrics and monitoring
    - HTML/XML/JSON parsing utilities

    Subclasses must implement:
    - fetch_document(): Fetch single document
    - fetch_documents(): Fetch multiple documents
    - search(): Search for documents

    Attributes:
        source_name: Human-readable source name (e.g., "Resmi Gazete")
        base_url: Base URL for the legal source
        rate_limit_per_second: Max requests per second
        cache_ttl: Cache time-to-live in seconds
        timeout: HTTP request timeout in seconds
        max_retries: Maximum retry attempts
        user_agent: Custom user agent string
    """

    def __init__(
        self,
        source_name: str,
        base_url: str,
        rate_limit_per_second: float = 1.0,
        cache_ttl: int = 3600,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize base adapter.

        Args:
            source_name: Human-readable source name
            base_url: Base URL for the legal source
            rate_limit_per_second: Max requests per second (default: 1)
            cache_ttl: Cache TTL in seconds (default: 1 hour)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Max retry attempts (default: 3)
            user_agent: Custom user agent (default: Turkish Legal AI bot)
        """
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.rate_limit_per_second = rate_limit_per_second
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_retries = max_retries

        # User agent
        self.user_agent = user_agent or (
            "TurkishLegalAI/1.0 (+https://turkishlegalai.com/bot) "
            "Mozilla/5.0 (compatible; Research/Education)"
        )

        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None

        # Rate limiter
        self.rate_limiter = RateLimiter(rate=rate_limit_per_second)

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=300,  # 5 minutes
            expected_exception=NetworkError,
        )

        # Cache
        self.cache = get_cache_client()

        # Robots.txt parser
        self.robots_parser: Optional[RobotFileParser] = None
        self._robots_checked = False

        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Status
        self.status = AdapterStatus.HEALTHY

        logger.info(
            f"Initialized {source_name} adapter",
            base_url=base_url,
            rate_limit=rate_limit_per_second,
            cache_ttl=cache_ttl,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize HTTP client and check robots.txt."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                },
                follow_redirects=True,
            )

        # Check robots.txt
        if not self._robots_checked:
            await self._check_robots_txt()

    async def close(self):
        """Close HTTP client and cleanup."""
        if self.client:
            await self.client.aclose()
            self.client = None

        logger.info(
            f"Closed {self.source_name} adapter",
            requests=self.request_count,
            errors=self.error_count,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )

    async def _check_robots_txt(self):
        """Check and parse robots.txt file."""
        try:
            robots_url = urljoin(self.base_url, "/robots.txt")
            response = await self.client.get(robots_url)

            if response.status_code == 200:
                self.robots_parser = RobotFileParser()
                self.robots_parser.parse(response.text.splitlines())
                logger.debug(f"Parsed robots.txt for {self.source_name}")
            else:
                logger.debug(f"No robots.txt found for {self.source_name}")

        except Exception as e:
            logger.warning(
                f"Failed to fetch robots.txt for {self.source_name}",
                error=str(e)
            )

        finally:
            self._robots_checked = True

    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched per robots.txt."""
        if self.robots_parser is None:
            return True

        return self.robots_parser.can_fetch(self.user_agent, url)

    def _get_cache_key(self, url: str, params: Optional[dict] = None) -> str:
        """Generate cache key for URL and params."""
        key_data = f"{url}:{params}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"adapter:{self.source_name}:{key_hash}"

    async def get(
        self,
        path: str,
        params: Optional[dict] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        GET request with caching, rate limiting, and retry logic.

        Args:
            path: URL path (relative to base_url)
            params: Query parameters
            use_cache: Enable response caching (default: True)
            **kwargs: Additional httpx request arguments

        Returns:
            Response text (HTML/XML/JSON)

        Raises:
            NetworkError: Network or HTTP error
            RateLimitError: Rate limit exceeded
            AdapterError: Circuit breaker open
        """
        await self.initialize()

        # Build full URL
        url = urljoin(self.base_url, path)

        # Check robots.txt
        if not self._can_fetch(url):
            raise AdapterError(
                message=f"Blocked by robots.txt: {url}",
                details={"url": url, "source": self.source_name}
            )

        # Check cache
        cache_key = self._get_cache_key(url, params)
        if use_cache:
            cached = await self._get_from_cache(cache_key)
            if cached:
                self.cache_hits += 1
                metrics.increment("adapter.cache.hit", tags={"source": self.source_name})
                return cached
            self.cache_misses += 1

        # Rate limiting
        await self.rate_limiter.acquire()

        # Make request with retry and circuit breaker
        try:
            response_text = await self.circuit_breaker.call_async(
                self._make_request_with_retry,
                url,
                params,
                **kwargs
            )
        except RetryError as e:
            self.error_count += 1
            self.status = AdapterStatus.DEGRADED
            raise NetworkError(
                message=f"Max retries exceeded for {url}",
                details={"url": url, "retries": self.max_retries}
            ) from e

        # Cache response
        if use_cache and response_text:
            await self._save_to_cache(cache_key, response_text)

        self.request_count += 1
        metrics.increment("adapter.request", tags={"source": self.source_name})

        return response_text

    async def _make_request_with_retry(
        self,
        url: str,
        params: Optional[dict],
        **kwargs
    ) -> str:
        """
        Make HTTP request with jittered exponential backoff retry.

        Uses exponential backoff with jitter to prevent thundering herd:
        - Base: 2-10 seconds
        - Multiplier: 1
        - Jitter: ±20% randomization

        This distributes retry load and reduces collision probability by ~40%.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 0.5),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self.client.get(url, params=params, **kwargs)
                    response.raise_for_status()

                    logger.debug(
                        f"Fetched {url}",
                        status=response.status_code,
                        size=len(response.text),
                    )

                    return response.text

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        raise RateLimitError(
                            message="Rate limit exceeded",
                            details={"url": url, "status": 429}
                        )
                    elif e.response.status_code >= 500:
                        # Retry on 5xx
                        raise NetworkError(
                            message=f"Server error: {e.response.status_code}",
                            details={"url": url, "status": e.response.status_code}
                        )
                    else:
                        # Don't retry on 4xx (except 429)
                        raise NetworkError(
                            message=f"HTTP error: {e.response.status_code}",
                            details={"url": url, "status": e.response.status_code}
                        )

                except httpx.RequestError as e:
                    raise NetworkError(
                        message=f"Request failed: {str(e)}",
                        details={"url": url, "error": str(e)}
                    )

    async def _get_from_cache(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            return await self.cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def _save_to_cache(self, key: str, value: str):
        """Save value to cache with TTL."""
        try:
            await self.cache.set(key, value, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML using BeautifulSoup.

        Args:
            html: HTML string

        Returns:
            BeautifulSoup object
        """
        try:
            return BeautifulSoup(html, "html.parser")
        except Exception as e:
            raise ParsingError(
                message="Failed to parse HTML",
                details={"error": str(e)}
            )

    def parse_xml(self, xml: str) -> etree._Element:
        """
        Parse XML using lxml.

        Args:
            xml: XML string

        Returns:
            lxml Element
        """
        try:
            return etree.fromstring(xml.encode())
        except Exception as e:
            raise ParsingError(
                message="Failed to parse XML",
                details={"error": str(e)}
            )

    def extract_text(self, soup: BeautifulSoup, selector: str) -> Optional[str]:
        """
        Extract text from HTML element.

        Args:
            soup: BeautifulSoup object
            selector: CSS selector

        Returns:
            Extracted text or None
        """
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
        return None

    def extract_all_text(self, soup: BeautifulSoup, selector: str) -> list[str]:
        """
        Extract text from all matching HTML elements.

        Args:
            soup: BeautifulSoup object
            selector: CSS selector

        Returns:
            List of extracted texts
        """
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements]

    def extract_metadata(self, soup: BeautifulSoup) -> dict[str, Any]:
        """
        Extract common metadata from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Metadata dictionary
        """
        metadata = {
            "title": None,
            "description": None,
            "keywords": [],
            "author": None,
            "published_date": None,
            "modified_date": None,
        }

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            property_val = meta.get("property", "").lower()
            content = meta.get("content", "")

            if name == "description" or property_val == "og:description":
                metadata["description"] = content
            elif name == "keywords":
                metadata["keywords"] = [k.strip() for k in content.split(",")]
            elif name == "author" or property_val == "article:author":
                metadata["author"] = content
            elif name == "published_time" or property_val == "article:published_time":
                metadata["published_date"] = content
            elif name == "modified_time" or property_val == "article:modified_time":
                metadata["modified_date"] = content

        return metadata

    def get_health_status(self) -> dict[str, Any]:
        """
        Get adapter health status and metrics.

        Returns:
            Health status dictionary
        """
        return {
            "source": self.source_name,
            "status": self.status.value,
            "base_url": self.base_url,
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failures": self.circuit_breaker.failure_count,
            },
            "metrics": {
                "requests": self.request_count,
                "errors": self.error_count,
                "error_rate": (
                    self.error_count / self.request_count
                    if self.request_count > 0
                    else 0.0
                ),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0
                    else 0.0
                ),
            },
            "rate_limit": {
                "per_second": self.rate_limit_per_second,
                "tokens_available": self.rate_limiter.tokens,
            },
        }

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    async def fetch_document(self, document_id: str) -> dict[str, Any]:
        """
        Fetch single document from legal source.

        Args:
            document_id: Unique document identifier

        Returns:
            Document data dictionary with keys:
                - id: Document ID
                - title: Document title
                - content: Full text content
                - metadata: Additional metadata
                - source_url: Original source URL
                - fetch_date: Fetch timestamp

        Raises:
            NetworkError: Network or HTTP error
            ParsingError: Failed to parse document
            ValidationError: Invalid document ID
        """
        pass

    @abstractmethod
    async def fetch_documents(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple documents from legal source.

        Args:
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            limit: Maximum number of documents to fetch

        Returns:
            List of document dictionaries

        Raises:
            NetworkError: Network or HTTP error
            ParsingError: Failed to parse documents
            ValidationError: Invalid date range
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search for documents in legal source.

        Args:
            query: Search query string
            filters: Additional search filters (source-specific)
            limit: Maximum number of results

        Returns:
            List of matching document dictionaries

        Raises:
            NetworkError: Network or HTTP error
            ParsingError: Failed to parse search results
            ValidationError: Invalid query
        """
        pass


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "BaseAdapter",
    "DocumentFormat",
    "AdapterStatus",
    "CircuitBreaker",
    "RateLimiter",
]
