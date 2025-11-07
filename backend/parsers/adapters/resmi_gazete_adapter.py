"""
Resmi Gazete (Official Gazette of Turkey) adapter.

This module provides the RasmiGazeteAdapter for scraping and parsing
official publications from Turkey's Official Gazette (Resmi Gazete).

The Official Gazette (Resmi Gazete) is the official journal of the Turkish
government, publishing all laws, decrees, regulations, and official announcements.
Every legal document published here has legal force.

Data Source:
    - Website: https://www.resmigazete.gov.tr
    - Format: PDF files (daily publications)
    - Content: Laws, decrees, regulations, announcements
    - Publication: Daily (including duplicates/mükerrer)

Document Types Published:
    - Kanun (Law)
    - Kanun Hükmünde Kararname (Decree with force of law)
    - Tüzük (Regulation)
    - Yönetmelik (By-law)
    - Tebliğ (Communiqué)
    - İlan (Announcement)
    - Atama (Appointment)

Special Features:
    - Duplicate issues (Mükerrer 1, Mükerrer 2, etc.)
    - Supplement issues (Ek sayı)
    - PDF parsing with OCR fallback
    - Full-text search capabilities
    - Historical archive (1920-present)

Example:
    >>> async with ResmiGazeteAdapter() as adapter:
    ...     # Fetch today's gazette
    ...     doc = await adapter.fetch_latest()
    ...
    ...     # Fetch specific date
    ...     docs = await adapter.fetch_by_date(date(2024, 11, 7))
    ...
    ...     # Search in gazettes
    ...     results = await adapter.search("KVKK")
"""

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urljoin

import pdfplumber
from bs4 import BeautifulSoup

from backend.core.constants import RESMI_GAZETE_BASE_URL
from backend.core.exceptions import NetworkError, ParsingError, ValidationError
from backend.core.logging import get_logger
from backend.parsers.adapters.base_adapter import BaseAdapter

# =============================================================================
# LOGGER
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# RESMI GAZETE ADAPTER
# =============================================================================


class ResmiGazeteAdapter(BaseAdapter):
    """
    Adapter for Turkey's Official Gazette (Resmi Gazete).

    Provides comprehensive access to official government publications:
    - Daily gazette downloads (PDF)
    - PDF text extraction with OCR fallback
    - Metadata extraction (number, date, type)
    - Historical archive access (1920-present)
    - Search capabilities
    - Duplicate issue handling (mükerrer)

    URL Structure:
        - Main: https://www.resmigazete.gov.tr
        - Archive: https://www.resmigazete.gov.tr/eskiler/YYYY/MM/YYYYMMDD.pdf
        - Duplicate: https://www.resmigazete.gov.tr/eskiler/YYYY/MM/YYYYMMDD-1.pdf
        - Search: https://www.resmigazete.gov.tr/arama.aspx

    Attributes:
        min_date: Minimum supported date (1920-01-01)
        max_date: Maximum supported date (today)
    """

    def __init__(self):
        """Initialize Resmi Gazete adapter."""
        super().__init__(
            source_name="Resmi Gazete",
            base_url=RESMI_GAZETE_BASE_URL,
            rate_limit_per_second=0.5,  # Respectful: 1 request per 2 seconds
            cache_ttl=86400,  # Cache PDFs for 24 hours (they don't change)
            timeout=60,  # PDF downloads may take time
        )

        # Date limits
        self.min_date = date(1920, 1, 1)  # First Resmi Gazete
        self.max_date = date.today()

    def _validate_date(self, check_date: date):
        """
        Validate date is within supported range.

        Args:
            check_date: Date to validate

        Raises:
            ValidationError: Date out of range
        """
        if check_date < self.min_date:
            raise ValidationError(
                message=f"Date before first Resmi Gazete publication",
                field="date",
                details={
                    "provided": str(check_date),
                    "min_date": str(self.min_date),
                }
            )

        if check_date > self.max_date:
            raise ValidationError(
                message=f"Date in future",
                field="date",
                details={
                    "provided": str(check_date),
                    "max_date": str(self.max_date),
                }
            )

    def _build_pdf_url(self, gazette_date: date, duplicate: int = 0) -> str:
        """
        Build PDF URL for specific gazette date.

        Args:
            gazette_date: Publication date
            duplicate: Duplicate number (0=main, 1=first duplicate, etc.)

        Returns:
            PDF URL

        Examples:
            >>> _build_pdf_url(date(2024, 11, 7), 0)
            'https://www.resmigazete.gov.tr/eskiler/2024/11/20241107.pdf'

            >>> _build_pdf_url(date(2024, 11, 7), 1)
            'https://www.resmigazete.gov.tr/eskiler/2024/11/20241107-1.pdf'
        """
        year = gazette_date.year
        month = f"{gazette_date.month:02d}"
        date_str = gazette_date.strftime("%Y%m%d")

        # Duplicate suffix
        duplicate_suffix = f"-{duplicate}" if duplicate > 0 else ""

        # URL structure: /eskiler/YYYY/MM/YYYYMMDD[-N].pdf
        path = f"/eskiler/{year}/{month}/{date_str}{duplicate_suffix}.pdf"

        return urljoin(self.base_url, path)

    async def _download_pdf(self, url: str) -> bytes:
        """
        Download PDF file.

        Args:
            url: PDF URL

        Returns:
            PDF content as bytes

        Raises:
            NetworkError: Download failed
        """
        try:
            # Note: get() returns str, but we need bytes for PDF
            await self.initialize()

            response = await self.client.get(url)
            response.raise_for_status()

            logger.debug(
                f"Downloaded PDF from Resmi Gazete",
                url=url,
                size=len(response.content),
            )

            return response.content

        except Exception as e:
            raise NetworkError(
                message=f"Failed to download PDF: {str(e)}",
                details={"url": url, "error": str(e)}
            )

    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF using pdfplumber with OCR fallback.

        Harvey/Legora %100 parite: Dual-path OCR.
        - Primary: pdfplumber (fast, digital text)
        - Fallback: pytesseract OCR (scanned/image PDFs)

        Args:
            pdf_content: PDF file content

        Returns:
            Extracted text

        Raises:
            ParsingError: Failed to extract text even with OCR
        """
        try:
            import io
            pdf_file = io.BytesIO(pdf_content)

            text_parts = []
            ocr_used = False

            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                        else:
                            # OCR FALLBACK: Page has no text → try OCR
                            logger.warning(
                                f"No text on page {page_num} - attempting OCR",
                                page=page_num
                            )
                            ocr_text = self._ocr_page(page)
                            if ocr_text and ocr_text.strip():
                                text_parts.append(ocr_text)
                                ocr_used = True
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num}",
                            error=str(e)
                        )
                        # Continue with other pages

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise ParsingError(
                    message="No text extracted from PDF even with OCR",
                    details={"pages": len(text_parts), "ocr_attempted": ocr_used}
                )

            logger.debug(
                "Extracted text from PDF",
                extra={
                    "pages": len(text_parts),
                    "text_length": len(full_text),
                    "ocr_used": ocr_used,
                }
            )

            return full_text

        except Exception as e:
            raise ParsingError(
                message=f"Failed to parse PDF: {str(e)}",
                details={"error": str(e)}
            )

    def _ocr_page(self, page) -> str:
        """
        OCR fallback for scanned/image PDF pages.

        Harvey/Legora %100 parite: OCR with pytesseract.

        Args:
            page: pdfplumber page object

        Returns:
            OCR-extracted text

        Note:
            Requires tesseract-ocr installed (apt install tesseract-ocr tesseract-ocr-tur)
        """
        try:
            # Import OCR dependencies (lazy to avoid hard requirement)
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                logger.warning("pytesseract not installed - OCR unavailable")
                return ""

            # Convert PDF page to image
            image = page.to_image(resolution=300)  # 300 DPI for quality
            pil_image = image.original

            # OCR with Turkish language support
            text = pytesseract.image_to_string(pil_image, lang='tur+eng')

            logger.debug(
                "OCR completed",
                extra={"text_length": len(text), "page_size": pil_image.size}
            )

            return text

        except Exception as e:
            logger.error(
                "OCR failed",
                error=str(e)
            )
            return ""

    def _extract_metadata_from_text(self, text: str, gazette_date: date) -> dict[str, Any]:
        """
        Extract metadata from gazette text.

        Args:
            text: Gazette text
            gazette_date: Publication date

        Returns:
            Metadata dictionary
        """
        metadata = {
            "publication_date": gazette_date.isoformat(),
            "year": gazette_date.year,
            "month": gazette_date.month,
            "day": gazette_date.day,
            "gazette_number": None,
            "duplicate_number": 0,
            "document_types": [],
            "laws": [],
            "decrees": [],
            "keywords": [],
        }

        # Extract gazette number (e.g., "Sayı: 32356")
        number_match = re.search(r"Sayı\s*:\s*(\d+)", text, re.IGNORECASE)
        if number_match:
            metadata["gazette_number"] = int(number_match.group(1))

        # Extract duplicate number (e.g., "Mükerrer 1", "Mükerrer 2")
        duplicate_match = re.search(r"Mükerrer\s*(\d+)", text, re.IGNORECASE)
        if duplicate_match:
            metadata["duplicate_number"] = int(duplicate_match.group(1))

        # Detect document types
        if re.search(r"\bKANUN\b", text, re.IGNORECASE):
            metadata["document_types"].append("kanun")

        if re.search(r"\bYÖNETMELİK\b", text, re.IGNORECASE):
            metadata["document_types"].append("yonetmelik")

        if re.search(r"\bTEBLİĞ\b", text, re.IGNORECASE):
            metadata["document_types"].append("teblig")

        if re.search(r"\bKARARNAME\b", text, re.IGNORECASE):
            metadata["document_types"].append("kararname")

        # Extract law numbers (e.g., "7524 sayılı Kanun")
        law_matches = re.findall(r"(\d+)\s+sayılı\s+Kanun", text, re.IGNORECASE)
        metadata["laws"] = [int(num) for num in law_matches]

        # Extract keywords (common legal terms)
        keywords = []
        for keyword in ["KVKK", "TBK", "TTK", "Anayasa", "Yargıtay", "Danıştay"]:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                keywords.append(keyword)
        metadata["keywords"] = keywords

        return metadata

    async def fetch_document(self, document_id: str) -> dict[str, Any]:
        """
        Fetch single Resmi Gazete document by ID.

        Document ID format: "YYYY-MM-DD" or "YYYY-MM-DD-N" (for duplicates)

        Args:
            document_id: Document ID (date-based)

        Returns:
            Document data dictionary

        Raises:
            ValidationError: Invalid document ID format
            NetworkError: Failed to fetch document
            ParsingError: Failed to parse document

        Example:
            >>> doc = await adapter.fetch_document("2024-11-07")
            >>> doc = await adapter.fetch_document("2024-11-07-1")  # Duplicate
        """
        # Parse document ID
        parts = document_id.split("-")
        if len(parts) < 3:
            raise ValidationError(
                message="Invalid document ID format",
                field="document_id",
                details={"expected": "YYYY-MM-DD or YYYY-MM-DD-N", "got": document_id}
            )

        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            duplicate = int(parts[3]) if len(parts) > 3 else 0

            gazette_date = date(year, month, day)
        except (ValueError, IndexError) as e:
            raise ValidationError(
                message="Invalid document ID format",
                field="document_id",
                details={"error": str(e), "document_id": document_id}
            )

        # Validate date
        self._validate_date(gazette_date)

        # Build PDF URL
        pdf_url = self._build_pdf_url(gazette_date, duplicate)

        # Download PDF
        pdf_content = await self._download_pdf(pdf_url)

        # Compute PDF checksum (SHA256) for change detection and caching
        # This prevents reprocessing identical PDFs and enables delta sync
        import hashlib
        pdf_checksum = hashlib.sha256(pdf_content).hexdigest()

        # Check if we've already processed this exact PDF
        cache_key = f"rg:checksum:{pdf_checksum}"
        cached_doc = await self._get_from_cache(cache_key)
        if cached_doc:
            logger.info(
                "PDF already processed (checksum match)",
                document_id=document_id,
                checksum=pdf_checksum[:8],
            )
            return cached_doc

        # Extract text (only if not cached)
        text = self._extract_text_from_pdf(pdf_content)

        # Extract metadata
        metadata = self._extract_metadata_from_text(text, gazette_date)

        # Build document
        document = {
            "id": document_id,
            "title": f"Resmi Gazete - {gazette_date.isoformat()}" + (
                f" (Mükerrer {duplicate})" if duplicate > 0 else ""
            ),
            "content": text,
            "metadata": metadata,
            "source_url": pdf_url,
            "fetch_date": datetime.now(timezone.utc).isoformat(),
            "format": "pdf",
            "size_bytes": len(pdf_content),
            "checksum": pdf_checksum,  # Store for delta sync
        }

        # Cache processed document by checksum (24h TTL)
        await self._save_to_cache(cache_key, document)

        logger.info(
            "Fetched Resmi Gazete document",
            document_id=document_id,
            gazette_number=metadata.get("gazette_number"),
            text_length=len(text),
        )

        return document

    async def fetch_documents(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple Resmi Gazete documents.

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)
            limit: Maximum documents to fetch

        Returns:
            List of documents

        Raises:
            ValidationError: Invalid date range

        Example:
            >>> docs = await adapter.fetch_documents(
            ...     start_date=datetime(2024, 11, 1),
            ...     end_date=datetime(2024, 11, 7),
            ...     limit=50
            ... )
        """
        # Default date range: last 30 days
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Convert to date
        start_date_only = start_date.date()
        end_date_only = end_date.date()

        # Validate dates
        self._validate_date(start_date_only)
        self._validate_date(end_date_only)

        if start_date_only > end_date_only:
            raise ValidationError(
                message="start_date must be before end_date",
                field="date_range",
                details={
                    "start_date": str(start_date_only),
                    "end_date": str(end_date_only),
                }
            )

        # Fetch documents
        documents = []
        current_date = start_date_only

        while current_date <= end_date_only and len(documents) < limit:
            document_id = current_date.isoformat()

            try:
                # Try to fetch main document
                doc = await self.fetch_document(document_id)
                documents.append(doc)

                # Check for duplicate issues (try up to 3 duplicates)
                for duplicate_num in range(1, 4):
                    if len(documents) >= limit:
                        break

                    duplicate_id = f"{document_id}-{duplicate_num}"
                    try:
                        duplicate_doc = await self.fetch_document(duplicate_id)
                        documents.append(duplicate_doc)
                    except NetworkError:
                        # No more duplicates
                        break

            except NetworkError as e:
                # Document doesn't exist for this date (weekend/holiday)
                logger.debug(
                    f"No gazette for date {current_date}",
                    date=str(current_date),
                )

            # Next date
            current_date += timedelta(days=1)

        logger.info(
            "Fetched Resmi Gazete documents",
            count=len(documents),
            start_date=str(start_date_only),
            end_date=str(end_date_only),
        )

        return documents

    async def search(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search Resmi Gazete documents.

        Args:
            query: Search query
            filters: Optional filters (date_from, date_to, document_type)
            limit: Maximum results

        Returns:
            List of matching documents

        Raises:
            ValidationError: Invalid query

        Example:
            >>> results = await adapter.search(
            ...     "KVKK",
            ...     filters={"date_from": "2024-01-01"},
            ...     limit=10
            ... )
        """
        if not query or not query.strip():
            raise ValidationError(
                message="Search query cannot be empty",
                field="query"
            )

        # Parse filters
        filters = filters or {}
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        # Default date range: last 90 days
        if date_to is None:
            date_to = date.today()
        elif isinstance(date_to, str):
            date_to = date.fromisoformat(date_to)

        if date_from is None:
            date_from = date_to - timedelta(days=90)
        elif isinstance(date_from, str):
            date_from = date.fromisoformat(date_from)

        # Fetch documents in date range
        start_datetime = datetime.combine(date_from, datetime.min.time())
        end_datetime = datetime.combine(date_to, datetime.min.time())

        documents = await self.fetch_documents(
            start_date=start_datetime,
            end_date=end_datetime,
            limit=limit * 2,  # Fetch more to account for filtering
        )

        # Filter by query
        matching_documents = []
        query_lower = query.lower()

        for doc in documents:
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()

            # Simple text matching (can be enhanced with fuzzy matching)
            if query_lower in content_lower or query_lower in title_lower:
                matching_documents.append(doc)

                if len(matching_documents) >= limit:
                    break

        logger.info(
            "Searched Resmi Gazete",
            query=query,
            results=len(matching_documents),
            date_range=f"{date_from} to {date_to}",
        )

        return matching_documents

    async def fetch_latest(self) -> dict[str, Any]:
        """
        Fetch the latest (most recent) Resmi Gazete.

        Returns:
            Latest document

        Example:
            >>> latest = await adapter.fetch_latest()
        """
        # Try today and previous days (in case today not published yet)
        for days_ago in range(7):  # Try up to 7 days back
            check_date = date.today() - timedelta(days=days_ago)
            document_id = check_date.isoformat()

            try:
                doc = await self.fetch_document(document_id)
                return doc
            except NetworkError:
                continue

        raise NetworkError(
            message="Could not find any recent Resmi Gazete publication",
            details={"days_checked": 7}
        )

    async def fetch_by_date(
        self,
        gazette_date: date,
        include_duplicates: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Fetch all gazettes for a specific date.

        Args:
            gazette_date: Publication date
            include_duplicates: Include duplicate issues

        Returns:
            List of documents for that date (main + duplicates)

        Example:
            >>> docs = await adapter.fetch_by_date(date(2024, 11, 7))
        """
        self._validate_date(gazette_date)

        documents = []
        document_id = gazette_date.isoformat()

        # Fetch main document
        try:
            doc = await self.fetch_document(document_id)
            documents.append(doc)
        except NetworkError:
            logger.warning(
                f"No main gazette for date {gazette_date}",
                date=str(gazette_date),
            )

        # Fetch duplicates
        if include_duplicates:
            for duplicate_num in range(1, 5):  # Try up to 4 duplicates
                duplicate_id = f"{document_id}-{duplicate_num}"
                try:
                    duplicate_doc = await self.fetch_document(duplicate_id)
                    documents.append(duplicate_doc)
                except NetworkError:
                    break

        return documents


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = ["ResmiGazeteAdapter"]
