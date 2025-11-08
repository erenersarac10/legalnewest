"""GİB (Revenue Administration) Adapter - Harvey/Legora CTO-Level
Fetches tax regulations, circulars, and rulings from gib.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata

class GIBAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("GİB Adapter", "1.0.0")
        self.base_url = "https://www.gib.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches GİB documents (Genel Tebliğ, Sirküler, Özelge)"""
        url = f"{self.base_url}/node/{document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1', class_='page-title') or soup.find('h1')
                content_div = soup.find('div', class_='field-item') or soup.find('div', class_='content')

                # Extract GİB document number (e.g., "123 Seri No.lu Genel Tebliğ")
                doc_number = None
                if title:
                    match = re.search(r'(\d+)\s+[Ss]eri\s+[Nn]o', title.text)
                    if match:
                        doc_number = match.group(1)

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'document_number': doc_number,
                    'source': 'GİB'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search GİB documents by keyword"""
        # GİB uses internal search - implement based on actual search endpoint
        return []

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from title
        title_lower = raw_data.get('title', '').lower()
        if 'genel tebliğ' in title_lower:
            doc_type = DocumentType.TEBLIG
        elif 'sirküler' in title_lower or 'sirkular' in title_lower:
            doc_type = DocumentType.GENELGE
        elif 'özelge' in title_lower:
            doc_type = DocumentType.OZELGE
        else:
            doc_type = DocumentType.TEBLIG

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.TEBLIG,
            source=SourceType.GIB,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            document_number=raw_data.get('document_number')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['GIBAdapter']
