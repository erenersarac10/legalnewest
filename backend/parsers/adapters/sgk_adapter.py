"""SGK (Social Security Institution) Adapter - Harvey/Legora CTO-Level
Fetches circulars, guides, and rulings from sgk.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata

class SGKAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("SGK Adapter", "1.0.0")
        self.base_url = "https://www.sgk.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches SGK documents (Genelge, Tebliğ, Rehber)"""
        url = f"{self.base_url}/Sayfa/Detail/{document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1') or soup.find('div', class_='baslik')
                content_div = soup.find('div', class_='icerik') or soup.find('div', class_='detay')

                # Extract SGK circular number (e.g., "2023/15 Sayılı Genelge")
                circular_number = None
                if title:
                    match = re.search(r'(\d{4}/\d+)', title.text)
                    if match:
                        circular_number = match.group(1)

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'circular_number': circular_number,
                    'source': 'SGK'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search SGK documents by keyword"""
        # SGK search implementation
        return []

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
