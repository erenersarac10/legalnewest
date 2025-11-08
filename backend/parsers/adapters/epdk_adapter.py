"""EPDK (Energy Market Regulatory Authority) Adapter - Harvey/Legora CTO-Level
Fetches energy sector regulations and decisions from epdk.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata

class EPDKAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("EPDK Adapter", "1.0.0")
        self.base_url = "https://www.epdk.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches EPDK documents (Kurul Kararı, Tebliğ, Yönetmelik)"""
        url = f"{self.base_url}/Detay/Icerik/{document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1', class_='icerik-baslik') or soup.find('h1')
                content_div = soup.find('div', class_='icerik-detay') or soup.find('div', class_='content')

                # Extract EPDK decision number
                decision_number = None
                if title:
                    # Match pattern: "Kurul Kararı: 1234-56"
                    match = re.search(r'(\d{4}-\d+)', title.text)
                    if match:
                        decision_number = match.group(1)

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'decision_number': decision_number,
                    'source': 'EPDK'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search EPDK documents by keyword"""
        # EPDK search implementation
        return []

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from title
        title_lower = raw_data.get('title', '').lower()
        if 'yönetmelik' in title_lower:
            doc_type = DocumentType.YONETMELIK
        elif 'tebliğ' in title_lower:
            doc_type = DocumentType.TEBLIG
        elif 'kurul kararı' in title_lower:
            doc_type = DocumentType.KURUL_KARARI
        else:
            doc_type = DocumentType.TEBLIG

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.TEBLIG if doc_type == DocumentType.TEBLIG else LegalHierarchy.YONETMELIK,
            source=SourceType.EPDK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            decision_number=raw_data.get('decision_number')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['EPDKAdapter']
