"""SPK (Capital Markets Board) Adapter - Harvey/Legora CTO-Level
Fetches regulations, communiqués, and decisions from spk.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata, CourtDecision

class SPKAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("SPK Adapter", "1.0.0")
        self.base_url = "https://www.spk.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches SPK documents (Tebliğ, Kurul Kararı, Duyuru)"""
        url = f"{self.base_url}/Sayfa/Dosya/{document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1') or soup.find('div', class_='title')
                content_div = soup.find('div', class_='icerik') or soup.find('div', class_='content')

                # Extract SPK decision number if present
                decision_number = None
                if title:
                    match = re.search(r'(\d+/\d+)', title.text)
                    if match:
                        decision_number = match.group(1)

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'decision_number': decision_number,
                    'source': 'SPK'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search SPK documents by keyword"""
        search_url = f"{self.base_url}/Arama?q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for item in soup.find_all('div', class_='search-result')[:10]:
                    title_elem = item.find('a')
                    if title_elem:
                        results.append({
                            'title': title_elem.text.strip(),
                            'url': self.base_url + title_elem['href'],
                            'snippet': item.find('p').text.strip() if item.find('p') else ''
                        })
                return results

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from title
        title_lower = raw_data.get('title', '').lower()
        if 'tebliğ' in title_lower:
            doc_type = DocumentType.TEBLIG
        elif 'karar' in title_lower or raw_data.get('decision_number'):
            doc_type = DocumentType.KURUL_KARARI
        else:
            doc_type = DocumentType.DUYURU

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.TEBLIG,
            source=SourceType.SPK,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            decision_number=raw_data.get('decision_number')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['SPKAdapter']
