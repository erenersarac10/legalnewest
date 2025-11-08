"""Rekabet Kurumu (Competition Authority) Adapter - Harvey/Legora CTO-Level
Fetches competition law decisions and merger clearances from rekabet.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata

class RekabetAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("Rekabet Kurumu Adapter", "1.0.0")
        self.base_url = "https://www.rekabet.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches Rekabet Kurumu decisions (Kurul Kararı, Soruşturma, Birleşme/Devralma)"""
        url = f"{self.base_url}/Karar?kararId={document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1', class_='karar-baslik') or soup.find('h1')
                content_div = soup.find('div', class_='karar-icerik') or soup.find('div', class_='content')

                # Extract decision number (e.g., "23-15/345-M")
                decision_number = None
                decision_date = None
                if title:
                    # Match pattern: YY-AA/BBB-C (Year-Order/Sequential-Type)
                    match = re.search(r'(\d{2}-\d+/\d+-[A-Z])', title.text)
                    if match:
                        decision_number = match.group(1)

                    # Match date
                    date_match = re.search(r'(\d{1,2}\.\d{1,2}\.\d{4})', title.text)
                    if date_match:
                        decision_date = date_match.group(1)

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'decision_number': decision_number,
                    'decision_date': decision_date,
                    'source': 'Rekabet Kurumu'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search Rekabet Kurumu decisions by keyword"""
        search_url = f"{self.base_url}/KararArama?q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for item in soup.find_all('div', class_='karar-item')[:10]:
                    title_elem = item.find('a')
                    if title_elem:
                        results.append({
                            'title': title_elem.text.strip(),
                            'url': self.base_url + title_elem['href'],
                            'summary': item.find('div', class_='ozet').text.strip() if item.find('div', class_='ozet') else ''
                        })
                return results

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from decision number suffix
        decision_num = raw_data.get('decision_number', '')
        if decision_num.endswith('-M'):
            doc_type = DocumentType.BIRLESMELER  # Merger/Acquisition
        elif decision_num.endswith('-K'):
            doc_type = DocumentType.KURUL_KARARI  # Board Decision
        else:
            doc_type = DocumentType.KURUL_KARARI

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_KARAR,
            source=SourceType.REKABET,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            decision_number=raw_data.get('decision_number'),
            decision_date=raw_data.get('decision_date')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['RekabetAdapter']
