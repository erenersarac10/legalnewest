"""TBMM (Turkish Grand National Assembly) Adapter - Harvey/Legora CTO-Level
Fetches legislation, parliamentary minutes, and commission reports from tbmm.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata

class TBMMAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("TBMM Adapter", "1.0.0")
        self.base_url = "https://www.tbmm.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches TBMM documents (Kanun Teklifi, Komisyon Raporu, Tutanak)"""
        # TBMM document ID format: "dönem/yasama_yılı/esas_no"
        url = f"{self.base_url}/sirasayi/{document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1', class_='baslik') or soup.find('h1')
                content_div = soup.find('div', class_='icerik') or soup.find('div', id='content')

                # Extract legislative info
                esas_no = None
                karar_no = None
                donem = None

                info_table = soup.find('table', class_='bilgi')
                if info_table:
                    for row in info_table.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].text.strip().lower()
                            value = cells[1].text.strip()
                            if 'esas' in label:
                                esas_no = value
                            elif 'karar' in label:
                                karar_no = value
                            elif 'dönem' in label:
                                donem = value

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'esas_no': esas_no,
                    'karar_no': karar_no,
                    'donem': donem,
                    'source': 'TBMM'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search TBMM documents by keyword"""
        search_url = f"{self.base_url}/arama?q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for item in soup.find_all('div', class_='sonuc')[:10]:
                    title_elem = item.find('a')
                    if title_elem:
                        results.append({
                            'title': title_elem.text.strip(),
                            'url': self.base_url + title_elem['href'],
                            'excerpt': item.find('p').text.strip() if item.find('p') else ''
                        })
                return results

    def _extract_raw_data(self, preprocessed: Any, **kwargs) -> Dict[str, Any]:
        return preprocessed

    def _transform_to_canonical(self, raw_data: Dict[str, Any], document_type: Optional[DocumentType], **kwargs):
        from ..core.canonical_schema import LegalDocument, Metadata, SourceType, JurisdictionType, LegalHierarchy, EffectivityStatus

        # Determine document type from title
        title_lower = raw_data.get('title', '').lower()
        if 'kanun teklif' in title_lower or 'kanun tasarı' in title_lower:
            doc_type = DocumentType.KANUN_TASARISI
        elif 'komisyon rapor' in title_lower:
            doc_type = DocumentType.KOMISYON_RAPORU
        elif 'tutanak' in title_lower:
            doc_type = DocumentType.TUTANAK
        elif 'karar' in title_lower:
            doc_type = DocumentType.MECLIS_KARARI
        else:
            doc_type = DocumentType.KANUN_TASARISI

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.YASAMA,
            hierarchy_level=LegalHierarchy.KANUN,  # Legislative level
            source=SourceType.TBMM,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.TASLAK,  # Draft until enacted
            esas_no=raw_data.get('esas_no'),
            karar_no=raw_data.get('karar_no'),
            donem=raw_data.get('donem')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['TBMMAdapter']
