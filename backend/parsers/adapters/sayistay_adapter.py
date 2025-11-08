"""Sayıştay (Court of Accounts) Adapter - Harvey/Legora CTO-Level
Fetches audit reports and rulings from sayistay.gov.tr
"""
from typing import Dict, List, Any, Optional
import aiohttp, re
from bs4 import BeautifulSoup
from ..core import SourceAdapter, ParsingResult, DocumentType, Metadata

class SayistayAdapter(SourceAdapter):
    def __init__(self):
        super().__init__("Sayıştay Adapter", "1.0.0")
        self.base_url = "https://www.sayistay.gov.tr"

    async def fetch_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Fetches Sayıştay documents (Denetim Raporu, Karar, Görüş)"""
        url = f"{self.base_url}/Rapor/{document_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title and content
                title = soup.find('h1', class_='rapor-baslik') or soup.find('h1')
                content_div = soup.find('div', class_='rapor-icerik') or soup.find('div', class_='content')

                # Extract audit info
                audit_year = None
                report_type = None

                if title:
                    # Extract year
                    year_match = re.search(r'(\d{4})', title.text)
                    if year_match:
                        audit_year = year_match.group(1)

                    # Determine report type
                    if 'mali denetim' in title.text.lower():
                        report_type = 'Mali Denetim'
                    elif 'uygunluk denetim' in title.text.lower():
                        report_type = 'Uygunluk Denetimi'
                    elif 'performans denetim' in title.text.lower():
                        report_type = 'Performans Denetimi'

                return {
                    'title': title.text.strip() if title else '',
                    'content': content_div.text.strip() if content_div else '',
                    'url': url,
                    'audit_year': audit_year,
                    'report_type': report_type,
                    'source': 'Sayıştay'
                }

    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search Sayıştay documents by keyword"""
        search_url = f"{self.base_url}/arama?q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for item in soup.find_all('div', class_='rapor-item')[:10]:
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

        # Determine document type from title
        title_lower = raw_data.get('title', '').lower()
        if 'denetim rapor' in title_lower:
            doc_type = DocumentType.DENETIM_RAPORU
        elif 'karar' in title_lower:
            doc_type = DocumentType.KURUL_KARARI
        elif 'görüş' in title_lower:
            doc_type = DocumentType.GORUS
        else:
            doc_type = DocumentType.DENETIM_RAPORU

        metadata = Metadata(
            document_type=doc_type,
            jurisdiction=JurisdictionType.IDARE,
            hierarchy_level=LegalHierarchy.IDARI_KARAR,
            source=SourceType.SAYISTAY,
            source_url=raw_data.get('url'),
            effectivity_status=EffectivityStatus.YURURLUKTE,
            audit_year=raw_data.get('audit_year'),
            report_type=raw_data.get('report_type')
        )

        return LegalDocument(
            metadata=metadata,
            title=raw_data.get('title', ''),
            full_text=raw_data.get('content', '')
        )

__all__ = ['SayistayAdapter']
