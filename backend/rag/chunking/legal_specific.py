"""Turkish Legal-Specific Chunking - Harvey/Legora CTO-Level
Specialized chunking for Turkish legal documents"""
from typing import List, Dict
import re

class TurkishLegalChunker:
    """Turkish legal document chunker"""
    
    def __init__(self, preserve_structure: bool = True):
        self.preserve_structure = preserve_structure
        
        # Turkish legal patterns
        self.madde_pattern = r'(MADDE|Madde)\s+(\d+|[A-Z]+)\s*[-–—]?\s*'
        self.fikra_pattern = r'\((\d+)\)'
        self.bent_pattern = r'([a-z])\)'
        self.bolum_pattern = r'(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ)?\s*BÖLÜM'
    
    def chunk_by_article(self, text: str) -> List[Dict]:
        """Chunk by Turkish legal articles
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split on MADDE
        parts = re.split(self.madde_pattern, text, flags=re.MULTILINE)
        
        i = 0
        while i < len(parts):
            if i + 2 < len(parts) and parts[i] in ['MADDE', 'Madde']:
                article_num = parts[i+1]
                content = parts[i+2]
                
                chunks.append({
                    'text': f"{parts[i]} {article_num} {content}",
                    'type': 'article',
                    'article_number': article_num,
                    'metadata': {
                        'article_start': True,
                        'has_paragraphs': bool(re.search(self.fikra_pattern, content))
                    }
                })
                i += 3
            else:
                if parts[i].strip():
                    chunks.append({
                        'text': parts[i].strip(),
                        'type': 'preamble',
                        'metadata': {}
                    })
                i += 1
        
        return chunks
    
    def chunk_by_section(self, text: str) -> List[Dict]:
        """Chunk by sections (BÖLÜM)"""
        sections = re.split(self.bolum_pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        
        chunks = []
        for i, section in enumerate(sections):
            if section and section.strip():
                chunks.append({
                    'text': section.strip(),
                    'type': 'section',
                    'section_number': i,
                    'metadata': {'is_section': 'BÖLÜM' in section.upper()}
                })
        
        return chunks

__all__ = ['TurkishLegalChunker']
