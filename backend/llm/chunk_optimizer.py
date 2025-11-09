"""LLM Chunk Optimizer - Harvey/Legora CTO-Level
Optimize document chunking for RAG"""
from typing import List
import re

class ChunkOptimizer:
    """Optimize chunking strategy"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_article(self, text: str) -> List[str]:
        """Chunk by Turkish legal articles"""
        # Split on MADDE markers
        pattern = r'(MADDE\s+\d+|Madde\s+\d+)'
        chunks = re.split(pattern, text)
        
        # Recombine with headers
        result = []
        for i in range(1, len(chunks), 2):
            if i+1 < len(chunks):
                result.append(chunks[i] + chunks[i+1])
        
        return [c.strip() for c in result if c.strip()]
    
    def chunk_semantic(self, text: str) -> List[str]:
        """Semantic chunking"""
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            current.append(sent)
            current_len += sent_len
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks

__all__ = ['ChunkOptimizer']
