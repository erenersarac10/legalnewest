"""LLM Context Optimizer - Harvey/Legora CTO-Level
Optimize context window usage for maximum relevance"""
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ContextOptimizer:
    """Optimize context for LLM calls"""
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
    
    def optimize(self, chunks: List[str], query: str) -> List[str]:
        """Optimize context chunks
        
        Args:
            chunks: Context chunks
            query: User query
            
        Returns:
            Optimized chunk list
        """
        # Score chunks by relevance
        scored = [(chunk, self._score_relevance(chunk, query)) for chunk in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select top chunks within token limit
        selected = []
        total_tokens = 0
        
        for chunk, score in scored:
            chunk_tokens = len(chunk) // 4
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            selected.append(chunk)
            total_tokens += chunk_tokens
        
        logger.debug(f"Selected {len(selected)}/{len(chunks)} chunks ({total_tokens} tokens)")
        return selected
    
    def _score_relevance(self, chunk: str, query: str) -> float:
        """Score chunk relevance to query"""
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk.lower().split())
        
        overlap = len(query_terms & chunk_terms)
        return overlap / len(query_terms) if query_terms else 0.0
    
    def truncate_to_limit(self, text: str, limit: int) -> str:
        """Truncate text to token limit"""
        tokens = len(text) // 4
        if tokens <= limit:
            return text
        
        # Truncate to approximately limit tokens
        char_limit = limit * 4
        return text[:char_limit] + "..."

__all__ = ['ContextOptimizer']
