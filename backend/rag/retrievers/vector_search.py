"""Vector Search Retriever - Harvey/Legora CTO-Level"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class VectorSearchRetriever:
    """Vector similarity search for legal documents"""
    
    def __init__(self, vector_store=None, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query_embedding: List[float], filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve similar chunks
        
        Args:
            query_embedding: Query vector
            filters: Optional metadata filters
            
        Returns:
            List of retrieved chunks with scores
        """
        if not self.vector_store:
            logger.warning("No vector store configured")
            return []
        
        try:
            results = self.vector_store.similarity_search(
                query_embedding,
                k=self.top_k,
                filter=filters
            )
            
            return [
                {
                    'text': r.get('text', ''),
                    'score': r.get('score', 0.0),
                    'metadata': r.get('metadata', {})
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

__all__ = ['VectorSearchRetriever']
