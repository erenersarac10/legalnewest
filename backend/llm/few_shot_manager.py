"""LLM Few-Shot Manager - Harvey/Legora CTO-Level
Manage and select few-shot examples"""
from typing import List, Dict, Optional
import random

class FewShotManager:
    """Manage few-shot learning examples"""
    
    def __init__(self):
        self.examples: Dict[str, List[Dict]] = {}
    
    def add_example(self, category: str, input_text: str, output_text: str, **metadata):
        """Add few-shot example"""
        if category not in self.examples:
            self.examples[category] = []
        
        self.examples[category].append({
            'input': input_text,
            'output': output_text,
            'metadata': metadata
        })
    
    def get_examples(
        self,
        category: str,
        n: int = 3,
        query: Optional[str] = None
    ) -> List[Dict]:
        """Get N examples for category
        
        Args:
            category: Example category
            n: Number of examples
            query: Optional query for similarity-based selection
            
        Returns:
            List of examples
        """
        if category not in self.examples:
            return []
        
        examples = self.examples[category]
        
        if query:
            # Score by similarity
            scored = [(ex, self._similarity(query, ex['input'])) for ex in examples]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [ex for ex, score in scored[:n]]
        
        # Random selection
        return random.sample(examples, min(n, len(examples)))
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0

__all__ = ['FewShotManager']
