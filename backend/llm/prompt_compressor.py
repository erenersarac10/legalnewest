"""LLM Prompt Compressor - Harvey/Legora CTO-Level
Token compression techniques for efficient prompting"""
from typing import List, Dict
import re

class PromptCompressor:
    """Compress prompts to reduce token usage"""
    
    TURKISH_STOP_WORDS = {'ve', 'veya', 'ile', 'için', 'bu', 'şu', 'o', 'bir', 'gibi', 'kadar', 'daha', 'çok', 'az'}
    
    def compress(self, text: str, target_ratio: float = 0.7) -> str:
        """Compress text to target ratio
        
        Args:
            text: Input text
            target_ratio: Target compression ratio (0-1)
            
        Returns:
            Compressed text
        """
        # Remove excessive whitespace
        compressed = re.sub(r'\s+', ' ', text)
        compressed = re.sub(r'\n\n+', '\n\n', compressed)
        
        # Remove stop words if needed
        if len(compressed) / len(text) > target_ratio:
            compressed = self._remove_stop_words(compressed)
        
        # Abbreviate common legal terms
        compressed = self._abbreviate_legal_terms(compressed)
        
        return compressed.strip()
    
    def _remove_stop_words(self, text: str) -> str:
        """Remove Turkish stop words"""
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.TURKISH_STOP_WORDS]
        return ' '.join(filtered)
    
    def _abbreviate_legal_terms(self, text: str) -> str:
        """Abbreviate common legal terms"""
        abbreviations = {
            'Türk Ceza Kanunu': 'TCK',
            'Türk Medeni Kanunu': 'TMK',
            'Türk Ticaret Kanunu': 'TTK',
            'Kişisel Verilerin Korunması Kanunu': 'KVKK',
            'madde': 'md.',
            'fıkra': 'f.',
            'bent': 'b.'
        }
        
        for full, abbr in abbreviations.items():
            text = text.replace(full, abbr)
        
        return text
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4

__all__ = ['PromptCompressor']
