"""LLM Early Exit Handler - Harvey/Legora CTO-Level
Early termination logic for cost optimization"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EarlyExitHandler:
    """Handle early exit conditions"""
    
    def __init__(
        self,
        max_tokens: Optional[int] = None,
        confidence_threshold: float = 0.9
    ):
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
    
    def should_exit_early(
        self,
        current_tokens: int,
        confidence: Optional[float] = None,
        has_answer: bool = False
    ) -> bool:
        """Check if should exit early
        
        Args:
            current_tokens: Current token count
            confidence: Answer confidence score
            has_answer: Whether answer is found
            
        Returns:
            True if should exit
        """
        # Token limit exceeded
        if self.max_tokens and current_tokens >= self.max_tokens:
            logger.info(f"Early exit: token limit ({current_tokens}/{self.max_tokens})")
            return True
        
        # High confidence answer found
        if has_answer and confidence and confidence >= self.confidence_threshold:
            logger.info(f"Early exit: high confidence answer (conf={confidence:.2f})")
            return True
        
        return False
    
    def truncate_response(self, response: str, max_length: int) -> str:
        """Truncate response to max length"""
        if len(response) <= max_length:
            return response
        
        # Try to truncate at sentence boundary
        truncated = response[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.8:
            return truncated[:last_period+1]
        
        return truncated + "..."

__all__ = ['EarlyExitHandler']
