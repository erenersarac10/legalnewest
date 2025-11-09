"""LLM Adaptive Router - Harvey/Legora CTO-Level
Intelligent model routing based on task complexity and cost"""
from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """Model performance tiers"""
    FAST = "fast"  # GPT-3.5, Claude Haiku
    BALANCED = "balanced"  # GPT-4, Claude Sonnet
    ADVANCED = "advanced"  # GPT-4 Turbo, Claude Opus

class AdaptiveRouter:
    """Route requests to optimal model"""
    
    def __init__(self):
        self.model_costs = {
            ModelTier.FAST: 0.001,
            ModelTier.BALANCED: 0.01,
            ModelTier.ADVANCED: 0.06
        }
        self.routing_history = []
    
    def route(self, query: str, context_length: int = 0, require_reasoning: bool = False) -> ModelTier:
        """Route to optimal model
        
        Args:
            query: User query
            context_length: Context length in tokens
            require_reasoning: Whether complex reasoning is needed
            
        Returns:
            Recommended model tier
        """
        # Simple routing logic
        complexity_score = self._assess_complexity(query)
        
        if require_reasoning or complexity_score > 0.7:
            tier = ModelTier.ADVANCED
        elif context_length > 8000 or complexity_score > 0.4:
            tier = ModelTier.BALANCED
        else:
            tier = ModelTier.FAST
        
        logger.debug(f"Routed to {tier.value} (complexity={complexity_score:.2f})")
        self.routing_history.append({"query": query, "tier": tier, "complexity": complexity_score})
        
        return tier
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1)"""
        complexity_indicators = [
            'değerlendir', 'analiz et', 'karşılaştır', 'açıkla',
            'neden', 'nasıl', 'adım adım', 'detaylı'
        ]
        
        query_lower = query.lower()
        matches = sum(1 for ind in complexity_indicators if ind in query_lower)
        
        # Normalize to 0-1
        return min(matches / 3.0, 1.0)
    
    def get_estimated_cost(self, tier: ModelTier, token_count: int) -> float:
        """Estimate cost for model tier"""
        return self.model_costs[tier] * (token_count / 1000)

__all__ = ['AdaptiveRouter', 'ModelTier']
