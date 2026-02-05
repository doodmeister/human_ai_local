from __future__ import annotations

from typing import List

"""Enhanced attention allocation.

Phase 3: attention selection is centralized in src.cognition.attention.AttentionManager.
This module remains as a compatibility shim for existing Executive wiring.
"""


class EnhancedAttentionService:
    def __init__(self, alpha_recency: float = 0.30, alpha_relevance: float = 0.50, alpha_novelty: float = 0.20):
        self.alpha_recency = alpha_recency
        self.alpha_relevance = alpha_relevance
        self.alpha_novelty = alpha_novelty

    async def allocate(self, *, query: str, candidates: List[str], capacity: int = 5) -> List[str]:
        from src.cognition.attention.attention_manager import get_attention_manager

        # Delegate to the single source of truth.
        return await get_attention_manager().allocate(query=query, candidates=candidates, capacity=capacity)