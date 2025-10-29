from __future__ import annotations
from typing import List

"""Enhanced attention allocation with simple multi-factor scoring.

Factors:
- recency: Later items (as given) boosted.
- relevance: crude substring or token overlap with query.
- novelty: penalize near-duplicates (exact match frequency).

This is intentionally lightweight and deterministic; can be upgraded to use
embeddings or a neural predictor later.
"""

class EnhancedAttentionService:
    def __init__(self, alpha_recency: float = 0.30, alpha_relevance: float = 0.50, alpha_novelty: float = 0.20):
        self.alpha_recency = alpha_recency
        self.alpha_relevance = alpha_relevance
        self.alpha_novelty = alpha_novelty

    async def allocate(self, *, query: str, candidates: List[str], capacity: int = 5) -> List[str]:  # Protocol-compatible
        if not candidates or capacity <= 0:
            return []
        # Precompute frequencies for novelty penalty
        freq: dict[str, int] = {}
        for c in candidates:
            freq[c] = freq.get(c, 0) + 1
        scored = []
        n = len(candidates)
        q_tokens = set(query.lower().split())
        for idx, c in enumerate(candidates):
            c_tokens = set(c.lower().split())
            overlap = len(q_tokens & c_tokens)
            relevance = overlap / max(len(q_tokens), 1)
            recency = (idx + 1) / n  # later entries higher idx -> larger value
            novelty_penalty = (freq[c] - 1) * 0.15
            raw_score = (
                self.alpha_relevance * relevance +
                self.alpha_recency * recency +
                self.alpha_novelty * (1.0 - novelty_penalty)
            )
            scored.append((raw_score, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for (_s, c) in scored[:capacity]]