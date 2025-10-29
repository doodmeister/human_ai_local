from __future__ import annotations
from typing import List, Protocol

"""Neural attention predictor stub interface.

Intended future integration point for a DPADRNN or other neural model
that estimates relevance scores for candidate memory snippets given a
query embedding.
"""

class AttentionPredictor(Protocol):
    async def score(self, query: str, candidates: List[str]) -> List[float]:
        """Return a relevance score (0..1) for each candidate (same length list)."""
        ...

class NoOpAttentionPredictor:
    async def score(self, query: str, candidates: List[str]) -> List[float]:  # pragma: no cover
        return [0.5] * len(candidates)
