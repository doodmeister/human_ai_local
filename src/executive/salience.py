from __future__ import annotations
from typing import List

"""Salience scoring for consolidation decisions.

Combines novelty, relevance (to recent STM), length, and sentiment.

Phase 3: sentiment/valence signals are centralized in AttentionManager.
"""

class SalienceScorer:
    def __init__(self, w_novelty: float = 0.3, w_relevance: float = 0.3, w_length: float = 0.25, w_sentiment: float = 0.15):
        self.w_novelty = w_novelty
        self.w_relevance = w_relevance
        self.w_length = w_length
        self.w_sentiment = w_sentiment

    def score(self, text: str, recent: List[str]) -> float:
        if not text:
            return 0.0
        tokens = set(text.lower().split())
        # Novelty: fraction of tokens not appearing in recent context
        recent_tokens = set()
        for r in recent[-6:]:
            recent_tokens.update(r.lower().split())
        novelty = len(tokens - recent_tokens) / max(len(tokens), 1)
        # Relevance: overlap with recent tokens
        relevance = len(tokens & recent_tokens) / max(len(tokens), 1)
        # Length (normalized)
        length_term = min(len(text) / 800.0, 1.0)
        # Sentiment/valence: centralized signal (map -1..1 -> 0..1)
        sentiment = 0.5
        try:
            from src.cognition.attention.attention_manager import get_attention_manager

            _sal, valence = get_attention_manager().estimate_salience_and_valence(text)
            sentiment = max(0.0, min((valence + 1.0) / 2.0, 1.0))
        except Exception:
            sentiment = 0.5
        raw = (
            self.w_novelty * novelty +
            self.w_relevance * relevance +
            self.w_length * length_term +
            self.w_sentiment * sentiment
        )
        return max(0.0, min(raw, 1.0))
