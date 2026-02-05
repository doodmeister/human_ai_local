from __future__ import annotations

import math
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .attention_mechanism import AttentionMechanism


_TOKEN_RE = re.compile(r"\w+")

_POSITIVE_WORDS = {"great", "good", "love", "nice", "wonderful", "happy", "excited"}
_NEGATIVE_WORDS = {"bad", "sad", "angry", "upset", "terrible", "hate", "awful", "tired"}
_INTENSIFIERS = {"very", "extremely", "incredibly", "totally", "really", "so", "super"}


class AttentionManager:
    """Central entry-point for attention and salience.

    Constraints enforced here:
    - Attention is a resource with limits.
    - Other modules may only query attention results (no independent scoring).
    """

    def __init__(self, attention: Optional[AttentionMechanism] = None) -> None:
        self._attention = attention or AttentionMechanism()
        self._lock = threading.RLock()

    # -----------------
    # Budget / results
    # -----------------

    def get_budget(self) -> Dict[str, int]:
        with self._lock:
            max_items = int(getattr(self._attention.config, "max_attention_items", 7))
            used = len(getattr(self._attention, "focused_items", {}) or {})
            return {"max_items": max_items, "used_items": used, "remaining_items": max(0, max_items - used)}

    @property
    def current_focus(self) -> List[Dict[str, Any]]:
        """ContextBuilder-compatible focus list.

        Returns list[dict] with at least: id, content, attention_score.
        """
        with self._lock:
            focused_items = getattr(self._attention, "focused_items", {}) or {}
            items = sorted(focused_items.values(), key=lambda it: float(getattr(it, "activation", 0.0)), reverse=True)
            out: List[Dict[str, Any]] = []
            for it in items:
                out.append(
                    {
                        "id": str(getattr(it, "id", "")),
                        "content": getattr(it, "content", ""),
                        "attention_score": float(getattr(it, "activation", 0.0)),
                        "salience": float(getattr(it, "salience", 0.0)),
                        "priority": float(getattr(it, "priority", 0.5)),
                        "effort_required": float(getattr(it, "effort_required", 0.5)),
                    }
                )
            return out

    # -----------------
    # Salience signals
    # -----------------

    def estimate_salience_and_valence(self, text: str) -> Tuple[float, float]:
        """Lightweight heuristic for salience & emotional valence.

        Returns (salience: 0..1, valence: -1..1).
        """
        if not text:
            return 0.0, 0.0

        tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
        length_factor = min(1.0, len(tokens) / 25.0)
        punctuation_count = text.count("!") + text.count("?")
        punctuation_boost = min(0.25, 0.07 * punctuation_count)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        emphasis_boost = min(0.25, uppercase_ratio * 0.6)
        intensifier_hits = sum(1 for t in tokens if t in _INTENSIFIERS)
        intensifier_boost = min(0.2, intensifier_hits * 0.05)

        salience = max(
            0.0,
            min(
                1.0,
                0.25 + length_factor * 0.4 + punctuation_boost + emphasis_boost + intensifier_boost,
            ),
        )

        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            valence = 0.0
        else:
            valence = (pos - neg) / total
        valence = math.tanh(valence)

        return salience, valence

    # -----------------
    # Attention scoring
    # -----------------

    def score_relevance(self, query: str, candidates: Sequence[str]) -> List[float]:
        """Return a relevance score (0..1) per candidate.

        This is intentionally lightweight and deterministic.
        """
        if not candidates:
            return []
        q_tokens = set((query or "").lower().split())
        denom = max(len(q_tokens), 1)
        scores: List[float] = []
        for c in candidates:
            c_tokens = set((c or "").lower().split())
            overlap = len(q_tokens & c_tokens)
            scores.append(overlap / denom)
        return scores

    def calculate_attention_score(
        self,
        *,
        relevance: float,
        novelty: float,
        emotional_salience: float = 0.0,
        current_fatigue: float = 0.0,
    ) -> float:
        """Calculate an attention score (0..1) from basic factors.

        This is provided so legacy helpers can delegate to a single central
        attention scoring implementation.
        """
        base_attention = (float(relevance) * 0.7) + (float(novelty) * 0.3)
        emotional_boost = abs(float(emotional_salience)) * 0.2
        fatigue_penalty = float(current_fatigue) * 0.3
        return max(0.0, min(1.0, base_attention + emotional_boost - fatigue_penalty))

    async def allocate(self, *, query: str, candidates: List[str], capacity: int = 5) -> List[str]:
        """Select <=capacity candidates, respecting attention resource limits."""
        if not candidates or capacity <= 0:
            return []

        with self._lock:
            budget = self.get_budget()
            effective_capacity = min(int(capacity), int(budget.get("remaining_items", 0)))
            if effective_capacity <= 0:
                return []

            # Novelty frequency map (exact match)
            freq: Dict[str, int] = {}
            for c in candidates:
                freq[c] = freq.get(c, 0) + 1

            n = len(candidates)
            q_tokens = set((query or "").lower().split())

            scored: List[Tuple[float, str, float]] = []  # (score, candidate, salience)
            for idx, c in enumerate(candidates):
                c_tokens = set((c or "").lower().split())
                overlap = len(q_tokens & c_tokens)
                relevance = overlap / max(len(q_tokens), 1)
                recency = (idx + 1) / max(n, 1)
                novelty_penalty = (freq[c] - 1) * 0.15

                # Salience is a first-class signal: compute once here.
                salience, _valence = self.estimate_salience_and_valence(c)

                raw_score = 0.50 * relevance + 0.30 * recency + 0.20 * max(0.0, 1.0 - novelty_penalty)
                # Modest salience influence (kept small to preserve prior behavior)
                score = raw_score + (salience * 0.10)
                scored.append((score, c, salience))

            scored.sort(key=lambda t: t[0], reverse=True)
            chosen = scored[:effective_capacity]

            # Update underlying attention state as the single place that consumes capacity.
            for rank, (_score, c, sal) in enumerate(chosen):
                stimulus_id = f"att:{hash(c)}"
                try:
                    self._attention.allocate_attention(
                        stimulus_id=stimulus_id,
                        content=c,
                        salience=float(sal),
                        novelty=0.0,
                        priority=max(0.0, min(1.0, 0.7 - (rank * 0.05))),
                        effort_required=0.15,
                    )
                except Exception:
                    # If capacity management fails, do not crash callers; just stop allocating.
                    break

            return [c for (_s, c, _sal) in chosen]


_attention_manager_singleton: Optional[AttentionManager] = None
_singleton_lock = threading.Lock()


def get_attention_manager() -> AttentionManager:
    global _attention_manager_singleton
    if _attention_manager_singleton is None:
        with _singleton_lock:
            if _attention_manager_singleton is None:
                _attention_manager_singleton = AttentionManager()
    return _attention_manager_singleton
