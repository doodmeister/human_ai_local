from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalPlan:
    query: str
    intent: str
    search_stm: bool = True
    search_ltm: bool = True
    search_episodic: bool = True
    include_prospective: bool = True
    relationship_target: str | None = None
    relationship_strength: float | None = None
    store_limits: dict[str, int] = field(default_factory=dict)
    fallback_policy: str = "graceful_degrade"
    reasoning: list[str] = field(default_factory=list)

    def limit_for(self, store: str, default: int) -> int:
        value = self.store_limits.get(store, default)
        try:
            return max(1, int(value))
        except Exception:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "search_stm": self.search_stm,
            "search_ltm": self.search_ltm,
            "search_episodic": self.search_episodic,
            "include_prospective": self.include_prospective,
            "relationship_target": self.relationship_target,
            "relationship_strength": self.relationship_strength,
            "store_limits": dict(self.store_limits),
            "fallback_policy": self.fallback_policy,
            "reasoning": list(self.reasoning),
        }