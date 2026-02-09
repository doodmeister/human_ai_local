"""SelfModel and SelfDiscovery data structures.

The self-model is the agent's *theory of itself* — what it believes
about its own patterns, drives, strengths, weaknesses, and values.
Crucially, this model is **partial and biased**.

The agent can be wrong about itself.  Self-discovery happens when
experience reveals a discrepancy between actual and perceived patterns.

Key design choices
------------------
* ``perceived_patterns`` may differ from actual ``PatternField`` strengths
  due to mood bias and blind spots.
* ``_blind_spots`` are tracked by the system but NOT accessible to the
  agent's self-report; they influence ``perceived_patterns`` via
  under-perception.
* ``self_regard`` is mood-biased: negative mood → lower self-regard.
* ``identity_stability`` recovers slowly and drops on significant
  pattern changes (destabilization).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ────────────────────────────────────────────────────────────────────
#  SelfDiscovery
# ────────────────────────────────────────────────────────────────────

@dataclass
class SelfDiscovery:
    """A moment when the agent learns something true about itself.

    Attributes
    ----------
    pattern_name : str
        Which pattern was revealed or re-evaluated.
    actual_strength : float
        The pattern's real strength.
    perceived_strength : float
        What the agent believed before the discovery.
    message : str
        Human-readable description of the discovery.
    timestamp : float
        When the discovery occurred (epoch seconds).
    """

    pattern_name: str
    actual_strength: float
    perceived_strength: float
    message: str
    timestamp: float = field(default_factory=time.time)

    @property
    def discrepancy(self) -> float:
        return abs(self.actual_strength - self.perceived_strength)

    def to_dict(self) -> Dict[str, object]:
        return {
            "pattern_name": self.pattern_name,
            "actual_strength": round(self.actual_strength, 4),
            "perceived_strength": round(self.perceived_strength, 4),
            "discrepancy": round(self.discrepancy, 4),
            "message": self.message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfDiscovery":
        return cls(
            pattern_name=str(data.get("pattern_name", "")),
            actual_strength=float(data.get("actual_strength", 0.0)),
            perceived_strength=float(data.get("perceived_strength", 0.0)),
            message=str(data.get("message", "")),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ────────────────────────────────────────────────────────────────────
#  SelfModel
# ────────────────────────────────────────────────────────────────────

@dataclass
class SelfModel:
    """What the agent believes about itself (may be inaccurate).

    ``perceived_patterns`` may differ from actual ``PatternField``
    due to mood bias, blind spots, and limited self-knowledge.

    ``_blind_spots`` are PRIVATE to the system — the agent cannot
    report on them.  They reduce the perceived strength of certain
    patterns, simulating the human tendency to not see certain
    truths about ourselves.

    Attributes
    ----------
    perceived_patterns : dict[str, float]
        Name → perceived strength. Biased by mood & blind spots.
    perceived_needs : dict[str, str]
        Drive name → self-narrative about that need.
    perceived_strengths : list[str]
        Pattern names the agent considers strengths.
    perceived_weaknesses : list[str]
        Pattern names the agent considers weaknesses.
    stated_values : list[str]
        Values the agent identifies with.
    _blind_spots : list[str]
        Patterns the agent doesn't fully see in itself (system-private).
    self_regard : float
        Overall self-evaluation (-1 to 1). Mood-biased.
    identity_stability : float
        How stable the self-model feels (0 to 1).
        Drops on significant pattern change, recovers slowly.
    recent_discoveries : list[SelfDiscovery]
        Recent self-discovery moments (ring buffer).
    last_updated_ts : float
        Epoch seconds of last rebuild.
    """

    perceived_patterns: Dict[str, float] = field(default_factory=dict)
    perceived_needs: Dict[str, str] = field(default_factory=dict)
    perceived_strengths: List[str] = field(default_factory=list)
    perceived_weaknesses: List[str] = field(default_factory=list)
    stated_values: List[str] = field(default_factory=list)
    _blind_spots: List[str] = field(default_factory=list)
    self_regard: float = 0.0
    identity_stability: float = 0.5
    recent_discoveries: List[SelfDiscovery] = field(default_factory=list)
    last_updated_ts: float = field(default_factory=time.time)

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def blind_spot_count(self) -> int:
        """How many blind spots exist (count only — contents are private)."""
        return len(self._blind_spots)

    def is_blind_spot(self, pattern_name: str) -> bool:
        """Check if a pattern name is a blind spot (system-level query)."""
        return pattern_name in self._blind_spots

    def top_perceived(self, n: int = 3) -> List[tuple]:
        """Return top-N perceived patterns sorted by strength desc."""
        pairs = sorted(
            self.perceived_patterns.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return pairs[:n]

    def self_description(self) -> str:
        """Natural-language self-description (from perceived data only).

        This is what the agent would say about itself.  Blind spots
        are NOT included.
        """
        parts = []

        if self.perceived_strengths:
            parts.append(
                f"I tend to be {', '.join(self.perceived_strengths[:3])}"
            )

        if self.perceived_weaknesses:
            parts.append(
                f"I sometimes struggle with {', '.join(self.perceived_weaknesses[:2])}"
            )

        if self.stated_values:
            parts.append(
                f"What matters to me: {', '.join(self.stated_values[:3])}"
            )

        if self.self_regard >= 0.3:
            parts.append("Overall I feel good about who I am")
        elif self.self_regard <= -0.3:
            parts.append("I'm not feeling great about myself right now")

        return ". ".join(parts) if parts else "I'm still getting to know myself"

    def summary(self) -> str:
        """Compact debug summary."""
        top = self.top_perceived(3)
        top_str = ", ".join(f"{n}={s:.2f}" for n, s in top)
        return (
            f"SelfModel(regard={self.self_regard:.2f} "
            f"stability={self.identity_stability:.2f} "
            f"perceived=[{top_str}] "
            f"blind_spots={self.blind_spot_count})"
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, object]:
        """Serialize to dict. Blind spots are included for persistence
        but should NOT be exposed to the agent's self-report channel."""
        return {
            "perceived_patterns": {
                k: round(v, 4) for k, v in self.perceived_patterns.items()
            },
            "perceived_needs": dict(self.perceived_needs),
            "perceived_strengths": list(self.perceived_strengths),
            "perceived_weaknesses": list(self.perceived_weaknesses),
            "stated_values": list(self.stated_values),
            "_blind_spots": list(self._blind_spots),
            "self_regard": round(self.self_regard, 4),
            "identity_stability": round(self.identity_stability, 4),
            "recent_discoveries": [d.to_dict() for d in self.recent_discoveries],
            "last_updated_ts": self.last_updated_ts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfModel":
        discoveries = [
            SelfDiscovery.from_dict(d)
            for d in data.get("recent_discoveries", [])
        ]
        return cls(
            perceived_patterns=dict(data.get("perceived_patterns", {})),
            perceived_needs=dict(data.get("perceived_needs", {})),
            perceived_strengths=list(data.get("perceived_strengths", [])),
            perceived_weaknesses=list(data.get("perceived_weaknesses", [])),
            stated_values=list(data.get("stated_values", [])),
            _blind_spots=list(data.get("_blind_spots", [])),
            self_regard=float(data.get("self_regard", 0.0)),
            identity_stability=float(data.get("identity_stability", 0.5)),
            recent_discoveries=discoveries,
            last_updated_ts=float(data.get("last_updated_ts", time.time())),
        )

    def __repr__(self) -> str:
        return self.summary()
