"""EmergentPattern and PatternField data structures.

A pattern is a behavioral/cognitive tendency that emerges from
experience — not predefined.  Patterns strengthen when repeatedly
activated and weaken when dormant.

The Big Five personality dimensions are offered as a *description
layer* — a way to interpret patterns in familiar terms.  They are
NOT the underlying representation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Big Five dimensions (description layer only)
BIG_FIVE_DIMENSIONS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

# Pattern source categories
PATTERN_CATEGORIES: tuple[str, ...] = (
    "drive_pattern",
    "coping_pattern",
    "relational_pattern",
    "felt_sense_pattern",
    "conflict_pattern",
    "procedural_pattern",
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ────────────────────────────────────────────────────────────────────
#  EmergentPattern
# ────────────────────────────────────────────────────────────────────

@dataclass
class EmergentPattern:
    """A behavioral/cognitive pattern that has emerged from experience.

    Attributes
    ----------
    name : str
        Unique identifier (snake_case, e.g. ``"curiosity_seeking"``).
    description : str
        Human-readable description.
    strength : float
        How established this pattern is (0.0 to 1.0).
    category : str
        Source category (drive_pattern, coping_pattern, etc.).
    context_triggers : list[str]
        Situations that activate this pattern.
    behavioral_tendencies : list[str]
        Observable behaviors this pattern produces.
    big_five_facet : str or None
        Which Big Five dimension this pattern maps to (if any).
    big_five_loading : float
        Direction and magnitude of Big Five mapping (-1 to +1).
    first_observed_ts : float
        When the pattern was first detected (epoch seconds).
    last_activated_ts : float
        When the pattern was last reinforced (epoch seconds).
    activation_count : int
        How many times the pattern has been detected/reinforced.
    """

    name: str
    description: str = ""
    strength: float = 0.1
    category: str = "drive_pattern"
    context_triggers: List[str] = field(default_factory=list)
    behavioral_tendencies: List[str] = field(default_factory=list)
    big_five_facet: Optional[str] = None
    big_five_loading: float = 0.0
    first_observed_ts: float = field(default_factory=time.time)
    last_activated_ts: float = field(default_factory=time.time)
    activation_count: int = 1

    def __post_init__(self) -> None:
        self.strength = _clamp(self.strength, 0.0, 1.0)
        if self.big_five_facet and self.big_five_facet not in BIG_FIVE_DIMENSIONS:
            self.big_five_facet = None
            self.big_five_loading = 0.0

    # ── Mutation ─────────────────────────────────────────────────────

    def activate(self, boost: float = 0.01) -> None:
        """Reinforce this pattern."""
        self.strength = _clamp(self.strength + boost, 0.0, 1.0)
        self.last_activated_ts = time.time()
        self.activation_count += 1

    def decay(self, amount: float = 0.005) -> None:
        """Weaken this pattern from inactivity."""
        self.strength = _clamp(self.strength - amount, 0.0, 1.0)

    # ── Queries ──────────────────────────────────────────────────────

    def hours_since_activation(self) -> float:
        """Hours since this pattern was last activated."""
        return (time.time() - self.last_activated_ts) / 3600.0

    def hours_since_observed(self) -> float:
        """Hours since this pattern was first observed."""
        return (time.time() - self.first_observed_ts) / 3600.0

    # ── Display ──────────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"Pattern({self.name} str={self.strength:.2f} "
            f"cat={self.category} acts={self.activation_count})"
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "strength": round(self.strength, 4),
            "category": self.category,
            "context_triggers": list(self.context_triggers),
            "behavioral_tendencies": list(self.behavioral_tendencies),
            "big_five_facet": self.big_five_facet,
            "big_five_loading": round(self.big_five_loading, 4),
            "first_observed_ts": self.first_observed_ts,
            "last_activated_ts": self.last_activated_ts,
            "activation_count": self.activation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmergentPattern":
        return cls(
            name=str(data.get("name", "unknown")),
            description=str(data.get("description", "")),
            strength=float(data.get("strength", 0.1)),
            category=str(data.get("category", "drive_pattern")),
            context_triggers=list(data.get("context_triggers", [])),
            behavioral_tendencies=list(data.get("behavioral_tendencies", [])),
            big_five_facet=data.get("big_five_facet"),
            big_five_loading=float(data.get("big_five_loading", 0.0)),
            first_observed_ts=float(data.get("first_observed_ts", time.time())),
            last_activated_ts=float(data.get("last_activated_ts", time.time())),
            activation_count=int(data.get("activation_count", 1)),
        )

    def __repr__(self) -> str:
        return self.summary()


# ────────────────────────────────────────────────────────────────────
#  PatternField
# ────────────────────────────────────────────────────────────────────

@dataclass
class PatternField:
    """Collection of all emergent patterns.

    Patterns live here and are managed by the ``PatternDetector``.
    The Big Five description is derived on demand, not stored.
    """

    patterns: List[EmergentPattern] = field(default_factory=list)
    max_patterns: int = 30

    # ── Lookup ───────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[EmergentPattern]:
        """Find a pattern by name."""
        for p in self.patterns:
            if p.name == name:
                return p
        return None

    # ── Mutation ─────────────────────────────────────────────────────

    def add_or_strengthen(
        self,
        pattern: EmergentPattern,
        strengthen_boost: float = 0.01,
    ) -> EmergentPattern:
        """Add a new pattern or strengthen an existing one.

        Returns the pattern (new or existing).
        """
        existing = self.get(pattern.name)
        if existing is not None:
            existing.activate(strengthen_boost)
            return existing

        # Add new (respect capacity)
        if len(self.patterns) >= self.max_patterns:
            self._prune_weakest()
        self.patterns.append(pattern)
        return pattern

    def weaken_inactive(
        self,
        threshold_hours: float = 24.0,
        decay: float = 0.005,
    ) -> int:
        """Weaken patterns not recently activated. Returns count weakened."""
        count = 0
        for p in self.patterns:
            if p.hours_since_activation() > threshold_hours:
                p.decay(decay)
                count += 1
        return count

    def prune_weak(self, min_strength: float = 0.02) -> int:
        """Remove patterns below minimum strength. Returns count pruned."""
        before = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.strength >= min_strength]
        return before - len(self.patterns)

    # ── Queries ──────────────────────────────────────────────────────

    def active_patterns(self, min_strength: float = 0.05) -> List[EmergentPattern]:
        """Return patterns above minimum strength, sorted by strength desc."""
        return sorted(
            [p for p in self.patterns if p.strength >= min_strength],
            key=lambda p: p.strength,
            reverse=True,
        )

    def dominant_patterns(self, n: int = 3) -> List[EmergentPattern]:
        """Return the *n* strongest patterns."""
        return self.active_patterns()[:n]

    def patterns_by_category(self, category: str) -> List[EmergentPattern]:
        """Return all patterns in a given category."""
        return [p for p in self.patterns if p.category == category]

    def describe_as_big_five(self) -> Dict[str, float]:
        """Map emergent patterns to Big Five scores (description layer).

        This is an INTERPRETATION, not the underlying representation.
        Scores are accumulated from pattern strengths and their
        ``big_five_loading`` values.

        Returns dict with openness, conscientiousness, extraversion,
        agreeableness, neuroticism — each clamped to [-1, 1].
        """
        scores = {dim: 0.0 for dim in BIG_FIVE_DIMENSIONS}

        for p in self.patterns:
            if p.big_five_facet and p.big_five_facet in scores:
                scores[p.big_five_facet] += p.strength * p.big_five_loading

        return {k: _clamp(v, -1.0, 1.0) for k, v in scores.items()}

    def count(self) -> int:
        return len(self.patterns)

    # ── Internal ─────────────────────────────────────────────────────

    def _prune_weakest(self) -> None:
        """Remove the weakest pattern to make room."""
        if not self.patterns:
            return
        weakest = min(self.patterns, key=lambda p: p.strength)
        self.patterns.remove(weakest)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, object]:
        return {
            "patterns": [p.to_dict() for p in self.patterns],
            "big_five": self.describe_as_big_five(),
            "count": self.count(),
            "dominant": [p.name for p in self.dominant_patterns()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternField":
        patterns_data = data.get("patterns", [])
        return cls(
            patterns=[EmergentPattern.from_dict(d) for d in patterns_data],
        )
