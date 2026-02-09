"""Mood — labeled emotional state derived from FeltSense.

The ``Mood`` dataclass carries the named emotion plus an explicit
``confidence`` score.  ``MoodLabeler`` converts a ``FeltSense`` into a
``Mood`` using rule-based pattern matching, optionally introducing
mislabeling to model self-opacity.

Design notes
-------------
* The labeling rules are intentionally simple — the architecture doc
  notes that "the agent might mislabel what it's feeling, just like
  humans do."  Sophistication comes from the *layered* approach
  (drives → felt sense → mood), not from a complex classifier here.
* Mood labels are drawn from a fixed vocabulary so downstream prompt
  engineering can rely on consistent strings.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .felt_sense_config import FeltSenseConfig
from .felt_sense_state import FeltSense

logger = logging.getLogger(__name__)

# ── Mood label vocabulary ───────────────────────────────────────────
MOOD_LABELS = (
    "content",
    "excited",
    "curious",
    "calm",
    "anxious",
    "sad",
    "lonely",
    "frustrated",
    "restless",
    "uncertain",
)


@dataclass
class Mood:
    """Labeled emotional state derived from a FeltSense snapshot.

    Attributes
    ----------
    label : str
        Named mood (from ``MOOD_LABELS``).
    valence : float
        -1 to 1 (mirrors felt_valence but may drift due to labeling).
    arousal : float
        0 to 1 (maps from felt-sense intensity).
    felt_sense : FeltSense
        The underlying pre-conceptual state.
    confidence : float
        0 to 1 — how sure the agent is about this label.
    since : datetime
        When the mood was labeled.
    """

    label: str = "calm"
    valence: float = 0.0
    arousal: float = 0.3
    felt_sense: Optional[FeltSense] = None
    confidence: float = 0.6
    since: datetime = field(default_factory=datetime.now)

    def describe(self) -> str:
        """Natural-language self-report, gated by confidence."""
        if self.confidence >= 0.7:
            return f"I'm feeling {self.label}"
        elif self.confidence >= 0.45:
            return f"I think I might be feeling {self.label}, but I'm not sure"
        return "I'm not sure what I'm feeling right now"

    def summary(self) -> str:
        return (
            f"Mood({self.label} v={self.valence:+.2f} "
            f"a={self.arousal:.2f} conf={self.confidence:.2f})"
        )

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "label": self.label,
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "confidence": round(self.confidence, 4),
            "description": self.describe(),
            "since": self.since.isoformat(),
        }
        if self.felt_sense is not None:
            result["felt_sense"] = self.felt_sense.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Mood":
        fs_data = data.get("felt_sense")
        fs = FeltSense.from_dict(fs_data) if isinstance(fs_data, dict) else None  # type: ignore[arg-type]
        since = data.get("since")
        if isinstance(since, str):
            try:
                since = datetime.fromisoformat(since)
            except (ValueError, TypeError):
                since = datetime.now()
        else:
            since = datetime.now()
        return cls(
            label=str(data.get("label", "calm")),
            valence=float(data.get("valence", 0.0)),  # type: ignore[arg-type]
            arousal=float(data.get("arousal", 0.3)),  # type: ignore[arg-type]
            felt_sense=fs,
            confidence=float(data.get("confidence", 0.6)),  # type: ignore[arg-type]
            since=since,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return self.summary()


# ── Labeler ─────────────────────────────────────────────────────────

class MoodLabeler:
    """Convert a FeltSense into a labeled Mood.

    The labeling can be imperfect when ``config.enable_mislabeling``
    is set, modelling the human tendency to mislabel one's own emotions.
    """

    def __init__(self, config: Optional[FeltSenseConfig] = None) -> None:
        self.config = config or FeltSenseConfig()

    def label_mood(self, felt: FeltSense) -> Mood:
        """Produce a Mood from a FeltSense snapshot."""
        label = self._classify(felt)
        confidence = self._compute_confidence(felt)

        # Optional mislabeling (self-opacity)
        if (
            self.config.enable_mislabeling
            and random.random() < self.config.mislabel_probability
        ):
            label = self._mislabel(label, felt)
            confidence *= 0.6  # lower confidence when mislabeled

        return Mood(
            label=label,
            valence=felt.felt_valence,
            arousal=felt.intensity,
            felt_sense=felt,
            confidence=round(confidence, 4),
            since=datetime.now(),
        )

    # ── Classification rules ─────────────────────────────────────────

    @staticmethod
    def _classify(felt: FeltSense) -> str:
        """Deterministic rule-based classification."""
        qs = set(felt.qualities)
        v = felt.felt_valence
        mov = felt.movement

        # Negative labels
        if "heavy" in qs and v < -0.3:
            return "sad"
        if ("tight" in qs or "sharp" in qs) and mov == "contracting":
            return "anxious"
        if "hollow" in qs or "aching" in qs:
            return "lonely"
        if "tight" in qs and v < -0.1:
            return "frustrated"
        if "churning" in qs or "constricted" in qs:
            return "restless"

        # Positive labels
        if "warm" in qs and "open" in qs:
            return "content"
        if ("buzzing" in qs or "bright" in qs) and v > 0.3:
            return "excited"
        if "foggy" not in qs and v > 0.1 and mov in ("expanding", "rising"):
            return "curious"
        if ("calm" in qs or "soft" in qs) and v >= 0.0:
            return "calm"

        # Ambiguous / unclear
        if "foggy" in qs:
            return "uncertain"

        # Default based on valence
        if v > 0.2:
            return "calm"
        if v < -0.2:
            return "uncertain"
        return "uncertain"

    def _compute_confidence(self, felt: FeltSense) -> float:
        """How confident is the agent in the label?"""
        cfg = self.config
        # Strong, clear (few qualities) + high intensity → high confidence
        if felt.intensity >= 0.7 and len(felt.qualities) <= 2:
            return cfg.mood_confidence_high
        if felt.intensity < 0.25:
            return cfg.mood_confidence_low
        # Mixed / moderate → mid confidence
        return 0.6

    @staticmethod
    def _mislabel(correct_label: str, felt: FeltSense) -> str:
        """Return a plausible but incorrect label (self-opacity)."""
        # Swap to a label from the same arousal family
        low_arousal_labels = ("calm", "sad", "lonely", "content")
        high_arousal_labels = ("anxious", "excited", "frustrated", "restless")

        if correct_label in low_arousal_labels:
            candidates = [l for l in low_arousal_labels if l != correct_label]
        elif correct_label in high_arousal_labels:
            candidates = [l for l in high_arousal_labels if l != correct_label]
        else:
            candidates = ["uncertain"]

        return random.choice(candidates) if candidates else "uncertain"

    # ── Context generation ───────────────────────────────────────────

    @staticmethod
    def mood_context_summary(mood: Mood) -> str:
        """Produce LLM-injectable context string for a mood state."""
        lines = [
            f"Current mood: {mood.label} (confidence: {mood.confidence:.0%})",
            f"  Valence: {mood.valence:+.2f}, Arousal: {mood.arousal:.2f}",
        ]
        if mood.felt_sense is not None:
            lines.append(f"  Felt sense: {mood.felt_sense.describe()}")
        lines.append(f"  Self-report: \"{mood.describe()}\"")
        return "\n".join(lines)
