"""FeltSense and FeltSenseHistory data structures.

A ``FeltSense`` snapshot captures the pre-conceptual texture of the
agent's current state.  It is intentionally *not* a mood label — the
labeling step happens downstream in ``MoodLabeler``.

``FeltSenseHistory`` maintains a bounded ring buffer of recent snapshots
and can report whether the trend is improving, worsening, or stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ── Quality vocabulary ──────────────────────────────────────────────
# These are the metaphorical sensation words the agent can "feel".
# Divided by valence family for convenience, but they mix freely.

NEGATIVE_QUALITIES = frozenset({
    "heavy", "tight", "constricted", "foggy", "empty", "hollow",
    "sharp", "churning", "numb", "aching",
})

POSITIVE_QUALITIES = frozenset({
    "warm", "open", "light", "flowing", "buzzing", "soft",
    "calm", "bright", "expansive", "buoyant",
})

ALL_QUALITIES = NEGATIVE_QUALITIES | POSITIVE_QUALITIES

# ── Movement vocabulary ─────────────────────────────────────────────
MOVEMENTS = frozenset({
    "contracting", "expanding", "still", "churning", "flowing",
    "pulsing", "sinking", "rising",
})

# ── Location vocabulary ─────────────────────────────────────────────
LOCATIONS = frozenset({
    "chest", "stomach", "throat", "head", "whole body",
})


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class FeltSense:
    """A single snapshot of pre-conceptual felt quality.

    Attributes
    ----------
    qualities : list[str]
        Up to 3 metaphorical sensation words (e.g. ``["heavy", "tight"]``).
    intensity : float
        0.0 (barely perceptible) to 1.0 (overwhelming).
    location : str
        Metaphorical body location (``"chest"``, ``"head"``, etc.).
    felt_valence : float
        -1.0 to 1.0 — *experienced*, not computed.
    movement : str
        Qualitative movement (``"contracting"``, ``"flowing"``, etc.).
    since : datetime
        When this felt sense snapshot was created.
    """

    qualities: List[str] = field(default_factory=lambda: ["calm"])
    intensity: float = 0.3
    location: str = "chest"
    felt_valence: float = 0.0
    movement: str = "still"
    since: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.felt_valence = _clamp(self.felt_valence)
        # Cap qualities at 3
        if len(self.qualities) > 3:
            self.qualities = self.qualities[:3]

    # ── Display ──────────────────────────────────────────────────────

    def describe(self) -> str:
        """Natural-language one-liner for context injection."""
        quality_str = " and ".join(self.qualities[:2]) if self.qualities else "neutral"
        return f"A {quality_str} feeling in my {self.location}, {self.movement}"

    def summary(self) -> str:
        """Compact representation for logs."""
        qs = ",".join(self.qualities)
        return (
            f"FeltSense({qs} | intensity={self.intensity:.2f} "
            f"valence={self.felt_valence:+.2f} loc={self.location} "
            f"mov={self.movement})"
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, object]:
        return {
            "qualities": list(self.qualities),
            "intensity": round(self.intensity, 4),
            "location": self.location,
            "felt_valence": round(self.felt_valence, 4),
            "movement": self.movement,
            "since": self.since.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FeltSense":
        since = data.get("since")
        if isinstance(since, str):
            try:
                since = datetime.fromisoformat(since)
            except (ValueError, TypeError):
                since = datetime.now()
        else:
            since = datetime.now()
        return cls(
            qualities=list(data.get("qualities", ["calm"])),  # type: ignore[arg-type]
            intensity=float(data.get("intensity", 0.3)),  # type: ignore[arg-type]
            location=str(data.get("location", "chest")),
            felt_valence=float(data.get("felt_valence", 0.0)),  # type: ignore[arg-type]
            movement=str(data.get("movement", "still")),
            since=since,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class FeltSenseHistory:
    """Bounded buffer of recent FeltSense snapshots with trend detection.

    Attributes
    ----------
    current : FeltSense
        Most recent snapshot.
    recent : list[FeltSense]
        Last ``max_size`` snapshots (oldest first).
    max_size : int
        Ring-buffer capacity.
    """

    current: FeltSense = field(default_factory=FeltSense)
    recent: List[FeltSense] = field(default_factory=list)
    max_size: int = 20

    def update(self, new_sense: FeltSense) -> None:
        """Push a new felt-sense snapshot, evicting oldest if full."""
        self.recent.append(self.current)
        if len(self.recent) > self.max_size:
            self.recent = self.recent[-self.max_size:]
        self.current = new_sense

    def trend(self) -> str:
        """Is the felt sense improving, worsening, or stable?

        Compares the mean valence of the last 3 snapshots against the
        previous 3 (or the overall mean when history is short).
        """
        if len(self.recent) < 2:
            return "stable"

        def _mean_valence(snapshots: List[FeltSense]) -> float:
            if not snapshots:
                return 0.0
            return sum(fs.felt_valence for fs in snapshots) / len(snapshots)

        recent_window = self.recent[-3:]
        recent_val = _mean_valence(recent_window)

        if len(self.recent) >= 6:
            older_val = _mean_valence(self.recent[-6:-3])
        else:
            older_val = _mean_valence(self.recent[:-3] or self.recent[:1])

        diff = recent_val - older_val
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "worsening"
        return "stable"

    def average_intensity(self, window: int = 5) -> float:
        """Mean intensity over the last *window* snapshots."""
        snapshots = [self.current] + list(reversed(self.recent))
        snapshots = snapshots[:window]
        if not snapshots:
            return 0.0
        return sum(fs.intensity for fs in snapshots) / len(snapshots)

    def to_dict(self) -> Dict[str, object]:
        return {
            "current": self.current.to_dict(),
            "trend": self.trend(),
            "average_intensity": round(self.average_intensity(), 4),
            "history_length": len(self.recent),
        }
