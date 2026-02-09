"""DriveState — current levels for all fundamental drives.

Drive levels run from 0.0 (fully satisfied) to 1.0 (desperately unsatisfied).
Each drive also carries a *baseline* (the level it drifts toward when nothing
happens) and a *sensitivity* (how quickly it becomes unsatisfied).

Baselines and sensitivities are mutable — they shift slowly over the agent's
lifetime as experience shapes who it becomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

# Canonical drive identifiers — order matches Self-Determination Theory
# (connection ≈ relatedness, competence, autonomy) plus Understanding + Meaning.
DRIVE_NAMES: tuple[str, ...] = (
    "connection",
    "competence",
    "autonomy",
    "understanding",
    "meaning",
)


def _default_baselines() -> Dict[str, float]:
    return {d: 0.3 for d in DRIVE_NAMES}


def _default_sensitivities() -> Dict[str, float]:
    return {d: 1.0 for d in DRIVE_NAMES}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class DriveState:
    """Current state of fundamental drives.

    Attributes
    ----------
    connection, competence, autonomy, understanding, meaning :
        Current pressure level for each drive (0 = satisfied, 1 = critical).
    baselines :
        The equilibrium pressure each drive drifts toward when unstimulated.
    sensitivities :
        Multiplier on how quickly each drive becomes unsatisfied.
    last_updated :
        Timestamp of the most recent state change.
    """

    connection: float = 0.3
    competence: float = 0.3
    autonomy: float = 0.3
    understanding: float = 0.3
    meaning: float = 0.3

    baselines: Dict[str, float] = field(default_factory=_default_baselines)
    sensitivities: Dict[str, float] = field(default_factory=_default_sensitivities)
    last_updated: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_level(self, drive: str) -> float:
        """Return the current level for a named drive."""
        if drive not in DRIVE_NAMES:
            raise ValueError(f"Unknown drive: {drive!r}")
        return getattr(self, drive)

    def set_level(self, drive: str, value: float) -> None:
        """Set a drive level, clamping to [0, 1]."""
        if drive not in DRIVE_NAMES:
            raise ValueError(f"Unknown drive: {drive!r}")
        setattr(self, drive, _clamp(value))
        self.last_updated = datetime.now()

    def get_pressure(self) -> Dict[str, float]:
        """Return a snapshot of all drive pressures."""
        return {d: self.get_level(d) for d in DRIVE_NAMES}

    def dominant_drive(self) -> str:
        """Which drive is creating the most pressure right now?"""
        pressures = self.get_pressure()
        return max(pressures, key=pressures.get)  # type: ignore[arg-type]

    def total_pressure(self) -> float:
        """Overall motivational tension (mean of all drives)."""
        return sum(self.get_pressure().values()) / len(DRIVE_NAMES)

    def high_pressure_drives(self, threshold: float = 0.6) -> list[str]:
        """Return drives above the given threshold, sorted descending."""
        return sorted(
            [d for d in DRIVE_NAMES if self.get_level(d) >= threshold],
            key=lambda d: self.get_level(d),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "levels": self.get_pressure(),
            "baselines": dict(self.baselines),
            "sensitivities": dict(self.sensitivities),
            "dominant": self.dominant_drive(),
            "total_pressure": round(self.total_pressure(), 4),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DriveState":
        levels = data.get("levels", {})
        state = cls(
            connection=float(levels.get("connection", 0.3)),  # type: ignore[union-attr]
            competence=float(levels.get("competence", 0.3)),  # type: ignore[union-attr]
            autonomy=float(levels.get("autonomy", 0.3)),  # type: ignore[union-attr]
            understanding=float(levels.get("understanding", 0.3)),  # type: ignore[union-attr]
            meaning=float(levels.get("meaning", 0.3)),  # type: ignore[union-attr]
        )
        if "baselines" in data and isinstance(data["baselines"], dict):
            state.baselines.update(data["baselines"])  # type: ignore[arg-type]
        if "sensitivities" in data and isinstance(data["sensitivities"], dict):
            state.sensitivities.update(data["sensitivities"])  # type: ignore[arg-type]
        return state

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable one-liner for logs / context injection."""
        parts = [f"{d}={self.get_level(d):.2f}" for d in DRIVE_NAMES]
        return f"Drives({', '.join(parts)} | total={self.total_pressure():.2f})"

    def __repr__(self) -> str:
        return self.summary()
