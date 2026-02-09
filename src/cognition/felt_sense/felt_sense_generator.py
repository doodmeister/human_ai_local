"""FeltSenseGenerator — creates a FeltSense snapshot from drive state.

The generator maps drive pressures to metaphorical qualities, location,
movement, and intensity.  It also accounts for recent turn-level valence
(from ``estimate_salience_and_valence``) to modulate the snapshot.

Design notes
-------------
* The mapping is intentionally deterministic so it remains testable.
  Randomness is reserved for the optional mislabeling in ``MoodLabeler``.
* A single FeltSenseGenerator instance is reused across turns.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from .felt_sense_config import FeltSenseConfig
from .felt_sense_state import FeltSense

logger = logging.getLogger(__name__)

# ── Drive → Quality mappings ────────────────────────────────────────
# When a specific drive pressure exceeds the high-pressure threshold,
# the associated *negative* quality is added.  When total pressure is
# low, positive qualities dominate.

_DRIVE_HIGH_QUALITY = {
    "connection": "hollow",
    "competence": "tight",
    "autonomy": "constricted",
    "understanding": "foggy",
    "meaning": "empty",
}

# Secondary modifiers when the drive is at extreme levels (>0.85)
_DRIVE_EXTREME_QUALITY = {
    "connection": "aching",
    "competence": "sharp",
    "autonomy": "churning",
    "understanding": "numb",
    "meaning": "heavy",
}

# ── Drive → metaphorical location ──────────────────────────────────
_DRIVE_LOCATION = {
    "connection": "chest",
    "competence": "stomach",
    "autonomy": "throat",
    "understanding": "head",
    "meaning": "whole body",
}


class FeltSenseGenerator:
    """Generate a FeltSense snapshot from drives and recent valence.

    Parameters
    ----------
    config : FeltSenseConfig, optional
        Tuning knobs.  A default config is created if omitted.
    """

    def __init__(self, config: Optional[FeltSenseConfig] = None) -> None:
        self.config = config or FeltSenseConfig()

    def generate(
        self,
        drives,  # DriveState — untyped to avoid hard import
        *,
        recent_valences: Optional[List[float]] = None,
    ) -> FeltSense:
        """Create a felt-sense snapshot from current drive state.

        Parameters
        ----------
        drives :
            ``DriveState`` instance (imported dynamically to avoid
            circular dependencies).
        recent_valences :
            Optional list of recent turn-level emotional valence values
            (most recent last, -1..1).  Used to modulate felt qualities.
        """
        cfg = self.config
        total_pressure = drives.total_pressure()

        # ── Determine qualities ──────────────────────────────────────
        qualities: list[str] = []

        # Check for low-pressure positive state first
        if total_pressure < cfg.low_pressure_quality_threshold:
            qualities = self._positive_qualities(total_pressure)
        else:
            # Map each high-pressure drive to a quality
            for drive_name, quality in _DRIVE_HIGH_QUALITY.items():
                level = drives.get_level(drive_name)
                if level >= cfg.high_pressure_quality_threshold:
                    qualities.append(quality)
                # Extreme levels add a second quality
                if level >= 0.85:
                    extreme_q = _DRIVE_EXTREME_QUALITY.get(drive_name)
                    if extreme_q and extreme_q not in qualities:
                        qualities.append(extreme_q)

        # Modulate from recent emotional valence
        if recent_valences:
            mean_val = sum(recent_valences[-3:]) / len(recent_valences[-3:])
            if mean_val < -0.5 and "heavy" not in qualities:
                qualities.append("heavy")
            elif mean_val > 0.5 and "light" not in qualities:
                qualities.append("light")

        # Default if nothing triggered
        if not qualities:
            qualities = ["calm"]

        # Cap at max qualities
        qualities = qualities[: cfg.max_qualities]

        # ── Location (from dominant drive) ───────────────────────────
        location = _DRIVE_LOCATION.get(drives.dominant_drive(), "chest")

        # ── Movement ─────────────────────────────────────────────────
        movement = self._determine_movement(total_pressure, recent_valences)

        # ── Intensity ────────────────────────────────────────────────
        intensity = min(1.0, total_pressure * 1.2)  # slight amplification

        # ── Felt valence ─────────────────────────────────────────────
        # Maps total_pressure 0→+1, 0.5→0, 1→-1
        felt_valence = 1.0 - (total_pressure * 2.0)
        felt_valence = max(-1.0, min(1.0, felt_valence))

        return FeltSense(
            qualities=qualities,
            intensity=round(intensity, 4),
            location=location,
            felt_valence=round(felt_valence, 4),
            movement=movement,
            since=datetime.now(),
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _positive_qualities(total_pressure: float) -> list[str]:
        """Return positive qualities for low-pressure states."""
        if total_pressure < 0.15:
            return ["open", "warm", "flowing"]
        return ["calm", "soft"]

    @staticmethod
    def _determine_movement(
        total_pressure: float,
        recent_valences: Optional[List[float]] = None,
    ) -> str:
        """Pick a movement quality from pressure and recent valence trend."""
        if total_pressure > 0.7:
            return "contracting"
        if total_pressure < 0.25:
            return "expanding"

        # Mid-range: check recent valence direction
        if recent_valences and len(recent_valences) >= 2:
            delta = recent_valences[-1] - recent_valences[-2]
            if delta > 0.2:
                return "rising"
            if delta < -0.2:
                return "sinking"

        if 0.4 < total_pressure < 0.6:
            return "pulsing"

        return "still"
