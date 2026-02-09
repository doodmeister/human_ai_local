"""DriveConfig — tuning knobs for the drive system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DriveConfig:
    """Configuration for drive dynamics.

    Attributes
    ----------
    drift_rate_per_minute :
        Base rate at which drives drift toward unsatisfied per minute of
        elapsed wall-clock time.  Multiplied by the per-drive *sensitivity*.
    satisfaction_magnitude :
        Default magnitude of a satisfying experience on a drive (subtracted).
    frustration_magnitude :
        Default magnitude of a frustrating experience on a drive (added).
    implicit_shift_magnitude :
        Micro-adjustment applied via implicit learning every turn.
    high_pressure_threshold :
        A drive above this level is considered "high pressure" (influences
        attention, conflicts, etc.).
    conflict_threshold :
        Two drives must both exceed this level for an internal conflict to
        be detected.
    baseline_adaptation_rate :
        How quickly baselines shift toward chronic drive levels.
    sensitivity_adaptation_rate :
        How quickly sensitivities adjust from repeated satisfaction /
        frustration patterns.
    enable_natural_drift :
        Whether to apply time-based drift at all (disable for unit tests).
    enable_implicit_learning :
        Whether to apply per-turn micro-shifts.
    """

    drift_rate_per_minute: float = 0.005
    satisfaction_magnitude: float = 0.10
    frustration_magnitude: float = 0.12
    implicit_shift_magnitude: float = 0.02
    high_pressure_threshold: float = 0.6
    conflict_threshold: float = 0.5
    baseline_adaptation_rate: float = 0.001
    sensitivity_adaptation_rate: float = 0.0005
    enable_natural_drift: bool = True
    enable_implicit_learning: bool = True
