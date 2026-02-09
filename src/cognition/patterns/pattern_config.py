"""PatternConfig — tuning knobs for the emergent patterns system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatternConfig:
    """Configuration for the emergent patterns system.

    Attributes
    ----------
    detection_interval : int
        Run pattern detection every N turns.
    strengthen_boost : float
        Strength increase when a pattern is reinforced.
    inactivity_decay : float
        Strength decrease per detection cycle for inactive patterns.
    inactivity_threshold_hours : float
        Hours after which a pattern is considered inactive.
    prune_threshold : float
        Patterns below this strength are removed.
    max_patterns : int
        Maximum number of patterns to track.
    drive_high_threshold : float
        Drive level above which a drive-based pattern is detected.
    drive_chronic_threshold : float
        Drive level considered chronically elevated (stronger signal).
    sensitivity_deviation_threshold : float
        How far sensitivity must deviate from 1.0 to trigger a pattern.
    felt_intensity_threshold : float
        Felt-sense average intensity above which depth pattern triggers.
    relational_quality_threshold : float
        Felt quality magnitude above which relational patterns trigger.
    conflict_tension_threshold : float
        Conflict tension above which conflict patterns form.
    initial_strength : float
        Starting strength for newly detected patterns.
    """

    detection_interval: int = 10
    strengthen_boost: float = 0.015
    inactivity_decay: float = 0.005
    inactivity_threshold_hours: float = 24.0
    prune_threshold: float = 0.02
    max_patterns: int = 30
    drive_high_threshold: float = 0.55
    drive_chronic_threshold: float = 0.7
    sensitivity_deviation_threshold: float = 0.3
    felt_intensity_threshold: float = 0.5
    relational_quality_threshold: float = 0.3
    conflict_tension_threshold: float = 0.4
    initial_strength: float = 0.1
