"""FeltSenseConfig — tuning knobs for the felt-sense subsystem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeltSenseConfig:
    """Configuration for FeltSenseGenerator and MoodLabeler.

    Attributes
    ----------
    high_pressure_quality_threshold : float
        Drive pressure above which negative felt qualities are generated.
    low_pressure_quality_threshold : float
        Drive pressure below which positive felt qualities dominate.
    max_qualities : int
        Maximum number of quality words per snapshot.
    history_size : int
        How many FeltSense snapshots to retain.
    mood_confidence_high : float
        Confidence assigned when felt sense is intense and clear.
    mood_confidence_low : float
        Confidence assigned when felt sense is faint or mixed.
    enable_mislabeling : bool
        If True the MoodLabeler can occasionally produce an inaccurate
        label, modeling human self-opacity.  Set False for deterministic
        testing.
    mislabel_probability : float
        Probability of mislabeling when ``enable_mislabeling`` is True.
    """

    high_pressure_quality_threshold: float = 0.7
    low_pressure_quality_threshold: float = 0.3
    max_qualities: int = 3
    history_size: int = 20
    mood_confidence_high: float = 0.85
    mood_confidence_low: float = 0.35
    enable_mislabeling: bool = False
    mislabel_probability: float = 0.10
