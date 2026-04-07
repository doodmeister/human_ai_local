"""SelfModelConfig — tuning knobs for the self-model system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfModelConfig:
    """Configuration for the self-model system (Phase 2, Layer 4).

    Attributes
    ----------
    update_interval : int
        Rebuild the self-model every N turns.
    negative_bias_threshold : float
        Felt-valence below which negative mood bias activates.
    negative_overweight : float
        Multiplier for negative-pattern strength in negative mood.
    negative_underweight : float
        Multiplier for positive-pattern strength in negative mood.
    positive_bias_threshold : float
        Felt-valence above which positive mood bias activates.
    positive_overweight : float
        Multiplier for positive-pattern strength in positive mood.
    positive_underweight : float
        Multiplier for negative-pattern strength in positive mood.
    blind_spot_recency_hours : float
        Patterns not activated in this many hours are blind-spot
        candidates (recency-based, NOT random).
    blind_spot_low_activation_threshold : int
        Patterns with fewer activations than this are blind-spot
        candidates (they haven't been noticed enough).
    blind_spot_perception_factor : float
        Multiplier applied to strength of blind-spot patterns
        in ``perceived_patterns`` (under-perceive them).
    strength_top_n : int
        Number of top patterns to consider for strengths list.
    strength_threshold : float
        Minimum perceived strength for a positive pattern to count as a strength.
    weakness_threshold : float
        Perceived strength below which a pattern counts as a weakness.
    discovery_discrepancy_threshold : float
        Minimum |actual - perceived| to trigger a self-discovery moment.
    identity_stability_recovery_rate : float
        How much identity_stability recovers per update cycle toward 1.0.
    identity_stability_pattern_change_penalty : float
        Stability penalty per significant pattern change detected.
    self_regard_pattern_weight : float
        Weight of pattern-derived self-regard vs. mood-based.
    self_regard_mood_weight : float
        Weight of mood-based self-regard.
    max_perceived_patterns : int
        Maximum number of patterns to track in self-model.
    max_strengths : int
        Maximum number of perceived strengths to list.
    max_weaknesses : int
        Maximum number of perceived weaknesses to list.
    max_values : int
        Maximum number of stated values.
    max_blind_spots : int
        Maximum number of tracked blind spots.
    max_recent_discoveries : int
        Maximum number of recent self-discovery moments to retain.
    """

    update_interval: int = 15
    negative_bias_threshold: float = -0.3
    negative_overweight: float = 1.3
    negative_underweight: float = 0.7
    positive_bias_threshold: float = 0.3
    positive_overweight: float = 1.2
    positive_underweight: float = 0.8
    blind_spot_recency_hours: float = 48.0
    blind_spot_low_activation_threshold: int = 3
    blind_spot_perception_factor: float = 0.3
    strength_top_n: int = 5
    strength_threshold: float = 0.1
    weakness_threshold: float = 0.15
    discovery_discrepancy_threshold: float = 0.3
    identity_stability_recovery_rate: float = 0.02
    identity_stability_pattern_change_penalty: float = 0.05
    self_regard_pattern_weight: float = 0.6
    self_regard_mood_weight: float = 0.4
    max_perceived_patterns: int = 20
    max_strengths: int = 5
    max_weaknesses: int = 5
    max_values: int = 5
    max_blind_spots: int = 10
    max_recent_discoveries: int = 10
