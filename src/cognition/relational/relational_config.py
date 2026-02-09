"""Tuning knobs for the relational-field subsystem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelationalConfig:
    """Configuration for the relational field subsystem.

    All fields have sensible defaults; override via ``CognitiveConfig``.
    """

    # ── Relationship quality dynamics ────────────────────────────────
    #: How much a single positive turn moves felt_quality (capped at 1.0)
    positive_quality_delta: float = 0.03
    #: How much a single negative turn moves felt_quality (capped at -1.0)
    negative_quality_delta: float = 0.05
    #: Neutral turns drift felt_quality toward 0 by this factor
    neutral_quality_decay: float = 0.005

    # ── Attachment strength ──────────────────────────────────────────
    #: Each interaction grows attachment by this amount (capped at 1.0)
    attachment_growth_per_turn: float = 0.005
    #: Long idle periods decay attachment toward 0 by this rate per hour
    attachment_decay_per_hour: float = 0.001
    #: Minimum interactions before a relationship is considered "significant"
    significant_interaction_threshold: int = 5

    # ── Drive effect integration ─────────────────────────────────────
    #: How strongly relationship-level drive effects blend into
    #: the global drive modulation (0 = no effect, 1 = full effect)
    drive_effect_weight: float = 0.5
    #: Smoothing factor for exponential-moving-average of per-turn drive
    #: impacts aggregated into the relationship model
    drive_effect_ema_alpha: float = 0.15

    # ── Storage / limits ─────────────────────────────────────────────
    #: Maximum number of relational models to track
    max_relationships: int = 50
    #: Maximum recurring patterns stored per relationship
    max_patterns: int = 10
    #: Maximum gifts stored per relationship
    max_gifts: int = 10
    #: Maximum significant moment IDs stored per relationship
    max_significant_moments: int = 20
