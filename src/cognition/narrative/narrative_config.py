"""NarrativeConfig — tuning knobs for the narrative system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NarrativeConfig:
    """Configuration for the narrative system (Phase 2, Layer 5).

    Attributes
    ----------
    update_interval : int
        Rebuild the narrative every N turns (at most).
    significance_threshold : float
        Minimum significance score (0-1) to trigger a narrative
        rebuild between scheduled intervals.
    identity_summary_max_tokens : int
        Approximate token budget for the identity_summary injected
        into the system prompt (~4 chars per token).
    max_chapters : int
        Maximum number of life-chapter entries to keep.
    max_themes : int
        Maximum number of active themes to track.
    max_defining_moments : int
        Maximum defining-moment IDs to retain.
    max_struggles : int
        Maximum ongoing-struggle entries.
    growth_window_turns : int
        How many turns of pattern history to use when detecting
        growth arcs ("I used to X, but now I Y").
    growth_change_threshold : float
        Minimum pattern-strength change to count as a growth arc.
    theme_recency_hours : float
        Themes older than this are pruned.
    stability_weight : float
        How much identity_stability influences narrative continuity
        vs. rewrite.  High → prefer keeping old narrative.
    """

    update_interval: int = 20
    significance_threshold: float = 0.6
    identity_summary_max_tokens: int = 150
    max_chapters: int = 5
    max_themes: int = 5
    max_defining_moments: int = 10
    max_struggles: int = 3
    growth_window_turns: int = 50
    growth_change_threshold: float = 0.15
    theme_recency_hours: float = 72.0
    stability_weight: float = 0.5
