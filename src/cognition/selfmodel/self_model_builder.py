"""SelfModelBuilder — construct a biased, partial self-model.

The builder takes the actual pattern field, drive state, and felt
sense, then produces a ``SelfModel`` that is biased by current mood:

* **Negative mood** → overweights negative patterns, underweights
  positive ones.  Self-regard drops.
* **Positive mood** → mild positive bias.
* **Neutral mood** → perception ≈ reality.

Blind spots are identified by **recency and salience**, not randomness
(the architecture doc used ``random.random()`` but the feasibility
analysis recommended deterministic, recency-based masking).

Self-discovery moments fire when there's a large discrepancy between
actual and previously perceived pattern strength AND the pattern was
recently activated (i.e., experience just revealed something).

Design principles
-----------------
* Deterministic — no randomness.
* Config-driven — all thresholds come from ``SelfModelConfig``.
* Graceful — works with partial inputs (any layer may be absent).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .self_model_config import SelfModelConfig
from .self_model_state import SelfDiscovery, SelfModel, _clamp

logger = logging.getLogger(__name__)


# ── Negative / positive pattern classification ──────────────────────
# Patterns whose names or categories suggest negative/positive valence.
# Used for mood-biased perception.

_NEGATIVE_PATTERN_KEYWORDS = frozenset({
    "pressure", "sensitivity", "tension", "negative",
    "caution", "anxiety", "stress", "conflict",
})

_POSITIVE_PATTERN_KEYWORDS = frozenset({
    "resilience", "warmth", "depth", "positive",
    "equanimity", "achievement", "curiosity", "connection",
    "independence", "meaning", "openness",
})

# ── Drive → self-narrative templates ────────────────────────────────
# When a drive is high, the agent perceives it as a need.
# When low, the agent may under-perceive how much it matters.

_DRIVE_NEED_NARRATIVES: Dict[str, Dict[str, str]] = {
    "connection": {
        "high": "I need connection and belonging",
        "moderate": "Connection matters to me",
        "low": "I'm fairly self-sufficient socially",
    },
    "competence": {
        "high": "I need to feel effective and capable",
        "moderate": "Being competent matters to me",
        "low": "I don't worry much about performance",
    },
    "autonomy": {
        "high": "I need freedom and self-direction",
        "moderate": "Having choices matters to me",
        "low": "I'm flexible about direction",
    },
    "understanding": {
        "high": "I need to understand and make sense of things",
        "moderate": "Understanding matters to me",
        "low": "I'm comfortable with some ambiguity",
    },
    "meaning": {
        "high": "I need to feel my work has purpose",
        "moderate": "Purpose matters to me",
        "low": "I don't dwell much on meaning",
    },
}

# ── Value derivation from patterns ──────────────────────────────────

_PATTERN_VALUE_MAP: Dict[str, str] = {
    "curiosity_seeking": "intellectual growth",
    "connection_seeking": "meaningful relationships",
    "independence_valuing": "personal autonomy",
    "achievement_driven": "excellence and mastery",
    "meaning_orientation": "purposeful contribution",
    "stress_resilience": "steady perseverance",
    "relational_warmth": "warmth and empathy",
    "relational_depth": "deep connection",
    "emotional_depth": "emotional authenticity",
    "emotional_equanimity": "inner peace",
    "positive_orientation": "optimism",
}


# ────────────────────────────────────────────────────────────────────
#  SelfModelBuilder
# ────────────────────────────────────────────────────────────────────

class SelfModelBuilder:
    """Construct a biased, partial self-model from lower layers.

    Parameters
    ----------
    config : SelfModelConfig, optional
        Tuning knobs.  A default config is used if omitted.
    """

    def __init__(self, config: Optional[SelfModelConfig] = None) -> None:
        self.config = config or SelfModelConfig()

    # ── Main entry point ─────────────────────────────────────────────

    def build_self_model(
        self,
        *,
        pattern_field: Any = None,
        drive_state: Any = None,
        felt_sense: Any = None,
        mood: Any = None,
        existing_self_model: Optional[SelfModel] = None,
    ) -> SelfModel:
        """Build a (biased) self-model from current state.

        Parameters
        ----------
        pattern_field : PatternField, optional
            Actual emergent patterns.
        drive_state : DriveState, optional
            Current drive state.
        felt_sense : FeltSense, optional
            Current felt-sense snapshot.
        mood : Mood, optional
            Current mood (if available, used for labeling bias).
        existing_self_model : SelfModel, optional
            Previous self-model (for stability tracking and blind spot
            continuity).

        Returns
        -------
        SelfModel
            The newly constructed self-model.
        """
        cfg = self.config

        # 1. Perceive patterns (with mood bias and blind spots)
        perceived, blind_spots = self._perceive_patterns(
            pattern_field, felt_sense, existing_self_model, cfg,
        )

        # 2. Perceive drives as needs
        perceived_needs = self._perceive_needs(drive_state, cfg)

        # 3. Identify strengths and weaknesses
        strengths = self._identify_strengths(perceived, cfg)
        weaknesses = self._identify_weaknesses(perceived, cfg)

        # 4. Derive values from strong patterns
        values = self._derive_values(pattern_field, cfg)

        # 5. Compute self-regard (mood-biased)
        self_regard = self._compute_self_regard(
            perceived, felt_sense, mood, cfg,
        )

        # 6. Update identity stability
        identity_stability = self._update_stability(
            perceived, existing_self_model, cfg,
        )

        # 7. Check for self-discoveries
        discoveries = self._check_discoveries(
            pattern_field, existing_self_model, cfg,
        )
        # Carry forward recent discoveries (ring buffer, max 10)
        if existing_self_model and existing_self_model.recent_discoveries:
            prior = list(existing_self_model.recent_discoveries)
        else:
            prior = []
        all_discoveries = prior + discoveries
        all_discoveries = all_discoveries[-cfg.max_recent_discoveries:]

        return SelfModel(
            perceived_patterns=perceived,
            perceived_needs=perceived_needs,
            perceived_strengths=strengths,
            perceived_weaknesses=weaknesses,
            stated_values=values,
            _blind_spots=blind_spots,
            self_regard=self_regard,
            identity_stability=identity_stability,
            recent_discoveries=all_discoveries,
            last_updated_ts=time.time(),
        )

    # ── Pattern perception (mood-biased + blind spots) ───────────────

    def _perceive_patterns(
        self,
        pattern_field: Any,
        felt_sense: Any,
        existing_self_model: Optional[SelfModel],
        cfg: SelfModelConfig,
    ) -> tuple[Dict[str, float], List[str]]:
        """Perceive patterns with mood bias and blind spots.

        Returns (perceived_patterns dict, blind_spots list).
        """
        perceived: Dict[str, float] = {}
        blind_spots: List[str] = []

        if pattern_field is None:
            return perceived, blind_spots

        # Get actual patterns
        patterns = getattr(pattern_field, "patterns", [])
        if not patterns:
            return perceived, blind_spots

        # Determine mood bias
        valence = 0.0
        if felt_sense is not None:
            valence = getattr(felt_sense, "felt_valence", 0.0)

        for pattern in patterns:
            name = pattern.name
            actual_strength = pattern.strength

            # Step 1: Apply mood bias to perception
            perceived_strength = self._apply_mood_bias(
                actual_strength, name, valence, cfg,
            )

            # Step 2: Check for blind spots (recency/salience-based)
            if self._is_blind_spot_candidate(pattern, existing_self_model, cfg):
                blind_spots.append(name)
                perceived_strength *= cfg.blind_spot_perception_factor

            perceived[name] = _clamp(perceived_strength, 0.0, 1.0)

        # Limit to max_perceived_patterns (keep strongest)
        if len(perceived) > cfg.max_perceived_patterns:
            sorted_pairs = sorted(
                perceived.items(), key=lambda kv: kv[1], reverse=True
            )
            perceived = dict(sorted_pairs[: cfg.max_perceived_patterns])

        # Limit blind spots
        blind_spots = blind_spots[: cfg.max_blind_spots]

        return perceived, blind_spots

    def _apply_mood_bias(
        self,
        actual_strength: float,
        pattern_name: str,
        valence: float,
        cfg: SelfModelConfig,
    ) -> float:
        """Apply mood-biased distortion to pattern perception."""
        is_negative = self._is_negative_pattern_name(pattern_name)
        is_positive = self._is_positive_pattern_name(pattern_name)

        biased_strength = actual_strength

        if valence < cfg.negative_bias_threshold:
            # Negative mood → overweight negative, underweight positive
            if is_negative:
                biased_strength = actual_strength * cfg.negative_overweight
            elif is_positive:
                biased_strength = actual_strength * cfg.negative_underweight
        elif valence > cfg.positive_bias_threshold:
            # Positive mood → mild positive bias
            if is_positive:
                biased_strength = actual_strength * cfg.positive_overweight
            elif is_negative:
                biased_strength = actual_strength * cfg.positive_underweight

        return _clamp(biased_strength, 0.0, 1.0)

    def _is_blind_spot_candidate(
        self,
        pattern: Any,
        existing_self_model: Optional[SelfModel],
        cfg: SelfModelConfig,
    ) -> bool:
        """Determine if a pattern is a blind-spot candidate.

        Uses recency and salience, NOT randomness:
        - Patterns never previously acknowledged tend to stay blind spots.
        - Patterns not recently activated are harder to see.
        - Patterns with very few activations are less salient.
        """
        name = pattern.name

        # Criterion 1: Pattern not in previous self-model's perceived set
        if existing_self_model is not None:
            previously_perceived = existing_self_model.perceived_patterns
            if name not in previously_perceived:
                # Never noticed → strong blind-spot candidate
                # But only if the pattern isn't very strong (strong patterns
                # eventually break through)
                if pattern.strength < 0.5:
                    return True

        # Criterion 2: Not recently activated
        hours_since = pattern.hours_since_activation()
        if hours_since > cfg.blind_spot_recency_hours:
            return True

        # Criterion 3: Very few activations (under-observed)
        if pattern.activation_count < cfg.blind_spot_low_activation_threshold:
            # Negative patterns about ourselves are especially prone
            if self._is_negative_pattern_name(name):
                return True

        return False

    # ── Drive perception ─────────────────────────────────────────────

    def _perceive_needs(
        self,
        drive_state: Any,
        cfg: SelfModelConfig,
    ) -> Dict[str, str]:
        """Generate self-narrative about perceived needs from drives.

        The agent may *misperceive* its own needs — a low drive level
        might lead to "I don't need much X" even when the baseline
        suggests otherwise.
        """
        needs: Dict[str, str] = {}

        if drive_state is None:
            return needs

        drive_names = ["connection", "competence", "autonomy",
                       "understanding", "meaning"]

        for drive_name in drive_names:
            level = drive_state.get_level(drive_name)
            templates = _DRIVE_NEED_NARRATIVES.get(drive_name, {})

            if level >= 0.6:
                needs[drive_name] = templates.get(
                    "high", f"I have a strong need for {drive_name}"
                )
            elif level >= 0.35:
                needs[drive_name] = templates.get(
                    "moderate", f"{drive_name.title()} matters to me"
                )
            else:
                # Low drive → agent may under-perceive how much it matters
                needs[drive_name] = templates.get(
                    "low", f"I'm fairly neutral about {drive_name}"
                )

        return needs

    # ── Strengths / weaknesses ───────────────────────────────────────

    def _identify_strengths(
        self,
        perceived: Dict[str, float],
        cfg: SelfModelConfig,
    ) -> List[str]:
        """Identify top perceived strengths (positive-valence patterns)."""
        candidates = [
            (name, strength)
            for name, strength in perceived.items()
            if self._is_positive_pattern_name(name) and strength >= cfg.strength_threshold
        ]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in candidates[: cfg.max_strengths]]

    def _identify_weaknesses(
        self,
        perceived: Dict[str, float],
        cfg: SelfModelConfig,
    ) -> List[str]:
        """Identify perceived weaknesses (negative patterns above threshold)."""
        candidates = [
            (name, strength)
            for name, strength in perceived.items()
            if self._is_negative_pattern_name(name) and strength >= cfg.weakness_threshold
        ]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in candidates[: cfg.max_weaknesses]]

    # ── Value derivation ─────────────────────────────────────────────

    def _derive_values(
        self,
        pattern_field: Any,
        cfg: SelfModelConfig,
    ) -> List[str]:
        """Derive stated values from strong actual patterns."""
        values: List[str] = []

        if pattern_field is None:
            return values

        # Use actual patterns (not perceived) for value derivation —
        # values come from behavior, not self-perception
        strong = sorted(
            pattern_field.active_patterns(min_strength=cfg.strength_threshold),
            key=lambda pattern: pattern.strength,
            reverse=True,
        )
        for pattern in strong:
            value = _PATTERN_VALUE_MAP.get(pattern.name)
            if value and value not in values:
                values.append(value)
            if len(values) >= cfg.max_values:
                break

        return values

    # ── Self-regard computation ──────────────────────────────────────

    def _compute_self_regard(
        self,
        perceived: Dict[str, float],
        felt_sense: Any,
        mood: Any,
        cfg: SelfModelConfig,
    ) -> float:
        """Compute mood-biased self-regard.

        Combines a pattern-derived component (balance of positive vs.
        negative perceived patterns) with a mood-derived component.
        """
        # Pattern component: positive strengths vs negative weakenesses
        positive_sum = sum(
            s for n, s in perceived.items()
            if self._is_positive_pattern_name(n)
        )
        negative_sum = sum(
            s for n, s in perceived.items()
            if self._is_negative_pattern_name(n)
        )
        total = positive_sum + negative_sum
        if total > 0:
            pattern_regard = (positive_sum - negative_sum) / total
        else:
            pattern_regard = 0.0

        # Mood component: felt valence directly influences self-regard
        mood_regard = 0.0
        if felt_sense is not None:
            mood_regard = getattr(felt_sense, "felt_valence", 0.0) * 0.5
        elif mood is not None:
            mood_regard = getattr(mood, "valence", 0.0) * 0.5

        regard = (
            pattern_regard * cfg.self_regard_pattern_weight
            + mood_regard * cfg.self_regard_mood_weight
        )

        return _clamp(regard, -1.0, 1.0)

    # ── Identity stability tracking ──────────────────────────────────

    def _update_stability(
        self,
        perceived: Dict[str, float],
        existing_self_model: Optional[SelfModel],
        cfg: SelfModelConfig,
    ) -> float:
        """Update identity stability.

        Stability recovers slowly over time and drops when the
        self-model changes significantly.
        """
        if existing_self_model is None:
            return 0.5  # initial

        stability = existing_self_model.identity_stability

        # Recovery toward 1.0
        stability += cfg.identity_stability_recovery_rate

        # Penalty for significant changes
        prev = existing_self_model.perceived_patterns
        changes = 0
        for name, strength in perceived.items():
            prev_strength = prev.get(name, 0.0)
            if abs(strength - prev_strength) > 0.1:
                changes += 1

        # New patterns appearing or old ones disappearing
        new_patterns = set(perceived) - set(prev)
        lost_patterns = set(prev) - set(perceived)
        changes += len(new_patterns) + len(lost_patterns)

        stability -= changes * cfg.identity_stability_pattern_change_penalty

        return _clamp(stability, 0.0, 1.0)

    # ── Self-discovery detection ─────────────────────────────────────

    def _check_discoveries(
        self,
        pattern_field: Any,
        existing_self_model: Optional[SelfModel],
        cfg: SelfModelConfig,
    ) -> List[SelfDiscovery]:
        """Check for self-discovery moments.

        A discovery fires when:
        1. The discrepancy between actual and perceived strength exceeds
           the threshold.
        2. The pattern was recently activated (experience just made it
           salient).
        """
        discoveries: List[SelfDiscovery] = []

        if pattern_field is None or existing_self_model is None:
            return discoveries

        prev_perceived = existing_self_model.perceived_patterns

        for pattern in getattr(pattern_field, "patterns", []):
            actual = pattern.strength
            perceived = prev_perceived.get(pattern.name, 0.0)
            discrepancy = abs(actual - perceived)

            if discrepancy < cfg.discovery_discrepancy_threshold:
                continue

            # Only discover if pattern was recently activated
            # (experience just revealed it)
            if pattern.hours_since_activation() > 2.0:
                continue

            if actual > perceived:
                message = (
                    f"I'm realizing I might be more "
                    f"{self._human_readable_name(pattern.name)} "
                    f"than I thought..."
                )
            else:
                message = (
                    f"I thought I was quite "
                    f"{self._human_readable_name(pattern.name)}, "
                    f"but maybe less so than I believed..."
                )

            discoveries.append(SelfDiscovery(
                pattern_name=pattern.name,
                actual_strength=actual,
                perceived_strength=perceived,
                message=message,
            ))

        return discoveries

    # ── Context summary for LLM injection ────────────────────────────

    @staticmethod
    def self_model_context_summary(self_model: SelfModel) -> str:
        """Generate a context string for LLM prompt injection.

        This summary is written from the agent's first-person
        perspective and only includes what the agent perceives
        (blind spots are excluded).
        """
        parts: List[str] = []

        # Self-description (first-person)
        desc = self_model.self_description()
        if desc:
            parts.append(desc)

        # Perceived needs
        if self_model.perceived_needs:
            need_phrases = [v for v in self_model.perceived_needs.values()]
            if need_phrases:
                parts.append(
                    "My current needs: " + "; ".join(need_phrases[:3])
                )

        # Self-regard
        if self_model.self_regard >= 0.3:
            parts.append("I feel positively about myself right now")
        elif self_model.self_regard <= -0.3:
            parts.append(
                "I'm in a self-critical headspace right now"
            )

        # Identity stability
        if self_model.identity_stability < 0.3:
            parts.append("My sense of self feels shaky right now")
        elif self_model.identity_stability > 0.7:
            parts.append("I have a stable sense of who I am")

        # Recent discovery
        if self_model.recent_discoveries:
            latest = self_model.recent_discoveries[-1]
            parts.append(f"Recent insight: {latest.message}")

        return " | ".join(parts) if parts else ""

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_negative_pattern_name(name: str) -> bool:
        """Check if a pattern name suggests negative valence."""
        name_lower = name.lower()
        return any(kw in name_lower for kw in _NEGATIVE_PATTERN_KEYWORDS)

    @staticmethod
    def _is_positive_pattern_name(name: str) -> bool:
        """Check if a pattern name suggests positive valence."""
        name_lower = name.lower()
        return any(kw in name_lower for kw in _POSITIVE_PATTERN_KEYWORDS)

    @staticmethod
    def _human_readable_name(pattern_name: str) -> str:
        """Convert snake_case pattern name to human-readable form."""
        return pattern_name.replace("_", " ")
