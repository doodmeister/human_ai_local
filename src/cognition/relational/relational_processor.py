"""RelationalProcessor — updates relationship models from turn data.

The processor handles:
* **Felt quality** — shift up/down based on turn-level valence.
* **Attachment strength** — slow growth per interaction, idle decay.
* **Drive effects** — exponential moving average of per-turn drive impacts.
* **Status classification** — active / dormant / strained / growing.
* **Recurring pattern capture** — simple heuristic detection from
  ``MemoryCaptureModule`` outputs.
* **Relationship context summary** — text block injected into the LLM
  system prompt via ``ContextBuilder``.

Design notes
-------------
* Deterministic — no randomness.  All dynamics are parameterised via
  ``RelationalConfig``.
* Safe to skip — if the relational module cannot be imported, the
  ChatService and ContextBuilder degrade gracefully.
* Serialisable — ``RelationalField.to_dict()`` / ``from_dict()``
  for persistence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .relational_config import RelationalConfig
from .relational_state import RelationalField, RelationalModel, _clamp

logger = logging.getLogger(__name__)


# ── Pattern detection keywords ──────────────────────────────────────
# Simple keyword heuristics — mirrors the approach in DriveProcessor.
# Each entry maps a pattern description to keywords that trigger it.

_PATTERN_KEYWORDS: Dict[str, List[str]] = {
    "They often teach me things": [
        "learn", "explain", "teach", "show me", "how to", "tutorial",
    ],
    "We explore ideas together": [
        "think about", "wonder", "what if", "imagine", "brainstorm",
    ],
    "We work on problems together": [
        "solve", "fix", "debug", "build", "implement", "create",
    ],
    "They give helpful feedback": [
        "great", "good job", "nice", "helpful", "well done", "thanks",
    ],
    "We sometimes disagree": [
        "wrong", "disagree", "no,", "not quite", "incorrect", "actually",
    ],
    "They share personal thoughts": [
        "i feel", "i think", "personally", "in my opinion", "honestly",
    ],
    "They set clear expectations": [
        "please", "i need", "make sure", "always", "never", "must",
    ],
}

# ── Gift detection keywords ─────────────────────────────────────────
_GIFT_KEYWORDS: Dict[str, List[str]] = {
    "Patience": ["take your time", "no rush", "whenever you're ready", "that's okay"],
    "Permission to fail": ["it's fine", "don't worry", "mistakes are okay", "no problem"],
    "New perspective": ["never thought of", "interesting point", "good idea", "that's clever"],
    "Encouragement": ["you can do", "keep going", "believe in", "great work", "proud"],
    "Trust": ["i trust you", "rely on you", "count on you", "depend on"],
}


class RelationalProcessor:
    """Updates relational models from per-turn interaction data.

    Parameters
    ----------
    config : RelationalConfig, optional
        Tuning knobs.  A default config is used if omitted.
    """

    def __init__(self, config: Optional[RelationalConfig] = None) -> None:
        self.config = config or RelationalConfig()

    # ── Main per-turn update ─────────────────────────────────────────

    def process_turn(
        self,
        field: RelationalField,
        person_id: str,
        message: str,
        *,
        valence: float = 0.0,
        salience: float = 0.5,
        drive_impact: Optional[Any] = None,
        person_name: str = "",
    ) -> RelationalModel:
        """Update (or create) the relationship for *person_id*.

        Parameters
        ----------
        field : RelationalField
            The global relational field container.
        person_id : str
            Unique identifier for the interlocutor.
        message : str
            The user message text for this turn.
        valence : float
            Turn-level emotional valence (-1 to +1).
        salience : float
            Turn-level salience (0 to 1).
        drive_impact : DriveImpact, optional
            The drive impact produced by DriveProcessor this turn.
            Used to update per-relationship drive effects.
        person_name : str
            Display name (used on first creation).

        Returns
        -------
        RelationalModel
            The updated relationship model.
        """
        cfg = self.config
        rel = field.get_or_create(person_id, person_name=person_name)

        # -- Interaction bookkeeping --
        rel.interaction_count += 1
        rel.last_interaction_ts = __import__("time").time()

        # -- Felt quality shift --
        rel.felt_quality = self._update_felt_quality(rel.felt_quality, valence, cfg)

        # -- Attachment growth --
        rel.attachment_strength = _clamp(
            rel.attachment_strength + cfg.attachment_growth_per_turn,
            0.0,
            1.0,
        )

        # -- Drive effect EMA --
        if drive_impact is not None:
            self._update_drive_effects(rel, drive_impact, cfg)

        # -- Pattern detection --
        self._detect_patterns(rel, message, cfg)

        # -- Gift detection --
        self._detect_gifts(rel, message, cfg)

        # -- Status classification --
        rel.current_status = self._classify_status(rel, cfg)

        # Enforce storage limits
        rel.recurring_patterns = rel.recurring_patterns[: cfg.max_patterns]
        rel.gifts = rel.gifts[: cfg.max_gifts]
        rel.significant_moment_ids = rel.significant_moment_ids[: cfg.max_significant_moments]

        return rel

    # ── Idle decay (called periodically, not every turn) ─────────────

    def apply_idle_decay(
        self,
        field: RelationalField,
    ) -> None:
        """Decay attachment for relationships that have been idle.

        Should be called periodically (e.g., once per session or on a
        timer).  Only relationships that are *not* the current
        interlocutor are decayed.
        """
        cfg = self.config
        for pid, rel in field.relationships.items():
            if pid == field.current_interlocutor:
                continue
            hours_idle = rel.hours_since_last_interaction()
            decay = cfg.attachment_decay_per_hour * hours_idle
            if decay > 0:
                rel.attachment_strength = _clamp(
                    rel.attachment_strength - decay, 0.0, 1.0,
                )
                # Update status if attachment has decayed significantly
                rel.current_status = self._classify_status(rel, cfg)

    # ── Relationship → drive modulation ──────────────────────────────

    def compute_drive_modulation(
        self,
        rel: RelationalModel,
    ) -> Dict[str, float]:
        """Return the drive modulation vector for *rel*.

        The values represent how much each drive should be *shifted*
        when interacting with this person.  Negative = satisfies,
        positive = frustrates.

        The modulation is scaled by ``drive_effect_weight`` from config.
        """
        weight = self.config.drive_effect_weight
        return {
            drive: effect * weight
            for drive, effect in rel.drive_effects.items()
        }

    # ── Context summary for LLM injection ────────────────────────────

    @staticmethod
    def relational_context_summary(rel: RelationalModel) -> str:
        """Build a short text block for the LLM system prompt.

        Included only when the relationship is significant.
        """
        parts: list[str] = []
        name = rel.person_name or rel.person_id
        parts.append(f"[Relationship with {name}]")
        parts.append(rel.describe_felt_quality())
        if rel.recurring_patterns:
            patterns_str = "; ".join(rel.recurring_patterns[:3])
            parts.append(f"Dynamics: {patterns_str}")
        if rel.gifts:
            gifts_str = ", ".join(rel.gifts[:3])
            parts.append(f"Gifts received: {gifts_str}")
        # Drive effects summary (only notable ones)
        notable = {
            d: v for d, v in rel.drive_effects.items() if abs(v) > 0.05
        }
        if notable:
            effects_parts = []
            for d, v in notable.items():
                direction = "satisfies" if v < 0 else "frustrates"
                effects_parts.append(f"{d} ({direction})")
            parts.append(f"Drive effects: {', '.join(effects_parts)}")
        return "\n".join(parts)

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _update_felt_quality(
        current: float,
        valence: float,
        cfg: RelationalConfig,
    ) -> float:
        """Shift felt quality based on turn valence."""
        if valence > 0.1:
            delta = cfg.positive_quality_delta * valence
        elif valence < -0.1:
            delta = cfg.negative_quality_delta * valence  # valence is negative → delta < 0
        else:
            # Neutral: small drift toward zero
            delta = -cfg.neutral_quality_decay if current > 0 else cfg.neutral_quality_decay
            if abs(current) < cfg.neutral_quality_decay:
                delta = -current  # snap to zero
        return _clamp(current + delta, -1.0, 1.0)

    @staticmethod
    def _update_drive_effects(
        rel: RelationalModel,
        drive_impact: Any,
        cfg: RelationalConfig,
    ) -> None:
        """Update per-relationship drive effects via EMA.

        ``drive_impact`` is expected to have a ``.effects`` dict
        ``{drive_name: float}`` (from ``DriveImpact.effects``).
        """
        effects = getattr(drive_impact, "effects", None)
        if effects is None:
            return
        alpha = cfg.drive_effect_ema_alpha
        for drive, impact_val in effects.items():
            old = rel.drive_effects.get(drive, 0.0)
            # EMA: new = alpha * observation + (1-alpha) * old
            rel.drive_effects[drive] = alpha * (-impact_val) + (1 - alpha) * old
            # Note: impact_val is positive when the drive was *satisfied*
            # and negative when *frustrated*.  We negate so that a positive
            # drive_effect in the model means "this relationship frustrates
            # this drive" (matching the architecture doc convention).

    def _detect_patterns(
        self,
        rel: RelationalModel,
        message: str,
        cfg: RelationalConfig,
    ) -> None:
        """Add recurring pattern observations via simple keyword matching."""
        msg_lower = message.lower()
        for pattern_desc, keywords in _PATTERN_KEYWORDS.items():
            if pattern_desc in rel.recurring_patterns:
                continue  # already captured
            if len(rel.recurring_patterns) >= cfg.max_patterns:
                break
            if any(kw in msg_lower for kw in keywords):
                rel.recurring_patterns.append(pattern_desc)

    def _detect_gifts(
        self,
        rel: RelationalModel,
        message: str,
        cfg: RelationalConfig,
    ) -> None:
        """Detect relational gifts via keyword matching."""
        msg_lower = message.lower()
        for gift_desc, keywords in _GIFT_KEYWORDS.items():
            if gift_desc in rel.gifts:
                continue  # already captured
            if len(rel.gifts) >= cfg.max_gifts:
                break
            if any(kw in msg_lower for kw in keywords):
                rel.gifts.append(gift_desc)

    @staticmethod
    def _classify_status(
        rel: RelationalModel,
        cfg: RelationalConfig,
    ) -> str:
        """Classify relationship status based on current state."""
        hours_idle = rel.hours_since_last_interaction()

        # Dormant: no interaction for > 24h and not very attached
        if hours_idle > 24 and rel.attachment_strength < 0.3:
            return "dormant"

        # Strained: low felt quality over several interactions
        if rel.felt_quality < -0.3 and rel.interaction_count >= cfg.significant_interaction_threshold:
            return "strained"

        # Growing: positive quality and rising attachment
        if (
            rel.felt_quality > 0.2
            and rel.attachment_strength > 0.1
            and rel.interaction_count >= cfg.significant_interaction_threshold
        ):
            return "growing"

        return "active"
