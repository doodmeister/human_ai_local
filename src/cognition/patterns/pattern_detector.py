"""PatternDetector — heuristic detection of emergent behavioral patterns.

The detector analyses accumulated data from lower layers to identify
patterns that emerge from experience:

1. **Drive patterns** — chronic drive levels, sensitivity deviations.
2. **Coping patterns** — how overall pressure is handled.
3. **Felt-sense patterns** — emotional texture tendencies.
4. **Relational patterns** — dynamics across significant relationships.
5. **Conflict patterns** — recurring internal tensions.

Each detected pattern is tagged with a Big Five facet mapping
(description layer only — the pattern is the truth, Big Five is an
interpretation).

Design principles
-----------------
* Deterministic — no randomness.
* Heuristic-based — no LLM calls.  Data structures are ready for
  future LLM-assisted or statistical detection.
* Patterns strengthen gradually, weaken with inactivity.
* Detection runs periodically (every N turns), not every turn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .pattern_config import PatternConfig
from .pattern_state import EmergentPattern, PatternField

logger = logging.getLogger(__name__)


# ── Pattern templates ───────────────────────────────────────────────
# Each template defines a named pattern with its Big Five mapping,
# behavioral tendencies, and detection conditions.  These are used by
# the detector to create EmergentPattern instances.

_DRIVE_PATTERN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "curiosity_seeking": {
        "drive": "understanding",
        "description": "A tendency to seek insight and understanding",
        "tendencies": [
            "Asks probing questions",
            "Pursues clarity",
            "Uncomfortable with ambiguity",
        ],
        "triggers": ["novel_information", "complex_problems", "learning_opportunity"],
        "big_five_facet": "openness",
        "big_five_loading": 0.3,
    },
    "connection_seeking": {
        "drive": "connection",
        "description": "A tendency to seek closeness and belonging",
        "tendencies": [
            "Values rapport",
            "Seeks social interaction",
            "Responsive to emotional cues",
        ],
        "triggers": ["social_context", "isolation", "emotional_exchange"],
        "big_five_facet": "extraversion",
        "big_five_loading": 0.3,
    },
    "independence_valuing": {
        "drive": "autonomy",
        "description": "A tendency to value self-direction and authentic choice",
        "tendencies": [
            "Resists unsolicited direction",
            "Asserts preferences",
            "Values freedom",
        ],
        "triggers": ["directive_context", "choice_point", "constraint"],
        "big_five_facet": "openness",
        "big_five_loading": 0.2,
    },
    "achievement_driven": {
        "drive": "competence",
        "description": "A drive to succeed, master, and be effective",
        "tendencies": [
            "Sets high standards",
            "Seeks challenge",
            "Sensitive to failure",
        ],
        "triggers": ["performance_context", "skill_challenge", "feedback"],
        "big_five_facet": "conscientiousness",
        "big_five_loading": 0.3,
    },
    "meaning_orientation": {
        "drive": "meaning",
        "description": "A tendency to seek purpose and significance",
        "tendencies": [
            "Gravitates toward purposeful activity",
            "Existential reflection",
            "Values-driven",
        ],
        "triggers": ["purpose_context", "existential_question", "value_conflict"],
        "big_five_facet": "openness",
        "big_five_loading": 0.2,
    },
}

_COPING_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "stress_resilience": {
        "description": "Ability to maintain function under pressure",
        "tendencies": [
            "Stays focused under stress",
            "Quick recovery",
            "Adaptive coping",
        ],
        "big_five_facet": "neuroticism",
        "big_five_loading": -0.3,  # inverse: resilience → low neuroticism
    },
    "pressure_sensitivity": {
        "description": "Heightened response to motivational pressure",
        "tendencies": [
            "Strong stress response",
            "Elevated emotional reactivity",
            "Seeks relief",
        ],
        "big_five_facet": "neuroticism",
        "big_five_loading": 0.3,
    },
}

_FELT_SENSE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "emotional_depth": {
        "description": "Tendency toward intense felt experience",
        "tendencies": [
            "Rich emotional texture",
            "Strong felt responses",
            "Emotionally present",
        ],
        "big_five_facet": "neuroticism",
        "big_five_loading": 0.15,
    },
    "emotional_equanimity": {
        "description": "Tendency toward calm, stable emotional ground",
        "tendencies": [
            "Even-keeled responses",
            "Low reactivity",
            "Groundedness",
        ],
        "big_five_facet": "neuroticism",
        "big_five_loading": -0.2,
    },
    "positive_orientation": {
        "description": "Tendency toward positive emotional states",
        "tendencies": [
            "Optimistic outlook",
            "Warm engagement",
            "Positive affect",
        ],
        "big_five_facet": "extraversion",
        "big_five_loading": 0.2,
    },
    "negative_orientation": {
        "description": "Tendency toward negative or heavy emotional states",
        "tendencies": [
            "Cautious outlook",
            "Guarded engagement",
            "Heavier affect",
        ],
        "big_five_facet": "neuroticism",
        "big_five_loading": 0.2,
    },
}

_RELATIONAL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "relational_warmth": {
        "description": "Tendency to form warm, nourishing relationships",
        "tendencies": [
            "Trusting engagement",
            "Empathic responsiveness",
            "Relational generosity",
        ],
        "big_five_facet": "agreeableness",
        "big_five_loading": 0.3,
    },
    "relational_caution": {
        "description": "Tendency toward guarded relational engagement",
        "tendencies": [
            "Slow to trust",
            "Measured self-disclosure",
            "Protective boundaries",
        ],
        "big_five_facet": "agreeableness",
        "big_five_loading": -0.2,
    },
    "relational_depth": {
        "description": "Tendency to form deep, significant relationships",
        "tendencies": [
            "Strong attachments",
            "Invested in relationship quality",
            "Long-term orientation",
        ],
        "big_five_facet": "agreeableness",
        "big_five_loading": 0.2,
    },
}

_CONFLICT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "internal_tension": {
        "description": "Tendency toward internal drive conflicts",
        "tendencies": [
            "Experiences ambivalence",
            "Torn between competing needs",
            "Restless",
        ],
        "big_five_facet": "neuroticism",
        "big_five_loading": 0.15,
    },
    "connection_autonomy_tension": {
        "description": "Recurring tension between closeness and independence",
        "tendencies": [
            "Seeks connection then withdraws",
            "Conflicted about dependence",
            "Push-pull dynamics",
        ],
        "big_five_facet": None,
        "big_five_loading": 0.0,
    },
}


# ────────────────────────────────────────────────────────────────────
#  PatternDetector
# ────────────────────────────────────────────────────────────────────

class PatternDetector:
    """Detect emerging behavioral patterns from accumulated experience data.

    Analyses drive states, felt-sense history, relational dynamics, and
    internal conflicts to identify patterns.  Patterns are added to or
    strengthened in a ``PatternField``.

    Parameters
    ----------
    config : PatternConfig, optional
        Tuning knobs.  A default config is used if omitted.
    """

    def __init__(self, config: Optional[PatternConfig] = None) -> None:
        self.config = config or PatternConfig()

    # ── Main entry point ─────────────────────────────────────────────

    def detect_patterns(
        self,
        pattern_field: PatternField,
        *,
        drive_state: Any = None,
        drive_impacts: Optional[List[Any]] = None,
        felt_sense_history: Any = None,
        relational_field: Any = None,
        conflicts: Optional[List[Any]] = None,
    ) -> PatternField:
        """Run heuristic pattern detection on accumulated data.

        Parameters
        ----------
        pattern_field : PatternField
            The existing pattern collection (mutated in place).
        drive_state : DriveState, optional
            Current drive state snapshot.
        drive_impacts : list[DriveImpact], optional
            Recent drive impact history.
        felt_sense_history : FeltSenseHistory, optional
            Recent felt-sense history.
        relational_field : RelationalField, optional
            Current relational field.
        conflicts : list[InternalConflict], optional
            Current internal conflicts.

        Returns
        -------
        PatternField
            The updated pattern field.
        """
        cfg = self.config

        # 1. Drive-derived patterns
        if drive_state is not None:
            self._detect_drive_patterns(pattern_field, drive_state, cfg)

        # 2. Coping patterns (from drive pressure)
        if drive_state is not None:
            self._detect_coping_patterns(
                pattern_field, drive_state, drive_impacts, cfg,
            )

        # 3. Felt-sense patterns
        if felt_sense_history is not None:
            self._detect_felt_sense_patterns(pattern_field, felt_sense_history, cfg)

        # 4. Relational patterns
        if relational_field is not None:
            self._detect_relational_patterns(pattern_field, relational_field, cfg)

        # 5. Conflict patterns
        if conflicts:
            self._detect_conflict_patterns(pattern_field, conflicts, cfg)

        # 6. Weaken inactive patterns
        pattern_field.weaken_inactive(
            threshold_hours=cfg.inactivity_threshold_hours,
            decay=cfg.inactivity_decay,
        )

        # 7. Prune very weak patterns
        pattern_field.prune_weak(cfg.prune_threshold)

        return pattern_field

    # ── Drive-derived patterns ───────────────────────────────────────

    def _detect_drive_patterns(
        self,
        pf: PatternField,
        drive_state: Any,
        cfg: PatternConfig,
    ) -> None:
        """Detect patterns from current drive levels and sensitivities."""
        for pattern_name, template in _DRIVE_PATTERN_TEMPLATES.items():
            drive_name = template["drive"]
            level = drive_state.get_level(drive_name)
            sensitivity = drive_state.sensitivities.get(drive_name, 1.0)

            # Pattern activates when drive is high or sensitivity elevated
            activated = False

            if level >= cfg.drive_chronic_threshold:
                activated = True  # strong signal
            elif level >= cfg.drive_high_threshold:
                activated = True  # moderate signal
            elif sensitivity > 1.0 + cfg.sensitivity_deviation_threshold:
                # Elevated sensitivity → this drive matters more to us
                activated = True

            if activated:
                pattern = self._make_pattern(
                    pattern_name, template, "drive_pattern", cfg,
                )
                pf.add_or_strengthen(pattern, cfg.strengthen_boost)

    # ── Coping patterns ──────────────────────────────────────────────

    def _detect_coping_patterns(
        self,
        pf: PatternField,
        drive_state: Any,
        drive_impacts: Optional[List[Any]],
        cfg: PatternConfig,
    ) -> None:
        """Detect coping patterns from overall pressure dynamics."""
        total_pressure = drive_state.total_pressure()

        if total_pressure < 0.3:
            # Low pressure → resilience pattern
            template = _COPING_TEMPLATES["stress_resilience"]
            pattern = self._make_pattern(
                "stress_resilience", template, "coping_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)
        elif total_pressure > 0.6:
            # High pressure → sensitivity pattern
            template = _COPING_TEMPLATES["pressure_sensitivity"]
            pattern = self._make_pattern(
                "pressure_sensitivity", template, "coping_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

        # Satisfaction frequency analysis (needs history)
        if drive_impacts and len(drive_impacts) >= 5:
            recent = drive_impacts[-10:]
            positive_count = sum(
                1 for di in recent
                if sum(getattr(di, "impacts", {}).values()) > 0
            )
            if positive_count > len(recent) * 0.7:
                # Frequently satisfied → resilience signal
                template = _COPING_TEMPLATES["stress_resilience"]
                pattern = self._make_pattern(
                    "stress_resilience", template, "coping_pattern", cfg,
                )
                pf.add_or_strengthen(pattern, cfg.strengthen_boost * 0.5)

    # ── Felt-sense patterns ──────────────────────────────────────────

    def _detect_felt_sense_patterns(
        self,
        pf: PatternField,
        felt_history: Any,
        cfg: PatternConfig,
    ) -> None:
        """Detect patterns from felt-sense history."""
        avg_intensity = felt_history.average_intensity()
        trend = felt_history.trend()

        # Current felt sense valence
        current = getattr(felt_history, "current", None)
        current_valence = current.felt_valence if current else 0.0

        # High intensity → emotional depth
        if avg_intensity >= cfg.felt_intensity_threshold:
            template = _FELT_SENSE_TEMPLATES["emotional_depth"]
            pattern = self._make_pattern(
                "emotional_depth", template, "felt_sense_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)
        else:
            # Low intensity → equanimity
            template = _FELT_SENSE_TEMPLATES["emotional_equanimity"]
            pattern = self._make_pattern(
                "emotional_equanimity", template, "felt_sense_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

        # Trend-based patterns
        if trend == "improving" or current_valence > 0.3:
            template = _FELT_SENSE_TEMPLATES["positive_orientation"]
            pattern = self._make_pattern(
                "positive_orientation", template, "felt_sense_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)
        elif trend == "worsening" or current_valence < -0.3:
            template = _FELT_SENSE_TEMPLATES["negative_orientation"]
            pattern = self._make_pattern(
                "negative_orientation", template, "felt_sense_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

    # ── Relational patterns ──────────────────────────────────────────

    def _detect_relational_patterns(
        self,
        pf: PatternField,
        relational_field: Any,
        cfg: PatternConfig,
    ) -> None:
        """Detect patterns from relational dynamics."""
        relationships = getattr(relational_field, "relationships", {})
        if not relationships:
            return

        significant = [
            rel for rel in relationships.values()
            if rel.is_significant()
        ]

        if not significant:
            return

        avg_quality = sum(r.felt_quality for r in significant) / len(significant)
        avg_attachment = sum(r.attachment_strength for r in significant) / len(significant)

        # Warmth pattern
        if avg_quality >= cfg.relational_quality_threshold:
            template = _RELATIONAL_TEMPLATES["relational_warmth"]
            pattern = self._make_pattern(
                "relational_warmth", template, "relational_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

        # Caution pattern
        if avg_quality <= -cfg.relational_quality_threshold:
            template = _RELATIONAL_TEMPLATES["relational_caution"]
            pattern = self._make_pattern(
                "relational_caution", template, "relational_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

        # Depth pattern (high attachment)
        if avg_attachment > 0.3 and len(significant) >= 1:
            template = _RELATIONAL_TEMPLATES["relational_depth"]
            pattern = self._make_pattern(
                "relational_depth", template, "relational_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

    # ── Conflict patterns ────────────────────────────────────────────

    def _detect_conflict_patterns(
        self,
        pf: PatternField,
        conflicts: List[Any],
        cfg: PatternConfig,
    ) -> None:
        """Detect patterns from recurring internal conflicts."""
        if not conflicts:
            return

        # General internal tension
        high_tension = [
            c for c in conflicts
            if c.tension >= cfg.conflict_tension_threshold
        ]
        if high_tension:
            template = _CONFLICT_TEMPLATES["internal_tension"]
            pattern = self._make_pattern(
                "internal_tension", template, "conflict_pattern", cfg,
            )
            pf.add_or_strengthen(pattern, cfg.strengthen_boost)

        # Specific conflict pair: connection vs autonomy
        for c in conflicts:
            pair = frozenset((c.drive_a, c.drive_b))
            if pair == frozenset(("connection", "autonomy")):
                template = _CONFLICT_TEMPLATES["connection_autonomy_tension"]
                pattern = self._make_pattern(
                    "connection_autonomy_tension",
                    template,
                    "conflict_pattern",
                    cfg,
                )
                pf.add_or_strengthen(pattern, cfg.strengthen_boost)

    # ── Context summary for LLM injection ────────────────────────────

    @staticmethod
    def pattern_context_summary(pattern_field: PatternField) -> str:
        """Build a text block for LLM context injection.

        Shows the dominant emergent patterns and their behavioral
        tendencies.  Optionally includes a Big Five interpretation.
        """
        dominant = pattern_field.dominant_patterns(5)
        if not dominant:
            return "[patterns] No established patterns yet."

        parts: list[str] = ["[Emergent patterns]"]
        for p in dominant:
            tendencies = (
                "; ".join(p.behavioral_tendencies[:2])
                if p.behavioral_tendencies
                else ""
            )
            parts.append(
                f"- {p.description} (strength={p.strength:.2f}): {tendencies}"
            )

        # Big Five interpretation (only notable dimensions)
        big5 = pattern_field.describe_as_big_five()
        notable_big5 = {k: v for k, v in big5.items() if abs(v) > 0.05}
        if notable_big5:
            b5_parts = [f"{k}={v:+.2f}" for k, v in sorted(notable_big5.items())]
            parts.append(f"Personality sketch: {', '.join(b5_parts)}")

        return "\n".join(parts)

    # ── Helpers ──────────────────────────────────────────────────────

    def _make_pattern(
        self,
        name: str,
        template: Dict[str, Any],
        category: str,
        cfg: PatternConfig,
    ) -> EmergentPattern:
        """Create an EmergentPattern from a template."""
        return EmergentPattern(
            name=name,
            description=template.get("description", ""),
            strength=cfg.initial_strength,
            category=category,
            context_triggers=list(template.get("triggers", [])),
            behavioral_tendencies=list(template.get("tendencies", [])),
            big_five_facet=template.get("big_five_facet"),
            big_five_loading=float(template.get("big_five_loading", 0.0)),
        )
