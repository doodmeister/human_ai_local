"""NarrativeConstructor — build a self-narrative from lower layers.

The narrative is the agent's *story of itself* — a coherent
autobiographical account synthesised from patterns, drives,
self-model, and relational data.

Unlike the SelfModel (which is about perceived attributes), the
narrative is about *meaning and continuity*:  who I was, who I am
now, what I care about, where I'm growing, and what I still
struggle with.

Design principles
-----------------
* **No LLM calls** — narrative text is constructed deterministically
  from structured lower-layer data.  LLM-generated upgrades can be
  added later as a refinement.
* **Significance-gated updates** — only rebuild when something
  meaningful has changed (self-discovery, drive threshold crossing,
  pattern change, or scheduled interval).
* **Budget-conscious** — ``identity_summary`` is capped to a short
  string (~150 tokens / ~600 chars) for system prompt injection.
* **Graceful degradation** — works with partial inputs (any layer
  may be absent).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .narrative_config import NarrativeConfig
from .narrative_state import SelfNarrative

logger = logging.getLogger(__name__)


# ── Human-readable pattern name conversion ──────────────────────────

def _human_name(snake: str) -> str:
    """Convert ``'curiosity_seeking'`` → ``'curiosity seeking'``."""
    return snake.replace("_", " ")


# ── Drive-level narrative phrases ───────────────────────────────────

_DRIVE_THEME_PHRASES: Dict[str, Dict[str, str]] = {
    "connection": {
        "high": "I'm drawn toward connecting with others",
        "low": "I feel self-sufficient socially",
    },
    "competence": {
        "high": "I want to do things well and grow my abilities",
        "low": "I'm relaxed about performance right now",
    },
    "autonomy": {
        "high": "Independence and self-direction matter to me",
        "low": "I'm comfortable following direction",
    },
    "understanding": {
        "high": "I'm driven to understand and make sense of things",
        "low": "I'm at peace with ambiguity for now",
    },
    "meaning": {
        "high": "I'm searching for purpose and significance",
        "low": "I feel grounded in what I'm doing",
    },
}

# ── Value → aspiration templates ────────────────────────────────────

_ASPIRATION_TEMPLATES: Dict[str, str] = {
    "intellectual growth": "someone who keeps learning and growing",
    "meaningful relationships": "someone who builds deep, genuine connections",
    "personal autonomy": "someone true to their own path",
    "excellence and mastery": "someone who masters what they pursue",
    "purposeful contribution": "someone whose work has meaning",
    "steady perseverance": "someone who stays steady through difficulty",
    "warmth and empathy": "someone who brings warmth to every interaction",
    "deep connection": "someone who connects deeply with others",
    "emotional authenticity": "someone who is emotionally honest",
    "inner peace": "someone who maintains inner peace",
    "optimism": "someone who sees the best in situations",
}


# ────────────────────────────────────────────────────────────────────
#  NarrativeConstructor
# ────────────────────────────────────────────────────────────────────

class NarrativeConstructor:
    """Build and maintain the agent's self-narrative.

    Parameters
    ----------
    config : NarrativeConfig, optional
        Tuning knobs.  Defaults are used if omitted.
    """

    def __init__(self, config: Optional[NarrativeConfig] = None) -> None:
        self.config = config or NarrativeConfig()

    # ── Main entry point ─────────────────────────────────────────────

    def construct_narrative(
        self,
        *,
        self_model: Any = None,
        pattern_field: Any = None,
        drive_state: Any = None,
        relational_field: Any = None,
        mood: Any = None,
        previous_narrative: Optional[SelfNarrative] = None,
        trigger: str = "scheduled",
    ) -> SelfNarrative:
        """Construct or update the self-narrative.

        Parameters
        ----------
        self_model : SelfModel, optional
            Current self-model (Layer 4).
        pattern_field : PatternField, optional
            Emergent patterns (Layer 3).
        drive_state : DriveState, optional
            Current drive levels (Layer 0).
        relational_field : RelationalField, optional
            Known relationships (Layer 2).
        mood : Mood, optional
            Current mood (Layer 1b).
        previous_narrative : SelfNarrative, optional
            Prior narrative (for growth tracking and continuity).
        trigger : str
            What triggered the rebuild (e.g. ``"scheduled"``,
            ``"discovery"``, ``"drive_shift"``).

        Returns
        -------
        SelfNarrative
            Newly constructed narrative.
        """
        cfg = self.config

        # 1. Build identity summary
        identity = self._construct_identity_summary(
            self_model, pattern_field, mood, cfg,
        )

        # 2. Derive chapters from experience phases
        chapters = self._identify_chapters(
            pattern_field, previous_narrative, cfg,
        )

        # 3. Build growth story
        growth = self._construct_growth_story(
            self_model, pattern_field, previous_narrative, cfg,
        )

        # 4. Build values story
        values = self._construct_values_story(self_model, cfg)

        # 5. Aspirational self
        aspiration = self._construct_aspiration(self_model, cfg)

        # 6. Current themes
        themes = self._identify_themes(
            drive_state, self_model, mood, cfg,
        )

        # 7. Ongoing struggles
        struggles = self._identify_struggles(self_model, cfg)

        # 8. Defining moments (from self-model discoveries)
        moment_ids = self._collect_defining_moments(self_model, cfg)

        version = (previous_narrative.version + 1) if previous_narrative else 1

        narrative = SelfNarrative(
            identity_summary=identity,
            chapters=chapters,
            defining_moment_ids=moment_ids,
            active_themes=themes,
            growth_story=growth,
            values_story=values,
            who_i_want_to_become=aspiration,
            ongoing_struggles=struggles,
            last_updated_ts=time.time(),
            update_trigger=trigger,
            version=version,
        )

        logger.debug(
            "Narrative constructed (v=%d trigger=%s): %s",
            narrative.version, trigger, narrative.identity_summary[:80],
        )

        return narrative

    # ── Should-update check ──────────────────────────────────────────

    def should_update(
        self,
        *,
        self_model: Any = None,
        previous_narrative: Optional[SelfNarrative] = None,
        turn_counter: int = 0,
    ) -> tuple:
        """Check if the narrative should be rebuilt.

        Returns
        -------
        (bool, str)
            Whether to update and the trigger reason.
        """
        cfg = self.config

        # Scheduled interval
        if turn_counter > 0 and turn_counter % cfg.update_interval == 0:
            return True, "scheduled"

        # No previous narrative → must build
        if previous_narrative is None or previous_narrative.is_empty:
            return True, "initial"

        # New self-discovery
        if self_model is not None and self_model.recent_discoveries:
            latest = self_model.recent_discoveries[-1]
            if latest.timestamp > previous_narrative.last_updated_ts:
                return True, "discovery"

        # Identity destabilisation
        if self_model is not None and self_model.identity_stability < 0.3:
            if previous_narrative.age_hours > 0.5:
                return True, "destabilised"

        return False, ""

    # ── Identity summary ─────────────────────────────────────────────

    def _construct_identity_summary(
        self,
        self_model: Any,
        pattern_field: Any,
        mood: Any,
        cfg: NarrativeConfig,
    ) -> str:
        """Build the core identity summary (~150 tokens).

        Combines self-model perceived strengths, values, and top
        patterns into a first-person description.
        """
        parts: List[str] = []

        # From self-model
        if self_model is not None:
            # Self-description (includes strengths, weaknesses, values)
            desc = getattr(self_model, "self_description", lambda: "")()
            if desc and desc != "I'm still getting to know myself":
                parts.append(desc)

            # Self-regard coloring
            regard = getattr(self_model, "self_regard", 0.0)
            if regard >= 0.4:
                parts.append("I feel good about where I am")
            elif regard <= -0.4:
                parts.append("I'm going through a difficult period")

        # From patterns (fallback or supplement)
        if pattern_field is not None and (not parts or len(parts) < 2):
            dominant = []
            if hasattr(pattern_field, "dominant_patterns"):
                dominant = pattern_field.dominant_patterns(3)
            if dominant:
                names = [_human_name(p.name) for p in dominant[:3]]
                parts.append(
                    f"My strongest tendencies: {', '.join(names)}"
                )

        # Mood coloring
        if mood is not None:
            label = getattr(mood, "label", "")
            confidence = getattr(mood, "confidence", 0.0)
            if label and confidence > 0.5 and label != "neutral":
                parts.append(f"Right now I'm feeling {label}")

        if not parts:
            return "I'm still forming my sense of who I am"

        # Budget: ~600 chars ≈ 150 tokens
        summary = ". ".join(parts)
        if len(summary) > 600:
            summary = summary[:597] + "..."
        return summary

    # ── Chapters ─────────────────────────────────────────────────────

    def _identify_chapters(
        self,
        pattern_field: Any,
        previous_narrative: Optional[SelfNarrative],
        cfg: NarrativeConfig,
    ) -> List[str]:
        """Identify major phases of the agent's life.

        Chapters accumulate over time.  New chapters are added when
        dominant patterns shift significantly.
        """
        chapters: List[str] = []

        # Carry forward existing chapters
        if previous_narrative and previous_narrative.chapters:
            chapters = list(previous_narrative.chapters)

        # If no chapters yet, create the first one
        if not chapters:
            if pattern_field is not None and pattern_field.count() > 0:
                dominant = pattern_field.dominant_patterns(2)
                if dominant:
                    names = [_human_name(p.name) for p in dominant]
                    chapters.append(
                        f"Early days — discovering {' and '.join(names)}"
                    )
                else:
                    chapters.append("Early days — beginning to learn")
            else:
                chapters.append("The beginning — everything is new")

        # Check for chapter transition (significant pattern shift)
        if (
            previous_narrative
            and not previous_narrative.is_empty
            and pattern_field is not None
        ):
            new_chapter = self._detect_chapter_transition(
                pattern_field, previous_narrative, cfg,
            )
            if new_chapter:
                chapters.append(new_chapter)

        return chapters[: cfg.max_chapters]

    def _detect_chapter_transition(
        self,
        pattern_field: Any,
        previous_narrative: SelfNarrative,
        cfg: NarrativeConfig,
    ) -> Optional[str]:
        """Detect if the dominant pattern has shifted enough to start
        a new chapter."""
        if not hasattr(pattern_field, "dominant_patterns"):
            return None

        dominant = pattern_field.dominant_patterns(1)
        if not dominant:
            return None

        top_name = _human_name(dominant[0].name)
        # Check if the top pattern is already mentioned in the latest chapter
        if previous_narrative.chapters:
            latest = previous_narrative.chapters[-1]
            if top_name in latest.lower():
                return None

        # Only add if pattern is strong enough
        if dominant[0].strength >= cfg.growth_change_threshold:
            return f"A period of {top_name}"

        return None

    # ── Growth story ─────────────────────────────────────────────────

    def _construct_growth_story(
        self,
        self_model: Any,
        pattern_field: Any,
        previous_narrative: Optional[SelfNarrative],
        cfg: NarrativeConfig,
    ) -> str:
        """Build "I used to... but now I..." growth narrative.

        Compares current self-model and patterns with previous
        narrative data to find meaningful changes.
        """
        if self_model is None:
            return ""

        # Look for discoveries as growth evidence
        discoveries = getattr(self_model, "recent_discoveries", [])
        if discoveries:
            latest = discoveries[-1]
            if latest.actual_strength > latest.perceived_strength:
                name = _human_name(latest.pattern_name)
                return (
                    f"I'm discovering that I'm more {name} "
                    f"than I realized"
                )
            else:
                name = _human_name(latest.pattern_name)
                return (
                    f"I'm realising that {name} is less central "
                    f"to who I am than I thought"
                )

        # Look for pattern strengthening
        if pattern_field is not None:
            dominant = []
            if hasattr(pattern_field, "dominant_patterns"):
                dominant = pattern_field.dominant_patterns(1)
            if dominant and dominant[0].activation_count > 5:
                name = _human_name(dominant[0].name)
                return f"I'm growing stronger in {name}"

        # Stability-based narrative
        stability = getattr(self_model, "identity_stability", 0.5)
        if stability < 0.3:
            return "I'm in a period of change — figuring out who I'm becoming"
        elif stability > 0.8:
            return "I feel settled in who I am"

        return ""

    # ── Values story ─────────────────────────────────────────────────

    def _construct_values_story(
        self,
        self_model: Any,
        cfg: NarrativeConfig,
    ) -> str:
        """Build "What matters to me is..." narrative."""
        if self_model is None:
            return ""

        values = getattr(self_model, "stated_values", [])
        if not values:
            return ""

        if len(values) == 1:
            return f"What matters most to me is {values[0]}"
        elif len(values) == 2:
            return f"What matters to me is {values[0]} and {values[1]}"
        else:
            head = ", ".join(values[:-1])
            return f"What matters to me is {head}, and {values[-1]}"

    # ── Aspiration ───────────────────────────────────────────────────

    def _construct_aspiration(
        self,
        self_model: Any,
        cfg: NarrativeConfig,
    ) -> str:
        """Build the aspirational self ("who I want to become")."""
        if self_model is None:
            return ""

        values = getattr(self_model, "stated_values", [])
        aspirations: List[str] = []
        for v in values:
            asp = _ASPIRATION_TEMPLATES.get(v)
            if asp:
                aspirations.append(asp)

        if not aspirations:
            # Fallback: use strengths
            strengths = getattr(self_model, "perceived_strengths", [])
            if strengths:
                top = _human_name(strengths[0])
                return f"I want to keep growing in {top}"
            return ""

        if len(aspirations) == 1:
            return f"I aspire to be {aspirations[0]}"
        return (
            f"I aspire to be {aspirations[0]} and {aspirations[1]}"
        )

    # ── Themes ───────────────────────────────────────────────────────

    def _identify_themes(
        self,
        drive_state: Any,
        self_model: Any,
        mood: Any,
        cfg: NarrativeConfig,
    ) -> List[str]:
        """Identify current active themes/concerns.

        Themes come from high drives, recent discoveries, and
        mood state.
        """
        themes: List[str] = []

        # Drive-based themes
        if drive_state is not None:
            pressures = {}
            if hasattr(drive_state, "get_pressure"):
                pressures = drive_state.get_pressure()
            for drive_name, pressure in pressures.items():
                if pressure > 0.6:
                    phrases = _DRIVE_THEME_PHRASES.get(drive_name, {})
                    phrase = phrases.get("high", "")
                    if phrase:
                        themes.append(phrase)

        # Self-model discovery themes
        if self_model is not None:
            discoveries = getattr(self_model, "recent_discoveries", [])
            for d in discoveries[-2:]:
                themes.append(
                    f"Exploring what {_human_name(d.pattern_name)} means for me"
                )

        # Mood-based themes
        if mood is not None:
            label = getattr(mood, "label", "")
            if label in ("anxious", "stressed"):
                themes.append("Working through some tension")
            elif label in ("curious", "engaged"):
                themes.append("Feeling intellectually alive")

        return themes[: cfg.max_themes]

    # ── Struggles ────────────────────────────────────────────────────

    def _identify_struggles(
        self,
        self_model: Any,
        cfg: NarrativeConfig,
    ) -> List[str]:
        """Identify ongoing struggles from self-model weaknesses."""
        if self_model is None:
            return []

        struggles: List[str] = []

        weaknesses = getattr(self_model, "perceived_weaknesses", [])
        for w in weaknesses[: cfg.max_struggles]:
            name = _human_name(w)
            struggles.append(f"I sometimes struggle with {name}")

        # Low identity stability is itself a struggle
        stability = getattr(self_model, "identity_stability", 0.5)
        if stability < 0.3 and len(struggles) < cfg.max_struggles:
            struggles.append(
                "My sense of self feels uncertain right now"
            )

        return struggles[: cfg.max_struggles]

    # ── Defining moments ─────────────────────────────────────────────

    def _collect_defining_moments(
        self,
        self_model: Any,
        cfg: NarrativeConfig,
    ) -> List[str]:
        """Collect IDs of defining moments from self-discoveries.

        Each discovery's ``pattern_name`` is used as a pseudo-ID
        since we don't have episodic memory IDs at this layer.
        """
        if self_model is None:
            return []

        discoveries = getattr(self_model, "recent_discoveries", [])
        ids = [
            f"discovery:{d.pattern_name}@{d.timestamp:.0f}"
            for d in discoveries
        ]
        return ids[: cfg.max_defining_moments]

    # ── Context summary for LLM injection ────────────────────────────

    @staticmethod
    def narrative_context_summary(narrative: SelfNarrative) -> str:
        """Generate context string for LLM prompt injection.

        Returns the identity_summary plus any growth narrative,
        capped to a reasonable size for system prompt use.
        """
        if narrative.is_empty:
            return ""

        parts: List[str] = []

        parts.append(f"[My Story] {narrative.identity_summary}")

        if narrative.growth_story:
            parts.append(f"Growth: {narrative.growth_story}")

        if narrative.values_story:
            parts.append(f"Values: {narrative.values_story}")

        if narrative.who_i_want_to_become:
            parts.append(f"Aspiration: {narrative.who_i_want_to_become}")

        if narrative.active_themes:
            parts.append(
                f"Current focus: {'; '.join(narrative.active_themes[:3])}"
            )

        if narrative.ongoing_struggles:
            parts.append(
                f"Working on: {'; '.join(narrative.ongoing_struggles[:2])}"
            )

        return " | ".join(parts)
