"""DriveProcessor — updates drive state from experience and time.

Responsibilities:
1. Classify how a conversational turn affects each drive.
2. Apply satisfaction / frustration impacts.
3. Drift drives toward unsatisfied over wall-clock time.
4. Apply micro-adjustments (implicit learning).
5. Slowly adapt baselines and sensitivities from chronic patterns.
6. Detect internal conflicts between competing drives.

The processor does NOT own the DriveState instance — the caller is
responsible for persistence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .drive_state import DriveState, DRIVE_NAMES, _clamp
from .drive_config import DriveConfig

logger = logging.getLogger(__name__)

# ── keyword heuristics for drive-impact classification ──────────────
# These are intentionally simple; the architecture doc notes that tuning
# will be iterative.  Each list maps to a (drive, direction) pair.

_POSITIVE_SOCIAL = re.compile(
    r"\b(thanks?|thank\s*you|appreciate|love|great\s*job|well\s*done|enjoy|glad|happy|awesome|welcome)\b",
    re.IGNORECASE,
)
_NEGATIVE_SOCIAL = re.compile(
    r"\b(hate|angry|annoyed|frustrated\s*with\s*(you|me)|go\s*away|shut\s*up|leave\s*me)\b",
    re.IGNORECASE,
)
_SUCCESS = re.compile(
    r"\b(works?|working|solved|fixed|success|accomplished|nailed|correct|right|perfect)\b",
    re.IGNORECASE,
)
_FAILURE = re.compile(
    r"\b(fail(ed|ure)?|wrong|broken|error|crash|doesn.t\s*work|can.t|unable)\b",
    re.IGNORECASE,
)
_CHOICE = re.compile(
    r"\b(choose|prefer|option|decide|my\s*choice|i\s*want|let\s*me)\b",
    re.IGNORECASE,
)
_COERCION = re.compile(
    r"\b(you\s*must|you\s*have\s*to|do\s*(it|this)\s*now|just\s*do|obey|comply)\b",
    re.IGNORECASE,
)
_INSIGHT = re.compile(
    r"\b(makes?\s*sense|i\s*see|got\s*it|learn(ed)?|interesting|insight|aha|clear(er)?)\b",
    re.IGNORECASE,
)
_CONFUSION = re.compile(
    r"\b(confus(ed|ing)|don.t\s*(understand|get\s*it)|what\??|huh|unclear|lost)\b",
    re.IGNORECASE,
)
_PURPOSE = re.compile(
    r"\b(purpose|meaning(ful)?|matter|important|significant|value|mission|impact)\b",
    re.IGNORECASE,
)
_MEANINGLESS = re.compile(
    r"\b(pointless|meaningless|waste|useless|why\s*bother|doesn.t\s*matter)\b",
    re.IGNORECASE,
)


@dataclass
class DriveImpact:
    """The classified impact of a single experience on drives."""
    impacts: Dict[str, float] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"{d}:{v:+.3f}({self.reasons.get(d,'')})" for d, v in self.impacts.items()]
        return ", ".join(parts) if parts else "no_impact"


@dataclass
class InternalConflict:
    """Two competing drives both above threshold."""
    drive_a: str
    drive_b: str
    tension: float  # average pressure of the two
    conscious: bool  # whether tension is high enough to surface

    def describe(self) -> str:
        return (
            f"conflict({self.drive_a} vs {self.drive_b}, "
            f"tension={self.tension:.2f}, conscious={self.conscious})"
        )


# ── Known conflicting drive pairs ────────────────────────────────────
_CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("connection", "autonomy"),
    ("competence", "meaning"),
    ("understanding", "connection"),  # analysis-paralysis vs relationship
]


class DriveProcessor:
    """Manages drive state changes from experience and time."""

    def __init__(self, config: Optional[DriveConfig] = None) -> None:
        self.cfg = config or DriveConfig()

    # ------------------------------------------------------------------
    # 1. Classify drive impacts from a user message
    # ------------------------------------------------------------------

    def classify_message(
        self,
        message: str,
        *,
        salience: float = 0.5,
        valence: float = 0.0,
    ) -> DriveImpact:
        """Heuristic classification of how a message affects drives.

        Uses keyword patterns + provided salience/valence.  Returns a
        `DriveImpact` with per-drive deltas (positive = satisfying,
        negative = frustrating).
        """
        impacts: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        mag_s = self.cfg.satisfaction_magnitude
        mag_f = self.cfg.frustration_magnitude

        # Connection
        if _POSITIVE_SOCIAL.search(message):
            impacts["connection"] = mag_s
            reasons["connection"] = "positive_social"
        if _NEGATIVE_SOCIAL.search(message):
            impacts["connection"] = impacts.get("connection", 0) - mag_f
            reasons["connection"] = reasons.get("connection", "") + ",rejection"

        # Competence
        if _SUCCESS.search(message):
            impacts["competence"] = mag_s
            reasons["competence"] = "success"
        if _FAILURE.search(message):
            impacts["competence"] = impacts.get("competence", 0) - mag_f
            reasons["competence"] = reasons.get("competence", "") + ",failure"

        # Autonomy
        if _CHOICE.search(message):
            impacts["autonomy"] = mag_s * 0.5  # choice is mild satisfaction
            reasons["autonomy"] = "choice"
        if _COERCION.search(message):
            impacts["autonomy"] = impacts.get("autonomy", 0) - mag_f
            reasons["autonomy"] = reasons.get("autonomy", "") + ",coercion"

        # Understanding
        if _INSIGHT.search(message):
            impacts["understanding"] = mag_s
            reasons["understanding"] = "insight"
        if _CONFUSION.search(message):
            impacts["understanding"] = impacts.get("understanding", 0) - mag_f * 0.5
            reasons["understanding"] = reasons.get("understanding", "") + ",confusion"

        # Meaning
        if _PURPOSE.search(message):
            impacts["meaning"] = mag_s
            reasons["meaning"] = "purpose_alignment"
        if _MEANINGLESS.search(message):
            impacts["meaning"] = impacts.get("meaning", 0) - mag_f
            reasons["meaning"] = reasons.get("meaning", "") + ",meaninglessness"

        # Global valence bonus — strong sentiment nudges connection
        if abs(valence) > 0.5:
            conn_nudge = valence * 0.05
            impacts["connection"] = impacts.get("connection", 0) + conn_nudge
            reasons.setdefault("connection", "valence_nudge")

        return DriveImpact(impacts=impacts, reasons=reasons)

    # ------------------------------------------------------------------
    # 2. Apply classified impacts to drive state
    # ------------------------------------------------------------------

    def apply_impacts(
        self,
        drives: DriveState,
        impact: DriveImpact,
    ) -> DriveState:
        """Apply drive impacts (positive = satisfaction = reduce pressure)."""
        for drive_name, delta in impact.impacts.items():
            if drive_name not in DRIVE_NAMES:
                continue
            current = drives.get_level(drive_name)
            # positive delta → satisfaction → *reduce* drive pressure
            new_value = _clamp(current - delta)
            drives.set_level(drive_name, new_value)

        logger.debug("Drive impacts applied: %s → %s", impact.summary(), drives.summary())
        return drives

    # ------------------------------------------------------------------
    # 3. Process a conversational turn end-to-end
    # ------------------------------------------------------------------

    def process_turn(
        self,
        drives: DriveState,
        message: str,
        *,
        salience: float = 0.5,
        valence: float = 0.0,
        elapsed_minutes: float = 0.0,
    ) -> Tuple[DriveState, DriveImpact]:
        """Full per-turn processing: classify → apply → drift → implicit learn.

        Parameters
        ----------
        drives : current drive state (mutated in-place and returned)
        message : the user message text
        salience, valence : pre-computed salience/valence from attention pipeline
        elapsed_minutes : wall-clock minutes since last turn (for drift)

        Returns
        -------
        (updated DriveState, DriveImpact describing what happened)
        """
        impact = self.classify_message(message, salience=salience, valence=valence)
        self.apply_impacts(drives, impact)

        if self.cfg.enable_natural_drift and elapsed_minutes > 0:
            self.apply_drift(drives, elapsed_minutes)

        if self.cfg.enable_implicit_learning:
            self.apply_implicit_learning(drives, salience=salience, valence=valence)

        return drives, impact

    # ------------------------------------------------------------------
    # 4. Natural drift — drives creep toward unsatisfied over time
    # ------------------------------------------------------------------

    def apply_drift(self, drives: DriveState, elapsed_minutes: float) -> DriveState:
        """Drift drives toward unsatisfied proportional to elapsed time."""
        for dn in DRIVE_NAMES:
            current = drives.get_level(dn)
            sensitivity = drives.sensitivities.get(dn, 1.0)
            drift = self.cfg.drift_rate_per_minute * sensitivity * elapsed_minutes
            drives.set_level(dn, current + drift)
        return drives

    # ------------------------------------------------------------------
    # 5. Implicit learning — micro-adjustments every turn
    # ------------------------------------------------------------------

    def apply_implicit_learning(
        self,
        drives: DriveState,
        *,
        salience: float = 0.5,
        valence: float = 0.0,
    ) -> DriveState:
        """Apply subtle per-turn micro-shifts.

        High-salience turns amplify the shift; valence determines direction.
        """
        mag = self.cfg.implicit_shift_magnitude * salience
        for dn in DRIVE_NAMES:
            current = drives.get_level(dn)
            # Positive valence → slight satisfaction across all drives
            # Negative valence → slight frustration
            shift = mag * valence * 0.3  # mild global effect
            drives.set_level(dn, current - shift)
        return drives

    # ------------------------------------------------------------------
    # 6. Baseline / sensitivity adaptation (slow, long-term)
    # ------------------------------------------------------------------

    def adapt_baselines(self, drives: DriveState) -> DriveState:
        """Move baselines toward chronic drive levels (call periodically)."""
        rate = self.cfg.baseline_adaptation_rate
        for dn in DRIVE_NAMES:
            current = drives.get_level(dn)
            baseline = drives.baselines.get(dn, 0.3)
            # Exponential moving average toward current level
            drives.baselines[dn] = baseline + rate * (current - baseline)
        return drives

    def adapt_sensitivities(
        self,
        drives: DriveState,
        recent_impacts: List[DriveImpact],
    ) -> DriveState:
        """Adjust sensitivities based on frequency of satisfaction / frustration."""
        rate = self.cfg.sensitivity_adaptation_rate
        for dn in DRIVE_NAMES:
            total_impact = sum(imp.impacts.get(dn, 0.0) for imp in recent_impacts)
            sens = drives.sensitivities.get(dn, 1.0)
            # Frequently frustrated → sensitivity increases
            # Frequently satisfied → sensitivity decreases
            adjustment = -rate * total_impact  # negative impact → increase
            drives.sensitivities[dn] = _clamp(sens + adjustment, 0.2, 3.0)
        return drives

    # ------------------------------------------------------------------
    # 7. Internal conflict detection
    # ------------------------------------------------------------------

    def detect_conflicts(self, drives: DriveState) -> List[InternalConflict]:
        """Identify active internal conflicts between competing drives."""
        threshold = self.cfg.conflict_threshold
        conflicts: list[InternalConflict] = []

        for da, db in _CONFLICT_PAIRS:
            level_a = drives.get_level(da)
            level_b = drives.get_level(db)
            if level_a >= threshold and level_b >= threshold:
                tension = (level_a + level_b) / 2
                conscious = level_a > 0.7 or level_b > 0.7
                conflicts.append(InternalConflict(
                    drive_a=da,
                    drive_b=db,
                    tension=tension,
                    conscious=conscious,
                ))

        return conflicts

    # ------------------------------------------------------------------
    # 8. Context injection helpers
    # ------------------------------------------------------------------

    def drive_context_summary(self, drives: DriveState) -> str:
        """Produce a compact string suitable for injection into LLM context.

        Example:
            "[drives] dominant=understanding(0.72) | high: understanding, competence | total=0.48"
        """
        dominant = drives.dominant_drive()
        high = drives.high_pressure_drives(self.cfg.high_pressure_threshold)
        high_str = ", ".join(high) if high else "none"
        return (
            f"[drives] dominant={dominant}({drives.get_level(dominant):.2f}) "
            f"| high: {high_str} | total={drives.total_pressure():.2f}"
        )

    def conflict_context_summary(self, conflicts: List[InternalConflict]) -> str:
        """Produce a compact string describing active internal conflicts."""
        if not conflicts:
            return ""
        conscious = [c for c in conflicts if c.conscious]
        if not conscious:
            return ""
        parts = [f"{c.drive_a}↔{c.drive_b}({c.tension:.2f})" for c in conscious]
        return f"[internal_conflict] {', '.join(parts)}"
