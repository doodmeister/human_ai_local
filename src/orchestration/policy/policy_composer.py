from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .response_policy import PolicyVector, ResponsePolicy, _as_mapping, _clamp01, _clamp_signed


@dataclass(slots=True)
class PolicyContribution:
    warmth: float = 0.0
    directness: float = 0.0
    curiosity: float = 0.0
    uncertainty: float = 0.0
    disclosure: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "warmth": round(self.warmth, 4),
            "directness": round(self.directness, 4),
            "curiosity": round(self.curiosity, 4),
            "uncertainty": round(self.uncertainty, 4),
            "disclosure": round(self.disclosure, 4),
        }


@dataclass(slots=True)
class PolicyComposer:
    stable_defaults: dict[str, float] = field(
        default_factory=lambda: {
            "warmth": 0.62,
            "directness": 0.68,
            "curiosity": 0.58,
            "uncertainty": 0.38,
            "disclosure": 0.34,
        }
    )

    def compose(
        self,
        *,
        stable_traits: Mapping[str, Any] | None = None,
        drive_state: Mapping[str, Any] | Any = None,
        mood_state: Mapping[str, Any] | Any = None,
        relationship_state: Mapping[str, Any] | Any = None,
        self_model_state: Mapping[str, Any] | Any = None,
        narrative_state: Mapping[str, Any] | Any = None,
    ) -> ResponsePolicy:
        stable_input = dict(self.stable_defaults)
        stable_input.update(_as_mapping(stable_traits))
        stable = PolicyVector(**stable_input)

        drive_data = _as_mapping(drive_state)
        mood_data = _as_mapping(mood_state)
        relationship_data = _as_mapping(relationship_state)
        self_model_data = _as_mapping(self_model_state)
        narrative_data = _as_mapping(narrative_state)

        contributions = {
            "drives": self._compose_drive_contribution(drive_data),
            "mood": self._compose_mood_contribution(mood_data),
            "relationship": self._compose_relationship_contribution(relationship_data),
            "self_model": self._compose_self_model_contribution(self_model_data),
            "narrative": self._compose_narrative_contribution(narrative_data),
        }
        dynamic = self._compose_dynamic_state(contributions)
        effective = self._compose_effective(stable, dynamic)
        trace = self._build_trace(
            drive_state=drive_data,
            mood_state=mood_data,
            relationship_state=relationship_data,
            self_model_state=self_model_data,
            narrative_state=narrative_data,
            contributions=contributions,
            stable=stable,
            dynamic=dynamic,
            effective=effective,
        )
        return ResponsePolicy(
            stable_traits=stable,
            dynamic_state=dynamic,
            effective=effective,
            trace=trace,
        )

    def _compose_drive_contribution(self, drive_state: Mapping[str, Any]) -> PolicyContribution:
        drive_levels = _as_mapping(drive_state.get("levels", {}))
        competence = _clamp01(drive_levels.get("competence", 0.3), default=0.3)
        autonomy = _clamp01(drive_levels.get("autonomy", 0.3), default=0.3)
        understanding = _clamp01(drive_levels.get("understanding", 0.3), default=0.3)
        meaning = _clamp01(drive_levels.get("meaning", 0.3), default=0.3)
        connection = _clamp01(drive_levels.get("connection", 0.3), default=0.3)
        return PolicyContribution(
            warmth=(connection - 0.3) * 0.12,
            directness=(competence - 0.3) * 0.18 + (autonomy - 0.3) * 0.12,
            curiosity=(understanding - 0.3) * 0.24 + (meaning - 0.3) * 0.14,
            uncertainty=max(0.0, 0.45 - competence) * 0.14,
            disclosure=(connection - 0.3) * 0.06,
        )

    def _compose_mood_contribution(self, mood_state: Mapping[str, Any]) -> PolicyContribution:
        label = str(mood_state.get("label", "")).lower()
        valence = _clamp_signed(mood_state.get("valence", 0.0), default=0.0)
        confidence = _clamp01(mood_state.get("confidence", 0.0), default=0.0)
        contribution = PolicyContribution(
            warmth=valence * 0.12,
            directness=max(0.0, valence) * 0.02,
            curiosity=max(0.0, valence) * 0.03,
            uncertainty=max(0.0, -valence) * 0.18,
            disclosure=max(0.0, -valence) * 0.02,
        )
        if label == "curious":
            contribution.curiosity += 0.12 * max(confidence, 0.5)
        elif label in {"uncertain", "anxious", "restless"}:
            contribution.uncertainty += 0.14 * max(confidence, 0.4)
        elif label in {"content", "calm"}:
            contribution.warmth += 0.05
        return contribution

    def _compose_relationship_contribution(self, relationship_state: Mapping[str, Any]) -> PolicyContribution:
        warmth = _clamp01(relationship_state.get("warmth", 0.5), default=0.5)
        trust = _clamp01(relationship_state.get("trust", 0.5), default=0.5)
        familiarity = _clamp01(relationship_state.get("familiarity", 0.0), default=0.0)
        rupture = _clamp01(relationship_state.get("rupture", 0.0), default=0.0)
        recurring_norms = [str(item).lower() for item in relationship_state.get("recurring_norms", [])]
        contribution = PolicyContribution(
            warmth=(warmth - 0.5) * 0.26 - rupture * 0.18,
            directness=0.0,
            curiosity=0.0,
            uncertainty=rupture * 0.10,
            disclosure=(trust - 0.5) * 0.18 + (familiarity - 0.3) * 0.10 - rupture * 0.16,
        )
        if any("direct" in norm or "checklist" in norm or "concision" in norm for norm in recurring_norms):
            contribution.directness += 0.10
        if any("warm" in norm or "kind" in norm or "empathy" in norm for norm in recurring_norms):
            contribution.warmth += 0.08
        return contribution

    def _compose_self_model_contribution(self, self_model_state: Mapping[str, Any]) -> PolicyContribution:
        self_regard = _clamp_signed(self_model_state.get("self_regard", 0.0), default=0.0)
        identity_stability = _clamp01(self_model_state.get("identity_stability", 0.5), default=0.5)
        stated_values = [str(item).lower() for item in self_model_state.get("stated_values", [])]
        contribution = PolicyContribution(
            warmth=self_regard * 0.08,
            directness=(identity_stability - 0.5) * 0.10,
            curiosity=0.0,
            uncertainty=(1.0 - identity_stability) * 0.16,
            disclosure=max(0.0, -self_regard) * 0.04,
        )
        if "clarity" in stated_values or "truth" in stated_values:
            contribution.directness += 0.06
        if "learning" in stated_values or "curiosity" in stated_values:
            contribution.curiosity += 0.08
        if "empathy" in stated_values or "care" in stated_values:
            contribution.warmth += 0.08
        return contribution

    def _compose_narrative_contribution(self, narrative_state: Mapping[str, Any]) -> PolicyContribution:
        active_themes = [str(item).lower() for item in narrative_state.get("active_themes", [])]
        ongoing_struggles = [str(item).lower() for item in narrative_state.get("ongoing_struggles", [])]
        contribution = PolicyContribution()
        if active_themes:
            contribution.curiosity += min(0.10, len(active_themes) * 0.02)
        if ongoing_struggles:
            contribution.uncertainty += min(0.08, len(ongoing_struggles) * 0.02)
            contribution.disclosure += min(0.06, len(ongoing_struggles) * 0.015)
        return contribution

    def _compose_dynamic_state(self, contributions: Mapping[str, PolicyContribution]) -> PolicyVector:
        dynamic = PolicyVector()
        for contribution in contributions.values():
            dynamic.warmth = _clamp01(dynamic.warmth + contribution.warmth, default=dynamic.warmth)
            dynamic.directness = _clamp01(dynamic.directness + contribution.directness, default=dynamic.directness)
            dynamic.curiosity = _clamp01(dynamic.curiosity + contribution.curiosity, default=dynamic.curiosity)
            dynamic.uncertainty = _clamp01(dynamic.uncertainty + contribution.uncertainty, default=dynamic.uncertainty)
            dynamic.disclosure = _clamp01(dynamic.disclosure + contribution.disclosure, default=dynamic.disclosure)
        return dynamic

    def _compose_effective(self, stable: PolicyVector, dynamic: PolicyVector) -> PolicyVector:
        return PolicyVector(
            warmth=(stable.warmth * 0.45) + (dynamic.warmth * 0.55),
            directness=(stable.directness * 0.45) + (dynamic.directness * 0.55),
            curiosity=(stable.curiosity * 0.40) + (dynamic.curiosity * 0.60),
            uncertainty=(stable.uncertainty * 0.50) + (dynamic.uncertainty * 0.50),
            disclosure=(stable.disclosure * 0.45) + (dynamic.disclosure * 0.55),
        )

    def _build_working_self_snapshot(
        self,
        *,
        drive_state: Mapping[str, Any],
        mood_state: Mapping[str, Any],
        relationship_state: Mapping[str, Any],
        self_model_state: Mapping[str, Any],
        narrative_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        drive_levels = _as_mapping(drive_state.get("levels", {}))
        dominant_drive = max(drive_levels, key=drive_levels.get) if drive_levels else None
        relationship_strength = round(
            (
                _clamp01(relationship_state.get("warmth", 0.5), default=0.5)
                + _clamp01(relationship_state.get("trust", 0.5), default=0.5)
                + _clamp01(relationship_state.get("familiarity", 0.0), default=0.0)
                + (1.0 - _clamp01(relationship_state.get("rupture", 0.0), default=0.0))
            )
            / 4.0,
            4,
        )
        return {
            "dominant_drive": dominant_drive,
            "drive_pressure": round(_clamp01(drive_state.get("total_pressure", 0.0), default=0.0), 4),
            "mood_label": mood_state.get("label"),
            "mood_confidence": round(_clamp01(mood_state.get("confidence", 0.0), default=0.0), 4),
            "relationship_strength": relationship_strength,
            "self_regard": round(_clamp_signed(self_model_state.get("self_regard", 0.0), default=0.0), 4),
            "identity_stability": round(_clamp01(self_model_state.get("identity_stability", 0.5), default=0.5), 4),
            "active_themes": list(narrative_state.get("active_themes", [])),
            "ongoing_struggles": list(narrative_state.get("ongoing_struggles", [])),
        }

    def _build_trace(
        self,
        *,
        drive_state: Mapping[str, Any],
        mood_state: Mapping[str, Any],
        relationship_state: Mapping[str, Any],
        self_model_state: Mapping[str, Any],
        narrative_state: Mapping[str, Any],
        contributions: Mapping[str, PolicyContribution],
        stable: PolicyVector,
        dynamic: PolicyVector,
        effective: PolicyVector,
    ) -> dict[str, Any]:
        contribution_dict = {key: value.to_dict() for key, value in contributions.items()}
        dominant_signals = sorted(
            (
                {
                    "source": source,
                    "magnitude": round(max(abs(v) for v in value.values()), 4),
                }
                for source, value in contribution_dict.items()
            ),
            key=lambda entry: entry["magnitude"],
            reverse=True,
        )
        return {
            "working_self": self._build_working_self_snapshot(
                drive_state=drive_state,
                mood_state=mood_state,
                relationship_state=relationship_state,
                self_model_state=self_model_state,
                narrative_state=narrative_state,
            ),
            "inputs": {
                "drive_signals": dict(_as_mapping(drive_state.get("levels", {}))),
                "mood": {
                    "label": mood_state.get("label"),
                    "valence": round(_clamp_signed(mood_state.get("valence", 0.0), default=0.0), 4),
                    "arousal": round(_clamp01(mood_state.get("arousal", 0.0), default=0.0), 4),
                    "confidence": round(_clamp01(mood_state.get("confidence", 0.0), default=0.0), 4),
                },
                "relationship": dict(relationship_state),
                "self_model": {
                    "self_regard": round(_clamp_signed(self_model_state.get("self_regard", 0.0), default=0.0), 4),
                    "identity_stability": round(_clamp01(self_model_state.get("identity_stability", 0.5), default=0.5), 4),
                    "stated_values": list(self_model_state.get("stated_values", [])),
                },
                "narrative": {
                    "active_themes": list(narrative_state.get("active_themes", [])),
                    "ongoing_struggles": list(narrative_state.get("ongoing_struggles", [])),
                },
            },
            "contributions": contribution_dict,
            "dominant_signals": dominant_signals[:3],
            "policy_vectors": {
                "stable": stable.to_dict(),
                "dynamic": dynamic.to_dict(),
                "effective": effective.to_dict(),
            },
            "trace_version": "mp-302-v1",
        }


def build_response_policy(**kwargs: Any) -> ResponsePolicy:
    return PolicyComposer().compose(**kwargs)