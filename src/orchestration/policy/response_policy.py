from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _clamp_signed(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(-1.0, min(1.0, numeric))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        maybe_mapping = value.to_dict()
        if isinstance(maybe_mapping, Mapping):
            return maybe_mapping
    return {}


@dataclass(slots=True)
class PolicyVector:
    warmth: float = 0.5
    directness: float = 0.5
    curiosity: float = 0.5
    uncertainty: float = 0.5
    disclosure: float = 0.5

    def __post_init__(self) -> None:
        self.warmth = _clamp01(self.warmth, default=0.5)
        self.directness = _clamp01(self.directness, default=0.5)
        self.curiosity = _clamp01(self.curiosity, default=0.5)
        self.uncertainty = _clamp01(self.uncertainty, default=0.5)
        self.disclosure = _clamp01(self.disclosure, default=0.5)

    def to_dict(self) -> dict[str, float]:
        return {
            "warmth": round(self.warmth, 4),
            "directness": round(self.directness, 4),
            "curiosity": round(self.curiosity, 4),
            "uncertainty": round(self.uncertainty, 4),
            "disclosure": round(self.disclosure, 4),
        }

    def combine(self, delta: PolicyVector) -> PolicyVector:
        return PolicyVector(
            warmth=self.warmth + delta.warmth,
            directness=self.directness + delta.directness,
            curiosity=self.curiosity + delta.curiosity,
            uncertainty=self.uncertainty + delta.uncertainty,
            disclosure=self.disclosure + delta.disclosure,
        )


@dataclass(slots=True)
class ResponsePolicy:
    stable_traits: PolicyVector = field(default_factory=PolicyVector)
    dynamic_state: PolicyVector = field(default_factory=PolicyVector)
    effective: PolicyVector = field(default_factory=PolicyVector)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_traits": self.stable_traits.to_dict(),
            "dynamic_state": self.dynamic_state.to_dict(),
            "effective": self.effective.to_dict(),
            "trace": dict(self.trace),
        }


def build_response_policy(
    *,
    stable_traits: Mapping[str, Any] | None = None,
    drive_state: Mapping[str, Any] | Any = None,
    mood_state: Mapping[str, Any] | Any = None,
    relationship_state: Mapping[str, Any] | Any = None,
    self_model_state: Mapping[str, Any] | Any = None,
    narrative_state: Mapping[str, Any] | Any = None,
) -> ResponsePolicy:
    from .policy_composer import PolicyComposer

    return PolicyComposer().compose(
        stable_traits=stable_traits,
        drive_state=drive_state,
        mood_state=mood_state,
        relationship_state=relationship_state,
        self_model_state=self_model_state,
        narrative_state=narrative_state,
    )