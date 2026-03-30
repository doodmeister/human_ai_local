from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _from_timestamp_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError):
        return None


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@dataclass(slots=True)
class RelationshipMemory:
    interlocutor_id: str
    display_name: str = ""
    warmth: float = 0.5
    trust: float = 0.5
    familiarity: float = 0.0
    rupture: float = 0.0
    recurring_norms: list[str] = field(default_factory=list)
    interaction_count: int = 0
    first_interaction: datetime | None = None
    last_interaction: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        interlocutor_id = str(self.interlocutor_id).strip()
        if not interlocutor_id:
            raise ValueError("interlocutor_id must be a non-empty string")
        self.interlocutor_id = interlocutor_id
        self.display_name = str(self.display_name or "").strip()
        self.warmth = _clamp_score(self.warmth)
        self.trust = _clamp_score(self.trust)
        self.familiarity = _clamp_score(self.familiarity)
        self.rupture = _clamp_score(self.rupture)
        self.interaction_count = max(0, int(self.interaction_count))
        self.first_interaction = _coerce_datetime(self.first_interaction)
        self.last_interaction = _coerce_datetime(self.last_interaction)
        self.recurring_norms = _dedupe_strings(list(self.recurring_norms))
        self.metadata = dict(self.metadata)

    def record_interaction(self, at: datetime | None = None, count: int = 1) -> None:
        timestamp = at or datetime.now()
        increment = max(0, int(count))
        if increment == 0:
            return
        self.interaction_count += increment
        if self.first_interaction is None:
            self.first_interaction = timestamp
        self.last_interaction = timestamp

    def merge_norms(self, norms: list[str]) -> None:
        self.recurring_norms = _dedupe_strings([*self.recurring_norms, *norms])

    def retrieval_features(self) -> dict[str, float]:
        relationship_strength = (self.warmth + self.trust + self.familiarity + (1.0 - self.rupture)) / 4.0
        return {
            "warmth": self.warmth,
            "trust": self.trust,
            "familiarity": self.familiarity,
            "rupture": self.rupture,
            "relationship_strength": round(_clamp_score(relationship_strength), 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "interlocutor_id": self.interlocutor_id,
            "display_name": self.display_name,
            "warmth": self.warmth,
            "trust": self.trust,
            "familiarity": self.familiarity,
            "rupture": self.rupture,
            "recurring_norms": list(self.recurring_norms),
            "interaction_count": self.interaction_count,
            "first_interaction": _iso_or_none(self.first_interaction),
            "last_interaction": _iso_or_none(self.last_interaction),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipMemory:
        return cls(
            interlocutor_id=str(data.get("interlocutor_id") or ""),
            display_name=str(data.get("display_name") or ""),
            warmth=float(data.get("warmth", 0.5) or 0.5),
            trust=float(data.get("trust", 0.5) or 0.5),
            familiarity=float(data.get("familiarity", 0.0) or 0.0),
            rupture=float(data.get("rupture", 0.0) or 0.0),
            recurring_norms=list(data.get("recurring_norms", [])),
            interaction_count=int(data.get("interaction_count", 0) or 0),
            first_interaction=_coerce_datetime(data.get("first_interaction")),
            last_interaction=_coerce_datetime(data.get("last_interaction")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_relational_model(cls, model: Any) -> RelationshipMemory:
        felt_quality = float(getattr(model, "felt_quality", 0.0) or 0.0)
        attachment_strength = float(getattr(model, "attachment_strength", 0.0) or 0.0)
        interaction_count = int(getattr(model, "interaction_count", 0) or 0)
        status = str(getattr(model, "current_status", "active") or "active")
        warmth = _clamp_score((felt_quality + 1.0) / 2.0)
        trust = _clamp_score(0.5 + (felt_quality * 0.2) + (attachment_strength * 0.3))
        familiarity = _clamp_score(max(attachment_strength, min(1.0, interaction_count / 20.0)))
        rupture = _clamp_score(max(0.0, -felt_quality, 1.0 - warmth if status == "strained" else 0.0))
        return cls(
            interlocutor_id=str(getattr(model, "person_id", "")),
            display_name=str(getattr(model, "person_name", "") or ""),
            warmth=warmth,
            trust=trust,
            familiarity=familiarity,
            rupture=rupture,
            recurring_norms=list(getattr(model, "recurring_patterns", []) or []),
            interaction_count=interaction_count,
            first_interaction=_from_timestamp_or_none(getattr(model, "first_interaction_ts", None)),
            last_interaction=_from_timestamp_or_none(getattr(model, "last_interaction_ts", None)),
            metadata={
                "status": status,
                "drive_effects": dict(getattr(model, "drive_effects", {}) or {}),
                "significant_moment_ids": list(getattr(model, "significant_moment_ids", []) or []),
            },
        )