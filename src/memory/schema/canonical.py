from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping, Self


class MemoryKind(StrEnum):
    TRACE = "trace"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    SELF_MODEL = "self_model"
    GOAL = "goal"
    PROSPECTIVE = "prospective"


def _parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _validate_unit_interval(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    raise ValueError(f"{name} must be between 0.0 and 1.0")


def _validate_signed_unit_interval(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if -1.0 <= value <= 1.0:
        return value
    raise ValueError(f"{name} must be between -1.0 and 1.0")


@dataclass(slots=True)
class MemoryTimeInterval:
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        self.start = _parse_datetime(self.start)
        self.end = _parse_datetime(self.end)
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("time_interval end must not be earlier than start")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            start=_parse_datetime(data.get("start")),
            end=_parse_datetime(data.get("end")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }


@dataclass(slots=True)
class CanonicalMemoryItem:
    memory_id: str
    memory_kind: MemoryKind
    content: str
    summary: str | None = None
    entities: list[str] = field(default_factory=list)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    time_interval: MemoryTimeInterval | None = None
    encoding_time: datetime | None = None
    last_access: datetime | None = None
    confidence: float | None = None
    importance: float = 0.5
    emotional_valence: float = 0.0
    arousal: float | None = None
    source: str = "unknown"
    source_memory_ids: list[str] = field(default_factory=list)
    contradiction_set_id: str | None = None
    relationship_target: str | None = None
    goal_ids: list[str] = field(default_factory=list)
    narrative_role: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("memory_id is required")
        if not self.content:
            raise ValueError("content is required")
        if not isinstance(self.memory_kind, MemoryKind):
            self.memory_kind = MemoryKind(str(self.memory_kind))
        if self.time_interval is not None and not isinstance(self.time_interval, MemoryTimeInterval):
            raise TypeError("time_interval must be a MemoryTimeInterval")
        self.encoding_time = _parse_datetime(self.encoding_time)
        self.last_access = _parse_datetime(self.last_access)
        self.confidence = _validate_unit_interval("confidence", self.confidence)
        self.importance = _validate_unit_interval("importance", self.importance) or 0.0
        self.emotional_valence = _validate_signed_unit_interval(
            "emotional_valence",
            self.emotional_valence,
        ) or 0.0
        self.arousal = _validate_unit_interval("arousal", self.arousal)
        self.entities = list(self.entities)
        self.source_memory_ids = list(self.source_memory_ids)
        self.goal_ids = list(self.goal_ids)
        self.tags = list(self.tags)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        time_interval_data = data.get("time_interval")
        time_interval = None
        if isinstance(time_interval_data, Mapping):
            time_interval = MemoryTimeInterval.from_dict(time_interval_data)

        return cls(
            memory_id=str(data["memory_id"]),
            memory_kind=MemoryKind(str(data["memory_kind"])),
            content=str(data["content"]),
            summary=data.get("summary"),
            entities=list(data.get("entities", [])),
            subject=data.get("subject"),
            predicate=data.get("predicate"),
            object=data.get("object"),
            time_interval=time_interval,
            encoding_time=_parse_datetime(data.get("encoding_time")),
            last_access=_parse_datetime(data.get("last_access")),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            importance=float(data.get("importance", 0.5)),
            emotional_valence=float(data.get("emotional_valence", 0.0)),
            arousal=float(data["arousal"]) if data.get("arousal") is not None else None,
            source=str(data.get("source", "unknown")),
            source_memory_ids=list(data.get("source_memory_ids", data.get("source_event_ids", []))),
            contradiction_set_id=data.get("contradiction_set_id"),
            relationship_target=data.get("relationship_target"),
            goal_ids=list(data.get("goal_ids", [])),
            narrative_role=data.get("narrative_role"),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_kind": self.memory_kind.value,
            "content": self.content,
            "summary": self.summary,
            "entities": list(self.entities),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "time_interval": self.time_interval.to_dict() if self.time_interval else None,
            "encoding_time": self.encoding_time.isoformat() if self.encoding_time else None,
            "last_access": self.last_access.isoformat() if self.last_access else None,
            "confidence": self.confidence,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "arousal": self.arousal,
            "source": self.source,
            "source_memory_ids": list(self.source_memory_ids),
            "contradiction_set_id": self.contradiction_set_id,
            "relationship_target": self.relationship_target,
            "goal_ids": list(self.goal_ids),
            "narrative_role": self.narrative_role,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }