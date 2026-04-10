from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .enums import CognitiveActType, GoalKind, InputType, PolicyName, ScheduledTaskStatus

__all__ = [
    "Stimulus",
    "ContradictionRecord",
    "RetrievalContextItem",
    "FocusItem",
    "MemoryStatus",
    "AttentionStatus",
    "GoalMetadata",
    "CognitiveGoal",
    "CognitiveActProposal",
    "SelfModel",
    "ScheduledCognitiveTask",
    "InternalStateSnapshot",
    "WorkspaceState",
    "PlanMetadata",
    "CognitivePlan",
    "MemoryUpdate",
    "AttentionUpdate",
    "ExecutionResult",
    "CriticReport",
    "MetacognitiveCycleResult",
]


def _normalize_unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return normalized


def _coerce_enum(name: str, enum_type: type[Enum], value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be one of {[member.value for member in enum_type]}, got {value!r}") from exc


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        normalized = float(value)
        return normalized if math.isfinite(normalized) else None
    try:
        normalized = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _normalize_flat_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


@dataclass(slots=True)
class ContradictionRecord:
    """Normalized contradiction detail shared across workspace and critic data."""

    kind: str
    description: str = ""
    severity: float | str | None = None
    confidence: float | None = None
    contradiction_set_id: str | None = None
    source_system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.description = str(self.description or "").strip()
        self.kind = str(self.kind or self.description or "unknown").strip() or "unknown"
        self.contradiction_set_id = _coerce_optional_text(self.contradiction_set_id)
        self.source_system = _coerce_optional_text(self.source_system)
        if isinstance(self.severity, (int, float)) and not isinstance(self.severity, bool):
            normalized = float(self.severity)
            self.severity = normalized if math.isfinite(normalized) else None
        elif self.severity is not None:
            self.severity = _coerce_optional_text(self.severity)
        self.confidence = _coerce_optional_float(self.confidence)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any) -> "ContradictionRecord":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            payload = dict(value)
            raw_kind = payload.pop("kind", payload.pop("type", None))
            raw_description = payload.pop("description", payload.pop("summary", payload.pop("value", None)))
            if raw_kind is None and raw_description is not None:
                raw_kind = raw_description
            return cls(
                kind=str(raw_kind or "unknown"),
                description="" if raw_description is None else str(raw_description),
                severity=payload.pop("severity", None),
                confidence=payload.pop("confidence", None),
                contradiction_set_id=payload.pop("contradiction_set_id", None),
                source_system=payload.pop("source_system", None),
                metadata=payload,
            )
        text = str(value or "").strip()
        return cls(kind=text or "unknown")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.description:
            payload["description"] = self.description
        if self.severity is not None:
            payload["severity"] = self.severity
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.contradiction_set_id is not None:
            payload["contradiction_set_id"] = self.contradiction_set_id
        if self.source_system is not None:
            payload["source_system"] = self.source_system
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload


def _normalize_contradictions(values: Any) -> tuple[ContradictionRecord, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(ContradictionRecord.from_value(item) for item in values)
    return (ContradictionRecord.from_value(values),)


@dataclass(slots=True)
class RetrievalContextItem(Mapping[str, Any]):
    """Normalized memory retrieval item carried through the workspace."""

    content: str
    source_system: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = str(self.content or "")
        self.source_system = _coerce_optional_text(self.source_system)
        self.source_id = _coerce_optional_text(self.source_id)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any, *, source_system: str | None = None) -> "RetrievalContextItem":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
            raw_metadata = payload.pop("metadata", None)
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            content = payload.pop("content", payload.pop("text", payload.pop("value", "")))
            item_source_system = payload.pop("source_system", source_system)
            source_id = payload.pop("source_id", payload.pop("id", None))
            metadata.update({str(key): item for key, item in payload.items()})
            return cls(
                content=str(content),
                source_system=item_source_system,
                source_id=source_id,
                metadata=metadata,
            )
        return cls(content=str(value), source_system=source_system)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": self.content}
        if self.source_system is not None:
            payload["source_system"] = self.source_system
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def _normalize_context_items(values: Any) -> tuple[RetrievalContextItem, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(RetrievalContextItem.from_value(item) for item in values)
    return (RetrievalContextItem.from_value(values),)


@dataclass(slots=True)
class FocusItem(Mapping[str, Any]):
    """Normalized attention focus item used in snapshots and workspaces."""

    label: str
    item_id: str | None = None
    kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = str(self.label or "").strip()
        self.item_id = _coerce_optional_text(self.item_id)
        self.kind = _coerce_optional_text(self.kind)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any) -> "FocusItem":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
            raw_metadata = payload.pop("metadata", None)
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            label = payload.pop("label", payload.pop("name", payload.pop("title", payload.pop("content", payload.pop("value", "")))))
            item_id = payload.pop("item_id", payload.pop("id", None))
            kind = payload.pop("kind", payload.pop("type", None))
            metadata.update({str(key): item for key, item in payload.items()})
            return cls(label=str(label), item_id=item_id, kind=kind, metadata=metadata)
        return cls(label=str(value or ""))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label}
        if self.item_id is not None:
            payload["item_id"] = self.item_id
        if self.kind is not None:
            payload["kind"] = self.kind
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload

    def to_primitive(self) -> Any:
        if self.item_id is None and self.kind is None and not self.metadata:
            return self.label
        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def _normalize_focus_items(values: Any) -> tuple[FocusItem, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(FocusItem.from_value(item) for item in values)
    return (FocusItem.from_value(values),)


def _normalize_status_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class MemoryStatus(Mapping[str, Any]):
    """Typed snapshot of memory-system status with flat-dict compatibility."""

    uncertainty: float | None = None
    health: str | None = None
    stm: dict[str, Any] = field(default_factory=dict)
    ltm: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.uncertainty = _coerce_optional_float(self.uncertainty)
        self.health = _coerce_optional_text(self.health)
        self.stm = dict(self.stm or {})
        self.ltm = dict(self.ltm or {})
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any, *, canonical_uncertainty: float | None = None) -> "MemoryStatus":
        if isinstance(value, cls):
            status = cls(
                uncertainty=value.uncertainty,
                health=value.health,
                stm=value.stm,
                ltm=value.ltm,
                metadata=value.metadata,
            )
            if canonical_uncertainty is not None and value.uncertainty is not None:
                status.uncertainty = canonical_uncertainty
            return status
        if not isinstance(value, Mapping):
            return cls()

        payload = dict(value)
        raw_metadata = payload.pop("metadata", None)
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        has_uncertainty = "uncertainty" in payload
        uncertainty = payload.pop("uncertainty", None)
        health = payload.pop("health", None)
        stm = payload.pop("stm", {})
        ltm = payload.pop("ltm", {})
        metadata.update({str(key): item for key, item in payload.items()})

        status = cls(
            uncertainty=uncertainty,
            health=health,
            stm=dict(stm) if isinstance(stm, Mapping) else {},
            ltm=dict(ltm) if isinstance(ltm, Mapping) else {},
            metadata=metadata,
        )
        if canonical_uncertainty is not None and has_uncertainty:
            status.uncertainty = canonical_uncertainty
        return status

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.health is not None:
            payload["health"] = self.health
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
        if self.stm:
            payload["stm"] = dict(self.stm)
        if self.ltm:
            payload["ltm"] = dict(self.ltm)
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass(slots=True)
class AttentionStatus(Mapping[str, Any]):
    """Typed snapshot of attention-system status with flat-dict compatibility."""

    cognitive_load: float | None = None
    current_focus: tuple[FocusItem, ...] = field(default_factory=tuple)
    focused_items: tuple[FocusItem, ...] = field(default_factory=tuple)
    available_capacity: float | None = None
    fatigue_level: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cognitive_load = _coerce_optional_float(self.cognitive_load)
        self.current_focus = _normalize_focus_items(self.current_focus)
        self.focused_items = _normalize_focus_items(self.focused_items)
        self.available_capacity = _coerce_optional_float(self.available_capacity)
        self.fatigue_level = _coerce_optional_float(self.fatigue_level)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any, *, canonical_cognitive_load: float | None = None) -> "AttentionStatus":
        if isinstance(value, cls):
            status = cls(
                cognitive_load=value.cognitive_load,
                current_focus=value.current_focus,
                focused_items=value.focused_items,
                available_capacity=value.available_capacity,
                fatigue_level=value.fatigue_level,
                metadata=value.metadata,
            )
            if canonical_cognitive_load is not None and value.cognitive_load is not None:
                status.cognitive_load = canonical_cognitive_load
            return status
        if not isinstance(value, Mapping):
            return cls()

        payload = dict(value)
        raw_metadata = payload.pop("metadata", None)
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        has_cognitive_load = "cognitive_load" in payload
        cognitive_load = payload.pop("cognitive_load", None)
        current_focus = payload.pop("current_focus", ())
        focused_items = payload.pop("focused_items", ())
        available_capacity = payload.pop("available_capacity", None)
        fatigue_level = payload.pop("fatigue_level", None)
        metadata.update({str(key): item for key, item in payload.items()})

        status = cls(
            cognitive_load=cognitive_load,
            current_focus=_normalize_focus_items(current_focus),
            focused_items=_normalize_focus_items(focused_items),
            available_capacity=available_capacity,
            fatigue_level=fatigue_level,
            metadata=metadata,
        )
        if canonical_cognitive_load is not None and has_cognitive_load:
            status.cognitive_load = canonical_cognitive_load
        return status

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.cognitive_load is not None:
            payload["cognitive_load"] = self.cognitive_load
        if self.current_focus:
            payload["current_focus"] = [item.to_primitive() for item in self.current_focus]
        if self.focused_items:
            payload["focused_items"] = [item.to_primitive() for item in self.focused_items]
        if self.available_capacity is not None:
            payload["available_capacity"] = self.available_capacity
        if self.fatigue_level is not None:
            payload["fatigue_level"] = self.fatigue_level
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass(slots=True)
class GoalMetadata(Mapping[str, Any]):
    """Typed goal metadata with flat-dict compatibility for scoring fields."""

    source: str | None = None
    contradiction_count: int | None = None
    title: str | None = None
    status: str | None = None
    progress: float | None = None
    source_kind: str | None = None
    heuristic_score: float | None = None
    heuristic_score_raw: float | None = None
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = _coerce_optional_text(self.source)
        self.contradiction_count = _coerce_optional_int(self.contradiction_count)
        self.title = _coerce_optional_text(self.title)
        self.status = _coerce_optional_text(self.status)
        self.progress = _coerce_optional_float(self.progress)
        self.source_kind = _coerce_optional_text(self.source_kind)
        self.heuristic_score = _coerce_optional_float(self.heuristic_score)
        self.heuristic_score_raw = _coerce_optional_float(self.heuristic_score_raw)
        self.score_breakdown = _normalize_flat_mapping(self.score_breakdown)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any) -> "GoalMetadata":
        if isinstance(value, cls):
            return cls(
                source=value.source,
                contradiction_count=value.contradiction_count,
                title=value.title,
                status=value.status,
                progress=value.progress,
                source_kind=value.source_kind,
                heuristic_score=value.heuristic_score,
                heuristic_score_raw=value.heuristic_score_raw,
                score_breakdown=value.score_breakdown,
                metadata=value.metadata,
            )
        if not isinstance(value, Mapping):
            return cls()

        payload = dict(value)
        raw_metadata = payload.pop("metadata", None)
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.update({str(key): item for key, item in payload.items() if key not in {
            "source",
            "contradiction_count",
            "title",
            "status",
            "progress",
            "source_kind",
            "heuristic_score",
            "heuristic_score_raw",
            "score_breakdown",
        }})
        return cls(
            source=payload.get("source"),
            contradiction_count=payload.get("contradiction_count"),
            title=payload.get("title"),
            status=payload.get("status"),
            progress=payload.get("progress"),
            source_kind=payload.get("source_kind"),
            heuristic_score=payload.get("heuristic_score"),
            heuristic_score_raw=payload.get("heuristic_score_raw"),
            score_breakdown=payload.get("score_breakdown", {}),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.source is not None:
            payload["source"] = self.source
        if self.contradiction_count is not None:
            payload["contradiction_count"] = self.contradiction_count
        if self.title is not None:
            payload["title"] = self.title
        if self.status is not None:
            payload["status"] = self.status
        if self.progress is not None:
            payload["progress"] = self.progress
        if self.source_kind is not None:
            payload["source_kind"] = self.source_kind
        if self.heuristic_score is not None:
            payload["heuristic_score"] = self.heuristic_score
        if self.heuristic_score_raw is not None:
            payload["heuristic_score_raw"] = self.heuristic_score_raw
        if self.score_breakdown:
            payload["score_breakdown"] = dict(self.score_breakdown)
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass(slots=True)
class PlanMetadata(Mapping[str, Any]):
    """Typed plan metadata with flat-dict compatibility for selection details."""

    contradiction_count: int | None = None
    uncertainty: float | None = None
    cognitive_load: float | None = None
    selection_strategy: str | None = None
    degenerate_plan: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.contradiction_count = _coerce_optional_int(self.contradiction_count)
        self.uncertainty = _coerce_optional_float(self.uncertainty)
        self.cognitive_load = _coerce_optional_float(self.cognitive_load)
        self.selection_strategy = _coerce_optional_text(self.selection_strategy)
        self.degenerate_plan = _coerce_optional_bool(self.degenerate_plan)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any) -> "PlanMetadata":
        if isinstance(value, cls):
            return cls(
                contradiction_count=value.contradiction_count,
                uncertainty=value.uncertainty,
                cognitive_load=value.cognitive_load,
                selection_strategy=value.selection_strategy,
                degenerate_plan=value.degenerate_plan,
                metadata=value.metadata,
            )
        if not isinstance(value, Mapping):
            return cls()

        payload = dict(value)
        raw_metadata = payload.pop("metadata", None)
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.update({str(key): item for key, item in payload.items() if key not in {
            "contradiction_count",
            "uncertainty",
            "cognitive_load",
            "selection_strategy",
            "degenerate_plan",
        }})
        return cls(
            contradiction_count=payload.get("contradiction_count"),
            uncertainty=payload.get("uncertainty"),
            cognitive_load=payload.get("cognitive_load"),
            selection_strategy=payload.get("selection_strategy"),
            degenerate_plan=payload.get("degenerate_plan"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.contradiction_count is not None:
            payload["contradiction_count"] = self.contradiction_count
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
        if self.cognitive_load is not None:
            payload["cognitive_load"] = self.cognitive_load
        if self.selection_strategy is not None:
            payload["selection_strategy"] = self.selection_strategy
        if self.degenerate_plan is not None:
            payload["degenerate_plan"] = self.degenerate_plan
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass(slots=True)
class MemoryUpdate:
    """Structured record of a memory-side effect emitted by execution."""

    description: str
    source_act_type: CognitiveActType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.description = str(self.description or "").strip()
        if self.source_act_type is not None:
            self.source_act_type = _coerce_enum("source_act_type", CognitiveActType, self.source_act_type)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any) -> "MemoryUpdate":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
            raw_metadata = payload.pop("metadata", None)
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            description = payload.pop("description", payload.pop("content", payload.pop("value", "")))
            source_act_type = payload.pop("source_act_type", payload.pop("act_type", None))
            metadata.update({str(key): item for key, item in payload.items()})
            return cls(description=str(description), source_act_type=source_act_type, metadata=metadata)
        return cls(description=str(value or ""))

    def to_dict(self) -> dict[str, Any]:
        payload = {"description": self.description}
        if self.source_act_type is not None:
            payload["source_act_type"] = self.source_act_type.value
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload


@dataclass(slots=True)
class AttentionUpdate:
    """Structured record of an attention-side effect emitted by execution."""

    changes: dict[str, Any] = field(default_factory=dict)
    source_act_type: CognitiveActType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.changes = dict(self.changes or {})
        if self.source_act_type is not None:
            self.source_act_type = _coerce_enum("source_act_type", CognitiveActType, self.source_act_type)
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(cls, value: Any) -> "AttentionUpdate":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
            raw_metadata = payload.pop("metadata", None)
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            source_act_type = payload.pop("source_act_type", payload.pop("act_type", None))
            raw_changes = payload.pop("changes", None)
            if raw_changes is None:
                changes = {str(key): item for key, item in payload.items()}
            else:
                changes = dict(raw_changes) if isinstance(raw_changes, Mapping) else {"value": raw_changes}
                metadata.update({str(key): item for key, item in payload.items()})
            return cls(changes=changes, source_act_type=source_act_type, metadata=metadata)
        return cls(changes={"value": value})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"changes": dict(self.changes)}
        if self.source_act_type is not None:
            payload["source_act_type"] = self.source_act_type.value
        for key, value in self.metadata.items():
            payload.setdefault(str(key), value)
        return payload


def _normalize_memory_updates(values: Any) -> tuple[MemoryUpdate, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(MemoryUpdate.from_value(item) for item in values)
    return (MemoryUpdate.from_value(values),)


def _normalize_attention_updates(values: Any) -> tuple[AttentionUpdate, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(AttentionUpdate.from_value(item) for item in values)
    return (AttentionUpdate.from_value(values),)


@dataclass(slots=True)
class Stimulus:
    """Input to a metacognitive cycle.

    Well-known metadata keys:
      - `background_task`: marks scheduler-launched cycles.
      - `task_id`: identifies the originating scheduled task.
      - `task_reason`: records the originating scheduled task reason code.
    """

    session_id: str
    user_input: str
    input_type: InputType = InputType.TEXT
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_type = _coerce_enum("input_type", InputType, self.input_type)


@dataclass(slots=True)
class CognitiveGoal:
    """Rankable cognitive goal for the current cycle."""

    goal_id: str
    description: str
    priority: float
    kind: GoalKind = GoalKind.RESPONSE
    urgency: float = 0.0
    salience: float = 0.0
    rationale: str = ""
    metadata: GoalMetadata = field(default_factory=GoalMetadata)

    def __post_init__(self) -> None:
        self.kind = _coerce_enum("kind", GoalKind, self.kind)
        self.priority = _normalize_unit_interval("priority", self.priority)
        self.urgency = _normalize_unit_interval("urgency", self.urgency)
        self.salience = _normalize_unit_interval("salience", self.salience)
        self.metadata = GoalMetadata.from_value(self.metadata)


@dataclass(slots=True)
class CognitiveActProposal:
    """Candidate act proposed by the policy engine."""

    act_type: CognitiveActType
    description: str
    priority_score: float
    rationale: str = ""
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.act_type = _coerce_enum("act_type", CognitiveActType, self.act_type)
        self.priority_score = _normalize_unit_interval("priority_score", self.priority_score)
        self.target = _coerce_optional_text(self.target)
        self.metadata = dict(self.metadata or {})

    @property
    def target_goal_id(self) -> str | None:
        return self.target

    @target_goal_id.setter
    def target_goal_id(self, value: Any) -> None:
        self.target = _coerce_optional_text(value)


@dataclass(slots=True)
class SelfModel:
    """Compact per-session self-model updated from critic feedback."""

    session_id: str
    confidence: float = 0.5
    traits: dict[str, float] = field(default_factory=dict)
    beliefs: tuple[str, ...] = field(default_factory=tuple)
    recent_updates: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _normalize_unit_interval("confidence", self.confidence)


@dataclass(slots=True)
class ScheduledCognitiveTask:
    """Deferred background task scheduled from critic output."""

    task_id: str
    description: str
    status: ScheduledTaskStatus = ScheduledTaskStatus.PENDING
    priority: float = 0.0
    due_at: float | None = None
    goal_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _coerce_enum("status", ScheduledTaskStatus, self.status)
        self.priority = _normalize_unit_interval("priority", self.priority)


@dataclass(slots=True)
class InternalStateSnapshot:
    """Structured snapshot assembled before workspace construction."""

    session_id: str
    turn_index: int
    memory_status: MemoryStatus = field(default_factory=MemoryStatus)
    attention_status: AttentionStatus = field(default_factory=AttentionStatus)
    active_goals: tuple[CognitiveGoal, ...] = field(default_factory=tuple)
    self_model: SelfModel = field(default_factory=lambda: SelfModel(session_id="default"))
    pending_tasks: tuple[ScheduledCognitiveTask, ...] = field(default_factory=tuple)
    cognitive_load: float = 0.0
    uncertainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cognitive_load = _normalize_unit_interval("cognitive_load", self.cognitive_load)
        self.uncertainty = _normalize_unit_interval("uncertainty", self.uncertainty)
        self.memory_status = MemoryStatus.from_value(self.memory_status, canonical_uncertainty=self.uncertainty)
        self.attention_status = AttentionStatus.from_value(
            self.attention_status,
            canonical_cognitive_load=self.cognitive_load,
        )
        if self.self_model.session_id == "default":
            self.self_model = replace(self.self_model, session_id=self.session_id)


@dataclass(slots=True)
class WorkspaceState:
    """Per-cycle workspace combining the stimulus, snapshot, and retrieved context."""

    stimulus: Stimulus
    snapshot: InternalStateSnapshot
    context_items: tuple[RetrievalContextItem, ...] = field(default_factory=tuple)
    focus_items: tuple[FocusItem, ...] = field(default_factory=tuple)
    contradictions: tuple[ContradictionRecord, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    dominant_goal: CognitiveGoal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.context_items = _normalize_context_items(self.context_items)
        self.focus_items = _normalize_focus_items(self.focus_items)
        self.contradictions = _normalize_contradictions(self.contradictions)


@dataclass(slots=True)
class CognitivePlan:
    """Selected plan of cognitive acts for the current cycle."""

    selected_goal: CognitiveGoal | None = None
    acts: tuple[CognitiveActProposal, ...] = field(default_factory=tuple)
    policy_name: PolicyName = PolicyName.UNSET
    rationale: str = ""
    metadata: PlanMetadata = field(default_factory=PlanMetadata)

    def __post_init__(self) -> None:
        self.policy_name = _coerce_enum("policy_name", PolicyName, self.policy_name)
        self.metadata = PlanMetadata.from_value(self.metadata)


@dataclass(slots=True)
class ExecutionResult:
    """Execution outcome emitted by the plan executor."""

    success: bool
    response_text: str = ""
    executed_acts: tuple[CognitiveActProposal, ...] = field(default_factory=tuple)
    memory_updates: tuple[MemoryUpdate, ...] = field(default_factory=tuple)
    attention_updates: tuple[AttentionUpdate, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory_updates = _normalize_memory_updates(self.memory_updates)
        self.attention_updates = _normalize_attention_updates(self.attention_updates)


@dataclass(slots=True)
class CriticReport:
    """Evaluation summary produced after a cycle completes."""

    cycle_id: str
    success_score: float
    # Wall-clock timestamp; persisted and compared across runs. Callers
    # tolerant to clock skew only. Do NOT substitute time.monotonic().
    timestamp: float = field(default_factory=time.time)
    goal_progress: float = 0.0
    follow_up_recommended: bool = False
    contradictions_detected: tuple[ContradictionRecord, ...] = field(default_factory=tuple)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.success_score = _normalize_unit_interval("success_score", self.success_score)
        self.goal_progress = _normalize_unit_interval("goal_progress", self.goal_progress)
        self.contradictions_detected = _normalize_contradictions(self.contradictions_detected)


@dataclass(slots=True)
class MetacognitiveCycleResult:
    """Aggregate outputs for one completed metacognitive cycle."""

    cycle_id: str
    workspace: WorkspaceState | None
    ranked_goals: tuple[CognitiveGoal, ...] = field(default_factory=tuple)
    plan: CognitivePlan | None = None
    execution_result: ExecutionResult | None = None
    critic_report: CriticReport | None = None
    updated_self_model: SelfModel | None = None
    scheduled_tasks: tuple[ScheduledCognitiveTask, ...] = field(default_factory=tuple)
    # Populated after successful persistence by CycleTracer.write_trace. `None`
    # indicates the cycle completed but was not persisted.
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
