from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Mapping

from src.memory.schema import CanonicalMemoryItem, MemoryKind, MemoryTimeInterval


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class EventEncoder:
    def encode_episode(
        self,
        episode: Any,
        *,
        goal_ids: list[str] | None = None,
        relationship_target: str | None = None,
    ) -> CanonicalMemoryItem:
        data = _to_mapping(episode)
        context = data.get("context") or {}
        participants = list(context.get("participants", data.get("participants", [])) or [])
        target = relationship_target or (participants[0] if participants else None)
        event_goal_ids = list(goal_ids or data.get("goal_ids", []) or [])
        timestamp = _coerce_datetime(data.get("timestamp") or data.get("created_at"))
        summary = str(data.get("summary") or "")
        content = str(data.get("detailed_content") or summary)
        importance = float(data.get("importance", 0.5) or 0.5)
        emotional_valence = float(data.get("emotional_valence", 0.0) or 0.0)
        metadata = dict(data)
        metadata["participants"] = participants
        metadata["life_period"] = data.get("life_period") or "general"
        metadata["related_episodes"] = list(data.get("related_episodes", []) or [])
        metadata["defining_moment"] = bool(
            data.get("defining_moment")
            or importance >= 0.75
            or abs(emotional_valence) >= 0.6
        )
        return CanonicalMemoryItem(
            memory_id=str(data.get("id") or data.get("memory_id") or "episode-unknown"),
            memory_kind=MemoryKind.EPISODIC,
            content=content,
            summary=summary or content,
            entities=participants,
            time_interval=MemoryTimeInterval(start=timestamp, end=timestamp) if timestamp else None,
            encoding_time=timestamp,
            last_access=_coerce_datetime(data.get("last_access")),
            confidence=float(data.get("confidence", 0.8) or 0.8),
            importance=importance,
            emotional_valence=emotional_valence,
            source=str(data.get("source") or data.get("episodic_source") or "episodic"),
            source_event_ids=list(data.get("source_memory_ids", []) or []),
            relationship_target=target,
            goal_ids=event_goal_ids,
            narrative_role=str(data.get("life_period") or "general"),
            tags=list(data.get("tags", []) or []),
            metadata=metadata,
        )

    def encode_events(self, episodes: list[Any]) -> list[CanonicalMemoryItem]:
        return [self.encode_episode(episode) for episode in episodes]