from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Mapping

from .canonical import CanonicalMemoryItem, MemoryKind, MemoryTimeInterval


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


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
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


def _content_from_mapping(data: Mapping[str, Any]) -> str:
    return str(
        data.get("content")
        or data.get("detailed_content")
        or data.get("text")
        or data.get("summary")
        or ""
    )


def _derive_activation(*, importance: float | None, confidence: float | None) -> float:
    importance_v = float(importance) if importance is not None else 0.5
    confidence_v = float(confidence) if confidence is not None else 0.5
    return max(0.0, min(1.0, importance_v * 0.6 + confidence_v * 0.4))


def _derive_recency(last_access: datetime | None, encoding_time: datetime | None) -> float:
    ts = last_access or encoding_time
    if ts is None:
        return 0.0
    age_hours = (datetime.now() - ts).total_seconds() / 3600.0
    if age_hours <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (age_hours / 24.0))


def _derive_salience(item: CanonicalMemoryItem) -> float:
    values: list[float] = [abs(float(item.emotional_valence))]
    if item.metadata.get("vividness") is not None:
        values.append(abs(float(item.metadata["vividness"])))
    if item.metadata.get("attention_score") is not None:
        values.append(abs(float(item.metadata["attention_score"])))
    if item.metadata.get("salience") is not None:
        values.append(abs(float(item.metadata["salience"])))
    return max(0.0, min(1.0, max(values) if values else 0.0))


def _derive_stm_activation(item: Any, fallback_importance: float = 0.5) -> tuple[float, float, float]:
    now = datetime.now()
    encoding_time = getattr(item, "encoding_time", None)
    encoding_dt = encoding_time if isinstance(encoding_time, datetime) else now
    age_hours = (now - encoding_dt).total_seconds() / 3600.0
    decay_rate = float(getattr(item, "decay_rate", 0.1) or 0.1)
    recency = max(0.0, 1.0 - age_hours * decay_rate)
    access_count = int(getattr(item, "access_count", 0) or 0)
    frequency = min(1.0, access_count / 10.0)
    importance = float(getattr(item, "importance", fallback_importance) or fallback_importance)
    attention_score = float(getattr(item, "attention_score", 0.0) or 0.0)
    salience = (importance + attention_score) / 2.0
    activation = (recency * 0.4) + (frequency * 0.3) + (salience * 0.3)
    return max(0.0, min(1.0, activation)), recency, max(0.0, min(1.0, salience))


def normalize_stm_item(item: Any, *, similarity: float = 0.0) -> CanonicalMemoryItem:
    activation, recency, salience = _derive_stm_activation(item)
    metadata = _to_mapping(item)
    metadata.update(
        {
            "similarity": float(similarity),
            "activation": activation,
            "recency": recency,
            "salience": salience,
            "source_system": "stm",
        }
    )
    return CanonicalMemoryItem(
        memory_id=str(getattr(item, "id", metadata.get("id", "")) or "stm-unknown"),
        memory_kind=MemoryKind.TRACE,
        content=str(getattr(item, "content", metadata.get("content", "")) or ""),
        encoding_time=getattr(item, "encoding_time", None),
        last_access=getattr(item, "last_access", None),
        importance=float(getattr(item, "importance", 0.5) or 0.5),
        emotional_valence=float(getattr(item, "emotional_valence", 0.0) or 0.0),
        source="stm",
        tags=list(metadata.get("tags", [])),
        metadata=metadata,
    )


def normalize_stm_mapping(record: Mapping[str, Any]) -> CanonicalMemoryItem:
    data = dict(record)
    encoding_time = _coerce_datetime(data.get("encoding_time") or data.get("timestamp"))
    last_access = _coerce_datetime(data.get("last_access"))
    similarity = float(data.get("similarity") or data.get("similarity_score") or 0.0)
    activation = float(data.get("activation", 0.0) or 0.0)
    recency = float(data.get("recency", _derive_recency(last_access, encoding_time)) or 0.0)
    salience = float(data.get("salience", data.get("attention_score", 0.0)) or 0.0)
    metadata = dict(data)
    metadata.update(
        {
            "similarity": similarity,
            "activation": activation,
            "recency": recency,
            "salience": salience,
            "source_system": "stm",
        }
    )
    return CanonicalMemoryItem(
        memory_id=str(data.get("id") or data.get("memory_id") or "stm-unknown"),
        memory_kind=MemoryKind.TRACE,
        content=_content_from_mapping(data),
        summary=data.get("summary"),
        entities=list(data.get("participants", [])),
        encoding_time=encoding_time,
        last_access=last_access,
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        importance=float(data.get("importance", 0.5) or 0.5),
        emotional_valence=float(data.get("emotional_valence", 0.0) or 0.0),
        source=str(data.get("source") or "stm"),
        relationship_target=data.get("relationship_target"),
        goal_ids=list(data.get("goal_ids", [])),
        narrative_role=data.get("narrative_role") or data.get("life_period"),
        tags=list(data.get("tags", [])),
        metadata=metadata,
    )


def normalize_ltm_record(record: Mapping[str, Any]) -> CanonicalMemoryItem:
    data = dict(record)
    content = _content_from_mapping(data)
    similarity = float(data.get("similarity") or data.get("similarity_score") or 0.0)
    encoding_time = _coerce_datetime(data.get("encoding_time"))
    last_access = _coerce_datetime(data.get("last_access"))
    metadata = dict(data)
    metadata.update(
        {
            "similarity": similarity,
            "activation": _derive_activation(
                importance=data.get("importance"),
                confidence=data.get("confidence"),
            ),
            "recency": _derive_recency(last_access, encoding_time),
            "source_system": "ltm",
        }
    )
    kind_name = str(data.get("memory_type") or MemoryKind.SEMANTIC.value).lower()
    kind = MemoryKind.SEMANTIC if kind_name not in {kind.value for kind in MemoryKind} else MemoryKind(kind_name)
    return CanonicalMemoryItem(
        memory_id=str(data.get("id") or data.get("memory_id") or "ltm-unknown"),
        memory_kind=kind,
        content=content,
        summary=data.get("summary"),
        entities=list(data.get("participants", [])),
        subject=data.get("subject"),
        predicate=data.get("predicate"),
        object=data.get("object"),
        encoding_time=encoding_time,
        last_access=last_access,
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        importance=float(data.get("importance", 0.5) or 0.5),
        emotional_valence=float(data.get("emotional_valence", 0.0) or 0.0),
        source=str(data.get("source") or "ltm"),
        relationship_target=data.get("relationship_target"),
        goal_ids=list(data.get("goal_ids", [])),
        narrative_role=data.get("narrative_role") or data.get("life_period"),
        tags=list(data.get("tags", [])),
        metadata=metadata,
    )


def normalize_episodic_value(memory: Any, *, relevance: float | None = None) -> CanonicalMemoryItem:
    data = _to_mapping(memory)
    timestamp = _coerce_datetime(data.get("timestamp") or getattr(memory, "timestamp", None))
    last_access = _coerce_datetime(data.get("last_access") or getattr(memory, "last_access", None))
    time_interval = MemoryTimeInterval(start=timestamp, end=timestamp) if timestamp else None
    metadata = dict(data)
    if relevance is not None:
        metadata["similarity"] = float(relevance)
    metadata.setdefault("source_system", "episodic")
    metadata.setdefault(
        "activation",
        _derive_activation(importance=data.get("importance"), confidence=data.get("confidence")),
    )
    metadata.setdefault("recency", _derive_recency(last_access, timestamp))
    metadata.setdefault("vividness", data.get("vividness"))
    return CanonicalMemoryItem(
        memory_id=str(data.get("id") or "episodic-unknown"),
        memory_kind=MemoryKind.EPISODIC,
        content=_content_from_mapping(data),
        summary=data.get("summary"),
        entities=list(data.get("participants", [])),
        time_interval=time_interval,
        encoding_time=timestamp,
        last_access=last_access,
        confidence=float(data.get("confidence", 0.8) or 0.8),
        importance=float(data.get("importance", 0.5) or 0.5),
        emotional_valence=float(data.get("emotional_valence", 0.0) or 0.0),
        source=str(data.get("source") or data.get("episodic_source") or "episodic"),
        source_event_ids=list(data.get("source_memory_ids", [])),
        relationship_target=data.get("relationship_target") or (data.get("participants") or [None])[0],
        goal_ids=list(data.get("goal_ids", [])),
        narrative_role=data.get("life_period"),
        tags=list(data.get("tags", [])),
        metadata=metadata,
    )


def normalize_prospective_value(reminder: Any) -> CanonicalMemoryItem:
    data = _to_mapping(reminder)
    due_time = _coerce_datetime(data.get("due_time"))
    created_at = _coerce_datetime(data.get("created_at"))
    time_interval = MemoryTimeInterval(start=created_at, end=due_time) if created_at or due_time else None
    metadata = dict(data)
    metadata.setdefault("source_system", "prospective")
    metadata.setdefault("activation", float(0.0 if data.get("completed") else 0.7))
    metadata.setdefault("recency", _derive_recency(created_at, created_at))
    return CanonicalMemoryItem(
        memory_id=str(data.get("id") or "prospective-unknown"),
        memory_kind=MemoryKind.PROSPECTIVE,
        content=str(data.get("content") or ""),
        summary=str(data.get("content") or ""),
        time_interval=time_interval,
        encoding_time=created_at,
        last_access=created_at,
        importance=float(data.get("importance", 0.7) or 0.7),
        emotional_valence=float(data.get("emotional_valence", 0.0) or 0.0),
        source="prospective",
        tags=list(data.get("tags", [])),
        metadata=metadata,
    )


def normalize_semantic_value(value: Mapping[str, Any]) -> CanonicalMemoryItem:
    data = dict(value)
    similarity = float(data.get("similarity") or data.get("similarity_score") or data.get("relevance") or 0.0)
    encoding_time = _coerce_datetime(data.get("encoding_time"))
    last_access = _coerce_datetime(data.get("last_access"))
    content = str(
        data.get("fact_text")
        or f"{data.get('subject', '')} {data.get('predicate', '')} {data.get('object', '')}".strip()
    )
    metadata = dict(data)
    metadata.setdefault("source_system", "semantic")
    metadata.setdefault("similarity", similarity)
    metadata.setdefault(
        "activation",
        _derive_activation(importance=data.get("importance"), confidence=data.get("confidence")),
    )
    metadata.setdefault("recency", _derive_recency(last_access, encoding_time))
    return CanonicalMemoryItem(
        memory_id=str(data.get("id") or data.get("memory_id") or "semantic-unknown"),
        memory_kind=MemoryKind.SEMANTIC,
        content=content,
        summary=data.get("fact_text"),
        entities=list(data.get("participants", [])),
        subject=data.get("subject"),
        predicate=data.get("predicate"),
        object=data.get("object"),
        encoding_time=encoding_time,
        last_access=last_access,
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        importance=float(data.get("importance", 0.5) or 0.5),
        emotional_valence=float(data.get("emotional_valence", 0.0) or 0.0),
        source=str(data.get("source") or "semantic"),
        contradiction_set_id=data.get("contradiction_set_id"),
        relationship_target=data.get("relationship_target"),
        goal_ids=list(data.get("goal_ids", [])),
        narrative_role=data.get("narrative_role") or data.get("life_period"),
        tags=list(data.get("tags", [])),
        metadata=metadata,
    )


def normalize_memory_results(raw: Any) -> list[CanonicalMemoryItem]:
    if not raw:
        return []

    normalized: list[CanonicalMemoryItem] = []
    if isinstance(raw, list) and raw and hasattr(raw[0], "item") and hasattr(raw[0], "similarity_score"):
        for result in raw:
            item = getattr(result, "item", None)
            if item is None:
                continue
            normalized.append(
                normalize_stm_item(item, similarity=float(getattr(result, "similarity_score", 0.0) or 0.0))
            )
        return normalized

    if isinstance(raw, list) and raw and isinstance(raw[0], tuple):
        for result in raw:
            if not result:
                continue
            item = result[0]
            similarity = float(result[1]) if len(result) > 1 else 0.0
            normalized.append(normalize_stm_item(item, similarity=similarity))
        return normalized

    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], Mapping)):
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            data = dict(entry)
            if data.get("activation") is not None and not any(
                key in data for key in ("subject", "predicate", "object", "fact_text", "due_time", "completed")
            ):
                normalized.append(normalize_stm_mapping(data))
            elif data.get("due_time") is not None or data.get("completed") is not None:
                normalized.append(normalize_prospective_value(data))
            elif data.get("detailed_content") is not None or data.get("life_period") is not None:
                normalized.append(normalize_episodic_value(data, relevance=float(data.get("relevance", 0.0) or 0.0)))
            elif any(key in data for key in ("subject", "predicate", "object", "fact_text")):
                normalized.append(normalize_semantic_value(data))
            else:
                normalized.append(normalize_ltm_record(data))
        return normalized

    return normalized


def canonical_item_to_context_payload(item: CanonicalMemoryItem) -> dict[str, Any]:
    similarity = float(item.metadata.get("similarity", 0.0) or 0.0)
    activation = float(
        item.metadata.get(
            "activation",
            _derive_activation(importance=item.importance, confidence=item.confidence),
        )
        or 0.0
    )
    recency = float(item.metadata.get("recency", _derive_recency(item.last_access, item.encoding_time)) or 0.0)
    salience = float(item.metadata.get("salience", _derive_salience(item)) or 0.0)
    return {
        "id": item.memory_id,
        "content": item.content,
        "summary": item.summary,
        "activation": activation,
        "strength": activation,
        "importance": item.importance,
        "similarity": similarity,
        "recency": recency,
        "salience": salience,
        "timestamp": _iso_or_none(item.encoding_time),
        "memory_kind": item.memory_kind.value,
        "canonical": item.to_dict(),
    }