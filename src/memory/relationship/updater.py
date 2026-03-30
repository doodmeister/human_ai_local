from __future__ import annotations

from datetime import datetime
from typing import Any

from .model import RelationshipMemory
from .store import RelationshipMemoryStore


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _blend(current: float, projected: float, weight: float) -> float:
    clamped_weight = _clamp(weight)
    return _clamp((current * (1.0 - clamped_weight)) + (projected * clamped_weight))


class RelationshipMemoryUpdater:
    """Persist relationship memory from turn-level and runtime relational signals."""

    def update_from_turn(
        self,
        *,
        store: RelationshipMemoryStore,
        interlocutor_id: str,
        message: str,
        valence: float,
        salience: float,
        relational_model: Any | None = None,
        display_name: str = "",
        observed_at: datetime | None = None,
    ) -> RelationshipMemory:
        base = store.create_or_get(interlocutor_id, display_name=display_name)
        memory = RelationshipMemory.from_dict(base.to_dict())

        if relational_model is not None:
            projected = RelationshipMemory.from_relational_model(relational_model)
            memory = self._merge_projected(memory, projected)
        else:
            memory.record_interaction(at=observed_at)

        if display_name and not memory.display_name:
            memory.display_name = display_name.strip()

        salience_weight = 0.5 + (_clamp(salience) * 0.5)
        positive_valence = max(0.0, float(valence))
        negative_valence = max(0.0, -float(valence))

        memory.warmth = _clamp(memory.warmth + (positive_valence * 0.08 * salience_weight) - (negative_valence * 0.1 * salience_weight))
        memory.trust = _clamp(memory.trust + (positive_valence * 0.05 * salience_weight) - (negative_valence * 0.08 * salience_weight))
        memory.familiarity = _clamp(memory.familiarity + (0.03 * salience_weight))
        memory.rupture = _clamp(memory.rupture + (negative_valence * 0.12 * salience_weight) - (positive_valence * 0.04 * salience_weight))
        memory.merge_norms(self._extract_norms(message))
        memory.metadata.update(
            {
                "last_turn_valence": float(valence),
                "last_turn_salience": float(salience),
            }
        )
        if observed_at is not None:
            if memory.first_interaction is None:
                memory.first_interaction = observed_at
            memory.last_interaction = observed_at

        return store.upsert(memory)

    def _merge_projected(self, existing: RelationshipMemory, projected: RelationshipMemory) -> RelationshipMemory:
        merged = RelationshipMemory.from_dict(existing.to_dict())
        if projected.display_name:
            merged.display_name = projected.display_name
        merged.warmth = _blend(merged.warmth, projected.warmth, 0.7)
        merged.trust = _blend(merged.trust, projected.trust, 0.7)
        merged.familiarity = max(merged.familiarity, projected.familiarity)
        merged.rupture = _blend(merged.rupture, projected.rupture, 0.7)
        merged.interaction_count = max(merged.interaction_count, projected.interaction_count)
        if merged.first_interaction is None or (
            projected.first_interaction is not None and projected.first_interaction < merged.first_interaction
        ):
            merged.first_interaction = projected.first_interaction
        if merged.last_interaction is None or (
            projected.last_interaction is not None and projected.last_interaction > merged.last_interaction
        ):
            merged.last_interaction = projected.last_interaction
        merged.merge_norms(projected.recurring_norms)
        merged.metadata.update(projected.metadata)
        return merged

    def _extract_norms(self, message: str) -> list[str]:
        lower = message.lower()
        norms: list[str] = []
        if any(token in lower for token in ("direct", "concise", "brief")):
            norms.append("prefers direct answers")
        if any(token in lower for token in ("step by step", "checklist", "bullet")):
            norms.append("likes structured guidance")
        if any(token in lower for token in ("why", "tradeoff", "reasoning")):
            norms.append("asks for rationale and tradeoffs")
        return norms