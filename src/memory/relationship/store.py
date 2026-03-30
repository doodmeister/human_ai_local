from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .model import RelationshipMemory


class RelationshipMemoryStore:
    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path or "data/memory_stores/relationships")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, RelationshipMemory] = {}
        self._load_from_disk()

    def list_all(self) -> list[RelationshipMemory]:
        return [self._cache[key] for key in sorted(self._cache)]

    def get(self, interlocutor_id: str) -> RelationshipMemory | None:
        return self._cache.get(interlocutor_id)

    def create_or_get(self, interlocutor_id: str, display_name: str = "") -> RelationshipMemory:
        memory = self.get(interlocutor_id)
        if memory is not None:
            return memory
        memory = RelationshipMemory(interlocutor_id=interlocutor_id, display_name=display_name)
        self.upsert(memory)
        return memory

    def upsert(self, memory: RelationshipMemory) -> RelationshipMemory:
        self._cache[memory.interlocutor_id] = memory
        self._save(memory)
        return memory

    def update(
        self,
        interlocutor_id: str,
        *,
        display_name: str | None = None,
        warmth: float | None = None,
        trust: float | None = None,
        familiarity: float | None = None,
        rupture: float | None = None,
        recurring_norms: list[str] | None = None,
        interaction_delta: int = 0,
        observed_at: datetime | None = None,
        metadata_updates: dict[str, object] | None = None,
    ) -> RelationshipMemory:
        memory = self.create_or_get(interlocutor_id, display_name=display_name or "")
        if display_name is not None:
            memory.display_name = display_name.strip()
        if warmth is not None:
            memory.warmth = max(0.0, min(1.0, float(warmth)))
        if trust is not None:
            memory.trust = max(0.0, min(1.0, float(trust)))
        if familiarity is not None:
            memory.familiarity = max(0.0, min(1.0, float(familiarity)))
        if rupture is not None:
            memory.rupture = max(0.0, min(1.0, float(rupture)))
        if recurring_norms:
            memory.merge_norms(recurring_norms)
        if metadata_updates:
            memory.metadata.update(metadata_updates)
        if interaction_delta > 0:
            memory.record_interaction(at=observed_at, count=interaction_delta)
        elif observed_at is not None:
            if memory.first_interaction is None:
                memory.first_interaction = observed_at
            memory.last_interaction = observed_at
        self._save(memory)
        return memory

    def _load_from_disk(self) -> None:
        for file_path in sorted(self.storage_path.glob("*.json")):
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                memory = RelationshipMemory.from_dict(payload)
            except Exception:
                continue
            self._cache[memory.interlocutor_id] = memory

    def _save(self, memory: RelationshipMemory) -> None:
        with self._path_for(memory.interlocutor_id).open("w", encoding="utf-8") as handle:
            json.dump(memory.to_dict(), handle, indent=2, ensure_ascii=False)

    def _path_for(self, interlocutor_id: str) -> Path:
        safe_name = quote(interlocutor_id, safe="")
        return self.storage_path / f"{safe_name}.json"