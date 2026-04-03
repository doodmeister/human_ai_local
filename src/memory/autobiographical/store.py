from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from .graph import AutobiographicalGraph, AutobiographicalGraphBuilder


class AutobiographicalGraphStore:
    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path or "data/memory_stores/autobiographical")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, AutobiographicalGraph] = {}
        self._load_from_disk()

    def get(self, key: str) -> AutobiographicalGraph | None:
        return self._cache.get(str(key))

    def upsert(self, key: str, graph: AutobiographicalGraph) -> AutobiographicalGraph:
        normalized_key = str(key)
        self._cache[normalized_key] = graph
        self._save(normalized_key, graph)
        return graph

    def merge(self, key: str, graph: AutobiographicalGraph) -> AutobiographicalGraph:
        existing = self.get(key)
        merged = graph if existing is None else existing.merged_with(graph)
        return self.upsert(key, merged)

    def ingest_items(self, key: str, items: list[object]) -> AutobiographicalGraph:
        graph = AutobiographicalGraphBuilder().build(items)
        return self.merge(key, graph)

    def _load_from_disk(self) -> None:
        for file_path in sorted(self.storage_path.glob("*.json")):
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                graph = AutobiographicalGraph.from_dict(dict(payload))
            except Exception:
                continue
            self._cache[file_path.stem] = graph

    def _save(self, key: str, graph: AutobiographicalGraph) -> None:
        path = self._path_for(key)
        path.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def _path_for(self, key: str) -> Path:
        return self.storage_path / f"{quote(str(key), safe='')}.json"