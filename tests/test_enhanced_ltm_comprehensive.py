from datetime import datetime, timedelta

import pytest

from src.memory.ltm.vector_ltm import VectorLongTermMemory


class FakeCollection:
    def __init__(self):
        self.records = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for memory_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.records[memory_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": dict(metadata),
            }

    def get(self, ids=None, where=None):
        items = list(self.records.items())
        if ids is not None:
            wanted_ids = set(ids)
            items = [(memory_id, record) for memory_id, record in items if memory_id in wanted_ids]
        if where:
            items = [(memory_id, record) for memory_id, record in items if self._matches_where(record["metadata"], where)]
        return {
            "ids": [memory_id for memory_id, _ in items],
            "documents": [record["document"] for _, record in items],
            "metadatas": [dict(record["metadata"]) for _, record in items],
        }

    def update(self, ids, metadatas):
        for memory_id, metadata in zip(ids, metadatas):
            self.records[memory_id]["metadata"] = dict(metadata)

    def delete(self, ids):
        for memory_id in ids:
            self.records.pop(memory_id, None)

    @staticmethod
    def _matches_where(metadata, where):
        for key, value in where.items():
            if isinstance(value, dict) and "$in" in value:
                raw = metadata.get(key, "")
                values = raw.split(",") if isinstance(raw, str) else list(raw)
                if not any(candidate in values for candidate in value["$in"]):
                    return False
                continue
            if metadata.get(key) != value:
                return False
        return True


@pytest.fixture
def ltm(monkeypatch, tmp_path):
    memory = VectorLongTermMemory(
        chroma_persist_dir=str(tmp_path / "chroma_ltm"),
        collection_name="test_ltm",
    )
    memory.collection = FakeCollection()
    memory.chroma_client = object()
    monkeypatch.setattr(memory, "_ensure_embedding_model", lambda: True)
    monkeypatch.setattr(memory, "_generate_embedding", lambda text: [float(len(str(text)) or 1)])
    return memory


def test_store_and_retrieve_round_trip(ltm):
    assert ltm.store(
        memory_id="memory-a",
        content="programming concept",
        importance=0.9,
        emotional_valence=0.2,
        source="unit-test",
        tags=["code", "python"],
    )

    record = ltm.retrieve("memory-a")

    assert record is not None
    assert record["id"] == "memory-a"
    assert record["content"] == "programming concept"
    assert record["importance"] == 0.9
    assert record["emotional_valence"] == 0.2
    assert record["source"] == "unit-test"
    assert record["tags"] == ["code", "python"]


def test_add_feedback_updates_record_and_summary(ltm):
    ltm.store(memory_id="memory-a", content="programming concept", tags=["code"])

    ltm.add_feedback("memory-a", "importance", 5)
    ltm.add_feedback("memory-a", "emotion", 0.7)

    record = ltm.retrieve("memory-a")
    summary = ltm.get_feedback_summary("memory-a")

    assert record is not None
    assert record["importance"] == 1.0
    assert record["emotional_valence"] == 0.7
    assert len(record["feedback"]) == 2
    assert summary["importance"] == 5.0
    assert summary["emotion"] == 0.7
    assert summary["count"] == 2


def test_search_by_tags_supports_or_and(ltm):
    ltm.store(memory_id="memory-a", content="python concept", tags=["code", "python"])
    ltm.store(memory_id="memory-b", content="java concept", tags=["code", "java"])
    ltm.store(memory_id="memory-c", content="archive note", tags=["archive"])

    or_results = ltm.search_by_tags(["python", "archive"], operator="OR")
    and_results = ltm.search_by_tags(["code", "python"], operator="AND")

    assert {result["id"] for result in or_results} == {"memory-a", "memory-c"}
    assert [result["id"] for result in and_results] == ["memory-a"]


def test_semantic_clusters_group_shared_tags(ltm):
    ltm.store(memory_id="memory-a", content="python concept", tags=["code", "python"])
    ltm.store(memory_id="memory-b", content="java concept", tags=["code", "java"])
    ltm.store(memory_id="memory-c", content="archive note", tags=["archive"])

    clusters = ltm.get_semantic_clusters(min_cluster_size=2)

    assert clusters["tag:code"] == ["memory-a", "memory-b"]


def test_decay_and_health_report_reflect_metadata_state(ltm):
    ltm.store(memory_id="stale", content="old memory", importance=0.8, tags=["archive"])
    ltm.store(memory_id="active", content="active memory", importance=0.6, tags=["code"])

    stale_metadata = ltm.collection.records["stale"]["metadata"]
    stale_metadata["last_access"] = (datetime.now() - timedelta(days=45)).isoformat()
    stale_metadata["importance"] = 0.8
    stale_metadata["confidence"] = 0.8

    active_metadata = ltm.collection.records["active"]["metadata"]
    active_metadata["access_count"] = 12
    active_metadata["confidence"] = 0.95

    decayed = ltm.decay_memories(decay_rate=0.2, half_life_days=30.0)
    stale_record = ltm.retrieve("stale")
    report = ltm.get_memory_health_report()

    assert decayed == 1
    assert stale_record is not None
    assert stale_record["importance"] < 0.8
    assert report["memory_categories"]["stale_memories"] == 1
    assert report["memory_categories"]["frequently_accessed"] == 1
    assert "potential_issues" in report