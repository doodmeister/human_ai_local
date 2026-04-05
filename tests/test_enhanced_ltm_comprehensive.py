from datetime import datetime, timedelta

import pytest

from src.memory.memory_system import VectorLongTermMemory


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.update_calls = 0

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
        self.update_calls += 1
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


def test_add_feedback_importance_uses_passed_value(ltm):
    ltm.store(memory_id="memory-a", content="programming concept", tags=["code"])

    ltm.add_feedback("memory-a", "importance", 0.25)

    record = ltm.retrieve("memory-a")

    assert record is not None
    assert record["importance"] == 0.25


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


def test_semantic_clusters_cap_content_keyword_output(ltm):
    shared_keywords = [f"sharedword{i}" for i in range(ltm.MAX_CONTENT_CLUSTERS + 20)]
    doc = " ".join(shared_keywords)
    ltm.store(memory_id="memory-a", content=doc, tags=["code"])
    ltm.store(memory_id="memory-b", content=doc, tags=["archive"])

    clusters = ltm.get_semantic_clusters(min_cluster_size=2)
    content_cluster_keys = [key for key in clusters if key.startswith("content:")]

    assert len(content_cluster_keys) <= ltm.MAX_CONTENT_CLUSTERS


def test_find_cross_system_links_matches_comma_delimited_associations(ltm):
    ltm.store(
        memory_id="memory-a",
        content="python concept",
        tags=["code", "python"],
        associations=["external-1", "external-2"],
    )
    ltm.store(
        memory_id="memory-b",
        content="java concept",
        tags=["code", "java"],
        associations=["external-3"],
    )

    matches = ltm.find_cross_system_links("external-2")

    assert [match["id"] for match in matches] == ["memory-a"]
    assert matches[0]["associations"] == ["external-1", "external-2"]


def test_get_associations_retrieves_each_node_once(ltm, monkeypatch):
    ltm.store(memory_id="root", content="root", associations=["child"])
    ltm.store(memory_id="child", content="child", associations=["grandchild"])
    ltm.store(memory_id="grandchild", content="grandchild")

    original_retrieve = ltm.retrieve
    retrieval_counts = {}

    def counting_retrieve(memory_id):
        retrieval_counts[memory_id] = retrieval_counts.get(memory_id, 0) + 1
        return original_retrieve(memory_id)

    monkeypatch.setattr(ltm, "retrieve", counting_retrieve)

    associated = ltm.get_associations("root", depth=2)

    assert [record["id"] for record in associated] == ["child", "grandchild"]
    assert retrieval_counts == {"root": 1, "child": 1, "grandchild": 1}


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


def test_decay_memories_batches_metadata_updates(ltm):
    ltm.store(memory_id="stale-a", content="old memory a", importance=0.8, tags=["archive"])
    ltm.store(memory_id="stale-b", content="old memory b", importance=0.7, tags=["archive"])

    old_ts = (datetime.now() - timedelta(days=45)).isoformat()
    for memory_id in ["stale-a", "stale-b"]:
        metadata = ltm.collection.records[memory_id]["metadata"]
        metadata["last_access"] = old_ts
        metadata["confidence"] = 0.8

    decayed = ltm.decay_memories(decay_rate=0.2, half_life_days=30.0)

    assert decayed == 2
    assert ltm.collection.update_calls == 1


def test_apply_forgetting_policy_batches_metadata_updates(ltm):
    ltm.store(memory_id="low-a", content="minor note a", importance=0.1)
    ltm.store(memory_id="low-b", content="minor note b", importance=0.15)

    old_ts = (datetime.now() - timedelta(days=45)).isoformat()
    for memory_id in ["low-a", "low-b"]:
        metadata = ltm.collection.records[memory_id]["metadata"]
        metadata["last_access"] = old_ts
        metadata["encoding_time"] = old_ts
        metadata["confidence"] = 0.1
        metadata["access_count"] = 0

    stats = ltm.apply_forgetting_policy(
        min_importance=0.2,
        min_confidence=0.2,
        min_access_count=0,
        min_age_days=30.0,
    )

    assert stats == {"suppressed": 2, "protected": 0}
    assert ltm.collection.update_calls == 1


def test_clear_returns_bool(ltm):
    ltm.store(memory_id="memory-a", content="programming concept", tags=["code"])

    cleared = ltm.clear()

    assert cleared is True
    assert ltm.collection.records == {}