from datetime import datetime, timedelta

from src.memory.episodic import episodic_memory as episodic_module
from src.memory.episodic.episodic_memory import (
    EpisodicContext,
    EpisodicMemory,
    EpisodicMemorySystem,
)


class _FakeCollection:
    def __init__(self):
        self.ids = []
        self.upsert_calls = []
        self.update_calls = []
        self.delete_calls = []

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)
        for memory_id in kwargs.get("ids", []):
            if memory_id not in self.ids:
                self.ids.append(memory_id)

    def update(self, **kwargs):
        self.update_calls.append(kwargs)

    def delete(self, ids):
        ids = list(ids)
        self.delete_calls.append(ids)
        self.ids = [memory_id for memory_id in self.ids if memory_id not in ids]

    def get(self):
        return {"ids": list(self.ids)}


class _BrokenChromaModule:
    def PersistentClient(self, *args, **kwargs):
        raise RuntimeError("persistent init failed")


def _build_system(tmp_path, name, monkeypatch):
    monkeypatch.setattr(episodic_module, "CHROMADB_AVAILABLE", False)
    monkeypatch.setattr(episodic_module, "ADVANCED_SEARCH_AVAILABLE", False)
    system = EpisodicMemorySystem(
        chroma_persist_dir=str(tmp_path / f"{name}_chroma"),
        collection_name=name,
        enable_json_backup=True,
        storage_path=str(tmp_path / f"{name}_json"),
    )
    system.collection = None
    system.chroma_client = None
    system._search_strategy = None
    return system


def test_episodic_context_round_trip():
    context = EpisodicContext(
        location="test_environment",
        emotional_state=0.7,
        cognitive_load=0.5,
        attention_focus=["memory", "testing"],
        interaction_type="test",
        participants=["tester", "ai"],
        environmental_factors={"noise_level": "low"},
    )

    restored = EpisodicContext.from_dict(context.to_dict())

    assert restored.location == "test_environment"
    assert restored.emotional_state == 0.7
    assert restored.attention_focus == ["memory", "testing"]
    assert restored.participants == ["tester", "ai"]


def test_episodic_memory_round_trip_tracks_access_and_rehearsal():
    memory = EpisodicMemory(
        id="episode-1",
        summary="Test summary",
        detailed_content="Detailed memory content",
        timestamp=datetime.now(),
        context=EpisodicContext(location="lab"),
        associated_stm_ids=["stm-1"],
        associated_ltm_ids=["ltm-1"],
        importance=0.8,
        emotional_valence=0.3,
        life_period="development",
    )

    initial_access_count = memory.access_count
    initial_consolidation = memory.consolidation_strength

    memory.update_access()
    memory.rehearse(0.2)
    restored = EpisodicMemory.from_dict(memory.to_dict())

    assert memory.access_count == initial_access_count + 2
    assert memory.consolidation_strength > initial_consolidation
    assert memory.rehearsal_count == 1
    assert restored.context.location == "lab"
    assert restored.associated_stm_ids == ["stm-1"]
    assert restored.life_period == "development"


def test_from_dict_uses_timestamp_fallbacks_for_missing_recency_fields():
    timestamp = datetime(2024, 1, 2, 3, 4, 5)

    restored = EpisodicMemory.from_dict(
        {
            "id": "episode-legacy",
            "summary": "Legacy import",
            "detailed_content": "Imported from older storage format.",
            "timestamp": timestamp.isoformat(),
            "context": {},
        }
    )

    assert restored.timestamp == timestamp
    assert restored.created_at == timestamp
    assert restored.last_access == timestamp
    assert restored.updated_at == timestamp


def test_system_initialization_creates_storage_paths(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_init", monkeypatch)

    assert system.collection_name == "episodic_init"
    assert system.enable_json_backup is True
    assert system.chroma_persist_dir.exists()
    assert system.storage_path.exists()


def test_initialize_chromadb_does_not_fallback_to_in_memory_client(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(episodic_module, "CHROMADB_AVAILABLE", True)
    monkeypatch.setattr(episodic_module, "ADVANCED_SEARCH_AVAILABLE", False)
    monkeypatch.setattr(episodic_module, "chromadb", _BrokenChromaModule())

    with caplog.at_level("WARNING"):
        system = EpisodicMemorySystem(
            chroma_persist_dir=str(tmp_path / "broken_chroma"),
            collection_name="broken_chroma",
            enable_json_backup=False,
            storage_path=str(tmp_path / "broken_json"),
        )

    assert system.chroma_client is None
    assert system.collection is None
    assert any("continue without ChromaDB persistence" in record.message for record in caplog.records)


def test_store_and_retrieve_preserves_context_and_references(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_store", monkeypatch)
    context = EpisodicContext(
        location="test_lab",
        emotional_state=0.6,
        cognitive_load=0.4,
        interaction_type="development_session",
        participants=["developer", "ai_system"],
    )

    memory_id = system.store_memory(
        detailed_content="Implemented and tested the episodic memory system with contextual metadata.",
        context=context,
        associated_stm_ids=["stm-dev-1", "stm-dev-2"],
        associated_ltm_ids=["ltm-memory-systems"],
        importance=0.9,
        emotional_valence=0.7,
        life_period="system_development",
    )
    retrieved = system.retrieve_memory(memory_id)

    assert retrieved is not None
    assert retrieved.importance == 0.9
    assert retrieved.context.location == "test_lab"
    assert retrieved.associated_stm_ids == ["stm-dev-1", "stm-dev-2"]
    assert retrieved.associated_ltm_ids == ["ltm-memory-systems"]
    assert retrieved.life_period == "system_development"


def test_store_memory_uses_chromadb_upsert(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_upsert", monkeypatch)
    fake_collection = _FakeCollection()
    system.collection = fake_collection

    memory_id = system.store_memory(
        detailed_content="Persist this episodic memory into the vector store.",
        importance=0.7,
    )

    assert memory_id in system._memory_cache
    assert len(fake_collection.upsert_calls) == 1
    assert fake_collection.upsert_calls[0]["ids"] == [memory_id]


def test_search_memories_uses_text_fallback_with_life_period_filter(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_search", monkeypatch)
    system.collection = None
    system._search_strategy = None

    target_id = system.store_memory(
        detailed_content="Deep conversation about neural networks and architecture planning.",
        importance=0.8,
        life_period="research_phase",
    )
    system.store_memory(
        detailed_content="Attention mechanism debugging session.",
        importance=0.6,
        life_period="development_phase",
    )

    results = system.search_memories(
        "neural networks",
        life_period="research_phase",
        limit=5,
        min_relevance=0.1,
    )

    assert [result.memory.id for result in results] == [target_id]
    assert results[0].match_type in {"text_match", "word_overlap"}
    assert results[0].memory.life_period == "research_phase"


def test_search_memories_fallback_respects_update_access_flag(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_search_access", monkeypatch)

    memory_id = system.store_memory(
        detailed_content="Discussed retrieval fallback behavior for episodic search.",
        importance=0.8,
    )
    memory = system._memory_cache[memory_id]
    access_before = memory.access_count

    results = system.search_memories(
        "retrieval fallback",
        limit=5,
        min_relevance=0.1,
        update_access=False,
    )

    assert [result.memory.id for result in results] == [memory_id]
    assert memory.access_count == access_before


def test_related_memories_returns_cross_reference_and_temporal_matches(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_related", monkeypatch)
    base_time = datetime.now()

    base_id = system.store_memory(
        detailed_content="Designed the architecture for STM and LTM integration.",
        associated_stm_ids=["stm-design-1", "stm-design-2"],
        associated_ltm_ids=["ltm-architecture"],
        importance=0.8,
        life_period="design_phase",
    )
    cross_ref_id = system.store_memory(
        detailed_content="Implemented memory systems with shared references.",
        associated_stm_ids=["stm-design-2"],
        associated_ltm_ids=["ltm-architecture"],
        importance=0.9,
        life_period="implementation_phase",
    )
    temporal_id = system.store_memory(
        detailed_content="Follow-up retrospective shortly after implementation.",
        importance=0.5,
        life_period="implementation_phase",
    )

    system._memory_cache[base_id].timestamp = base_time
    system._memory_cache[cross_ref_id].timestamp = base_time + timedelta(minutes=30)
    system._memory_cache[temporal_id].timestamp = base_time + timedelta(minutes=45)

    related = system.get_related_memories(
        base_id,
        relationship_types=["cross_reference", "temporal"],
        limit=10,
    )

    related_by_id = {result.memory.id: result for result in related}

    assert cross_ref_id in related_by_id
    assert temporal_id in related_by_id
    assert related_by_id[cross_ref_id].match_type == "cross_reference"
    assert related_by_id[temporal_id].match_type == "temporal"


def test_autobiographical_timeline_sorts_chronologically(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_timeline", monkeypatch)
    base_time = datetime.now() - timedelta(hours=3)

    first_id = system.store_memory(
        detailed_content="Started the cognitive architecture project.",
        importance=0.7,
        life_period="project_development",
    )
    second_id = system.store_memory(
        detailed_content="Designed STM and LTM systems.",
        importance=0.8,
        life_period="project_development",
    )
    third_id = system.store_memory(
        detailed_content="Implemented core memory systems.",
        importance=0.9,
        life_period="project_development",
    )

    system._memory_cache[first_id].timestamp = base_time
    system._memory_cache[second_id].timestamp = base_time + timedelta(minutes=60)
    system._memory_cache[third_id].timestamp = base_time + timedelta(minutes=120)

    timeline = system.get_autobiographical_timeline(life_period="project_development", limit=10)

    assert [memory.id for memory in timeline] == [first_id, second_id, third_id]
    assert all(memory.life_period == "project_development" for memory in timeline)


def test_consolidation_candidates_and_statistics_reflect_memory_state(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_stats", monkeypatch)

    primary_id = system.store_memory(
        detailed_content="Deep learning about cognitive architectures.",
        importance=0.8,
        emotional_valence=0.6,
        life_period="phase_1",
    )
    system.store_memory(
        detailed_content="Routine implementation task.",
        importance=0.3,
        emotional_valence=-0.4,
        life_period="phase_2",
    )
    neutral_id = system.store_memory(
        detailed_content="Neutral project checkpoint.",
        importance=0.6,
        emotional_valence=0.0,
        life_period="phase_3",
    )

    before = system.retrieve_memory(primary_id)
    assert before is not None
    before_consolidation = before.consolidation_strength

    assert system.consolidate_memory(primary_id, strength_increment=0.3) is True

    after = system.retrieve_memory(primary_id)
    candidates = system.get_consolidation_candidates(min_importance=0.5, max_consolidation=0.9, limit=10)
    stats = system.get_memory_statistics()

    assert after is not None
    assert after.consolidation_strength > before_consolidation
    assert {candidate.id for candidate in candidates} == {primary_id, neutral_id}
    assert stats["total_memories"] == 3
    assert stats["life_period_count"] == 3
    assert stats["memory_system_status"] == "active"
    assert stats["emotional_stats"]["positive_memories"] == 1
    assert stats["emotional_stats"]["negative_memories"] == 1
    assert stats["emotional_stats"]["neutral_memories"] == 1


def test_apply_forgetting_policy_batches_only_changed_memories(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_forgetting", monkeypatch)
    fake_collection = _FakeCollection()
    system.collection = fake_collection

    anchor_id = system.store_memory(
        detailed_content="Important relationship milestone with participants present.",
        context=EpisodicContext(participants=["user"]),
        importance=0.9,
    )
    transient_id = system.store_memory(
        detailed_content="Low-signal transient note.",
        importance=0.1,
        emotional_valence=0.0,
    )
    stable_id = system.store_memory(
        detailed_content="Another protected milestone that is already unsuppressed.",
        importance=0.85,
    )

    old_timestamp = datetime.now() - timedelta(days=30)
    system._memory_cache[anchor_id].timestamp = old_timestamp
    system._memory_cache[transient_id].timestamp = old_timestamp
    system._memory_cache[stable_id].timestamp = old_timestamp
    system._memory_cache[anchor_id].suppressed = True
    system._save_to_json_backup(system._memory_cache[anchor_id])

    stats = system.apply_forgetting_policy(min_importance=0.3, min_confidence=0.9, min_age_days=14.0)

    assert stats["suppressed"] == 1
    assert len(fake_collection.update_calls) == 1
    updated_ids = set(fake_collection.update_calls[0]["ids"])
    assert updated_ids == {anchor_id, transient_id}
    assert stable_id not in updated_ids
    assert system._memory_cache[anchor_id].suppressed is False
    assert system._memory_cache[transient_id].suppressed is True


def test_clear_memory_removes_old_and_low_importance_entries(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_clear", monkeypatch)
    fake_collection = _FakeCollection()
    system.collection = fake_collection

    old_id = system.store_memory(
        detailed_content="This is an old memory with low importance.",
        importance=0.2,
    )
    recent_id = system.store_memory(
        detailed_content="This is a recent memory with high importance.",
        importance=0.8,
    )
    low_importance_id = system.store_memory(
        detailed_content="This memory has low importance.",
        importance=0.1,
    )

    system._memory_cache[old_id].timestamp = datetime.now() - timedelta(days=2)

    system.clear_memory(older_than=timedelta(days=1))
    system.clear_memory(importance_threshold=0.2)

    assert old_id not in system._memory_cache
    assert low_importance_id not in system._memory_cache
    assert recent_id in system._memory_cache
    assert not (system.storage_path / f"{old_id}.json").exists()
    assert not (system.storage_path / f"{low_importance_id}.json").exists()
    assert (system.storage_path / f"{recent_id}.json").exists()
    deleted_ids = {memory_id for call in fake_collection.delete_calls for memory_id in call}
    assert old_id in deleted_ids
    assert low_importance_id in deleted_ids


def test_clear_all_memories_removes_persisted_artifacts(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_clear_all", monkeypatch)
    fake_collection = _FakeCollection()
    system.collection = fake_collection

    first_id = system.store_memory(detailed_content="First persisted episodic memory.", importance=0.4)
    second_id = system.store_memory(detailed_content="Second persisted episodic memory.", importance=0.6)

    system.clear_all_memories()

    assert system._memory_cache == {}
    assert fake_collection.delete_calls == [[first_id, second_id]]
    assert list(system.storage_path.glob("*.json")) == []


def test_summarize_content_handles_abbreviations_and_decimals(tmp_path, monkeypatch):
    system = _build_system(tmp_path, "episodic_summary", monkeypatch)

    summary = system._summarize_content(
        "Meeting with Dr. Smith about version 3.2 rollout. Follow-up tomorrow.",
        max_length=128,
    )

    assert summary == "Meeting with Dr. Smith about version 3.2 rollout."