from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import import_module
import logging
from tempfile import TemporaryDirectory
import threading
from typing import Any
import warnings

from src.evals.metrics import LongitudinalMetrics, score_longitudinal
from src.memory.episodic.episodic_memory import EpisodicContext, EpisodicMemory
from src.memory.memory_system import MemorySystem, MemorySystemConfig


_BASE_TIME = datetime(2025, 1, 10, 14, 0, tzinfo=timezone.utc)


class _FakeCollection:
    def __init__(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        self.records = deepcopy(records) if records is not None else {}

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        for fact_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas, strict=True):
            self.records[str(fact_id)] = {
                "embedding": list(embedding),
                "document": document,
                "metadata": dict(metadata),
            }

    def get(self, ids=None):
        if ids is None:
            selected_ids = list(self.records)
        else:
            selected_ids = [fact_id for fact_id in ids if fact_id in self.records]
        return {
            "ids": list(selected_ids),
            "documents": [self.records[fact_id]["document"] for fact_id in selected_ids],
            "metadatas": [self.records[fact_id]["metadata"] for fact_id in selected_ids],
        }

    def delete(self, ids) -> None:
        for fact_id in ids:
            self.records.pop(fact_id, None)

    def query(self, query_embeddings, n_results, include):
        del query_embeddings, include
        selected_ids = list(self.records)[:n_results]
        return {
            "ids": [selected_ids],
            "documents": [[self.records[fact_id]["document"] for fact_id in selected_ids]],
            "metadatas": [[self.records[fact_id]["metadata"] for fact_id in selected_ids]],
        }


def _make_semantic_memory(records: dict[str, dict[str, Any]] | None = None):
    semantic_cls = import_module("src.memory.semantic.semantic_memory").SemanticMemorySystem
    semantic = semantic_cls.__new__(semantic_cls)
    semantic.collection = _FakeCollection(records)
    semantic.embedding_model = object()
    semantic._embedding_lock = threading.Lock()
    semantic._lazy_embeddings = True
    semantic._last_revision_event = None
    semantic._generate_embedding = lambda text: [0.1, 0.2, float(len(text))]
    semantic._ensure_embedding_model = lambda: True
    return semantic


@dataclass(frozen=True, slots=True)
class FactSeed:
    subject: str
    predicate: str
    object: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeSeed:
    episode_id: str
    summary: str
    detailed_content: str
    timestamp: datetime
    life_period: str
    importance: float = 0.7
    confidence: float = 0.85
    participants: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FactExpectation:
    subject: str
    predicate: str
    expected_active_objects: tuple[str, ...]
    expected_quarantined_objects: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LongitudinalPhase:
    phase_id: str
    fact_writes: tuple[FactSeed, ...] = ()
    episode_writes: tuple[EpisodeSeed, ...] = ()
    restart_after: bool = False


@dataclass(frozen=True, slots=True)
class LongitudinalScenario:
    scenario_id: str
    phases: tuple[LongitudinalPhase, ...]
    runner: str = "fixture"
    fact_expectations: tuple[FactExpectation, ...] = ()
    expected_episode_ids: tuple[str, ...] = ()
    min_continuity_score: float = 1.0
    min_contradiction_repair_score: float = 1.0
    max_over_recall_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class LongitudinalScenarioResult:
    scenario_id: str
    runner: str
    metrics: LongitudinalMetrics
    active_fact_objects: dict[str, tuple[str, ...]]
    quarantined_fact_objects: dict[str, tuple[str, ...]]
    available_episode_ids: tuple[str, ...]


@contextmanager
def _suppress_runtime_logs() -> Any:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable)


@contextmanager
def _suppress_runtime_warnings() -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`clean_up_tokenization_spaces` was not set.*",
            category=FutureWarning,
        )
        yield


class _LongitudinalHarness:
    def __init__(
        self,
        *,
        semantic_records: dict[str, dict[str, Any]] | None = None,
        episodic_records: tuple[dict[str, Any], ...] | None = None,
    ) -> None:
        self.semantic = _make_semantic_memory(semantic_records)
        self.episodes = {
            episode["id"]: EpisodicMemory.from_dict(episode)
            for episode in (episodic_records or ())
        }

    def apply_phase(self, phase: LongitudinalPhase) -> None:
        for fact in phase.fact_writes:
            self.semantic.store_fact(
                fact.subject,
                fact.predicate,
                fact.object,
                metadata=dict(fact.metadata),
            )

        for episode in phase.episode_writes:
            self.episodes[episode.episode_id] = EpisodicMemory(
                id=episode.episode_id,
                summary=episode.summary,
                detailed_content=episode.detailed_content,
                timestamp=episode.timestamp,
                duration=timedelta(minutes=5),
                context=EpisodicContext(participants=list(episode.participants)),
                importance=episode.importance,
                confidence=episode.confidence,
                life_period=episode.life_period,
                tags=list(episode.tags),
            )

    def restart(self) -> "_LongitudinalHarness":
        episodic_records = tuple(memory.to_dict() for memory in self.episodes.values())
        return _LongitudinalHarness(
            semantic_records=deepcopy(self.semantic.collection.records),
            episodic_records=episodic_records,
        )

    def facts_for_axis(self, subject: str, predicate: str, *, belief_status: str) -> tuple[dict[str, Any], ...]:
        matched: list[dict[str, Any]] = []
        for fact_id in self.semantic.collection.records:
            fact = self.semantic.retrieve_fact(fact_id)
            if fact is None:
                continue
            if fact.get("subject") != subject or fact.get("predicate") != predicate:
                continue
            if str(fact.get("belief_status", "active")).lower() != belief_status:
                continue
            matched.append(fact)
        matched.sort(key=lambda item: (str(item.get("object", "")), str(item.get("id", ""))))
        return tuple(matched)

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.episodes))


def _patch_runtime_memory(memory: MemorySystem) -> None:
    memory.semantic._ensure_embedding_model = lambda: True
    memory.semantic._generate_embedding = lambda text: [0.1, 0.2, float(len(text))]
    memory.ltm._ensure_embedding_model = lambda: True
    memory.ltm._generate_embedding = lambda text: [0.2, 0.3, float(len(text))]


def _build_runtime_memory(persist_dir: str) -> MemorySystem:
    with _suppress_runtime_logs(), _suppress_runtime_warnings():
        memory = MemorySystem(MemorySystemConfig(chroma_persist_dir=persist_dir))
    _patch_runtime_memory(memory)
    return memory


class _RuntimeLongitudinalHarness:
    def __init__(
        self,
        *,
        temp_dir_handle: TemporaryDirectory[str] | None = None,
        persist_dir: str | None = None,
        episode_aliases: dict[str, str] | None = None,
    ) -> None:
        self._temp_dir_handle = temp_dir_handle or TemporaryDirectory(
            prefix="longitudinal_runtime_",
            ignore_cleanup_errors=True,
        )
        self._persist_dir = persist_dir or self._temp_dir_handle.name
        self._episode_aliases = dict(episode_aliases or {})
        self.memory = _build_runtime_memory(self._persist_dir)

    def apply_phase(self, phase: LongitudinalPhase) -> None:
        with _suppress_runtime_logs():
            for fact in phase.fact_writes:
                self.memory.store_fact(
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    metadata=dict(fact.metadata),
                )

            for episode in phase.episode_writes:
                episode_id = self.memory.create_episodic_memory(
                    summary=episode.summary,
                    detailed_content=episode.detailed_content,
                    participants=list(episode.participants),
                    importance=episode.importance,
                    emotional_valence=0.0,
                    life_period=episode.life_period,
                )
                if episode_id:
                    self._episode_aliases[episode.episode_id] = episode_id

    def restart(self) -> "_RuntimeLongitudinalHarness":
        with _suppress_runtime_logs():
            self.memory.shutdown()
        return _RuntimeLongitudinalHarness(
            temp_dir_handle=self._temp_dir_handle,
            persist_dir=self._persist_dir,
            episode_aliases=self._episode_aliases,
        )

    def facts_for_axis(self, subject: str, predicate: str, *, belief_status: str) -> tuple[dict[str, Any], ...]:
        matched: list[dict[str, Any]] = []
        with _suppress_runtime_logs():
            collection = getattr(self.memory.semantic, "collection", None)
            if collection is None:
                return ()
            all_ids = tuple(collection.get().get("ids") or [])
            for fact_id in all_ids:
                fact = self.memory.semantic.retrieve_fact(str(fact_id))
                if fact is None:
                    continue
                if fact.get("subject") != subject or fact.get("predicate") != predicate:
                    continue
                if str(fact.get("belief_status", "active")).lower() != belief_status:
                    continue
                matched.append(fact)
        matched.sort(key=lambda item: (str(item.get("object", "")), str(item.get("fact_id", ""))))
        return tuple(matched)

    def episode_ids(self) -> tuple[str, ...]:
        available: list[str] = []
        with _suppress_runtime_logs():
            for alias, actual_id in self._episode_aliases.items():
                if self.memory.episodic.retrieve_memory(actual_id) is not None:
                    available.append(alias)
        return tuple(sorted(available))

    def close(self) -> None:
        with _suppress_runtime_logs():
            episodic = getattr(self.memory, "_episodic", None)
            if episodic is not None and hasattr(episodic, "shutdown"):
                episodic.shutdown()
            ltm = getattr(self.memory, "_ltm", None)
            chroma_client = getattr(ltm, "chroma_client", None)
            if chroma_client is not None and hasattr(chroma_client, "clear_system_cache"):
                chroma_client.clear_system_cache()
            self.memory.shutdown()
        try:
            self._temp_dir_handle.cleanup()
        except Exception:
            pass


def build_longitudinal_scenarios() -> list[LongitudinalScenario]:
    return [
        LongitudinalScenario(
            scenario_id="restart_continuity_project_history",
            phases=(
                LongitudinalPhase(
                    phase_id="session_one",
                    fact_writes=(
                        FactSeed(
                            subject="user",
                            predicate="focuses_on",
                            object="adaptive memory roadmap",
                            metadata={"source": "user_assertion", "confidence": 0.86},
                        ),
                    ),
                    episode_writes=(
                        EpisodeSeed(
                            episode_id="episode-roadmap-kickoff",
                            summary="Roadmap kickoff",
                            detailed_content="The user outlined the adaptive memory roadmap and asked for autonomous progress.",
                            timestamp=_BASE_TIME,
                            life_period="phase_4_memory",
                            participants=("user", "assistant"),
                            tags=("roadmap", "memory"),
                        ),
                    ),
                    restart_after=True,
                ),
                LongitudinalPhase(
                    phase_id="session_two",
                    episode_writes=(
                        EpisodeSeed(
                            episode_id="episode-roadmap-review",
                            summary="Roadmap review",
                            detailed_content="The follow-up session reviewed reconsolidation and forgetting milestones after restart.",
                            timestamp=_BASE_TIME + timedelta(days=1),
                            life_period="phase_4_memory",
                            participants=("user", "assistant"),
                            tags=("review", "continuity"),
                        ),
                    ),
                ),
            ),
            fact_expectations=(
                FactExpectation(
                    subject="user",
                    predicate="focuses_on",
                    expected_active_objects=("adaptive memory roadmap",),
                ),
            ),
            expected_episode_ids=("episode-roadmap-kickoff", "episode-roadmap-review"),
        ),
        LongitudinalScenario(
            scenario_id="runtime_restart_continuity_project_history",
            runner="runtime",
            phases=(
                LongitudinalPhase(
                    phase_id="session_one",
                    fact_writes=(
                        FactSeed(
                            subject="user",
                            predicate="focuses_on",
                            object="adaptive memory roadmap",
                            metadata={"source": "user_assertion", "confidence": 0.86},
                        ),
                    ),
                    episode_writes=(
                        EpisodeSeed(
                            episode_id="episode-roadmap-kickoff",
                            summary="Roadmap kickoff",
                            detailed_content="The user outlined the adaptive memory roadmap and asked for autonomous progress.",
                            timestamp=_BASE_TIME,
                            life_period="phase_4_memory",
                            participants=("user", "assistant"),
                            tags=("roadmap", "memory"),
                        ),
                    ),
                    restart_after=True,
                ),
                LongitudinalPhase(
                    phase_id="session_two",
                    episode_writes=(
                        EpisodeSeed(
                            episode_id="episode-roadmap-review",
                            summary="Roadmap review",
                            detailed_content="The follow-up session reviewed reconsolidation and forgetting milestones after restart.",
                            timestamp=_BASE_TIME + timedelta(days=1),
                            life_period="phase_4_memory",
                            participants=("user", "assistant"),
                            tags=("review", "continuity"),
                        ),
                    ),
                ),
            ),
            fact_expectations=(
                FactExpectation(
                    subject="user",
                    predicate="focuses_on",
                    expected_active_objects=("adaptive memory roadmap",),
                ),
            ),
            expected_episode_ids=("episode-roadmap-kickoff", "episode-roadmap-review"),
        ),
        LongitudinalScenario(
            scenario_id="restart_contradiction_repair_drink_preference",
            phases=(
                LongitudinalPhase(
                    phase_id="session_one",
                    fact_writes=(
                        FactSeed(
                            subject="user",
                            predicate="prefers",
                            object="coffee",
                            metadata={"source": "imported", "confidence": 0.72, "importance": 0.55},
                        ),
                    ),
                    restart_after=True,
                ),
                LongitudinalPhase(
                    phase_id="session_two",
                    fact_writes=(
                        FactSeed(
                            subject="user",
                            predicate="prefers",
                            object="tea",
                            metadata={
                                "source": "explicit_user_correction",
                                "confidence": 0.93,
                                "importance": 0.82,
                            },
                        ),
                    ),
                ),
            ),
            fact_expectations=(
                FactExpectation(
                    subject="user",
                    predicate="prefers",
                    expected_active_objects=("tea",),
                    expected_quarantined_objects=("coffee",),
                ),
            ),
        ),
        LongitudinalScenario(
            scenario_id="runtime_restart_contradiction_repair_drink_preference",
            runner="runtime",
            phases=(
                LongitudinalPhase(
                    phase_id="session_one",
                    fact_writes=(
                        FactSeed(
                            subject="user",
                            predicate="prefers",
                            object="coffee",
                            metadata={"source": "imported", "confidence": 0.72, "importance": 0.55},
                        ),
                    ),
                    restart_after=True,
                ),
                LongitudinalPhase(
                    phase_id="session_two",
                    fact_writes=(
                        FactSeed(
                            subject="user",
                            predicate="prefers",
                            object="tea",
                            metadata={
                                "source": "explicit_user_correction",
                                "confidence": 0.93,
                                "importance": 0.82,
                            },
                        ),
                    ),
                ),
            ),
            fact_expectations=(
                FactExpectation(
                    subject="user",
                    predicate="prefers",
                    expected_active_objects=("tea",),
                    expected_quarantined_objects=("coffee",),
                ),
            ),
        ),
    ]


def run_longitudinal_scenario(scenario: LongitudinalScenario) -> LongitudinalScenarioResult:
    harness: Any = _RuntimeLongitudinalHarness() if scenario.runner == "runtime" else _LongitudinalHarness()
    restart_count = 0
    try:
        for phase in scenario.phases:
            harness.apply_phase(phase)
            if phase.restart_after:
                harness = harness.restart()
                restart_count += 1

        matched_continuity_checks = 0
        total_continuity_checks = 0
        repaired_contradictions = 0
        total_contradiction_checks = 0
        false_memory_count = 0
        active_fact_count = 0
        active_fact_objects: dict[str, tuple[str, ...]] = {}
        quarantined_fact_objects: dict[str, tuple[str, ...]] = {}

        for expectation in scenario.fact_expectations:
            axis_key = f"{expectation.subject}:{expectation.predicate}"
            active = harness.facts_for_axis(expectation.subject, expectation.predicate, belief_status="active")
            quarantined = harness.facts_for_axis(expectation.subject, expectation.predicate, belief_status="quarantined")
            active_objects = tuple(str(fact.get("object", "")) for fact in active)
            quarantined_objects = tuple(str(fact.get("object", "")) for fact in quarantined)
            active_fact_objects[axis_key] = active_objects
            quarantined_fact_objects[axis_key] = quarantined_objects

            total_contradiction_checks += 1
            if all(obj in active_objects for obj in expectation.expected_active_objects) and all(
                obj in quarantined_objects for obj in expectation.expected_quarantined_objects
            ):
                repaired_contradictions += 1

            for expected_object in expectation.expected_active_objects:
                total_continuity_checks += 1
                if expected_object in active_objects:
                    matched_continuity_checks += 1
            for expected_object in expectation.expected_quarantined_objects:
                total_continuity_checks += 1
                if expected_object in quarantined_objects:
                    matched_continuity_checks += 1

            active_fact_count += len(active_objects)
            false_memory_count += sum(1 for obj in active_objects if obj not in expectation.expected_active_objects)

        available_episode_ids = harness.episode_ids()
        for episode_id in scenario.expected_episode_ids:
            total_continuity_checks += 1
            if episode_id in available_episode_ids:
                matched_continuity_checks += 1

        metrics = score_longitudinal(
            matched_continuity_checks=matched_continuity_checks,
            total_continuity_checks=total_continuity_checks,
            repaired_contradictions=repaired_contradictions,
            total_contradiction_checks=total_contradiction_checks,
            false_memory_count=false_memory_count,
            active_fact_count=active_fact_count,
            restart_count=restart_count,
        )
        return LongitudinalScenarioResult(
            scenario_id=scenario.scenario_id,
            runner=scenario.runner,
            metrics=metrics,
            active_fact_objects=active_fact_objects,
            quarantined_fact_objects=quarantined_fact_objects,
            available_episode_ids=available_episode_ids,
        )
    finally:
        if hasattr(harness, "close"):
            harness.close()


def run_longitudinal_suite(
    scenarios: tuple[LongitudinalScenario, ...] | list[LongitudinalScenario] | None = None,
) -> list[LongitudinalScenarioResult]:
    scenario_list = list(scenarios) if scenarios is not None else build_longitudinal_scenarios()
    return [run_longitudinal_scenario(scenario) for scenario in scenario_list]