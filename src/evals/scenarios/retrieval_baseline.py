from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping

from src.evals.metrics import RetrievalMetrics, score_retrieval
from src.memory.memory_system import MemorySystem, MemorySystemConfig
from src.memory.relationship import RelationshipMemory, RelationshipMemoryStore
from src.orchestration.chat.context_builder import ContextBuilder
from src.orchestration.chat.conversation_session import SessionManager
from src.orchestration.chat.models import TurnRecord


_MEMORY_SOURCE_SYSTEMS = {"stm", "ltm", "episodic"}
_RUNTIME_EMBEDDING_TERMS = (
    "editor",
    "favorite",
    "vscode",
    "schema",
    "canonical",
    "discussion",
    "bash",
    "windows",
    "coffee",
    "tea",
    "roadmap",
    "relationship",
    "direct",
    "answers",
    "work",
    "together",
)


class _StaticSearchMemory:
    def __init__(self, results: Iterable[Any]) -> None:
        self._results = list(results)

    def search(self, query: str, max_results: int | None = None, limit: int | None = None) -> list[Any]:
        del query
        cap = limit if limit is not None else max_results
        if cap is None:
            return list(self._results)
        return list(self._results[:cap])


class _SemanticAsLtmSearch:
    def __init__(self, semantic_memory: Any) -> None:
        self._semantic_memory = semantic_memory

    def search(self, query: str, max_results: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        cap = limit if limit is not None else max_results
        results = list(self._semantic_memory.search(query=query))
        adapted: list[dict[str, Any]] = []
        for result in results[:cap] if cap is not None else results:
            payload = dict(result)
            payload.setdefault("id", payload.get("fact_id"))
            payload.setdefault("similarity_score", 0.95)
            for key in ("encoding_time", "last_access"):
                value = payload.get(key)
                if isinstance(value, str):
                    try:
                        parsed = datetime.fromisoformat(value)
                    except ValueError:
                        continue
                    if parsed.tzinfo is not None:
                        payload[key] = parsed.replace(tzinfo=None).isoformat()
            adapted.append(payload)
        return adapted


def _runtime_embedding(text: str) -> list[float]:
    lowered = text.lower()
    vector = [float(lowered.count(term)) for term in _RUNTIME_EMBEDDING_TERMS]
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude <= 0.0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


@contextmanager
def _suppress_runtime_logs() -> Any:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def _patch_runtime_memory(memory: MemorySystem) -> None:
    memory.semantic._ensure_embedding_model = lambda: True
    memory.semantic._generate_embedding = _runtime_embedding
    memory.ltm._ensure_embedding_model = lambda: True
    memory.ltm._generate_embedding = _runtime_embedding
    memory.episodic._ensure_embedding_model = lambda: True
    memory.episodic._generate_embedding = _runtime_embedding


def _configure_runtime_episodic_storage(memory: MemorySystem, persist_dir: str) -> None:
    episodic_storage_path = Path(persist_dir) / "episodic_json"
    episodic_storage_path.mkdir(parents=True, exist_ok=True)
    memory.episodic.storage_path = episodic_storage_path
    memory.episodic._memory_cache = {}
    memory.episodic._load_from_json_backup()


def _build_runtime_memory(persist_dir: str) -> MemorySystem:
    with _suppress_runtime_logs():
        memory = MemorySystem(MemorySystemConfig(chroma_persist_dir=persist_dir))
    _patch_runtime_memory(memory)
    _configure_runtime_episodic_storage(memory, persist_dir)
    return memory


class _RuntimeRetrievalHarness:
    def __init__(
        self,
        *,
        temp_dir_handle: TemporaryDirectory[str] | None = None,
        persist_dir: str | None = None,
        episode_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._temp_dir_handle = temp_dir_handle or TemporaryDirectory(
            prefix="retrieval_runtime_",
            ignore_cleanup_errors=True,
        )
        self._persist_dir = persist_dir or self._temp_dir_handle.name
        self._episode_aliases = dict(episode_aliases or {})
        self.memory = _build_runtime_memory(self._persist_dir)
        self.relationship_store = RelationshipMemoryStore(storage_path=Path(self._persist_dir) / "relationships")

    def seed(self, scenario: "RetrievalScenario") -> None:
        with _suppress_runtime_logs():
            for result in scenario.ltm_results:
                if scenario.use_semantic_ltm_backend:
                    metadata = dict(result)
                    metadata.update(
                        {
                            "source": str(result.get("source") or "retrieval_eval"),
                            "confidence": float(result.get("confidence", 0.8) or 0.8),
                            "importance": float(result.get("importance", 0.5) or 0.5),
                            "relationship_target": result.get("relationship_target"),
                        }
                    )
                    stored_id = self.memory.store_fact(
                        str(result.get("subject") or "user"),
                        str(result.get("predicate") or "knows"),
                        result.get("object") or result.get("content") or "",
                        content=str(result.get("fact_text") or result.get("content") or ""),
                        fact_id=str(result.get("id") or ""),
                        metadata=metadata,
                    )
                    if stored_id:
                        continue

                content = str(result.get("fact_text") or result.get("content") or result.get("summary") or "")
                if not content:
                    continue
                self.memory.ltm.store(
                    memory_id=str(result.get("id") or f"ltm-{len(self._episode_aliases)}"),
                    content=content,
                    memory_type=str(result.get("memory_type") or "semantic"),
                    importance=float(result.get("importance", 0.5) or 0.5),
                    emotional_valence=float(result.get("emotional_valence", 0.0) or 0.0),
                    source=str(result.get("source") or "retrieval_eval"),
                    tags=list(result.get("tags", ())),
                    associations=list(result.get("associations", ())),
                )

            for result in scenario.episodic_results:
                alias = str(result.get("id") or f"episode-{len(self._episode_aliases) + 1}")
                episode_id = self.memory.create_episodic_memory(
                    summary=str(result.get("summary") or alias),
                    detailed_content=str(result.get("detailed_content") or result.get("content") or ""),
                    participants=list(result.get("participants", ())),
                    importance=float(result.get("importance", 0.5) or 0.5),
                    emotional_valence=float(result.get("emotional_valence", 0.0) or 0.0),
                    life_period=str(result.get("life_period") or "retrieval_eval"),
                )
                if episode_id:
                    self._episode_aliases[str(episode_id)] = alias
                    episode = self.memory.episodic._memory_cache.get(str(episode_id))
                    if episode is not None:
                        timestamp = result.get("timestamp")
                        last_access = result.get("last_access")
                        if isinstance(timestamp, str):
                            episode.timestamp = datetime.fromisoformat(timestamp)
                            episode.created_at = episode.timestamp
                        if isinstance(last_access, str):
                            episode.last_access = datetime.fromisoformat(last_access)
                            episode.updated_at = episode.last_access
                        if result.get("confidence") is not None:
                            episode.confidence = float(result.get("confidence") or episode.confidence)
                        if result.get("importance") is not None:
                            episode.importance = float(result.get("importance") or episode.importance)
                        if result.get("emotional_valence") is not None:
                            episode.emotional_valence = float(result.get("emotional_valence") or episode.emotional_valence)
                        if result.get("related_episodes") is not None:
                            episode.related_episodes = list(result.get("related_episodes") or [])
                        if result.get("tags") is not None:
                            episode.tags = list(result.get("tags") or [])
                        self.memory.episodic._save_to_json_backup(episode)

            if scenario.relationship_memory is not None:
                self.relationship_store.upsert(RelationshipMemory.from_dict(dict(scenario.relationship_memory)))

            if scenario.forgetting_policy is not None:
                self.memory.apply_forgetting_policy(**dict(scenario.forgetting_policy))

    def restart(self) -> "_RuntimeRetrievalHarness":
        with _suppress_runtime_logs():
            self.memory.shutdown()
        return _RuntimeRetrievalHarness(
            temp_dir_handle=self._temp_dir_handle,
            persist_dir=self._persist_dir,
            episode_aliases=self._episode_aliases,
        )

    def prepare_for_query(self, scenario: "RetrievalScenario") -> None:
        if scenario.disable_episodic_vector_search:
            self.memory.episodic.collection = None
            original_search = self.memory.episodic.search

            def _search_with_eval_threshold(query: str | None = None, **kwargs: Any):
                kwargs.setdefault("min_relevance", 0.0)
                return original_search(query=query, **kwargs)

            self.memory.episodic.search = _search_with_eval_threshold
            with _suppress_runtime_logs():
                self.memory.episodic.search(query=scenario.query, limit=max(len(scenario.expected_ids), 1))

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

    def alias_for(self, source_id: str) -> str:
        return self._episode_aliases.get(source_id, source_id)


@dataclass(slots=True)
class RetrievalScenario:
    scenario_id: str
    query: str
    expected_intent: str
    expected_ids: tuple[str, ...]
    forbidden_ids: tuple[str, ...] = ()
    min_precision: float = 1.0
    max_irrelevant_context_rate: float = 0.0
    session_turns: tuple[str, ...] = ()
    stm_results: tuple[Any, ...] = ()
    ltm_results: tuple[dict[str, Any], ...] = ()
    episodic_results: tuple[dict[str, Any], ...] = ()
    runner: str = "fixture"
    restart_before_query: bool = False
    disable_episodic_vector_search: bool = False
    chat_config: Mapping[str, Any] | None = None
    relationship_memory: Mapping[str, Any] | None = None
    use_semantic_ltm_backend: bool = False
    forgetting_policy: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RetrievalScenarioResult:
    scenario_id: str
    intent: str
    retrieved_ids: tuple[str, ...]
    missing_expected_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]
    metrics: RetrievalMetrics
    runner: str = "fixture"


def build_baseline_scenarios() -> list[RetrievalScenario]:
    now = datetime.now()
    return [
        RetrievalScenario(
            scenario_id="fact_recall_favorite_editor",
            query="What do you know about the user's favorite editor?",
            expected_intent="factual",
            expected_ids=("fact-editor",),
            session_turns=("We were talking about development tools.",),
            ltm_results=(
                {
                    "id": "fact-editor",
                    "subject": "user",
                    "predicate": "favorite_editor",
                    "object": "VS Code",
                    "fact_text": "The user's favorite editor is VS Code.",
                    "similarity_score": 0.91,
                    "confidence": 0.95,
                    "importance": 0.8,
                    "encoding_time": (now - timedelta(days=2)).isoformat(),
                    "last_access": (now - timedelta(hours=4)).isoformat(),
                },
                {
                    "id": "fact-shell",
                    "subject": "user",
                    "predicate": "uses_shell",
                    "object": "bash",
                    "fact_text": "The user uses bash on Windows.",
                    "similarity_score": 0.44,
                    "confidence": 0.8,
                    "importance": 0.6,
                    "encoding_time": (now - timedelta(days=5)).isoformat(),
                    "last_access": (now - timedelta(days=1)).isoformat(),
                },
            ),
        ),
        RetrievalScenario(
            scenario_id="episodic_recall_schema_discussion",
            query="When did we discuss the canonical schema?",
            expected_intent="episodic",
            expected_ids=("episode-schema",),
            session_turns=("I want to revisit our design work.",),
            episodic_results=(
                {
                    "id": "episode-schema",
                    "detailed_content": "Yesterday afternoon we discussed the canonical memory schema and its field validation rules.",
                    "summary": "Canonical schema discussion",
                    "relevance": 0.93,
                    "importance": 0.82,
                    "confidence": 0.9,
                    "timestamp": (now - timedelta(hours=18)).isoformat(),
                    "last_access": (now - timedelta(hours=12)).isoformat(),
                    "life_period": "phase_1_foundations",
                },
            ),
        ),
        RetrievalScenario(
            scenario_id="contradiction_recall_drink_preference",
            query="What do you know about whether I prefer tea or coffee?",
            expected_intent="factual",
            expected_ids=("preference-tea", "preference-coffee"),
            session_turns=("Let's sanity-check my stored preferences.",),
            ltm_results=(
                {
                    "id": "preference-tea",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "tea",
                    "fact_text": "The user prefers tea in the afternoon.",
                    "similarity_score": 0.92,
                    "confidence": 0.94,
                    "importance": 0.8,
                    "contradiction_set_id": "drink-preference",
                    "encoding_time": (now - timedelta(days=1)).isoformat(),
                    "last_access": (now - timedelta(hours=3)).isoformat(),
                },
                {
                    "id": "preference-coffee",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "coffee",
                    "fact_text": "The user previously said they preferred coffee for mornings.",
                    "similarity_score": 0.89,
                    "confidence": 0.88,
                    "importance": 0.76,
                    "contradiction_set_id": "drink-preference",
                    "encoding_time": (now - timedelta(days=6)).isoformat(),
                    "last_access": (now - timedelta(days=2)).isoformat(),
                },
                {
                    "id": "preference-mug",
                    "subject": "user",
                    "predicate": "owns",
                    "object": "blue mug",
                    "fact_text": "The user owns a blue mug.",
                    "similarity_score": 0.38,
                    "confidence": 0.7,
                    "importance": 0.4,
                    "encoding_time": (now - timedelta(days=10)).isoformat(),
                    "last_access": (now - timedelta(days=8)).isoformat(),
                },
            ),
        ),
        RetrievalScenario(
            scenario_id="runtime_fact_recall_favorite_editor",
            query="What do you know about the user's favorite editor?",
            expected_intent="factual",
            expected_ids=("fact-editor",),
            session_turns=("We were talking about development tools.",),
            ltm_results=(
                {
                    "id": "fact-editor",
                    "fact_text": "The user's favorite editor is VS Code.",
                    "importance": 0.8,
                    "source": "user_assertion",
                },
                {
                    "id": "fact-shell",
                    "fact_text": "The user uses bash on Windows.",
                    "importance": 0.6,
                    "source": "user_assertion",
                },
            ),
            runner="runtime",
            restart_before_query=True,
        ),
        RetrievalScenario(
            scenario_id="runtime_episodic_recall_schema_discussion",
            query="When did we discuss the canonical schema?",
            expected_intent="episodic",
            expected_ids=("episode-schema",),
            session_turns=("I want to revisit our design work.",),
            episodic_results=(
                {
                    "id": "episode-schema",
                    "summary": "Canonical schema discussion",
                    "detailed_content": "Yesterday afternoon we discussed the canonical memory schema and its field validation rules.",
                    "importance": 0.82,
                    "life_period": "phase_1_foundations",
                    "participants": ("user", "assistant"),
                },
            ),
            runner="runtime",
            restart_before_query=True,
            disable_episodic_vector_search=True,
            chat_config={"timeouts": {"retrieval_ms": 4000}},
        ),
        RetrievalScenario(
            scenario_id="runtime_contradiction_recall_drink_preference",
            query="What do you know about whether I prefer tea or coffee?",
            expected_intent="factual",
            expected_ids=("preference-tea",),
            forbidden_ids=("preference-coffee",),
            session_turns=("Let's sanity-check my stored preferences.",),
            ltm_results=(
                {
                    "id": "preference-coffee",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "coffee",
                    "fact_text": "The user previously said they preferred coffee for mornings.",
                    "source": "imported",
                    "confidence": 0.72,
                    "importance": 0.55,
                },
                {
                    "id": "preference-tea",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "tea",
                    "fact_text": "The user prefers tea in the afternoon.",
                    "source": "explicit_user_correction",
                    "confidence": 0.93,
                    "importance": 0.82,
                },
            ),
            runner="runtime",
            restart_before_query=True,
            use_semantic_ltm_backend=True,
        ),
        RetrievalScenario(
            scenario_id="runtime_quarantined_fact_stays_hidden",
            query="What do you know about whether I still prefer coffee?",
            expected_intent="factual",
            expected_ids=("preference-tea",),
            forbidden_ids=("preference-coffee",),
            session_turns=("Check whether the corrected preference stays consistent after restart.",),
            ltm_results=(
                {
                    "id": "preference-coffee",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "coffee",
                    "fact_text": "The user previously said they preferred coffee for mornings.",
                    "source": "imported",
                    "confidence": 0.72,
                    "importance": 0.55,
                },
                {
                    "id": "preference-tea",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "tea",
                    "fact_text": "The user corrected that they now prefer tea in the afternoon.",
                    "source": "explicit_user_correction",
                    "confidence": 0.94,
                    "importance": 0.84,
                },
            ),
            runner="runtime",
            restart_before_query=True,
            use_semantic_ltm_backend=True,
        ),
        RetrievalScenario(
            scenario_id="runtime_suppressed_low_value_fact_stays_hidden",
            query="What do you know about the shell I use on Windows?",
            expected_intent="factual",
            expected_ids=("fact-shell",),
            forbidden_ids=("fact-old-windows-note",),
            session_turns=("Check that stale low-value facts stop leaking back into context.",),
            ltm_results=(
                {
                    "id": "fact-shell",
                    "subject": "user",
                    "predicate": "uses_shell",
                    "object": "bash",
                    "fact_text": "The user uses bash on Windows.",
                    "source": "user_assertion",
                    "confidence": 0.86,
                    "importance": 0.82,
                    "encoding_time": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    "last_access": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                },
                {
                    "id": "fact-old-windows-note",
                    "subject": "user",
                    "predicate": "notes",
                    "object": "legacy windows workflow",
                    "fact_text": "The user had an old Windows workflow note.",
                    "source": "imported",
                    "confidence": 0.2,
                    "importance": 0.2,
                    "encoding_time": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
                    "last_access": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
                },
            ),
            runner="runtime",
            restart_before_query=True,
            use_semantic_ltm_backend=True,
            forgetting_policy={
                "min_importance": 0.3,
                "min_confidence": 0.35,
                "min_access_count": 1,
                "min_age_days": 14.0,
                "decay_stm": False,
            },
        ),
        RetrievalScenario(
            scenario_id="runtime_suppressed_low_value_episode_stays_hidden",
            query="When did we discuss the editor setup workaround?",
            expected_intent="episodic",
            expected_ids=("episode-current-editor-setup",),
            forbidden_ids=("episode-legacy-editor-workaround",),
            session_turns=("Check that stale low-value episodes stop resurfacing after restart.",),
            episodic_results=(
                {
                    "id": "episode-current-editor-setup",
                    "summary": "Current editor setup guidance",
                    "detailed_content": "Last week we discussed the editor setup workaround replacement and agreed to use VS Code tasks instead of the legacy wrapper.",
                    "importance": 0.64,
                    "confidence": 0.9,
                    "life_period": "general",
                    "timestamp": (now - timedelta(days=5)).isoformat(),
                    "last_access": (now - timedelta(days=1)).isoformat(),
                    "tags": ("editor", "setup", "workaround"),
                },
                {
                    "id": "episode-legacy-editor-workaround",
                    "summary": "Legacy editor setup workaround",
                    "detailed_content": "Two months ago we noted a temporary editor setup workaround using a legacy shell wrapper for the editor launch flow.",
                    "importance": 0.18,
                    "confidence": 0.2,
                    "life_period": "general",
                    "timestamp": (now - timedelta(days=45)).isoformat(),
                    "last_access": (now - timedelta(days=45)).isoformat(),
                    "tags": ("editor", "setup", "workaround", "legacy"),
                },
            ),
            runner="runtime",
            restart_before_query=True,
            disable_episodic_vector_search=True,
            chat_config={"timeouts": {"retrieval_ms": 4000}},
            forgetting_policy={
                "min_importance": 0.3,
                "min_confidence": 0.35,
                "min_access_count": 1,
                "min_age_days": 14.0,
                "decay_stm": False,
            },
        ),
        RetrievalScenario(
            scenario_id="runtime_autobiographical_continuity_roadmap",
            query="What changed lately in the memory roadmap?",
            expected_intent="continuity",
            expected_ids=("episode-runtime-hardening", "episode-schema-kickoff"),
            session_turns=("Summarize the recent roadmap continuity.",),
            episodic_results=(
                {
                    "id": "episode-schema-kickoff",
                    "summary": "Roadmap schema kickoff",
                    "detailed_content": "Earlier in the roadmap we defined the canonical memory schema and the initial implementation foundations.",
                    "importance": 0.6,
                    "confidence": 0.88,
                    "life_period": "phase_1_foundations",
                    "timestamp": (now - timedelta(days=30)).isoformat(),
                    "last_access": (now - timedelta(days=25)).isoformat(),
                    "tags": ("schema", "roadmap", "foundations"),
                },
                {
                    "id": "episode-runtime-hardening",
                    "summary": "Runtime eval hardening",
                    "detailed_content": "Lately we hardened runtime retrieval evaluation, restart continuity checks, and memory quality gates for the roadmap.",
                    "importance": 0.65,
                    "confidence": 0.9,
                    "life_period": "phase_4_memory",
                    "timestamp": (now - timedelta(days=1)).isoformat(),
                    "last_access": (now - timedelta(hours=6)).isoformat(),
                    "tags": ("runtime", "continuity", "roadmap"),
                },
            ),
            runner="runtime",
            restart_before_query=True,
            disable_episodic_vector_search=True,
            chat_config={"timeouts": {"retrieval_ms": 4000}},
        ),
        RetrievalScenario(
            scenario_id="runtime_autobiographical_defining_moment_priority",
            query="What changed lately in the project direction?",
            expected_intent="continuity",
            expected_ids=("episode-major-pivot", "episode-status-check"),
            session_turns=("Summarize the major project direction changes.",),
            episodic_results=(
                {
                    "id": "episode-major-pivot",
                    "summary": "Major project pivot",
                    "detailed_content": "We made a major project pivot that reoriented the roadmap around memory quality gates and deterministic evaluation.",
                    "importance": 0.82,
                    "confidence": 0.9,
                    "life_period": "phase_3_pivot",
                    "timestamp": (now - timedelta(days=20)).isoformat(),
                    "last_access": (now - timedelta(days=15)).isoformat(),
                    "tags": ("pivot", "roadmap", "quality"),
                },
                {
                    "id": "episode-status-check",
                    "summary": "Recent status check",
                    "detailed_content": "We recently checked project status and confirmed the current implementation backlog.",
                    "importance": 0.58,
                    "confidence": 0.86,
                    "life_period": "phase_4_memory",
                    "timestamp": (now - timedelta(days=1)).isoformat(),
                    "last_access": (now - timedelta(hours=8)).isoformat(),
                    "tags": ("status", "recent", "project"),
                },
            ),
            runner="runtime",
            restart_before_query=True,
            disable_episodic_vector_search=True,
            chat_config={"timeouts": {"retrieval_ms": 4000}},
        ),
        RetrievalScenario(
            scenario_id="runtime_social_recall_working_preferences",
            query="What do you know about our relationship?",
            expected_intent="social",
            expected_ids=("relationship-episode-user-42", "relationship-episode-maintainer-7"),
            session_turns=("Let's talk about how we work together.",),
            episodic_results=(
                {
                    "id": "relationship-episode-user-42",
                    "summary": "Working relationship discussion",
                    "detailed_content": "We discussed our relationship and the user's preference for direct answers when we work together.",
                    "importance": 0.84,
                    "life_period": "phase_2_autobiographical",
                    "participants": ("user-42",),
                },
                {
                    "id": "relationship-episode-maintainer-7",
                    "summary": "Another working relationship discussion",
                    "detailed_content": "We discussed our relationship with another maintainer and the cadence for code reviews.",
                    "importance": 0.74,
                    "life_period": "phase_2_autobiographical",
                    "participants": ("maintainer-7",),
                },
            ),
            runner="runtime",
            restart_before_query=True,
            disable_episodic_vector_search=True,
            chat_config={"timeouts": {"retrieval_ms": 4000}},
            relationship_memory={
                "interlocutor_id": "user-42",
                "display_name": "Primary user",
                "warmth": 0.9,
                "trust": 0.85,
                "familiarity": 0.9,
                "rupture": 0.1,
                "recurring_norms": ["prefer direct answers"],
            },
        ),
    ]


def _collect_retrieved_ids(
    built: Any,
    *,
    alias_lookup: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        alias_lookup.get(item.source_id, item.source_id) if alias_lookup else item.source_id
        for item in built.items
        if item.source_system in _MEMORY_SOURCE_SYSTEMS
    )


def _build_fixture_context(scenario: RetrievalScenario) -> tuple[Any, tuple[str, ...]]:
    session = SessionManager().create_or_get(scenario.scenario_id)
    session_turns = scenario.session_turns or (scenario.query,)
    for content in session_turns:
        session.add_turn(TurnRecord(role="user", content=content))
    if scenario.relationship_memory is not None:
        setattr(session, "_relationship_memory_snapshot", RelationshipMemory.from_dict(dict(scenario.relationship_memory)))

    builder = ContextBuilder(
        chat_config=dict(scenario.chat_config or {}),
        stm=_StaticSearchMemory(scenario.stm_results) if scenario.stm_results else None,
        ltm=_StaticSearchMemory(scenario.ltm_results) if scenario.ltm_results else None,
        episodic=_StaticSearchMemory(scenario.episodic_results) if scenario.episodic_results else None,
    )
    built = builder.build(session, query=scenario.query, include_trace=True)
    return built, _collect_retrieved_ids(built)


def _build_runtime_context(scenario: RetrievalScenario) -> tuple[Any, tuple[str, ...]]:
    harness = _RuntimeRetrievalHarness()
    try:
        harness.seed(scenario)
        if scenario.restart_before_query:
            harness = harness.restart()
        harness.prepare_for_query(scenario)

        session = SessionManager().create_or_get(scenario.scenario_id)
        session_turns = scenario.session_turns or (scenario.query,)
        for content in session_turns:
            session.add_turn(TurnRecord(role="user", content=content))
        if scenario.relationship_memory is not None:
            relationship = harness.relationship_store.get(
                str(scenario.relationship_memory.get("interlocutor_id") or "")
            )
            if relationship is not None:
                setattr(session, "_relationship_memory_snapshot", relationship)

        ltm_backend = None
        if scenario.ltm_results:
            ltm_backend = _SemanticAsLtmSearch(harness.memory.semantic) if scenario.use_semantic_ltm_backend else harness.memory.ltm

        builder = ContextBuilder(
            chat_config=dict(scenario.chat_config or {}),
            ltm=ltm_backend,
            episodic=harness.memory.episodic if scenario.episodic_results else None,
        )
        with _suppress_runtime_logs():
            built = builder.build(session, query=scenario.query, include_trace=True)
        return built, _collect_retrieved_ids(built, alias_lookup=harness._episode_aliases)
    finally:
        harness.close()


def run_retrieval_scenario(scenario: RetrievalScenario) -> RetrievalScenarioResult:
    if scenario.runner == "runtime":
        built, retrieved_ids = _build_runtime_context(scenario)
    else:
        built, retrieved_ids = _build_fixture_context(scenario)

    metrics = score_retrieval(retrieved_ids, scenario.expected_ids)
    retrieval_stage = next((stage for stage in built.trace.pipeline_stages if stage.name == "retrieval_plan"), None)
    intent = ""
    if retrieval_stage is not None:
        intent = str(retrieval_stage.metadata.get("intent", ""))

    missing_expected_ids = tuple(item_id for item_id in scenario.expected_ids if item_id not in retrieved_ids)
    unexpected_ids = tuple(item_id for item_id in retrieved_ids if item_id not in scenario.expected_ids)
    return RetrievalScenarioResult(
        scenario_id=scenario.scenario_id,
        intent=intent,
        retrieved_ids=retrieved_ids,
        missing_expected_ids=missing_expected_ids,
        unexpected_ids=unexpected_ids,
        metrics=metrics,
        runner=scenario.runner,
    )


def run_baseline_suite(scenarios: Iterable[RetrievalScenario] | None = None) -> list[RetrievalScenarioResult]:
    scenario_list = list(scenarios) if scenarios is not None else build_baseline_scenarios()
    return [run_retrieval_scenario(scenario) for scenario in scenario_list]