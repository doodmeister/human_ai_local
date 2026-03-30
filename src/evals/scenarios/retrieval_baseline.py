from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

from src.evals.metrics import RetrievalMetrics, score_retrieval
from src.orchestration.chat.context_builder import ContextBuilder
from src.orchestration.chat.conversation_session import SessionManager
from src.orchestration.chat.models import TurnRecord


_MEMORY_SOURCE_SYSTEMS = {"stm", "ltm", "episodic"}


class _StaticSearchMemory:
    def __init__(self, results: Iterable[Any]) -> None:
        self._results = list(results)

    def search(self, query: str, max_results: int | None = None, limit: int | None = None) -> list[Any]:
        del query
        cap = limit if limit is not None else max_results
        if cap is None:
            return list(self._results)
        return list(self._results[:cap])


@dataclass(slots=True)
class RetrievalScenario:
    scenario_id: str
    query: str
    expected_intent: str
    expected_ids: tuple[str, ...]
    min_precision: float = 1.0
    max_irrelevant_context_rate: float = 0.0
    session_turns: tuple[str, ...] = ()
    stm_results: tuple[Any, ...] = ()
    ltm_results: tuple[dict[str, Any], ...] = ()
    episodic_results: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalScenarioResult:
    scenario_id: str
    intent: str
    retrieved_ids: tuple[str, ...]
    missing_expected_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]
    metrics: RetrievalMetrics


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
    ]


def run_retrieval_scenario(scenario: RetrievalScenario) -> RetrievalScenarioResult:
    session = SessionManager().create_or_get(scenario.scenario_id)
    session_turns = scenario.session_turns or (scenario.query,)
    for content in session_turns:
        session.add_turn(TurnRecord(role="user", content=content))

    builder = ContextBuilder(
        stm=_StaticSearchMemory(scenario.stm_results) if scenario.stm_results else None,
        ltm=_StaticSearchMemory(scenario.ltm_results) if scenario.ltm_results else None,
        episodic=_StaticSearchMemory(scenario.episodic_results) if scenario.episodic_results else None,
    )
    built = builder.build(session, query=scenario.query, include_trace=True)

    retrieved_ids = tuple(
        item.source_id
        for item in built.items
        if item.source_system in _MEMORY_SOURCE_SYSTEMS
    )
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
    )


def run_baseline_suite(scenarios: Iterable[RetrievalScenario] | None = None) -> list[RetrievalScenarioResult]:
    scenario_list = list(scenarios) if scenarios is not None else build_baseline_scenarios()
    return [run_retrieval_scenario(scenario) for scenario in scenario_list]