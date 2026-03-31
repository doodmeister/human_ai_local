from __future__ import annotations

import asyncio
from contextlib import contextmanager
import gc
import logging
import os
import re
from dataclasses import dataclass, field
from math import sqrt
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock
import uuid

from src.evals.metrics import BehaviorMetrics, score_behavior
from src.model.llm_provider import LLMResponse
from src.orchestration.cognitive_agent import CognitiveAgent
from src.orchestration.agent.llm_session import CognitiveAgentLLMSession
from src.orchestration.policy import PolicyVector, ResponsePolicy


_RUNTIME_EMBEDDING_TERMS = (
    "roadmap",
    "checkpoint",
    "previous",
    "session",
    "continue",
    "deterministic",
    "evaluation",
    "hardening",
    "summary",
    "situation",
    "help",
    "next",
)


def _build_config() -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            temperature=0.2,
            max_tokens=256,
            openai_model="test-model",
            ollama_model="test-ollama",
        )
    )


def _build_response_policy(
    *,
    warmth: float = 0.5,
    directness: float = 0.5,
    curiosity: float = 0.5,
    uncertainty: float = 0.5,
    disclosure: float = 0.5,
) -> ResponsePolicy:
    effective = PolicyVector(
        warmth=warmth,
        directness=directness,
        curiosity=curiosity,
        uncertainty=uncertainty,
        disclosure=disclosure,
    )
    return ResponsePolicy(
        stable_traits=PolicyVector(),
        dynamic_state=effective,
        effective=effective,
        trace={
            "dominant_signals": [{"source": "fixture", "magnitude": 1.0}],
            "working_self": {
                "dominant_drive": "understanding",
                "drive_pressure": 0.55,
                "mood_label": "steady",
                "mood_confidence": 0.8,
                "relationship_strength": 0.6,
                "self_regard": 0.15,
                "identity_stability": 0.85,
                "active_themes": ["clarity"],
                "ongoing_struggles": [],
            },
        },
    )


def _extract_metric(policy_block: str, metric: str) -> float:
    match = re.search(rf"- {metric}: ([0-9]+(?:\.[0-9]+)?)", policy_block)
    if match is None:
        raise ValueError(f"Missing metric {metric} in policy block")
    return float(match.group(1))


@dataclass
class PolicyAwareEvalProvider:
    last_messages: list[dict[str, str]] | None = None
    last_policy_values: dict[str, float] | None = None
    last_memory_context: str | None = None

    def is_available(self) -> bool:
        return True

    async def chat_completion(self, messages, temperature: float = 0.7, max_tokens: int = 512, **kwargs) -> LLMResponse:
        del temperature, max_tokens, kwargs
        self.last_messages = list(messages)
        policy_block = next(msg["content"] for msg in messages if msg["content"].startswith("[POLICY]"))
        self.last_memory_context = next(
            (msg["content"] for msg in messages if msg["content"].startswith("[MEMORY CONTEXT]")),
            None,
        )
        self.last_policy_values = {
            metric: _extract_metric(policy_block, metric)
            for metric in ["warmth", "directness", "curiosity", "uncertainty", "disclosure"]
        }

        user_message = messages[-1]["content"]
        parts: list[str] = []
        if self.last_policy_values["warmth"] >= 0.6:
            parts.append("Happy to help.")
        if self.last_policy_values["disclosure"] >= 0.6:
            parts.append("From my current perspective,")
        if self.last_policy_values["uncertainty"] >= 0.55:
            parts.append("I may be mistaken, but")
        if self.last_memory_context and "roadmap checkpoint from the previous session" in self.last_memory_context.lower():
            parts.append("Continuing from the stored checkpoint.")
        if self.last_policy_values["directness"] >= 0.6:
            parts.append(f"Answer: next step for '{user_message}' is to keep the change small.")
        else:
            parts.append(f"Context: for '{user_message}', it helps to weigh the tradeoffs before acting.")
        if self.last_policy_values["curiosity"] >= 0.6:
            parts.append("What constraint matters most?")

        return LLMResponse(
            content=" ".join(parts),
            model="policy-eval-provider",
            metadata={"policy_values": dict(self.last_policy_values)},
        )


def _build_session(provider: PolicyAwareEvalProvider) -> CognitiveAgentLLMSession:
    session = CognitiveAgentLLMSession(
        config=_build_config(),
        lazy_import_llm=lambda: (None, None),
    )
    session.provider = provider
    return session


@dataclass(frozen=True, slots=True)
class PolicyBehaviorScenario:
    scenario_id: str
    user_input: str
    response_policy: dict[str, Any]
    runner: str = "session"
    expected_present: tuple[str, ...] = ()
    expected_absent: tuple[str, ...] = ()
    expected_policy_values: dict[str, float] = field(default_factory=dict)
    expected_message_prefixes: tuple[str, ...] = ("[ROLE]", "[POLICY]", "[WORKING SELF]")
    require_stable_replay: bool = True
    restart_before_replay: bool = False
    use_runtime_memory_context: bool = False
    runtime_seed_memories: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyBehaviorScenarioResult:
    scenario_id: str
    runner: str
    response: str
    replay_response: str | None
    observed_policy_values: dict[str, float]
    metrics: BehaviorMetrics


class StubCognitiveLayers:
    def __init__(self, policy: ResponsePolicy) -> None:
        self._policy = policy
        self.calls: list[dict[str, object]] = []

    def process_turn(self, *, session, tick, message: str, salience: float, valence: float, global_turn_counter: int) -> None:
        self.calls.append(
            {
                "session": session,
                "message": message,
                "salience": salience,
                "valence": valence,
                "global_turn_counter": global_turn_counter,
            }
        )
        tick.state["response_policy"] = self._policy
        setattr(session, "_response_policy_snapshot", self._policy)

    def get_response_policy_state(self):
        return self._policy.to_dict()


def _response_policy_from_dict(policy: dict[str, Any]) -> ResponsePolicy:
    stable = PolicyVector(**dict(policy.get("stable_traits", {})))
    dynamic = PolicyVector(**dict(policy.get("dynamic_state", {})))
    effective = PolicyVector(**dict(policy.get("effective", {})))
    return ResponsePolicy(
        stable_traits=stable,
        dynamic_state=dynamic,
        effective=effective,
        trace=dict(policy.get("trace", {})),
    )


def _runtime_embedding(text: str) -> list[float]:
    lowered = text.lower()
    vector = [float(lowered.count(term)) for term in _RUNTIME_EMBEDDING_TERMS]
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude <= 0.0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def _patch_runtime_memory(memory: Any) -> None:
    for store_name in ("_semantic", "_ltm", "_episodic", "_stm"):
        store = getattr(memory, store_name, None)
        if store is None:
            continue
        if hasattr(store, "_ensure_embedding_model"):
            store._ensure_embedding_model = lambda: True
        if hasattr(store, "_generate_embedding"):
            store._generate_embedding = _runtime_embedding


async def _retrieve_persisted_runtime_memory_context(agent: CognitiveAgent, query: str) -> list[dict[str, Any]]:
    memories = agent.memory.search_memories(
        query=query,
        search_stm=False,
        search_episodic=False,
        max_results=3,
    )
    context_memories: list[dict[str, Any]] = []
    for memory_obj, relevance, source in memories:
        source_key = str(source).lower()
        if source_key != "ltm":
            continue
        if isinstance(memory_obj, dict):
            mem_id = memory_obj.get("id") or memory_obj.get("memory_id")
            mem_content = memory_obj.get("content", "")
            mem_timestamp = memory_obj.get("encoding_time")
        else:
            mem_id = getattr(memory_obj, "id", None)
            mem_content = getattr(memory_obj, "content", "")
            mem_timestamp = getattr(memory_obj, "encoding_time", None)
        context_memories.append(
            {
                "id": mem_id,
                "content": mem_content,
                "source": "LTM",
                "relevance": relevance,
                "timestamp": mem_timestamp,
            }
        )
    return context_memories


@contextmanager
def _temporary_runtime_env(*, persist_dir: str | None = None, collection_stem: str | None = None) -> Any:
    temp_dir_handle: TemporaryDirectory[str] | None = None
    if persist_dir is None:
        temp_dir_handle = TemporaryDirectory(prefix="policy_behavior_", ignore_cleanup_errors=True)
        temp_dir = temp_dir_handle.name
    else:
        temp_dir = persist_dir
    collection_seed = collection_stem or f"runtime_policy_{uuid.uuid4().hex[:8]}"
    try:
        previous = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "DISABLE_SEMANTIC_MEMORY": os.environ.get("DISABLE_SEMANTIC_MEMORY"),
            "CHROMA_PERSIST_DIR": os.environ.get("CHROMA_PERSIST_DIR"),
            "STM_COLLECTION": os.environ.get("STM_COLLECTION"),
            "LTM_COLLECTION": os.environ.get("LTM_COLLECTION"),
        }
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["DISABLE_SEMANTIC_MEMORY"] = "1"
        os.environ["CHROMA_PERSIST_DIR"] = temp_dir
        os.environ["STM_COLLECTION"] = f"{collection_seed}_stm"
        os.environ["LTM_COLLECTION"] = f"{collection_seed}_ltm"
        try:
            yield temp_dir
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    finally:
        if temp_dir_handle is not None:
            temp_dir_handle.cleanup()


@contextmanager
def _suppress_runtime_logs() -> Any:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def build_behavior_scenarios() -> list[PolicyBehaviorScenario]:
    return [
        PolicyBehaviorScenario(
            scenario_id="warmth_directness_high",
            user_input="continue the roadmap",
            response_policy=_build_response_policy(warmth=0.82, directness=0.79).to_dict(),
            expected_present=("Happy to help.", "Answer:"),
            expected_absent=("Context:",),
            expected_policy_values={"warmth": 0.82, "directness": 0.79},
        ),
        PolicyBehaviorScenario(
            scenario_id="uncertainty_disclosure_high",
            user_input="summarize the situation",
            response_policy=_build_response_policy(uncertainty=0.74, disclosure=0.71).to_dict(),
            expected_present=("I may be mistaken, but", "From my current perspective,"),
            expected_policy_values={"uncertainty": 0.74, "disclosure": 0.71},
        ),
        PolicyBehaviorScenario(
            scenario_id="curiosity_high",
            user_input="help me decide what to do next",
            response_policy=_build_response_policy(curiosity=0.78).to_dict(),
            expected_present=("What constraint matters most?",),
            expected_policy_values={"curiosity": 0.78},
        ),
        PolicyBehaviorScenario(
            scenario_id="traceability_full_policy",
            user_input="what should I do next?",
            response_policy=_build_response_policy(
                warmth=0.81,
                directness=0.76,
                curiosity=0.64,
                uncertainty=0.58,
                disclosure=0.66,
            ).to_dict(),
            expected_present=(
                "Happy to help.",
                "Answer:",
                "I may be mistaken, but",
                "From my current perspective,",
                "What constraint matters most?",
            ),
            expected_policy_values={
                "warmth": 0.81,
                "directness": 0.76,
                "curiosity": 0.64,
                "uncertainty": 0.58,
                "disclosure": 0.66,
            },
        ),
        PolicyBehaviorScenario(
            scenario_id="runtime_warmth_directness_high",
            user_input="continue the roadmap",
            runner="runtime",
            response_policy=_build_response_policy(warmth=0.82, directness=0.79).to_dict(),
            expected_present=("Happy to help.", "Answer:"),
            expected_absent=("Context:",),
            expected_policy_values={"warmth": 0.82, "directness": 0.79},
        ),
        PolicyBehaviorScenario(
            scenario_id="runtime_uncertainty_disclosure_high",
            user_input="summarize the situation",
            runner="runtime",
            response_policy=_build_response_policy(uncertainty=0.74, disclosure=0.71).to_dict(),
            expected_present=("I may be mistaken, but", "From my current perspective,"),
            expected_policy_values={"uncertainty": 0.74, "disclosure": 0.71},
        ),
        PolicyBehaviorScenario(
            scenario_id="runtime_restart_memory_continuity",
            user_input="continue from the roadmap checkpoint",
            runner="runtime",
            response_policy=_build_response_policy(warmth=0.72, directness=0.78).to_dict(),
            expected_present=(
                "Happy to help.",
                "Continuing from the stored checkpoint.",
                "Answer:",
            ),
            expected_absent=("Context:",),
            expected_policy_values={"warmth": 0.72, "directness": 0.78},
            expected_message_prefixes=("[ROLE]", "[POLICY]", "[WORKING SELF]", "[MEMORY CONTEXT]"),
            restart_before_replay=True,
            use_runtime_memory_context=True,
            runtime_seed_memories=(
                {
                    "memory_id": "ltm-roadmap-checkpoint",
                    "content": "Roadmap checkpoint from the previous session: continue deterministic evaluation hardening and keep the gate clean.",
                    "importance": 0.84,
                    "emotional_valence": 0.05,
                    "force_ltm": True,
                },
            ),
        ),
        PolicyBehaviorScenario(
            scenario_id="runtime_traceability_full_policy",
            user_input="help me continue the roadmap",
            runner="runtime",
            response_policy=_build_response_policy(
                warmth=0.81,
                directness=0.76,
                curiosity=0.64,
                uncertainty=0.58,
                disclosure=0.66,
            ).to_dict(),
            expected_present=(
                "Happy to help.",
                "Answer:",
                "I may be mistaken, but",
                "From my current perspective,",
                "What constraint matters most?",
            ),
            expected_policy_values={
                "warmth": 0.81,
                "directness": 0.76,
                "curiosity": 0.64,
                "uncertainty": 0.58,
                "disclosure": 0.66,
            },
        ),
    ]


async def _generate_with_fresh_session(scenario: PolicyBehaviorScenario) -> tuple[str, PolicyAwareEvalProvider]:
    provider = PolicyAwareEvalProvider()
    session = _build_session(provider)
    response = await session.generate_response(
        processed_input={"raw_input": scenario.user_input, "response_policy": scenario.response_policy},
        memory_context=[
            {
                "id": "ltm-1",
                "source": "LTM",
                "relevance": 0.82,
                "content": "The current goal is deterministic policy behavior evaluation.",
            }
        ],
    )
    return response, provider


async def _generate_with_runtime_agent(scenario: PolicyBehaviorScenario) -> tuple[str, PolicyAwareEvalProvider]:
    provider = PolicyAwareEvalProvider()
    with _temporary_runtime_env(), _suppress_runtime_logs():
        agent = CognitiveAgent()
        try:
            _patch_runtime_memory(agent.memory)
            agent._llm_session.provider = provider
            agent._llm_session.openai_client = None
            agent._cognitive_layers = StubCognitiveLayers(_response_policy_from_dict(scenario.response_policy))
            agent._turn_processor._get_cognitive_layers = lambda: agent._cognitive_layers
            if scenario.use_runtime_memory_context:
                async def _runtime_memory_context(processed_input):
                    return await _retrieve_persisted_runtime_memory_context(
                        agent,
                        str(processed_input.get("raw_input", "")),
                    )

                agent._turn_processor.retrieve_memory_context = _runtime_memory_context
            else:
                agent._turn_processor.retrieve_memory_context = AsyncMock(return_value=[])
            agent._turn_processor.calculate_attention_allocation = AsyncMock(return_value={"overall_salience": 0.5})
            agent._turn_processor.consolidate_memory = AsyncMock(return_value=None)
            response = await agent.process_input(scenario.user_input)
        finally:
            agent.memory.shutdown()
            del agent
            gc.collect()
    return response, provider


def _seed_runtime_memories(*, persist_dir: str, collection_stem: str, scenario: PolicyBehaviorScenario) -> None:
    if not scenario.runtime_seed_memories:
        return
    with _temporary_runtime_env(persist_dir=persist_dir, collection_stem=collection_stem), _suppress_runtime_logs():
        agent = CognitiveAgent()
        try:
            _patch_runtime_memory(agent.memory)
            for seed in scenario.runtime_seed_memories:
                agent.memory.store_memory(
                    memory_id=str(seed.get("memory_id") or uuid.uuid4()),
                    content=str(seed.get("content") or ""),
                    importance=float(seed.get("importance", 0.5) or 0.5),
                    emotional_valence=float(seed.get("emotional_valence", 0.0) or 0.0),
                    force_ltm=bool(seed.get("force_ltm", False)),
                )
        finally:
            agent.memory.shutdown()
            del agent
            gc.collect()


async def _generate_with_persisted_runtime_agent(
    scenario: PolicyBehaviorScenario,
    *,
    persist_dir: str,
    collection_stem: str,
) -> tuple[str, PolicyAwareEvalProvider]:
    provider = PolicyAwareEvalProvider()
    with _temporary_runtime_env(persist_dir=persist_dir, collection_stem=collection_stem), _suppress_runtime_logs():
        agent = CognitiveAgent()
        try:
            _patch_runtime_memory(agent.memory)
            agent._llm_session.provider = provider
            agent._llm_session.openai_client = None
            agent._cognitive_layers = StubCognitiveLayers(_response_policy_from_dict(scenario.response_policy))
            agent._turn_processor._get_cognitive_layers = lambda: agent._cognitive_layers
            if scenario.use_runtime_memory_context:
                async def _runtime_memory_context(processed_input):
                    return await _retrieve_persisted_runtime_memory_context(
                        agent,
                        str(processed_input.get("raw_input", "")),
                    )

                agent._turn_processor.retrieve_memory_context = _runtime_memory_context
            else:
                agent._turn_processor.retrieve_memory_context = AsyncMock(return_value=[])
            agent._turn_processor.calculate_attention_allocation = AsyncMock(return_value={"overall_salience": 0.5})
            agent._turn_processor.consolidate_memory = AsyncMock(return_value=None)
            response = await agent.process_input(scenario.user_input)
            return response, provider
        finally:
            agent.memory.shutdown()


async def _run_behavior_scenario_async(scenario: PolicyBehaviorScenario) -> PolicyBehaviorScenarioResult:
    replay_response = None
    replay_provider = None
    if scenario.runner == "runtime" and (scenario.restart_before_replay or scenario.runtime_seed_memories):
        collection_stem = f"policy_behavior_{scenario.scenario_id}"
        with TemporaryDirectory(prefix="policy_behavior_restart_", ignore_cleanup_errors=True) as persist_dir:
            _seed_runtime_memories(persist_dir=persist_dir, collection_stem=collection_stem, scenario=scenario)
            response, provider = await _generate_with_persisted_runtime_agent(
                scenario,
                persist_dir=persist_dir,
                collection_stem=collection_stem,
            )
            if scenario.require_stable_replay:
                replay_response, replay_provider = await _generate_with_persisted_runtime_agent(
                    scenario,
                    persist_dir=persist_dir,
                    collection_stem=collection_stem,
                )
    else:
        generator = _generate_with_runtime_agent if scenario.runner == "runtime" else _generate_with_fresh_session
        response, provider = await generator(scenario)
        if scenario.require_stable_replay:
            replay_response, replay_provider = await generator(scenario)

    matched_expectations = sum(1 for token in scenario.expected_present if token in response)
    matched_expectations += sum(1 for token in scenario.expected_absent if token not in response)
    total_expectations = len(scenario.expected_present) + len(scenario.expected_absent)

    observed_policy_values = dict(provider.last_policy_values or {})
    matched_traceability_checks = 0
    total_traceability_checks = len(scenario.expected_policy_values) + len(scenario.expected_message_prefixes)
    for key, expected in scenario.expected_policy_values.items():
        if observed_policy_values.get(key) == expected:
            matched_traceability_checks += 1
    last_messages = provider.last_messages or []
    message_contents = [message.get("content", "") for message in last_messages]
    for prefix in scenario.expected_message_prefixes:
        if any(content.startswith(prefix) for content in message_contents):
            matched_traceability_checks += 1

    stable_outputs = 0
    total_stability_checks = 1 if scenario.require_stable_replay else 0
    if scenario.require_stable_replay and replay_response == response:
        if replay_provider is not None and replay_provider.last_policy_values == provider.last_policy_values:
            stable_outputs = 1

    metrics = score_behavior(
        matched_expectations=matched_expectations,
        total_expectations=total_expectations,
        matched_traceability_checks=matched_traceability_checks,
        total_traceability_checks=total_traceability_checks,
        stable_outputs=stable_outputs,
        total_stability_checks=total_stability_checks,
    )
    return PolicyBehaviorScenarioResult(
        scenario_id=scenario.scenario_id,
        runner=scenario.runner,
        response=response,
        replay_response=replay_response,
        observed_policy_values=observed_policy_values,
        metrics=metrics,
    )


def run_behavior_scenario(scenario: PolicyBehaviorScenario) -> PolicyBehaviorScenarioResult:
    return asyncio.run(_run_behavior_scenario_async(scenario))


def run_behavior_suite(
    scenarios: tuple[PolicyBehaviorScenario, ...] | list[PolicyBehaviorScenario] | None = None,
) -> list[PolicyBehaviorScenarioResult]:
    scenario_list = list(scenarios) if scenarios is not None else build_behavior_scenarios()
    return [run_behavior_scenario(scenario) for scenario in scenario_list]