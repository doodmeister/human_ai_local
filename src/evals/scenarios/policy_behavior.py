from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from src.evals.metrics import BehaviorMetrics, score_behavior
from src.model.llm_provider import LLMResponse
from src.orchestration.agent.llm_session import CognitiveAgentLLMSession
from src.orchestration.policy import PolicyVector, ResponsePolicy


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

    def is_available(self) -> bool:
        return True

    async def chat_completion(self, messages, temperature: float = 0.7, max_tokens: int = 512, **kwargs) -> LLMResponse:
        del temperature, max_tokens, kwargs
        self.last_messages = list(messages)
        policy_block = next(msg["content"] for msg in messages if msg["content"].startswith("[POLICY]"))
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
    expected_present: tuple[str, ...] = ()
    expected_absent: tuple[str, ...] = ()
    expected_policy_values: dict[str, float] = field(default_factory=dict)
    expected_message_prefixes: tuple[str, ...] = ("[ROLE]", "[POLICY]", "[WORKING SELF]")
    require_stable_replay: bool = True


@dataclass(frozen=True, slots=True)
class PolicyBehaviorScenarioResult:
    scenario_id: str
    response: str
    replay_response: str | None
    observed_policy_values: dict[str, float]
    metrics: BehaviorMetrics


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


async def _run_behavior_scenario_async(scenario: PolicyBehaviorScenario) -> PolicyBehaviorScenarioResult:
    response, provider = await _generate_with_fresh_session(scenario)
    replay_response = None
    replay_provider = None
    if scenario.require_stable_replay:
        replay_response, replay_provider = await _generate_with_fresh_session(scenario)

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