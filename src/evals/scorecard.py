from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping

from src.evals.scenarios import (
    LongitudinalScenarioResult,
    PolicyBehaviorScenarioResult,
    RetrievalScenarioResult,
    run_baseline_suite,
    run_behavior_suite,
    run_longitudinal_suite,
)


GateStatus = Literal["pass", "fail"]
GateComparator = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class ScorecardGate:
    metric_key: str
    comparator: GateComparator
    threshold: float


@dataclass(frozen=True, slots=True)
class ScorecardGateResult:
    metric_key: str
    comparator: GateComparator
    threshold: float
    observed: float
    status: GateStatus


@dataclass(frozen=True, slots=True)
class MemoryQualityScorecard:
    retrieval_summary: dict[str, float]
    retrieval_runner_summaries: dict[str, dict[str, float]]
    longitudinal_summary: dict[str, float]
    longitudinal_runner_summaries: dict[str, dict[str, float]]
    behavior_summary: dict[str, float]
    behavior_runner_summaries: dict[str, dict[str, float]]
    runner_gate_failures: dict[str, dict[str, tuple[str, ...]]]
    gates: tuple[ScorecardGateResult, ...]
    gate_failures: tuple[str, ...]
    telemetry_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_summary": dict(self.retrieval_summary),
            "retrieval_runner_summaries": {
                runner: dict(summary)
                for runner, summary in self.retrieval_runner_summaries.items()
            },
            "longitudinal_summary": dict(self.longitudinal_summary),
            "longitudinal_runner_summaries": {
                runner: dict(summary)
                for runner, summary in self.longitudinal_runner_summaries.items()
            },
            "behavior_summary": dict(self.behavior_summary),
            "behavior_runner_summaries": {
                runner: dict(summary)
                for runner, summary in self.behavior_runner_summaries.items()
            },
            "runner_gate_failures": {
                domain: {runner: list(failures) for runner, failures in domain_failures.items()}
                for domain, domain_failures in self.runner_gate_failures.items()
            },
            "gates": [asdict(gate) for gate in self.gates],
            "gate_failures": list(self.gate_failures),
            "telemetry_snapshot": dict(self.telemetry_snapshot) if isinstance(self.telemetry_snapshot, dict) else None,
        }


def default_scorecard_gates() -> tuple[ScorecardGate, ...]:
    return (
        ScorecardGate(metric_key="retrieval_precision_avg", comparator="min", threshold=0.95),
        ScorecardGate(metric_key="retrieval_expected_coverage_avg", comparator="min", threshold=1.0),
        ScorecardGate(metric_key="retrieval_irrelevant_context_rate_avg", comparator="max", threshold=0.05),
        ScorecardGate(metric_key="longitudinal_continuity_score_avg", comparator="min", threshold=1.0),
        ScorecardGate(metric_key="longitudinal_chapter_continuity_score_avg", comparator="min", threshold=1.0),
        ScorecardGate(metric_key="longitudinal_contradiction_repair_score_avg", comparator="min", threshold=1.0),
        ScorecardGate(metric_key="longitudinal_over_recall_rate_avg", comparator="max", threshold=0.0),
        ScorecardGate(metric_key="longitudinal_false_memory_count_total", comparator="max", threshold=0.0),
        ScorecardGate(metric_key="behavior_alignment_score_avg", comparator="min", threshold=1.0),
        ScorecardGate(metric_key="behavior_traceability_score_avg", comparator="min", threshold=1.0),
        ScorecardGate(metric_key="behavior_stability_score_avg", comparator="min", threshold=1.0),
    )


def summarize_retrieval_results(results: Iterable[RetrievalScenarioResult]) -> dict[str, float]:
    result_list = list(results)
    count = len(result_list)
    if count == 0:
        return {
            "retrieval_scenario_count": 0.0,
            "retrieval_precision_avg": 0.0,
            "retrieval_expected_coverage_avg": 0.0,
            "retrieval_irrelevant_context_rate_avg": 0.0,
            "retrieval_missing_expected_total": 0.0,
        }

    return {
        "retrieval_scenario_count": float(count),
        "retrieval_precision_avg": sum(result.metrics.precision for result in result_list) / count,
        "retrieval_expected_coverage_avg": sum(result.metrics.expected_coverage for result in result_list) / count,
        "retrieval_irrelevant_context_rate_avg": sum(result.metrics.irrelevant_context_rate for result in result_list) / count,
        "retrieval_missing_expected_total": float(sum(len(result.missing_expected_ids) for result in result_list)),
    }


def summarize_retrieval_results_by_runner(results: Iterable[RetrievalScenarioResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RetrievalScenarioResult]] = {}
    for result in results:
        grouped.setdefault(str(result.runner or "unknown"), []).append(result)
    return {
        runner: summarize_retrieval_results(grouped_results)
        for runner, grouped_results in sorted(grouped.items())
    }


def summarize_longitudinal_results(results: Iterable[LongitudinalScenarioResult]) -> dict[str, float]:
    result_list = list(results)
    count = len(result_list)
    if count == 0:
        return {
            "longitudinal_scenario_count": 0.0,
            "longitudinal_continuity_score_avg": 0.0,
            "longitudinal_chapter_continuity_score_avg": 0.0,
            "longitudinal_contradiction_repair_score_avg": 0.0,
            "longitudinal_over_recall_rate_avg": 0.0,
            "longitudinal_false_memory_count_total": 0.0,
            "longitudinal_restart_count_total": 0.0,
        }

    return {
        "longitudinal_scenario_count": float(count),
        "longitudinal_continuity_score_avg": sum(result.metrics.continuity_score for result in result_list) / count,
        "longitudinal_chapter_continuity_score_avg": sum(
            result.metrics.chapter_continuity_score for result in result_list
        ) / count,
        "longitudinal_contradiction_repair_score_avg": sum(result.metrics.contradiction_repair_score for result in result_list) / count,
        "longitudinal_over_recall_rate_avg": sum(result.metrics.over_recall_rate for result in result_list) / count,
        "longitudinal_false_memory_count_total": float(sum(result.metrics.false_memory_count for result in result_list)),
        "longitudinal_restart_count_total": float(sum(result.metrics.restart_count for result in result_list)),
    }


def summarize_longitudinal_results_by_runner(results: Iterable[LongitudinalScenarioResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[LongitudinalScenarioResult]] = {}
    for result in results:
        grouped.setdefault(str(result.runner or "unknown"), []).append(result)
    return {
        runner: summarize_longitudinal_results(grouped_results)
        for runner, grouped_results in sorted(grouped.items())
    }


def summarize_behavior_results(results: Iterable[PolicyBehaviorScenarioResult]) -> dict[str, float]:
    result_list = list(results)
    count = len(result_list)
    if count == 0:
        return {
            "behavior_scenario_count": 0.0,
            "behavior_alignment_score_avg": 0.0,
            "behavior_traceability_score_avg": 0.0,
            "behavior_stability_score_avg": 0.0,
            "behavior_checked_expectation_count_total": 0.0,
        }

    return {
        "behavior_scenario_count": float(count),
        "behavior_alignment_score_avg": sum(result.metrics.alignment_score for result in result_list) / count,
        "behavior_traceability_score_avg": sum(result.metrics.traceability_score for result in result_list) / count,
        "behavior_stability_score_avg": sum(result.metrics.stability_score for result in result_list) / count,
        "behavior_checked_expectation_count_total": float(
            sum(result.metrics.checked_expectation_count for result in result_list)
        ),
    }


def summarize_behavior_results_by_runner(results: Iterable[PolicyBehaviorScenarioResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[PolicyBehaviorScenarioResult]] = {}
    for result in results:
        grouped.setdefault(str(result.runner or "unknown"), []).append(result)
    return {
        runner: summarize_behavior_results(grouped_results)
        for runner, grouped_results in sorted(grouped.items())
    }


def evaluate_scorecard_gates(
    metrics: Mapping[str, float],
    gates: Iterable[ScorecardGate] | None = None,
) -> tuple[ScorecardGateResult, ...]:
    gate_list = tuple(gates) if gates is not None else default_scorecard_gates()
    results: list[ScorecardGateResult] = []
    for gate in gate_list:
        observed = float(metrics.get(gate.metric_key, 0.0))
        passed = observed >= gate.threshold if gate.comparator == "min" else observed <= gate.threshold
        results.append(
            ScorecardGateResult(
                metric_key=gate.metric_key,
                comparator=gate.comparator,
                threshold=gate.threshold,
                observed=observed,
                status="pass" if passed else "fail",
            )
        )
    return tuple(results)


def evaluate_runner_gate_failures(
    runner_summaries: Mapping[str, Mapping[str, float]],
    *,
    metric_prefix: str,
    gates: Iterable[ScorecardGate] | None = None,
) -> dict[str, tuple[str, ...]]:
    gate_list = tuple(gates) if gates is not None else default_scorecard_gates()
    applicable_gates = tuple(gate for gate in gate_list if gate.metric_key.startswith(metric_prefix))
    failures: dict[str, tuple[str, ...]] = {}
    for runner, summary in sorted(runner_summaries.items()):
        results = evaluate_scorecard_gates(summary, gates=applicable_gates)
        failures[runner] = tuple(result.metric_key for result in results if result.status == "fail")
    return failures


def generate_memory_quality_scorecard(
    *,
    retrieval_results: Iterable[RetrievalScenarioResult] | None = None,
    longitudinal_results: Iterable[LongitudinalScenarioResult] | None = None,
    behavior_results: Iterable[PolicyBehaviorScenarioResult] | None = None,
    telemetry_snapshot: dict[str, Any] | None = None,
    gates: Iterable[ScorecardGate] | None = None,
) -> MemoryQualityScorecard:
    retrieval_result_list = list(retrieval_results) if retrieval_results is not None else run_baseline_suite()
    longitudinal_result_list = list(longitudinal_results) if longitudinal_results is not None else run_longitudinal_suite()
    behavior_result_list = list(behavior_results) if behavior_results is not None else run_behavior_suite()

    retrieval_summary = summarize_retrieval_results(retrieval_result_list)
    retrieval_runner_summaries = summarize_retrieval_results_by_runner(retrieval_result_list)
    longitudinal_summary = summarize_longitudinal_results(longitudinal_result_list)
    longitudinal_runner_summaries = summarize_longitudinal_results_by_runner(longitudinal_result_list)
    behavior_summary = summarize_behavior_results(behavior_result_list)
    behavior_runner_summaries = summarize_behavior_results_by_runner(behavior_result_list)
    runner_gate_failures = {
        "retrieval": evaluate_runner_gate_failures(
            retrieval_runner_summaries,
            metric_prefix="retrieval_",
            gates=gates,
        ),
        "longitudinal": evaluate_runner_gate_failures(
            longitudinal_runner_summaries,
            metric_prefix="longitudinal_",
            gates=gates,
        ),
        "behavior": evaluate_runner_gate_failures(
            behavior_runner_summaries,
            metric_prefix="behavior_",
            gates=gates,
        ),
    }
    merged_metrics = {**retrieval_summary, **longitudinal_summary, **behavior_summary}
    gate_results = evaluate_scorecard_gates(merged_metrics, gates=gates)
    gate_failures = tuple(result.metric_key for result in gate_results if result.status == "fail")
    return MemoryQualityScorecard(
        retrieval_summary=retrieval_summary,
        retrieval_runner_summaries=retrieval_runner_summaries,
        longitudinal_summary=longitudinal_summary,
        longitudinal_runner_summaries=longitudinal_runner_summaries,
        behavior_summary=behavior_summary,
        behavior_runner_summaries=behavior_runner_summaries,
        runner_gate_failures=runner_gate_failures,
        gates=gate_results,
        gate_failures=gate_failures,
        telemetry_snapshot=telemetry_snapshot,
    )