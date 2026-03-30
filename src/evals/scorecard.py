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
    longitudinal_summary: dict[str, float]
    behavior_summary: dict[str, float]
    gates: tuple[ScorecardGateResult, ...]
    gate_failures: tuple[str, ...]
    telemetry_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_summary": dict(self.retrieval_summary),
            "longitudinal_summary": dict(self.longitudinal_summary),
            "behavior_summary": dict(self.behavior_summary),
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


def summarize_longitudinal_results(results: Iterable[LongitudinalScenarioResult]) -> dict[str, float]:
    result_list = list(results)
    count = len(result_list)
    if count == 0:
        return {
            "longitudinal_scenario_count": 0.0,
            "longitudinal_continuity_score_avg": 0.0,
            "longitudinal_contradiction_repair_score_avg": 0.0,
            "longitudinal_over_recall_rate_avg": 0.0,
            "longitudinal_false_memory_count_total": 0.0,
            "longitudinal_restart_count_total": 0.0,
        }

    return {
        "longitudinal_scenario_count": float(count),
        "longitudinal_continuity_score_avg": sum(result.metrics.continuity_score for result in result_list) / count,
        "longitudinal_contradiction_repair_score_avg": sum(result.metrics.contradiction_repair_score for result in result_list) / count,
        "longitudinal_over_recall_rate_avg": sum(result.metrics.over_recall_rate for result in result_list) / count,
        "longitudinal_false_memory_count_total": float(sum(result.metrics.false_memory_count for result in result_list)),
        "longitudinal_restart_count_total": float(sum(result.metrics.restart_count for result in result_list)),
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
    longitudinal_summary = summarize_longitudinal_results(longitudinal_result_list)
    behavior_summary = summarize_behavior_results(behavior_result_list)
    merged_metrics = {**retrieval_summary, **longitudinal_summary, **behavior_summary}
    gate_results = evaluate_scorecard_gates(merged_metrics, gates=gates)
    gate_failures = tuple(result.metric_key for result in gate_results if result.status == "fail")
    return MemoryQualityScorecard(
        retrieval_summary=retrieval_summary,
        longitudinal_summary=longitudinal_summary,
        behavior_summary=behavior_summary,
        gates=gate_results,
        gate_failures=gate_failures,
        telemetry_snapshot=telemetry_snapshot,
    )