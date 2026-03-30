from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BehaviorMetrics:
    alignment_score: float
    traceability_score: float
    stability_score: float
    checked_expectation_count: int
    matched_expectation_count: int


def score_behavior(
    *,
    matched_expectations: int,
    total_expectations: int,
    matched_traceability_checks: int,
    total_traceability_checks: int,
    stable_outputs: int,
    total_stability_checks: int,
) -> BehaviorMetrics:
    alignment_score = 1.0 if total_expectations == 0 else matched_expectations / total_expectations
    traceability_score = (
        1.0 if total_traceability_checks == 0 else matched_traceability_checks / total_traceability_checks
    )
    stability_score = 1.0 if total_stability_checks == 0 else stable_outputs / total_stability_checks
    return BehaviorMetrics(
        alignment_score=alignment_score,
        traceability_score=traceability_score,
        stability_score=stability_score,
        checked_expectation_count=total_expectations,
        matched_expectation_count=matched_expectations,
    )