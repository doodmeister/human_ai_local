from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LongitudinalMetrics:
    continuity_score: float
    contradiction_repair_score: float
    over_recall_rate: float
    false_memory_count: int
    restart_count: int
    checked_artifact_count: int
    active_fact_count: int


def score_longitudinal(
    *,
    matched_continuity_checks: int,
    total_continuity_checks: int,
    repaired_contradictions: int,
    total_contradiction_checks: int,
    false_memory_count: int,
    active_fact_count: int,
    restart_count: int,
) -> LongitudinalMetrics:
    continuity_score = 1.0
    if total_continuity_checks > 0:
        continuity_score = matched_continuity_checks / total_continuity_checks

    contradiction_repair_score = 1.0
    if total_contradiction_checks > 0:
        contradiction_repair_score = repaired_contradictions / total_contradiction_checks

    over_recall_rate = 0.0
    if active_fact_count > 0:
        over_recall_rate = false_memory_count / active_fact_count

    return LongitudinalMetrics(
        continuity_score=continuity_score,
        contradiction_repair_score=contradiction_repair_score,
        over_recall_rate=over_recall_rate,
        false_memory_count=false_memory_count,
        restart_count=restart_count,
        checked_artifact_count=total_continuity_checks,
        active_fact_count=active_fact_count,
    )