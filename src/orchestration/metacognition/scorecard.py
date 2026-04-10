from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Iterable, Mapping

__all__ = ["MetacognitiveScorecard", "build_metacognitive_scorecard"]


@dataclass(frozen=True, slots=True)
class MetacognitiveScorecard:
    """Aggregate metacognition trace metrics for display and monitoring.

    The dataclass is frozen at the attribute level, but nested dictionaries are
    still mutable and should be treated as read-only by callers.
    """

    session_id: str | None
    trace_count: int
    summary: dict[str, Any]
    contradictions: dict[str, Any]
    self_model: dict[str, Any]
    goals: dict[str, Any]
    latest_trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dictionary view of the scorecard for transport.

        Top-level dictionaries are copied, but nested structures remain shared.
        """
        return {
            "available": self.trace_count > 0,
            "session_id": self.session_id,
            "trace_count": self.trace_count,
            "latest_trace_id": self.latest_trace_id,
            "summary": dict(self.summary),
            "contradictions": dict(self.contradictions),
            "self_model": dict(self.self_model),
            "goals": dict(self.goals),
        }


def build_metacognitive_scorecard(
    traces: Iterable[Mapping[str, Any]],
    *,
    session_id: str | None = None,
) -> MetacognitiveScorecard:
    """Summarize persisted metacognition traces into scorecard metrics.

    `None` means a metric could not be computed from the available traces,
    whereas numeric zero means the metric was computed and its value is zero.
    """
    trace_list = [dict(trace) for trace in traces]
    if not trace_list:
        return MetacognitiveScorecard(
            session_id=session_id,
            trace_count=0,
            latest_trace_id=None,
            summary={
                "cycle_success_avg": None,
                "follow_up_rate": None,
            },
            contradictions={
                "contradiction_rate": None,
                "contradiction_count_avg": None,
            },
            self_model={
                "self_model_drift_avg": None,
                "confidence_drift_avg": None,
                "trait_drift_avg": None,
                "belief_churn_avg": None,
            },
            goals={
                "goal_churn_rate": None,
                "selected_goal_kind_counts": {},
            },
        )

    success_scores: list[float] = []
    follow_up_count = 0
    contradiction_counts: list[int] = []
    selected_goal_ids: list[str] = []
    selected_goal_kind_counts: dict[str, int] = {}
    self_models: list[dict[str, Any]] = []
    latest_trace_id = None
    latest_trace_timestamp = float("-inf")

    for trace in trace_list:
        cycle_id = trace.get("cycle_id")
        timestamp = _to_float((trace.get("critic_report") or {}).get("timestamp")) or 0.0
        if cycle_id and timestamp >= latest_trace_timestamp:
            latest_trace_timestamp = timestamp
            latest_trace_id = str(cycle_id)

        critic_report = trace.get("critic_report") or {}
        success_score = _to_float(critic_report.get("success_score"))
        if success_score is not None:
            success_scores.append(success_score)
        if critic_report.get("follow_up_recommended"):
            follow_up_count += 1

        contradiction_counts.append(_contradiction_count(trace))

        selected_goal = (trace.get("plan") or {}).get("selected_goal") or {}
        goal_id = selected_goal.get("goal_id")
        if goal_id:
            selected_goal_ids.append(str(goal_id))
        raw_goal_kind = selected_goal.get("kind")
        raw_goal_kind = getattr(raw_goal_kind, "value", raw_goal_kind)
        goal_kind = str(raw_goal_kind) if raw_goal_kind else "unknown"
        selected_goal_kind_counts[goal_kind] = selected_goal_kind_counts.get(goal_kind, 0) + 1

        self_model = trace.get("updated_self_model")
        if isinstance(self_model, Mapping):
            self_models.append(dict(self_model))

    confidence_drifts: list[float] = []
    trait_drifts: list[float] = []
    belief_churns: list[float] = []
    for previous, current in pairwise(self_models):
        prev_confidence = _to_float(previous.get("confidence"))
        curr_confidence = _to_float(current.get("confidence"))
        if prev_confidence is not None and curr_confidence is not None:
            confidence_drifts.append(abs(curr_confidence - prev_confidence))
        trait_drifts.append(_trait_drift(previous.get("traits"), current.get("traits")))
        belief_churns.append(_belief_churn(previous.get("beliefs"), current.get("beliefs")))

    self_model_drift_components = [
        value
        for value in (
            _average(confidence_drifts),
            _average(trait_drifts),
            _average(belief_churns),
        )
        if value is not None
    ]

    goal_changes = sum(1 for previous, current in pairwise(selected_goal_ids) if previous != current)
    goal_churn_rate = None
    if len(selected_goal_ids) > 1:
        goal_churn_rate = goal_changes / (len(selected_goal_ids) - 1)

    return MetacognitiveScorecard(
        session_id=session_id,
        trace_count=len(trace_list),
        latest_trace_id=latest_trace_id,
        summary={
            "cycle_success_avg": _average(success_scores),
            "follow_up_rate": follow_up_count / len(trace_list),
        },
        contradictions={
            "contradiction_rate": sum(1 for count in contradiction_counts if count > 0) / len(trace_list),
            "contradiction_count_avg": sum(contradiction_counts) / len(trace_list),
        },
        self_model={
            "self_model_drift_avg": _average(self_model_drift_components),
            "confidence_drift_avg": _average(confidence_drifts),
            "trait_drift_avg": _average(trait_drifts),
            "belief_churn_avg": _average(belief_churns),
        },
        goals={
            "goal_churn_rate": goal_churn_rate,
            "selected_goal_kind_counts": selected_goal_kind_counts,
        },
    )


def _contradiction_count(trace: Mapping[str, Any]) -> int:
    """Return the contradiction count, preferring critic output over workspace fallback."""
    critic_report = trace.get("critic_report") or {}
    contradictions = critic_report.get("contradictions_detected")
    if isinstance(contradictions, (list, tuple)):
        return len(contradictions)
    workspace = trace.get("workspace") or {}
    workspace_contradictions = workspace.get("contradictions")
    if isinstance(workspace_contradictions, (list, tuple)):
        return len(workspace_contradictions)
    return 0


def _trait_drift(previous: Any, current: Any) -> float:
    """Return the mean absolute numeric drift across trait keys."""
    prev_traits = dict(previous) if isinstance(previous, Mapping) else {}
    curr_traits = dict(current) if isinstance(current, Mapping) else {}
    keys = set(prev_traits) | set(curr_traits)
    if not keys:
        return 0.0
    deltas: list[float] = []
    for key in keys:
        previous_value = _to_float(prev_traits.get(key))
        current_value = _to_float(curr_traits.get(key))
        if previous_value is None and current_value is None:
            continue
        deltas.append(abs((current_value or 0.0) - (previous_value or 0.0)))
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _belief_churn(previous: Any, current: Any) -> float:
    """Return a Jaccard-style churn ratio for belief collections."""
    prev_beliefs = {str(value) for value in previous} if isinstance(previous, (list, tuple)) else set()
    curr_beliefs = {str(value) for value in current} if isinstance(current, (list, tuple)) else set()
    union = prev_beliefs | curr_beliefs
    if not union:
        return 0.0
    return len(prev_beliefs ^ curr_beliefs) / len(union)


def _to_float(value: Any) -> float | None:
    """Return a numeric value for ints and floats, rejecting bools and non-numerics."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _average(values: Iterable[float]) -> float | None:
    """Return the mean of numeric values, skipping non-numeric entries."""
    items: list[float] = []
    for value in values:
        coerced = _to_float(value)
        if coerced is not None:
            items.append(coerced)
    if not items:
        return None
    return sum(items) / len(items)