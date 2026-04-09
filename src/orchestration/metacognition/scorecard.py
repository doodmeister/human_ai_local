from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class MetacognitiveScorecard:
    session_id: str | None
    trace_count: int
    summary: dict[str, Any]
    contradictions: dict[str, Any]
    self_model: dict[str, Any]
    goals: dict[str, Any]
    latest_trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
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

    for trace in trace_list:
        latest_trace_id = str(trace.get("cycle_id") or latest_trace_id or "") or latest_trace_id
        critic_report = trace.get("critic_report") or {}
        success_score = critic_report.get("success_score")
        if isinstance(success_score, (int, float)):
            success_scores.append(float(success_score))
        if critic_report.get("follow_up_recommended"):
            follow_up_count += 1

        contradiction_counts.append(_contradiction_count(trace))

        selected_goal = (trace.get("plan") or {}).get("selected_goal") or {}
        goal_id = selected_goal.get("goal_id")
        if goal_id:
            selected_goal_ids.append(str(goal_id))
        goal_kind = str(selected_goal.get("kind") or "unknown")
        selected_goal_kind_counts[goal_kind] = selected_goal_kind_counts.get(goal_kind, 0) + 1

        self_model = trace.get("updated_self_model")
        if isinstance(self_model, Mapping):
            self_models.append(dict(self_model))

    confidence_drifts: list[float] = []
    trait_drifts: list[float] = []
    belief_churns: list[float] = []
    for previous, current in zip(self_models, self_models[1:]):
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

    goal_changes = sum(1 for previous, current in zip(selected_goal_ids, selected_goal_ids[1:]) if previous != current)
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
    critic_report = trace.get("critic_report") or {}
    contradictions = critic_report.get("contradictions_detected")
    if isinstance(contradictions, list):
        return len(contradictions)
    if isinstance(contradictions, tuple):
        return len(contradictions)
    workspace = trace.get("workspace") or {}
    workspace_contradictions = workspace.get("contradictions")
    if isinstance(workspace_contradictions, list):
        return len(workspace_contradictions)
    return 0


def _trait_drift(previous: Any, current: Any) -> float:
    prev_traits = dict(previous) if isinstance(previous, Mapping) else {}
    curr_traits = dict(current) if isinstance(current, Mapping) else {}
    keys = sorted(set(prev_traits) | set(curr_traits))
    if not keys:
        return 0.0
    deltas = [abs(float(curr_traits.get(key, 0.0)) - float(prev_traits.get(key, 0.0))) for key in keys]
    return sum(deltas) / len(deltas)


def _belief_churn(previous: Any, current: Any) -> float:
    prev_beliefs = {str(value) for value in previous} if isinstance(previous, (list, tuple)) else set()
    curr_beliefs = {str(value) for value in current} if isinstance(current, (list, tuple)) else set()
    union = prev_beliefs | curr_beliefs
    if not union:
        return 0.0
    return len(prev_beliefs ^ curr_beliefs) / len(union)


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _average(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    if not items:
        return None
    return sum(items) / len(items)