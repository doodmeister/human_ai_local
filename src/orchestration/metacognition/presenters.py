from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from .enums import ReflectionTrigger, ScheduledTaskStatus, TaskReason
from .models import AttentionStatus, AttentionUpdate, ContradictionRecord, FocusItem, GoalMetadata, MemoryStatus, MemoryUpdate, PlanMetadata, RetrievalContextItem

__all__ = [
    "BackgroundState",
    "present_background_state",
    "present_cycle",
    "present_dashboard",
    "present_goals",
    "present_reflection_episodes",
    "present_scorecard",
    "present_self_model",
    "present_status",
    "present_tasks",
]


class BackgroundState(TypedDict):
    available: bool
    session_id: str | None
    scheduler_running: bool
    tick_interval_seconds: float | None
    idle_reflection_interval_seconds: float | None
    last_idle_reflection_ts: float | None
    pending_task_count: int
    due_task_count: int
    unresolved_contradiction_count: int
    audit_task_count: int
    pending_audit_task_count: int
    idle_reflection_count: int
    last_idle_reflection: dict[str, Any] | None


def present_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project controller status into a normalized mapping for API consumers.

    Ensures an `available` boolean is always present. When the producer omits
    it, availability means at least one persisted metacognition artifact key is
    present.
    """
    status = _normalize_mapping(payload)
    status.setdefault(
        "available",
        any(key in status for key in ("last_cycle", "last_trace", "persisted_self_model")),
    )
    return status


def present_background_state(
    status_payload: Mapping[str, Any],
    *,
    tasks: list[dict[str, Any]] | None = None,
    reflections: list[dict[str, Any]] | None = None,
) -> BackgroundState:
    """Project scheduler and reflection state into a stable background block.

    The status payload typically comes from `MetacognitiveController.build_status`
    or `CognitiveAgent.get_metacognitive_status`; task and reflection lists come
    from the task queue and cycle-tracer APIs.
    """
    status = present_status(status_payload)
    normalized_tasks = _normalize_list(tasks or [], expected_name="tasks")
    normalized_reflections = _normalize_list(reflections or [], expected_name="reflections")
    audit_tasks = [
        task
        for task in normalized_tasks
        if isinstance(task, dict)
        and str((task.get("metadata") or {}).get("reason") or "")
        == TaskReason.BACKGROUND_CONTRADICTION_AUDIT.value
    ]
    idle_reflections = [
        report
        for report in normalized_reflections
        if isinstance(report, dict)
        and str((report.get("metadata") or {}).get("trigger") or "")
        == ReflectionTrigger.IDLE_SCHEDULER.value
    ]
    last_idle_reflection = idle_reflections[-1] if idle_reflections else None

    # When the producer omits queue counts, the caller has already supplied the
    # loaded task list so we can still derive pending tasks locally. Due-task
    # and contradiction counts remain producer-owned because they depend on the
    # controller's timing and contradiction-resolution state.
    return {
        "available": bool(status.get("available") or normalized_tasks or normalized_reflections),
        "session_id": status.get("session_id"),
        "scheduler_running": bool(status.get("background_scheduler_running", False)),
        "tick_interval_seconds": status.get("background_tick_interval_seconds"),
        "idle_reflection_interval_seconds": status.get("idle_reflection_interval_seconds"),
        "last_idle_reflection_ts": status.get("last_idle_reflection_ts"),
        "pending_task_count": status.get("pending_task_count", len(normalized_tasks)),
        "due_task_count": status.get("due_task_count", 0),
        "unresolved_contradiction_count": status.get("unresolved_contradiction_count", 0),
        "audit_task_count": len(audit_tasks),
        "pending_audit_task_count": sum(
            1 for task in audit_tasks if str(task.get("status") or "") == ScheduledTaskStatus.PENDING.value
        ),
        "idle_reflection_count": len(idle_reflections),
        "last_idle_reflection": last_idle_reflection,
    }


def present_scorecard(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a scorecard payload for dashboard and API consumers."""
    scorecard = _normalize_mapping(payload)
    scorecard.setdefault("available", bool(scorecard.get("trace_count")))
    return scorecard


def present_dashboard(
    *,
    status_payload: Mapping[str, Any],
    background_payload: Mapping[str, Any],
    scorecard_payload: Mapping[str, Any],
    tasks: list[dict[str, Any]] | None = None,
    reflections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the metacognition dashboard payload.

    When `tasks` or `reflections` are provided, the background block is derived
    through `present_background_state` so the dashboard matches the standalone
    background endpoint. Without them, `background_payload` is treated as an
    already-projected background slice for backward compatibility.
    """
    status = present_status(status_payload)
    if tasks is not None or reflections is not None:
        background = present_background_state(
            background_payload,
            tasks=tasks,
            reflections=reflections,
        )
    else:
        background = _normalize_mapping(background_payload)
    scorecard = present_scorecard(scorecard_payload)
    return {
        "available": bool(status.get("available") or background.get("available") or scorecard.get("available")),
        "session_id": status.get("session_id") or background.get("session_id") or scorecard.get("session_id"),
        "status": status,
        "background": background,
        "scorecard": scorecard,
    }


def present_cycle(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the last-cycle payload while preserving a stable API seam.

    This remains a thin wrapper so callers can depend on a public presenter even
    though cycle payloads are currently returned almost verbatim.
    """
    cycle = _normalize_mapping(payload)
    cycle.setdefault("available", bool(cycle))
    return cycle


def present_goals(goals: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap normalized active-goal payloads in a count-and-items envelope."""
    return _present_list(goals, expected_name="goals")


def present_self_model(self_model: Mapping[str, Any]) -> dict[str, Any]:
    """Project a persisted self-model payload and ensure `available` exists."""
    normalized = _normalize_mapping(self_model)
    normalized.setdefault("available", bool(normalized))
    return normalized


def present_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap normalized scheduled-task payloads in a count-and-items envelope."""
    return _present_list(tasks, expected_name="tasks")


def present_reflection_episodes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap normalized reflection reports in a count-and-items envelope."""
    return _present_list(reports, expected_name="reports")


def _present_list(items: Any, *, expected_name: str) -> dict[str, Any]:
    normalized = _normalize_list(items, expected_name=expected_name)
    return {"count": len(normalized), "items": normalized}


def _normalize_mapping(value: Any) -> dict[str, Any]:
    normalized = _normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected mapping, got {type(value).__name__}")
    return normalized


def _normalize_list(value: Any, *, expected_name: str) -> list[Any]:
    normalized = _normalize(value)
    if not isinstance(normalized, list):
        raise TypeError(f"Expected {expected_name} to normalize to a list, got {type(value).__name__}")
    return normalized


def _normalize(value: Any) -> Any:
    if isinstance(value, (ContradictionRecord, RetrievalContextItem, MemoryStatus, AttentionStatus, GoalMetadata, PlanMetadata, MemoryUpdate, AttentionUpdate)):
        return _normalize(value.to_dict())
    if isinstance(value, FocusItem):
        return _normalize(value.to_primitive())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    # Payloads emitted by the tracer and controller are JSON-serializable, so
    # this normalizer assumes acyclic mappings and sequences.
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value