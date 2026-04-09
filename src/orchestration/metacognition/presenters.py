from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def present_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = _normalize(payload)
    status.setdefault("available", bool(status.get("last_cycle") or status.get("last_trace") or status.get("persisted_self_model")))
    return status


def present_background_state(
    status_payload: Mapping[str, Any],
    *,
    tasks: list[dict[str, Any]] | None = None,
    reflections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status = present_status(status_payload)
    normalized_tasks = _normalize(tasks or [])
    normalized_reflections = _normalize(reflections or [])
    audit_tasks = [
        task
        for task in normalized_tasks
        if isinstance(task, dict) and str((task.get("metadata") or {}).get("reason") or "") == "background_contradiction_audit"
    ]
    idle_reflections = [
        report
        for report in normalized_reflections
        if isinstance(report, dict) and str((report.get("metadata") or {}).get("trigger") or "") == "idle_background_scheduler"
    ]
    last_idle_reflection = idle_reflections[-1] if idle_reflections else None
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
        "pending_audit_task_count": len(
            [task for task in audit_tasks if isinstance(task, dict) and str(task.get("status") or "") == "pending"]
        ),
        "idle_reflection_count": len(idle_reflections),
        "last_idle_reflection": last_idle_reflection,
    }


def present_scorecard(payload: Mapping[str, Any]) -> dict[str, Any]:
    scorecard = _normalize(payload)
    scorecard.setdefault("available", bool(scorecard.get("trace_count")))
    return scorecard


def present_dashboard(
    *,
    status_payload: Mapping[str, Any],
    background_payload: Mapping[str, Any],
    scorecard_payload: Mapping[str, Any],
) -> dict[str, Any]:
    status = present_status(status_payload)
    background = _normalize(background_payload)
    scorecard = present_scorecard(scorecard_payload)
    return {
        "available": bool(status.get("available") or background.get("available") or scorecard.get("available")),
        "session_id": status.get("session_id") or background.get("session_id") or scorecard.get("session_id"),
        "status": status,
        "background": background,
        "scorecard": scorecard,
    }


def present_cycle(payload: Mapping[str, Any]) -> dict[str, Any]:
    cycle = _normalize(payload)
    cycle.setdefault("available", bool(cycle))
    return cycle


def present_goals(goals: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize(goals)
    return {"count": len(normalized), "items": normalized}


def present_self_model(self_model: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize(self_model)
    normalized.setdefault("available", bool(normalized))
    return normalized


def present_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize(tasks)
    return {"count": len(normalized), "items": normalized}


def present_reflection_episodes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize(reports)
    return {"count": len(normalized), "items": normalized}


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value