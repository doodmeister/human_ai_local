from __future__ import annotations

from typing import Any, Callable

from .enums import GoalKind, ScheduledTaskStatus
from .models import (
    CognitiveGoal,
    InternalStateSnapshot,
    ScheduledCognitiveTask,
    SelfModel,
    Stimulus,
)


class DefaultStateProvider:
    """Adapter that snapshots current memory, attention, goal, self-model, and task state."""

    def __init__(
        self,
        *,
        memory: Any = None,
        attention: Any = None,
        executive: Any = None,
        self_model_provider: Callable[[str], Any] | None = None,
        task_provider: Callable[[str], list[Any]] | None = None,
    ) -> None:
        self._memory = memory
        self._attention = attention
        self._executive = executive
        self._self_model_provider = self_model_provider
        self._task_provider = task_provider

    def snapshot(self, stimulus: Stimulus) -> InternalStateSnapshot:
        memory_status = self._get_memory_status()
        attention_status = self._get_attention_status()
        active_goals = tuple(self._get_active_goals())
        self_model = self._get_self_model(stimulus.session_id)
        pending_tasks = tuple(self._get_tasks(stimulus.session_id))
        cognitive_load = float(attention_status.get("cognitive_load", 0.0) or 0.0)
        uncertainty = float(memory_status.get("uncertainty", 0.0) or 0.0)

        return InternalStateSnapshot(
            session_id=stimulus.session_id,
            turn_index=stimulus.turn_index,
            memory_status=memory_status,
            attention_status=attention_status,
            active_goals=active_goals,
            self_model=self_model,
            pending_tasks=pending_tasks,
            cognitive_load=cognitive_load,
            uncertainty=uncertainty,
            metadata={"input_type": stimulus.input_type},
        )

    def _get_memory_status(self) -> dict[str, Any]:
        if self._memory is None or not hasattr(self._memory, "get_status"):
            return {}
        try:
            status = self._memory.get_status()
        except Exception:
            return {}
        return status if isinstance(status, dict) else {}

    def _get_attention_status(self) -> dict[str, Any]:
        if self._attention is None or not hasattr(self._attention, "get_attention_status"):
            return {}
        try:
            status = self._attention.get_attention_status()
        except Exception:
            return {}
        return status if isinstance(status, dict) else {}

    def _get_active_goals(self) -> list[CognitiveGoal]:
        if self._executive is None:
            return []

        goal_source = getattr(self._executive, "goal_manager", self._executive)
        if hasattr(goal_source, "get_active_goals"):
            try:
                goals = goal_source.get_active_goals() or []
            except Exception:
                goals = []
        else:
            goals = []

        normalized: list[CognitiveGoal] = []
        for goal in goals:
            normalized.append(self._coerce_goal(goal))
        return normalized

    def _get_self_model(self, session_id: str) -> SelfModel:
        if self._self_model_provider is None:
            return SelfModel(session_id=session_id)

        try:
            raw_model = self._self_model_provider(session_id)
        except Exception:
            return SelfModel(session_id=session_id)

        if isinstance(raw_model, SelfModel):
            return raw_model
        if isinstance(raw_model, dict):
            traits = raw_model.get("traits") or {}
            beliefs = raw_model.get("beliefs") or ()
            recent_updates = raw_model.get("recent_updates") or ()
            return SelfModel(
                session_id=session_id,
                confidence=float(raw_model.get("confidence", 0.5) or 0.5),
                traits=dict(traits) if isinstance(traits, dict) else {},
                beliefs=tuple(beliefs) if isinstance(beliefs, (list, tuple)) else (),
                recent_updates=tuple(recent_updates) if isinstance(recent_updates, (list, tuple)) else (),
                metadata={
                    key: value
                    for key, value in raw_model.items()
                    if key not in {"confidence", "traits", "beliefs", "recent_updates"}
                },
            )
        return SelfModel(session_id=session_id)

    def _get_tasks(self, session_id: str) -> list[ScheduledCognitiveTask]:
        if self._task_provider is None:
            return []
        try:
            tasks = self._task_provider(session_id) or []
        except Exception:
            return []

        normalized: list[ScheduledCognitiveTask] = []
        for task in tasks:
            if isinstance(task, ScheduledCognitiveTask):
                normalized.append(task)
                continue
            if not isinstance(task, dict):
                continue
            normalized.append(
                ScheduledCognitiveTask(
                    task_id=str(task.get("task_id") or task.get("id") or f"task-{len(normalized)}"),
                    description=str(task.get("description") or task.get("content") or ""),
                    status=ScheduledTaskStatus(str(task.get("status") or ScheduledTaskStatus.PENDING.value)),
                    priority=float(task.get("priority", 0.0) or 0.0),
                    due_at=task.get("due_at"),
                    goal_id=task.get("goal_id"),
                    metadata={k: v for k, v in task.items() if k not in {"task_id", "id", "description", "content", "status", "priority", "due_at", "goal_id"}},
                )
            )
        return normalized

    @staticmethod
    def _coerce_goal(goal: Any) -> CognitiveGoal:
        priority = getattr(goal, "priority", 0.0)
        if hasattr(priority, "value"):
            priority_value = float(priority.value)
        else:
            priority_value = float(priority or 0.0)

        return CognitiveGoal(
            goal_id=str(getattr(goal, "id", "")),
            description=str(getattr(goal, "description", getattr(goal, "title", ""))),
            priority=priority_value,
            kind=GoalKind.PLANNING,
            urgency=float(getattr(goal, "progress", 0.0) or 0.0),
            salience=priority_value,
            rationale=str(getattr(goal, "status", "")),
            metadata={
                "title": getattr(goal, "title", ""),
                "status": getattr(getattr(goal, "status", None), "value", getattr(goal, "status", None)),
            },
        )
