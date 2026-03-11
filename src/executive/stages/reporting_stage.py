from __future__ import annotations

from typing import Any, Callable, Dict

from src.executive.goal_manager import GoalStatus


class ExecutiveReportingStage:
    def __init__(
        self,
        *,
        get_goal_manager: Callable[[], Any],
        get_execution_contexts: Callable[[], Dict[str, Any]],
        get_active_schedules: Callable[[], Dict[str, Any]],
        get_outcome_tracker: Callable[[], Any],
    ) -> None:
        self._get_goal_manager = get_goal_manager
        self._get_execution_contexts = get_execution_contexts
        self._get_active_schedules = get_active_schedules
        self._get_outcome_tracker = get_outcome_tracker

    def get_system_health(self, *, executing_status: Any, failed_status: Any) -> Dict[str, Any]:
        goal_manager = self._get_goal_manager()
        execution_contexts = self._get_execution_contexts()
        active_schedules = self._get_active_schedules()

        active_goals = len([goal for goal in goal_manager.goals.values() if goal.status == GoalStatus.ACTIVE])
        executing_contexts = len([context for context in execution_contexts.values() if context.status == executing_status])
        failed_contexts = len([context for context in execution_contexts.values() if context.status == failed_status])

        return {
            "status": "healthy" if failed_contexts == 0 else "degraded",
            "active_goals": active_goals,
            "executing_workflows": executing_contexts,
            "failed_workflows": failed_contexts,
            "total_contexts": len(execution_contexts),
            "active_schedules": len(active_schedules),
            "components": {
                "goal_manager": "operational",
                "decision_engine": "operational",
                "goap_planner": "operational",
                "scheduler": "operational",
            },
        }

    def clear_execution_history(self, goal_id: str | None = None) -> None:
        execution_contexts = self._get_execution_contexts()
        active_schedules = self._get_active_schedules()
        if goal_id:
            execution_contexts.pop(goal_id, None)
            active_schedules.pop(goal_id, None)
            return

        execution_contexts.clear()
        active_schedules.clear()

    def get_learning_metrics(self) -> Dict[str, Any]:
        outcome_tracker = self._get_outcome_tracker()
        return {
            "decision_accuracy": outcome_tracker.analyze_decision_accuracy(),
            "planning_accuracy": outcome_tracker.analyze_planning_accuracy(),
            "scheduling_accuracy": outcome_tracker.analyze_scheduling_accuracy(),
            "improvement_trends": outcome_tracker.get_improvement_trends(),
        }