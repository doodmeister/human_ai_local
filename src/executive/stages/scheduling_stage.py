from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any, Optional

from src.executive.goal_manager import Goal, GoalPriority
from src.executive.planning.goap_planner import Plan
from src.executive.scheduling import (
    OptimizationObjective,
    Resource,
    ResourceType,
    Schedule,
    SchedulingProblem,
    Task,
)


logger = logging.getLogger(__name__)


class ExecutiveSchedulingStage:
    def __init__(self, *, config: Any, scheduler: Any, metrics: Any) -> None:
        self._config = config
        self._scheduler = scheduler
        self._metrics = metrics

    def create_schedule_from_plan(self, plan: Plan, goal: Goal) -> Optional[Schedule]:
        tasks = []
        for index, step in enumerate(plan.steps):
            action = step.action
            task = Task(
                id=f"action_{index}_{action.name}",
                name=action.name,
                duration=timedelta(hours=1),
                priority=self.goal_priority_to_float(goal.priority),
                cognitive_load=0.5,
                dependencies=set([f"action_{index-1}_{plan.steps[index-1].action.name}"] if index > 0 else []),
            )
            tasks.append(task)

        resources = [
            Resource(
                id="cognitive",
                name="Cognitive Capacity",
                type=ResourceType.COGNITIVE,
                capacity=1.0,
            )
        ]

        problem = SchedulingProblem(
            tasks=tasks,
            resources=resources,
            objectives=[
                OptimizationObjective(
                    name="minimize_makespan",
                    description="Minimize total schedule duration",
                    weight=1.0,
                )
            ],
            horizon=timedelta(hours=self._config.scheduling_horizon_hours),
        )

        try:
            schedule = self._scheduler.create_initial_schedule(problem)
            self._metrics.inc("executive_schedules_created_total")
            return schedule
        except Exception as exc:
            logger.error("Scheduling failed: %s", exc)
            self._metrics.inc("executive_scheduling_failures_total")
            return None

    def goal_priority_to_float(self, priority: GoalPriority) -> float:
        mapping = {
            GoalPriority.LOW: 0.3,
            GoalPriority.MEDIUM: 0.6,
            GoalPriority.HIGH: 0.9,
            GoalPriority.CRITICAL: 1.0,
        }
        return mapping.get(priority, 0.5)
