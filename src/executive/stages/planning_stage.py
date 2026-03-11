from __future__ import annotations

from typing import Any, Optional

from src.executive.goal_manager import Goal
from src.executive.decision_engine import DecisionResult
from src.executive.planning.goap_planner import Plan, WorldState


class ExecutivePlanningStage:
    def __init__(self, *, config: Any, goap_planner: Any, metrics: Any) -> None:
        self._config = config
        self._goap_planner = goap_planner
        self._metrics = metrics

    def create_goal_plan(
        self,
        goal: Goal,
        decision: DecisionResult,
        initial_state: Optional[WorldState] = None,
    ) -> Optional[Plan]:
        goal_state = self.goal_to_world_state(goal)
        if initial_state is None:
            initial_state = WorldState()

        plan = self._goap_planner.plan(
            initial_state=initial_state,
            goal_state=goal_state,
            max_iterations=self._config.planning_max_iterations,
            plan_context={"goal_id": goal.id, "decision": decision},
        )

        if plan:
            self._metrics.inc("executive_plans_created_total")
            self._metrics.observe("executive_plan_length", len(plan.steps))
        else:
            self._metrics.inc("executive_planning_failures_total")

        return plan

    def goal_to_world_state(self, goal: Goal) -> WorldState:
        goal_conditions = {}
        if goal.success_criteria:
            for criterion in goal.success_criteria:
                if "=" in criterion:
                    var, val = criterion.split("=", 1)
                    var = var.strip()
                    val = val.strip()

                    if val.lower() == "true":
                        goal_conditions[var] = True
                    elif val.lower() == "false":
                        goal_conditions[var] = False
                    elif val.isdigit():
                        goal_conditions[var] = int(val)
                    else:
                        try:
                            goal_conditions[var] = float(val)
                        except ValueError:
                            goal_conditions[var] = val
                else:
                    goal_conditions[criterion.strip()] = True
        else:
            goal_conditions[f"goal_{goal.id}_completed"] = True

        return WorldState(goal_conditions)