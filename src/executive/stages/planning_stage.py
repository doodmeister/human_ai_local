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
        *,
        procedural_matches: Optional[list[dict[str, Any]]] = None,
        planning_metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Plan]:
        goal_state = self.goal_to_world_state(goal)
        if initial_state is None:
            initial_state = WorldState()

        preferred_action_names, planning_procedure_id = self._select_preferred_actions(procedural_matches)
        plan_context = {
            "goal_id": goal.id,
            "decision": decision,
            "planning_procedure_id": planning_procedure_id,
        }
        if procedural_matches:
            plan_context["procedural_match_count"] = len(procedural_matches)

        plan = self._goap_planner.plan(
            initial_state=initial_state,
            goal_state=goal_state,
            max_iterations=self._config.planning_max_iterations,
            plan_context=plan_context,
            preferred_action_names=preferred_action_names,
        )

        if planning_metadata is not None:
            for key in [
                "planning_source",
                "planning_procedure_id",
                "preferred_action_names",
                "resolved_preferred_actions",
                "procedural_replay_attempted",
                "procedural_match_count",
            ]:
                if key in plan_context:
                    planning_metadata[key] = plan_context[key]

        if plan:
            self._metrics.inc("executive_plans_created_total")
            self._metrics.observe("executive_plan_length", len(plan.steps))
        else:
            self._metrics.inc("executive_planning_failures_total")

        return plan

    def _select_preferred_actions(
        self,
        procedural_matches: Optional[list[dict[str, Any]]],
    ) -> tuple[list[str], str | None]:
        for procedure in procedural_matches or []:
            steps = [str(step).strip() for step in procedure.get("steps", []) or [] if str(step).strip()]
            if steps:
                return steps, str(procedure.get("id", "") or "") or None
        return [], None

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