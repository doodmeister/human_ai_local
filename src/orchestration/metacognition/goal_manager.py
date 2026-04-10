from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from dataclasses import replace

from .enums import GoalKind, InputType
from .models import CognitiveGoal, WorkspaceState


class HeuristicGoalManager:
    """Deterministically rank active and synthesized cognitive goals."""

    def __init__(
        self,
        *,
        priority_weight: float = 0.45,
        urgency_weight: float = 0.20,
        salience_weight: float = 0.20,
        contradiction_weight: float = 0.25,
        uncertainty_weight: float = 0.20,
        cognitive_load_weight: float = 0.15,
        planning_weight: float = 0.20,
        response_bias: float = 0.15,
        min_score: float = 0.0,
    ) -> None:
        self._priority_weight = priority_weight
        self._urgency_weight = urgency_weight
        self._salience_weight = salience_weight
        self._contradiction_weight = contradiction_weight
        self._uncertainty_weight = uncertainty_weight
        self._cognitive_load_weight = cognitive_load_weight
        self._planning_weight = planning_weight
        self._response_bias = response_bias
        self._min_score = min_score
        self._background_goal_counters: dict[str, int] = defaultdict(int)

    def rank_goals(self, workspace: WorkspaceState) -> list[CognitiveGoal]:
        candidates = list(workspace.snapshot.active_goals or ())
        candidates.extend(self._synthesized_goals(workspace, candidates))

        ranked: list[CognitiveGoal] = []
        contradiction_pressure = min(1.0, len(workspace.contradictions or ()) / 5.0)
        uncertainty = self._bounded_float(getattr(workspace.snapshot, "uncertainty", 0.0))
        cognitive_load = self._bounded_float(getattr(workspace.snapshot, "cognitive_load", 0.0))

        for goal in candidates:
            try:
                ranked_goal = self._scored_goal(
                    goal,
                    workspace=workspace,
                    contradiction_pressure=contradiction_pressure,
                    uncertainty=uncertainty,
                    cognitive_load=cognitive_load,
                )
            except Exception:
                continue
            if ranked_goal is not None:
                ranked.append(ranked_goal)

        ranked.sort(
            key=lambda goal: (
                -float(goal.metadata.get("heuristic_score_raw", goal.metadata.get("heuristic_score", 0.0))),
                -self._bounded_float(getattr(goal, "priority", 0.0)),
                -self._bounded_float(getattr(goal, "salience", 0.0)),
                goal.kind.value,
                goal.goal_id,
            )
        )
        return ranked

    def _synthesized_goals(
        self,
        workspace: WorkspaceState,
        existing_goals: list[CognitiveGoal],
    ) -> list[CognitiveGoal]:
        kinds = {
            goal_kind
            for goal in existing_goals
            if (goal_kind := self._goal_kind(getattr(goal, "kind", None))) is not None
        }
        synthesized: list[CognitiveGoal] = []
        if GoalKind.RESPONSE not in kinds and self._has_actionable_user_input(workspace):
            synthesized.append(
                CognitiveGoal(
                    goal_id=self._synthesized_goal_id("response", workspace),
                    description="Respond to the current user input",
                    priority=0.60,
                    kind=GoalKind.RESPONSE,
                    urgency=0.70,
                    salience=0.70,
                    rationale="synthetic_response_goal",
                    metadata={"source": "heuristic"},
                )
            )
        if workspace.contradictions and GoalKind.REFLECTION not in kinds:
            synthesized.append(
                CognitiveGoal(
                    goal_id=self._synthesized_goal_id("reflect", workspace),
                    description="Inspect contradictory signals before answering",
                    priority=0.75,
                    kind=GoalKind.REFLECTION,
                    urgency=min(1.0, 0.60 + 0.10 * len(workspace.contradictions)),
                    salience=0.85,
                    rationale="contradiction_pressure",
                    metadata={"source": "heuristic", "contradiction_count": len(workspace.contradictions)},
                )
            )
        if workspace.snapshot.uncertainty >= 0.45 and GoalKind.MEMORY not in kinds:
            synthesized.append(
                CognitiveGoal(
                    goal_id=self._synthesized_goal_id("memory", workspace),
                    description="Retrieve supporting memory context",
                    priority=0.45,
                    kind=GoalKind.MEMORY,
                    urgency=self._bounded_float(workspace.snapshot.uncertainty),
                    salience=0.55,
                    rationale="high_uncertainty",
                    metadata={"source": "heuristic"},
                )
            )
        if workspace.snapshot.cognitive_load >= 0.75 and GoalKind.ATTENTION not in kinds:
            synthesized.append(
                CognitiveGoal(
                    goal_id=self._synthesized_goal_id("attention", workspace),
                    description="Reduce cognitive load before continuing",
                    priority=0.58,
                    kind=GoalKind.ATTENTION,
                    urgency=workspace.snapshot.cognitive_load,
                    salience=0.60,
                    rationale="high_cognitive_load",
                    metadata={"source": "heuristic"},
                )
            )
        return synthesized

    def _response_bonus(self, goal_kind: GoalKind, workspace: WorkspaceState) -> float:
        if goal_kind == GoalKind.RESPONSE and self._has_actionable_user_input(workspace):
            return self._response_bias
        return 0.0

    def _contradiction_bonus(self, goal_kind: GoalKind, contradiction_pressure: float) -> float:
        if goal_kind == GoalKind.REFLECTION:
            return contradiction_pressure * self._contradiction_weight
        return 0.0

    def _uncertainty_bonus(self, goal_kind: GoalKind, uncertainty: float) -> float:
        if goal_kind == GoalKind.MEMORY:
            return uncertainty * self._uncertainty_weight
        return 0.0

    def _cognitive_load_bonus(self, goal_kind: GoalKind, cognitive_load: float) -> float:
        if goal_kind == GoalKind.ATTENTION:
            return cognitive_load * self._cognitive_load_weight
        return 0.0

    def _planning_bonus(self, goal_kind: GoalKind, priority: float) -> float:
        if goal_kind == GoalKind.PLANNING:
            return self._planning_weight * priority
        return 0.0

    def _scored_goal(
        self,
        goal: CognitiveGoal,
        *,
        workspace: WorkspaceState,
        contradiction_pressure: float,
        uncertainty: float,
        cognitive_load: float,
    ) -> CognitiveGoal | None:
        goal_kind = self._goal_kind(getattr(goal, "kind", None))
        priority = self._goal_metric(goal.priority)
        urgency = self._goal_metric(goal.urgency)
        salience = self._goal_metric(goal.salience)
        score_breakdown = {
            "priority": round(self._priority_weight * priority, 6),
            "urgency": round(self._urgency_weight * urgency, 6),
            "salience": round(self._salience_weight * salience, 6),
            "response_bias": round(self._response_bonus(goal_kind, workspace), 6),
            "contradiction_pressure": round(self._contradiction_bonus(goal_kind, contradiction_pressure), 6),
            "uncertainty": round(self._uncertainty_bonus(goal_kind, uncertainty), 6),
            "cognitive_load": round(self._cognitive_load_bonus(goal_kind, cognitive_load), 6),
            "planning_bonus": round(self._planning_bonus(goal_kind, priority), 6),
        }
        raw_score = sum(score_breakdown.values())
        heuristic_score = max(0.0, min(1.0, raw_score))
        if heuristic_score < self._min_score:
            return None
        metadata = dict(goal.metadata) if isinstance(goal.metadata, Mapping) else {}
        metadata["heuristic_score"] = round(heuristic_score, 6)
        metadata["heuristic_score_raw"] = round(raw_score, 6)
        metadata["score_breakdown"] = score_breakdown
        return replace(goal, kind=goal_kind, metadata=metadata)

    @staticmethod
    def _goal_kind(value: Any) -> GoalKind:
        if isinstance(value, GoalKind):
            return value
        try:
            return GoalKind(str(value))
        except ValueError:
            return GoalKind.PLANNING

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _goal_metric(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid goal metric: {value!r}") from exc

    @staticmethod
    def _has_actionable_user_input(workspace: WorkspaceState) -> bool:
        if getattr(workspace.stimulus, "input_type", InputType.TEXT) == InputType.BACKGROUND_TASK:
            return False
        return bool(str(getattr(workspace.stimulus, "user_input", "")).strip())

    def _synthesized_goal_id(self, prefix: str, workspace: WorkspaceState) -> str:
        base = f"{prefix}:{workspace.stimulus.session_id}:{workspace.snapshot.turn_index}"
        if getattr(workspace.stimulus, "input_type", InputType.TEXT) == InputType.BACKGROUND_TASK:
            session_id = str(workspace.stimulus.session_id)
            self._background_goal_counters[session_id] += 1
            return f"{base}:{self._background_goal_counters[session_id]}"
        return base
