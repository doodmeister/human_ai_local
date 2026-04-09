from __future__ import annotations

from dataclasses import replace

from .enums import GoalKind
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
        response_bias: float = 0.15,
    ) -> None:
        self._priority_weight = priority_weight
        self._urgency_weight = urgency_weight
        self._salience_weight = salience_weight
        self._contradiction_weight = contradiction_weight
        self._uncertainty_weight = uncertainty_weight
        self._cognitive_load_weight = cognitive_load_weight
        self._response_bias = response_bias

    def rank_goals(self, workspace: WorkspaceState) -> list[CognitiveGoal]:
        candidates = list(workspace.snapshot.active_goals)
        candidates.extend(self._synthesized_goals(workspace, candidates))

        ranked: list[CognitiveGoal] = []
        contradiction_pressure = min(1.0, len(workspace.contradictions) / 2.0)
        uncertainty = max(0.0, min(1.0, workspace.snapshot.uncertainty))
        cognitive_load = max(0.0, min(1.0, workspace.snapshot.cognitive_load))

        for goal in candidates:
            score_breakdown = {
                "priority": self._priority_weight * max(0.0, min(1.0, goal.priority)),
                "urgency": self._urgency_weight * max(0.0, min(1.0, goal.urgency)),
                "salience": self._salience_weight * max(0.0, min(1.0, goal.salience)),
                "response_bias": self._response_bonus(goal, workspace),
                "contradiction_pressure": self._contradiction_bonus(goal, contradiction_pressure),
                "uncertainty": self._uncertainty_bonus(goal, uncertainty),
                "cognitive_load": self._cognitive_load_bonus(goal, cognitive_load),
            }
            score = sum(score_breakdown.values())
            metadata = dict(goal.metadata)
            metadata["heuristic_score"] = round(score, 6)
            metadata["score_breakdown"] = score_breakdown
            ranked.append(replace(goal, metadata=metadata))

        ranked.sort(
            key=lambda goal: (
                -float(goal.metadata.get("heuristic_score", 0.0)),
                -goal.priority,
                -goal.salience,
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
        kinds = {goal.kind for goal in existing_goals}
        synthesized: list[CognitiveGoal] = []
        if GoalKind.RESPONSE not in kinds:
            synthesized.append(
                CognitiveGoal(
                    goal_id=f"response:{workspace.stimulus.session_id}:{workspace.snapshot.turn_index}",
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
                    goal_id=f"reflect:{workspace.stimulus.session_id}:{workspace.snapshot.turn_index}",
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
                    goal_id=f"memory:{workspace.stimulus.session_id}:{workspace.snapshot.turn_index}",
                    description="Retrieve supporting memory context",
                    priority=0.45,
                    kind=GoalKind.MEMORY,
                    urgency=min(0.50, workspace.snapshot.uncertainty),
                    salience=0.55,
                    rationale="high_uncertainty",
                    metadata={"source": "heuristic"},
                )
            )
        if workspace.snapshot.cognitive_load >= 0.75 and GoalKind.ATTENTION not in kinds:
            synthesized.append(
                CognitiveGoal(
                    goal_id=f"attention:{workspace.stimulus.session_id}:{workspace.snapshot.turn_index}",
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

    def _response_bonus(self, goal: CognitiveGoal, workspace: WorkspaceState) -> float:
        if goal.kind is GoalKind.RESPONSE and bool(workspace.stimulus.user_input.strip()):
            return self._response_bias
        return 0.0

    def _contradiction_bonus(self, goal: CognitiveGoal, contradiction_pressure: float) -> float:
        if goal.kind is GoalKind.REFLECTION:
            return contradiction_pressure * self._contradiction_weight
        return 0.0

    def _uncertainty_bonus(self, goal: CognitiveGoal, uncertainty: float) -> float:
        if goal.kind is GoalKind.MEMORY:
            return uncertainty * self._uncertainty_weight
        return 0.0

    def _cognitive_load_bonus(self, goal: CognitiveGoal, cognitive_load: float) -> float:
        if goal.kind is GoalKind.ATTENTION:
            return cognitive_load * self._cognitive_load_weight
        return 0.0
