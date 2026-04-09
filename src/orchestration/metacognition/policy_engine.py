from __future__ import annotations

from .enums import CognitiveActType, GoalKind
from .models import CognitiveActProposal, CognitiveGoal, CognitivePlan, WorkspaceState


class HeuristicPolicyEngine:
    """Produce deterministic cognitive act proposals and a simple selected plan."""

    def __init__(
        self,
        *,
        uncertainty_threshold: float = 0.45,
        cognitive_load_threshold: float = 0.75,
        defer_threshold: float = 0.90,
        max_acts: int = 3,
    ) -> None:
        self._uncertainty_threshold = uncertainty_threshold
        self._cognitive_load_threshold = cognitive_load_threshold
        self._defer_threshold = defer_threshold
        self._max_acts = max_acts

    def propose_acts(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
    ) -> list[CognitiveActProposal]:
        contradiction_pressure = min(1.0, len(workspace.contradictions) / 2.0)
        uncertainty = max(0.0, min(1.0, workspace.snapshot.uncertainty))
        cognitive_load = max(0.0, min(1.0, workspace.snapshot.cognitive_load))
        dominant_goal = ranked_goals[0] if ranked_goals else workspace.dominant_goal

        proposals: list[CognitiveActProposal] = []
        if contradiction_pressure > 0.0:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.INSPECT_CONFLICT,
                    description="Inspect contradictory context before responding",
                    priority_score=0.50 + 0.40 * contradiction_pressure,
                    rationale="contradiction_pressure",
                    metadata={"contradiction_count": len(workspace.contradictions)},
                )
            )
        if uncertainty >= self._uncertainty_threshold or not workspace.context_items:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.RETRIEVE_CONTEXT,
                    description="Retrieve additional context to reduce uncertainty",
                    priority_score=0.40 + 0.35 * max(uncertainty, 0.35),
                    rationale="uncertainty_or_sparse_context",
                    metadata={"uncertainty": uncertainty, "context_count": len(workspace.context_items)},
                )
            )
        if cognitive_load >= self._cognitive_load_threshold:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.REFOCUS_ATTENTION,
                    description="Refocus attention before continuing",
                    priority_score=0.35 + 0.35 * cognitive_load,
                    rationale="high_cognitive_load",
                    metadata={"cognitive_load": cognitive_load},
                )
            )
        if cognitive_load >= self._defer_threshold:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.DEFER,
                    description="Defer deeper work until load is reduced",
                    priority_score=0.30 + 0.40 * cognitive_load,
                    rationale="extreme_cognitive_load",
                    metadata={"cognitive_load": cognitive_load},
                )
            )
        if workspace.stimulus.user_input.strip():
            response_bonus = 0.10 if dominant_goal and dominant_goal.kind is GoalKind.RESPONSE else 0.0
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.RESPOND,
                    description="Respond to the user",
                    priority_score=0.55 + response_bonus,
                    rationale="user_turn_requires_response",
                    target=dominant_goal.goal_id if dominant_goal else None,
                )
            )
        proposals.append(
            CognitiveActProposal(
                act_type=CognitiveActType.STORE_MEMORY,
                description="Capture useful state from this turn",
                priority_score=0.25 + 0.10 * min(1.0, len(workspace.context_items) / 4.0),
                rationale="turn_state_capture",
            )
        )

        deduped: dict[CognitiveActType, CognitiveActProposal] = {}
        for proposal in proposals:
            incumbent = deduped.get(proposal.act_type)
            if incumbent is None or proposal.priority_score > incumbent.priority_score:
                deduped[proposal.act_type] = proposal

        ranked = sorted(
            deduped.values(),
            key=lambda proposal: (-proposal.priority_score, proposal.act_type.value),
        )
        return ranked

    def select_plan(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
        proposals: list[CognitiveActProposal],
    ) -> CognitivePlan:
        selected_goal = ranked_goals[0] if ranked_goals else workspace.dominant_goal
        proposal_by_type = {proposal.act_type: proposal for proposal in proposals}
        selected_acts: list[CognitiveActProposal] = []

        if CognitiveActType.INSPECT_CONFLICT in proposal_by_type:
            selected_acts.append(proposal_by_type[CognitiveActType.INSPECT_CONFLICT])
        if CognitiveActType.RETRIEVE_CONTEXT in proposal_by_type:
            selected_acts.append(proposal_by_type[CognitiveActType.RETRIEVE_CONTEXT])

        if selected_goal and selected_goal.kind is GoalKind.ATTENTION:
            if CognitiveActType.REFOCUS_ATTENTION in proposal_by_type:
                selected_acts.append(proposal_by_type[CognitiveActType.REFOCUS_ATTENTION])
        elif CognitiveActType.DEFER in proposal_by_type:
            selected_acts.append(proposal_by_type[CognitiveActType.DEFER])
        elif CognitiveActType.REFOCUS_ATTENTION in proposal_by_type:
            selected_acts.append(proposal_by_type[CognitiveActType.REFOCUS_ATTENTION])

        if CognitiveActType.RESPOND in proposal_by_type:
            selected_acts.append(proposal_by_type[CognitiveActType.RESPOND])
        elif CognitiveActType.STORE_MEMORY in proposal_by_type:
            selected_acts.append(proposal_by_type[CognitiveActType.STORE_MEMORY])

        if len(selected_acts) < self._max_acts:
            for proposal in proposals:
                if proposal in selected_acts:
                    continue
                selected_acts.append(proposal)
                if len(selected_acts) >= self._max_acts:
                    break

        unique_acts: list[CognitiveActProposal] = []
        seen_types: set[CognitiveActType] = set()
        for proposal in selected_acts:
            if proposal.act_type in seen_types:
                continue
            unique_acts.append(proposal)
            seen_types.add(proposal.act_type)
            if len(unique_acts) >= self._max_acts:
                break

        rationale_parts = [
            f"goal={selected_goal.goal_id}" if selected_goal else "goal=none",
            f"acts={','.join(proposal.act_type.value for proposal in unique_acts)}",
        ]
        return CognitivePlan(
            selected_goal=selected_goal,
            acts=tuple(unique_acts),
            policy_name="heuristic_policy_v1",
            rationale="; ".join(rationale_parts),
            metadata={
                "contradiction_count": len(workspace.contradictions),
                "uncertainty": workspace.snapshot.uncertainty,
                "cognitive_load": workspace.snapshot.cognitive_load,
            },
        )
