from __future__ import annotations

import math

from .enums import CognitiveActType, GoalKind, PolicyName
from .models import CognitiveActProposal, CognitiveGoal, CognitivePlan, WorkspaceState
from .thresholds import DEFAULT_COGNITIVE_THRESHOLDS


class HeuristicPolicyEngine:
    """Produce deterministic cognitive act proposals and a simple selected plan.

    The default cognitive-load and uncertainty thresholds are shared with the
    workspace builder and scheduler through DEFAULT_COGNITIVE_THRESHOLDS.
    """

    # Contradiction pressure saturates after roughly two concurrent conflicts.
    _CONTRADICTION_SATURATION = 2.0
    # Sparse-context retrieval still gets a meaningful score even when uncertainty is low.
    _UNCERTAINTY_FLOOR = 0.35
    # Store-memory urgency tops out once a few context items were available this turn.
    _CONTEXT_COUNT_SATURATION = 4.0
    # Goal-kind preferences only influence template selection, not proposal scoring.
    _GOAL_KIND_PREFERRED_ACT: dict[GoalKind, CognitiveActType] = {
        GoalKind.ATTENTION: CognitiveActType.REFOCUS_ATTENTION,
    }
    _SCORE_TABLE: dict[CognitiveActType, tuple[float, float]] = {
        CognitiveActType.INSPECT_CONFLICT: (0.50, 0.40),
        CognitiveActType.RETRIEVE_CONTEXT: (0.40, 0.35),
        CognitiveActType.REFOCUS_ATTENTION: (0.35, 0.35),
        CognitiveActType.DEFER: (0.30, 0.40),
        CognitiveActType.RESPOND: (0.55, 0.10),
        CognitiveActType.STORE_MEMORY: (0.25, 0.10),
    }

    def __init__(
        self,
        *,
        uncertainty_threshold: float = DEFAULT_COGNITIVE_THRESHOLDS.uncertainty_threshold,
        cognitive_load_threshold: float = DEFAULT_COGNITIVE_THRESHOLDS.cognitive_load_threshold,
        defer_threshold: float = 0.90,
        max_acts: int = 3,
    ) -> None:
        for name, value in (
            ("uncertainty_threshold", uncertainty_threshold),
            ("cognitive_load_threshold", cognitive_load_threshold),
            ("defer_threshold", defer_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
        if defer_threshold < cognitive_load_threshold:
            raise ValueError(
                f"defer_threshold ({defer_threshold}) must be >= "
                f"cognitive_load_threshold ({cognitive_load_threshold})"
            )
        if not isinstance(max_acts, int) or isinstance(max_acts, bool) or max_acts < 1:
            raise ValueError(f"max_acts must be an integer >= 1, got {max_acts!r}")
        self._uncertainty_threshold = uncertainty_threshold
        self._cognitive_load_threshold = cognitive_load_threshold
        self._defer_threshold = defer_threshold
        self._max_acts = max_acts

    def propose_acts(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
    ) -> list[CognitiveActProposal]:
        """Emit at most one proposal per act type, sorted by descending score.

        Only the top-ranked goal influences act scoring; secondary goals are
        ignored by design.
        """
        contradiction_pressure = min(1.0, len(workspace.contradictions) / self._CONTRADICTION_SATURATION)
        uncertainty = self._clamp_unit_interval(workspace.snapshot.uncertainty)
        cognitive_load = self._clamp_unit_interval(workspace.snapshot.cognitive_load)
        dominant_goal = ranked_goals[0] if ranked_goals else workspace.dominant_goal

        proposals: list[CognitiveActProposal] = []
        if contradiction_pressure > 0.0:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.INSPECT_CONFLICT,
                    description="Inspect contradictory context before responding",
                    priority_score=self._score(CognitiveActType.INSPECT_CONFLICT, contradiction_pressure),
                    rationale="contradiction_pressure",
                    metadata={"contradiction_count": len(workspace.contradictions)},
                )
            )
        if uncertainty >= self._uncertainty_threshold or not workspace.context_items:
            effective_uncertainty = max(uncertainty, self._UNCERTAINTY_FLOOR)
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.RETRIEVE_CONTEXT,
                    description="Retrieve additional context to reduce uncertainty",
                    priority_score=self._score(CognitiveActType.RETRIEVE_CONTEXT, effective_uncertainty),
                    rationale="uncertainty_or_sparse_context",
                    metadata={"uncertainty": uncertainty, "context_count": len(workspace.context_items)},
                )
            )
        if cognitive_load >= self._cognitive_load_threshold:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.REFOCUS_ATTENTION,
                    description="Refocus attention before continuing",
                    priority_score=self._score(CognitiveActType.REFOCUS_ATTENTION, cognitive_load),
                    rationale="high_cognitive_load",
                    metadata={"cognitive_load": cognitive_load},
                )
            )
        if cognitive_load >= self._defer_threshold:
            proposals.append(
                CognitiveActProposal(
                    act_type=CognitiveActType.DEFER,
                    description="Defer deeper work until load is reduced",
                    priority_score=self._score(CognitiveActType.DEFER, cognitive_load),
                    rationale="extreme_cognitive_load",
                    metadata={"cognitive_load": cognitive_load},
                )
            )
        if workspace.stimulus.user_input.strip():
            response_bonus = 0.10 if dominant_goal and dominant_goal.kind is GoalKind.RESPONSE else 0.0
            response_proposal = CognitiveActProposal(
                act_type=CognitiveActType.RESPOND,
                description="Respond to the user",
                priority_score=self._score(CognitiveActType.RESPOND, response_bonus / 0.10 if response_bonus else 0.0),
                rationale="user_turn_requires_response",
            )
            response_proposal.target_goal_id = dominant_goal.goal_id if dominant_goal else None
            proposals.append(response_proposal)
        proposals.append(
            CognitiveActProposal(
                act_type=CognitiveActType.STORE_MEMORY,
                description="Capture useful state from this turn",
                priority_score=self._score(
                    CognitiveActType.STORE_MEMORY,
                    min(1.0, len(workspace.context_items) / self._CONTEXT_COUNT_SATURATION),
                ),
                rationale="turn_state_capture",
            )
        )

        seen_types: set[CognitiveActType] = set()
        for proposal in proposals:
            if proposal.act_type in seen_types:
                raise AssertionError("propose_acts must produce at most one proposal per act type")
            seen_types.add(proposal.act_type)

        return sorted(
            proposals,
            key=lambda proposal: (-proposal.priority_score, proposal.act_type.value),
        )

    def select_plan(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
        proposals: list[CognitiveActProposal],
    ) -> CognitivePlan:
        """Select a fixed-template plan from the proposal set.

        This policy always prefers a fixed act sequence over proposal-score
        ordering: inspect conflict, retrieve context, goal-kind preference or
        defer/refocus, then respond or store memory. Remaining proposals backfill
        slots up to max_acts.
        """
        selected_goal = ranked_goals[0] if ranked_goals else workspace.dominant_goal
        proposal_by_type = {proposal.act_type: proposal for proposal in proposals}
        selected_acts: list[CognitiveActProposal] = []
        selected_act_types: set[CognitiveActType] = set()

        def _append_if_available(act_type: CognitiveActType) -> None:
            if len(selected_acts) >= self._max_acts or act_type in selected_act_types:
                return
            proposal = proposal_by_type.get(act_type)
            if proposal is None:
                return
            selected_acts.append(proposal)
            selected_act_types.add(act_type)

        _append_if_available(CognitiveActType.INSPECT_CONFLICT)
        _append_if_available(CognitiveActType.RETRIEVE_CONTEXT)

        preferred_act = self._GOAL_KIND_PREFERRED_ACT.get(selected_goal.kind) if selected_goal else None
        if preferred_act is not None:
            _append_if_available(preferred_act)
        elif CognitiveActType.DEFER in proposal_by_type:
            _append_if_available(CognitiveActType.DEFER)
        elif CognitiveActType.REFOCUS_ATTENTION in proposal_by_type:
            _append_if_available(CognitiveActType.REFOCUS_ATTENTION)

        if CognitiveActType.RESPOND in proposal_by_type:
            _append_if_available(CognitiveActType.RESPOND)
        elif CognitiveActType.STORE_MEMORY in proposal_by_type:
            _append_if_available(CognitiveActType.STORE_MEMORY)

        if len(selected_acts) < self._max_acts:
            for proposal in proposals:
                if proposal.act_type in selected_act_types:
                    continue
                selected_acts.append(proposal)
                selected_act_types.add(proposal.act_type)
                if len(selected_acts) >= self._max_acts:
                    break

        acts_tuple = tuple(selected_acts[: self._max_acts])

        rationale_parts = [
            f"goal={selected_goal.goal_id}" if selected_goal else "goal=none",
            f"acts={','.join(proposal.act_type.value for proposal in acts_tuple)}",
        ]
        return CognitivePlan(
            selected_goal=selected_goal,
            acts=acts_tuple,
            policy_name=PolicyName.HEURISTIC_V1,
            rationale="; ".join(rationale_parts),
            metadata={
                "contradiction_count": len(workspace.contradictions),
                "uncertainty": workspace.snapshot.uncertainty,
                "cognitive_load": workspace.snapshot.cognitive_load,
                "selection_strategy": "fixed_template",
                "degenerate_plan": len(acts_tuple) <= 1,
            },
        )

    @staticmethod
    def _clamp_unit_interval(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def _score(self, act_type: CognitiveActType, signal: float) -> float:
        base, slope = self._SCORE_TABLE[act_type]
        priority_score = base + slope * self._clamp_unit_interval(signal)
        return self._clamp_unit_interval(priority_score)
