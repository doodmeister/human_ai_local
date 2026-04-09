from __future__ import annotations

from .models import CognitivePlan, CriticReport, ExecutionResult, WorkspaceState


class DefaultCritic:
    """Heuristically evaluate the outcome of a metacognitive cycle."""

    def evaluate(
        self,
        workspace: WorkspaceState,
        plan: CognitivePlan,
        result: ExecutionResult,
    ) -> CriticReport:
        cycle_id = f"{workspace.stimulus.session_id}:{workspace.snapshot.turn_index}"
        response_quality = 0.35 if result.response_text.strip() else 0.0
        execution_quality = 0.30 if result.success else 0.10
        plan_quality = min(0.20, 0.05 * len(plan.acts))
        contradiction_penalty = min(0.15, 0.05 * len(workspace.contradictions))
        success_score = max(0.0, min(1.0, response_quality + execution_quality + plan_quality - contradiction_penalty))

        if plan.selected_goal is None:
            goal_progress = 0.0
        elif result.response_text.strip() or result.memory_updates or result.attention_updates:
            goal_progress = 1.0 if result.success else 0.5
        else:
            goal_progress = 0.0

        contradictions = tuple(
            str(item.get("kind") or item.get("description") or item)
            for item in workspace.contradictions
        )
        follow_up = bool(contradictions) or workspace.snapshot.uncertainty >= 0.5 or not result.success
        rationale = (
            f"acts={len(result.executed_acts)}; response_len={len(result.response_text)}; "
            f"contradictions={len(contradictions)}"
        )
        return CriticReport(
            cycle_id=cycle_id,
            success_score=round(success_score, 6),
            goal_progress=goal_progress,
            follow_up_recommended=follow_up,
            contradictions_detected=contradictions,
            rationale=rationale,
            metadata={
                "executed_acts": [act.act_type.value for act in result.executed_acts],
                "memory_updates": len(result.memory_updates),
                "attention_updated": bool(result.attention_updates),
            },
        )
