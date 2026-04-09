from __future__ import annotations

from typing import Any, Callable

from .enums import CognitiveActType
from .models import CognitiveActProposal, CognitivePlan, ExecutionResult, WorkspaceState


ActDispatcher = Callable[[CognitiveActProposal, WorkspaceState], dict[str, Any]]
ResponseBuilder = Callable[[CognitivePlan, WorkspaceState, list[dict[str, Any]]], str]


class DefaultPlanExecutor:
    """Dispatch plan acts through injected handlers and return an execution summary."""

    def __init__(
        self,
        *,
        dispatch_act: ActDispatcher | None = None,
        build_response_text: ResponseBuilder | None = None,
    ) -> None:
        self._dispatch_act = dispatch_act or self._default_dispatch
        self._build_response_text = build_response_text or self._default_response_text

    def execute(self, workspace: WorkspaceState, plan: CognitivePlan) -> ExecutionResult:
        executed_acts: list[CognitiveActProposal] = []
        memory_updates: list[str] = []
        attention_updates: dict[str, Any] = {}
        dispatch_log: list[dict[str, Any]] = []

        for act in plan.acts:
            dispatch_result = self._dispatch_act(act, workspace) or {}
            executed_acts.append(act)
            dispatch_log.append({"act_type": act.act_type.value, **dispatch_result})
            if dispatch_result.get("memory_updated"):
                memory_updates.append(act.description)
            if isinstance(dispatch_result.get("attention"), dict):
                attention_updates.update(dispatch_result["attention"])

        response_text = self._build_response_text(plan, workspace, dispatch_log).strip()
        if not response_text and workspace.stimulus.user_input.strip():
            response_text = self._fallback_response(workspace, plan)

        success = bool(executed_acts) and bool(response_text or memory_updates or attention_updates)
        return ExecutionResult(
            success=success,
            response_text=response_text,
            executed_acts=tuple(executed_acts),
            memory_updates=tuple(memory_updates),
            attention_updates=attention_updates,
            metadata={"dispatch_log": dispatch_log},
        )

    @staticmethod
    def _default_dispatch(act: CognitiveActProposal, workspace: WorkspaceState) -> dict[str, Any]:
        if act.act_type is CognitiveActType.RETRIEVE_CONTEXT:
            return {"memory_updated": False, "retrieved": len(workspace.context_items)}
        if act.act_type is CognitiveActType.STORE_MEMORY:
            return {"memory_updated": True, "stored": workspace.stimulus.user_input[:80]}
        if act.act_type is CognitiveActType.REFOCUS_ATTENTION:
            return {"attention": {"refocused": True, "focus_count": len(workspace.focus_items)}}
        if act.act_type is CognitiveActType.INSPECT_CONFLICT:
            return {"inspected_conflicts": len(workspace.contradictions)}
        if act.act_type is CognitiveActType.DEFER:
            return {"deferred": True}
        if act.act_type is CognitiveActType.RESPOND:
            return {"response_candidate": True}
        return {}

    @staticmethod
    def _default_response_text(
        plan: CognitivePlan,
        workspace: WorkspaceState,
        dispatch_log: list[dict[str, Any]],
    ) -> str:
        if not workspace.stimulus.user_input.strip():
            return ""
        if any(entry.get("deferred") for entry in dispatch_log):
            return "I should defer the deeper work briefly and continue with the most stable response."
        if any(entry.get("inspected_conflicts") for entry in dispatch_log):
            return "I found conflicting context and will answer conservatively while keeping that conflict visible."
        if any(entry.get("retrieved") for entry in dispatch_log):
            return "I pulled in supporting context and can answer with better grounding."
        if plan.selected_goal is not None:
            return f"Advancing goal: {plan.selected_goal.description}."
        return "Responding to the current input."

    @staticmethod
    def _fallback_response(workspace: WorkspaceState, plan: CognitivePlan) -> str:
        if plan.selected_goal is not None:
            return f"Proceeding with {plan.selected_goal.kind.value}: {plan.selected_goal.description}."
        return f"Processing input: {workspace.stimulus.user_input.strip()}"
