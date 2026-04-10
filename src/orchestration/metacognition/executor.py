from __future__ import annotations

import inspect
from typing import Any, Callable, cast

from .enums import CognitiveActType
from .models import AttentionUpdate, CognitiveActProposal, CognitivePlan, ExecutionResult, MemoryUpdate, WorkspaceState


ActDispatcher = Callable[..., dict[str, Any]]
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
        self._uses_stub_dispatch = dispatch_act is None
        self._dispatch_accepts_history = self._callable_accepts_history(self._dispatch_act)

    def execute(self, workspace: WorkspaceState, plan: CognitivePlan) -> ExecutionResult:
        executed_acts: list[CognitiveActProposal] = []
        memory_updates: list[MemoryUpdate] = []
        attention_updates: list[AttentionUpdate] = []
        attention_log: list[dict[str, Any]] = []
        dispatch_log: list[dict[str, Any]] = []
        plan_acts = tuple(plan.acts or ())

        for act in plan_acts:
            try:
                dispatch_result = self._dispatch_with_context(act, workspace, dispatch_log) or {}
            except Exception as exc:
                dispatch_result = {"error": str(exc), "error_type": type(exc).__name__}
            if not isinstance(dispatch_result, dict):
                dispatch_result = {"result": dispatch_result}
            dispatch_entry = {"act_type": act.act_type.value, **dispatch_result}
            dispatch_log.append(dispatch_entry)
            if dispatch_result and not dispatch_result.get("error"):
                executed_acts.append(act)
            if dispatch_result.get("memory_updated") and not dispatch_result.get("stub"):
                memory_updates.append(MemoryUpdate(description=act.description, source_act_type=act.act_type))
            attention_payload = dispatch_result.get("attention")
            if isinstance(attention_payload, dict):
                attention_entry = {
                    "act_type": act.act_type.value,
                    "stub": bool(dispatch_result.get("stub")),
                    **attention_payload,
                }
                attention_log.append(attention_entry)
                if not dispatch_result.get("stub"):
                    attention_updates.append(AttentionUpdate(changes=attention_payload, source_act_type=act.act_type))

        response_builder_error: dict[str, str] | None = None
        try:
            response_text = (self._build_response_text(plan, workspace, dispatch_log) or "").strip()
        except Exception as exc:
            response_text = ""
            response_builder_error = {
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        if not response_text and workspace.stimulus.user_input.strip():
            response_text = self._fallback_response(workspace, plan)

        error_count = sum(1 for entry in dispatch_log if entry.get("error"))
        all_errored = bool(dispatch_log) and error_count == len(dispatch_log)
        has_output = bool(response_text or memory_updates or attention_updates)
        dry_run = self._uses_stub_dispatch or any(entry.get("stub") for entry in dispatch_log)
        success = bool(executed_acts) and has_output and not all_errored
        metadata: dict[str, Any] = {
            "dispatch_log": dispatch_log,
            "attempted_acts": [act.act_type.value for act in plan_acts],
            "attempted_act_count": len(plan_acts),
            "error_count": error_count,
            "dry_run": dry_run,
            "attention_log": attention_log,
        }
        if response_builder_error is not None:
            metadata["response_builder_error"] = response_builder_error
        return ExecutionResult(
            success=success,
            response_text=response_text,
            executed_acts=tuple(executed_acts),
            memory_updates=tuple(memory_updates),
            attention_updates=tuple(attention_updates),
            metadata=metadata,
        )

    @staticmethod
    def _default_dispatch(act: CognitiveActProposal, workspace: WorkspaceState) -> dict[str, Any]:
        """Stub dispatcher for testing and dry-run execution without side effects."""
        if act.act_type is CognitiveActType.RETRIEVE_CONTEXT:
            return {"stub": True, "retrieved": len(workspace.context_items)}
        if act.act_type is CognitiveActType.STORE_MEMORY:
            return {"stub": True, "would_store": workspace.stimulus.user_input[:80]}
        if act.act_type is CognitiveActType.REFOCUS_ATTENTION:
            return {"stub": True, "attention": {"refocused": True, "focus_count": len(workspace.focus_items)}}
        if act.act_type is CognitiveActType.INSPECT_CONFLICT:
            return {"stub": True, "inspected_conflicts": len(workspace.contradictions)}
        if act.act_type is CognitiveActType.DEFER:
            return {"stub": True, "deferred": True}
        if act.act_type is CognitiveActType.RESPOND:
            return {"stub": True, "response_candidate": True}
        return {}

    @staticmethod
    def _default_response_text(
        plan: CognitivePlan,
        workspace: WorkspaceState,
        dispatch_log: list[dict[str, Any]],
    ) -> str:
        if not workspace.stimulus.user_input.strip():
            return ""
        parts: list[str] = []
        if any(entry.get("error") for entry in dispatch_log):
            parts.append("Some planned actions could not be completed cleanly.")
        if any(entry.get("deferred") for entry in dispatch_log):
            parts.append("Deferring deeper work while conditions are elevated.")
        if any(entry.get("inspected_conflicts") for entry in dispatch_log):
            parts.append("Found conflicting context; answering conservatively.")
        if any(entry.get("retrieved") for entry in dispatch_log):
            parts.append("Retrieved supporting context for better grounding.")
        if not parts and plan.selected_goal is not None:
            parts.append(f"Advancing goal: {plan.selected_goal.description}.")
        if not parts:
            parts.append("Responding to the current input.")
        return " ".join(parts)

    @staticmethod
    def _fallback_response(workspace: WorkspaceState, plan: CognitivePlan) -> str:
        if plan.selected_goal is not None:
            return f"Proceeding with {plan.selected_goal.kind.value}: {plan.selected_goal.description}."
        return "Processing the current input."

    def _dispatch_with_context(
        self,
        act: CognitiveActProposal,
        workspace: WorkspaceState,
        dispatch_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dispatch_callable = cast(Callable[..., dict[str, Any]], self._dispatch_act)
        if self._dispatch_accepts_history:
            return dispatch_callable(act, workspace, list(dispatch_log))
        return dispatch_callable(act, workspace)

    @staticmethod
    def _callable_accepts_history(dispatch_act: ActDispatcher) -> bool:
        try:
            parameters = tuple(inspect.signature(dispatch_act).parameters.values())
        except (TypeError, ValueError):
            return False
        positional_count = sum(
            1
            for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        return any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters) or positional_count >= 3

    @staticmethod
    def _merge_attention_updates(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(current)
        for key, value in incoming.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = DefaultPlanExecutor._merge_attention_updates(existing, value)
            else:
                merged[key] = value
        return merged
