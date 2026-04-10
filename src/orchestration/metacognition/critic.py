from __future__ import annotations

import time
from typing import Any

from .enums import CognitiveActType, GoalKind
from .models import AttentionUpdate, CognitivePlan, ContradictionRecord, CriticReport, ExecutionResult, MemoryUpdate, WorkspaceState


class DefaultCritic:
    """Heuristically evaluate the outcome of a metacognitive cycle."""

    def evaluate(
        self,
        workspace: WorkspaceState,
        plan: CognitivePlan,
        result: ExecutionResult,
        *,
        cycle_id: str | None = None,
    ) -> CriticReport:
        cycle_id = cycle_id or self._default_cycle_id(workspace)
        timestamp = time.time()

        try:
            response_text = ((result.response_text or "") if result is not None else "").strip()
            executed_acts = tuple((result.executed_acts or ()) if result is not None else ())
            plan_acts = tuple((plan.acts or ()) if plan is not None else ())
            contradictions_raw = tuple((workspace.contradictions or ()) if workspace is not None else ())
            context_items = tuple((workspace.context_items or ()) if workspace is not None else ())
            memory_updates = self._normalize_memory_updates(getattr(result, "memory_updates", ()) if result is not None else ())
            attention_updates = self._normalize_attention_updates(
                getattr(result, "attention_updates", ()) if result is not None else ()
            )
            result_metadata = dict((result.metadata or {}) if result is not None else {})
            dispatch_log = list((result_metadata.get("dispatch_log") or []) if result is not None else [])
            productive_act_count = sum(1 for entry in dispatch_log if self._dispatch_entry_is_productive(entry))
            response_length = len(response_text)
            contradictions = tuple(ContradictionRecord.from_value(item) for item in contradictions_raw)
            executed_act_types = [act.act_type.value for act in executed_acts]
            contradiction_count = len(contradictions)
            plan_count = len(plan_acts)
            executed_count = len(executed_acts)
            productive_ratio = min(1.0, productive_act_count / max(1, plan_count))
            executed_ratio = min(1.0, executed_count / max(1, plan_count))
            completion_ratio = productive_ratio if dispatch_log else executed_ratio
            if dispatch_log:
                completion_ratio = (completion_ratio + executed_ratio) / 2.0

            response_quality = 0.0
            if response_text:
                response_quality = min(0.35, 0.05 + 0.001 * response_length)
            execution_quality = 0.10 + 0.20 * completion_ratio
            plan_quality = self._plan_quality(plan_acts, getattr(plan, "selected_goal", None))
            contradiction_penalty = self._contradiction_penalty(contradiction_count, executed_acts)
            goal_alignment = self._goal_alignment_score(getattr(plan, "selected_goal", None), executed_acts)
            context_utilization = min(0.10, 0.02 * len(context_items))

            cognitive_load = self._coerce_float(getattr(getattr(workspace, "snapshot", None), "cognitive_load", 0.0))
            uncertainty = self._coerce_float(getattr(getattr(workspace, "snapshot", None), "uncertainty", 0.0))
            retrieved_context = any(act.act_type is CognitiveActType.RETRIEVE_CONTEXT for act in executed_acts)
            inspected_conflicts = any(act.act_type is CognitiveActType.INSPECT_CONFLICT for act in executed_acts)
            load_resilience_bonus = 0.05 if cognitive_load >= 0.75 and bool(getattr(result, "success", False)) else 0.0
            uncertainty_reduction_bonus = 0.05 if uncertainty >= 0.5 and retrieved_context else 0.0
            dry_run = bool(result_metadata.get("dry_run"))
            dry_run_penalty = 0.15 if dry_run else 0.0

            success_score = response_quality
            success_score += execution_quality
            success_score += plan_quality
            success_score += goal_alignment
            success_score += context_utilization
            success_score += load_resilience_bonus
            success_score += uncertainty_reduction_bonus
            success_score -= contradiction_penalty
            success_score -= dry_run_penalty
            success_score = max(0.0, min(1.0, success_score))

            goal_progress = self._goal_progress(
                selected_goal=getattr(plan, "selected_goal", None),
                plan_acts=plan_acts,
                executed_acts=executed_acts,
                response_text=response_text,
                success=bool(getattr(result, "success", False)),
                memory_updates=memory_updates,
                attention_updates=attention_updates,
            )

            unresolved_contradictions = bool(contradictions) and not inspected_conflicts
            unresolved_uncertainty = uncertainty >= 0.5 and not retrieved_context
            follow_up = unresolved_contradictions or unresolved_uncertainty or not bool(getattr(result, "success", False))

            metadata = {
                "executed_acts": executed_act_types,
                "executed_act_count": executed_count,
                "planned_act_count": plan_count,
                "productive_act_count": productive_act_count,
                "response_length": response_length,
                "memory_updates": len(memory_updates),
                "attention_updated": bool(attention_updates),
                "contradiction_count": contradiction_count,
                "contradiction_details": [item.to_dict() for item in contradictions],
                "cognitive_load": cognitive_load,
                "uncertainty": uncertainty,
                "dispatch_summary": self._summarize_dispatch_log(dispatch_log),
                "dry_run": dry_run,
                "score_components": {
                    "response_quality": round(response_quality, 6),
                    "execution_quality": round(execution_quality, 6),
                    "plan_quality": round(plan_quality, 6),
                    "goal_alignment": round(goal_alignment, 6),
                    "context_utilization": round(context_utilization, 6),
                    "load_resilience_bonus": round(load_resilience_bonus, 6),
                    "uncertainty_reduction_bonus": round(uncertainty_reduction_bonus, 6),
                    "contradiction_penalty": round(contradiction_penalty, 6),
                    "dry_run_penalty": round(dry_run_penalty, 6),
                },
                "completion_ratio": round(completion_ratio, 6),
                "productive_ratio": round(productive_ratio, 6),
                "executed_ratio": round(executed_ratio, 6),
                "follow_up_reasons": {
                    "unresolved_contradictions": unresolved_contradictions,
                    "unresolved_uncertainty": unresolved_uncertainty,
                    "execution_failure": not bool(getattr(result, "success", False)),
                },
            }
            rationale = (
                f"score={success_score:.2f}; acts={productive_act_count}/{max(1, plan_count)} productive; "
                f"response_len={response_length}; contradictions={contradiction_count}; follow_up={follow_up}"
            )
            return CriticReport(
                cycle_id=cycle_id,
                success_score=round(success_score, 6),
                timestamp=timestamp,
                goal_progress=round(goal_progress, 6),
                follow_up_recommended=follow_up,
                contradictions_detected=contradictions,
                rationale=rationale,
                metadata=metadata,
            )
        except Exception as exc:
            return CriticReport(
                cycle_id=cycle_id,
                success_score=0.0,
                timestamp=timestamp,
                goal_progress=0.0,
                follow_up_recommended=True,
                contradictions_detected=(),
                rationale="critic evaluation failed",
                metadata={
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )

    @staticmethod
    def _default_cycle_id(workspace: WorkspaceState) -> str:
        stimulus = getattr(workspace, "stimulus", None)
        snapshot = getattr(workspace, "snapshot", None)
        session_id = getattr(stimulus, "session_id", "unknown-session")
        turn_index = getattr(snapshot, "turn_index", "unknown-turn")
        return f"{session_id}:{turn_index}"

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _dispatch_entry_is_productive(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if entry.get("error") or entry.get("stub"):
            return False
        return any(value for key, value in entry.items() if key != "act_type" and value)

    def _goal_alignment_score(
        self,
        selected_goal: Any,
        executed_acts: tuple[Any, ...],
    ) -> float:
        if selected_goal is None or not executed_acts:
            return 0.0
        core_matches = sum(1 for act in executed_acts if self._act_matches_goal_kind(act, selected_goal.kind))
        relevant_matches = sum(1 for act in executed_acts if self._act_relevant_to_goal(act, selected_goal))
        if core_matches == 0 and relevant_matches == 0:
            return 0.0
        score = 0.10 if core_matches else 0.0
        score += 0.05 * min(1.0, relevant_matches / max(1, len(executed_acts)))
        return min(0.15, score)

    def _plan_quality(
        self,
        plan_acts: tuple[Any, ...],
        selected_goal: Any,
    ) -> float:
        if not plan_acts:
            return 0.0
        relevant_acts = sum(1 for act in plan_acts if self._act_relevant_to_goal(act, selected_goal))
        relevance_ratio = relevant_acts / max(1, len(plan_acts))
        breadth_ratio = min(len(plan_acts), 3) / 3.0
        return min(0.20, 0.12 * relevance_ratio + 0.08 * breadth_ratio)

    @staticmethod
    def _contradiction_penalty(contradiction_count: int, executed_acts: tuple[Any, ...]) -> float:
        raw_penalty = min(0.15, 0.05 * contradiction_count)
        inspected_conflicts = any(act.act_type is CognitiveActType.INSPECT_CONFLICT for act in executed_acts)
        return raw_penalty * (0.25 if inspected_conflicts else 1.0)

    def _goal_progress(
        self,
        *,
        selected_goal: Any,
        plan_acts: tuple[Any, ...],
        executed_acts: tuple[Any, ...],
        response_text: str,
        success: bool,
        memory_updates: tuple[MemoryUpdate, ...],
        attention_updates: tuple[AttentionUpdate, ...],
    ) -> float:
        if selected_goal is None:
            return 0.0
        base = 0.6 if success else 0.3 if response_text else 0.0
        matching_acts = sum(1 for act in executed_acts if self._act_matches_goal_kind(act, selected_goal.kind))
        base += 0.2 * matching_acts / max(1, len(plan_acts))
        if memory_updates:
            base += 0.1
        if attention_updates:
            base += 0.1
        return min(1.0, base)

    @staticmethod
    def _normalize_memory_updates(values: Any) -> tuple[MemoryUpdate, ...]:
        if values is None:
            return ()
        if isinstance(values, (list, tuple)):
            return tuple(MemoryUpdate.from_value(item) for item in values)
        return (MemoryUpdate.from_value(values),)

    @staticmethod
    def _normalize_attention_updates(values: Any) -> tuple[AttentionUpdate, ...]:
        if values is None:
            return ()
        if isinstance(values, (list, tuple)):
            return tuple(AttentionUpdate.from_value(item) for item in values)
        return (AttentionUpdate.from_value(values),)

    @staticmethod
    def _act_matches_goal_kind(act: Any, goal_kind: GoalKind) -> bool:
        match goal_kind:
            case GoalKind.RESPONSE:
                return act.act_type is CognitiveActType.RESPOND
            case GoalKind.REFLECTION:
                return act.act_type is CognitiveActType.INSPECT_CONFLICT
            case GoalKind.MEMORY:
                return act.act_type in {CognitiveActType.RETRIEVE_CONTEXT, CognitiveActType.STORE_MEMORY}
            case GoalKind.ATTENTION:
                return act.act_type is CognitiveActType.REFOCUS_ATTENTION
            case GoalKind.PLANNING:
                return act.act_type is CognitiveActType.DEFER
        return False

    def _act_relevant_to_goal(self, act: Any, selected_goal: Any) -> bool:
        if selected_goal is None:
            return False
        if self._act_matches_goal_kind(act, selected_goal.kind):
            return True
        support_map = {
            GoalKind.RESPONSE: {CognitiveActType.RETRIEVE_CONTEXT, CognitiveActType.INSPECT_CONFLICT},
            GoalKind.REFLECTION: {CognitiveActType.RETRIEVE_CONTEXT, CognitiveActType.RESPOND},
            GoalKind.MEMORY: {CognitiveActType.RESPOND},
            GoalKind.ATTENTION: {CognitiveActType.DEFER},
            GoalKind.PLANNING: {CognitiveActType.RETRIEVE_CONTEXT, CognitiveActType.RESPOND},
        }
        return act.act_type in support_map.get(selected_goal.kind, set())

    def _summarize_dispatch_log(self, dispatch_log: list[Any]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for entry in dispatch_log:
            if not isinstance(entry, dict):
                summary.append({"act_type": "unknown", "productive": False, "details": {"value": str(entry)}})
                continue
            details = {key: value for key, value in entry.items() if key != "act_type" and value}
            summary.append(
                {
                    "act_type": str(entry.get("act_type", "unknown")),
                    "productive": bool(details),
                    "details": details,
                }
            )
        return summary
