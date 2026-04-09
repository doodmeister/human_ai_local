from __future__ import annotations

from typing import Any, Callable

from .models import CognitiveGoal, InternalStateSnapshot, Stimulus, WorkspaceState


class DefaultWorkspaceBuilder:
    """Build a deterministic cognition workspace from a stimulus and snapshot."""

    def __init__(
        self,
        *,
        memory_context_provider: Callable[[str, str], Any] | None = None,
        contradiction_provider: Callable[[str], Any] | None = None,
    ) -> None:
        self._memory_context_provider = memory_context_provider
        self._contradiction_provider = contradiction_provider

    def build(self, stimulus: Stimulus, snapshot: InternalStateSnapshot) -> WorkspaceState:
        context_items = tuple(self._get_context_items(stimulus))
        contradictions = tuple(self._get_contradictions(stimulus.session_id))
        dominant_goal = self._select_dominant_goal(snapshot.active_goals)
        focus_items = tuple(snapshot.attention_status.get("current_focus", ()) or ())
        notes = self._build_notes(snapshot, contradictions)

        return WorkspaceState(
            stimulus=stimulus,
            snapshot=snapshot,
            context_items=context_items,
            focus_items=focus_items,
            contradictions=contradictions,
            notes=notes,
            dominant_goal=dominant_goal,
            metadata={
                "context_count": len(context_items),
                "contradiction_count": len(contradictions),
            },
        )

    def _get_context_items(self, stimulus: Stimulus) -> list[dict[str, Any]]:
        if self._memory_context_provider is None:
            return []
        try:
            context = self._memory_context_provider(stimulus.user_input, stimulus.session_id)
        except Exception:
            return []

        if isinstance(context, dict):
            flattened: list[dict[str, Any]] = []
            for source_system, items in context.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        normalized = dict(item)
                    else:
                        normalized = {"content": str(item)}
                    normalized.setdefault("source_system", source_system)
                    flattened.append(normalized)
            return flattened

        if isinstance(context, list):
            return [dict(item) if isinstance(item, dict) else {"content": str(item)} for item in context]
        return []

    def _get_contradictions(self, session_id: str) -> list[dict[str, Any]]:
        if self._contradiction_provider is None:
            return []
        try:
            contradictions = self._contradiction_provider(session_id)
        except Exception:
            return []
        if not isinstance(contradictions, list):
            return []
        return [dict(item) for item in contradictions if isinstance(item, dict)]

    def contradictions_for_session(self, session_id: str) -> list[dict[str, Any]]:
        return self._get_contradictions(session_id)

    @staticmethod
    def _select_dominant_goal(goals: tuple[CognitiveGoal, ...]) -> CognitiveGoal | None:
        if not goals:
            return None
        return max(goals, key=lambda goal: (goal.priority, goal.salience, goal.urgency))

    @staticmethod
    def _build_notes(
        snapshot: InternalStateSnapshot,
        contradictions: tuple[dict[str, Any], ...],
    ) -> tuple[str, ...]:
        notes: list[str] = []
        if snapshot.cognitive_load >= 0.75:
            notes.append(f"high_cognitive_load:{snapshot.cognitive_load:.2f}")
        if snapshot.uncertainty >= 0.5:
            notes.append(f"high_uncertainty:{snapshot.uncertainty:.2f}")
        if contradictions:
            notes.append(f"contradictions:{len(contradictions)}")
        return tuple(notes)
