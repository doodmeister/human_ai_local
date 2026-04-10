from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from .interfaces import ContradictionProvider, MemoryContextProvider, WorkspaceBuilder
from .models import ContradictionRecord, CognitiveGoal, InternalStateSnapshot, RetrievalContextItem, Stimulus, WorkspaceState
from .thresholds import DEFAULT_COGNITIVE_THRESHOLDS

__all__ = ["DefaultWorkspaceBuilder"]

_LOGGER = logging.getLogger(__name__)


class DefaultWorkspaceBuilder(WorkspaceBuilder):
    """Build a deterministic cognition workspace from a stimulus and snapshot.

    Providers are invoked sequentially and the resulting workspace is not
    atomic across them. Callers that require a coherent point-in-time view must
    quiesce background writers before calling build().

    The default cognitive-load and uncertainty thresholds are shared with the
    policy engine and scheduler through DEFAULT_COGNITIVE_THRESHOLDS.
    """

    def __init__(
        self,
        *,
        memory_context_provider: MemoryContextProvider | None = None,
        contradiction_provider: ContradictionProvider | None = None,
        cognitive_load_threshold: float = DEFAULT_COGNITIVE_THRESHOLDS.cognitive_load_threshold,
        uncertainty_threshold: float = DEFAULT_COGNITIVE_THRESHOLDS.uncertainty_threshold,
    ) -> None:
        for name, value in (
            ("cognitive_load_threshold", cognitive_load_threshold),
            ("uncertainty_threshold", uncertainty_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
        self._memory_context_provider = memory_context_provider
        self._contradiction_provider = contradiction_provider
        self._cognitive_load_threshold = cognitive_load_threshold
        self._uncertainty_threshold = uncertainty_threshold

    def build(self, stimulus: Stimulus, snapshot: InternalStateSnapshot) -> WorkspaceState:
        context_items, context_error = self._get_context_items(stimulus)
        contradictions, contradiction_error = self._get_contradictions(stimulus.session_id)
        context_items_tuple = tuple(context_items)
        contradictions_tuple = tuple(contradictions)
        dominant_goal = self._select_dominant_goal(snapshot.active_goals)
        focus_items = tuple(snapshot.attention_status.current_focus or ())
        notes = self._build_notes(snapshot, contradictions_tuple)

        metadata: dict[str, Any] = {
            "context_count": len(context_items_tuple),
            "contradiction_count": len(contradictions_tuple),
            "high_cognitive_load": snapshot.cognitive_load >= self._cognitive_load_threshold,
            "high_uncertainty": snapshot.uncertainty >= self._uncertainty_threshold,
        }
        provider_errors: dict[str, bool] = {}
        if context_error:
            provider_errors["memory_context"] = True
        if contradiction_error:
            provider_errors["contradictions"] = True
        if provider_errors:
            metadata["provider_errors"] = provider_errors

        return WorkspaceState(
            stimulus=stimulus,
            snapshot=snapshot,
            context_items=context_items_tuple,
            focus_items=focus_items,
            contradictions=contradictions_tuple,
            notes=notes,
            dominant_goal=dominant_goal,
            metadata=metadata,
        )

    def _get_context_items(self, stimulus: Stimulus) -> tuple[list[RetrievalContextItem], bool]:
        """Return normalized retrieval context from the configured provider.

        Supported provider shapes are a flat list/tuple of items, or a mapping
        from source-system name to list/tuple of items. Unsupported shapes are
        ignored after emitting a warning and marking the provider as degraded.
        """
        if self._memory_context_provider is None:
            return [], False
        try:
            context = self._memory_context_provider(stimulus.user_input, stimulus.session_id)
        except Exception as exc:
            _LOGGER.warning(
                "memory_context_provider failed for session %s: %s",
                stimulus.session_id,
                exc,
                exc_info=True,
            )
            return [], True

        if isinstance(context, Mapping):
            flattened: list[RetrievalContextItem] = []
            degraded = False
            for source_system, items in context.items():
                if not isinstance(items, (list, tuple)):
                    _LOGGER.warning(
                        "memory_context_provider returned unsupported collection type %s for source %s in session %s",
                        type(items).__name__,
                        source_system,
                        stimulus.session_id,
                    )
                    degraded = True
                    continue
                for item in items:
                    flattened.append(RetrievalContextItem.from_value(item, source_system=source_system))
            return flattened, degraded

        if isinstance(context, (list, tuple)):
            return [RetrievalContextItem.from_value(item) for item in context], False

        _LOGGER.warning(
            "memory_context_provider returned unsupported type %s for session %s; expected list/tuple or mapping of list/tuple",
            type(context).__name__,
            stimulus.session_id,
        )
        return [], True

    def _get_contradictions(self, session_id: str) -> tuple[list[ContradictionRecord], bool]:
        """Return normalized contradictions from the configured provider."""
        if self._contradiction_provider is None:
            return [], False
        try:
            contradictions = self._contradiction_provider(session_id)
        except Exception as exc:
            _LOGGER.warning(
                "contradiction_provider failed for session %s: %s",
                session_id,
                exc,
                exc_info=True,
            )
            return [], True
        if not isinstance(contradictions, (list, tuple)):
            _LOGGER.warning(
                "contradiction_provider returned unsupported type %s for session %s; expected list/tuple",
                type(contradictions).__name__,
                session_id,
            )
            return [], True
        return [ContradictionRecord.from_value(item) for item in contradictions], False

    def contradictions_for_session(self, session_id: str) -> list[ContradictionRecord]:
        """Fetch contradictions for a session.

        This lookup is not cached and not guaranteed idempotent. Callers that
        need a stable snapshot should build a WorkspaceState and read its
        contradictions field.
        """
        contradictions, _ = self._get_contradictions(session_id)
        return contradictions

    @staticmethod
    def _select_dominant_goal(goals: tuple[CognitiveGoal, ...]) -> CognitiveGoal | None:
        """Return the highest-priority goal, with salience then urgency as tie-breakers."""
        if not goals:
            return None
        return max(goals, key=lambda goal: (goal.priority, goal.salience, goal.urgency))

    def _build_notes(
        self,
        snapshot: InternalStateSnapshot,
        contradictions: tuple[ContradictionRecord, ...],
    ) -> tuple[str, ...]:
        """Return human-readable notes summarizing threshold crossings and contradictions."""
        notes: list[str] = []
        if snapshot.cognitive_load >= self._cognitive_load_threshold:
            notes.append(f"high_cognitive_load:{snapshot.cognitive_load:.2f}")
        if snapshot.uncertainty >= self._uncertainty_threshold:
            notes.append(f"high_uncertainty:{snapshot.uncertainty:.2f}")
        if contradictions:
            notes.append(f"contradictions:{len(contradictions)}")
        return tuple(notes)
