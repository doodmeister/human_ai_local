from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from .enums import GoalKind, ScheduledTaskStatus
from .interfaces import StateProvider
from .models import (
    CognitiveGoal,
    InternalStateSnapshot,
    ScheduledCognitiveTask,
    SelfModel,
    Stimulus,
)

__all__ = ["DefaultStateProvider"]

_LOGGER = logging.getLogger(__name__)
_SELF_MODEL_FIELD_NAMES = {field.name for field in fields(SelfModel)}
_GOAL_KIND_MAP = {
    "response": GoalKind.RESPONSE,
    "reply": GoalKind.RESPONSE,
    "reflection": GoalKind.REFLECTION,
    "reflect": GoalKind.REFLECTION,
    "memory": GoalKind.MEMORY,
    "attention": GoalKind.ATTENTION,
    "planning": GoalKind.PLANNING,
    "plan": GoalKind.PLANNING,
}


@runtime_checkable
class _MemoryStatusSource(Protocol):
    def get_status(self) -> dict[str, Any]: ...


@runtime_checkable
class _AttentionStatusSource(Protocol):
    def get_attention_status(self) -> dict[str, Any]: ...


@runtime_checkable
class _GoalSource(Protocol):
    def get_active_goals(self) -> list[Any]: ...


class _GoalSourceContainer(Protocol):
    goal_manager: _GoalSource


class DefaultStateProvider(StateProvider):
    """Snapshot current memory, attention, goal, self-model, and task state.

    Snapshots are not atomic across providers. Callers that require a coherent
    cross-provider view must quiesce background writers before invoking this
    adapter.
    """

    def __init__(
        self,
        *,
        memory: _MemoryStatusSource | None = None,
        attention: _AttentionStatusSource | None = None,
        executive: _GoalSource | _GoalSourceContainer | None = None,
        self_model_provider: Callable[[str], Any] | None = None,
        task_provider: Callable[[str], list[Any]] | None = None,
    ) -> None:
        self._memory = memory
        self._attention = attention
        self._executive = executive
        self._self_model_provider = self_model_provider
        self._task_provider = task_provider

    def snapshot(self, stimulus: Stimulus) -> InternalStateSnapshot:
        memory_status = self._get_memory_status()
        attention_status = self._get_attention_status()
        active_goals = tuple(self._get_active_goals())
        self_model = self._get_self_model(stimulus.session_id)
        pending_tasks = tuple(self._get_tasks(stimulus.session_id))
        cognitive_load = self._coerce_unit_interval(attention_status.get("cognitive_load"), default=0.0)
        uncertainty = self._coerce_unit_interval(memory_status.get("uncertainty"), default=0.0)

        return InternalStateSnapshot(
            session_id=stimulus.session_id,
            turn_index=stimulus.turn_index,
            memory_status=memory_status,
            attention_status=attention_status,
            active_goals=active_goals,
            self_model=self_model,
            pending_tasks=pending_tasks,
            cognitive_load=cognitive_load,
            uncertainty=uncertainty,
            metadata={**stimulus.metadata, "input_type": getattr(stimulus.input_type, "value", stimulus.input_type)},
        )

    def _get_memory_status(self) -> dict[str, Any]:
        """Return normalized memory status from a provider exposing get_status()."""
        if self._memory is None or not isinstance(self._memory, _MemoryStatusSource):
            return {}
        try:
            status = self._memory.get_status()
        except Exception as exc:
            _LOGGER.warning("memory.get_status failed: %s", exc, exc_info=True)
            return {}
        if not isinstance(status, Mapping):
            _LOGGER.debug("memory.get_status returned unsupported type %s", type(status).__name__)
            return {}
        return dict(status)

    def _get_attention_status(self) -> dict[str, Any]:
        """Return normalized attention status from a provider exposing get_attention_status()."""
        if self._attention is None or not isinstance(self._attention, _AttentionStatusSource):
            return {}
        try:
            status = self._attention.get_attention_status()
        except Exception as exc:
            _LOGGER.warning("attention.get_attention_status failed: %s", exc, exc_info=True)
            return {}
        if not isinstance(status, Mapping):
            _LOGGER.debug("attention.get_attention_status returned unsupported type %s", type(status).__name__)
            return {}
        return dict(status)

    def _get_active_goals(self) -> list[CognitiveGoal]:
        """Return normalized active goals from either the executive or executive.goal_manager."""
        goal_source = self._resolve_goal_source()
        if goal_source is None:
            return []
        try:
            goals = goal_source.get_active_goals() or []
        except Exception as exc:
            _LOGGER.warning("executive.get_active_goals failed: %s", exc, exc_info=True)
            return []

        normalized: list[CognitiveGoal] = []
        for goal in goals:
            try:
                normalized.append(self._coerce_goal(goal))
            except Exception as exc:
                _LOGGER.warning("failed to coerce goal of type %s: %s", type(goal).__name__, exc, exc_info=True)
        return normalized

    def _get_self_model(self, session_id: str) -> SelfModel:
        """Return a normalized self-model for the session from the configured provider."""
        if self._self_model_provider is None:
            return SelfModel(session_id=session_id)

        try:
            raw_model = self._self_model_provider(session_id)
        except Exception as exc:
            _LOGGER.warning("self_model_provider failed for session %s: %s", session_id, exc, exc_info=True)
            return SelfModel(session_id=session_id)

        if isinstance(raw_model, SelfModel):
            return raw_model

        normalized_model = self._normalize_self_model_source(raw_model)
        if isinstance(normalized_model, Mapping):
            return self._coerce_self_model_mapping(session_id, normalized_model)

        _LOGGER.warning(
            "self_model_provider returned unsupported type %s for session %s; using default",
            type(raw_model).__name__,
            session_id,
        )
        return SelfModel(session_id=session_id)

    def _get_tasks(self, session_id: str) -> list[ScheduledCognitiveTask]:
        """Return normalized scheduled tasks from the configured task provider."""
        if self._task_provider is None:
            return []
        try:
            tasks = self._task_provider(session_id) or []
        except Exception as exc:
            _LOGGER.warning("task_provider failed for session %s: %s", session_id, exc, exc_info=True)
            return []

        normalized: list[ScheduledCognitiveTask] = []
        for task in tasks:
            if isinstance(task, ScheduledCognitiveTask):
                normalized.append(task)
                continue
            if not isinstance(task, Mapping):
                _LOGGER.debug("skipping task of unsupported type %s", type(task).__name__)
                continue
            normalized.append(
                ScheduledCognitiveTask(
                    task_id=self._coerce_task_id(task),
                    description=str(task.get("description") or task.get("content") or ""),
                    status=self._coerce_task_status(task.get("status")),
                    priority=self._coerce_unit_interval(task.get("priority"), default=0.0),
                    due_at=task.get("due_at"),
                    goal_id=(str(task.get("goal_id")) if task.get("goal_id") not in (None, "") else None),
                    metadata={
                        str(key): value
                        for key, value in task.items()
                        if key not in {"task_id", "id", "description", "content", "status", "priority", "due_at", "goal_id"}
                    },
                )
            )
        return normalized

    def _resolve_goal_source(self) -> _GoalSource | None:
        """Resolve a goal source from either the executive object or its goal_manager attribute."""
        if self._executive is None:
            return None
        if isinstance(self._executive, _GoalSource):
            return self._executive
        goal_source = getattr(self._executive, "goal_manager", None)
        if isinstance(goal_source, _GoalSource):
            return goal_source
        return None

    @classmethod
    def _coerce_goal(cls, goal: Any) -> CognitiveGoal:
        """Coerce a source goal object or mapping into the metacognition goal model."""
        description = cls._goal_attr(goal, "description", None) or cls._goal_attr(goal, "title", "")
        title = cls._goal_attr(goal, "title", "") or description
        source_id = cls._goal_attr(goal, "id", None) or cls._goal_attr(goal, "goal_id", None)
        progress = cls._coerce_float(cls._goal_attr(goal, "progress", None), default=0.0)
        raw_status = cls._goal_attr(goal, "status", None)
        raw_kind = cls._extract_goal_kind(goal)
        raw_urgency = cls._goal_attr(goal, "urgency", None)
        raw_salience = cls._goal_attr(goal, "salience", None)

        # Goal urgency and salience are consumed independently by HeuristicGoalManager
        # and WorkspaceBuilder, so progress is preserved in metadata instead of being
        # treated as an urgency proxy.
        metadata: dict[str, Any] = {
            "title": title,
            "status": cls._enum_value(raw_status),
            "progress": progress,
        }
        if raw_kind is not None:
            metadata["source_kind"] = cls._enum_value(raw_kind)

        return CognitiveGoal(
            goal_id=str(source_id) if source_id not in (None, "") else f"goal-{uuid4().hex[:8]}",
            description=str(description),
            priority=cls._coerce_goal_priority(cls._goal_attr(goal, "priority", None), default=0.0),
            kind=cls._coerce_goal_kind(raw_kind),
            urgency=cls._coerce_unit_interval(raw_urgency, default=0.0),
            salience=cls._coerce_unit_interval(raw_salience, default=0.0),
            rationale=str(cls._enum_value(raw_status) or ""),
            metadata=metadata,
        )

    @staticmethod
    def _normalize_self_model_source(raw_model: Any) -> Any:
        """Normalize alternate self-model provider objects into mapping-like payloads."""
        if hasattr(raw_model, "model_dump"):
            return raw_model.model_dump()
        if hasattr(raw_model, "dict"):
            return raw_model.dict()
        if hasattr(raw_model, "to_dict"):
            return raw_model.to_dict()
        return raw_model

    @classmethod
    def _coerce_self_model_mapping(cls, session_id: str, raw_model: Mapping[str, Any]) -> SelfModel:
        """Coerce a self-model mapping while preserving unknown fields in metadata."""
        metadata: dict[str, Any] = {}
        raw_metadata = raw_model.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata.update({str(key): value for key, value in raw_metadata.items()})

        kwargs: dict[str, Any] = {}
        for field_name in _SELF_MODEL_FIELD_NAMES:
            if field_name in {"session_id", "metadata"} or field_name not in raw_model:
                continue
            raw_value = raw_model[field_name]
            if field_name == "confidence":
                kwargs[field_name] = cls._coerce_unit_interval(raw_value, default=0.5)
            elif field_name == "traits":
                kwargs[field_name] = dict(raw_value) if isinstance(raw_value, Mapping) else {}
            elif field_name in {"beliefs", "recent_updates"}:
                kwargs[field_name] = tuple(raw_value) if isinstance(raw_value, (list, tuple)) else ()
            else:
                kwargs[field_name] = raw_value

        for key, value in raw_model.items():
            if key not in _SELF_MODEL_FIELD_NAMES:
                metadata[str(key)] = value

        return SelfModel(session_id=session_id, metadata=metadata, **kwargs)

    @staticmethod
    def _goal_attr(goal: Any, name: str, default: Any = None) -> Any:
        if isinstance(goal, Mapping):
            return goal.get(name, default)
        return getattr(goal, name, default)

    @classmethod
    def _extract_goal_kind(cls, goal: Any) -> Any:
        for field_name in ("kind", "goal_kind"):
            raw_kind = cls._goal_attr(goal, field_name, None)
            if raw_kind is not None:
                return raw_kind

        for container_name in ("metadata", "context"):
            container = cls._goal_attr(goal, container_name, None)
            if isinstance(container, Mapping):
                raw_kind = container.get("kind") or container.get("goal_kind")
                if raw_kind is not None:
                    return raw_kind
        return None

    @classmethod
    def _coerce_goal_kind(cls, raw_kind: Any) -> GoalKind:
        if raw_kind is None:
            return GoalKind.PLANNING
        normalized_kind = str(cls._enum_value(raw_kind)).strip().lower()
        return _GOAL_KIND_MAP.get(normalized_kind, GoalKind.PLANNING)

    @staticmethod
    def _coerce_task_status(raw_status: Any) -> ScheduledTaskStatus:
        normalized_status = str(raw_status) if raw_status is not None else ScheduledTaskStatus.PENDING.value
        try:
            return ScheduledTaskStatus(normalized_status)
        except ValueError:
            _LOGGER.warning("unsupported scheduled task status %r; defaulting to pending", normalized_status)
            return ScheduledTaskStatus.PENDING

    @staticmethod
    def _coerce_task_id(task: Mapping[str, Any]) -> str:
        source_id = task.get("task_id") or task.get("id")
        if source_id not in (None, ""):
            return str(source_id)
        fallback_payload = {
            "description": task.get("description") or task.get("content") or "",
            "due_at": task.get("due_at"),
            "goal_id": task.get("goal_id"),
            "status": task.get("status"),
        }
        digest = hashlib.sha1(json.dumps(fallback_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
        return f"task-{digest}"

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        if value is None:
            return default
        raw_value = getattr(value, "value", value)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _coerce_unit_interval(cls, value: Any, *, default: float) -> float:
        normalized = cls._coerce_float(value, default=default)
        return max(0.0, min(1.0, normalized))

    @classmethod
    def _coerce_goal_priority(cls, value: Any, *, default: float) -> float:
        normalized = cls._coerce_float(value, default=default)
        if normalized <= 1.0:
            return max(0.0, normalized)
        if normalized <= 5.0:
            return min(1.0, normalized / 5.0)
        return min(1.0, normalized / 10.0)
