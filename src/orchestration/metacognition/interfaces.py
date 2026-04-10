"""Protocol contracts for the metacognition pipeline's pluggable components.

This module defines the typed interfaces used to connect state loading,
workspace construction, goal ranking, policy selection, execution, critique,
event publication, scheduling, and trace persistence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .models import (
    CognitiveActProposal,
    CognitiveGoal,
    CognitivePlan,
    ContradictionRecord,
    CriticReport,
    ExecutionResult,
    InternalStateSnapshot,
    ScheduledCognitiveTask,
    SelfModel,
    Stimulus,
    WorkspaceState,
)

EventHandler = Callable[[Mapping[str, Any]], None]
Unsubscribe = Callable[[], None]

__all__ = [
    "CognitiveScheduler",
    "ContradictionProvider",
    "Critic",
    "CycleTracer",
    "EventBus",
    "EventHandler",
    "GoalManager",
    "MemoryContextProvider",
    "PlanExecutor",
    "PolicyEngine",
    "SelfModelUpdater",
    "StateProvider",
    "Unsubscribe",
    "WorkspaceBuilder",
]


class StateProvider(Protocol):
    """Build an internal state snapshot for the current stimulus."""

    def snapshot(self, stimulus: Stimulus) -> InternalStateSnapshot:
        """Return the current metacognitive state for the incoming stimulus."""
        ...


class MemoryContextProvider(Protocol):
    """Return retrieval context as a flat sequence or source-tagged mapping."""

    def __call__(
        self,
        user_input: str,
        session_id: str,
    ) -> Sequence[Mapping[str, Any] | Any] | Mapping[str, Sequence[Mapping[str, Any] | Any]]:
        ...


class ContradictionProvider(Protocol):
    """Return unresolved contradictions for a session."""

    def __call__(self, session_id: str) -> Sequence[Mapping[str, Any] | Any]:
        ...


class WorkspaceBuilder(Protocol):
    """Translate a stimulus and snapshot into a scored working context."""

    def build(self, stimulus: Stimulus, snapshot: InternalStateSnapshot) -> WorkspaceState:
        """Construct the workspace used by goal ranking and policy selection."""
        ...

    def contradictions_for_session(self, session_id: str) -> list[ContradictionRecord]:
        """Return unresolved contradictions already known for a session."""
        ...


class GoalManager(Protocol):
    """Rank active and synthesized cognitive goals for the current workspace."""

    def rank_goals(self, workspace: WorkspaceState) -> list[CognitiveGoal]:
        """Return goals ordered from most to least relevant for this cycle."""
        ...


class PolicyEngine(Protocol):
    """Propose and select cognitive acts for the ranked goals."""

    def propose_acts(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
    ) -> list[CognitiveActProposal]:
        """Return candidate acts that could advance the current workspace goals."""
        ...

    def select_plan(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
        proposals: list[CognitiveActProposal],
    ) -> CognitivePlan:
        """Choose a deterministic execution plan from the proposed acts."""
        ...


class PlanExecutor(Protocol):
    """Execute a cognitive plan and summarize the observable outcome."""

    def execute(self, workspace: WorkspaceState, plan: CognitivePlan) -> ExecutionResult:
        """Run the chosen plan and return the execution result and metadata."""
        ...


class Critic(Protocol):
    """Evaluate a completed cycle and produce critique metadata."""

    def evaluate(
        self,
        workspace: WorkspaceState,
        plan: CognitivePlan,
        result: ExecutionResult,
        *,
        cycle_id: str | None = None,
    ) -> CriticReport:
        """Return a critic report describing success, progress, and follow-up needs."""
        ...


class SelfModelUpdater(Protocol):
    """Apply critic feedback to the compact metacognitive self-model."""

    def apply(self, current: SelfModel, report: CriticReport) -> SelfModel:
        """Return an updated self-model derived from the latest critic report."""
        ...


class CognitiveScheduler(Protocol):
    """Turn critic output into deferred follow-up work items."""

    def schedule(
        self,
        workspace: WorkspaceState,
        report: CriticReport,
    ) -> list[ScheduledCognitiveTask]:
        """Return scheduled cognitive tasks that should be persisted for later execution."""
        ...


class EventBus(Protocol):
    """Publish synchronous metacognition events to in-process subscribers."""

    def subscribe(self, event_name: str, handler: EventHandler) -> Unsubscribe:
        """Register a handler and return the canonical unsubscribe callback."""
        ...

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Remove a previously subscribed handler when an unsubscribe callback is unavailable."""
        ...

    def publish(self, event_name: str, payload: Mapping[str, Any]) -> int:
        """Dispatch an event payload and return the number of handler failures."""
        ...


class CycleTracer(Protocol):
    """Persist cycle traces and related metacognition artifacts on disk."""

    def write_trace(self, trace: Mapping[str, Any]) -> str:
        """Persist a cycle trace and return its stable trace identifier."""
        ...

    def persist_self_model(self, session_id: str, self_model: Mapping[str, Any] | SelfModel) -> Path:
        """Overwrite the persisted self-model snapshot for a session and return its path."""
        ...

    def upsert_task_queue(
        self,
        session_id: str,
        tasks: Sequence[Mapping[str, Any] | ScheduledCognitiveTask],
    ) -> Path:
        """Merge tasks into the persisted queue by task ID and return the queue file path."""
        ...

    def load_self_model(self, session_id: str) -> dict[str, Any] | None:
        """Load the persisted self-model for a session, if present and valid."""
        ...

    def latest_trace_for_session(self, session_id: str) -> dict[str, Any] | None:
        """Return the most recent persisted trace for a session, if one exists."""
        ...

    def list_traces(self, *, session_id: str | None = None, limit: int | None = 20) -> list[dict[str, Any]]:
        """List persisted traces, optionally filtered by session and capped by limit."""
        ...

    def append_reflection_episode(self, session_id: str, report: Mapping[str, Any]) -> Path:
        """Append a reflection report for a session and return the created file path."""
        ...

    def list_reflection_episodes(self, session_id: str, *, limit: int | None = 20) -> list[dict[str, Any]]:
        """List persisted reflection episodes for a session, optionally capped by limit."""
        ...

    def prune_reflection_episodes(self, session_id: str, *, max_episodes: int) -> int:
        """Delete the oldest reflection episodes beyond max_episodes and return the prune count."""
        ...

    def build_regression_metrics(
        self,
        *,
        traces: Sequence[Mapping[str, Any]] | None = None,
        session_id: str | None = None,
        limit: int | None = 20,
    ) -> dict[str, Any]:
        """Compute regression metrics from provided traces or from persisted traces on disk."""
        ...

    def load_task_queue(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Load the persisted task queue for a session, optionally limited to the newest tasks."""
        ...

    def persist_task_queue(
        self,
        session_id: str,
        tasks: Sequence[Mapping[str, Any] | ScheduledCognitiveTask],
    ) -> Path:
        """Overwrite the persisted task queue for a session and return the queue file path."""
        ...

    def latest_trace(self) -> dict[str, Any] | None:
        """Return the most recent persisted trace across all sessions, if available."""
        ...
