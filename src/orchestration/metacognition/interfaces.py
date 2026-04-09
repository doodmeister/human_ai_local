from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

from .models import (
    CognitiveActProposal,
    CognitiveGoal,
    CognitivePlan,
    CriticReport,
    ExecutionResult,
    InternalStateSnapshot,
    ScheduledCognitiveTask,
    SelfModel,
    Stimulus,
    WorkspaceState,
)

EventHandler = Callable[[Mapping[str, Any]], None]


class StateProvider(Protocol):
    def snapshot(self, stimulus: Stimulus) -> InternalStateSnapshot:
        ...


class WorkspaceBuilder(Protocol):
    def build(self, stimulus: Stimulus, snapshot: InternalStateSnapshot) -> WorkspaceState:
        ...

    def contradictions_for_session(self, session_id: str) -> list[dict[str, Any]]:
        ...


class GoalManager(Protocol):
    def rank_goals(self, workspace: WorkspaceState) -> list[CognitiveGoal]:
        ...


class PolicyEngine(Protocol):
    def propose_acts(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
    ) -> list[CognitiveActProposal]:
        ...

    def select_plan(
        self,
        workspace: WorkspaceState,
        ranked_goals: list[CognitiveGoal],
        proposals: list[CognitiveActProposal],
    ) -> CognitivePlan:
        ...


class PlanExecutor(Protocol):
    def execute(self, workspace: WorkspaceState, plan: CognitivePlan) -> ExecutionResult:
        ...


class Critic(Protocol):
    def evaluate(
        self,
        workspace: WorkspaceState,
        plan: CognitivePlan,
        result: ExecutionResult,
    ) -> CriticReport:
        ...


class SelfModelUpdater(Protocol):
    def apply(self, current: SelfModel, report: CriticReport) -> SelfModel:
        ...


class CognitiveScheduler(Protocol):
    def schedule(
        self,
        workspace: WorkspaceState,
        report: CriticReport,
    ) -> list[ScheduledCognitiveTask]:
        ...


class EventBus(Protocol):
    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        ...

    def publish(self, event_name: str, payload: Mapping[str, Any]) -> None:
        ...


class CycleTracer(Protocol):
    def write_trace(self, trace: Mapping[str, Any]) -> str:
        ...

    def persist_self_model(self, session_id: str, self_model: Mapping[str, Any] | SelfModel) -> Any:
        ...

    def upsert_task_queue(
        self,
        session_id: str,
        tasks: Sequence[Mapping[str, Any] | ScheduledCognitiveTask],
    ) -> Any:
        ...

    def load_self_model(self, session_id: str) -> dict[str, Any] | None:
        ...

    def latest_trace_for_session(self, session_id: str) -> dict[str, Any] | None:
        ...

    def list_traces(self, *, session_id: str | None = None, limit: int | None = 20) -> list[dict[str, Any]]:
        ...

    def append_reflection_episode(self, session_id: str, report: Mapping[str, Any]) -> Any:
        ...

    def list_reflection_episodes(self, session_id: str, *, limit: int | None = 20) -> list[dict[str, Any]]:
        ...

    def prune_reflection_episodes(self, session_id: str, *, max_episodes: int) -> int:
        ...

    def build_regression_metrics(self, *, session_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        ...

    def load_task_queue(self, session_id: str) -> list[dict[str, Any]]:
        ...

    def persist_task_queue(
        self,
        session_id: str,
        tasks: Sequence[Mapping[str, Any] | ScheduledCognitiveTask],
    ) -> Any:
        ...

    def latest_trace(self) -> dict[str, Any] | None:
        ...
