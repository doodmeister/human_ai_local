from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

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
