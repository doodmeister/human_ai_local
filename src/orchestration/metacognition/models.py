from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import CognitiveActType, GoalKind, ScheduledTaskStatus


@dataclass(slots=True)
class Stimulus:
    session_id: str
    user_input: str
    input_type: str = "text"
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CognitiveGoal:
    goal_id: str
    description: str
    priority: float
    kind: GoalKind = GoalKind.RESPONSE
    urgency: float = 0.0
    salience: float = 0.0
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CognitiveActProposal:
    act_type: CognitiveActType
    description: str
    priority_score: float
    rationale: str = ""
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelfModel:
    session_id: str
    confidence: float = 0.5
    traits: dict[str, float] = field(default_factory=dict)
    beliefs: tuple[str, ...] = field(default_factory=tuple)
    recent_updates: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScheduledCognitiveTask:
    task_id: str
    description: str
    status: ScheduledTaskStatus = ScheduledTaskStatus.PENDING
    priority: float = 0.0
    due_at: float | None = None
    goal_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InternalStateSnapshot:
    session_id: str
    turn_index: int
    memory_status: dict[str, Any] = field(default_factory=dict)
    attention_status: dict[str, Any] = field(default_factory=dict)
    active_goals: tuple[CognitiveGoal, ...] = field(default_factory=tuple)
    self_model: SelfModel = field(default_factory=lambda: SelfModel(session_id="default"))
    pending_tasks: tuple[ScheduledCognitiveTask, ...] = field(default_factory=tuple)
    cognitive_load: float = 0.0
    uncertainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkspaceState:
    stimulus: Stimulus
    snapshot: InternalStateSnapshot
    context_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    focus_items: tuple[Any, ...] = field(default_factory=tuple)
    contradictions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    dominant_goal: CognitiveGoal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CognitivePlan:
    selected_goal: CognitiveGoal | None
    acts: tuple[CognitiveActProposal, ...] = field(default_factory=tuple)
    policy_name: str = "deterministic"
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    response_text: str = ""
    executed_acts: tuple[CognitiveActProposal, ...] = field(default_factory=tuple)
    memory_updates: tuple[str, ...] = field(default_factory=tuple)
    attention_updates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticReport:
    cycle_id: str
    success_score: float
    goal_progress: float = 0.0
    follow_up_recommended: bool = False
    contradictions_detected: tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetacognitiveCycleResult:
    cycle_id: str
    workspace: WorkspaceState
    ranked_goals: tuple[CognitiveGoal, ...] = field(default_factory=tuple)
    plan: CognitivePlan | None = None
    execution_result: ExecutionResult | None = None
    critic_report: CriticReport | None = None
    updated_self_model: SelfModel | None = None
    scheduled_tasks: tuple[ScheduledCognitiveTask, ...] = field(default_factory=tuple)
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
