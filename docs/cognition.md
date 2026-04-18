Target architecture

Implementation status as of 2026-04-09

- [x] Phase 0 safety rails landed
- [x] Phase 1 types and seams landed
- [x] Phase 2 state and workspace adapters landed
- [x] Phase 3 goals and policy
- [x] Phase 4 execution loop
- [x] Phase 5 runtime integration
- [x] Phase 6 persistence and observability
- [x] Phase 7 background cognition

Add a new subsystem:

src/orchestration/metacognition/
- [x] __init__.py
- [x] interfaces.py
- [x] models.py
- [x] enums.py
- [x] controller.py
- [x] workspace_builder.py
- [x] state_provider.py
- [x] goal_manager.py
- [x] policy_engine.py
- [x] executor.py
- [x] critic.py
- [x] self_model_updater.py
- [x] scheduler.py
- [x] event_bus.py
- [x] cycle_tracer.py
- [x] presenters.py

How it fits:

User input
  -> ChatService facade
  -> MetacognitiveController.run_cycle()
      -> StateProvider snapshot
      -> WorkspaceBuilder
      -> GoalManager rank/prioritize
      -> PolicyEngine propose/select cognitive acts
      -> Executor dispatch to memory/attention/chat/executive
      -> Critic evaluate result
      -> SelfModelUpdater apply evidence
      -> Scheduler enqueue follow-up cognitive work
      -> CycleTracer persist trace
  -> ChatService returns response

Division of responsibilities:

ChatService: ingress/egress, request normalization, final response packaging
MetacognitiveController: cognition loop coordinator
AttentionManager / AttentionMechanism: salience/focus gating
MemorySystem: storage/retrieval services
ExecutiveSystem: goal planning/scheduling where needed
CognitiveAgent: top-level facade that exposes process_input() and status

This keeps the current architecture intact while adding the missing “mind loop.”

2. Core design principles for the coding agent

The AI coding agent should follow these hard rules:

create interfaces before implementations
create data models before service logic
never let routers or UI talk to underscore-prefixed internals
every new public method gets a contract test
every phase ends with one integration test proving the new seam works
no monolithic “smart” class over 500 lines
no implicit dict payloads across module boundaries; use dataclasses or pydantic models
no cross-module side effects without an event or explicit returned command

That matters because your repo already identified monolith drift, split composition paths, and private-field leakage as real architectural problems.

3. New core models

Create src/orchestration/metacognition/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(slots=True)
class Stimulus:
    session_id: str
    user_input: str
    source: Literal["chat", "api", "system", "scheduled_task"] = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class InternalStateSnapshot:
    session_id: str
    active_goals: list["CognitiveGoal"]
    cognitive_load: float
    fatigue: float
    attention_focus: list[dict[str, Any]]
    unresolved_conflicts: list[dict[str, Any]]
    self_model: "SelfModel"
    pending_tasks: list["ScheduledCognitiveTask"]
    memory_summary: dict[str, Any]
    captured_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class WorkspaceState:
    cycle_id: str
    session_id: str
    stimulus: Stimulus
    internal_state: InternalStateSnapshot
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    dominant_goal: "CognitiveGoal | None" = None
    candidate_goals: list["CognitiveGoal"] = field(default_factory=list)
    candidate_acts: list["CognitiveActProposal"] = field(default_factory=list)
    selected_plan: "CognitivePlan | None" = None
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class CognitiveGoal:
    goal_id: str
    goal_type: str
    description: str
    priority: float
    urgency: float
    expected_value: float
    uncertainty: float
    fatigue_cost: float
    status: Literal["active", "blocked", "deferred", "completed", "cancelled"]
    source: str = "system"
    blockers: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CognitiveActProposal:
    act_type: str
    rationale: str
    confidence: float
    estimated_cost: float
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CognitivePlan:
    plan_id: str
    session_id: str
    acts: list[CognitiveActProposal]
    selected_goal_id: str | None
    policy_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionResult:
    cycle_id: str
    success: bool
    response_text: str | None
    executed_acts: list[str]
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CriticReport:
    cycle_id: str
    outcome_score: float
    retrieval_quality: float
    contradiction_delta: float
    goal_progress_delta: float
    failures: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelfModel:
    identity_claims: dict[str, float] = field(default_factory=dict)
    stable_preferences: dict[str, float] = field(default_factory=dict)
    capability_estimates: dict[str, float] = field(default_factory=dict)
    blind_spots: list[str] = field(default_factory=list)
    recurring_failures: list[str] = field(default_factory=list)
    source_trust: dict[str, float] = field(default_factory=dict)
    active_roles: list[str] = field(default_factory=list)
    narrative_summary: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ScheduledCognitiveTask:
    task_id: str
    task_type: str
    due_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    status: Literal["pending", "running", "done", "failed", "cancelled"] = "pending"

Why this is coding-agent-friendly:

typed
small
explicit
stable contracts for tests and refactors
4. New interfaces

Create src/orchestration/metacognition/interfaces.py

from __future__ import annotations

from typing import Protocol

from .models import (
    CriticReport,
    ExecutionResult,
    InternalStateSnapshot,
    SelfModel,
    Stimulus,
    WorkspaceState,
    CognitiveGoal,
    CognitivePlan,
    ScheduledCognitiveTask,
)


class StateProvider(Protocol):
    def snapshot(self, session_id: str) -> InternalStateSnapshot: ...


class WorkspaceBuilder(Protocol):
    def build(self, stimulus: Stimulus, state: InternalStateSnapshot) -> WorkspaceState: ...


class GoalManager(Protocol):
    def get_active_goals(self, session_id: str) -> list[CognitiveGoal]: ...
    def rank_goals(self, goals: list[CognitiveGoal], workspace: WorkspaceState) -> list[CognitiveGoal]: ...


class PolicyEngine(Protocol):
    def propose(self, workspace: WorkspaceState) -> list: ...
    def select(self, workspace: WorkspaceState) -> CognitivePlan: ...


class PlanExecutor(Protocol):
    def execute(self, workspace: WorkspaceState, plan: CognitivePlan) -> ExecutionResult: ...


class Critic(Protocol):
    def evaluate(self, workspace: WorkspaceState, result: ExecutionResult) -> CriticReport: ...


class SelfModelUpdater(Protocol):
    def apply(self, session_id: str, current: SelfModel, critic: CriticReport, result: ExecutionResult) -> SelfModel: ...


class CognitiveScheduler(Protocol):
    def schedule_followups(self, session_id: str, workspace: WorkspaceState, critic: CriticReport) -> list[ScheduledCognitiveTask]: ...


class EventBus(Protocol):
    def publish(self, event_type: str, payload: dict) -> None: ...

This lets the coding agent build each service behind a stable seam.

5. Controller implementation

Create src/orchestration/metacognition/controller.py

from __future__ import annotations

import uuid

from .models import Stimulus, WorkspaceState, ExecutionResult, CriticReport
from .interfaces import (
    StateProvider,
    WorkspaceBuilder,
    GoalManager,
    PolicyEngine,
    PlanExecutor,
    Critic,
    SelfModelUpdater,
    CognitiveScheduler,
    EventBus,
)


class MetacognitiveController:
    def __init__(
        self,
        state_provider: StateProvider,
        workspace_builder: WorkspaceBuilder,
        goal_manager: GoalManager,
        policy_engine: PolicyEngine,
        executor: PlanExecutor,
        critic: Critic,
        self_model_updater: SelfModelUpdater,
        scheduler: CognitiveScheduler,
        event_bus: EventBus,
        cycle_tracer,
    ) -> None:
        self._state_provider = state_provider
        self._workspace_builder = workspace_builder
        self._goal_manager = goal_manager
        self._policy_engine = policy_engine
        self._executor = executor
        self._critic = critic
        self._self_model_updater = self_model_updater
        self._scheduler = scheduler
        self._event_bus = event_bus
        self._cycle_tracer = cycle_tracer

    def run_cycle(self, stimulus: Stimulus) -> tuple[WorkspaceState, ExecutionResult, CriticReport]:
        state = self._state_provider.snapshot(stimulus.session_id)
        workspace = self._workspace_builder.build(stimulus, state)
        workspace.cycle_id = str(uuid.uuid4())

        goals = self._goal_manager.get_active_goals(stimulus.session_id)
        ranked_goals = self._goal_manager.rank_goals(goals, workspace)
        workspace.candidate_goals = ranked_goals
        workspace.dominant_goal = ranked_goals[0] if ranked_goals else None

        proposals = self._policy_engine.propose(workspace)
        workspace.candidate_acts = proposals
        plan = self._policy_engine.select(workspace)
        workspace.selected_plan = plan

        self._event_bus.publish("cognition.cycle.started", {
            "cycle_id": workspace.cycle_id,
            "session_id": stimulus.session_id,
        })

        result = self._executor.execute(workspace, plan)
        critic_report = self._critic.evaluate(workspace, result)

        updated_self_model = self._self_model_updater.apply(
            session_id=stimulus.session_id,
            current=state.self_model,
            critic=critic_report,
            result=result,
        )

        scheduled = self._scheduler.schedule_followups(
            session_id=stimulus.session_id,
            workspace=workspace,
            critic=critic_report,
        )

        self._cycle_tracer.record(
            workspace=workspace,
            result=result,
            critic=critic_report,
            self_model=updated_self_model,
            scheduled=scheduled,
        )

        self._event_bus.publish("cognition.cycle.completed", {
            "cycle_id": workspace.cycle_id,
            "success": result.success,
            "outcome_score": critic_report.outcome_score,
        })

        return workspace, result, critic_report

This becomes the missing cognitive spine.

6. Minimal first implementations

The first version should be simple, deterministic, and easy for an agent to code correctly.

state_provider.py

Responsibilities:

query MemorySystem.get_status()
ask attention for current focus/cognitive load/fatigue
load active goals from executive integration
load self-model from a small persisted JSON or semantic memory namespace
load due/pending cognitive tasks
workspace_builder.py

Responsibilities:

retrieve top memories for stimulus using MemorySystem.get_context_for_query() or equivalent
pull current attention focus snapshot
attach unresolved contradiction set summary
compress into WorkspaceState
goal_manager.py

Responsibilities:

retrieve current goals
compute score:
priority + urgency + expected_value + conflict_pressure - fatigue_cost - load_penalty
return ranked list
policy_engine.py

First version should not be fancy. It should choose from a short menu:

respond_to_user
inspect_conflict
store_episode
revise_belief
schedule_followup
defer
executor.py

Responsibilities:

call attention allocate
invoke retrieval if plan says retrieve
call semantic belief revision if needed
call episodic/semantic memory storage
call ChatService or a response composer for final text
return ExecutionResult
critic.py

Evaluate:

was a response produced
how many relevant memories were used
contradiction count before/after
whether a dominant goal progressed
whether the system deferred under uncertainty properly
self_model_updater.py

First version:

update capability estimates by simple moving average
add recurring failure if same failure repeats N times
update source trust if user corrections repeatedly override inferred facts
refresh narrative summary lazily, not every turn
scheduler.py

First version schedules only:

contradiction review if contradictions remain
self-model review after repeated failures
memory consolidation after heavy interaction
deferred follow-up if uncertainty high
7. Event bus and traceability

Create event_bus.py

from __future__ import annotations

from collections import defaultdict
from typing import Callable

class InProcessEventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[dict], None]) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event_type: str, payload: dict) -> None:
        for handler in self._handlers.get(event_type, []):
            handler(payload)

Create cycle_tracer.py

writes one JSON trace per cycle under data/cognition_traces/
optionally mirrors summary into episodic memory as a reflection episode

A coding agent can build this fast and test it easily.

8. Reflection episodes as a memory type

Add a new memory concept without destabilizing core memory first:

not a whole new storage backend initially
store as episodic entries with episode_type="reflection"
tags: ["reflection", "critic", "self_model_update"]

Example payload:

{
  "cycle_id": "...",
  "dominant_goal": "...",
  "selected_acts": [...],
  "outcome_score": 0.62,
  "failures": [...],
  "recommendations": [...],
}

Why this is high leverage:

gives future self-model updates real evidence
supports longitudinal introspection
creates data for dashboards and tuning
9. Integrating with the current repo
src/orchestration/runtime/app_container.py

Add builders:

build_metacognitive_controller()
get_metacognitive_controller()
src/orchestration/cognitive_agent.py

Keep facade stable. Add:

def process_input(self, user_input: str, session_id: str) -> dict:
    stimulus = Stimulus(session_id=session_id, user_input=user_input)
    workspace, result, critic = self._metacognitive_controller.run_cycle(stimulus)
    return {
        "response": result.response_text,
        "cycle_id": workspace.cycle_id,
        "goal": workspace.dominant_goal.description if workspace.dominant_goal else None,
        "critic": {
            "outcome_score": critic.outcome_score,
            "failures": critic.failures,
        },
    }
src/orchestration/chat/chat_service.py

Refactor so process_user_message() delegates into CognitiveAgent.process_input() instead of owning the whole turn pipeline itself.

src/interfaces/api/

Add endpoints:

GET /agent/metacog/status
GET /agent/metacog/background
GET /agent/metacog/scorecard
GET /agent/metacog/dashboard
GET /agent/metacog/last-cycle
GET /agent/metacog/goals
GET /agent/metacog/self-model
GET /agent/metacog/tasks

Do not expose internals directly; go through facade methods.

10. File-by-file implementation order for an AI coding agent

This is the best coding order.

Phase 0: Safety rails

- [x] Add contract tests for existing ChatService.process_user_message
- [x] Add contract tests for CognitiveAgent.process_input
- [x] Add one characterization test for MemorySystem.get_context_for_query
- [x] Add one characterization test for attention status/focus snapshot

Deliverable:

- [x] existing behavior pinned before refactor
Phase 1: Types and seams

- [x] Create metacognition/models.py
- [x] Create metacognition/interfaces.py
- [x] Create metacognition/event_bus.py
- [x] Create metacognition/cycle_tracer.py

Deliverable:

- [x] no behavior change, just scaffolding
Phase 2: State and workspace

- [x] Implement state_provider.py
- [x] Implement workspace_builder.py
- [x] Add tests for snapshot shape and workspace shape

Deliverable:

- [x] can build a deterministic workspace from a stimulus
Phase 3: Goals and policy

- [x] Implement goal_manager.py
- [x] Implement policy_engine.py
- [x] Add scoring tests
- [x] Add selection tests

Deliverable:

- [x] dominant goal and selected acts are deterministic
Phase 4: Execution loop

- [x] Implement executor.py
- [x] Implement critic.py
- [x] Implement self_model_updater.py
- [x] Implement scheduler.py
- [x] Implement controller.py

Deliverable:

- [x] end-to-end closed loop works in-memory
Phase 5: Runtime integration

- [x] Wire controller in app_container.py
- [x] Update cognitive_agent.py
- [x] Update chat_service.py delegation path
- [x] Add integration tests

Deliverable:

- [x] normal chat path now runs through metacognitive loop
Phase 6: Persistence and observability

- [x] Persist self-model
- [x] Persist reflection episodes
- [x] Persist cycle traces
- [x] Add status endpoints
- [x] Add regression metrics

Deliverable:

- [x] inspectable, replayable cognition traces
Phase 7: Background cognition

- [x] Add periodic scheduler tick
- [x] Execute due cognitive tasks
- [x] Add idle reflection
- [x] Add contradiction audits

Deliverable:

- [x] cognition continues across time, not only turns
11. Prompt-ready task slices for an AI coding agent

These are good exact tasks to hand to a coding agent.

Task 1

Create src/orchestration/metacognition/models.py with dataclasses for:
Stimulus, InternalStateSnapshot, WorkspaceState, CognitiveGoal, CognitiveActProposal, CognitivePlan, ExecutionResult, CriticReport, SelfModel, ScheduledCognitiveTask.
Use slots=True, type hints, and no external dependencies.

- [x] Implemented

Task 2

Create src/orchestration/metacognition/interfaces.py using typing.Protocol for:
StateProvider, WorkspaceBuilder, GoalManager, PolicyEngine, PlanExecutor, Critic, SelfModelUpdater, CognitiveScheduler, EventBus.

- [x] Implemented

Task 3

Implement InProcessEventBus in event_bus.py and add unit tests verifying:

- [x] subscriber registration
- [x] multiple handlers per event
- [x] no crash on unhandled event

- [x] Implemented
Task 4

Implement FilesystemCycleTracer in cycle_tracer.py that writes JSON traces into data/cognition_traces/.
Add tests using a temp directory.

- [x] Implemented

Task 5

Implement DefaultStateProvider that composes:

- [x] MemorySystem
- [x] attention facade/status provider
- [x] executive integration
- [x] self-model store
- [x] task store
- [x] Return InternalStateSnapshot.

- [x] Implemented
Task 6

Implement DefaultWorkspaceBuilder that:

- [x] creates WorkspaceState
- [x] retrieves top memory context
- [x] copies attention focus and contradictions
- [x] includes notes on uncertainty/load
- [x] Add deterministic tests with stubbed dependencies.

- [x] Implemented
Task 7

Implement HeuristicGoalManager with weighted scoring.
Expose weights as constructor params.
Add tests for ranking order.

- [x] Implemented

Task 8

Implement HeuristicPolicyEngine that proposes acts based on:

contradiction pressure
uncertainty
cognitive load
presence/absence of dominant goal
Then selects a simple plan.
Add tests.

- [x] Implemented
Task 9

Implement DefaultPlanExecutor that:

allocates attention
optionally retrieves/store memories
optionally triggers semantic revision
creates a response via existing chat/response component
Return ExecutionResult.

- [x] Implemented
Task 10

Implement DefaultCritic, DefaultSelfModelUpdater, and DefaultCognitiveScheduler.
Keep version 1 deterministic and heuristic.
Add tests.

- [x] Implemented

Task 11

Implement MetacognitiveController and wire it into app_container.py.
Add one integration test proving:
input -> workspace -> plan -> execution -> critic -> trace

- [x] Implement MetacognitiveController
- [x] Wire it into app_container.py
- [x] Add one integration test proving input -> workspace -> plan -> execution -> critic -> trace

Task 12

Refactor CognitiveAgent and ChatService to delegate into the controller while preserving payload shape.
Add regression tests.

- [x] Implemented

12. Suggested package boundaries

Keep each file narrow:

controller.py: orchestration only
workspace_builder.py: no side effects beyond retrieval
goal_manager.py: scoring only
policy_engine.py: proposal/selection only
executor.py: action dispatch only
critic.py: evaluation only
self_model_updater.py: state updates only
scheduler.py: deferred task creation only

That matters because your repo has already suffered from oversized facade classes and mixed responsibilities.

13. Recommended first public facade methods

Add these methods to CognitiveAgent:

process_input(user_input, session_id)
get_metacognitive_status(session_id)
get_last_cycle_summary(session_id)
get_self_model(session_id)
list_cognitive_tasks(session_id)
get_active_goals(session_id)

Add these methods to MemorySystem if missing:

store_reflection_episode(...)
get_unresolved_conflicts(session_id)
get_context_for_query(query, session_id)

Add these methods to attention facade if missing:

get_focus_snapshot(session_id=None)
get_cognitive_metrics()
14. Test strategy

For an AI coding agent, test shape matters more than perfect sophistication at first.

Unit tests
model construction
goal ranking
act proposal
critic scoring
self-model update rules
scheduler task creation
Contract tests
CognitiveAgent.process_input() response shape
ChatService.process_user_message() response shape unchanged
GET /agent/metacog/status shape
GET /agent/metacog/background shape
Integration tests
one end-to-end turn
one contradiction turn that triggers inspect_conflict
one high-load turn that triggers defer
one repeated failure pattern that updates self-model
Nonfunctional tests
cycle trace file is written
no private field reads from API routes
no crash if no goals exist
no crash if retrieval returns nothing
15. What not to do in v1

Do not let the coding agent:

build an LLM-driven planner first
rewrite memory architecture first
add distributed queues first
create a giant “brain.py”
store self-model only as an unstructured prompt blob
let ChatService and MetacognitiveController both orchestrate turns

That would recreate the same architectural drift you just finished cleaning up.

16. Concrete phased refactor plan
Phase A: Add metacognition scaffolding

- [x] models, interfaces, event bus, and tracer landed
- [x] success condition met: no production behavior change

Goal:

land models, interfaces, event bus, tracer
Risk:
low
Success condition:
no production behavior change
Phase B: Build deterministic cognition loop

- [x] state snapshot adapter landed
- [x] workspace builder landed
- [x] goals layer
- [x] policy layer
- [x] controller
- [x] success condition met: closed loop runs in unit/integration tests

Goal:

state snapshot, workspace, goals, policy, controller
Risk:
medium
Success condition:
closed loop runs in unit/integration tests
Phase C: Route chat through the controller

- [x] ChatService delegates through the metacognitive controller sidecar
- [x] CognitiveAgent records metacognitive cycles through the controller sidecar
- [x] success condition met: chat payload compatibility preserved in focused regression tests

Goal:

make ChatService delegate to CognitiveAgent, and CognitiveAgent delegate to the metacognitive controller
Risk:
medium-high
Success condition:
chat payload compatibility preserved
Phase D: Persist introspective state

- [x] completed

Goal:

self-model persistence, reflection episodes, cycle traces
Risk:
medium
Success condition:
continuity across sessions demonstrable
Phase E: Add background cognition

- [x] completed

Goal:

scheduled contradiction review, self-model review, consolidation jobs
Risk:
medium-high
Success condition:
system performs cognitive work without a direct user turn
Phase F: Add dashboards and scorecards

- [x] completed
- [x] Streamlit dashboard exposes metacognition status, background state, scorecards, tasks, and reflections
- [x] Chainlit /metacog command exposes the same metacognition inspection surface in the chat-first UI

Goal:

expose cycle success, contradiction rates, self-model drift, goal churn
Risk:
low
Success condition:
easier tuning and regression detection
17. Highest-leverage first commit set

If I were batching the very first engineering move, I’d make exactly these commits:

- [x] feat(metacognition): add typed core models and service protocols
- [x] feat(metacognition): add in-process event bus and filesystem cycle tracer
- [x] feat(metacognition): add default state provider and workspace builder
- [x] feat(metacognition): add heuristic goal manager and policy engine
- [x] feat(metacognition): add controller, executor, critic, self-model updater, and scheduler
- [x] refactor(runtime): wire metacognitive controller into app container
- [x] refactor(agent): delegate process_input through metacognitive controller
- [x] refactor(chat): route process_user_message through cognitive agent facade
- [x] feat(api): add metacognitive status, self-model, task, reflection, and background endpoints
- [x] test(metacognition): add closed-loop contract and integration tests

That sequence is easy for an AI coding agent to follow and review.

18. Bottom line

The missing piece is not more memory sophistication first. It is a small, explicit, testable recurrent controller.

Your repo already has the right substrate:

multi-store memory
attention gating
persistence
runtime composition
public facades
a refactor path aimed at truthful boundaries

The blueprint above adds the missing spine in a way that is:

incremental
compatible with your current architecture
observable
testable
coding-agent friendly
