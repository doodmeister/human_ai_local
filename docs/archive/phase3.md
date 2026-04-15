# Phase 3 Refactor Plan

Last updated: 2026-03-28

## Purpose

This document is the working plan for Phase 3: architecture consolidation and refactoring.

It has four jobs:

1. Record the deeper architecture review.
2. Identify the best extraction boundaries.
3. Turn those boundaries into a concrete refactor sequence.
4. Serve as the checklist we update as work is completed.

## Executive Summary

The project is beyond prototype stage. It already has a real platform shape:

- A single top-level entrypoint in `main.py`
- A central runtime in `src/orchestration/cognitive_agent.py`
- A chat orchestration layer in `src/orchestration/chat/`
- A unified memory facade in `src/memory/memory_system.py`
- An executive pipeline in `src/executive/integration.py`
- Mounted FastAPI routers in `src/interfaces/api/`

The primary Phase 3 problem is not missing capability. The primary problem is architectural drift:

- Newer orchestration code coexists with legacy bootstrapping.
- Several large classes now own too many responsibilities.
- API/runtime boundaries sometimes cross layers or reach into private state.
- Public docs, roadmap docs, and default test execution no longer tell the same story.

Phase 3 should therefore focus on consolidation, extraction, and truthfulness of boundaries rather than adding major new cognitive features.

## Current State Review

The sections in this review capture the baseline state observed before the Phase 3 extraction work landed. The later phase checklists and final architecture summary record the current post-refactor state.

## Current Test Lanes

These lanes reflect the current repository truth after Phase 3 reconciliation.

### Lane: `default`

- Scope: `tests/`
- Current default via `pytest.ini`
- Purpose: maintained regression coverage for the repository's active test suite
- Command: `python -m pytest -q`

### Lane: `targeted`

- Scope: explicit files or populated subdirectories under `tests/`
- Purpose: focused validation while refactoring a subsystem or endpoint
- Commands: `python -m pytest tests/executive -q`, `python -m pytest tests/unit -q`, or a specific file under `tests/`

### Lane: `legacy-manual`

- Scope: `archived_tests/manual_legacy/`
- Purpose: intentional validation of older or historical paths outside the maintained default suite
- Command: invoke the relevant subdirectory under `archived_tests/manual_legacy/`

## Current Startup Paths

These were the startup paths before Phase 3 extraction.

### Canonical user-facing entrypoint

- `main.py`

### Current startup flow

1. CLI path:
   - `main.py chat`
   - now resolves the shared agent through `src/orchestration/runtime/app_container.py`

2. API path:
   - `main.py api`
   - builds app in `main.py`
   - currently starts from legacy `scripts.legacy.george_api_simple.app`
   - mounts current routers from `src/interfaces/api/`

3. Chat API runtime path:
   - routers call `src/orchestration/chat/api_runtime.py`
   - that module lazily builds shared agent/chat instances

4. Secondary factory path:
   - `src/orchestration/chat/factory.py`
   - can independently construct `ChatService`

### Why this matters

There are currently at least three partially overlapping composition paths. Phase 3B should collapse them into one runtime container.

## Current Singleton Ownership

This was the effective ownership model before refactor.

### Agent singleton

- Owner today: `src/orchestration/agent_singleton.py`
- Behavior: lazy global `CognitiveAgent` creation

### Chat runtime singleton

- Owner today: `src/orchestration/chat/api_runtime.py`
- Behavior: lazy module-level `SessionManager`, `ContextBuilder`, and `ChatService`

### Prospective reminder singleton

- Owner today: in-memory prospective memory helper used via `api_runtime.py` and chat code

### Risk

Singleton ownership is currently distributed across multiple modules rather than a single composition root. That is acceptable for now, but it is the first structural issue to remove.

### What is working well

- Subsystems are already separated at a coarse level: memory, executive, orchestration, interfaces, cognition.
- The runtime is feature-complete enough to support chat, memory retrieval, reminders, planning, scheduling, and layered cognition.
- The codebase already uses lazy initialization and fallback behavior in several places, which reduces startup fragility.
- The repository has evidence of real hardening work: contracts, status endpoints, telemetry, and resilience utilities.

### What is not working well

#### 1. Composition is split across too many places

Current composition is spread across:

- `main.py`
- `src/orchestration/agent_singleton.py`
- `src/orchestration/chat/factory.py`
- `src/orchestration/chat/api_runtime.py`
- `src/interfaces/api/*.py`

This makes it harder to answer basic questions like:

- What is the canonical way to build the runtime?
- Which objects are singletons versus request-scoped?
- Which dependencies are optional versus required?
- Which startup path is authoritative for CLI, API, and UI?

#### 2. ChatService has become the largest orchestration monolith

`src/orchestration/chat/chat_service.py` currently mixes:

- Session handling
- Turn pipeline orchestration
- Intent classification
- Goal detection and goal orchestration
- Memory query handling
- Reminder CRUD behavior
- Performance reporting
- System status summarization
- Consolidation heuristics
- Metacognitive state tracking
- Phase 2 layer runtime state for drives, felt sense, relational field, patterns, self-model, and narrative

The file is functionally grouped already, which is good. But those groups are still trapped in one class.

#### 3. MemorySystem is doing too much at too many levels

`src/memory/memory_system.py` currently owns:

- Backend initialization
- Storage routing
- Session tracking
- Search orchestration
- Consolidation scheduling
- Dream-state consolidation bridge
- Semantic fact CRUD
- Context retrieval
- Episodic helpers
- Prospective reminder helpers
- Proactive recall and summarization
- Status and lifecycle management

This is a facade, but it is also an implementation bucket. That makes it the second major extraction target.

#### 4. CognitiveAgent is both runtime container and behavior surface

`src/orchestration/cognitive_agent.py` currently owns:

- Component initialization
- Cognitive loop execution
- LLM conversation state
- Fact convenience methods
- Cognitive state updates
- Reflection lifecycle and background scheduling
- Dream processing accessors
- Status reporting

This file wants to be at least three things:

- Runtime bootstrap/container
- Turn processor
- Reflection and maintenance service

#### 5. Executive integration is mostly good, but still mixing stages and side effects

`src/executive/integration.py` is in better shape than the chat and memory layers. It already models a pipeline. But it still mixes:

- Goal creation compatibility logic
- Decision orchestration
- Plan creation
- Schedule generation
- Reminder projection side effects
- Health reporting
- Outcome tracking access

This is a smaller refactor than ChatService or MemorySystem, but it should still be cleaned up.

#### 6. Interface truthfulness is uneven

Examples:

- `main.py` still builds the API from `scripts.legacy.george_api_simple`.
- `src/interfaces/api/executive_api.py` still describes itself as simplified and returns some simulated data.
- `src/interfaces/api/chat_endpoints.py` reads internal fields like `_last_metacog_snapshot` and `_metacog_history` directly.

This creates maintenance drag and weakens contracts.

#### 7. Documentation and test signals are out of sync

- `pytest.ini` defaults to `tests/contracts` only.
- Historical docs report much larger test counts and broader completion claims.
- README examples still reference older import paths.

This does not mean the code is weak. It means the repo lacks one authoritative current-state narrative.

## Best Extraction Boundaries

The following are the recommended extraction boundaries. They are ordered by architectural value, not by implementation difficulty.

### Boundary 1: Runtime Composition Root

#### Why this is the best first boundary

Everything else becomes safer once there is one canonical place to assemble the system.

#### Current source area

- `main.py`
- `src/orchestration/agent_singleton.py`
- `src/orchestration/chat/factory.py`
- `src/orchestration/chat/api_runtime.py`

#### Target shape

Create a single runtime assembly package, for example:

```text
src/orchestration/runtime/
  app_container.py
  builders.py
  service_registry.py
  bootstrap.py
```

#### Responsibilities to move here

- Build `CognitiveAgent`
- Build `ChatService`
- Wire memory, attention, executive, prospective, and consolidator dependencies
- Define singleton ownership
- Define startup and shutdown semantics
- Remove legacy bootstrapping from interface code

#### Rules

- API routers should request services from the runtime container, not build them.
- `main.py` should call the runtime container, not perform partial assembly itself.
- There should be one canonical path for CLI, API, and UI startup.

### Boundary 2: Chat Turn Pipeline

#### Why this is the highest-value domain extraction

`ChatService` currently owns the most cross-cutting behavior in the system. Extracting it will reduce cognitive load across the whole repo.

#### Current source area

- `src/orchestration/chat/chat_service.py`

#### Target shape

```text
src/orchestration/chat/
  service.py
  turn_pipeline.py
  intent_router.py
  turn_context.py
  response_builder.py
  status_service.py
```

#### Recommended sub-boundaries

1. `turn_pipeline.py`
   - Own `process_user_message`
   - Coordinate perceive, classify, retrieve, respond, consolidate, report

2. `intent_router.py`
   - Own `_plan_intent_execution`
   - Own `_run_intent_handlers`
   - Own goal query/update, reminder request, memory query, performance query, and system status dispatch

3. `status_service.py`
   - Own `_handle_performance_query`
   - Own `_handle_system_status`
   - Own formatting helpers for latency, percentages, STM usage, due phrases, reminder summaries

4. `response_builder.py`
   - Own `_merge_intent_sections`
   - Own context item summarization and trace serialization
   - Own response assembly behavior

5. `turn_context.py`
   - Own `_build_session_context`
   - Own turn metadata packaging for UI/API consumers

#### What should stay in ChatService

- Public facade methods only
- Thin dependency holder
- Backward-compatible adapter behavior during migration

### Boundary 3: Phase 2 Cognitive Layer Runtime

#### Why this deserves its own boundary

The drives through narrative stack is conceptually distinct from core chat flow. Right now it is embedded inside `ChatService` state and helpers.

#### Current source area

- Lazy-init helpers and state in `src/orchestration/chat/chat_service.py`

#### Target shape

```text
src/orchestration/cognitive_layers/
  runtime.py
  drive_runtime.py
  felt_runtime.py
  relational_runtime.py
  pattern_runtime.py
  self_model_runtime.py
  narrative_runtime.py
```

#### Responsibilities to move here

- Layer state ownership
- Lazy initialization
- Turn-by-turn updates
- Public snapshots for telemetry and API use

#### Rule

The chat layer should ask the cognitive-layers runtime for updates and snapshots. It should not own six layer-specific state machines directly.

### Boundary 4: Memory Facade vs Memory Services

#### Why this is the second major refactor target

The current memory facade is valuable, but it has absorbed too many implementation concerns.

#### Current source area

- `src/memory/memory_system.py`

#### Target shape

```text
src/memory/
  runtime/
    initializer.py
    lifecycle.py
  services/
    storage_router.py
    retrieval_service.py
    consolidation_service.py
    context_service.py
    fact_service.py
    prospective_service.py
    recall_service.py
    status_service.py
  facade/
    memory_system.py
```

#### Recommended sub-boundaries

1. `runtime/initializer.py`
   - Subsystem construction and backend selection

2. `services/storage_router.py`
   - `store_memory`
   - `_determine_storage_system`
   - `_store_in_ltm`
   - `_store_in_stm`

3. `services/retrieval_service.py`
   - `retrieve_memory`
   - `search_memories`
   - `search_stm_semantic`
   - episodic and hierarchical retrieval logic

4. `services/context_service.py`
   - `get_context_for_query`
   - context aggregation helpers

5. `services/consolidation_service.py`
   - `consolidate_memories`
   - dream-state bridge
   - schedule checks

6. `services/fact_service.py`
   - semantic fact CRUD

7. `services/prospective_service.py`
   - prospective reminder helpers

8. `services/recall_service.py`
   - proactive recall
   - summary generation

9. `services/status_service.py`
   - status, metrics, session reset, lifecycle reporting

#### What should stay in the facade

- Stable public API
- Dependency references
- Thin delegation layer

### Boundary 5: CognitiveAgent Runtime vs Turn Processing vs Reflection

#### Why this boundary matters

`CognitiveAgent` is currently both a service container and a behavior object. That makes testing and lifecycle management harder than necessary.

#### Current source area

- `src/orchestration/cognitive_agent.py`

#### Target shape

```text
src/orchestration/agent/
  runtime.py
  turn_processor.py
  llm_session.py
  reflection_service.py
  maintenance_service.py
  status_service.py
```

#### Recommended sub-boundaries

1. `runtime.py`
   - component initialization
   - dependency wiring
   - startup/shutdown

2. `turn_processor.py`
   - `process_input`
   - sensory retrieval, attention allocation, response generation, consolidation flow

3. `llm_session.py`
   - system prompt state
   - LLM provider init and conversation buffer management

4. `reflection_service.py`
   - reflection reports
   - scheduler start/stop
   - reflection status APIs

5. `status_service.py`
   - `get_cognitive_status`
   - break/recovery reporting
   - dream statistics accessors

#### Rule

The `CognitiveAgent` class can remain as a facade for compatibility, but it should delegate almost all implementation to smaller services.

### Boundary 6: Executive Planning Pipeline Stages

#### Why this is a secondary extraction

`ExecutiveSystem` is already organized around a pipeline, so the extraction is more straightforward.

#### Current source area

- `src/executive/integration.py`

#### Target shape

```text
src/executive/integration/
  system.py
  goal_facade.py
  decision_stage.py
  planning_stage.py
  scheduling_stage.py
  reminder_projection.py
  health_service.py
```

#### Responsibilities to split

- Goal creation compatibility logic
- Decision stage orchestration
- Planning stage orchestration
- Scheduling stage orchestration
- Reminder generation side effects
- Health and metrics reporting

#### Rule

Advisor-only semantics must remain explicit. Reminder creation should stay opt-in and be separated from core planning stages.

### Boundary 7: API Interface Adapters

#### Why this boundary matters

Routers should depend on stable service methods, not internals.

#### Current source area

- `src/interfaces/api/chat_endpoints.py`
- `src/interfaces/api/executive_api.py`

#### Target shape

```text
src/interfaces/api/
  dependencies.py
  presenters/
    chat_presenter.py
    executive_presenter.py
```

#### Responsibilities to move

- Request-to-service adaptation
- Response shaping
- Internal field translation

#### Rule

No router should read private underscore-prefixed fields from core services.

## Extraction Order

This is the recommended implementation order.

1. Runtime composition root
2. Chat turn pipeline
3. Phase 2 cognitive layer runtime
4. Memory services
5. CognitiveAgent split
6. Executive integration split
7. API adapter cleanup
8. Docs and test lane reconciliation

This order reduces risk because each step creates a cleaner seam for the next one.

## Refactor Strategy

### Rules for the Phase 3 workstream

- Keep public APIs stable unless there is a strong reason to change them.
- Prefer extract-and-delegate over rename-and-break.
- Move behavior behind new modules first, then shrink old files.
- Add characterization tests before moving critical orchestration behavior.
- Do not mix major feature work into this phase.
- Keep legacy compatibility shims until callers are migrated.

### Definition of done for Phase 3

Phase 3 is complete when:

- `main.py` uses one canonical runtime container.
- `ChatService`, `MemorySystem`, and `CognitiveAgent` are facades instead of implementation monoliths.
- API routers do not rely on private service internals.
- The executive integration layer has explicit stage modules.
- README, roadmap, and test commands describe the same current system.
- The default maintained suite is documented truthfully, without claiming a curated contract-only lane that does not exist.

## Concrete Refactor Plan

## Phase 3A: Baseline and Safety Net

- [x] Create a current architecture inventory doc section in this file as modules move.
- [x] Define the supported verification lanes explicitly in docs and config.
- [x] Add characterization tests for current `ChatService` response payload shape.
- [x] Add characterization tests for current `MemorySystem` routing and status behaviors.
- [x] Add characterization tests for current `CognitiveAgent` status and reflection behaviors.
- [x] Add characterization tests for current `ExecutiveSystem.plan_goal` artifacts.
- [x] Document startup paths and singleton ownership before code movement.

### Exit criteria

- [x] We can refactor without changing external behavior accidentally.

## Phase 3B: Composition Root Consolidation

- [x] Create `src/orchestration/runtime/` package.
- [x] Introduce a single app/container builder for agent, chat service, memory, and executive wiring.
- [x] Move singleton ownership out of interface helpers into the runtime container.
- [x] Refactor `src/orchestration/chat/api_runtime.py` to delegate to the container.
- [x] Refactor `src/orchestration/chat/factory.py` to delegate to the container.
- [x] Update `main.py` to route both CLI and API startup through the container-backed runtime path.
- [x] Remove dependency on `scripts.legacy.george_api_simple` for the canonical API startup path.

### Exit criteria

- [x] There is one authoritative runtime assembly path.

## Phase 3C: ChatService Extraction

- [x] Create `turn_pipeline.py` and move turn orchestration logic behind it.
- [x] Create `intent_router.py` and move intent planning and handler dispatch there.
- [x] Create `status_service.py` and move performance/system status formatting there.
- [x] Create `response_builder.py` and move response merge and trace serialization there.
- [x] Create `turn_context.py` and move session context packaging there.
- [x] Convert `ChatService` into a thin facade over extracted collaborators.
- [x] Keep `process_user_message` public behavior unchanged.

### Exit criteria

- [x] `chat_service.py` is primarily public API plus delegation.

## Phase 3D: Phase 2 Cognitive Layer Runtime Extraction

- [x] Create `src/orchestration/cognitive_layers/` package.
- [x] Move drive runtime state and updates out of `ChatService`.
- [x] Move felt-sense runtime state and updates out of `ChatService`.
- [x] Move relational runtime state and updates out of `ChatService`.
- [x] Move pattern runtime state and updates out of `ChatService`.
- [x] Move self-model runtime state and updates out of `ChatService`.
- [x] Move narrative runtime state and updates out of `ChatService`.
- [x] Replace direct layer fields in `ChatService` with a single collaborator.

### Exit criteria

- [x] Chat orchestration no longer owns six separate layer-specific state machines.

## Phase 3E: MemorySystem Extraction

- [x] Create `src/memory/runtime/` and `src/memory/services/` packages.
- [x] Extract subsystem initialization into a dedicated initializer.
- [x] Extract storage routing into `storage_router.py`.
- [x] Extract retrieval and search into `retrieval_service.py`.
- [x] Extract context aggregation into `context_service.py`.
- [x] Extract consolidation behavior into `consolidation_service.py`.
- [x] Extract fact CRUD into `fact_service.py`.
- [x] Extract prospective helper behavior into `prospective_service.py`.
- [x] Extract proactive recall and summarization into `recall_service.py`.
- [x] Extract status and lifecycle reporting into `status_service.py`.
- [x] Extract the first retrieval and search service behind `src/memory/services/retrieval_service.py`.
- [x] Extract context aggregation behind `src/memory/services/context_service.py`.
- [x] Convert `MemorySystem` into a thin facade that delegates to these services.

### Exit criteria

- [x] `memory_system.py` is a stable facade, not the implementation bucket.

## Phase 3F: CognitiveAgent Extraction

- [x] Create `src/orchestration/agent/` package.
- [x] Move runtime initialization into `runtime.py`.
- [x] Move main turn loop into `turn_processor.py`.
- [x] Move LLM provider/session handling into `llm_session.py`.
- [x] Move reflection logic and scheduler into `reflection_service.py`.
- [x] Move maintenance and status helpers into dedicated services.
- [x] Keep `CognitiveAgent` as a backward-compatible facade.

### Exit criteria

- [x] `cognitive_agent.py` reads as a facade over composed services.

## Phase 3G: Executive Integration Extraction

- [x] Create stage modules for decision, planning, and scheduling orchestration.
- [x] Separate reminder projection from core planning stages.
- [x] Separate health and metrics reporting from the main system class.
- [x] Preserve advisor-only semantics in naming and behavior.
- [x] Migrate callers without changing plan artifact shapes.

### Exit criteria

- [x] `integration.py` becomes orchestration glue, not stage implementation.

## Phase 3H: API and Interface Cleanup

- [x] Add explicit interface dependency helpers.
- [x] Stop reading underscore-prefixed internal fields from routers.
- [x] Replace simulated executive API values with real service-backed values or clearly marked placeholders.
- [x] Align API responses with service contracts and tests.
- [x] Remove outdated “simplified” wording where it is no longer accurate.
- [x] Replace chat router metacog, dream, and LLM config internal-field access with public service methods.
- [x] Replace reflection router app-state and module-global report caches with agent-backed reflection report lifecycle methods.
- [x] Normalize canonical API routers onto shared request dependency helpers instead of ad hoc app-state access.
- [x] Replace memory API status placeholders with real memory facade status output.
- [x] Remove unmounted legacy executive API variants so only the canonical executive router remains in the interface surface.

### Exit criteria

- [x] Routers depend on stable service methods and presenters only.

## Phase 3I: Documentation and Test Reconciliation

- [x] Update README imports and architecture references to current module paths.
- [x] Write one authoritative “current architecture” section and archive conflicting status docs.
- [x] Document the supported pytest lanes and what each one guarantees.
- [x] Ensure task names, `pytest.ini`, and docs match the current verification workflow.
- [x] Record the final extracted architecture in this file.

### Final extracted architecture

- `main.py` remains the single user-facing entrypoint for `chat`, `api`, `ui`, and `chainlit` modes.
- `src/orchestration/runtime/` is the canonical composition root for shared CLI/API runtime construction and FastAPI app bootstrap.
- `src/orchestration/cognitive_agent.py` is now a stable facade over dedicated runtime, LLM-session, reflection, maintenance, and turn-processing collaborators.
- `src/orchestration/chat/chat_service.py` is now a stable facade over dedicated response, status, context, intent-routing, turn-pipeline, and turn-support collaborators.
- `src/memory/memory_system.py` is now a stable facade over dedicated status, fact, retrieval, context, consolidation, prospective, recall, and initialization services.
- `src/executive/integration.py` remains the executive orchestration facade over dedicated decision, planning, scheduling, reporting, and reminder projection stages.
- `src/interfaces/api/` now depends on public facade methods and shared request dependency helpers instead of ad hoc app-state access or underscore-prefixed internals.

### Exit criteria

- [x] Repo status, docs, and default commands describe the same system.

## Risks and Mitigations

### Risk: Extracting too much at once breaks runtime wiring

Mitigation:

- Extract one boundary at a time.
- Keep old facades in place until callers are migrated.
- Verify each phase with contract tests before moving on.

### Risk: Hidden coupling to private fields

Mitigation:

- Add explicit accessor methods before moving code.
- Replace private-field access in routers early.

### Risk: Docs remain stale while code changes fast

Mitigation:

- Update this file at the end of each completed refactor slice.
- Treat documentation updates as part of the exit criteria.

### Risk: Performance regressions during delegation refactors

Mitigation:

- Keep the same lazy-init behavior where possible.
- Preserve current fallback paths until equivalent behavior is verified.
- Compare latency/status outputs before and after each slice.

## Progress Log

### Completed during planning

- [x] Review top-level architecture and entrypoints.
- [x] Review chat orchestration shape.
- [x] Review memory facade shape.
- [x] Review cognitive agent runtime shape.
- [x] Review executive integration shape.
- [x] Identify composition-root drift.
- [x] Identify best extraction boundaries.
- [x] Convert the review into a phased refactor plan.
- [x] Add missing contract coverage for `CognitiveAgent` and `ExecutiveSystem` baseline behavior.
- [x] Introduce the first shared runtime container and route chat/api runtime helpers through it.
- [x] Replace `main.py` legacy API bootstrap with a canonical runtime FastAPI app builder.
- [x] Extract the first `ChatService` response assembly collaborator behind a stable facade.
- [x] Extract `ChatService` status and reminder formatting helpers behind a dedicated status service.
- [x] Extract `ChatService` session-context packaging behind a dedicated turn-context builder.
- [x] Extract `ChatService` intent planning and dispatch behind a dedicated intent router.
- [x] Extract `ChatService.process_user_message` orchestration behind a dedicated turn pipeline collaborator.
- [x] Extract Phase 2 layer runtime state and updates behind a dedicated cognitive-layer runtime collaborator.
- [x] Extract ChatService turn-support and remaining intent-action helpers behind dedicated collaborators.
- [x] Extract the first MemorySystem status and session-lifecycle service behind `src/memory/services/status_service.py`.
- [x] Extract MemorySystem fact CRUD behind `src/memory/services/fact_service.py`.
- [x] Extract MemorySystem retrieval and search behind `src/memory/services/retrieval_service.py`.
- [x] Extract MemorySystem context aggregation behind `src/memory/services/context_service.py`.
- [x] Extract MemorySystem consolidation policy and scheduling behind `src/memory/services/consolidation_service.py`.
- [x] Extract MemorySystem prospective scheduling helpers behind `src/memory/services/prospective_service.py`.
- [x] Extract MemorySystem proactive recall and summarization behind `src/memory/services/recall_service.py`.
- [x] Normalize canonical API routers onto shared request dependency helpers and remove remaining router access to private runtime state.
- [x] Remove unmounted legacy executive API variants so the mounted interface surface is canonical.
- [x] Route `main.py chat` through the shared runtime container so CLI and API startup use the same composition root.
- [x] Reconcile primary startup, architecture, and test-lane docs with the extracted runtime and current verification model.

## Notes for Execution

- Start with boundaries that reduce ambiguity first, not with the biggest file first.
- Do not begin by renaming modules. Begin by extracting collaborators and delegating to them.
- Every time a slice lands, shrink the old facade file immediately so drift does not persist.
- Update this file as tasks are completed so it remains the source of truth for Phase 3.