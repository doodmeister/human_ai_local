# Memory And Personality Implementation Tickets

Last updated: 2026-04-02

## Purpose

This document turns the roadmap and target architecture into concrete implementation tickets by phase.

It complements:

- `memory_personality_roadmap.md` for goals, sequencing, and success criteria
- `memory_personality_architecture.md` for the target design
- `memory_personality_issue_backlog.md` for issue-style tracking items
- `../phase3.md` for current runtime-consolidation status

Use this document when deciding what to build next, what files to touch, how to stage migrations, and what validation should exist before a slice is considered done.

## Planning Rules

- each ticket should fit the current repo rather than assume a greenfield rewrite
- prefer thin vertical slices over broad framework additions
- keep the runtime container and current entrypoints stable while introducing new layers
- every ticket should name code seams, tests, and docs updates
- docs and implementation should move together when a ticket changes current behavior

## Current Code Seams

The current implementation seams that matter most for this plan are:

- `src/memory/memory_system.py`
- `src/memory/services/retrieval_service.py`
- `src/orchestration/chat/context_builder.py`
- `src/orchestration/cognitive_layers/runtime.py`
- `src/orchestration/agent/llm_session.py`

These are the highest-leverage insertion points for phased memory and personality work without destabilizing the runtime.

## Audit Summary

As of 2026-04-02, this ticket list no longer represents mostly unstarted work.

Verified implemented and wired in the current runtime:

- MP-101 through MP-105
- MP-201, MP-202, MP-204, and MP-205
- MP-301 through MP-304
- MP-401 through MP-404

Implemented but still limited in depth or durability:

- MP-203 now has concrete event encoding, graph and chapter builders, persisted autobiographical graph snapshots, plus autobiographical promotion on both the main chat path and the lower-level cognitive-agent path; the same promoted turn now writes multiple semantic products including relationship-derived preference facts, narrative-theme focus facts, and explicit follow-up reminders that now round-trip through persisted prospective memory across restart, but richer generalized multi-product updates are still incomplete

Primary next follow-up slice after this audit:

- finish durable autobiographical continuity by persisting chapter and graph state and promoting it from live turn outputs
- deepen continuity retrieval so restart-aware behavior depends on persisted autobiographical state rather than only on episodic overlap
- keep reconsolidation, contradiction, forgetting, and quality-gate work aligned with that durable autobiographical layer

## Phase 1: Foundations

Outcome:
Canonical memory representation plus planned retrieval infrastructure.

Current status:
Implemented in the active runtime and backed by targeted tests and evaluation scenarios.

### MP-101 Canonical Memory Schema

- Goal: define a normalized memory item contract shared across stores and retrieval consumers.
- Existing seams: `src/memory/memory_system.py`, `src/orchestration/chat/models.py`, `src/orchestration/chat/context_builder.py`
- New modules:
  - `src/memory/schema/canonical.py`
- Deliverables:
  - typed canonical memory item model
  - allowed `memory_kind` values and required metadata fields
  - provenance, confidence, contradiction, and relationship fields
- Tests:
  - schema construction and validation tests
  - representative fixtures for STM, LTM, episodic, and prospective items
- Docs:
  - update `docs/memory_personality_architecture.md` if field names shift

### MP-102 Store Normalization Adapters

- Goal: normalize current retrieval outputs into the canonical schema without rewriting storage backends.
- Existing seams: `src/memory/services/retrieval_service.py`, `src/memory/memory_system.py`
- New modules:
  - `src/memory/schema/normalization.py`
- Deliverables:
  - adapters for STM, LTM, episodic, and prospective results
  - source tagging and provenance preservation
  - duplicate-safe identity rules for normalized items
- Tests:
  - adapter tests for each store type
  - duplicate and missing-field fallback tests
- Docs:
  - update `docs/memory_taxonomy.md` if store responsibilities are clarified

### MP-103 Retrieval Plan Contract

- Goal: introduce a retrieval-plan object that decides which stores to query and with what limits.
- Existing seams: `src/orchestration/chat/context_builder.py`, `src/memory/services/retrieval_service.py`
- New modules:
  - `src/memory/retrieval/retrieval_plan.py`
  - `src/memory/retrieval/planner.py`
- Deliverables:
  - retrieval plan dataclass
  - initial routing based on query, intent, and task type
  - explicit per-store budgets and fallback policy
- Tests:
  - planner routing tests for factual, episodic, and reminder queries
  - fallback behavior tests when a store is unavailable
- Docs:
  - update `docs/memory_personality_architecture.md` if planner outputs evolve

### MP-104 Cross-Store Reranker

- Goal: rerank normalized candidates across stores using relevance, recency, confidence, and diversity.
- Existing seams: `src/orchestration/chat/scoring.py`, `src/orchestration/chat/context_builder.py`
- New modules:
  - `src/memory/retrieval/reranker.py`
- Deliverables:
  - reranker scoring contract
  - diversity suppression for duplicate or near-duplicate items
  - explainable scoring metadata for top items
- Tests:
  - ranking and dedupe tests
  - regression test showing reduced duplicate context items
- Docs:
  - update `docs/memory_personality_roadmap.md` if scoring priorities change

### MP-105 Retrieval Evaluation Baseline

- Goal: make Phase 1 measurable before broader memory expansion.
- Existing seams: `tests/`, current chat/context tests
- New modules:
  - `src/evals/scenarios/`
  - `src/evals/metrics/`
- Deliverables:
  - baseline scenarios for fact recall, episodic recall, and contradiction detection
  - precision and irrelevant-context metrics
- Tests:
  - scenario runner smoke tests
  - fixed fixtures for expected retrieval sets
- Docs:
  - add evaluation notes to `docs/memory_personality_roadmap.md`

## Phase 2: Autobiographical Continuity

Outcome:
Persistent relationship and life-phase continuity across sessions.

Current status:
Largely implemented. Relationship persistence, updater integration, contradiction repair, durable autobiographical graph snapshots, live autobiographical promotion, and chapter-summary context injection are live; the main remaining depth items are persisted-chapter-first continuity retrieval, restart-independent continuity evaluation, and broader multi-product promotion depth.

### MP-201 Relationship Memory Model

- Goal: introduce a first-class per-user relationship memory store.
- Existing seams: `src/orchestration/cognitive_layers/runtime.py`, `src/memory/memory_system.py`
- New modules:
  - `src/memory/relationship/model.py`
  - `src/memory/relationship/store.py`
- Deliverables:
  - relationship state model with warmth, trust, familiarity, rupture, and norms
  - persistence keyed by user or interlocutor identity
- Tests:
  - create/update/load relationship model tests
  - restart continuity tests

### MP-202 Relationship Updater Integration

- Goal: update relationship state from turn-level signals instead of keeping it only in transient runtime state.
- Existing seams: `src/orchestration/cognitive_layers/runtime.py`
- New modules:
  - `src/memory/relationship/updater.py`
  - optionally `src/memory/encoding/relationship_encoder.py`
- Deliverables:
  - mapping from turn salience, valence, and relational cues into persisted relationship updates
  - session snapshot to persistent-store sync rules
- Tests:
  - relationship update policy tests
  - repeated-turn trend tests

### MP-203 Autobiographical Graph And Chapters

- Goal: connect episodes, projects, goals, and people into durable autobiographical continuity.
- Existing seams: episodic memory integrations and prospective/goals references
- New modules:
  - `src/memory/encoding/event_encoder.py`
  - graph or link model under `src/memory/relationship/` or a dedicated autobiographical package
- Deliverables:
  - event-link model
  - chapter summary contract
  - defining-moment markers
- Tests:
  - graph link creation tests
  - chapter summary generation tests

### MP-204 Relationship-Aware Retrieval

- Goal: allow retrieval planning and ranking to consume relationship and autobiographical signals.
- Existing seams: `src/memory/retrieval/planner.py`, `src/memory/retrieval/reranker.py`, `src/orchestration/chat/context_builder.py`
- Deliverables:
  - relationship-weighted recall for user-specific prompts
  - chapter-aware retrieval for “what changed lately?” style queries
- Tests:
  - relationship-conditioned recall tests
  - autobiographical continuity scenario tests

### MP-205 Contradiction Sets And Belief Revision

- Goal: track conflicting beliefs explicitly instead of silent overwrite.
- Existing seams: normalized schema and retrieval metadata from Phase 1
- New modules:
  - `src/memory/schema/contradiction.py`
- Deliverables:
  - contradiction-set IDs
  - quarantine vs promote rules
  - source-weighted belief revision policy
- Tests:
  - contradictory preference tests
  - belief repair regression tests

## Phase 3: Personality Binding

Outcome:
Internal state becomes explicit response policy rather than prompt-side decoration.

Current status:
Implemented in runtime composition, prompt assembly, API payloads, and behavior evaluation paths.

### MP-301 Response Policy Contract

- Goal: define a structured response-policy object for warmth, directness, curiosity, uncertainty, and disclosure.
- Existing seams: `src/orchestration/cognitive_layers/runtime.py`, `src/orchestration/agent/llm_session.py`
- New modules:
  - `src/orchestration/policy/response_policy.py`
- Deliverables:
  - typed response-policy contract
  - stable trait layer plus dynamic state layer
- Tests:
  - policy schema tests
  - deterministic policy derivation tests from fixed internal-state fixtures

### MP-302 Policy Composer

- Goal: compose response policy from self-model, narrative, mood, drives, and relationship state.
- Existing seams: `src/orchestration/cognitive_layers/runtime.py`
- New modules:
  - `src/orchestration/policy/policy_composer.py`
- Deliverables:
  - composer that consumes current cognitive layer snapshots
  - explicit policy trace fields for observability
- Tests:
  - policy-composition tests for changing mood, relationship stance, and self-model inputs

### MP-303 Structured LLM Prompt Assembly

- Goal: move `llm_session.py` from generic memory dump to structured policy-aware message blocks.
- Existing seams: `src/orchestration/agent/llm_session.py`
- New modules:
  - `src/orchestration/policy/policy_rendering.py`
- Deliverables:
  - distinct sections for role, policy, working-self state, and memory context
  - bounded prompt assembly with clear source blocks
- Tests:
  - prompt assembly shape tests
  - policy rendering tests

### MP-304 Personality Behavior Evaluation

- Goal: verify that personality is visible in outputs and traceable to policy inputs.
- Existing seams: Phase 1 eval harness
- Deliverables:
  - scenario set for warmth, directness, uncertainty style, and memory disclosure
  - traceability checks from input state to output stance
- Tests:
  - behavioral regression tests with fixed scenarios

## Phase 4: Adaptive Memory And Believability

Outcome:
Memory revision, forgetting, and long-horizon evaluation become first-class.

Current status:
Implemented for the current deterministic quality-gate scope, including reconsolidation feedback, forgetting and suppression policy, restart-aware longitudinal scenarios, and scorecard gates.

### MP-401 Reconsolidation Hooks

- Goal: update memory strength and metadata after successful or failed recall.
- Existing seams: `src/memory/memory_system.py`, retrieval services, reflection hooks
- Deliverables:
  - post-recall reinforcement and weakening rules
  - confidence drift updates
- Tests:
  - repeated-recall strengthening tests
  - miss or correction weakening tests

### MP-402 Selective Forgetting And Suppression

- Goal: reduce stale or low-value memory noise without losing autobiographical anchors.
- Existing seams: STM/LTM decay and consolidation configuration
- Deliverables:
  - suppression or decay policy for low-value memories
  - protection rules for high-value autobiographical or relationship anchors
- Tests:
  - forgetting policy tests
  - anchor preservation tests

### MP-403 Longitudinal Evaluation Harness

- Goal: measure continuity, contradiction repair, and personality stability over long-running scenarios.
- Existing seams: eval scaffolding from Phase 1 and personality tests from Phase 3
- Deliverables:
  - restart-aware scenarios
  - multi-session continuity suite
  - false-memory and over-recall checks
- Tests:
  - end-to-end scenario runner tests

### MP-404 Quality Gates And Scorecards

- Goal: turn memory and personality metrics into pre-merge decision support.
- Existing seams: test suite, telemetry, metrics registry
- Deliverables:
  - scorecard output for key metrics
  - recommended CI or pre-merge gates for regressions
- Tests:
  - scorecard generation tests
  - threshold regression tests

## Recommended Next Slice After Audit

Build these next in sequence:

1. deepen MP-204 so continuity retrieval prefers persisted autobiographical chapters and defining moments rather than mostly graphing the current episodic candidate set
2. keep MP-401, MP-403, and MP-404 aligned with that durable autobiographical layer so restart-aware evals and scorecards require persisted chapter continuity rather than episodic overlap alone
3. extend MP-203's promotion seam into a broader generalized event-to-product pipeline for additional semantic and self-model updates

Why this next:

- the original Phase 1 and most Phase 3 scaffolding already landed and is wired into runtime code
- autobiographical continuity retrieval and evaluation depth are now the clearest remaining gaps between the current system and the roadmap's target behavior
- this slice compounds existing work instead of reopening already-solved schema, planner, and prompt-assembly seams

## Suggested Follow-Up Breakdown

### Slice A

- thread persisted autobiographical state more aggressively into `ContextBuilder`, retrieval planning, and reranking
- prefer chapter summaries and defining moments for continuity-style prompts and "what changed lately?" queries
- keep retrieval budgets bounded and explainable while doing so

### Slice B

- extend longitudinal and scorecard scenarios so continuity success requires persisted autobiographical state
- add regression checks that distinguish episodic overlap from true chapter continuity
- ensure restart-aware quality gates fail when chapter persistence is missing but episodic recall still looks superficially relevant

### Slice C

- broaden the promoted-turn multi-product seam for additional semantic classes or self-model updates
- keep the new products lazy, optional, and measurable through the existing eval harnesses
- avoid reopening already-complete persistence or prompt-assembly work unless the new products require it

## Phase Exit Gates

- Phase 1 exits when retrieval is normalized, planned, and minimally measurable.
- Phase 2 exits when the system can maintain relationship and autobiographical continuity across restarts.
- Phase 3 exits when internal state becomes explicit response policy and output behavior is traceable to it.
- Phase 4 exits when revision, forgetting, and long-horizon evaluation are measurable and stable.
