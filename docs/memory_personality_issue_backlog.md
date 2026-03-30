# Memory And Personality Issue Backlog

Last updated: 2026-03-28

## Purpose

This document converts the implementation ticket plan into issue-style backlog items suitable for project tracking.

It complements:

- `memory_personality_roadmap.md`
- `memory_personality_architecture.md`
- `memory_personality_implementation_tickets.md`

Use this file when you want a backlog that is ready for GitHub issues, milestones, sprint planning, or handoff.

## Conventions

- Priority values: `P0`, `P1`, `P2`
- Milestones map to roadmap phases
- Dependencies refer to other backlog IDs in this file
- Acceptance criteria should be satisfied before a ticket is marked done

## Milestones

- `Phase 1 Foundations`
- `Phase 2 Autobiographical Continuity`
- `Phase 3 Personality Binding`
- `Phase 4 Adaptive Memory And Believability`

## Phase 1 Foundations

### MP-101 Add Canonical Memory Schema

- Priority: `P0`
- Milestone: `Phase 1 Foundations`
- Depends on: none

Problem:
Current retrieval outputs vary by store shape, which makes context assembly and downstream reasoning brittle.

Scope:

- add canonical memory item model under `src/memory/schema/canonical.py`
- define required fields for memory kind, provenance, confidence, contradiction metadata, relationship target, and tags
- document allowed `memory_kind` values

Acceptance criteria:

- canonical model exists and is typed
- representative STM, LTM, episodic, and prospective fixtures can be expressed through the model
- no current active path depends on ad hoc undocumented fields for core memory identity

Validation:

- schema construction tests
- validation or coercion tests for required fields

Docs:

- update `docs/memory_personality_architecture.md` if field names differ from the target draft

### MP-102 Add Store Normalization Adapters

- Priority: `P0`
- Milestone: `Phase 1 Foundations`
- Depends on: `MP-101`

Problem:
Store-specific retrieval results cannot yet be consumed uniformly by planners, rerankers, or policy-aware prompt assembly.

Scope:

- add normalization adapters under `src/memory/schema/normalization.py`
- support STM, LTM, episodic, and prospective outputs
- preserve source and provenance metadata during normalization

Acceptance criteria:

- all currently queried store results normalize into one schema
- normalization handles missing or partial metadata safely
- duplicate identity rules are defined consistently

Validation:

- adapter tests by store type
- missing-field fallback tests
- duplicate normalization tests

Docs:

- update `docs/memory_taxonomy.md` if store-role wording needs clarification

### MP-103 Add Retrieval Plan Contract

- Priority: `P0`
- Milestone: `Phase 1 Foundations`
- Depends on: `MP-101`, `MP-102`

Problem:
The current retrieval flow is mostly blended and stage-based, not query-planned.

Scope:

- add retrieval plan dataclass under `src/memory/retrieval/retrieval_plan.py`
- add first planner under `src/memory/retrieval/planner.py`
- encode store selection, per-store limits, and fallback policy

Acceptance criteria:

- planner can emit distinct plans for factual, episodic, social, and reminder-oriented queries
- plan outputs can be consumed without changing store implementations
- fallback behavior is explicit rather than implicit

Validation:

- planner routing tests
- degraded-store fallback tests

Docs:

- update `docs/memory_personality_architecture.md` if planner shape changes

### MP-104 Add Cross-Store Reranker

- Priority: `P1`
- Milestone: `Phase 1 Foundations`
- Depends on: `MP-102`, `MP-103`

Problem:
Even with normalized candidates, the system still needs a principled way to choose which items reach the working set.

Scope:

- add reranker under `src/memory/retrieval/reranker.py`
- score on relevance, recency, confidence, and diversity
- suppress duplicate or near-duplicate context items

Acceptance criteria:

- reranker produces deterministic ordering for fixed fixtures
- duplicate context items are reduced in regression cases
- selected items carry enough score metadata to explain ranking

Validation:

- ranking and dedupe tests
- regression fixture showing fewer repeated items

Docs:

- update `docs/memory_personality_roadmap.md` if ranking priorities materially shift

### MP-105 Add Retrieval Evaluation Baseline

- Priority: `P1`
- Milestone: `Phase 1 Foundations`
- Depends on: `MP-103`

Problem:
Retrieval quality improvements are hard to prove without stable scenarios and measurable expectations.

Scope:

- add eval scaffolding under `src/evals/scenarios/` and `src/evals/metrics/`
- define baseline retrieval scenarios for facts, episodes, and contradictions
- record precision and irrelevant-context metrics

Acceptance criteria:

- a repeatable evaluation runner exists for baseline scenarios
- expected retrieval sets are explicit for seed fixtures
- retrieval regressions become observable before Phase 2 expansion

Validation:

- scenario-runner smoke tests
- fixture-based expected-result tests

Docs:

- update `docs/memory_personality_roadmap.md` with eval notes if needed

## Phase 2 Autobiographical Continuity

### MP-201 Add Relationship Memory Model

- Priority: `P0`
- Milestone: `Phase 2 Autobiographical Continuity`
- Depends on: `MP-101`

Problem:
Relationship state currently lives mainly in transient runtime structures instead of a persistent, recallable memory model.

Scope:

- add `src/memory/relationship/model.py`
- add `src/memory/relationship/store.py`
- define warmth, trust, familiarity, rupture, and recurring norms fields

Acceptance criteria:

- relationship state can be stored, loaded, and updated per user/interlocutor
- relationship memory survives restart
- schema can support future retrieval weighting

Validation:

- persistence tests
- load/update roundtrip tests

Docs:

- update `docs/memory_personality_architecture.md` if relationship fields evolve

### MP-202 Persist Relationship Updates From Runtime Signals

- Priority: `P1`
- Milestone: `Phase 2 Autobiographical Continuity`
- Depends on: `MP-201`

Problem:
Relational state is computed during turns, but not consistently promoted into long-lived relationship memory.

Scope:

- add `src/memory/relationship/updater.py`
- optionally add `src/memory/encoding/relationship_encoder.py`
- map turn salience, valence, and relational cues into persisted deltas

Acceptance criteria:

- turn-level relational signals can update persistent relationship memory
- repeated interactions accumulate into stable trends
- sync rules between runtime snapshots and stored relationship state are explicit

Validation:

- updater policy tests
- repeated-turn trend tests

Docs:

- update `docs/memory_personality_implementation_tickets.md` if integration boundaries shift

### MP-203 Add Autobiographical Graph And Chapter Summaries

- Priority: `P1`
- Milestone: `Phase 2 Autobiographical Continuity`
- Depends on: `MP-101`, `MP-102`

Problem:
The system can retrieve episodes, but it does not yet organize them into life-phase continuity or defining moments.

Scope:

- add event-encoding path under `src/memory/encoding/event_encoder.py`
- add graph or link structures for people, projects, goals, and events
- add chapter summary contract and defining-moment markers

Acceptance criteria:

- episode links can be created across related events
- chapter summaries can be generated from linked events
- defining moments can be marked and retrieved distinctly

Validation:

- event-link creation tests
- chapter summary tests

Docs:

- update `docs/memory_personality_architecture.md` if the graph model differs from the initial proposal

### MP-204 Add Relationship-Aware Retrieval

- Priority: `P1`
- Milestone: `Phase 2 Autobiographical Continuity`
- Depends on: `MP-103`, `MP-104`, `MP-201`

Problem:
Retrieval planning and reranking do not yet account for user-specific relationship relevance.

Scope:

- extend planner and reranker to consume relationship signals
- allow chapter-aware recall for continuity questions
- integrate relationship weighting into `ContextBuilder`

Acceptance criteria:

- user-specific recall changes when relationship memory differs
- continuity-style prompts can use chapter-aware retrieval
- recall remains bounded and explainable

Validation:

- relationship-conditioned recall tests
- autobiographical continuity scenarios

Docs:

- update `docs/memory_personality_roadmap.md` if retrieval priorities change materially

### MP-205 Add Contradiction Sets And Belief Revision

- Priority: `P0`
- Milestone: `Phase 2 Autobiographical Continuity`
- Depends on: `MP-101`, `MP-102`

Problem:
Conflicting beliefs and corrected user facts are not yet tracked as explicit contradiction sets.

Scope:

- add `src/memory/schema/contradiction.py`
- define contradiction-set IDs and quarantine rules
- add source-weighted revision policy

Acceptance criteria:

- conflicting facts can be grouped instead of silently overwritten
- revised beliefs preserve source history and confidence context
- uncertain beliefs can be quarantined before promotion

Validation:

- contradictory preference tests
- belief repair regression tests

Docs:

- update `docs/memory_personality_architecture.md` if revision policy semantics shift

## Phase 3 Personality Binding

### MP-301 Add Response Policy Contract

- Priority: `P0`
- Milestone: `Phase 3 Personality Binding`
- Depends on: `MP-201`, `MP-205`

Problem:
Internal state exists, but there is no explicit contract translating it into response behavior.

Scope:

- add `src/orchestration/policy/response_policy.py`
- define warmth, directness, curiosity, uncertainty, and disclosure fields
- separate stable traits from dynamic state

Acceptance criteria:

- response-policy model is typed and inspectable
- fixed internal-state fixtures produce deterministic policy outputs
- policy fields are suitable for prompt assembly and evaluation

Validation:

- policy schema tests
- deterministic derivation tests

Docs:

- update `docs/memory_personality_architecture.md` if policy field names shift

### MP-302 Add Policy Composer

- Priority: `P0`
- Milestone: `Phase 3 Personality Binding`
- Depends on: `MP-301`

Problem:
The cognitive layer runtime computes drives, mood, self-model, and narrative, but not a unified response-policy output.

Scope:

- add `src/orchestration/policy/policy_composer.py`
- compose policy from self-model, narrative, mood, drives, and relationship state
- expose traceable policy inputs for observability

Acceptance criteria:

- current cognitive layer snapshots can be composed into one response policy
- policy composition is traceable and testable
- changes in internal state lead to coherent policy deltas

Validation:

- policy-composition tests
- changed-state regression fixtures

Docs:

- update `docs/memory_personality_implementation_tickets.md` if integration seams shift

### MP-303 Add Structured Policy-Aware Prompt Assembly

- Priority: `P1`
- Milestone: `Phase 3 Personality Binding`
- Depends on: `MP-301`, `MP-302`, `MP-103`

Problem:
`llm_session.py` still builds prompts around a generic system prompt and a flat memory block.

Scope:

- add `src/orchestration/policy/policy_rendering.py`
- restructure `src/orchestration/agent/llm_session.py` to emit role, policy, working-self state, and memory blocks
- keep prompt budgets bounded

Acceptance criteria:

- prompt assembly separates policy from retrieved memory context
- message construction remains bounded and inspectable
- fallback behavior still works when the LLM is unavailable

Validation:

- prompt assembly shape tests
- policy rendering tests

Docs:

- update `docs/memory_personality_architecture.md` if message structure changes materially

### MP-304 Add Personality Behavior Evaluation

- Priority: `P1`
- Milestone: `Phase 3 Personality Binding`
- Depends on: `MP-302`, `MP-303`

Problem:
Personality changes are hard to verify unless output behavior is evaluated against explicit policy inputs.

Scope:

- extend eval harness with warmth, directness, uncertainty, and disclosure scenarios
- add traceability checks from input state to output stance

Acceptance criteria:

- fixed scenarios can detect policy-to-output regressions
- behavior differences can be tied back to policy inputs
- core style traits become measurable rather than anecdotal

Validation:

- behavioral regression tests
- traceability fixture tests

Docs:

- update `docs/memory_personality_roadmap.md` if evaluation emphasis changes

## Phase 4 Adaptive Memory And Believability

### MP-401 Add Reconsolidation Hooks

- Priority: `P1`
- Milestone: `Phase 4 Adaptive Memory And Believability`
- Depends on: `MP-205`

Problem:
Recall does not yet modify memory strength and confidence in a first-class way.

Scope:

- add post-recall reinforcement and weakening rules
- update confidence and source weighting after successful or failed recall
- wire reconsolidation into existing memory flows without destabilizing storage backends

Acceptance criteria:

- repeated successful recall can strengthen memory metadata
- correction or failed recall can weaken or revise metadata
- reconsolidation behavior is observable through tests or diagnostics

Validation:

- repeated-recall tests
- weakening-on-correction tests

Docs:

- update `docs/memory_personality_architecture.md` if reconsolidation semantics change

### MP-402 Add Selective Forgetting And Suppression

- Priority: `P1`
- Milestone: `Phase 4 Adaptive Memory And Believability`
- Depends on: `MP-401`

Problem:
Long-horizon memory quality will degrade if every low-value item competes indefinitely for retrieval.

Scope:

- add suppression or decay policy for low-value memories
- define protection rules for autobiographical and relationship anchors
- integrate forgetting policy with existing decay and consolidation settings where appropriate

Acceptance criteria:

- low-value memories can decay or suppress without deleting anchors prematurely
- high-value autobiographical and relationship items remain protected
- forgetting rules are explicit and testable

Validation:

- forgetting-policy tests
- anchor-preservation tests

Docs:

- update `docs/memory_personality_roadmap.md` if forgetting priorities shift materially

### MP-403 Add Longitudinal Evaluation Harness

- Priority: `P0`
- Milestone: `Phase 4 Adaptive Memory And Believability`
- Depends on: `MP-105`, `MP-304`, `MP-401`

Problem:
Long-term continuity and believability cannot be validated with only short single-turn tests.

Scope:

- add restart-aware multi-session scenarios
- add false-memory and over-recall checks
- measure continuity and contradiction-repair behavior over longer horizons

Acceptance criteria:

- multi-session scenarios can be run deterministically
- restart continuity and over-recall metrics are produced
- long-horizon regressions become visible before broad release

Validation:

- end-to-end scenario runner tests
- restart-aware fixture tests

Docs:

- update `docs/memory_personality_implementation_tickets.md` if scenario structure changes

### MP-404 Add Quality Gates And Scorecards

- Priority: `P1`
- Milestone: `Phase 4 Adaptive Memory And Believability`
- Depends on: `MP-403`

Problem:
Memory and personality quality need a stable decision surface for pre-merge review.

Scope:

- add scorecard generation for key metrics
- define recommended CI or pre-merge thresholds
- connect scorecard output to eval metrics and telemetry where useful

Acceptance criteria:

- scorecards summarize continuity, contradiction, retrieval quality, and behavior stability
- thresholds can fail a gate or at least flag review-worthy regressions
- scorecard generation is deterministic for fixed fixtures

Validation:

- scorecard generation tests
- threshold regression tests

Docs:

- update `README.md` or `docs/README.md` if scorecard workflow becomes part of the standard process

## Recommended Execution Order

Build in this order first:

1. `MP-101`
2. `MP-102`
3. `MP-103`
4. `MP-104`
5. `MP-105`

This keeps the first milestone focused on normalized memory contracts and retrieval quality before relationship, self-model, and behavior-control expansion.
