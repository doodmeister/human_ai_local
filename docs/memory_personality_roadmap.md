# World-Class Memory And Personality Roadmap

Last updated: 2026-04-02

## Purpose

This document is the forward-looking roadmap for evolving the project from a strong modular cognitive architecture into a world-class memory system and personality system that more closely mimics human continuity, selfhood, and social behavior.

It complements:

- `../README.md` for current runtime and developer workflow
- `../phase3.md` for architecture-consolidation status
- `memory_personality_architecture.md` for the target design
- `memory_personality_implementation_tickets.md` for phase-by-phase build tickets

## North Star

The system should:

- remember the right things, not just many things
- forget selectively and believably
- maintain a stable but revisable identity over time
- form and update relationship-specific models of users
- reconcile contradiction instead of silently overwriting memory
- use personality and internal state to shape behavior, not only telemetry
- remain inspectable, testable, and safe under bounded context limits

## Current Position

The repository already has strong foundations:

- multi-store memory across STM, LTM, episodic, semantic, and prospective systems
- a consolidated runtime and thin facades after Phase 3
- attention, metacognition, consolidation, and reflection hooks
- a layered cognitive runtime for drives, felt sense, relational state, self-model, and narrative
- a canonical memory item schema plus normalization adapters for STM, LTM, episodic, semantic, and prospective retrieval outputs
- chat `ContextBuilder` and direct runtime memory-context assembly now both consume canonicalized memory payloads instead of store-specific prompt dicts

The main gap is not missing modules. The main gap is that memory encoding, retrieval, autobiographical continuity, and personality control are still too shallow or too loosely coupled to produce convincingly human-like continuity.

## Audit Snapshot

As of 2026-04-02, the roadmap needs to be read as a forward-looking design document rather than a list of unstarted work.

Verified against the current runtime and tests:

- Phase 1 foundations are implemented and wired into the active chat/runtime path
- Phase 2 relationship memory and contradiction repair are implemented; autobiographical continuity is present but still limited by on-demand graph construction rather than a durable autobiographical store
- Phase 3 response-policy and policy-aware prompt assembly are implemented and exercised in runtime and evaluation paths
- Phase 4 reconsolidation, forgetting, longitudinal evaluation, and scorecard gating all exist for the current deterministic quality-gate scope

The highest-value remaining gaps are now depth and durability:

- stronger event-to-memory product generation from live turns
- persisted autobiographical graph and chapter state beyond retrieval-time derivation
- tighter coupling between recall outcomes, contradiction handling, and long-horizon autobiographical continuity

## Strategic Workstreams

### 1. Canonical Memory Representation

Goal:
Create one memory schema that all stores can produce and consume.

Why:
The current system has rich data spread across multiple stores and adapters, but no single canonical representation for beliefs, events, relationship facts, uncertainty, and narrative role.

Deliverables:

- canonical memory item schema
- explicit event-to-memory transformation pipeline
- typed support for entities, time intervals, source, confidence, affect, contradiction markers, and narrative role
- compatibility adapters for STM, LTM, episodic, semantic, and prospective memory

Current implementation note:

- `src/memory/schema/canonical.py` defines the shared memory contract
- `src/memory/schema/normalization.py` normalizes raw store outputs and unified `MemorySystem.search_memories()` tuples into canonical items
- prompt-facing context payloads for both chat and direct runtime retrieval are now derived from canonical items rather than ad hoc per-store formatting

Success criteria:

- every retrieved item can be normalized into one schema
- every stored event can produce multiple memory products deterministically
- contradiction and provenance become first-class metadata instead of side conventions

### 2. Retrieval Planner And Reranking

Goal:
Move from flat blended search to query-aware retrieval planning.

Why:
Human-like memory depends on selecting the right type of memory for the task: episode, fact, relationship history, self-belief, pending intention, or narrative continuity.

Deliverables:

- query-intent to retrieval-plan routing
- source-specific candidate generation
- cross-store reranking using relevance, recency, confidence, affect, relationship relevance, and novelty/diversity
- explicit retrieval budgets and failure modes

Current implementation note:

- `src/memory/retrieval/retrieval_plan.py` and `src/memory/retrieval/planner.py` define and route retrieval plans by query intent
- `src/memory/retrieval/reranker.py` reranks canonical items using similarity, recency, activation, salience, confidence, relationship relevance, and continuity signals
- `src/orchestration/chat/context_builder.py` now plans retrieval before querying stores and reranks normalized candidates before prompt-facing context assembly

Success criteria:

- memory lookup precision improves in targeted eval scenarios
- repeated irrelevant or duplicate context items decrease
- system can explain why a memory was selected

### 3. Autobiographical And Relationship Memory

Goal:
Turn isolated recalled snippets into durable personal continuity.

Why:
Human-like behavior depends on remembering not just facts, but episodes, relationships, unfinished tensions, promises, and recurring themes.

Deliverables:

- autobiographical graph linking episodes, people, projects, goals, and chapters
- per-user relationship memory with trust, warmth, familiarity, rupture/repair, and recurring norms
- chapter and defining-moment summaries
- relationship-aware retrieval and response conditioning

Current implementation note:

- `src/memory/relationship/model.py`, `src/memory/relationship/store.py`, and `src/memory/relationship/updater.py` persist per-user relationship memory and update it from turn-level relational signals
- `src/memory/encoding/event_encoder.py` and `src/memory/autobiographical/graph.py` provide event encoding plus graph and chapter construction primitives
- `src/memory/autobiographical/store.py` now persists autobiographical graph snapshots, and `src/orchestration/chat/context_builder.py` can merge persisted chapter state with on-demand graph construction during continuity reranking
- `src/orchestration/chat/context_builder.py` now also surfaces persisted autobiographical chapter summaries as live context for continuity-oriented queries, so restart-aware chapter state affects both reranking and the actual working context seen by the response path
- `src/orchestration/chat/turn_pipeline.py` and `src/orchestration/chat/chat_service.py` now promote consolidated chat turns into episodic memories and refresh the session's persisted autobiographical snapshot on the main chat path
- `src/orchestration/agent/turn_processor.py` now uses the same autobiographical promotion seam for the lower-level cognitive-agent path, so direct `CognitiveAgent.process_input()` calls also refresh persisted autobiographical snapshots
- `src/orchestration/autobiographical_promotion.py` now lets the same promoted turn emit semantic preference facts from relationship-memory norms, so one interaction can update episodic, autobiographical, and semantic stores together
- the shared promotion path now emits a broader product set: preference-style facts from relationship norms, focus facts from narrative themes, and explicit follow-up reminders into prospective memory, so promoted turns can update semantic and prospective state without going back through the capture-frequency path
- the same shared promotion path now also marks explicit milestone and turning-point interactions as autobiographical defining moments, so persisted chapter state can preserve those pivots even when raw episodic salience is not high enough to infer them implicitly
- `src/evals/scenarios/longitudinal_memory.py` now includes restart-aware fixture and runtime scenarios for promoted preference continuity and promoted preference contradiction repair, so the new semantic writeback path is covered by the quality-gate eval lane
- `src/evals/scenarios/retrieval_baseline.py` now includes runtime restart scenarios proving promoted preference facts, promoted focus facts, promoted follow-up reminders, promoted defining-moment episodes, and promoted chapter summaries surface through the live persisted retrieval/context path after restart
- `tests/test_chat_autobiographical_promotion.py` now includes a direct chat-path summary check proving the runtime can summarize a recent life phase together with a relationship trajectory from continuity-oriented context
- `src/evals/scenarios/policy_behavior.py` now includes a persisted-runtime relationship continuity scenario where a fresh agent reloads relationship memory, recomposes response policy from it, and changes wording through the live policy-aware response path
- `src/orchestration/cognitive_layers/runtime.py`, `src/orchestration/chat/context_builder.py`, and `src/memory/retrieval/reranker.py` already consume relationship memory and chapter-aware continuity signals in the live runtime
- the remaining gap is richer multi-product encoding depth: promoted turns now update episodic, autobiographical, multiple semantic axes, and restart-persisted explicit follow-up reminders, but there is still no broader generalized event-to-product pipeline for additional semantic classes, self-model updates, or chapter-linked continuity artifacts

Success criteria:

- system can summarize recent life phase and relationship trajectory
- system resumes after restart with stable user-specific interaction context
- relationship continuity affects recall and wording in measurable ways

### 4. Personality As Behavior Control

Goal:
Make personality and internal state shape response policy directly.

Why:
Current mood, narrative, and self-model are valuable, but they should do more than add descriptive prompt text. They should affect tone, boundaries, directness, curiosity, warmth, and tradeoff behavior.

Deliverables:

- response-policy composer driven by self-model, narrative, drives, mood, and relationship state
- explicit behavior controls for tone, interpersonal stance, uncertainty policy, and memory disclosure
- stable trait layer plus dynamic state layer
- controllable and testable prompt assembly for policy-conditioned generation
- behavior-level evaluation fixtures for warmth, directness, curiosity, uncertainty, and disclosure

Current implementation note:

- `src/orchestration/policy/response_policy.py` and `src/orchestration/policy/policy_composer.py` define and compose stable traits, dynamic state, effective policy, and trace data
- `src/orchestration/cognitive_layers/runtime.py` updates response policy from drives, mood, relationship state, self-model, and narrative snapshots during live turns
- `src/orchestration/policy/policy_rendering.py` and `src/orchestration/agent/llm_session.py` render structured role, policy, working-self, and memory-context prompt blocks
- deterministic behavior and runtime-path checks live in `src/evals/scenarios/policy_behavior.py` and related tests

Success criteria:

- behavior remains stable across sessions for the same persona
- behavior changes coherently when internal state changes
- style and stance can be explained through policy inputs rather than prompt accidents
- fixed scenarios can detect policy-to-output regressions in a repeatable way

### 5. Reconsolidation, Revision, And Forgetting

Goal:
Make memory updates more human-like after retrieval and reflection.

Why:
Human memory is not append-only. Recall changes memory. Contradictions are negotiated. Confidence shifts over time.

Deliverables:

- reconsolidation on successful recall
- confidence updates and source weighting
- explicit contradiction sets and belief revision logic
- selective forgetting and suppression policies

Current implementation note:

- `src/memory/schema/contradiction.py` and `src/memory/semantic/semantic_memory.py` now treat contradiction sets and belief revision as first-class semantic-memory behavior
- `src/memory/services/reconsolidation_service.py` applies post-recall reinforcement, correction, and failed-recall weakening across STM, LTM, episodic, and semantic stores when supported
- low-value long-term, episodic, and semantic memories can now be suppressed through an explicit memory-facade forgetting policy
- autobiographical and relationship anchors are preserved by rule-based protection checks instead of relying on ad hoc deletion avoidance
- retrieval and proactive recall skip suppressed or quarantined items by default so decayed memories stop competing indefinitely

Success criteria:

- repeated recall changes memory strength and structure
- contradictory user facts are tracked and repaired instead of silently clobbered
- stale or low-value memories decay without destroying high-value autobiographical anchors

### 6. Evaluation And Quality Gates

Goal:
Make progress measurable.

Why:
This project will otherwise accumulate theory and modules faster than real gains in continuity and believability.

Deliverables:

- longitudinal memory/personality eval suite
- scenario-based tests for identity, preference, contradiction, relationship, and narrative continuity
- false-memory and over-recall checks
- human-believability and memory-discipline scorecards

Current implementation note:

- `src/evals/scenarios/retrieval_baseline.py`, `src/evals/scenarios/longitudinal_memory.py`, and `src/evals/scenarios/policy_behavior.py` provide deterministic retrieval, continuity, contradiction-repair, and behavior scenarios
- `src/evals/scorecard.py` and `scripts/generate_memory_scorecard.py` generate runner-split scorecards and quality gates for retrieval, longitudinal, and behavior domains

Success criteria:

- PRs can be evaluated against stable memory/personality metrics
- regressions in continuity become detectable in CI or pre-merge workflows

## Sequenced Roadmap

## Phase 1: Foundations

Target window:
2 to 4 weeks

Primary outcome:
Establish canonical representations and retrieval infrastructure.

Status:
Implemented in the active runtime and test suite.

Deliverables:

- canonical memory schema
- event-to-memory encoder contract
- retrieval planner draft
- first reranker with diversity and confidence terms
- baseline eval scenarios for memory lookup and contradiction handling

Recommended implementation order:

1. canonical schema
2. retrieval planner
3. reranker
4. baseline eval harness

Exit criteria:

- all stores can normalize into the shared schema
- context assembly no longer relies on ad hoc dict-shape guessing alone
- a targeted retrieval evaluation suite exists

## Phase 2: Autobiographical Continuity

Target window:
4 to 8 weeks

Primary outcome:
Make the system feel like one persistent mind across sessions.

Status:
Largely implemented, with the main remaining gap in durable autobiographical graph and chapter persistence.

Deliverables:

- autobiographical graph model
- chapter summaries and defining moments
- per-user relationship memory subsystem
- relationship-aware retrieval and context injection
- contradiction clustering and belief revision policies

Exit criteria:

- the system can summarize what happened, what changed, and what matters now
- the system can describe its current relationship stance with a user from memory alone

## Phase 3: Personality Binding

Target window:
6 to 10 weeks

Primary outcome:
Turn internal state into consistent behavior.

Status:
Implemented in the active runtime, prompt assembly, and behavior evaluation paths.

Deliverables:

- response-policy composer
- separation of stable traits vs dynamic state
- prompt assembly with behavior controls
- explicit uncertainty, warmth, directness, and memory-disclosure policies

Exit criteria:

- personality is observable in outputs and traceable to explicit policy inputs
- users can distinguish different internal states or trait presets consistently

## Phase 4: Adaptive Memory And Believability

Target window:
8 to 12 weeks

Primary outcome:
Improve realism under long-running usage.

Status:
Implemented for the current deterministic reconsolidation, forgetting, longitudinal, and scorecard-gate scope; further realism work is now incremental rather than foundational.

Deliverables:

- reconsolidation after recall
- confidence drift and source weighting
- selective forgetting and memory suppression
- longitudinal eval harness with restart-aware scenarios
- human-believability test battery

Exit criteria:

- the system improves continuity without ballooning irrelevant recall
- memory and personality regressions are measurable over long horizons

## First 30 Days

### Week 1

- define the canonical memory schema
- map current store outputs to normalization adapters
- identify contradiction and provenance requirements

### Week 2

- implement retrieval planner skeleton
- refactor context assembly to consume normalized memory items
- add first focused retrieval quality tests

### Week 3

- introduce relationship-memory persistence model
- add autobiographical linking between episodes, projects, and people
- draft response-policy composer contract

### Week 4

- connect self-model, narrative, mood, and relationship state into response policy
- create longitudinal evaluation scenarios
- document metrics and add regression gates for core scenarios

Current status:

- deterministic restart-aware longitudinal scenarios now live in `src/evals/scenarios/longitudinal_memory.py`
- the initial suite scores restart continuity, contradiction repair, false-memory count, and over-recall rate in end-to-end tests
- the retrieval baseline now includes persisted runtime roundtrips for both vector LTM fact recall and episodic restart recall, alongside the fast fixture-backed scenarios
- the retrieval baseline also covers restart-persisted semantic contradiction repair and relationship-aware social recall using disk-backed relationship memory snapshots
- the scorecard now reports retrieval quality both in aggregate and split by runner so fixture-only and persisted-runtime regressions are visible separately
- retrieval runtime coverage now includes an explicit negative case proving quarantined contradiction facts stay hidden after restart
- the scorecard now splits retrieval, longitudinal, and behavior summaries by runner so fixture and persisted-runtime regressions can be separated quickly
- retrieval runtime coverage now also includes a real forgetting-policy case proving suppressed low-value facts stay hidden after restart
- retrieval runtime coverage now also includes a persisted episodic forgetting case proving suppressed low-value episodes stay hidden after restart
- retrieval runtime coverage now includes a persisted autobiographical continuity case that exercises chapter-aware reranking after restart
- the scorecard now reports runner-local gate failures by domain so drift can be spotted even before aggregate gating is adjusted
- deterministic scorecard generation now lives in `src/evals/scorecard.py` with a developer entrypoint at `scripts/generate_memory_scorecard.py`
- the scorecard now also summarizes policy behavior alignment, traceability, and replay stability using deterministic policy fixtures
- the behavior suite now includes multiple stubbed `CognitiveAgent.process_input()` runtime scenarios, including a restart-aware continuity case that only passes when persisted memory context survives across fresh agent instances
- the longitudinal suite now includes persisted `MemorySystem` roundtrips for restart continuity and contradiction repair, alongside the fixture-backed fast baseline scenarios

## Priority Backlog

1. Persist autobiographical graph and chapter state as a durable subsystem rather than deriving it only from the current episodic candidate set
2. Expand event-to-memory product generation so live turns can promote richer semantic, relationship, self-model, and chapter-linked outputs deterministically
3. Deepen chapter-aware retrieval so restart continuity depends on persisted autobiographical state rather than only on retrieved episodic overlap
4. Tighten cross-store adaptive-memory coupling so reconsolidation, contradiction repair, and forgetting share anchor semantics and provenance rules
5. Extend behavior and believability evaluation beyond deterministic fixtures into broader long-horizon and social-continuity scenarios
6. Harden quality gates around runner-local regressions so fixture-only passes cannot hide runtime drift

## Metrics That Matter

### Memory Quality

- recall precision for user facts, preferences, and episodic history
- contradiction detection and repair rate
- over-recall / irrelevant context rate
- false-memory rate
- autobiographical continuity across restarts

### Personality Quality

- identity stability across sessions
- relationship continuity and rapport consistency
- tone stability under equivalent contexts
- state-dependent behavioral coherence
- user-rated authenticity and trustworthiness

### System Quality

- bounded retrieval latency
- prompt budget efficiency
- memory-store health and decay quality
- stability under long-session workloads

## Risks

### Risk: More memory makes the system noisier

Mitigation:

- retrieval planner before adding more stores
- diversity-aware reranking
- strict working-set budget

### Risk: Personality becomes prompt theater instead of behavior

Mitigation:

- explicit response-policy layer
- behavior-focused evaluations, not just state snapshots

### Risk: Contradiction handling destabilizes the self-model

Mitigation:

- confidence-weighted revision
- source-aware belief tracking
- quarantine uncertain beliefs before promotion

### Risk: Complexity outruns testability

Mitigation:

- typed contracts and normalized interfaces
- scenario-based evals before broad expansion
- phase gates tied to measurable outcomes

## Recommended Next Step

Start with the next real continuity depth slice rather than repeating already-landed persistence work:

1. deepen continuity retrieval so persisted autobiographical chapters and defining moments are preferred earlier for continuity-style prompts instead of depending mostly on reassembled episodic overlap
2. extend longitudinal and scorecard evaluations so chapter-aware continuity must survive restart from persisted autobiographical state, not just from episodic-memory overlap
3. broaden the promoted-turn event-to-product pipeline beyond the current episodic, autobiographical, semantic-preference, semantic-focus, and prospective-follow-up outputs
