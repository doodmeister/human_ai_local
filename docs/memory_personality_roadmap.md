# World-Class Memory And Personality Roadmap

Last updated: 2026-03-28

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

The main gap is not missing modules. The main gap is that memory encoding, retrieval, autobiographical continuity, and personality control are still too shallow or too loosely coupled to produce convincingly human-like continuity.

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

Success criteria:

- PRs can be evaluated against stable memory/personality metrics
- regressions in continuity become detectable in CI or pre-merge workflows

## Sequenced Roadmap

## Phase 1: Foundations

Target window:
2 to 4 weeks

Primary outcome:
Establish canonical representations and retrieval infrastructure.

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
- deterministic scorecard generation now lives in `src/evals/scorecard.py` with a developer entrypoint at `scripts/generate_memory_scorecard.py`

## Priority Backlog

1. Canonical memory schema and normalization layer
2. Query-aware retrieval planner
3. Cross-store reranker with diversity and confidence signals
4. Relationship memory subsystem
5. Autobiographical graph and chapter model
6. Contradiction tracking and belief revision
7. Response-policy composer
8. Reconsolidation after recall
9. Longitudinal evaluation harness
10. Human-believability benchmark pack

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

Start with the architecture in `memory_personality_architecture.md` and implement the first foundation slice:

1. canonical memory schema
2. retrieval planner skeleton
3. evaluation scenarios for fact recall and contradiction repair
