# World-Class Memory And Personality Architecture

Last updated: 2026-03-29

## Purpose

This document defines the target architecture for a world-class memory system and personality system built on top of the current repository.

It is intended to answer four questions:

1. What should the system remember?
2. How should it retrieve and revise memory?
3. How should personality and internal state shape behavior?
4. How do we evolve the current implementation without breaking the runtime?

This is a target design, not a claim that all components already exist.

## Design Goals

- preserve continuity across turns and sessions
- maintain a stable but revisable identity
- remember social relationships and prior commitments
- forget selectively instead of retaining everything forever
- distinguish facts, episodes, beliefs, and self-narrative
- keep retrieval bounded, explainable, and measurable
- make personality visible in behavior, not just descriptions

## Non-Goals

- unlimited raw transcript retention as a substitute for memory
- unconstrained prompt stuffing
- monolithic “one vector store solves everything” design
- personality defined only by a static system prompt

## Architectural Principles

### 1. One Event, Multiple Memory Products

A single interaction should not map to a single stored blob.

Each significant event may produce:

- a raw turn trace
- an episodic event record
- one or more semantic beliefs
- a relationship update
- a self-model update
- a goal or prospective-memory update

### 2. Retrieval Must Be Planned

The system should decide what kind of memory it needs before it searches.

### 3. Personality Is A Control Plane

Personality should shape response policy directly through stable traits, dynamic state, and relationship context.

### 4. Memory Is Revisable

Recall, contradiction, new evidence, and reflection should change memory structure and confidence.

### 5. Every Important Decision Should Be Inspectable

Why a memory was selected, why a contradiction was flagged, and why a personality stance was chosen should all be available in traces or diagnostics.

## Target Top-Level Flow

1. Input interpretation
2. Event encoding
3. Memory-product generation
4. Retrieval planning
5. Candidate retrieval
6. Cross-store reranking
7. Working-self composition
8. Response-policy composition
9. LLM synthesis
10. Reconsolidation and reflection

## Core Subsystems

## 1. Input Interpretation Layer

Purpose:
Convert raw user input into structured cognitive signals.

Inputs:

- raw text
- session state
- recent interaction state

Outputs:

- utterance intent
- entities and references
- temporal expressions
- propositions and negations
- affective appraisal
- social significance
- uncertainty markers

Current repo touchpoints:

- `src/orchestration/chat/intent_classifier_v2.py`
- `src/orchestration/chat/memory_capture.py`
- `src/cognition/attention/attention_manager.py`

Target additions:

- proposition extractor
- entity resolver
- contradiction candidate detector
- social-act classifier

## 2. Event Encoder

Purpose:
Turn interpreted input plus context into a normalized event object.

Suggested event fields:

- `event_id`
- `session_id`
- `actor`
- `target_actor`
- `timestamp`
- `utterance_type`
- `propositions`
- `entities`
- `time_refs`
- `affect`
- `salience`
- `goal_relevance`
- `relationship_relevance`
- `source`
- `confidence`

This layer is the bridge between chat logic and memory logic.

## 3. Canonical Memory Schema

Purpose:
Normalize all stored and retrieved memory items across memory systems.

Suggested canonical fields:

- `memory_id`
- `memory_kind`
  values: `trace`, `episodic`, `semantic`, `relationship`, `self_model`, `goal`, `prospective`
- `content`
- `summary`
- `entities`
- `subject`
- `predicate`
- `object`
- `time_interval`
- `encoding_time`
- `last_access`
- `confidence`
- `importance`
- `emotional_valence`
- `arousal`
- `source`
- `source_event_ids`
- `contradiction_set_id`
- `relationship_target`
- `goal_ids`
- `narrative_role`
- `tags`
- `metadata`

Rules:

- every store-specific retrieval must be mappable into this schema
- every context item should be derivable from this schema without ad hoc shape guessing
- provenance and contradiction fields are mandatory for mutable beliefs

## 4. Memory Products

### Raw Trace Memory

Purpose:
Preserve exact interaction evidence.

Use for:

- debugging
- provenance
- high-fidelity episode reconstruction

### Episodic Memory

Purpose:
Store what happened, when, in what context.

Use for:

- autobiographical continuity
- narrative chapters
- relationship history

### Semantic / Belief Memory

Purpose:
Store generalized facts, preferences, stable user data, and world knowledge.

Use for:

- durable recall
- contradiction resolution
- preference consistency

### Relationship Memory

Purpose:
Store interaction-specific social state for a user or interlocutor.

Examples:

- familiarity
- trust
- warmth
- unresolved rupture
- recurring collaboration style
- known boundaries or preferences in conversation

### Self-Model Memory

Purpose:
Store persistent self-beliefs, stable traits, current vulnerabilities, and narrative discoveries.

### Goal / Prospective Memory

Purpose:
Track pending obligations, reminders, and goal-linked commitments.

## 5. Store Roles

### STM

Role:
high-speed working memory and local recency buffer

Should optimize for:

- rapid recall
- activation dynamics
- rehearsal and decay
- temporary task coherence

Should not be the main source of durable identity.

### LTM

Role:
durable semantic and generalized memory

Should optimize for:

- belief retrieval
- preference stability
- confidence and source tracking
- long-horizon access patterns

### Episodic Store

Role:
autobiographical event history

Should optimize for:

- time-aware retrieval
- sequence continuity
- chaptering and episode links

### Relationship Store

Role:
per-user interpersonal continuity

Should optimize for:

- user-specific recall
- social-policy conditioning
- rapport and trust updates

### Prospective Store

Role:
future commitments and reminders

Should optimize for:

- due detection
- importance ranking
- link-back to goals, users, and promises

## 6. Retrieval Planner

Purpose:
Route each query or turn to the right memory retrieval strategy.

Planner inputs:

- current intent
- entities and temporal expressions
- whether the task is factual, episodic, social, or self-referential
- current relationship target
- active goals and reminders

Planner outputs:

- which stores to query
- per-store limits
- reranking weights
- fallback policy

Example strategies:

- factual question about the user → semantic + relationship
- “what happened last week?” → episodic + chapter graph
- “how are you feeling?” → self-model + mood + narrative
- “remind me later” → prospective + goal linkage

## 7. Candidate Retrieval And Normalization

All store-specific search results should pass through:

1. store adapter
2. canonical normalizer
3. provenance annotator
4. contradiction checker

This replaces brittle shape-specific context assembly.

## 8. Cross-Store Reranker

Purpose:
Choose the working set that actually reaches the model.

Suggested scoring terms:

- task relevance
- semantic similarity
- recency
- confidence
- importance
- emotional salience
- relationship relevance
- novelty/diversity
- narrative continuity value
- contradiction risk

Rules:

- no duplicate or near-duplicate items in the final working set
- maintain source diversity
- reserve budget for social or self-model context when relevant

## 9. Working-Self Composer

Purpose:
Assemble current internal state into a structured “working self” for the turn.

Components:

- drives
- felt sense / mood
- relationship stance toward current interlocutor
- active goals and obligations
- self-model snapshot
- narrative chapter context
- uncertainty posture

Output:

a bounded, structured state object used by policy composition and response generation

## 10. Response-Policy Composer

Purpose:
Turn working-self state into explicit behavioral controls.

Suggested policy fields:

- warmth
- directness
- curiosity
- assertiveness
- apology tendency
- uncertainty disclosure style
- memory disclosure policy
- value priorities
- conflict stance
- help-vs-boundary tradeoff

This is the missing bridge between “personality state exists” and “outputs consistently feel like one person.”

## 11. LLM Synthesis Layer

The model should receive distinct input blocks, not just one generic memory dump.

Suggested message structure:

1. stable role and safety system prompt
2. current response policy
3. current working-self state
4. relevant autobiographical and relationship context
5. relevant factual and episodic memories
6. current user message

Generation should be constrained by policy rather than assuming the LLM will infer personality from raw memory snippets.

Current implementation note:

- prompt assembly now uses bounded source blocks for role, policy, working-self state, and retrieved memory context
- policy blocks are optional when no response-policy snapshot is available, but the message shape remains inspectable
- retrieved memory is rendered as context, not instruction text, to reduce prompt leakage from flat memory dumps

## 12. Reconsolidation And Revision

Every retrieval can update memory.

Mechanisms:

- strengthen or weaken confidence after successful or failed recall
- merge similar beliefs into abstractions
- split contradictory beliefs into tracked alternatives
- update self-model or relationship model when repeated evidence appears
- create narrative links when an event becomes identity-relevant

Current implementation note:

- recalled STM, LTM, and episodic items now support explicit reconsolidation feedback through the memory facade
- successful recall can reinforce importance, confidence, access metadata, or episodic consolidation strength
- correction-style feedback can weaken metadata without requiring backend-specific callers

## 13. Reflection Layer

Reflection remains useful, but should not be the only adaptation mechanism.

Reflection should:

- inspect memory health
- summarize relationship trends
- detect identity drift
- identify stale or contradictory beliefs
- recommend consolidation, forgetting, or clarification actions

## 14. Evaluation Architecture

The target architecture must include a quality system, not only runtime components.

Core eval suites:

- fact retention
- preference continuity
- episodic chronology
- contradiction repair
- relationship continuity
- self-model stability
- narrative coherence
- false-memory resistance
- restart continuity
- human-believability scoring

## Mapping To Current Repository

### Current Strengths

- memory facades and services already exist
- cognitive layers already compute self-model and narrative state
- context builder already injects personality-adjacent context
- reflection and metacognition already exist as infrastructure

### Current Gaps

- no canonical memory schema across all stores
- retrieval is still heuristic and blended rather than planned
- relationship memory is not a first-class persistent store
- generation policy is underpowered relative to the richness of internal state
- contradiction and reconsolidation are not first-class architectural concepts

## Proposed Module Additions

Suggested new packages over time:

```text
src/memory/schema/
  canonical.py
  normalization.py
  contradiction.py

src/memory/encoding/
  event_encoder.py
  proposition_extractor.py
  relationship_encoder.py
  self_model_encoder.py

src/memory/retrieval/
  planner.py
  reranker.py
  retrieval_plan.py

src/memory/relationship/
  store.py
  model.py
  updater.py

src/orchestration/policy/
  response_policy.py
  policy_composer.py
  policy_rendering.py

src/evals/
  scenarios/
  metrics/
  runners/
```

## Migration Strategy

### Slice 1

- add canonical schema
- normalize retrieval outputs
- keep existing store implementations intact

### Slice 2

- introduce retrieval planner while retaining existing search backends
- move context builder to consume planned retrieval outputs

### Slice 3

- add relationship store and autobiographical graph links
- begin persisting per-user social state

### Slice 4

- add response-policy composer
- connect personality state to generation

### Slice 5

- add reconsolidation and contradiction handling
- add longitudinal evaluation harness

## Definition Of Success

This architecture is successful when:

- the system can explain why it recalled a memory
- the system can maintain preferences and identity over long horizons
- the system can form user-specific relationships that survive restarts
- the system can notice and manage contradiction rather than hiding it
- the system behaves consistently enough that users experience it as one continuous mind
