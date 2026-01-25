# Memory taxonomy (ownership + constraints)

This project treats **memory_system.py** as the only public API for memory systems.

Outside of the memory package, code must not directly import:
- semantic memory modules
- episodic memory modules
- procedural memory modules
- STM/LTM implementation modules

Use `src.memory.memory_system.MemorySystem` (and `MemorySystemConfig`) instead.

## Definitions (high level)

- **Short-Term Memory (STM)**: transient, limited-capacity working store for the current context window.
- **Long-Term Memory (LTM)**: persistent store for consolidated items; optimized for retrieval over time.
- **Episodic memory**: concrete, time-anchored records of specific experiences/events.
- **Semantic memory**: atemporal knowledge (facts/relationships) divorced from a specific time/place.
- **Procedural memory**: skills, procedures, and how-to sequences (action policies).
- **Prospective memory**: future intentions/reminders (things to do later).

## Ownership rules

### STM (Short-Term Memory)

**Can store**
- recent user/assistant turn fragments relevant to the current session
- transient working notes (short-lived summaries, scratchpad-like cues)
- short-lived preferences or constraints that are not yet validated

**Must never store**
- long-lived “facts about the world” that should be consolidated to semantic memory
- durable personal preferences/goals intended to persist across sessions (promote to LTM/semantic via consolidation)
- large documents or long transcripts (store references/IDs instead)

### LTM (Long-Term Memory)

**Can store**
- consolidated items promoted from STM (facts, preferences, stable summaries)
- durable user preferences/goals, if they’ve met promotion criteria
- stable artifacts (summaries, extracted constraints) useful for future retrieval

**Must never store**
- raw, high-frequency ephemeral working context (belongs in STM)
- event logs that require accurate timelines (belongs in episodic)

### Episodic memory

**Can store**
- specific events/experiences with time anchors (e.g., a session event, an action taken, a conversation milestone)
- contextual metadata about an episode (participants, location/channel, surrounding state)
- pointers to artifacts created during the episode (IDs/paths), plus a minimal description

**Must never store (hard constraint)**
- abstractions/generalizations/rules divorced from the specific episode (belongs in semantic/procedural/LTM)

Examples of forbidden episodic content:
- “The user usually prefers X” (a generalization)
- “Always do Y when Z” (a rule/procedure)

### Semantic memory

**Can store**
- atemporal facts and relationships (e.g., subject–predicate–object triples)
- stable preferences represented as timeless statements (e.g., “User prefers concise answers”)
- definitions, taxonomies, and conceptual relationships

**Must never store (hard constraint)**
- temporal data: timestamps, durations, sequences of events, or “when/then” narratives

Examples of forbidden semantic content:
- “On 2026-01-21 the user said …”
- “Yesterday we decided …”
- “First we did A, then B”

When a fact is learned inside an episode, the *episode record* (with time) belongs in episodic memory, while the *time-free fact* belongs in semantic memory.

### Procedural memory

**Can store**
- how-to knowledge: procedures, action sequences, strategies, policies
- conditional action patterns (preconditions → steps → expected outcome)
- tool-usage playbooks (e.g., how to run a workflow safely)

**Must never store**
- time-anchored logs of when a procedure was executed (episodic)
- factual world knowledge that is not procedural (semantic)

### Prospective memory

**Can store**
- reminders/intentions with due times or triggering conditions
- tasks the system must execute later or prompt the user about

**Must never store**
- generalized knowledge (semantic) or detailed event narratives (episodic)
- large bodies of content; store references and a short description

## Summary of the two key constraints

- **Semantic memory must not store temporal data.**
- **Episodic memory must not store abstractions.**
