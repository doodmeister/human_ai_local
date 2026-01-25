# Cognitive Theory (Phase 0 — Source of Truth)

## What “cognition” means in this system

In this repository, **cognition** is the end-to-end process that turns an incoming user message and an evolving internal state into:

- a coherent, goal-aligned response,
- an updated internal state (memories, metrics, plans, reminders), and
- controlled, repeatable behavior under constraints (context limits, latency, dependency availability).

Cognition here is not “a single model thinking.” It is a **pipeline of cooperating subsystems** that:

1) interpret the current turn,
2) assemble a bounded context (“working set”),
3) decide what matters now (attention/executive control),
4) ask an LLM to produce language and/or actions,
5) capture outcomes back into memory, and
6) adapt thresholds/behaviors based on load and feedback.

The defining property is **closed-loop control**: each turn both consumes and updates state.

## The canonical cognitive loop

The canonical loop is “turn-based” and repeats for every user message (or tool/event that is treated as a turn):

1. **Perception (Input + Parse)**
   - Receive the user’s message and relevant system signals (time, session metadata, load/telemetry).
   - Normalize into internal turn structures.

2. **Working-Set Construction (Context Building)**
   - Collect **recent conversation turns**.
   - Retrieve from memory systems (see below):
     - short-term/working memories,
     - long-term semantic memories,
     - episodic/situation-specific items,
     - prospective reminders (due/upcoming),
     - any configured external knowledge sources.
   - Score and rank candidate context items.
   - Truncate to a strict budget (a “working set”), preserving top-ranked items.

3. **Attention Allocation (Prioritization Under Constraint)**
   - Apply attention rules/metrics (fatigue/capacity/load) to determine:
     - what is salient now,
     - how aggressive retrieval should be,
     - whether to simplify/fallback.
   - Attention is the mechanism that makes bounded context behave like bounded cognition.

4. **Executive Control (Goal/Decision/Plan/Schedule as Needed)**
   - Decide **what to do** (respond, ask a question, plan steps, schedule tasks, create reminders).
   - When enabled/needed, convert intents into structured outputs:
     - decisions among options,
     - plans (action sequences),
     - schedules (time/resource constrained).
   - Executive control is responsible for coherence across turns and for aligning actions with goals.

5. **LLM Synthesis (Language + Optional Tool Intent)**
   - Provide the LLM with the bounded working set.
   - The LLM produces:
     - the user-facing response, and
     - any structured directives the system chooses to support (e.g., tool calls, reminder creation intents).

6. **Reflection & Capture (State Update)**
   - Extract salient facts/preferences/goals/outcomes from the turn.
   - Store to short-term memory immediately; promote/cluster into long-term memory when gating conditions are met.
   - Update metrics/telemetry (retrieval counts, attention load, truncation decisions, planning/scheduling stats).

7. **Adaptive Control (Meta-cognition)**
   - Adjust retrieval thresholds and fallback behavior for the next turn (especially under load).
   - Preserve safety and reliability: degraded paths must still produce a usable response.

This loop is canonical: implementations may vary, but any change to the system should be judged against whether it preserves this loop’s intent.

## Roles of memory, attention, executive functions, and LLMs

### Memory (persistence + recall)

Memory provides continuity across turns and sessions.

- **Short-Term Memory (STM / working memory)**
  - Small capacity, fast access.
  - Captures the most recent salient items and activations.
  - Used to stabilize local coherence (“what we were just doing”).

- **Long-Term Memory (LTM / semantic memory)**
  - Larger store, slower/structured retrieval.
  - Holds stable facts, preferences, and learned associations.
  - Retrieved when relevant to the current turn; maintained with decay/health checks.

- **Episodic memory (situation-specific context)**
  - Captures events and their context (“what happened when”), enabling narrative continuity.
  - Retrieval is driven by similarity and recency.

- **Prospective memory (future intentions/reminders)**
  - Holds reminders with due times or triggers.
  - Injects due/upcoming reminders into the working set as high-priority context.

Core principle: memory is only useful if it is **selectively recalled** into the bounded working set; raw persistence without retrieval discipline creates noise.

### Attention (bounded cognition)

Attention mediates scarcity:

- enforces that only a limited set of items enter the working set,
- prioritizes items under time/context constraints,
- measures load (fatigue/capacity) and moderates how much retrieval/planning to attempt.

In other words, attention is the system’s mechanism for making “too much available information” behave like “a manageable present focus.”

### Executive functions (control + alignment)

Executive functions ensure that behavior is purposeful and consistent.

They:

- select strategies (e.g., fast response vs. deliberative planning),
- make decisions when trade-offs exist,
- generate or repair plans,
- schedule tasks under constraints,
- manage goals and success criteria,
- coordinate across subsystems without requiring the LLM to invent structure.

Executive outputs are treated as **control signals** for the loop, not as prose.

### LLMs (generative interface, not the architecture)

LLMs are used for:

- natural language understanding and generation,
- summarization and synthesis from the working set,
- producing structured intents when the system asks for them.

LLMs are **not** the source of truth for cognition in this repo. They operate inside the constraints and signals defined by the pipeline:

- They receive a bounded, curated working set.
- They are expected to be fallible; the system must degrade gracefully.
- The system’s cognitive identity is defined by the loop and subsystems, not by any single model.

## System-level constraints (non-negotiable)

- **Bounded context**: The working set must be scored and truncated to a configured maximum; unlimited accumulation is disallowed.
- **Graceful degradation**: If optional heavy dependencies are missing/unavailable, the loop must still function via fallbacks.
- **Turn-based state updates**: The system must capture and persist relevant outcomes each turn; otherwise cognition cannot improve.
- **Telemetry-first**: Cognitive behavior must be measurable (retrieval counts, truncation, planning/scheduling performance) to support adaptive control.

This document is the Phase 0 lock-in: any future code change should be consistent with this canonical loop and role separation.
