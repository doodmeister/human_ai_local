# Phase 4 — Natural Conversational Interface

## Overview

The Chainlit UI in `scripts/chainlit_app/app.py` is still command-first and debug-heavy. The backend chat pipeline already does more than the UI currently exposes: it classifies natural-language intents, handles memory queries and reminder requests inside normal chat, detects goals, returns proactive reminders, and periodically attaches metacognitive snapshots.

Phase 4 should make Chainlit a conversation-first shell over the existing backend behavior. The main goal is not to add a second intent system in the UI. The main goal is to let normal chat use the backend as the source of truth, while slash commands remain explicit shortcuts and debug escape hatches.

Implementation status: all Phase 4 tasks complete as of 2026-04-15.

**Guiding principles**
- Normal chat goes through `/agent/chat`; backend intent handling is the primary path.
- Slash commands remain supported for power users and for features that do not yet have strong conversational handlers.
- The UI should surface notable state, not render every internal detail on every turn.
- Add backend payloads only when the UI truly lacks the signal it needs.

**Current backend capabilities to leverage**
- `/agent/chat` already returns `intent`, `intent_sections`, `intent_results`, `detected_goal`, `proactive_reminders`, `metrics`, `context_items`, and `captured_memories`.
- The backend chat service already has natural-language handlers for `memory_query`, `reminder_request`, `goal_creation`, and `system_status`.
- Periodic metacognitive snapshots are already attached to chat responses as `metacog`; richer diagnostics remain available via `/agent/metacog/*`.

---

## Task 1 — Backend-first conversational routing in `on_message`

Status: complete on 2026-04-14.

### Problem
`on_message` in `app.py` treats non-slash input as a generic `send_chat()` call, then mostly ignores the structured response fields the backend already returns. The earlier idea of adding a local `intent_classifier.py` in Chainlit would duplicate the classifier and handler logic that already exists in the backend chat pipeline.

### What to change

**File:** `scripts/chainlit_app/app.py` — `on_message` function

Restructure `on_message` into two clear paths:

1. **Explicit command path**
   - Keep existing slash command handling for `/memory`, `/remind`, `/goal`, `/dream`, `/reflect`, `/learning`, and `/metacog`.
   - These remain shortcuts and debug tools.
   - Do not remove them.

2. **Normal chat path**
   - For any non-slash message, send the content directly to `send_chat()`.
   - Treat the backend response as structured UI state, not just text plus metrics.
   - Use these fields intentionally:
     - `intent`: identify what the backend thought the user was trying to do.
     - `intent_sections` and `intent_results`: determine whether a conversational intent handler already produced a useful section.
     - `detected_goal`: surface goal execution actions.
     - `proactive_reminders`: surface due reminders and upcoming reminders.
     - `metrics`, `context_items`, `captured_memories`: render only when notable.

### Required constraint
Do **not** create `scripts/chainlit_app/intent_classifier.py` in this phase. The backend is already the source of truth for conversational intent. Adding a second classifier in Chainlit would create drift between the UI and the runtime behavior.

### Implementation shape

```python
@cl.on_message
async def on_message(message: cl.Message):
    cmd = getattr(message, "command", None) or ""
    content = message.content.strip()

    if cmd or content.startswith("/"):
        await _handle_command_message(cmd, content)
        return

    await _handle_chat_message(content)


async def _handle_chat_message(content: str):
    session_id = cl.user_session.get("session_id") or "default"
    flags = cl.user_session.get("flags") or {}
    salience = cl.user_session.get("salience_threshold")

    response = await send_chat(content, session_id, flags=flags, salience_threshold=salience)
    await _render_chat_response(response, session_id)
```

Add a helper such as `_render_chat_response()` that:
- sends the assistant reply;
- adds actions from `detected_goal`;
- surfaces due reminders;
- uses `intent` and `intent_sections` to make the UI feel intentional instead of generic.

### Notes
- If a future backend gap is proven, add the missing handler to the backend chat service first.
- Features that are still primarily operational rather than conversational, such as `/dream`, `/reflect`, `/learning`, and raw `/metacog`, can remain command-driven in this phase.

---

## Task 2 — Unified memory search for explicit memory browsing

Status: complete on 2026-04-15.

### Problem
The backend already handles conversational memory queries in normal chat, but the Chainlit `/memory` command still forces the user to know which memory system to inspect. That makes slash-mode browsing harder than it needs to be.

### What to change

**File:** `scripts/chainlit_app/app.py` — `_cmd_memory`  
**File:** `scripts/chainlit_app/george_api.py` — add new function

Add a unified search helper in `george_api.py` for the systems that currently support search over HTTP:

```python
async def unified_memory_search(query: str, max_per_system: int = 5) -> list[dict[str, Any]]:
    systems = ["stm", "ltm", "episodic"]
    tasks = [search_memories(system, query, max_results=max_per_system) for system in systems]
    tasks.append(fetch_semantic_facts(query))

    results_per_system = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[dict[str, Any]] = []
    labels = systems + ["semantic"]
    for label, result in zip(labels, results_per_system):
        if isinstance(result, Exception):
            continue
        for item in result:
            enriched = dict(item)
            enriched["_source_system"] = label
            merged.append(enriched)

    merged.sort(
        key=lambda item: (
            -float(item.get("relevance", item.get("similarity", 0.0))),
            -float(item.get("importance", item.get("activation", 0.0))),
        )
    )
    return merged[:20]
```

Update `_cmd_memory` to support both explicit system browsing and a unified mode:

- `/memory stm [query]`, `/memory ltm [query]`, `/memory episodic [query]`, `/memory semantic [query]`:
  keep the existing system-specific behavior.
- `/memory prospective` and `/memory procedural`:
  keep explicit browse/list behavior for those systems, because the current UI API surface exposes list endpoints rather than equivalent search endpoints.
- `/memory <query>` or `/memory all <query>`:
  run the unified fan-out search and render source labels.

### Rationale
- Normal chat memory requests already have a backend path; do not re-implement that in Chainlit.
- Unified search improves the explicit `/memory` tool without requiring the user to think in storage taxonomy.
- If this fan-out becomes generally useful outside Chainlit, move it into a backend endpoint later.

---

## Task 3 — Progressive disclosure on response metadata

Status: complete on 2026-04-15.

### Problem
The current chat path in `app.py` emits follow-up UI messages for metrics, context items, and captured memories on nearly every turn. That creates noise, especially when the values are routine or uninteresting.

### What to change

**File:** `scripts/chainlit_app/app.py` — normal chat response rendering block

Replace unconditional follow-up rendering with notable-only rules.

```python
# Metrics: only render when the turn has something worth calling out
if metrics:
    notable = []
    latency = metrics.get("turn_latency_ms")
    if latency is not None and latency > 500:
        notable.append(f"latency {latency:.0f}ms")

    consolidation = metrics.get("consolidation_status")
    if consolidation and consolidation not in {"none", "skipped"}:
        notable.append(consolidation)

    salience = metrics.get("user_salience")
    if salience is not None and salience >= 0.8:
        notable.append(f"salience={salience:.2f}")

    if notable:
        await cl.Message(content=" | ".join(notable), author="metrics", parent_id=msg.id).send()

# Context: only show when retrieval looks materially useful
if context_items and len(context_items) >= 3:
    lines = [f"Retrieved {len(context_items)} context items:"]
    for item in context_items[:6]:
        source = item.get("source_system", "?")
        snippet = str(item.get("content", ""))[:100]
        lines.append(f"- [{source}] {snippet}")
    await cl.Message(content="\n".join(lines), author="context", parent_id=msg.id).send()

# Captured memories: only show new captures, not routine reinforcement
new_captures = [item for item in captured if not item.get("reinforced")]
if new_captures:
    ...
```

### Additional cleanup
- The current `Intent: ...` line should not appear on every normal turn. Move intent display behind a debug flag, trace flag, or developer-only mode.
- Keep low-level diagnostics available when `Include_Trace` is enabled.

---

## Task 4 — Narrative metacognition as the default UI

Status: complete on 2026-04-15.

### Problem
`/metacog` in the Chainlit UI is still written like a diagnostic dump. That is useful for debugging, but not as the default conversational presentation.

### What to change

**File:** `scripts/chainlit_app/app.py` — `_cmd_metacog`

Split the metacog command into:
- a default narrative mode for human-readable self-report;
- a raw/debug mode for the current detailed dashboard.

```python
async def _cmd_metacog(content: str = ""):
    parts = content.split() if content else []
    raw_mode = any(flag in parts for flag in ["--raw", "debug", "verbose", "detailed"])

    session_id = cl.user_session.get("session_id") or "default"
    dashboard, tasks, reflections = await asyncio.gather(
        fetch_metacognition_dashboard(session_id=session_id, history_limit=10, limit=50),
        fetch_metacognition_tasks(session_id=session_id),
        fetch_metacognition_reflections(session_id=session_id, limit=10),
    )

    if raw_mode:
        await _render_metacog_raw(dashboard, tasks, reflections, session_id)
        return

    await _render_metacog_narrative(dashboard, tasks, reflections, session_id)
```

The narrative renderer should summarize:
- recent cycle success;
- unresolved contradictions or high contradiction rate;
- self-model drift when notable;
- pending or due background tasks;
- follow-up rate when it indicates cognitive load or unfinished work.

### Important nuance
The backend already has a conversational `system_status` intent. That means prompts like "How are you doing?" may already produce a useful status answer through normal chat. The Chainlit-side narrative renderer is still worth doing, but it should be the default presentation for explicit `/metacog`, not a replacement for the backend's own conversational status handling.

---

## Task 5 — Proactive suggestions from backend chat state

Status: complete on 2026-04-15.

### Problem
The backend chat response already includes proactive reminders and may include a metacognitive snapshot, but the UI only surfaces due reminders and ignores the rest of the structured state.

### What to change

**File:** `scripts/chainlit_app/app.py`

Add a post-response helper such as `_maybe_surface_suggestions(response, session_id)`.

Use existing response fields first:
- `response["proactive_reminders"]` for due and upcoming reminders;
- `response["metacog"]` when present;
- `response["metrics"]["consolidation_status"]`;
- `response["intent"]` where helpful for context.

Example behavior:
- if `metacog.stm_utilization >= 0.85`, suggest a dream cycle;
- if `metacog.performance.degraded` is true, offer diagnostics;
- if the turn already promoted or consolidated significant material, surface that passively rather than adding more prompts.

```python
async def _maybe_surface_suggestions(response: dict[str, Any], session_id: str):
    metrics = response.get("metrics") or {}
    metacog = response.get("metacog") or {}

    suggestions: list[tuple[str, str, str]] = []

    stm_util = metacog.get("stm_utilization")
    if isinstance(stm_util, (int, float)) and stm_util >= 0.85:
        suggestions.append((
            f"Short-term memory is at {stm_util:.0%} capacity.",
            "Run consolidation",
            "/dream",
        ))

    perf = metacog.get("performance") or {}
    if perf.get("degraded"):
        suggestions.append((
            "Response latency is elevated. Want the detailed diagnostic view?",
            "Open diagnostics",
            "/metacog --raw",
        ))

    for text, label, command in suggestions[:2]:
        await cl.Message(
            content=text,
            actions=[cl.Action(name="suggestion_action", label=label, payload={"command": command})],
            author="system",
        ).send()
```

Add an action callback that re-enters the existing command path.

### Backend dependency
Current chat responses only attach `metacog` on the configured cadence, not every turn. If reliable suggestioning needs every-turn signals, add a lightweight always-on payload block from the backend, such as `ui_signals` or `metacog_summary`, instead of building a second classifier or separate UI inference layer.

Suggested fields:
- `stm_utilization`
- `performance_degraded`
- `unresolved_contradiction_count`
- `pending_task_count`

---

## Task 6 — Auto-suggest dream and reflect without auto-running them

Status: complete on 2026-04-15.

### Problem
`/dream` and `/reflect` remain discoverability traps. Users should not have to memorize them, but the UI also should not silently start background cycles on their behalf.

### What to change

**File:** `scripts/chainlit_app/app.py`

Build on Task 5's suggestion infrastructure.

Add two classes of non-blocking suggestion:
1. **Pressure-triggered suggestions**
   - Suggest `/dream` when STM utilization is high or consolidation pressure is evident.
   - Suggest `/metacog` or `/reflect` when degradation or unresolved contradictions are present and the supporting signal is available.

2. **Session-shape suggestions**
   - Track `turn_count` and `last_dream_turn` in `cl.user_session`.
   - After a configurable number of turns, suggest a consolidation cycle if the session looks heavy enough.
   - Do not auto-run the cycle; ask.

Also add cooldown tracking so the same suggestion is not repeated within a short span.

### Constraint
Reflection suggestions should depend on real backend signals, not UI guesswork. If contradiction counts or task backlog are not present in the chat payload, fetch them sparingly from the metacog dashboard or add them to the backend summary payload.

---

## Task 7 — Adaptive settings cleanup

Status: complete on 2026-04-15.

### Problem
The current ChatSettings panel exposes a fixed salience slider. That makes the user manage a tuning parameter even though the backend already supports running without an explicit override.

### What to change

**File:** `scripts/chainlit_app/app.py` — `on_chat_start` and `on_settings_update`

Replace the raw slider with a simpler user intent:

```python
Select(
    id="Salience_Mode",
    label="Memory capture sensitivity",
    values=["adaptive", "capture_more", "capture_less"],
    initial_index=0,
    description="Adaptive uses backend defaults. Override only if you want broader or narrower capture.",
)
```

Map the mode in `on_settings_update`:

```python
salience_mode = settings.get("Salience_Mode", "adaptive")
salience_map = {
    "adaptive": None,
    "capture_more": 0.35,
    "capture_less": 0.75,
}
cl.user_session.set("salience_threshold", salience_map.get(salience_mode))
```

This matches the current `send_chat()` contract, which already treats `consolidation_salience_threshold` as optional.

### Optional follow-up
If the backend later exposes the effective adaptive threshold, show it as read-only UI state instead of restoring a raw slider.

---

## File change summary

| File | Changes |
|------|---------|
| `scripts/chainlit_app/app.py` | Restructure `on_message` into command vs chat paths, add backend-aware chat response rendering, add progressive disclosure, add narrative `/metacog`, add suggestion actions, simplify salience settings, update `/memory` behavior |
| `scripts/chainlit_app/george_api.py` | Add `unified_memory_search()` and, if needed, small helpers for metacog status/dashboard fallback calls |
| `src/interfaces/api/chat_endpoints.py` or chat payload builder | Optional follow-up only if the UI needs lightweight every-turn suggestion fields that are not currently in the chat payload |

---

## Implementation order

1. **Task 1** — Backend-first `on_message` restructuring
   - Make normal chat consume structured backend response fields deliberately.
2. **Task 3** — Progressive disclosure
   - Reduce noise before adding new UI affordances.
3. **Task 4** — Narrative `/metacog`
   - Turn diagnostics into a readable default presentation.
4. **Task 2** — Unified `/memory` fan-out
   - Improve explicit browsing without touching backend conversational logic.
5. **Task 5** — Suggestion infrastructure
   - Surface proactive backend state in the UI.
6. **Task 6** — Dream and reflect suggestion policies
   - Add cooldowns and session heuristics.
7. **Task 7** — Adaptive settings cleanup
   - Simplify settings after the main interaction model is stable.

---

## Testing notes

- Non-slash messages must continue to flow through `/agent/chat`; do not add a duplicate Chainlit intent classifier in this phase.
- Natural-language requests for memory, reminders, goals, and general system status should be handled by the existing backend intent pipeline and still produce a coherent UI presentation.
- All existing slash commands must continue to work unchanged.
- Unified `/memory` search must tolerate partial system failure; one failing source must not block the others.
- Progressive disclosure must not hide genuinely useful context. Verify turns with strong retrieval, high latency, or new memory captures still surface the right details.
- Narrative `/metacog` must degrade gracefully when dashboard fields are absent or sparse.
- Suggestion actions must never spam. Cap visible suggestions per turn and add a cooldown across nearby turns.
- If `metacog` is missing from a chat response because the interval was not reached, the UI should skip metacog-based suggestions or use a throttled fallback fetch.

---

## Acceptance criteria

- A user can ask conversationally about remembered information, reminders, goals, or system status and get a backend-driven result without knowing the slash command names.
- `/memory <query>` performs a cross-system search across the currently searchable systems, while `/memory stm <query>` and the other explicit system modes continue to work.
- `/metacog` defaults to a readable narrative self-report, and `/metacog --raw` still exposes the detailed debug view.
- Response metadata only appears when it contains notable information.
- Dream and diagnostic suggestions appear when supported by real backend signals, without auto-running disruptive actions.
- No separate Chainlit-side intent classifier is introduced for Phase 4.
- All slash commands continue to work for power users and debugging.
