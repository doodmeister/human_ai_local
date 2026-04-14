# Phase 4 — Natural Conversational Interface

## Overview

The Chainlit UI (`scripts/chainlit_app/app.py`) currently requires explicit slash commands and manual toggling for most cognitive features. The backend already returns intent detection, adaptive thresholds, and proactive signals — but the UI treats them as display-only artifacts rather than routing signals.

This phase rewires `on_message` to route through natural language intent before falling back to slash commands, unifies memory search across systems, and surfaces metacognitive state proactively instead of requiring manual `/metacog` invocation.

**Guiding principle:** The user should never need to know a slash command exists. Every feature reachable via `/command` must also be reachable by just talking naturally.

---

## Task 1 — Intent-routing pre-classifier in `on_message`

### Problem
`on_message` in `app.py` checks `cmd` attribute and `/prefix` strings. If neither matches, everything falls through to `send_chat()`. Natural requests like "What do you remember about the database?" or "Remind me in an hour to check the PR" go through the generic chat path and never hit the specialized handlers.

### What to change

**File:** `scripts/chainlit_app/app.py` — `on_message` function (line ~183)

Insert an intent classification step BEFORE the existing command routing block. The classifier should:

1. Accept the raw `message.content` string.
2. Return one of these intent categories with extracted parameters:
   - `memory_query` → extracted: `query` (str), optional `system` (str or None for fan-out)
   - `create_reminder` → extracted: `text` (str), `due_seconds` (int, parsed from NL time expression)
   - `create_goal` → extracted: `title` (str), optional `deadline` (str)
   - `list_reminders` → no params
   - `list_goals` → no params
   - `dream_request` → no params
   - `reflect_request` → no params
   - `metacog_query` → no params
   - `learning_query` → no params
   - `general_chat` → fallthrough, no routing

### Implementation approach

**Option A (lightweight, no extra LLM call):** Pattern-based classifier using keyword/regex matching. Create a new file `scripts/chainlit_app/intent_classifier.py`:

```python
import re
from dataclasses import dataclass
from typing import Any

@dataclass
class IntentResult:
    intent: str  # one of the categories above
    params: dict[str, Any]
    confidence: float  # 0.0–1.0

# Time expression patterns for reminder parsing
TIME_PATTERNS = [
    (r"in\s+(\d+)\s+min(?:ute)?s?", lambda m: int(m.group(1)) * 60),
    (r"in\s+(\d+)\s+hours?", lambda m: int(m.group(1)) * 3600),
    (r"in\s+half\s+an?\s+hour", lambda _: 1800),
    (r"in\s+an?\s+hour", lambda _: 3600),
    (r"tomorrow", lambda _: 86400),
]

# Keyword triggers — order matters, first match wins
INTENT_RULES = [
    # memory queries
    (r"(?:what do (?:you|I|we) (?:know|remember)|recall|search (?:my )?memor)", "memory_query"),
    (r"(?:remind me|set a reminder|don't let me forget)", "create_reminder"),
    (r"(?:(?:show|list|check) (?:my )?reminders|what reminders)", "list_reminders"),
    (r"(?:(?:show|list|check) (?:my )?goals|what (?:are my )?goals)", "list_goals"),
    (r"(?:I need to|I have to|my goal is|I want to achieve)", "create_goal"),
    (r"(?:run (?:a )?dream|consolidat|compress memor)", "dream_request"),
    (r"(?:how are you doing|cognitive state|how.*feeling|self.?report)", "metacog_query"),
    (r"(?:run (?:a )?reflect|self.?analy|introspect)", "reflect_request"),
    (r"(?:learning (?:metrics|dashboard|progress)|how.*learning)", "learning_query"),
]

def classify_intent(text: str) -> IntentResult:
    """Classify user message into an intent category."""
    lower = text.lower().strip()
    
    for pattern, intent in INTENT_RULES:
        if re.search(pattern, lower):
            params = _extract_params(intent, text, lower)
            return IntentResult(intent=intent, params=params, confidence=0.75)
    
    return IntentResult(intent="general_chat", params={}, confidence=1.0)
```

**Option B (higher accuracy, uses one LLM call):** Send a structured classification prompt to the already-configured LLM via `send_chat` with a system-level flag, or call the LLM provider directly. Only use this if pattern matching proves too brittle — start with Option A.

### Integration point in `on_message`

```python
@cl.on_message
async def on_message(message: cl.Message):
    cmd = getattr(message, "command", None) or ""
    content = message.content.strip()

    # --- NEW: Intent-based routing (before slash command checks) ---
    if not cmd and not content.startswith("/"):
        intent = classify_intent(content)
        if intent.intent != "general_chat" and intent.confidence >= 0.6:
            routed = await _route_by_intent(intent, content)
            if routed:
                return

    # --- Existing slash command routing (unchanged) ---
    if cmd == "memory" or content.startswith("/memory"):
        ...
```

Create `_route_by_intent()` that maps each intent to the existing `_cmd_*` handlers with extracted params. For `memory_query`, call the new unified search (Task 2). For `create_reminder`, parse the NL time and call `_cmd_create_reminder` with a synthesized content string. For `create_goal`, call `_cmd_create_goal`.

---

## Task 2 — Unified memory search with fan-out

### Problem
`/memory stm query` requires the user to know which of the 6 memory systems (stm, ltm, episodic, semantic, prospective, procedural) holds the information they want. Users don't think in memory-system taxonomy.

### What to change

**File:** `scripts/chainlit_app/app.py` — `_cmd_memory` function (line ~371)  
**File:** `scripts/chainlit_app/george_api.py` — add new function

Add a unified search function to `george_api.py`:

```python
async def unified_memory_search(query: str, max_per_system: int = 5) -> list[dict[str, Any]]:
    """Fan out a query across all searchable memory systems, merge by relevance."""
    systems = ["stm", "ltm", "episodic", "semantic"]
    tasks = [search_memories(system, query, max_results=max_per_system) for system in systems]
    # semantic facts use a different endpoint
    tasks.append(fetch_semantic_facts(query))
    
    results_per_system = await asyncio.gather(*tasks, return_exceptions=True)
    
    merged: list[dict[str, Any]] = []
    system_labels = systems + ["semantic_facts"]
    for system_label, result in zip(system_labels, results_per_system):
        if isinstance(result, Exception):
            continue
        for item in result:
            item["_source_system"] = system_label
            merged.append(item)
    
    # Sort by relevance score if available, else by importance/activation
    merged.sort(key=lambda m: (
        -float(m.get("relevance", m.get("similarity", 0.0))),
        -float(m.get("importance", m.get("activation", 0.0))),
    ))
    return merged[:20]
```

Modify `_cmd_memory` to use `unified_memory_search` when no system is specified:

```python
async def _cmd_memory(content: str):
    parts = content.split(maxsplit=2)
    system = parts[1] if len(parts) > 1 else None
    query = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 and parts[1] not in valid_systems else "")
    
    if system is None or system not in valid_systems:
        # Fan-out search across all systems
        query_text = " ".join(parts[1:]) if len(parts) > 1 else ""
        if not query_text:
            await cl.Message(content="What would you like to search for? Or specify a system: stm, ltm, episodic, semantic, prospective, procedural").send()
            return
        memories = await unified_memory_search(query_text)
        # ... render with source system tags
    else:
        # Existing system-specific path (unchanged)
        ...
```

When the intent classifier routes a `memory_query` intent, it should call this unified path with just the extracted query string.

---

## Task 3 — Proactive metacognitive suggestions

### Problem
The metacognition layer already computes STM utilization, contradiction counts, and performance degradation. The adaptive thresholds in `src/orchestration/metacognition/thresholds.py` define when action should be taken (STM ≥ 85%, performance degraded). But the UI never reads these signals to prompt the user.

### What to change

**File:** `scripts/chainlit_app/app.py` — end of the normal chat handler in `on_message` (after response rendering, line ~265)

After every `send_chat` response, check the returned metrics and proactive signals. Add a post-response advisor:

```python
async def _maybe_surface_suggestions(response: dict[str, Any], session_id: str):
    """Check response metrics and surface proactive suggestions when thresholds are crossed."""
    metrics = response.get("metrics") or {}
    metacog = response.get("metacog") or {}
    
    suggestions: list[tuple[str, str, str]] = []  # (message, action_label, action_command)
    
    # STM pressure — suggest consolidation
    stm_util = metrics.get("stm_utilization") or metacog.get("stm_utilization")
    if stm_util is not None and stm_util >= 0.85:
        suggestions.append((
            f"Short-term memory is at {stm_util:.0%} capacity.",
            "Run consolidation",
            "/dream",
        ))
    
    # Unresolved contradictions — surface them
    contradiction_count = metacog.get("contradiction_count", 0)
    if contradiction_count > 0:
        suggestions.append((
            f"{contradiction_count} unresolved contradiction{'s' if contradiction_count != 1 else ''} detected in recent context.",
            "Inspect",
            "/metacog",
        ))
    
    # Performance degradation
    if metrics.get("performance_degraded"):
        suggestions.append((
            "Response latency is elevated. Cognitive load may be high.",
            "Check diagnostics",
            "/metacog",
        ))
    
    # Consolidation happened — inform passively
    consolidation = metrics.get("consolidation_status")
    if consolidation and "promoted" in str(consolidation).lower():
        await cl.Message(
            content=f"Memory note: {consolidation}",
            author="system",
        ).send()
        return  # Don't pile on with other suggestions
    
    for message_text, label, command in suggestions[:2]:  # max 2 suggestions per turn
        actions = [cl.Action(
            name="suggestion_action",
            label=label,
            payload={"command": command},
        )]
        await cl.Message(content=message_text, actions=actions, author="system").send()
```

Add the action callback:

```python
@cl.action_callback("suggestion_action")
async def on_suggestion_action(action: cl.Action):
    command = action.payload.get("command", "")
    if command:
        # Synthesize a message object and re-route through on_message
        synthetic = cl.Message(content=command)
        await on_message(synthetic)
```

Call `_maybe_surface_suggestions(response, session_id)` at the end of the normal chat path in `on_message`, after `_surface_due_reminders`.

### Backend dependency
The `/agent/chat` response must include `metacog` data (STM utilization, contradiction count) in the response payload. Check `src/interfaces/api/` routes to confirm this is already included; if not, add it to the chat response schema.

---

## Task 4 — Conversational metacognition self-report

### Problem
`/metacog` dumps a dense wall of markdown stats plus a raw JSON block. This is useful for debugging but hostile as a user-facing feature. "How are you doing?" should produce a narrative answer.

### What to change

**File:** `scripts/chainlit_app/app.py` — `_cmd_metacog` function (line ~547)

Add a narrative rendering mode that is the default. Keep the current stat dump available via `/metacog --raw` or `/metacog debug`.

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
        # Existing stat-dump rendering (move current implementation here)
        await _render_metacog_raw(dashboard, tasks, reflections, session_id)
        return
    
    # Narrative self-report
    await _render_metacog_narrative(dashboard, tasks, reflections, session_id)


async def _render_metacog_narrative(dashboard, tasks, reflections, session_id):
    """Render metacognition state as a natural-language self-report."""
    if not dashboard or not dashboard.get("available"):
        await cl.Message(content="I don't have enough data for a self-report yet. Send a few messages first and I'll be able to tell you how things are going.").send()
        return
    
    scorecard = dashboard.get("scorecard", {})
    summary = scorecard.get("summary", {})
    contradictions = scorecard.get("contradictions", {})
    self_model = scorecard.get("self_model", {})
    background = dashboard.get("background", {})
    
    # Build narrative parts
    parts: list[str] = []
    
    # Overall health
    success_avg = summary.get("cycle_success_avg")
    if isinstance(success_avg, (int, float)):
        if success_avg >= 0.7:
            parts.append(f"I'm doing well — {success_avg:.0%} cycle success rate over recent turns.")
        elif success_avg >= 0.4:
            parts.append(f"I'm performing moderately — {success_avg:.0%} cycle success. There's room to improve.")
        else:
            parts.append(f"I'm struggling a bit — only {success_avg:.0%} cycle success. Something may need attention.")
    
    # Contradictions
    contradiction_rate = contradictions.get("contradiction_rate")
    unresolved = background.get("unresolved_contradiction_count", 0)
    if unresolved > 0:
        parts.append(f"I've noticed {unresolved} unresolved contradiction{'s' if unresolved != 1 else ''} in the context I'm working with.")
    elif isinstance(contradiction_rate, (int, float)) and contradiction_rate > 0.3:
        parts.append(f"Contradictions are showing up in about {contradiction_rate:.0%} of cycles — worth investigating.")
    
    # Self-model confidence
    drift = self_model.get("self_model_drift_avg")
    if isinstance(drift, (int, float)) and drift > 0.1:
        parts.append(f"My self-model has been shifting noticeably (drift: {drift:.2f}). My understanding of context may be evolving.")
    
    # Pending tasks
    pending = background.get("pending_task_count", 0)
    due = background.get("due_task_count", 0)
    if due > 0:
        parts.append(f"I have {due} background task{'s' if due != 1 else ''} that {'are' if due != 1 else 'is'} due now.")
    elif pending > 0:
        parts.append(f"There {'are' if pending != 1 else 'is'} {pending} background task{'s' if pending != 1 else ''} queued.")
    
    # Follow-up rate
    follow_up = summary.get("follow_up_rate")
    if isinstance(follow_up, (int, float)) and follow_up > 0.5:
        parts.append(f"About {follow_up:.0%} of recent cycles flagged follow-up work, which suggests some topics need deeper processing.")
    
    if not parts:
        parts.append("Things look stable. No notable issues in the recent cycle history.")
    
    narrative = " ".join(parts)
    
    actions = [cl.Action(
        name="suggestion_action",
        label="Show full diagnostics",
        payload={"command": "/metacog --raw"},
    )]
    await cl.Message(content=narrative, actions=actions).send()
```

---

## Task 5 — Progressive disclosure on response metadata

### Problem
Every chat response renders three nested Chainlit Steps (metrics, context items, captured memories) regardless of whether they contain meaningful data. This adds visual noise to routine turns.

### What to change

**File:** `scripts/chainlit_app/app.py` — response rendering block in `on_message` (lines ~237–265)

Replace the unconditional rendering with conditional checks:

```python
# Metrics — only show if something is notable
if metrics:
    notable_parts = []
    latency = metrics.get("turn_latency_ms")
    if latency is not None and latency > 500:  # Only surface slow turns
        notable_parts.append(f"latency {latency:.0f}ms")
    salience = metrics.get("user_salience")
    if salience is not None and salience >= 0.8:  # High-salience message
        notable_parts.append(f"salience={salience:.2f}")
    consolidation = metrics.get("consolidation_status")
    if consolidation and consolidation != "none":
        notable_parts.append(consolidation)
    if notable_parts:  # Only render if there's something worth showing
        await cl.Message(content=" | ".join(notable_parts), author="metrics", parent_id=msg.id).send()

# Context — only show if retrieval was meaningful (3+ items)
if context_items and len(context_items) >= 3:
    lines = [f"**Retrieved {len(context_items)} context items:**"]
    for item in context_items[:6]:  # Cap at 6, down from 10
        source = item.get("source_system", "?")
        snippet = str(item.get("content", ""))[:100]
        lines.append(f"- `[{source}]` {snippet}")
    await cl.Message(content="\n".join(lines), author="context", parent_id=msg.id).send()

# Captured memories — only show if something new was stored (not just reinforced)
new_captures = [m for m in captured if not m.get("reinforced")]
if new_captures:
    lines = [f"**Captured {len(new_captures)} new {'memory' if len(new_captures) == 1 else 'memories'}:**"]
    for mem in new_captures[:5]:
        tag = f"[{mem.get('memory_type', '')}] " if mem.get("memory_type") else ""
        lines.append(f"- {tag}{mem.get('content', '')}")
    await cl.Message(content="\n".join(lines), author="memory", parent_id=msg.id).send()
```

### Rationale
- Latency only matters to the user when it's slow (>500ms threshold).
- Salience is only interesting when the message was deemed highly salient.
- Context items below 3 are routine retrieval, not worth calling out.
- Reinforced memories are not new information — only show genuinely new captures.

---

## Task 6 — Auto-suggest dream/reflect based on backend state

### Problem
`/dream` and `/reflect` are manual commands. Users must remember they exist and know when to run them. The metacog layer already knows when consolidation pressure is high.

### What to change

**File:** `scripts/chainlit_app/app.py`

This is handled by Task 3's `_maybe_surface_suggestions` function. The STM utilization check already suggests consolidation. Two additional triggers to add:

1. **Turn-count heuristic**: After every N turns (configurable, default 15), check if a dream cycle has been run this session. If not, suggest it.

```python
# In on_message, after send_chat:
turn_count = cl.user_session.get("turn_count", 0) + 1
cl.user_session.set("turn_count", turn_count)
last_dream_turn = cl.user_session.get("last_dream_turn", 0)

if turn_count - last_dream_turn >= 15:
    # Check STM item count from metrics
    stm_hits = (response.get("metrics") or {}).get("stm_hits")
    if stm_hits and stm_hits > 10:
        actions = [cl.Action(
            name="suggestion_action",
            label="Run consolidation",
            payload={"command": "/dream"},
        )]
        await cl.Message(
            content="You've had a productive session. A memory consolidation cycle might help strengthen key memories.",
            actions=actions,
            author="system",
        ).send()
```

2. **Post-dream, track the turn**: When `/dream` completes, set `cl.user_session.set("last_dream_turn", turn_count)`.

---

## Task 7 — Remove manual salience slider in favor of adaptive thresholds

### Problem
The ChatSettings panel exposes a manual salience threshold slider (0–1). The metacog layer in `src/orchestration/metacognition/` already dynamically adjusts consolidation thresholds based on STM utilization and performance state (see `docs/metacog_features.md`: threshold increases +0.05 when STM ≥ 85% or performance degraded, capped at 0.85). The manual slider conflicts with the adaptive behavior.

### What to change

**File:** `scripts/chainlit_app/app.py` — `on_chat_start` ChatSettings block (line ~120)

Replace the Salience slider with a simpler toggle:

```python
Select(
    id="Salience_Mode",
    label="Memory capture sensitivity",
    values=["adaptive", "capture_more", "capture_less"],
    initial_index=0,
    description="Adaptive lets the system tune itself. Override if you want more or fewer memories captured.",
),
```

In `on_settings_update`, map the selection:

```python
salience_mode = settings.get("Salience_Mode", "adaptive")
salience_map = {
    "adaptive": None,       # Don't send threshold — let backend adapt
    "capture_more": 0.35,   # Low threshold = capture more
    "capture_less": 0.75,   # High threshold = only emphatic
}
cl.user_session.set("salience_threshold", salience_map.get(salience_mode))
```

In `send_chat`, only include `consolidation_salience_threshold` in the payload if it's not None (adaptive mode omits it, letting the backend's adaptive logic take over).

---

## File change summary

| File | Changes |
|------|---------|
| `scripts/chainlit_app/intent_classifier.py` | **NEW** — Pattern-based intent classifier with NL time parsing |
| `scripts/chainlit_app/app.py` | Modify `on_message` to call intent classifier before slash routing. Add `_route_by_intent()`, `_maybe_surface_suggestions()`, `_render_metacog_narrative()`. Modify `_cmd_memory` for fan-out. Modify `_cmd_metacog` for narrative mode. Modify response rendering for progressive disclosure. Add turn-count dream suggestion. Simplify salience setting. Add `suggestion_action` callback. |
| `scripts/chainlit_app/george_api.py` | Add `unified_memory_search()` function |

---

## Implementation order

1. **Task 1** — Intent classifier + routing (highest impact, enables natural conversation)
2. **Task 2** — Unified memory search (removes the most confusing user-facing decision)
3. **Task 5** — Progressive disclosure (quick cleanup, reduces noise immediately)
4. **Task 4** — Narrative metacognition (transforms `/metacog` from debug tool to conversational feature)
5. **Task 3** — Proactive suggestions (requires Task 1 working so suggestions can re-enter the router)
6. **Task 6** — Auto-suggest dream/reflect (builds on Task 3's suggestion infrastructure)
7. **Task 7** — Adaptive salience (small change, do last since it touches settings schema)

---

## Testing notes

- All existing `/slash` commands must continue to work unchanged. Intent routing is additive, not replacing.
- Test intent classifier with edge cases: "I remember something about databases" (should NOT trigger memory search — user is providing info, not querying), "Remember to check the PR" (reminder, not memory query).
- Unified memory search should gracefully handle individual system failures (one system down should not block results from others).
- Proactive suggestions should never fire more than 2 per turn and should not repeat the same suggestion within 3 turns (track in session state).
- Narrative metacog should degrade gracefully when scorecard fields are None/missing.
- The `/metacog --raw` escape hatch must always produce the full stat dump for debugging.

---

## Acceptance criteria

- A user can type "What do you remember about the project timeline?" and get memory results without knowing `/memory` exists.
- A user can type "Remind me in 2 hours to review the deployment" and get a reminder created without knowing `/remind` syntax.
- A user can type "How are you doing?" and get a narrative cognitive self-report.
- STM pressure above 85% triggers a visible suggestion to run consolidation within the next chat turn.
- Response metadata (metrics, context, captures) only appears when it contains notable information.
- All slash commands continue to work for power users who prefer them.
