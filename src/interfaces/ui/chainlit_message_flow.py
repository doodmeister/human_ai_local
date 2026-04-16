"""Helpers for Chainlit command parsing and backend-driven UI actions."""

from __future__ import annotations

from typing import Any, Iterable


KNOWN_CHAINLIT_COMMANDS = frozenset(
    {
        "memory",
        "reminders",
        "remind",
        "goals",
        "goal",
        "dream",
        "reflect",
        "learning",
        "metacog",
    }
)

SEARCHABLE_MEMORY_SYSTEMS = frozenset({"stm", "ltm", "episodic", "semantic", "procedural"})
LIST_ONLY_MEMORY_SYSTEMS = frozenset({"prospective"})
ALL_MEMORY_SYSTEMS = frozenset(SEARCHABLE_MEMORY_SYSTEMS | LIST_ONLY_MEMORY_SYSTEMS)

DREAM_SUGGESTION_INTERVAL_TURNS = 15
REFLECT_SUGGESTION_INTERVAL_TURNS = 12
REFLECTION_DASHBOARD_REFRESH_TURNS = 6
SUGGESTION_COOLDOWN_TURNS = 3

SALIENCE_MODE_VALUES = ("adaptive", "capture_more", "capture_less")
SALIENCE_THRESHOLD_BY_MODE = {
    "adaptive": None,
    "capture_more": 0.35,
    "capture_less": 0.75,
}


def synthesize_command_content(command_name: str, content: str) -> str:
    """Ensure command-driven messages always include the slash command prefix."""
    normalized_command = str(command_name or "").strip().lower()
    normalized_content = str(content or "").strip()
    if not normalized_command:
        return normalized_content
    if not normalized_content:
        return f"/{normalized_command}"
    if normalized_content.startswith("/"):
        return normalized_content
    return f"/{normalized_command} {normalized_content}".strip()


def normalize_salience_mode(mode: Any) -> str:
    """Normalize memory capture sensitivity mode to a supported value."""
    normalized = str(mode or "").strip().lower()
    if normalized in SALIENCE_THRESHOLD_BY_MODE:
        return normalized
    return "adaptive"


def salience_threshold_for_mode(mode: Any) -> float | None:
    """Return the backend salience threshold override for a UI salience mode."""
    normalized = normalize_salience_mode(mode)
    return SALIENCE_THRESHOLD_BY_MODE[normalized]


def normalize_command_request(
    command: str | None,
    content: str,
    *,
    known_commands: Iterable[str] = KNOWN_CHAINLIT_COMMANDS,
) -> tuple[str | None, str]:
    """Return a normalized command name and content if the message is command-like."""
    known = {str(name).strip().lower() for name in known_commands}
    normalized_content = str(content or "").strip()
    normalized_command = str(command or "").strip().lower()

    if normalized_command in known:
        return normalized_command, synthesize_command_content(normalized_command, normalized_content)

    if not normalized_content.startswith("/"):
        return None, normalized_content

    token = normalized_content[1:].split(maxsplit=1)[0].strip().lower()
    if token in known:
        return token, normalized_content
    return None, normalized_content


def get_primary_intent_type(intent_payload: dict[str, Any] | None) -> str | None:
    """Return the canonical intent type from backend chat payloads."""
    if not isinstance(intent_payload, dict):
        return None
    intent_type = intent_payload.get("type") or intent_payload.get("intent_type")
    if intent_type is None:
        return None
    normalized = str(intent_type).strip()
    return normalized or None


def collect_response_intents(
    intent_sections: Iterable[dict[str, Any]] | None,
    intent_results: Iterable[dict[str, Any]] | None,
) -> tuple[str, ...]:
    """Collect unique handled intent names from response sections and execution logs."""
    collected: list[str] = []

    for section in intent_sections or ():
        if not isinstance(section, dict):
            continue
        intent_name = str(section.get("intent") or "").strip()
        if intent_name and intent_name not in collected:
            collected.append(intent_name)

    for result in intent_results or ():
        if not isinstance(result, dict) or not result.get("handled"):
            continue
        intent_name = str(result.get("intent") or "").strip()
        if intent_name and intent_name not in collected:
            collected.append(intent_name)

    return tuple(collected)


def build_command_action_specs(
    *,
    primary_intent: str | None,
    handled_intents: Iterable[str],
    proactive_reminders: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Map backend response state into lightweight UI command actions."""
    handled = {str(name).strip() for name in handled_intents if str(name).strip()}
    proactive = proactive_reminders or {}
    upcoming = proactive.get("upcoming") or []
    due = proactive.get("due") or []

    specs: list[dict[str, str]] = []
    if "reminder_request" in handled or upcoming or due:
        specs.append({"label": "Show reminders", "command": "/reminders", "icon": "bell"})
    if primary_intent == "system_status" or "system_status" in handled:
        specs.append({"label": "Open diagnostics", "command": "/metacog", "icon": "compass"})
    if primary_intent == "goal_query" or "goal_query" in handled:
        specs.append({"label": "Show goals", "command": "/goals", "icon": "target"})

    deduped: list[dict[str, str]] = []
    seen_commands: set[str] = set()
    for spec in specs:
        command_name = spec["command"]
        if command_name in seen_commands:
            continue
        seen_commands.add(command_name)
        deduped.append(spec)
    return deduped


def build_backend_suggestion_specs(response: dict[str, Any] | None) -> list[dict[str, str]]:
    """Build suggestion messages from backend chat state already returned to the UI."""
    payload = response or {}
    metrics = payload.get("metrics") or {}
    metacog = payload.get("metacog") or {}
    proactive = payload.get("proactive_reminders") or {}

    suggestions: list[dict[str, str]] = []

    stm_util = _safe_float(metacog.get("stm_utilization"))
    if stm_util is not None and stm_util >= 0.85:
        suggestions.append(
            {
                "message": f"Short-term memory is at {stm_util:.0%} capacity. A consolidation cycle could help.",
                "label": "Run consolidation",
                "command": "/dream",
                "icon": "moon",
            }
        )

    performance = metacog.get("performance") or {}
    performance_degraded = bool(performance.get("degraded"))
    slow_turn = _safe_float(metrics.get("turn_latency_ms"))
    if performance_degraded or (slow_turn is not None and slow_turn >= 1500):
        suggestions.append(
            {
                "message": "Response latency is elevated. Want the detailed diagnostic view?",
                "label": "Open diagnostics",
                "command": "/metacog --raw",
                "icon": "compass",
            }
        )

    due = proactive.get("due") or []
    upcoming = proactive.get("upcoming") or []
    if upcoming and not due:
        noun = "reminder" if len(upcoming) == 1 else "reminders"
        suggestions.append(
            {
                "message": f"You have {len(upcoming)} {noun} coming up soon.",
                "label": "Show reminders",
                "command": "/reminders",
                "icon": "bell",
            }
        )

    return suggestions


def build_response_artifact_sections(
    response: dict[str, Any] | None,
    *,
    include_trace: bool = False,
) -> dict[str, list[str]]:
    """Build reply artifact sections with progressive disclosure by default."""
    payload = response or {}
    metrics = payload.get("metrics") or {}
    context_items = payload.get("context_items") or []
    captured = payload.get("captured_memories") or []
    intent = payload.get("intent") or {}

    return {
        "metrics": _build_metrics_artifact_lines(metrics, intent=intent, include_trace=include_trace),
        "context": _build_context_artifact_lines(context_items, include_trace=include_trace),
        "memory": _build_memory_artifact_lines(captured, include_trace=include_trace),
    }


def should_refresh_reflection_dashboard(
    *,
    turn_count: int,
    last_reflect_turn: int,
    reflect_interval: int = REFLECT_SUGGESTION_INTERVAL_TURNS,
    refresh_interval: int = REFLECTION_DASHBOARD_REFRESH_TURNS,
) -> bool:
    """Return whether the UI should fetch dashboard state for reflection suggestioning."""
    return (
        turn_count > 0
        and turn_count % refresh_interval == 0
        and turn_count - max(last_reflect_turn, 0) >= reflect_interval
    )


def build_maintenance_suggestion_specs(
    *,
    turn_count: int,
    last_dream_turn: int,
    last_reflect_turn: int,
    response: dict[str, Any] | None,
    dashboard: dict[str, Any] | None = None,
    dream_interval: int = DREAM_SUGGESTION_INTERVAL_TURNS,
    reflect_interval: int = REFLECT_SUGGESTION_INTERVAL_TURNS,
) -> list[dict[str, str]]:
    """Build lower-frequency dream and reflection suggestions from session state."""
    payload = response or {}
    metrics = payload.get("metrics") or {}
    metacog = payload.get("metacog") or {}
    suggestions: list[dict[str, str]] = []

    if turn_count - max(last_dream_turn, 0) >= dream_interval:
        stm_util = _safe_float(metacog.get("stm_utilization"))
        stm_hits = _safe_int(metrics.get("stm_hits"))
        if (stm_util is not None and stm_util >= 0.70) or stm_hits >= 10:
            suggestions.append(
                {
                    "message": "You have had a productive session. A consolidation cycle could help strengthen the most useful memories.",
                    "label": "Run consolidation",
                    "command": "/dream",
                    "icon": "moon",
                }
            )

    dashboard_payload = dashboard or {}
    if (
        dashboard_payload.get("available")
        and turn_count - max(last_reflect_turn, 0) >= reflect_interval
    ):
        background = dashboard_payload.get("background") or {}
        scorecard = dashboard_payload.get("scorecard") or {}
        summary = scorecard.get("summary") or {}

        unresolved = _safe_int(background.get("unresolved_contradiction_count"))
        due_tasks = _safe_int(background.get("due_task_count"))
        pending_tasks = _safe_int(background.get("pending_task_count"))
        follow_up_rate = _safe_float(summary.get("follow_up_rate"))

        message = ""
        if unresolved > 0:
            noun = "contradiction" if unresolved == 1 else "contradictions"
            message = f"I still see {unresolved} unresolved {noun}. A reflection pass could help inspect them."
        elif due_tasks > 0:
            noun = "task" if due_tasks == 1 else "tasks"
            message = f"I have {due_tasks} background {noun} due now. A reflection pass could help prioritize them."
        elif pending_tasks >= 3:
            noun = "task" if pending_tasks == 1 else "tasks"
            message = f"I have {pending_tasks} background {noun} queued. A reflection pass could help decide what matters most next."
        elif follow_up_rate is not None and follow_up_rate >= 0.50:
            message = f"About {follow_up_rate:.0%} of recent cycles are generating follow-up work. A reflection pass could help consolidate that backlog."

        if message:
            suggestions.append(
                {
                    "message": message,
                    "label": "Run reflection",
                    "command": "/reflect",
                    "icon": "scan-eye",
                }
            )

    return suggestions


def filter_suggestion_specs(
    suggestions: Iterable[dict[str, str]],
    *,
    existing_commands: Iterable[str],
    last_suggested_turns: dict[str, int] | None,
    turn_count: int,
    cooldown_turns: int = SUGGESTION_COOLDOWN_TURNS,
) -> list[dict[str, str]]:
    """Deduplicate suggestions against reply actions and per-command cooldowns."""
    filtered: list[dict[str, str]] = []
    seen_commands = {str(command).strip() for command in existing_commands if str(command).strip()}
    recent = last_suggested_turns or {}

    for spec in suggestions:
        command = str(spec.get("command") or "").strip()
        if command:
            if command in seen_commands:
                continue
            last_turn = _safe_int(recent.get(command))
            if last_turn and turn_count - last_turn < cooldown_turns:
                continue
            seen_commands.add(command)
        filtered.append(spec)
    return filtered


def parse_memory_command_content(content: str) -> dict[str, str | None]:
    """Parse `/memory` content into help, explicit system, or unified-search modes."""
    normalized_content = str(content or "").strip()
    parts = normalized_content.split()
    if not parts:
        return {"mode": "help", "system": None, "query": ""}

    first_token = parts[0].lstrip("/").lower()
    args = parts[1:] if first_token == "memory" else parts
    if not args:
        return {"mode": "help", "system": None, "query": ""}

    first_arg = args[0].strip().lower()
    if first_arg == "all":
        query = " ".join(args[1:]).strip()
        if not query:
            return {"mode": "help", "system": None, "query": ""}
        return {"mode": "unified", "system": None, "query": query}

    if first_arg in ALL_MEMORY_SYSTEMS:
        query = " ".join(args[1:]).strip()
        return {"mode": "system", "system": first_arg, "query": query}

    query = " ".join(args).strip()
    if not query:
        return {"mode": "help", "system": None, "query": ""}
    return {"mode": "unified", "system": None, "query": query}


def is_metacog_raw_request(content: str) -> bool:
    """Return whether `/metacog` should render the detailed diagnostics view."""
    parts = {token.strip().lower() for token in str(content or "").split()}
    return any(flag in parts for flag in {"--raw", "debug", "verbose", "detailed"})


def build_metacog_narrative(
    dashboard: dict[str, Any] | None,
    *,
    tasks: Iterable[dict[str, Any]] | None = None,
    reflections: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Summarize metacognitive dashboard state as a natural-language self-report."""
    dashboard_payload = dashboard or {}
    if not dashboard_payload.get("available"):
        return (
            "I do not have enough metacognitive data for a self-report yet. "
            "Send a few messages first and I will be able to summarize how things are going."
        )

    scorecard = dashboard_payload.get("scorecard") or {}
    summary = scorecard.get("summary") or {}
    contradictions = scorecard.get("contradictions") or {}
    self_model = scorecard.get("self_model") or {}
    background = dashboard_payload.get("background") or {}

    pending_tasks = _safe_int(background.get("pending_task_count"))
    due_tasks = _safe_int(background.get("due_task_count"))
    unresolved = _safe_int(background.get("unresolved_contradiction_count"))
    if pending_tasks == 0 and tasks is not None:
        pending_tasks = len(list(tasks))
    idle_reflections = _safe_int(background.get("idle_reflection_count"))
    recent_reflections = len(list(reflections)) if reflections is not None else 0
    reflection_count = idle_reflections or recent_reflections

    parts: list[str] = []

    success_avg = _safe_float(summary.get("cycle_success_avg"))
    if success_avg is not None:
        if success_avg >= 0.70:
            parts.append(f"I am doing well - {success_avg:.0%} cycle success over recent turns.")
        elif success_avg >= 0.40:
            parts.append(f"I am doing moderately - {success_avg:.0%} cycle success over recent turns.")
        else:
            parts.append(f"I am under some strain - only {success_avg:.0%} cycle success over recent turns.")

    contradiction_rate = _safe_float(contradictions.get("contradiction_rate"))
    if unresolved > 0:
        noun = "contradiction" if unresolved == 1 else "contradictions"
        parts.append(f"I still have {unresolved} unresolved {noun} in the current context.")
    elif contradiction_rate is not None and contradiction_rate > 0.30:
        parts.append(f"Contradictions are appearing in about {contradiction_rate:.0%} of recent cycles.")

    drift = _safe_float(self_model.get("self_model_drift_avg"))
    if drift is not None and drift > 0.10:
        parts.append(
            f"My self-model has been shifting noticeably (drift {drift:.2f}), so context interpretation may still be moving."
        )

    if due_tasks > 0:
        noun = "task" if due_tasks == 1 else "tasks"
        parts.append(f"I have {due_tasks} background {noun} due now.")
    elif pending_tasks > 0:
        noun = "task" if pending_tasks == 1 else "tasks"
        parts.append(f"I have {pending_tasks} background {noun} queued.")

    follow_up_rate = _safe_float(summary.get("follow_up_rate"))
    if follow_up_rate is not None and follow_up_rate > 0.50:
        parts.append(f"About {follow_up_rate:.0%} of recent cycles flagged follow-up work.")

    if not parts and reflection_count > 0:
        noun = "reflection" if reflection_count == 1 else "reflections"
        if idle_reflections > 0:
            parts.append(f"Things look stable right now. I have logged {reflection_count} recent idle {noun}.")
        else:
            parts.append(f"Things look stable right now. I have logged {reflection_count} recent {noun}.")

    if not parts:
        return "Things look stable right now. No notable issues stand out in the recent cycle history."
    return " ".join(parts)


def _build_metrics_artifact_lines(
    metrics: dict[str, Any],
    *,
    intent: dict[str, Any] | None,
    include_trace: bool,
) -> list[str]:
    if include_trace:
        parts: list[str] = []
        latency = _safe_float(metrics.get("turn_latency_ms"))
        if latency is not None:
            parts.append(f"latency {latency:.0f}ms")
        stm_hits = metrics.get("stm_hits")
        if stm_hits is not None:
            parts.append(f"STM={stm_hits}")
        ltm_hits = metrics.get("ltm_hits")
        if ltm_hits is not None:
            parts.append(f"LTM={ltm_hits}")
        procedural_hits = metrics.get("procedural_hits")
        if procedural_hits is not None:
            parts.append(f"PROC={procedural_hits}")
        salience = _safe_float(metrics.get("user_salience"))
        if salience is not None:
            parts.append(f"salience={salience:.2f}")
        consolidation = str(metrics.get("consolidation_status") or "").strip()
        if consolidation:
            parts.append(consolidation)

        lines: list[str] = [" | ".join(parts)] if parts else []
        intent_type = get_primary_intent_type(intent)
        if intent_type:
            confidence = _safe_float((intent or {}).get("confidence"))
            if confidence is not None:
                lines.append(f"Intent: **{intent_type}** ({confidence:.0%})")
            else:
                lines.append(f"Intent: **{intent_type}**")
        return lines

    notable: list[str] = []
    latency = _safe_float(metrics.get("turn_latency_ms"))
    if latency is not None and latency > 500:
        notable.append(f"latency {latency:.0f}ms")

    salience = _safe_float(metrics.get("user_salience"))
    if salience is not None and salience >= 0.80:
        notable.append(f"salience={salience:.2f}")

    consolidation = str(metrics.get("consolidation_status") or "").strip().lower()
    if consolidation and consolidation not in {"none", "pending", "skipped"}:
        notable.append(consolidation)

    return [" | ".join(notable)] if notable else []


def _build_context_artifact_lines(context_items: list[dict[str, Any]], *, include_trace: bool) -> list[str]:
    if not context_items:
        return []

    if include_trace:
        lines = ["**Context used (STM -> LTM):**"]
        for item in context_items[:10]:
            source = item.get("source_system", "?")
            reason = str(item.get("reason") or "").strip()
            snippet = str(item.get("content", ""))[:120]
            line = f"- `[{source}]` {snippet}"
            if reason:
                line += f"  _{reason}_"
            lines.append(line)
        return lines

    if len(context_items) < 3:
        return []

    lines = [f"**Retrieved {len(context_items)} context items:**"]
    for item in context_items[:6]:
        source = item.get("source_system", "?")
        snippet = str(item.get("content", ""))[:100]
        lines.append(f"- `[{source}]` {snippet}")
    return lines


def _build_memory_artifact_lines(captured: list[dict[str, Any]], *, include_trace: bool) -> list[str]:
    if not captured:
        return []

    if include_trace:
        lines = ["**Captured memories:**"]
        for mem in captured[:8]:
            tag = f"[{mem.get('memory_type', '')}] " if mem.get("memory_type") else ""
            extras: list[str] = []
            if mem.get("reinforced"):
                extras.append("reinforced")
            if mem.get("contradiction"):
                extras.append("contradiction")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"- {tag}{mem.get('content', '')}{suffix}")
        return lines

    new_captures = [item for item in captured if not item.get("reinforced")]
    if not new_captures:
        return []

    noun = "memory" if len(new_captures) == 1 else "memories"
    lines = [f"**Captured {len(new_captures)} new {noun}:**"]
    for mem in new_captures[:5]:
        tag = f"[{mem.get('memory_type', '')}] " if mem.get("memory_type") else ""
        suffix = " (contradiction)" if mem.get("contradiction") else ""
        lines.append(f"- {tag}{mem.get('content', '')}{suffix}")
    return lines


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0
