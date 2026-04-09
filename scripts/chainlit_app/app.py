"""George Cognitive Chat — Chainlit interface.

Replaces the Streamlit UI with an event-driven, chat-first experience.
Run via:  chainlit run scripts/chainlit_app/app.py -w
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from george_api import (
    API_BASE,
    check_health,
    complete_reminder,
    create_goal,
    create_reminder,
    delete_reminder,
    execute_goal,
    fetch_execution_outcomes,
    fetch_experiments,
    fetch_learning_metrics,
    fetch_metacognition_dashboard,
    fetch_metacognition_reflections,
    fetch_metacognition_tasks,
    fetch_memories,
    fetch_procedural_memories,
    fetch_prospective_memories,
    fetch_semantic_facts,
    format_due_display,
    list_goals,
    list_reminders,
    search_memories,
    send_chat,
    trigger_dream,
    trigger_reflection,
    update_llm_config,
)


# ============================================================================
# Lifecycle Hooks
# ============================================================================

@cl.set_starters
async def set_starters(user=None, language=None):
    """Suggest conversation starters on an empty chat."""
    return [
        cl.Starter(
            label="Set a goal",
            message="I need to finish the project report by Friday",
            icon="target",
        ),
        cl.Starter(
            label="Check my memories",
            message="/memory stm",
            icon="brain",
        ),
        cl.Starter(
            label="Dream cycle",
            message="/dream",
            icon="moon",
        ),
        cl.Starter(
            label="Show reminders",
            message="/reminders",
            icon="bell",
        ),
        cl.Starter(
            label="Inspect metacognition",
            message="/metacog",
            icon="compass",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new session."""
    session_id = uuid.uuid4().hex[:12]
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("flags", {
        "include_memory": True,
        "include_attention": True,
        "include_trace": False,
        "reflection": False,
    })
    cl.user_session.set("salience_threshold", 0.55)
    cl.user_session.set("snooze_minutes", 15)

    # Set up commands for the input bar
    await cl.context.emitter.set_commands([
        {"id": "memory", "icon": "brain", "description": "Browse a memory system (stm, ltm, episodic, semantic, prospective, procedural)", "button": True, "persistent": True},
        {"id": "reminders", "icon": "bell", "description": "List active reminders", "button": True, "persistent": True},
        {"id": "remind", "icon": "alarm-clock", "description": "Create a reminder: /remind <minutes> <text>", "button": False, "persistent": False},
        {"id": "goals", "icon": "target", "description": "List active goals", "button": True, "persistent": True},
        {"id": "goal", "icon": "plus", "description": "Create a goal: /goal <title>", "button": False, "persistent": False},
        {"id": "dream", "icon": "moon", "description": "Run a dream consolidation cycle", "button": True, "persistent": False},
        {"id": "reflect", "icon": "scan-eye", "description": "Run metacognitive reflection", "button": True, "persistent": False},
        {"id": "learning", "icon": "chart-line", "description": "Show learning dashboard metrics", "button": True, "persistent": False},
        {"id": "metacog", "icon": "compass", "description": "Show metacognition dashboard summary", "button": True, "persistent": False},
    ])

    # Chat settings (gear icon)
    await cl.ChatSettings([
        Select(
            id="LLM_Provider",
            label="LLM Provider",
            values=["openai", "ollama"],
            initial_index=0,
            description="Backend LLM provider",
        ),
        Select(
            id="OpenAI_Model",
            label="OpenAI Model",
            values=["gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            initial_index=0,
        ),
        Slider(
            id="Salience",
            label="STM Consolidation Threshold",
            initial=0.55,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Lower = capture more memories, higher = only emphatic messages",
        ),
        Slider(
            id="Snooze_Minutes",
            label="Default Snooze (minutes)",
            initial=15,
            min=5,
            max=120,
            step=5,
        ),
        Switch(
            id="Include_Memory",
            label="Include memory retrieval",
            initial=True,
        ),
        Switch(
            id="Include_Attention",
            label="Include attention signals",
            initial=True,
        ),
        Switch(
            id="Include_Trace",
            label="Include trace details",
            initial=False,
        ),
    ]).send()

    # Check backend health
    healthy = await check_health()
    if healthy:
        await cl.Message(
            content=f"Connected to backend at `{API_BASE}`.  Session `{session_id}`.\n\nType a message to chat, or use a **command** (click the `/` icon) for goals, reminders, memory browsing, and more.",
            author="system",
        ).send()
    else:
        await cl.Message(
            content=f"Backend at `{API_BASE}` is **not reachable**.\n\nStart it with `python main.py api`, then refresh this page.",
            author="system",
        ).send()


@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]):
    """Apply changed settings."""
    cl.user_session.set("salience_threshold", settings.get("Salience", 0.55))
    cl.user_session.set("snooze_minutes", int(settings.get("Snooze_Minutes", 15)))
    cl.user_session.set("flags", {
        "include_memory": settings.get("Include_Memory", True),
        "include_attention": settings.get("Include_Attention", True),
        "include_trace": settings.get("Include_Trace", False),
        "reflection": False,
    })

    # Push LLM config to backend
    provider = settings.get("LLM_Provider", "openai")
    model = settings.get("OpenAI_Model", "gpt-4.1-nano")
    result = await update_llm_config(provider=provider, openai_model=model)
    if result.get("status") == "ok":
        await cl.Message(content=f"LLM config updated: **{provider}** / `{model}`", author="system").send()


# ============================================================================
# Message Handler
# ============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    """Handle every user message — route commands or chat."""
    cmd = getattr(message, "command", None) or ""
    content = message.content.strip()

    # ------- Command routing -------
    if cmd == "memory" or content.startswith("/memory"):
        await _cmd_memory(content)
        return
    if cmd == "reminders" or content.startswith("/reminders"):
        await _cmd_list_reminders()
        return
    if cmd == "remind" or content.startswith("/remind"):
        await _cmd_create_reminder(content)
        return
    if cmd == "goals" or content.startswith("/goals"):
        await _cmd_list_goals()
        return
    if cmd == "goal" or content.startswith("/goal"):
        await _cmd_create_goal(content)
        return
    if cmd == "dream" or content.startswith("/dream"):
        await _cmd_dream()
        return
    if cmd == "reflect" or content.startswith("/reflect"):
        await _cmd_reflect()
        return
    if cmd == "learning" or content.startswith("/learning"):
        await _cmd_learning()
        return
    if cmd == "metacog" or content.startswith("/metacog"):
        await _cmd_metacog()
        return

    # ------- Normal chat -------
    session_id = cl.user_session.get("session_id") or "default"
    flags = cl.user_session.get("flags") or {}
    salience = cl.user_session.get("salience_threshold")

    # Show a thinking indicator via a Step
    async with cl.Step(name="George thinking", type="llm") as step:
        step.input = content
        try:
            response = await send_chat(
                content, session_id, flags=flags, salience_threshold=salience
            )
        except Exception as exc:
            await cl.Message(content=f"Chat request failed: `{exc}`").send()
            return
        step.output = response.get("response", "")

    reply_text = response.get("response") or "[Empty response]"

    # Build action buttons for any detected goal
    actions: List[cl.Action] = []
    detected_goal = response.get("detected_goal")
    if detected_goal:
        goal_id = detected_goal.get("goal_id", "")
        actions.append(cl.Action(
            name="execute_goal",
            label=f"Execute: {detected_goal.get('title', 'Goal')[:40]}",
            icon="play",
            payload={"goal_id": goal_id},
        ))

    # Send the reply
    msg = cl.Message(content=reply_text, actions=actions)
    await msg.send()

    # Render context items, captured memories, and metrics as nested Steps
    context_items = response.get("context_items") or []
    captured = response.get("captured_memories") or []
    metrics = response.get("metrics") or {}
    intent = response.get("intent") or {}

    elements: List[str] = []
    if metrics:
        parts = []
        if metrics.get("turn_latency_ms") is not None:
            parts.append(f"latency {metrics['turn_latency_ms']:.0f}ms")
        if metrics.get("stm_hits") is not None:
            parts.append(f"STM={metrics['stm_hits']}")
        if metrics.get("ltm_hits") is not None:
            parts.append(f"LTM={metrics['ltm_hits']}")
        if metrics.get("user_salience") is not None:
            parts.append(f"salience={metrics['user_salience']:.2f}")
        if metrics.get("consolidation_status"):
            parts.append(metrics["consolidation_status"])
        if parts:
            elements.append(" | ".join(parts))

    if intent.get("intent_type"):
        elements.append(f"Intent: **{intent['intent_type']}** ({intent.get('confidence', 0):.0%})")

    if elements:
        await cl.Message(content="\n".join(elements), author="metrics", parent_id=msg.id).send()

    if context_items:
        lines = ["**Context used (STM -> LTM):**"]
        for item in context_items[:10]:
            source = item.get("source_system", "?")
            reason = item.get("reason", "")
            snippet = str(item.get("content", ""))[:120]
            lines.append(f"- `[{source}]` {snippet}  _{reason}_")
        await cl.Message(content="\n".join(lines), author="context", parent_id=msg.id).send()

    if captured:
        lines = ["**Captured memories:**"]
        for mem in captured[:8]:
            tag = f"[{mem.get('memory_type', '')}] " if mem.get("memory_type") else ""
            extra = ""
            if mem.get("reinforced"):
                extra = " (reinforced)"
            if mem.get("contradiction"):
                extra = " (contradiction)"
            lines.append(f"- {tag}{mem.get('content', '')}{extra}")
        await cl.Message(content="\n".join(lines), author="memory", parent_id=msg.id).send()

    # Surface due reminders proactively
    proactive = response.get("proactive_reminders") or {}
    due = proactive.get("due") or []
    if due:
        await _surface_due_reminders(due)


# ============================================================================
# Action Callbacks
# ============================================================================

@cl.action_callback("execute_goal")
async def on_execute_goal(action: cl.Action):
    goal_id = action.payload.get("goal_id")
    if not goal_id:
        return
    await cl.Message(content=f"Executing goal `{goal_id}`...").send()
    result = await execute_goal(goal_id)
    if result and result.get("status") == "success":
        ctx = result.get("context", {})
        lines = [
            f"Goal **{ctx.get('goal_title', goal_id)}** executed.",
            f"- Status: {ctx.get('status', 'UNKNOWN')}",
            f"- Decision: {ctx.get('decision_time_ms', 0)}ms",
            f"- Planning: {ctx.get('planning_time_ms', 0)}ms ({ctx.get('total_actions', 0)} actions)",
            f"- Scheduling: {ctx.get('scheduling_time_ms', 0)}ms ({ctx.get('scheduled_tasks', 0)} tasks)",
        ]
        await cl.Message(content="\n".join(lines)).send()
    else:
        await cl.Message(content="Goal execution failed or returned no result.").send()


@cl.action_callback("complete_reminder")
async def on_complete_reminder(action: cl.Action):
    rid = action.payload.get("reminder_id")
    if rid and await complete_reminder(rid):
        await cl.Message(content=f"Reminder `{rid[:8]}...` marked complete.", author="system").send()


@cl.action_callback("snooze_reminder")
async def on_snooze_reminder(action: cl.Action):
    rid = action.payload.get("reminder_id")
    content_text = action.payload.get("content", "Snoozed reminder")
    minutes = cl.user_session.get("snooze_minutes") or 15
    if rid:
        await delete_reminder(rid)
        await create_reminder(content_text, minutes * 60)
        label = f"{minutes}min"
        await cl.Message(content=f"Reminder snoozed for {label}.", author="system").send()


@cl.action_callback("delete_reminder")
async def on_delete_reminder(action: cl.Action):
    rid = action.payload.get("reminder_id")
    if rid and await delete_reminder(rid):
        await cl.Message(content="Reminder deleted.", author="system").send()


# ============================================================================
# Slash Commands
# ============================================================================

async def _cmd_memory(content: str):
    """Browse a memory system.  Usage: /memory <system> [query]"""
    parts = content.split(maxsplit=2)
    system = parts[1] if len(parts) > 1 else "stm"
    query = parts[2] if len(parts) > 2 else ""
    valid = {"stm", "ltm", "episodic", "semantic", "prospective", "procedural"}
    if system not in valid:
        await cl.Message(content=f"Unknown system `{system}`. Choose from: {', '.join(sorted(valid))}").send()
        return

    async with cl.Step(name=f"Fetching {system} memories", type="tool") as step:
        if system == "semantic":
            memories = await fetch_semantic_facts(query or None)
        elif system == "prospective":
            memories = await fetch_prospective_memories()
        elif system == "procedural":
            memories = await fetch_procedural_memories()
        elif query:
            memories = await search_memories(system, query)
        else:
            memories = await fetch_memories(system)
        step.output = f"Found {len(memories)} items"

    if not memories:
        await cl.Message(content=f"No memories in **{system}**." + (f" Query: _{query}_" if query else "")).send()
        return

    lines = [f"**{system.upper()} Memories** ({len(memories)} items):\n"]
    for i, mem in enumerate(memories[:20], 1):
        text = mem.get("content") or mem.get("detailed_content") or mem.get("summary") or str(mem)
        text = str(text)[:200]
        importance = mem.get("importance")
        activation = mem.get("activation")
        extras = []
        if importance is not None:
            extras.append(f"imp={importance:.2f}")
        if activation is not None:
            extras.append(f"act={activation:.2f}")
        suffix = f"  _({', '.join(extras)})_" if extras else ""
        lines.append(f"{i}. {text}{suffix}")

    await cl.Message(content="\n".join(lines)).send()


async def _cmd_list_reminders():
    """List all active reminders with action buttons."""
    reminders = await list_reminders()
    if not reminders:
        await cl.Message(content="No active reminders.").send()
        return

    for r in reminders:
        rid = r.get("id", "")
        content_text = r.get("content", "Reminder")
        due = format_due_display(r)
        actions = [
            cl.Action(name="complete_reminder", label="Complete", icon="check", payload={"reminder_id": rid}),
            cl.Action(name="snooze_reminder", label="Snooze", icon="alarm-clock", payload={"reminder_id": rid, "content": content_text}),
            cl.Action(name="delete_reminder", label="Delete", icon="trash-2", payload={"reminder_id": rid}),
        ]
        await cl.Message(
            content=f"**{content_text}** — {due}",
            actions=actions,
        ).send()


async def _cmd_create_reminder(content: str):
    """Create a reminder.  Usage: /remind <minutes> <text>"""
    parts = content.split(maxsplit=2)
    if len(parts) < 3:
        await cl.Message(content="Usage: `/remind <minutes> <text>`\nExample: `/remind 30 Review the pull request`").send()
        return
    try:
        minutes = int(parts[1])
    except ValueError:
        await cl.Message(content=f"Invalid minutes: `{parts[1]}`").send()
        return
    text = parts[2]
    reminder = await create_reminder(text, minutes * 60)
    if reminder:
        due = format_due_display(reminder)
        await cl.Message(content=f"Reminder created: **{text}** — {due}").send()
    else:
        await cl.Message(content="Failed to create reminder.").send()


async def _cmd_list_goals():
    """List active goals with execute actions."""
    goals = await list_goals()
    if not goals:
        await cl.Message(content="No active goals. Create one with `/goal <title>`.").send()
        return

    lines = ["**Active Goals:**\n"]
    actions: List[cl.Action] = []
    for g in goals:
        gid = g.get("id", "")
        title = g.get("title", "Untitled")
        priority = g.get("priority", 0.5)
        status = g.get("status", "pending")
        lines.append(f"- **{title}** (priority: {priority:.0%}, status: {status})")
        actions.append(cl.Action(
            name="execute_goal",
            label=f"Execute: {title[:35]}",
            icon="play",
            payload={"goal_id": gid},
        ))

    await cl.Message(content="\n".join(lines), actions=actions).send()


async def _cmd_create_goal(content: str):
    """Create a goal.  Usage: /goal <title>"""
    parts = content.split(maxsplit=1)
    title = parts[1] if len(parts) > 1 else ""
    if not title.strip():
        await cl.Message(content="Usage: `/goal <title>`\nExample: `/goal Finish the quarterly report`").send()
        return
    result = await create_goal(title.strip())
    if result and result.get("status") == "success":
        gid = result.get("goal_id", "")
        actions = [cl.Action(
            name="execute_goal",
            label="Execute now",
            icon="play",
            payload={"goal_id": gid},
        )]
        await cl.Message(content=f"Goal created: **{title.strip()}**", actions=actions).send()
    else:
        await cl.Message(content="Failed to create goal.").send()


async def _cmd_dream():
    """Run a dream consolidation cycle."""
    async with cl.Step(name="Dream Cycle", type="tool") as step:
        step.input = "Consolidating STM -> LTM"
        try:
            result = await trigger_dream()
        except Exception as exc:
            await cl.Message(content=f"Dream cycle failed: `{exc}`").send()
            return
        dr = result.get("dream_results", {})
        step.output = json.dumps(dr, indent=2) if dr else "No results"

    lines = ["**Dream Cycle Complete**\n"]
    if dr:
        lines.append(f"- Consolidated: {dr.get('memories_consolidated', 0)}")
        lines.append(f"- Candidates: {dr.get('candidates_identified', 0)}")
        lines.append(f"- Associations: {dr.get('associations_created', 0)}")
        lines.append(f"- Duration: {dr.get('actual_duration', 0):.2f}s")
    await cl.Message(content="\n".join(lines)).send()


async def _cmd_reflect():
    """Run metacognitive reflection."""
    async with cl.Step(name="Reflection", type="tool") as step:
        step.input = "Running self-analysis"
        try:
            result = await trigger_reflection()
        except Exception as exc:
            await cl.Message(content=f"Reflection failed: `{exc}`").send()
            return
        report = result.get("report", {})
        step.output = json.dumps(report, indent=2)[:2000] if report else "No report"

    lines = ["**Reflection Report**\n"]
    stm = report.get("stm_metacognitive_stats")
    if stm:
        lines.append(f"**STM Health:** {stm.get('memory_count', 0)} items, "
                      f"{stm.get('capacity_utilization', 0)*100:.0f}% utilized, "
                      f"error rate {stm.get('error_rate', 0)*100:.1f}%")
    ltm = report.get("ltm_health_report")
    if ltm:
        status = ltm.get("health_status", "unknown").upper()
        lines.append(f"**LTM Health:** {status} | {ltm.get('total_memories', 0)} memories, "
                      f"confidence {ltm.get('average_confidence', 0):.2f}")
        for rec in (ltm.get("recommendations") or [])[:3]:
            lines.append(f"  - {rec}")
    if len(lines) == 1:
        lines.append("No data available.")
    await cl.Message(content="\n".join(lines)).send()


async def _cmd_learning():
    """Show learning dashboard metrics."""
    async with cl.Step(name="Learning Metrics", type="tool") as step:
        metrics = await fetch_learning_metrics()
        experiments = await fetch_experiments()
        outcomes = await fetch_execution_outcomes(limit=5)
        step.output = f"metrics={bool(metrics)}, experiments={len(experiments)}, outcomes={len(outcomes)}"

    lines = ["**Learning Dashboard**\n"]

    # Decision accuracy
    da = metrics.get("decision_accuracy", {})
    if da:
        lines.append(f"Decision Accuracy: **{da.get('overall_accuracy', 0):.1%}** ({da.get('total_decisions', 0)} decisions)")
    pa = metrics.get("planning_accuracy", {})
    if pa:
        lines.append(f"Planning Success: **{pa.get('plan_success_rate', 0):.1%}**")
    sa = metrics.get("scheduling_accuracy", {})
    if sa:
        lines.append(f"Scheduling Accuracy: **{sa.get('time_accuracy', 0):.1%}**")

    # Experiments
    if experiments:
        lines.append(f"\n**A/B Experiments:** {len(experiments)}")
        for exp in experiments[:3]:
            status = exp.get("status", "?")
            name = exp.get("name", "Experiment")
            winner = exp.get("recommended_strategy")
            line = f"- {name} ({status})"
            if winner:
                line += f" -> Winner: **{winner}**"
            lines.append(line)

    # Recent outcomes
    if outcomes:
        lines.append(f"\n**Recent Outcomes:** ({len(outcomes)} shown)")
        for o in outcomes[:5]:
            icon = "pass" if o.get("success") else "fail"
            lines.append(f"- [{icon}] {o.get('goal_title', 'Goal')} (score: {o.get('outcome_score', 0):.2f})")

    if len(lines) == 1:
        lines.append("No learning data available yet. Execute some goals to generate data.")

    await cl.Message(content="\n".join(lines)).send()


async def _cmd_metacog():
    """Show metacognition dashboard data for the current session."""
    session_id = cl.user_session.get("session_id") or "default"

    async with cl.Step(name="Metacognition Dashboard", type="tool") as step:
        dashboard, tasks, reflections = await asyncio.gather(
            fetch_metacognition_dashboard(session_id=session_id, history_limit=10, limit=50),
            fetch_metacognition_tasks(session_id=session_id),
            fetch_metacognition_reflections(session_id=session_id, limit=10),
        )
        step.output = (
            f"available={bool(dashboard.get('available'))}, "
            f"tasks={len(tasks)}, reflections={len(reflections)}"
        )

    if not dashboard or not dashboard.get("available"):
        await cl.Message(
            content="No metacognition trace data is available yet. Send a few chat turns or let background cognition run first."
        ).send()
        return

    status = dashboard.get("status", {})
    background = dashboard.get("background", {})
    scorecard = dashboard.get("scorecard", {})
    summary = scorecard.get("summary", {})
    contradictions = scorecard.get("contradictions", {})
    self_model = scorecard.get("self_model", {})
    goals = scorecard.get("goals", {})

    lines = ["**Metacognition Dashboard**\n"]
    lines.append(f"Session: `{dashboard.get('session_id') or session_id}`")
    lines.append(f"Trace Count: **{scorecard.get('trace_count', 0)}**")

    cycle_success = summary.get("cycle_success_avg")
    contradiction_rate = contradictions.get("contradiction_rate")
    drift = self_model.get("self_model_drift_avg")
    churn = goals.get("goal_churn_rate")
    follow_up_rate = summary.get("follow_up_rate")

    lines.append(
        "Cycle Success: **{}** | Follow-Up Rate: **{}**".format(
            f"{cycle_success:.1%}" if isinstance(cycle_success, (int, float)) else "N/A",
            f"{follow_up_rate:.1%}" if isinstance(follow_up_rate, (int, float)) else "N/A",
        )
    )
    lines.append(
        "Contradiction Rate: **{}** | Goal Churn: **{}**".format(
            f"{contradiction_rate:.1%}" if isinstance(contradiction_rate, (int, float)) else "N/A",
            f"{churn:.1%}" if isinstance(churn, (int, float)) else "N/A",
        )
    )
    lines.append(
        "Self-Model Drift: **{}** | Pending Tasks: **{}** | Due Tasks: **{}**".format(
            f"{drift:.2f}" if isinstance(drift, (int, float)) else "N/A",
            background.get("pending_task_count", 0),
            background.get("due_task_count", 0),
        )
    )
    lines.append(
        "Scheduler: **{}** | Open Contradictions: **{}** | Idle Reflections: **{}**".format(
            "running" if background.get("scheduler_running") else "stopped",
            background.get("unresolved_contradiction_count", 0),
            background.get("idle_reflection_count", 0),
        )
    )

    last_cycle = status.get("last_cycle") or {}
    if last_cycle:
        lines.append("\n**Last Cycle**")
        lines.append(f"- Cycle ID: `{last_cycle.get('cycle_id', 'N/A')}`")
        lines.append(f"- Goal Kind: `{last_cycle.get('selected_goal_kind', 'N/A')}`")
        success_score = last_cycle.get("success_score")
        lines.append(
            f"- Success Score: {success_score:.2f}" if isinstance(success_score, (int, float)) else "- Success Score: N/A"
        )

    goal_mix = goals.get("selected_goal_kind_counts", {})
    if goal_mix:
        lines.append("\n**Goal Selection Mix**")
        for goal_kind, count in goal_mix.items():
            lines.append(f"- {goal_kind}: {count}")

    if tasks:
        lines.append("\n**Tasks**")
        for task in tasks[:5]:
            reason = (task.get("metadata") or {}).get("reason")
            due_at = task.get("due_at", "N/A")
            label = task.get("task_type") or task.get("description") or task.get("task_id", "task")
            suffix = f" ({reason})" if reason else ""
            lines.append(f"- {label}: {task.get('status', 'unknown')} due `{due_at}`{suffix}")

    if reflections:
        lines.append("\n**Recent Reflections**")
        for report in reflections[:3]:
            trigger = (report.get("metadata") or {}).get("trigger")
            summary_text = report.get("summary") or report.get("timestamp") or "reflection"
            suffix = f" [{trigger}]" if trigger else ""
            lines.append(f"- {summary_text}{suffix}")

    await cl.Message(content="\n".join(lines)).send()

    detail_payload = {
        "dashboard": dashboard,
        "tasks": tasks[:10],
        "reflections": reflections[:10],
    }
    await cl.Message(
        content=f"```json\n{json.dumps(detail_payload, indent=2)[:3500]}\n```",
        author="metacognition",
    ).send()


# ============================================================================
# Helpers
# ============================================================================

async def _surface_due_reminders(due: List[Dict[str, Any]]):
    """Post action-bearing messages for due reminders."""
    for r in due[:5]:
        rid = r.get("id", "")
        content_text = r.get("content", "Reminder")
        due_str = format_due_display(r)
        actions = [
            cl.Action(name="complete_reminder", label="Done", icon="check", payload={"reminder_id": rid}),
            cl.Action(name="snooze_reminder", label="Snooze", icon="alarm-clock", payload={"reminder_id": rid, "content": content_text}),
            cl.Action(name="delete_reminder", label="Dismiss", icon="x", payload={"reminder_id": rid}),
        ]
        await cl.Message(
            content=f"**Reminder due:** {content_text} ({due_str})",
            actions=actions,
            author="reminders",
        ).send()
