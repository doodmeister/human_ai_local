"""Minimal Streamlit chat interface for the George cognitive backend."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# Allow overriding the API host via environment variable.
DEFAULT_API_BASE = os.getenv("GEORGE_API_BASE_URL", "http://localhost:8000")


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO8601 parser that tolerates trailing Z notation."""
    if not value:
        return None
    cleaned = value
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _format_due_display(reminder: Dict[str, Any]) -> str:
    """Return a user-friendly due phrase for a reminder."""
    phrase = reminder.get("due_phrase")
    if phrase:
        return phrase
    due_dt = _parse_iso_datetime(reminder.get("due_time"))
    if not due_dt:
        return "no specific time"
    if due_dt.tzinfo is None:
        due_dt = due_dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = due_dt - now
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "due now"
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24
    if minutes < 1:
        return "in <1 min"
    if minutes < 60:
        return f"in {minutes} min"
    if hours < 24:
        return f"in {hours} hr"
    return f"in {days} day" + ("s" if days != 1 else "")


def _mark_acknowledged(reminder_id: Optional[str]) -> None:
    """Remember that a reminder was surfaced so we do not spam announcements."""
    if not reminder_id:
        return
    if "acknowledged_reminder_ids" not in st.session_state:
        st.session_state.acknowledged_reminder_ids = set()
    st.session_state.acknowledged_reminder_ids.add(reminder_id)


def _drop_reminder_from_cache(reminder_id: Optional[str]) -> None:
    if not reminder_id:
        return
    data = st.session_state.get("proactive_reminders") or {}
    for bucket in ("due", "upcoming"):
        items = data.get(bucket) or []
        filtered = [item for item in items if item.get("id") != reminder_id]
        data[bucket] = filtered
    st.session_state.proactive_reminders = data
    pending = st.session_state.get("new_due_reminders", [])
    st.session_state.new_due_reminders = [item for item in pending if item.get("id") != reminder_id]


def _log_reminder_event(event_type: str, reminder: Optional[Dict[str, Any]] = None) -> None:
    metrics = st.session_state.get("reminder_metrics")
    if metrics is None:
        metrics = {"counts": {}, "events": []}
    counts = metrics.setdefault("counts", {})
    counts[event_type] = counts.get(event_type, 0) + 1
    event = {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if reminder:
        event["reminder_id"] = reminder.get("id")
        event["content"] = reminder.get("content")
        event["due_time"] = reminder.get("due_time")
    events = metrics.setdefault("events", [])
    events.append(event)
    metrics["events"] = events[-50:]
    st.session_state.reminder_metrics = metrics


def _add_reminder_to_upcoming_cache(reminder: Dict[str, Any]) -> None:
    data = st.session_state.get("proactive_reminders") or {"summary": None, "due": [], "upcoming": []}
    upcoming = data.get("upcoming") or []
    if reminder.get("id") not in {item.get("id") for item in upcoming}:
        upcoming.append(reminder)
        data["upcoming"] = upcoming
    st.session_state.proactive_reminders = data


def create_reminder_remote(
    base_url: str,
    content: str,
    due_in_seconds: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "content": content,
        "due_in_seconds": due_in_seconds,
    }
    if metadata:
        payload["metadata"] = metadata
    try:
        resp = requests.post(f"{base_url}/agent/reminders", json=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Unable to create reminder: {exc}")
        return None
    reminder = resp.json().get("reminder")
    if reminder:
        _log_reminder_event("created", reminder)
    return reminder


def complete_reminder_remote(base_url: str, reminder: Dict[str, Any]) -> bool:
    """Mark a reminder complete via the API and update local caches."""
    reminder_id = reminder.get("id")
    if not reminder_id:
        st.error("Missing reminder identifier")
        return False
    try:
        resp = requests.post(f"{base_url}/agent/reminders/{reminder_id}/complete", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Unable to complete reminder: {exc}")
        return False
    _mark_acknowledged(reminder_id)
    _drop_reminder_from_cache(reminder_id)
    _log_reminder_event("completed", reminder)
    return True


def delete_reminder_remote(base_url: str, reminder: Dict[str, Any], log_event: bool = True) -> bool:
    """Delete a reminder via API and update local caches."""
    reminder_id = reminder.get("id")
    if not reminder_id:
        st.error("Missing reminder identifier")
        return False
    try:
        resp = requests.delete(f"{base_url}/agent/reminders/{reminder_id}", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Unable to delete reminder: {exc}")
        return False
    _mark_acknowledged(reminder_id)
    _drop_reminder_from_cache(reminder_id)
    if log_event:
        _log_reminder_event("deleted", reminder)
    return True


def snooze_reminder_remote(base_url: str, reminder: Dict[str, Any], minutes: int = 15) -> bool:
    """Snooze a reminder by creating a new one and removing the original."""
    content = reminder.get("content", "")
    if not content:
        st.error("Cannot snooze reminder without content")
        return False
    if minutes <= 0:
        st.warning("Snooze minutes must be positive")
        return False
    metadata = dict(reminder.get("metadata") or {})
    metadata.update({
        "snoozed_from_id": reminder.get("id"),
        "snoozed_minutes": minutes,
    })
    new_reminder = create_reminder_remote(base_url, content, int(minutes) * 60, metadata)
    if new_reminder is None:
        return False
    if reminder.get("id"):
        delete_reminder_remote(base_url, reminder, log_event=False)
    _add_reminder_to_upcoming_cache(new_reminder)
    _log_reminder_event("snoozed", reminder)
    return True


def render_proactive_banner(base_url: str) -> None:
    """Show a top-of-feed announcement whenever fresh reminders arrive."""
    new_due: List[Dict[str, Any]] = st.session_state.get("new_due_reminders", [])
    if not new_due:
        return

    with st.container():
        st.markdown("### ⏰ Coming up soon")
        summary = st.session_state.get("proactive_reminders", {}).get("summary")
        if summary:
            st.caption(summary)
        for reminder in new_due:
            _render_reminder_row(reminder, base_url, source="banner")
        if st.button("Dismiss reminders", key="dismiss_proactive_banner"):
            for reminder in list(new_due):
                _mark_acknowledged(reminder.get("id"))
                _log_reminder_event("dismissed", reminder)
            st.session_state.new_due_reminders = []


def render_session_context_panel() -> None:
    """Render a lightweight session context snapshot when available."""
    context = st.session_state.get("session_context")
    if not context:
        return

    with st.expander("Session context snapshot", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Captured", context.get("captured_memory_count", 0))
            st.metric("Last intent", context.get("last_intent", "unknown"))
        with cols[1]:
            st.metric("Due reminders", context.get("prospective_due_count", 0))
            st.metric("Upcoming", context.get("prospective_upcoming_count", 0))
        with cols[2]:
            goals = context.get("active_goal_ids") or []
            st.metric("Active goals", len(goals))
            st.caption(context.get("session_id", ""))

        next_upcoming = context.get("next_upcoming_reminder")
        if next_upcoming:
            due_phrase = _format_due_display({"due_time": next_upcoming.get("due_time")})
            st.info(
                f"Next reminder: **{next_upcoming.get('content', '')}** ({due_phrase})",
                icon="⌛",
            )

        classifier_context = context.get("classifier_context")
        if classifier_context:
            with st.expander("Classifier context", expanded=False):
                st.json(classifier_context)

        if goals:
            with st.expander("Active goal IDs", expanded=False):
                for goal_id in goals:
                    st.write(f"- {goal_id}")


def _render_reminder_row(reminder: Dict[str, Any], base_url: str, source: str) -> None:
    reminder_id = reminder.get("id")
    snooze_minutes = (
        st.session_state.get("reminder_form", {}).get("snooze_minutes")
        or 15
    )
    cols = st.columns([0.5, 0.2, 0.2, 0.1])
    with cols[0]:
        st.markdown(f"**{reminder.get('content', 'Reminder')}**")
        st.caption(_format_due_display(reminder))
    with cols[1]:
        if st.button("Mark done", key=f"complete_{reminder_id}_{source}"):
            if complete_reminder_remote(base_url, reminder):
                st.success("Reminder completed")
    with cols[2]:
        label = f"Snooze +{snooze_minutes}m"
        if st.button(label, key=f"snooze_{reminder_id}_{source}"):
            if snooze_reminder_remote(base_url, reminder, minutes=int(snooze_minutes)):
                st.info(f"Snoozed for {snooze_minutes} minutes")
    with cols[3]:
        if st.button("Dismiss", key=f"dismiss_{reminder_id}_{source}"):
            _mark_acknowledged(reminder_id)
            _drop_reminder_from_cache(reminder_id)
            _log_reminder_event("dismissed", reminder)
            st.session_state.new_due_reminders = [
                item for item in st.session_state.get("new_due_reminders", []) if item.get("id") != reminder_id
            ]


def render_sidebar_reminders(base_url: str) -> None:
    """Render the reminder timeline in the sidebar."""
    data = st.session_state.get("proactive_reminders") or {}
    due = data.get("due") or []
    upcoming = data.get("upcoming") or []
    combined = due + [r for r in upcoming if r.get("id") not in {item.get("id") for item in due}]

    if not combined:
        st.caption("No reminders scheduled")
        return

    def _sort_key(item: Dict[str, Any]) -> float:
        dt = _parse_iso_datetime(item.get("due_time"))
        if dt is None:
            return float("inf")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    sorted_items = sorted(combined, key=_sort_key)
    snooze_minutes = (
        st.session_state.get("reminder_form", {}).get("snooze_minutes")
        or 15
    )
    for reminder in sorted_items:
        reminder_id = reminder.get("id")
        st.markdown(f"**{reminder.get('content', 'Reminder')}**")
        st.caption(_format_due_display(reminder))
        btn_cols = st.columns([0.4, 0.3, 0.3])
        with btn_cols[0]:
            if st.button("Complete", key=f"sidebar_complete_{reminder_id}"):
                if complete_reminder_remote(base_url, reminder):
                    st.success("Reminder completed")
        with btn_cols[1]:
            label = f"Snooze +{snooze_minutes}m"
            if st.button(label, key=f"sidebar_snooze_{reminder_id}"):
                if snooze_reminder_remote(base_url, reminder, minutes=int(snooze_minutes)):
                    st.info(f"Snoozed {snooze_minutes} minutes")
        with btn_cols[2]:
            if st.button("Delete", key=f"sidebar_delete_{reminder_id}"):
                if delete_reminder_remote(base_url, reminder):
                    st.info("Reminder removed")


def _update_proactive_state_from_response(response: Dict[str, Any]) -> None:
    proactive = response.get("proactive_reminders") or {}
    st.session_state.proactive_reminders = proactive
    due = proactive.get("due") or []
    ack = st.session_state.get("acknowledged_reminder_ids", set())
    surfaced = st.session_state.get("surfaced_reminder_ids", set())
    fresh_due: List[Dict[str, Any]] = []
    for reminder in due:
        rem_id = reminder.get("id")
        if not rem_id or rem_id in ack:
            continue
        fresh_due.append(reminder)
        if rem_id not in surfaced:
            _log_reminder_event("surfaced", reminder)
            surfaced.add(rem_id)
    st.session_state.surfaced_reminder_ids = surfaced
    st.session_state.new_due_reminders = fresh_due


def check_backend(base_url: str) -> bool:
    """Return True when the backend health check responds successfully."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def post_chat(
    base_url: str,
    message: str,
    session_id: str,
    flags: Optional[Dict[str, bool]] = None,
    salience_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Send a chat request to the backend and return the parsed JSON response."""
    payload: Dict[str, Any] = {"message": message, "session_id": session_id}
    if flags:
        filtered = {key: value for key, value in flags.items() if value}
        if filtered:
            payload["flags"] = filtered
    if salience_threshold is not None:
        payload["consolidation_salience_threshold"] = salience_threshold
    resp = requests.post(f"{base_url}/agent/chat", json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def trigger_dream_cycle(base_url: str, cycle_type: str = "light") -> Dict[str, Any]:
    """Trigger a dream state consolidation cycle."""
    payload = {"cycle_type": cycle_type}
    resp = requests.post(f"{base_url}/agent/dream/start", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def trigger_reflection(base_url: str) -> Dict[str, Any]:
    """Trigger agent-level metacognitive reflection and return the report."""
    resp = requests.post(f"{base_url}/reflect", timeout=30)
    resp.raise_for_status()
    return resp.json()


def ensure_state() -> None:
    """Initialize Streamlit session state fields used by the UI."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE
    if "last_raw_response" not in st.session_state:
        st.session_state.last_raw_response = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "salience_threshold" not in st.session_state:
        st.session_state.salience_threshold = 0.55  # Default from ChatConfig
    if "session_context" not in st.session_state:
        st.session_state.session_context = None
    if "proactive_reminders" not in st.session_state:
        st.session_state.proactive_reminders = {"summary": None, "due": [], "upcoming": []}
    if "new_due_reminders" not in st.session_state:
        st.session_state.new_due_reminders = []
    if "acknowledged_reminder_ids" not in st.session_state:
        st.session_state.acknowledged_reminder_ids = set()
    if "surfaced_reminder_ids" not in st.session_state:
        st.session_state.surfaced_reminder_ids = set()
    if "reminder_form" not in st.session_state:
        st.session_state.reminder_form = {
            "content": "",
            "due_in_minutes": 15,
            "snooze_minutes": 15,
        }
    if "reminder_metrics" not in st.session_state:
        st.session_state.reminder_metrics = {"counts": {}, "events": []}
    # LLM provider settings
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "openai"
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4.1-nano"
    if "ollama_base_url" not in st.session_state:
        st.session_state.ollama_base_url = "http://localhost:11434"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "llama3.2"
    # Executive function settings
    if "active_goals" not in st.session_state:
        st.session_state.active_goals = []
    if "active_goals_refresh" not in st.session_state:
        st.session_state.active_goals_refresh = False
    if "selected_goal_id" not in st.session_state:
        st.session_state.selected_goal_id = None


def create_goal_remote(
    base_url: str,
    title: str,
    description: str = "",
    priority: float = 0.5,
    deadline: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Create a new goal via executive API."""
    payload = {
        "title": title,
        "description": description,
        "priority": priority,
    }
    if deadline:
        payload["deadline"] = deadline
    
    try:
        resp = requests.post(f"{base_url}/executive/goals", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to create goal: {exc}")
        return None


def get_goals_remote(base_url: str, active_only: bool = True) -> List[Dict[str, Any]]:
    """Fetch goals from executive API."""
    try:
        params = {"active_only": "true" if active_only else "false"}
        resp = requests.get(f"{base_url}/executive/goals", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("goals", [])
    except requests.RequestException as exc:
        st.error(f"Failed to fetch goals: {exc}")
        return []


def render_goal_creator(base_url: str) -> None:
    """Render goal creation form in sidebar."""
    with st.expander("➕ Create Goal", expanded=False):
        goal_title = st.text_input(
            "Goal title",
            placeholder="e.g., Deploy new feature to production",
            key="goal_title_input"
        )
        
        goal_description = st.text_area(
            "Description (optional)",
            placeholder="Additional context about this goal...",
            key="goal_description_input",
            height=80
        )
        
        col1, col2 = st.columns(2)
        with col1:
            priority_label = st.selectbox(
                "Priority",
                options=["Low (0.25)", "Medium (0.5)", "High (0.75)", "Critical (0.95)"],
                index=1,
                key="goal_priority_input"
            )
            priority_map = {
                "Low (0.25)": 0.25,
                "Medium (0.5)": 0.5,
                "High (0.75)": 0.75,
                "Critical (0.95)": 0.95
            }
            priority = priority_map[priority_label]
        
        with col2:
            use_deadline = st.checkbox("Set deadline", key="goal_has_deadline")
        
        deadline_str = None
        if use_deadline:
            deadline_date = st.date_input("Deadline", key="goal_deadline_input")
            deadline_time = st.time_input("Time (optional)", key="goal_deadline_time_input")
            if deadline_date:
                if deadline_time:
                    deadline_str = f"{deadline_date} {deadline_time}"
                else:
                    deadline_str = str(deadline_date)
        
        if st.button("Create Goal", key="create_goal_btn"):
            if not goal_title.strip():
                st.warning("Please provide a goal title")
                return
            
            result = create_goal_remote(
                base_url,
                goal_title.strip(),
                description=goal_description.strip(),
                priority=priority,
                deadline=deadline_str
            )
            
            if result and result.get("status") == "success":
                goal_id = result.get("goal_id")
                st.success(f"Goal created: {goal_id[:8] if goal_id else 'success'}")
                st.session_state.active_goals_refresh = True
                st.rerun()


def render_active_goals(base_url: str) -> None:
    """Render active goals panel in sidebar."""
    st.subheader("🎯 Active Goals")
    
    # Refresh trigger
    if st.session_state.get("active_goals_refresh"):
        st.session_state.active_goals_refresh = False
    
    goals = get_goals_remote(base_url, active_only=True)
    st.session_state.active_goals = goals
    
    if not goals:
        st.caption("No active goals")
        return
    
    for goal in goals:
        goal_id = goal.get("id", "")
        title = goal.get("title", "Untitled goal")
        priority = goal.get("priority", 0.5)
        status = goal.get("status", "unknown")
        progress = goal.get("progress", 0.0)
        
        # Priority emoji
        if priority >= 0.75:
            priority_emoji = "🔴"
        elif priority >= 0.5:
            priority_emoji = "🟡"
        else:
            priority_emoji = "🟢"
        
        with st.container():
            st.markdown(f"{priority_emoji} **{title}**")
            st.caption(f"Status: {status} | Progress: {progress*100:.0f}%")
            
            if progress > 0:
                st.progress(progress)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Details", key=f"view_goal_{goal_id}"):
                    st.session_state.selected_goal_id = goal_id
                    st.rerun()
            with col2:
                if st.button("🔗 Link", key=f"link_goal_{goal_id}", help="Copy goal ID to link with reminders"):
                    st.info(f"Goal ID: {goal_id[:12]}")
            
            st.markdown("---")


def update_llm_config(base_url: str, provider: str, openai_model: str, ollama_base_url: str, ollama_model: str) -> Dict[str, Any]:
    """Update the backend LLM configuration"""
    payload = {
        "provider": provider,
        "openai_model": openai_model,
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
    }
    try:
        resp = requests.post(f"{base_url}/agent/config/llm", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def render_sidebar(connection_ok: bool) -> Dict[str, Any]:
    """Render sidebar controls and return the current chat flags and salience threshold."""
    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API base URL", value=st.session_state.api_base_url)
        cleaned = (api_url or "").rstrip("/")
        st.session_state.api_base_url = cleaned or DEFAULT_API_BASE
        
        # LLM Provider Configuration
        st.subheader("🤖 LLM Provider")
        provider_changed = False
        
        new_provider = st.selectbox(
            "Provider",
            options=["openai", "ollama"],
            index=0 if st.session_state.llm_provider == "openai" else 1,
            help="Select between OpenAI API or locally-hosted Ollama"
        )
        
        if new_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = new_provider
            provider_changed = True
        
        if st.session_state.llm_provider == "openai":
            new_openai_model = st.text_input(
                "OpenAI Model",
                value=st.session_state.openai_model,
                help="e.g., gpt-4.1-nano, gpt-4o, gpt-3.5-turbo"
            )
            if new_openai_model != st.session_state.openai_model:
                st.session_state.openai_model = new_openai_model
                provider_changed = True
        else:  # ollama
            new_ollama_url = st.text_input(
                "Ollama Base URL",
                value=st.session_state.ollama_base_url,
                help="URL of your local Ollama server"
            )
            new_ollama_model = st.text_input(
                "Ollama Model",
                value=st.session_state.ollama_model,
                help="e.g., llama3.2, mistral, codellama"
            )
            if new_ollama_url != st.session_state.ollama_base_url:
                st.session_state.ollama_base_url = new_ollama_url
                provider_changed = True
            if new_ollama_model != st.session_state.ollama_model:
                st.session_state.ollama_model = new_ollama_model
                provider_changed = True
        
        if provider_changed or st.button("🔄 Apply LLM Config"):
            result = update_llm_config(
                st.session_state.api_base_url,
                st.session_state.llm_provider or "openai",
                st.session_state.openai_model or "gpt-4.1-nano",
                st.session_state.ollama_base_url or "http://localhost:11434",
                st.session_state.ollama_model or "llama3.2",
            )
            if result.get("status") == "ok":
                st.success("LLM configuration updated!")
            else:
                st.error(f"Failed to update config: {result.get('message', 'Unknown error')}")
        
        st.markdown("---")
        
        st.subheader("Memory Controls")
        st.session_state.salience_threshold = st.slider(
            "STM Consolidation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.salience_threshold,
            step=0.05,
            help="Lower = capture more in STM (e.g., 0.35 for casual chat). Higher = only capture emphatic messages (default 0.55)."
        )
        st.caption(f"Current threshold: {st.session_state.salience_threshold:.2f}")
        
        if st.button("🌙 Trigger Dream Cycle", help="Consolidate STM → LTM (promotes memories based on rehearsal)"):
            try:
                with st.spinner("Running dream consolidation..."):
                    result = trigger_dream_cycle(st.session_state.api_base_url, cycle_type="light")
                st.success("Dream cycle complete!")
                
                # Parse and display results nicely
                dream_results = result.get("dream_results", {})
                if dream_results:
                    # Summary metrics
                    memories_consolidated = dream_results.get("memories_consolidated", 0)
                    candidates = dream_results.get("candidates_identified", 0)
                    associations = dream_results.get("associations_created", 0)
                    duration = dream_results.get("actual_duration", 0)
                    
                    st.metric("Memories Consolidated", memories_consolidated)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Candidates", candidates)
                    with col2:
                        st.metric("Associations", associations)
                    with col3:
                        st.metric("Duration", f"{duration:.2f}s")
                    
                    # Cleanup info
                    cleanup = dream_results.get("cleanup", {})
                    if cleanup:
                        with st.expander("Cleanup details"):
                            st.write(f"Weak memories removed: {cleanup.get('weak_memories_removed', 0)}")
                            st.write(f"Duplicates cleaned: {cleanup.get('duplicate_associations_cleaned', 0)}")
                            st.write(f"Decay applied: {cleanup.get('decay_applied', False)}")
                            st.write(f"Items decayed: {cleanup.get('items_decayed', 0)}")
                
                with st.expander("Full dream cycle results"):
                    st.json(result)
            except Exception as e:
                st.error(f"Dream cycle failed: {e}")
        
        if st.button("🧠 Trigger Reflection", help="Run metacognitive self-analysis on memory health & performance"):
            try:
                with st.spinner("Running reflection analysis..."):
                    result = trigger_reflection(st.session_state.api_base_url)
                st.success("Reflection complete!")
                
                # Display the reflection report
                report = result.get("report", {})
                if report:
                    # STM Stats
                    stm_stats = report.get("stm_metacognitive_stats")
                    if stm_stats:
                        st.subheader("📊 Short-Term Memory Health")
                        col1, col2 = st.columns(2)
                        with col1:
                            capacity_util = stm_stats.get("capacity_utilization", 0)
                            st.metric("Capacity Utilization", f"{capacity_util*100:.1f}%")
                            st.metric("Memory Count", stm_stats.get("memory_count", 0))
                        with col2:
                            error_rate = stm_stats.get("error_rate", 0)
                            st.metric("Error Rate", f"{error_rate*100:.2f}%")
                            avg_importance = stm_stats.get("avg_importance", 0)
                            st.metric("Avg Importance", f"{avg_importance:.2f}")
                    
                    # LTM Health Report
                    ltm_health = report.get("ltm_health_report")
                    if ltm_health:
                        st.subheader("🧠 Long-Term Memory Health")
                        
                        # Core metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Memories", ltm_health.get("total_memories", 0))
                        with col2:
                            avg_conf = ltm_health.get("average_confidence", 0)
                            st.metric("Avg Confidence", f"{avg_conf:.2f}")
                        with col3:
                            success_rate = ltm_health.get("search_success_rate", 0)
                            st.metric("Search Success Rate", f"{success_rate*100:.1f}%")
                        
                        # Recommendations
                        recommendations = ltm_health.get("recommendations", [])
                        if recommendations:
                            with st.expander("💡 Recommendations", expanded=True):
                                for rec in recommendations:
                                    st.info(rec)
                        
                        # Health status
                        health_status = ltm_health.get("health_status", "unknown")
                        status_color = {"good": "🟢", "warning": "🟡", "poor": "🔴"}.get(health_status, "⚪")
                        st.write(f"**Overall Health:** {status_color} {health_status.upper()}")
                    
                    # Full report details
                    with st.expander("Full reflection report"):
                        st.json(report)
                else:
                    st.warning("No reflection data available")
                    
            except Exception as e:
                st.error(f"Reflection failed: {e}")
        
        st.subheader("Chat Options")
        include_memory = st.checkbox("Include memory retrieval", value=True, key="include_memory")
        include_attention = st.checkbox("Include attention signals", value=True, key="include_attention")
        include_trace = st.checkbox("Include trace details", value=False, key="include_trace")
        include_reflection = st.checkbox("Request reflection", value=False, key="include_reflection")
        
        if st.button("New conversation"):
            st.session_state.session_id = uuid.uuid4().hex[:12]
            st.session_state.messages = []
            st.session_state.last_raw_response = None
            st.session_state.last_error = None
            st.session_state.session_context = None
            st.session_state.proactive_reminders = {"summary": None, "due": [], "upcoming": []}
            st.session_state.new_due_reminders = []
            st.session_state.acknowledged_reminder_ids.clear()
            st.session_state.surfaced_reminder_ids.clear()
        st.caption(f"Session ID: {st.session_state.session_id}")
        if connection_ok:
            st.success("Backend reachable")
        else:
            st.error("Backend unreachable")
        if st.session_state.last_error:
            st.warning(f"Last error: {st.session_state.last_error}")
        with st.expander("Last backend response", expanded=False):
            if st.session_state.last_raw_response is None:
                st.caption("No responses yet")
            else:
                st.json(st.session_state.last_raw_response)
        
        st.markdown("---")
        render_active_goals(st.session_state.api_base_url)
        render_goal_creator(st.session_state.api_base_url)
        
        st.markdown("---")
        st.subheader("Upcoming reminders")
        render_sidebar_reminders(st.session_state.api_base_url)
        st.markdown("---")
        st.subheader("Create reminder")
        reminder_form = st.session_state.reminder_form
        reminder_content = st.text_input(
            "Content",
            value=reminder_form.get("content", ""),
            key="reminder_content_input",
        )
        reminder_minutes = st.number_input(
            "Due in (minutes)",
            min_value=1,
            max_value=1440,
            value=int(reminder_form.get("due_in_minutes", 15)),
            step=5,
            key="reminder_due_minutes_input",
        )
        snooze_options = [5, 10, 15, 30, 45, 60]
        current_snooze = reminder_form.get("snooze_minutes", 15)
        if current_snooze not in snooze_options:
            snooze_options.append(current_snooze)
            snooze_options = sorted(set(snooze_options))
        default_index = snooze_options.index(current_snooze)
        snooze_minutes = st.selectbox(
            "Default snooze length (minutes)",
            options=snooze_options,
            index=default_index,
            key="reminder_snooze_minutes_input",
        )
        st.session_state.reminder_form["snooze_minutes"] = snooze_minutes
        reminder_metadata_note = st.text_area(
            "Metadata notes (optional)",
            value=reminder_form.get("metadata_note", ""),
            key="reminder_metadata_input",
        )
        
        # NEW: Goal linking
        active_goals = st.session_state.get("active_goals", [])
        goal_options = ["None (not linked)"] + [
            f"{g.get('title', 'Untitled')} ({g.get('id', '')[:8]})" 
            for g in active_goals
        ]
        selected_goal_idx = st.selectbox(
            "Link to goal (optional)",
            options=range(len(goal_options)),
            format_func=lambda i: goal_options[i],
            key="reminder_goal_link",
            help="Associate this reminder with an active goal"
        )

        if st.button("➕ Add reminder", key="create_reminder_button"):
            trimmed = reminder_content.strip()
            if not trimmed:
                st.warning("Provide reminder content first")
            else:
                metadata = {"note": reminder_metadata_note.strip()} if reminder_metadata_note.strip() else {}
                
                # Add goal link to metadata if selected
                if selected_goal_idx > 0:  # 0 is "None"
                    goal = active_goals[selected_goal_idx - 1]
                    metadata["goal_id"] = goal.get("id")
                    metadata["goal_title"] = goal.get("title")
                
                reminder = create_reminder_remote(
                    st.session_state.api_base_url,
                    trimmed,
                    int(reminder_minutes) * 60,
                    metadata=metadata if metadata else None,
                )
                if reminder:
                    st.success("Reminder created")
                    _add_reminder_to_upcoming_cache(reminder)
                    st.session_state.reminder_form = {
                        "content": "",
                        "due_in_minutes": reminder_minutes,
                        "snooze_minutes": snooze_minutes,
                        "metadata_note": reminder_metadata_note,
                    }

        metrics = st.session_state.get("reminder_metrics", {"counts": {}, "events": []})
        with st.expander("Reminder telemetry", expanded=False):
            counts = metrics.get("counts", {})
            if counts:
                cols = st.columns(len(counts))
                for idx, (label, value) in enumerate(counts.items()):
                    with cols[idx]:
                        st.metric(label.replace("_", " ").title(), value)
            else:
                st.caption("No reminder interactions yet")
            events = metrics.get("events", [])
            if events:
                st.caption("Recent events")
                for event in reversed(events[-5:]):
                    stamp = event.get("timestamp", "")
                    ev_label = event.get("type", "")
                    content = event.get("content", "")
                    st.write(f"{stamp}: **{ev_label}** – {content}")
            else:
                st.caption("No telemetry events recorded")

        return {
            "flags": {
                "include_memory": include_memory,
                "include_attention": include_attention,
                "include_trace": include_trace,
                "reflection": include_reflection,
            },
            "salience_threshold": st.session_state.salience_threshold,
        }


def render_message(message: Dict[str, Any]) -> None:
    """Render a single chat message with optional context details."""
    role = message.get("role", "assistant")
    # Streamlit expects "user" or "assistant"; map anything else to assistant.
    role_key = role if role in {"user", "assistant"} else "assistant"
    with st.chat_message(role_key):
        st.markdown(message.get("content", ""))
        if role_key == "assistant":
            context_items = message.get("context_items") or []
            captured = message.get("captured_memories") or []
            metrics = message.get("metrics") or {}
            if context_items:
                with st.expander("Context used (STM -> LTM ordering)"):
                    for item in context_items:
                        source = item.get("source_system", "unknown")
                        reason = item.get("reason", "")
                        rank = item.get("rank")
                        content = item.get("content", "")
                        st.markdown(f"**{rank}. [{source}]** {content}\n*reason:* {reason}")
            if captured:
                with st.expander("Captured memories"):
                    for mem in captured:
                        summary = mem.get("content", "")
                        mem_type = mem.get("memory_type", "")
                        freq = mem.get("frequency")
                        tag = f"[{mem_type}] " if mem_type else ""
                        details = f"{tag}{summary}"
                        if freq:
                            details += f" - frequency {freq}"
                        if mem.get("contradiction"):
                            details += " [contradiction]"
                        if mem.get("reinforced"):
                            details += " [reinforced]"
                        st.write(details)
            if metrics:
                latency = metrics.get("turn_latency_ms")
                stm_hits = metrics.get("stm_hits")
                ltm_hits = metrics.get("ltm_hits")
                fallback = metrics.get("fallback_used")
                # Debug consolidation
                salience = metrics.get("user_salience")
                valence = metrics.get("user_valence")
                importance = metrics.get("user_importance")
                cons_status = metrics.get("consolidation_status")
                
                parts = []
                if latency is not None:
                    parts.append(f"latency {latency:.0f} ms")
                if stm_hits is not None or ltm_hits is not None:
                    parts.append(f"context hits STM={stm_hits} LTM={ltm_hits}")
                if fallback:
                    parts.append("fallback retrieval active")
                if salience is not None:
                    parts.append(f"salience={salience:.2f}")
                if valence is not None:
                    parts.append(f"valence={valence:.2f}")
                if importance is not None:
                    parts.append(f"importance={importance:.2f}")
                if cons_status:
                    parts.append(f"consolidation: {cons_status}")
                if parts:
                    st.caption("; ".join(parts))


def main() -> None:
    st.set_page_config(page_title="George Chat", page_icon="G", layout="wide")
    ensure_state()

    base_url = st.session_state.api_base_url
    backend_ok = check_backend(base_url)
    config = render_sidebar(backend_ok)
    flags = config["flags"]
    salience_threshold = config["salience_threshold"]

    st.title("George Chat Interface")
    st.write("Chat with the cognitive backend. Each turn retrieves short-term memories first, then falls back to long-term memories when needed.")
    render_proactive_banner(base_url)
    render_session_context_panel()

    prompt: Optional[str] = None
    if backend_ok:
        prompt = st.chat_input("Send a message")
    else:
        st.info("Start the backend with `python start_server.py` to enable chatting.")

    assistant_record: Optional[Dict[str, Any]] = None
    if prompt:
        user_record = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        st.session_state.messages.append(user_record)
        st.session_state.last_error = None

        try:
            with st.spinner("Retrieving context and generating response..."):
                response = post_chat(
                    base_url,
                    prompt,
                    st.session_state.session_id,
                    flags,
                    salience_threshold,
                )
            st.session_state.last_raw_response = response
            st.session_state.session_context = response.get("session_context")
            _update_proactive_state_from_response(response)
            reply_text = response.get("response") or "[Empty response from backend]"
            assistant_record = {
                "role": "assistant",
                "content": reply_text,
                "context_items": response.get("context_items", []),
                "captured_memories": response.get("captured_memories", []),
                "metrics": response.get("metrics", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except requests.HTTPError as exc:
            st.error(f"Chat request failed: {exc}")
            st.session_state.last_error = str(exc)
            st.session_state.last_raw_response = None
        except requests.RequestException as exc:
            st.error(f"Network error: {exc}")
            st.session_state.last_error = str(exc)
            st.session_state.last_raw_response = None

        if assistant_record:
            st.session_state.messages.append(assistant_record)

    for msg in st.session_state.messages:
        render_message(msg)


if __name__ == "__main__":
    main()
