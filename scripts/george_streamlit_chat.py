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


# ============================================================================
# Memory Browser API Functions
# ============================================================================

def fetch_memories(base_url: str, system: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch memories from a specific memory system."""
    try:
        resp = requests.get(f"{base_url}/memory/{system}/list", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("memories", data.get("items", []))
    except requests.RequestException as exc:
        st.error(f"Failed to fetch {system} memories: {exc}")
        return []


def search_memories(base_url: str, system: str, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Search memories in a specific memory system."""
    try:
        payload = {"query": query, "max_results": max_results}
        resp = requests.post(f"{base_url}/memory/{system}/search", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", data.get("memories", []))
    except requests.RequestException as exc:
        st.error(f"Failed to search {system} memories: {exc}")
        return []


def delete_memory(base_url: str, system: str, memory_id: str) -> bool:
    """Delete a memory from a specific memory system."""
    try:
        resp = requests.delete(f"{base_url}/memory/{system}/delete/{memory_id}", timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        st.error(f"Failed to delete memory: {exc}")
        return False


def fetch_semantic_facts(base_url: str, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch or search semantic facts."""
    try:
        if query:
            payload = {"query": query}
            resp = requests.post(f"{base_url}/semantic/fact/search", json=payload, timeout=10)
        else:
            # List all - use search with empty query
            payload = {"query": ""}
            resp = requests.post(f"{base_url}/semantic/fact/search", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("facts", data.get("results", []))
    except requests.RequestException as exc:
        st.error(f"Failed to fetch semantic facts: {exc}")
        return []


def fetch_prospective_memories(base_url: str) -> List[Dict[str, Any]]:
    """Fetch prospective memories (reminders)."""
    try:
        # Prefer the canonical reminders API.
        resp = requests.get(f"{base_url}/agent/reminders", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("reminders", [])
    except requests.RequestException as exc:
        # Fallback for older servers that only expose /prospective/*.
        try:
            resp = requests.get(f"{base_url}/prospective/search", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("events", data.get("reminders", []))
        except requests.RequestException:
            st.error(f"Failed to fetch prospective memories: {exc}")
            return []


def fetch_procedural_memories(base_url: str) -> List[Dict[str, Any]]:
    """Fetch procedural memories (skills/procedures)."""
    try:
        resp = requests.get(f"{base_url}/procedure/list", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("procedures", data.get("items", []))
    except requests.RequestException as exc:
        st.error(f"Failed to fetch procedural memories: {exc}")
        return []


# ============== Learning Dashboard Helpers ==============

def fetch_learning_metrics(base_url: str) -> Dict[str, Any]:
    """Fetch learning metrics from the executive system."""
    try:
        resp = requests.get(f"{base_url}/executive/learning/metrics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        # Try alternative endpoint
        try:
            resp = requests.get(f"{base_url}/executive/status", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("learning_metrics", {})
        except requests.RequestException as exc:
            st.error(f"Failed to fetch learning metrics: {exc}")
            return {}


def fetch_experiments(base_url: str) -> List[Dict[str, Any]]:
    """Fetch A/B test experiments."""
    try:
        resp = requests.get(f"{base_url}/executive/experiments", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("experiments", [])
    except requests.RequestException:
        return []


def fetch_execution_outcomes(base_url: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent execution outcomes."""
    try:
        resp = requests.get(f"{base_url}/executive/outcomes", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("outcomes", [])
    except requests.RequestException:
        return []


def render_learning_dashboard(base_url: str) -> None:
    """Render the learning dashboard interface."""
    st.header("📊 Learning Dashboard")
    st.write("Monitor ML model performance, A/B test results, and system learning trends.")
    
    # Refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        refresh = st.button("🔄 Refresh", key="learning_refresh")
    
    # Fetch learning metrics
    metrics = {}
    if refresh or "learning_metrics_cache" not in st.session_state:
        with st.spinner("Loading learning metrics..."):
            metrics = fetch_learning_metrics(base_url)
            st.session_state.learning_metrics_cache = metrics
    else:
        metrics = st.session_state.get("learning_metrics_cache", {})
    
    # Overview metrics
    st.subheader("📈 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    decision_acc = metrics.get("decision_accuracy", {})
    planning_acc = metrics.get("planning_accuracy", {})
    scheduling_acc = metrics.get("scheduling_accuracy", {})
    trends = metrics.get("improvement_trends", {})
    
    with col1:
        overall_acc = decision_acc.get("overall_accuracy", 0) if decision_acc else 0
        st.metric(
            "Decision Accuracy",
            f"{overall_acc:.1%}" if overall_acc else "N/A",
            delta=f"{trends.get('decision_improvement', 0):.1%}" if trends.get('decision_improvement') else None
        )
    
    with col2:
        plan_success = planning_acc.get("plan_success_rate", 0) if planning_acc else 0
        st.metric(
            "Planning Success",
            f"{plan_success:.1%}" if plan_success else "N/A",
            delta=f"{trends.get('planning_improvement', 0):.1%}" if trends.get('planning_improvement') else None
        )
    
    with col3:
        sched_acc = scheduling_acc.get("time_accuracy", 0) if scheduling_acc else 0
        st.metric(
            "Scheduling Accuracy",
            f"{sched_acc:.1%}" if sched_acc else "N/A",
            delta=f"{trends.get('scheduling_improvement', 0):.1%}" if trends.get('scheduling_improvement') else None
        )
    
    with col4:
        total_outcomes = decision_acc.get("total_decisions", 0) if decision_acc else 0
        st.metric("Total Decisions", total_outcomes if total_outcomes else "N/A")
    
    st.divider()
    
    # Tabs for different views
    tab_strategies, tab_experiments, tab_outcomes = st.tabs([
        "🎯 Strategy Performance",
        "🧪 A/B Experiments",
        "📜 Recent Outcomes"
    ])
    
    with tab_strategies:
        render_strategy_performance(metrics)
    
    with tab_experiments:
        render_experiments(base_url)
    
    with tab_outcomes:
        render_recent_outcomes(base_url)


def render_strategy_performance(metrics: Dict[str, Any]) -> None:
    """Render strategy performance comparison."""
    st.markdown("#### Strategy Comparison")
    
    decision_acc = metrics.get("decision_accuracy", {})
    
    if not decision_acc:
        st.info("No strategy performance data available yet. Execute some goals to generate data.")
        return
    
    # Strategy breakdown
    strategy_stats = decision_acc.get("by_strategy", {})
    
    if strategy_stats:
        st.markdown("**Performance by Strategy:**")
        
        for strategy, stats in strategy_stats.items():
            with st.expander(f"📊 {strategy.replace('_', ' ').title()}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    success_rate = stats.get("success_rate", 0)
                    st.metric("Success Rate", f"{success_rate:.1%}")
                
                with col2:
                    avg_confidence = stats.get("avg_confidence", 0)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                with col3:
                    count = stats.get("count", 0)
                    st.metric("Uses", count)
    else:
        # Show sample data placeholder
        st.info("Strategy comparison will appear here after multiple goals are executed with different strategies.")
        
        # Show available strategies
        st.markdown("**Available Decision Strategies:**")
        strategies = [
            ("weighted_scoring", "Classic weighted sum of criteria scores"),
            ("ahp", "Analytic Hierarchy Process - multi-criteria with consistency checks"),
            ("pareto", "Multi-objective Pareto frontier analysis"),
        ]
        
        for name, desc in strategies:
            st.caption(f"• **{name}**: {desc}")


def render_experiments(base_url: str) -> None:
    """Render A/B experiment results."""
    st.markdown("#### A/B Test Experiments")
    
    experiments = fetch_experiments(base_url)
    
    if not experiments:
        st.info("No A/B experiments running. Experiments allow comparing decision strategies to find the best performer.")
        
        # Show how to create an experiment
        with st.expander("💡 How to create an experiment"):
            st.markdown("""
            A/B experiments compare different decision strategies:
            
            ```python
            from src.executive.learning import create_experiment_manager, AssignmentMethod
            
            manager = create_experiment_manager()
            exp = manager.create_experiment(
                name="Strategy Comparison",
                strategies=["weighted_scoring", "ahp"],
                assignment_method=AssignmentMethod.EPSILON_GREEDY
            )
            manager.start_experiment(exp.experiment_id)
            ```
            
            The system will automatically assign strategies and track outcomes.
            """)
        return
    
    for exp in experiments:
        status_emoji = {
            "running": "🟢",
            "completed": "✅",
            "stopped": "🔴"
        }.get(exp.get("status", ""), "⚪")
        
        with st.expander(f"{status_emoji} {exp.get('name', 'Experiment')}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.caption(f"**Status:** {exp.get('status', 'unknown')}")
                st.caption(f"**Method:** {exp.get('assignment_method', 'N/A')}")
            
            with col2:
                st.caption(f"**Strategies:** {', '.join(exp.get('strategies', []))}")
                st.caption(f"**Total Assignments:** {exp.get('total_assignments', 0)}")
            
            with col3:
                if exp.get("recommended_strategy"):
                    st.success(f"**Winner:** {exp.get('recommended_strategy')}")
                    st.caption(f"**Confidence:** {exp.get('confidence', 'N/A')}")
            
            # Strategy results
            results = exp.get("strategy_results", {})
            if results:
                st.markdown("**Results by Strategy:**")
                
                result_cols = st.columns(len(results))
                for i, (strategy, data) in enumerate(results.items()):
                    with result_cols[i]:
                        st.markdown(f"**{strategy}**")
                        st.metric("Success Rate", f"{data.get('success_rate', 0):.1%}")
                        st.caption(f"Samples: {data.get('count', 0)}")


def render_recent_outcomes(base_url: str) -> None:
    """Render recent execution outcomes."""
    st.markdown("#### Recent Execution Outcomes")
    
    outcomes = fetch_execution_outcomes(base_url)
    
    if not outcomes:
        st.info("No execution outcomes recorded yet. Complete some goal executions to see results here.")
        return
    
    for outcome in outcomes:
        success = outcome.get("success", False)
        status_emoji = "✅" if success else "❌"
        
        with st.expander(f"{status_emoji} {outcome.get('goal_title', 'Goal')}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption(f"**Goal ID:** {outcome.get('goal_id', 'N/A')}")
                st.caption(f"**Success:** {'Yes' if success else 'No'}")
                st.caption(f"**Score:** {outcome.get('outcome_score', 0):.2f}")
            
            with col2:
                st.caption(f"**Strategy:** {outcome.get('strategy', 'N/A')}")
                st.caption(f"**Plan Steps:** {outcome.get('plan_length', 0)}")
                st.caption(f"**Time:** {outcome.get('execution_time', 'N/A')}")
            
            # Accuracy metrics
            acc_metrics = outcome.get("accuracy_metrics", {})
            if acc_metrics:
                st.markdown("**Accuracy Metrics:**")
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    time_acc = acc_metrics.get("time_accuracy", 0)
                    st.metric("Time Accuracy", f"{time_acc:.1%}" if time_acc else "N/A")
                
                with metric_cols[1]:
                    plan_adh = acc_metrics.get("plan_adherence", 0)
                    st.metric("Plan Adherence", f"{plan_adh:.1%}" if plan_adh else "N/A")
                
                with metric_cols[2]:
                    goal_ach = acc_metrics.get("goal_achievement", 0)
                    st.metric("Goal Achievement", f"{goal_ach:.1%}" if goal_ach else "N/A")
            
            # Deviations
            deviations = outcome.get("deviations", [])
            if deviations:
                st.markdown("**Deviations:**")
                for dev in deviations:
                    st.caption(f"⚠️ {dev}")


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
    # Memory browser settings
    if "memory_browser_system" not in st.session_state:
        st.session_state.memory_browser_system = "stm"
    if "memory_browser_search" not in st.session_state:
        st.session_state.memory_browser_search = ""
    if "memory_browser_cache" not in st.session_state:
        st.session_state.memory_browser_cache = {}
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"
    # Learning dashboard settings
    if "learning_metrics_cache" not in st.session_state:
        st.session_state.learning_metrics_cache = {}


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
    """Render compact goal creation form."""
    # Inline quick-add form
    col1, col2 = st.columns([3, 1])
    with col1:
        goal_title = st.text_input(
            "Quick goal",
            placeholder="What do you want to accomplish?",
            key="quick_goal_input",
            label_visibility="collapsed"
        )
    with col2:
        create_clicked = st.button("➕", key="quick_create_goal", help="Create goal")
    
    if create_clicked and goal_title.strip():
        result = create_goal_remote(base_url, goal_title.strip(), priority=0.5)
        if result and result.get("status") == "success":
            st.success("✓ Created")
            st.session_state.active_goals_refresh = True
            st.rerun()
    
    # Advanced options in expander
    with st.expander("⚙️ Advanced options", expanded=False):
        adv_description = st.text_area(
            "Description",
            placeholder="Additional context...",
            key="adv_goal_description",
            height=60
        )
        
        col1, col2 = st.columns(2)
        with col1:
            priority_label = st.selectbox(
                "Priority",
                options=["🟢 Low", "🟡 Medium", "🔴 High", "⚫ Critical"],
                index=1,
                key="adv_goal_priority"
            )
            priority_map = {"🟢 Low": 0.25, "🟡 Medium": 0.5, "🔴 High": 0.75, "⚫ Critical": 0.95}
            priority = priority_map[priority_label]
        
        with col2:
            deadline_date = st.date_input("Deadline (optional)", value=None, key="adv_goal_deadline")
        
        if st.button("Create with options", key="adv_create_goal"):
            if not goal_title.strip() and not st.session_state.get("quick_goal_input", "").strip():
                st.warning("Enter a goal title above")
            else:
                title = goal_title.strip() or st.session_state.get("quick_goal_input", "").strip()
                deadline_str = str(deadline_date) if deadline_date else None
                result = create_goal_remote(
                    base_url, title,
                    description=adv_description.strip(),
                    priority=priority,
                    deadline=deadline_str
                )
                if result and result.get("status") == "success":
                    st.success("✓ Goal created")
                    st.session_state.active_goals_refresh = True
                    st.rerun()


def execute_goal_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Execute a goal through the integrated pipeline."""
    try:
        resp = requests.post(f"{base_url}/executive/goals/{goal_id}/execute", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to execute goal: {exc}")
        return None


def get_goal_plan_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Get GOAP plan for a goal."""
    try:
        resp = requests.get(f"{base_url}/executive/goals/{goal_id}/plan", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to get plan: {exc}")
        return None


def get_goal_schedule_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Get CP-SAT schedule for a goal."""
    try:
        resp = requests.get(f"{base_url}/executive/goals/{goal_id}/schedule", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to get schedule: {exc}")
        return None


def get_goal_execution_status_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Get execution status for a goal."""
    try:
        resp = requests.get(f"{base_url}/executive/goals/{goal_id}/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to get execution status: {exc}")
        return None


def render_active_goals(base_url: str) -> None:
    """Render compact active goals panel in sidebar."""
    st.subheader("🎯 Goals")
    
    # Refresh trigger
    if st.session_state.get("active_goals_refresh"):
        st.session_state.active_goals_refresh = False
    
    goals = get_goals_remote(base_url, active_only=True)
    st.session_state.active_goals = goals
    
    if not goals:
        st.caption("No active goals. Add one below!")
        return
    
    # Compact goal list
    for goal in goals:
        goal_id = goal.get("id", "")
        title = goal.get("title", "Untitled goal")
        priority = goal.get("priority", 0.5)
        status = goal.get("status", "pending")
        progress = goal.get("progress", 0.0)
        
        # Priority indicator
        priority_dot = "🔴" if priority >= 0.75 else "🟡" if priority >= 0.5 else "🟢"
        
        # Compact single-line display with hover actions
        col1, col2 = st.columns([4, 1])
        with col1:
            # Clickable goal title
            if st.button(
                f"{priority_dot} {title[:25]}{'...' if len(title) > 25 else ''}",
                key=f"select_goal_{goal_id}",
                help=f"{title}\nStatus: {status} | Progress: {progress*100:.0f}%",
                use_container_width=True
            ):
                st.session_state.selected_goal_id = goal_id
                st.rerun()
        
        with col2:
            # Single action button - execute (most important action)
            exec_result = st.session_state.get(f"exec_result_{goal_id}")
            if exec_result:
                st.caption("✅")
            elif st.button("▶️", key=f"exec_{goal_id}", help="Execute"):
                with st.spinner("..."):
                    result = execute_goal_remote(base_url, goal_id)
                    if result and result.get("status") == "success":
                        st.session_state[f"exec_result_{goal_id}"] = result
                        st.rerun()


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
        
        st.markdown("---")
        render_active_goals(st.session_state.api_base_url)
        render_goal_creator(st.session_state.api_base_url)
        
        st.markdown("---")
        st.subheader("⏰ Reminders")
        render_sidebar_reminders(st.session_state.api_base_url)
        
        # Compact reminder creation
        col1, col2 = st.columns([3, 1])
        with col1:
            reminder_content = st.text_input(
                "Quick reminder",
                placeholder="Remind me to...",
                key="quick_reminder_input",
                label_visibility="collapsed"
            )
        with col2:
            reminder_minutes = st.number_input(
                "min",
                min_value=1,
                max_value=480,
                value=15,
                step=5,
                key="quick_reminder_minutes",
                label_visibility="collapsed"
            )
        
        if st.button("➕ Add", key="quick_add_reminder", use_container_width=True):
            trimmed = reminder_content.strip()
            if trimmed:
                # Auto-link to selected goal if viewing one
                metadata = {}
                active_goals = st.session_state.get("active_goals", [])
                if active_goals and len(active_goals) == 1:
                    # Auto-link to only goal
                    metadata["goal_id"] = active_goals[0].get("id")
                    metadata["goal_title"] = active_goals[0].get("title")
                
                reminder = create_reminder_remote(
                    st.session_state.api_base_url,
                    trimmed,
                    int(reminder_minutes) * 60,
                    metadata=metadata if metadata else None,
                )
                if reminder:
                    st.success("✓")
                    _add_reminder_to_upcoming_cache(reminder)
                    st.rerun()
            else:
                st.warning("Enter reminder text")
        
        # Debug section (collapsed)
        with st.expander("🔧 Debug", expanded=False):
            if st.session_state.last_error:
                st.warning(f"Last error: {st.session_state.last_error}")
            if st.session_state.last_raw_response:
                st.json(st.session_state.last_raw_response)
            metrics = st.session_state.get("reminder_metrics", {"counts": {}})
            if metrics.get("counts"):
                st.caption("Reminder telemetry")
                st.json(metrics["counts"])

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
            # Show detected goal notification
            detected_goal = message.get("detected_goal")
            if detected_goal:
                goal_id = detected_goal.get("goal_id", "")
                goal_title = detected_goal.get("title", "Untitled goal")
                goal_priority = detected_goal.get("priority", "MEDIUM")
                deadline = detected_goal.get("deadline")
                
                st.success(f"🎯 **Goal Created:** {goal_title}")
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.caption(f"Priority: {goal_priority}")
                with col2:
                    st.caption(f"Deadline: {deadline or 'None'}")
                with col3:
                    if st.button("📊 View", key=f"view_detected_{goal_id}", help="View goal details"):
                        st.session_state.selected_goal_id = goal_id
                        st.session_state.active_goals_refresh = True
                        st.rerun()
            
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
            
            # Show intent classification (in debug expander)
            intent = message.get("intent")
            if intent and intent.get("intent_type"):
                with st.expander("🧠 Intent Classification", expanded=False):
                    st.caption(f"**Type:** {intent.get('intent_type')}")
                    st.caption(f"**Confidence:** {intent.get('confidence', 0):.0%}")
                    if intent.get("entities"):
                        st.caption(f"**Entities:** {intent.get('entities')}")


def render_goal_details_page(base_url: str, goal_id: str) -> None:
    """Render detailed goal execution view."""
    st.subheader("📋 Goal Execution Details")
    
    if st.button("⬅️ Back to Chat"):
        st.session_state.selected_goal_id = None
        st.rerun()
    
    # Get execution status
    status_response = get_goal_execution_status_remote(base_url, goal_id)
    
    if not status_response or status_response.get("status") == "not_executed":
        st.info("This goal has not been executed yet. Click the ▶️ button to execute.")
        return
    
    context = status_response.get("context", {})
    
    # Header with status
    st.markdown(f"### {context.get('goal_title', 'Goal Details')}")
    exec_status = context.get("status", "UNKNOWN")
    
    status_colors = {
        "COMPLETED": "green",
        "FAILED": "red",
        "EXECUTING": "orange",
        "PLANNING": "blue",
        "SCHEDULING": "blue"
    }
    color = status_colors.get(exec_status, "gray")
    st.markdown(f"**Status:** :{color}[{exec_status}]")
    
    # Pipeline stages
    st.markdown("#### Pipeline Stages")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("1️⃣ Decision", f"{context.get('decision_time_ms', 0)}ms")
        if context.get("decision_result"):
            st.caption(f"Strategy: {context['decision_result'].get('strategy', 'N/A')}")
            st.caption(f"Confidence: {context['decision_result'].get('confidence', 0):.1%}")
    
    with col2:
        st.metric("2️⃣ Planning", f"{context.get('planning_time_ms', 0)}ms")
        st.caption(f"Actions: {context.get('total_actions', 0)}")
    
    with col3:
        st.metric("3️⃣ Scheduling", f"{context.get('scheduling_time_ms', 0)}ms")
        st.caption(f"Tasks: {context.get('scheduled_tasks', 0)}")
    
    # GOAP Plan viewer
    st.markdown("#### 📝 GOAP Plan")
    plan_response = get_goal_plan_remote(base_url, goal_id)
    
    if plan_response and plan_response.get("status") == "success":
        plan = plan_response.get("plan", {})
        steps = plan.get("steps", [])
        
        st.info(f"**Plan:** {plan.get('length', 0)} steps | Cost: {plan.get('total_cost', 0):.1f} | Planning: {plan.get('planning_time_ms', 0)}ms")
        
        for i, step in enumerate(steps, 1):
            with st.expander(f"Step {i}: {step.get('name', 'Unknown')}" + (" ✅" if i == 1 else "")):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Preconditions:**")
                    preconditions = step.get("preconditions", {})
                    if preconditions:
                        for key, value in preconditions.items():
                            st.caption(f"- {key}: {value}")
                    else:
                        st.caption("None")
                
                with col2:
                    st.markdown("**Effects:**")
                    effects = step.get("effects", {})
                    if effects:
                        for key, value in effects.items():
                            st.caption(f"- {key}: {value}")
                    else:
                        st.caption("None")
                
                st.metric("Cost", f"{step.get('cost', 0):.1f}")
    else:
        st.warning("No plan available yet.")
    
    # Schedule viewer (simplified for now)
    st.markdown("#### 📅 Schedule")
    schedule_response = get_goal_schedule_remote(base_url, goal_id)
    
    if schedule_response and schedule_response.get("status") == "success":
        schedule = schedule_response.get("schedule", {})
        tasks = schedule.get("tasks", [])
        
        st.info(f"**Schedule:** {len(tasks)} tasks | Makespan: {schedule.get('makespan', 0)}s | Robustness: {schedule.get('robustness_score', 0):.1%}")
        
        for task in tasks:
            with st.expander(f"Task: {task.get('name', task.get('id', 'Unknown'))}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption(f"**Start:** {task.get('scheduled_start', 'N/A')}")
                    st.caption(f"**End:** {task.get('scheduled_end', 'N/A')}")
                    st.caption(f"**Duration:** {task.get('duration', 0)}s")
                
                with col2:
                    st.caption(f"**Priority:** {task.get('priority', 0)}")
                    resources = task.get('resources', [])
                    st.caption(f"**Resources:** {', '.join(resources) if resources else 'None'}")
                    dependencies = task.get('dependencies', [])
                    st.caption(f"**Dependencies:** {', '.join(dependencies) if dependencies else 'None'}")
    else:
        st.warning("No schedule available yet.")


# ============================================================================
# Memory Browser UI
# ============================================================================

def render_memory_item(memory: Dict[str, Any], system: str, base_url: str, index: int) -> None:
    """Render a single memory item with actions."""
    memory_id = memory.get("id") or memory.get("memory_id") or f"memory_{index}"
    content = memory.get("content") or memory.get("detailed_content") or memory.get("summary") or str(memory)
    
    # Truncate long content for display
    display_content = content[:300] + "..." if len(str(content)) > 300 else content
    
    with st.expander(f"📝 {display_content[:60]}...", expanded=False):
        st.markdown(f"**Content:** {content}")
        
        # Show metadata in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if memory.get("importance"):
                st.metric("Importance", f"{memory.get('importance', 0):.2f}")
            if memory.get("activation"):
                st.metric("Activation", f"{memory.get('activation', 0):.2f}")
        
        with col2:
            if memory.get("encoding_time"):
                st.caption(f"**Created:** {memory.get('encoding_time', 'Unknown')[:19]}")
            elif memory.get("timestamp"):
                ts = memory.get("timestamp")
                if isinstance(ts, str):
                    st.caption(f"**Created:** {ts[:19]}")
            if memory.get("last_access"):
                st.caption(f"**Last Access:** {memory.get('last_access', 'N/A')[:19]}")
        
        with col3:
            if memory.get("access_count"):
                st.caption(f"**Access Count:** {memory.get('access_count', 0)}")
            if memory.get("tags"):
                tags = memory.get("tags", [])
                if isinstance(tags, list):
                    st.caption(f"**Tags:** {', '.join(tags[:5])}")
                elif isinstance(tags, str):
                    st.caption(f"**Tags:** {tags}")
        
        # Additional metadata
        with st.expander("📊 Full Metadata", expanded=False):
            st.json(memory)
        
        # Delete button
        if st.button("🗑️ Delete", key=f"delete_{system}_{memory_id}_{index}"):
            if delete_memory(base_url, system, memory_id):
                st.success("Memory deleted successfully")
                # Clear cache to force refresh
                if system in st.session_state.memory_browser_cache:
                    del st.session_state.memory_browser_cache[system]
                st.rerun()


def render_memory_browser(base_url: str) -> None:
    """Render the memory browser interface."""
    st.header("🧠 Memory Browser")
    st.write("Browse, search, and manage memories across all memory systems.")
    
    # Memory system selector
    col1, col2 = st.columns([2, 3])
    
    with col1:
        system = st.selectbox(
            "Memory System",
            options=["stm", "ltm", "episodic", "semantic", "prospective", "procedural"],
            format_func=lambda x: {
                "stm": "📝 Short-Term Memory (STM)",
                "ltm": "💾 Long-Term Memory (LTM)",
                "episodic": "📖 Episodic Memory",
                "semantic": "🔤 Semantic Memory",
                "prospective": "⏰ Prospective Memory",
                "procedural": "⚙️ Procedural Memory"
            }.get(x, x),
            key="memory_browser_system"
        )
    
    with col2:
        search_query = st.text_input(
            "Search memories",
            value=st.session_state.memory_browser_search,
            placeholder="Enter search query...",
            key="memory_search_input"
        )
        st.session_state.memory_browser_search = search_query
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        refresh = st.button("🔄 Refresh", use_container_width=True)
    
    with col2:
        clear_cache = st.button("🧹 Clear Cache", use_container_width=True)
        if clear_cache:
            st.session_state.memory_browser_cache = {}
            st.success("Cache cleared")
    
    # Fetch memories based on system type
    memories = []
    cache_key = f"{system}_{search_query}"
    
    if refresh or cache_key not in st.session_state.memory_browser_cache:
        with st.spinner(f"Loading {system} memories..."):
            if system in ["stm", "ltm", "episodic"]:
                if search_query:
                    memories = search_memories(base_url, system, search_query)
                else:
                    memories = fetch_memories(base_url, system)
            elif system == "semantic":
                memories = fetch_semantic_facts(base_url, search_query if search_query else None)
            elif system == "prospective":
                memories = fetch_prospective_memories(base_url)
            elif system == "procedural":
                memories = fetch_procedural_memories(base_url)
            
            st.session_state.memory_browser_cache[cache_key] = memories
    else:
        memories = st.session_state.memory_browser_cache.get(cache_key, [])
    
    # Display memories
    st.divider()
    
    if not memories:
        st.info(f"No memories found in {system}. Try a different search or memory system.")
    else:
        st.success(f"Found **{len(memories)}** memories in {system}")
        
        # Display memories
        for i, memory in enumerate(memories):
            render_memory_item(memory, system, base_url, i)


def main() -> None:
    st.set_page_config(page_title="George Chat", page_icon="G", layout="wide")
    ensure_state()

    base_url = st.session_state.api_base_url
    backend_ok = check_backend(base_url)
    config = render_sidebar(backend_ok)
    flags = config["flags"]
    salience_threshold = config["salience_threshold"]
    
    # Check if viewing goal details
    if st.session_state.get("selected_goal_id"):
        render_goal_details_page(base_url, st.session_state.selected_goal_id)
        return

    st.title("George Chat Interface")
    
    # Create tabs for Chat, Memory Browser, and Learning Dashboard
    tab_chat, tab_memory, tab_learning = st.tabs(["💬 Chat", "🧠 Memory Browser", "📊 Learning"])
    
    with tab_chat:
        st.write("Chat with the cognitive backend. Each turn retrieves short-term memories first, then falls back to long-term memories when needed.")
        
        # Tip about natural language goal creation
        with st.expander("💡 **Tip:** You can create goals naturally!", expanded=False):
            st.markdown("""
            Try saying things like:
            - *"I need to finish the report by Friday"*
            - *"Help me plan a project for next week"*
            - *"My goal is to learn Python this month"*
            - *"I should review the code before the meeting"*
            
            George will automatically detect your goal, create it, and start the **Goal→Decision→Plan→Schedule** pipeline!
            """)
        
        render_proactive_banner(base_url)
        render_session_context_panel()

        prompt: Optional[str] = None
        if backend_ok:
            prompt = st.chat_input("Send a message")
        else:
            st.info("Start the backend with `python main.py api` to enable chatting.")

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
                    "detected_goal": response.get("detected_goal"),
                    "intent": response.get("intent"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                # Auto-refresh goals sidebar if a goal was detected
                if response.get("detected_goal"):
                    st.session_state.active_goals_refresh = True
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
    
    with tab_memory:
        if backend_ok:
            render_memory_browser(base_url)
        else:
            st.warning("Start the backend with `python main.py api` to browse memories.")
    
    with tab_learning:
        if backend_ok:
            render_learning_dashboard(base_url)
        else:
            st.warning("Start the backend with `python main.py api` to view learning metrics.")


if __name__ == "__main__":
    main()
