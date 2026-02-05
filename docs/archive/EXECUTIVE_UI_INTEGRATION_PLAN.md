# Executive Function UI Integration Plan

> **Historical plan:** This document describes a Nov 2025 integration plan. For current status and priorities, use:
> - Current state: [nextsteps.md](../nextsteps.md)
> - P1 plan (Jan 2026): [docs/P1_ACTION_PLAN.md](P1_ACTION_PLAN.md)
> - Long-term roadmap: [planning/roadmap.md](planning/roadmap.md)

**Date**: November 20, 2025  
**Goal**: Bridge the 90% gap between backend executive capabilities and UI exposure  
**Duration**: 6-8 weeks  
**Priority**: CRITICAL for roadmap alignment

---

## Overview

We have a fully functional executive function pipeline in the backend:
- Goal management with hierarchical dependencies
- GOAP planning with A* search
- CP-SAT constraint scheduling
- Full Goal→Decision→Plan→Schedule integration
- ML-based learning and A/B testing

**Current Problem:** 0% of this is visible in the George UI.

**Solution:** 4-phase incremental integration plan.

---

## Phase 1: Foundation & Goal Management (Week 1-2) ✅ COMPLETE

### Task 1.1: Mount Executive API Routes ✅ COMPLETE
**Priority**: P0  
**Effort**: 4 hours  
**Status**: ✅ DONE (January 2025)
**Files**: `start_server.py` (modified)

**Implementation:**
```python
# start_server.py - Executive routes mounted
from src.interfaces.api.executive_api import router as executive_router

app.include_router(chat_router, prefix="")
app.include_router(executive_router, prefix="/executive", tags=["executive"])
```

**API Endpoints Mounted (9 total):**

**Original CRUD (4):**
- `POST /executive/goals` - Create goal
- `GET /executive/goals` - List goals
- `GET /executive/goals/{goal_id}` - Get goal details
- `GET /executive/status` - Basic status

**New Integration Endpoints (5):**
- `POST /executive/goals/{goal_id}/execute` - Execute integrated pipeline (Decision → Plan → Schedule)
- `GET /executive/goals/{goal_id}/plan` - Get GOAP plan from execution context
- `GET /executive/goals/{goal_id}/schedule` - Get CP-SAT schedule from execution context
- `GET /executive/goals/{goal_id}/status` - Get full execution context and pipeline status
- `GET /executive/system/health` - System health metrics (active goals, success rate, timing)
- `GET /executive/goals/{goal_id}/status` - Get execution context
- `GET /executive/status` - Executive system health

**Validation:**
```bash
curl http://localhost:8000/executive/goals
# Should return: {"goals": []}
```

---

### Task 1.2: Create Goal Management Widget
**Priority**: P0  
**Effort**: 8 hours  
**Files**: `scripts/george_streamlit_chat.py`

**New Component: `render_goal_creator()`**

```python
def create_goal_remote(
    base_url: str,
    description: str,
    success_criteria: List[str],
    priority: str = "MEDIUM",
    deadline: Optional[datetime] = None,
    parent_goal_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Create a new goal via executive API."""
    payload = {
        "description": description,
        "success_criteria": success_criteria,
        "priority": priority,
    }
    if deadline:
        payload["deadline"] = deadline.isoformat()
    if parent_goal_id:
        payload["parent_goal_id"] = parent_goal_id
    
    try:
        resp = requests.post(f"{base_url}/executive/goals", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("goal")
    except requests.RequestException as exc:
        st.error(f"Failed to create goal: {exc}")
        return None


def render_goal_creator(base_url: str) -> None:
    """Render goal creation form in sidebar."""
    with st.expander("➕ Create Goal", expanded=False):
        goal_desc = st.text_input(
            "Goal description",
            placeholder="e.g., Deploy new feature to production",
            key="goal_desc_input"
        )
        
        success_criteria_text = st.text_area(
            "Success criteria (one per line)",
            placeholder="deployment_complete=True\ntests_passing=True\ndocs_updated=True",
            key="goal_criteria_input",
            help="Format: variable=value (e.g., 'data_analyzed=True')"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            priority = st.selectbox(
                "Priority",
                options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                index=1,
                key="goal_priority_input"
            )
        with col2:
            use_deadline = st.checkbox("Set deadline", key="goal_has_deadline")
        
        deadline = None
        if use_deadline:
            deadline = st.date_input("Deadline", key="goal_deadline_input")
        
        # TODO: Parent goal selection (hierarchical goals)
        
        if st.button("Create Goal", key="create_goal_btn"):
            if not goal_desc.strip():
                st.warning("Please provide a goal description")
                return
            
            criteria = [
                line.strip() 
                for line in success_criteria_text.split('\n') 
                if line.strip()
            ]
            
            if not criteria:
                st.warning("Please provide at least one success criterion")
                return
            
            goal = create_goal_remote(
                base_url,
                goal_desc.strip(),
                criteria,
                priority,
                deadline=datetime.combine(deadline, datetime.min.time()) if deadline else None
            )
            
            if goal:
                st.success(f"Goal created: {goal.get('id')}")
                st.session_state.active_goals_refresh = True
```

**UI Location:** Sidebar, below "Create reminder" section

---

### Task 1.3: Active Goals Sidebar Panel
**Priority**: P0  
**Effort**: 6 hours  
**Files**: `scripts/george_streamlit_chat.py`

**New Component: `render_active_goals()`**

```python
def get_goals_remote(base_url: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch goals from executive API."""
    try:
        params = {"status": status} if status else {}
        resp = requests.get(f"{base_url}/executive/goals", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("goals", [])
    except requests.RequestException as exc:
        st.error(f"Failed to fetch goals: {exc}")
        return []


def execute_goal_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Trigger goal execution via executive API."""
    try:
        resp = requests.post(
            f"{base_url}/executive/goals/{goal_id}/execute",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to execute goal: {exc}")
        return None


def render_active_goals(base_url: str) -> None:
    """Render active goals panel in sidebar."""
    st.subheader("🎯 Active Goals")
    
    # Refresh trigger
    if st.session_state.get("active_goals_refresh"):
        st.session_state.active_goals_refresh = False
    
    goals = get_goals_remote(base_url, status="active")
    
    if not goals:
        st.caption("No active goals")
        return
    
    for goal in goals:
        goal_id = goal.get("id")
        description = goal.get("description", "Untitled goal")
        priority = goal.get("priority", "MEDIUM")
        status = goal.get("status", "unknown")
        
        # Priority emoji
        priority_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }.get(priority, "⚪")
        
        with st.container():
            st.markdown(f"{priority_emoji} **{description}**")
            st.caption(f"Status: {status} | ID: {goal_id[:8]}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Execute", key=f"exec_goal_{goal_id}"):
                    with st.spinner("Starting execution..."):
                        result = execute_goal_remote(base_url, goal_id)
                        if result:
                            st.success("Execution started!")
                            st.session_state.selected_goal_id = goal_id
            with col2:
                if st.button("📊 Details", key=f"view_goal_{goal_id}"):
                    st.session_state.selected_goal_id = goal_id
            
            st.markdown("---")
```

**UI Location:** Sidebar, above "Upcoming reminders" section

---

### Task 1.4: Link Reminders to Goals
**Priority**: P1  
**Effort**: 3 hours  
**Files**: `scripts/george_streamlit_chat.py`

**Modification: Update `create_reminder_remote()`**

```python
def render_goal_linked_reminder_creator(base_url: str) -> None:
    """Enhanced reminder creator with goal linking."""
    st.subheader("Create reminder")
    
    # Existing reminder fields...
    reminder_content = st.text_input("Content", ...)
    reminder_minutes = st.number_input("Due in (minutes)", ...)
    
    # NEW: Goal selection
    goals = get_goals_remote(base_url, status="active")
    goal_options = ["None"] + [
        f"{g.get('description')} ({g.get('id')[:8]})" 
        for g in goals
    ]
    selected_goal_idx = st.selectbox(
        "Related goal (optional)",
        options=range(len(goal_options)),
        format_func=lambda i: goal_options[i],
        key="reminder_goal_link"
    )
    
    if st.button("➕ Add reminder", key="create_reminder_button"):
        # ... existing validation ...
        
        # NEW: Add goal_id to metadata
        metadata = {"note": reminder_metadata_note.strip()} if reminder_metadata_note.strip() else {}
        if selected_goal_idx > 0:  # 0 is "None"
            goal_id = goals[selected_goal_idx - 1].get("id")
            metadata["goal_id"] = goal_id
            metadata["goal_description"] = goals[selected_goal_idx - 1].get("description")
        
        reminder = create_reminder_remote(
            base_url,
            trimmed,
            int(reminder_minutes) * 60,
            metadata=metadata
        )
        # ... rest of existing code ...
```

**Benefit:** Reminders can now trigger goal-related actions or surface goal context.

---

## Phase 2: Planning Visualization (Week 3-4)

### Task 2.1: GOAP Plan Viewer Component
**Priority**: P0  
**Effort**: 10 hours  
**Files**: `scripts/george_streamlit_chat.py`

**New Component: `render_goal_plan()`**

```python
def get_goal_plan_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Fetch GOAP plan for a goal."""
    try:
        resp = requests.get(
            f"{base_url}/executive/goals/{goal_id}/plan",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch plan: {exc}")
        return None


def render_goal_plan(base_url: str, goal_id: str) -> None:
    """Render GOAP action plan visualization."""
    plan_data = get_goal_plan_remote(base_url, goal_id)
    
    if not plan_data or not plan_data.get("plan"):
        st.info("No plan generated yet. Execute the goal to create a plan.")
        return
    
    plan = plan_data["plan"]
    actions = plan.get("actions", [])
    
    st.subheader("📋 Action Plan")
    
    # Plan metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actions", len(actions))
    with col2:
        st.metric("Total Cost", f"{plan.get('total_cost', 0):.2f}")
    with col3:
        st.metric("Planning Time", f"{plan.get('planning_time_ms', 0):.0f} ms")
    
    # Action sequence
    st.markdown("### Action Sequence")
    for i, action in enumerate(actions, 1):
        name = action.get("name", "Unknown")
        cost = action.get("cost", 0)
        
        with st.expander(f"{i}. {name} (cost: {cost:.2f})", expanded=i==1):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Preconditions:**")
                preconditions = action.get("preconditions", {})
                if preconditions:
                    for key, value in preconditions.items():
                        st.write(f"- `{key} = {value}`")
                else:
                    st.caption("None")
            
            with col2:
                st.markdown("**Effects:**")
                effects = action.get("effects", {})
                if effects:
                    for key, value in effects.items():
                        st.write(f"- `{key} = {value}`")
                else:
                    st.caption("None")
    
    # Planning details
    with st.expander("Planning Metrics", expanded=False):
        st.write(f"**Heuristic:** {plan.get('heuristic', 'unknown')}")
        st.write(f"**Nodes Expanded:** {plan.get('nodes_expanded', 0)}")
        st.write(f"**Plan Length:** {plan.get('length', 0)}")
```

**UI Location:** Main content area when goal is selected

---

### Task 2.2: Goal Detail View
**Priority**: P0  
**Effort**: 6 hours  
**Files**: `scripts/george_streamlit_chat.py`

**New Page Component: `render_goal_detail_page()`**

```python
def render_goal_detail_page(base_url: str, goal_id: str) -> None:
    """Render full goal detail page with plan and schedule."""
    # Back button
    if st.button("← Back to Chat"):
        st.session_state.selected_goal_id = None
        st.rerun()
    
    # Fetch goal details
    try:
        resp = requests.get(f"{base_url}/executive/goals/{goal_id}", timeout=10)
        resp.raise_for_status()
        goal = resp.json().get("goal")
    except requests.RequestException as exc:
        st.error(f"Failed to fetch goal: {exc}")
        return
    
    # Goal header
    st.title(f"🎯 {goal.get('description')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", goal.get("status", "unknown"))
    with col2:
        st.metric("Priority", goal.get("priority", "MEDIUM"))
    with col3:
        deadline = goal.get("deadline")
        st.metric("Deadline", deadline or "None")
    
    # Success criteria
    st.subheader("Success Criteria")
    criteria = goal.get("success_criteria", [])
    for criterion in criteria:
        st.write(f"✓ {criterion}")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Plan", "📅 Schedule", "📊 Execution"])
    
    with tab1:
        render_goal_plan(base_url, goal_id)
    
    with tab2:
        render_goal_schedule(base_url, goal_id)  # Task 2.3
    
    with tab3:
        render_execution_context(base_url, goal_id)  # Phase 3
```

**Integration Point:** Main app routing

```python
def main():
    ensure_state()
    connection_ok = check_backend(st.session_state.api_base_url)
    
    # Check if viewing goal detail
    if st.session_state.get("selected_goal_id"):
        render_goal_detail_page(
            st.session_state.api_base_url,
            st.session_state.selected_goal_id
        )
        return
    
    # ... existing chat interface ...
```

---

### Task 2.3: Schedule Gantt Chart Viewer
**Priority**: P1  
**Effort**: 12 hours  
**Files**: `scripts/george_streamlit_chat.py`  
**Dependencies**: `plotly` (add to requirements.txt)

**New Component: `render_goal_schedule()`**

```python
def get_goal_schedule_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Fetch CP-SAT schedule for a goal."""
    try:
        resp = requests.get(
            f"{base_url}/executive/goals/{goal_id}/schedule",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch schedule: {exc}")
        return None


def render_goal_schedule(base_url: str, goal_id: str) -> None:
    """Render CP-SAT schedule with Gantt chart."""
    schedule_data = get_goal_schedule_remote(base_url, goal_id)
    
    if not schedule_data or not schedule_data.get("schedule"):
        st.info("No schedule generated yet.")
        return
    
    schedule = schedule_data["schedule"]
    tasks = schedule.get("tasks", [])
    
    st.subheader("📅 Task Schedule")
    
    # Schedule metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tasks", len(tasks))
    with col2:
        makespan = schedule.get("makespan", 0)
        st.metric("Makespan", f"{makespan} min")
    with col3:
        robustness = schedule.get("robustness_score", 0)
        st.metric("Robustness", f"{robustness:.2f}")
    with col4:
        cognitive = schedule.get("cognitive_smoothness", 0)
        st.metric("Cognitive Load", f"{cognitive:.2f}")
    
    # Gantt chart
    if tasks:
        import plotly.express as px
        import pandas as pd
        
        # Prepare data for Gantt
        gantt_data = []
        for task in tasks:
            gantt_data.append({
                "Task": task.get("name", "Unknown"),
                "Start": pd.to_datetime(task.get("scheduled_start")),
                "Finish": pd.to_datetime(task.get("scheduled_end")),
                "Priority": task.get("priority", 0),
                "Resource": ", ".join(task.get("resources", []))
            })
        
        df = pd.DataFrame(gantt_data)
        
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Priority",
            hover_data=["Resource"],
            title="Task Timeline"
        )
        fig.update_yaxes(autorange="reversed")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource utilization
    resources = schedule.get("resource_utilization", {})
    if resources:
        st.markdown("### Resource Utilization")
        for resource_name, utilization in resources.items():
            st.write(f"**{resource_name}**")
            st.progress(utilization)
            st.caption(f"{utilization*100:.1f}%")
    
    # Critical path
    critical_tasks = schedule.get("critical_path", [])
    if critical_tasks:
        st.markdown("### ⚠️ Critical Path")
        st.warning(f"{len(critical_tasks)} tasks have zero slack time")
        for task_name in critical_tasks:
            st.write(f"- {task_name}")
```

**Requirements Update:**
```txt
# Add to requirements.txt
plotly>=5.18.0
pandas>=2.1.0
```

---

## Phase 3: Execution Monitoring (Week 5-6)

### Task 3.1: Execution Context Display
**Priority**: P0  
**Effort**: 8 hours  
**Files**: `scripts/george_streamlit_chat.py`

**New Component: `render_execution_context()`**

```python
def get_execution_context_remote(base_url: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """Fetch execution context for a goal."""
    try:
        resp = requests.get(
            f"{base_url}/executive/goals/{goal_id}/status",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch execution context: {exc}")
        return None


def render_execution_context(base_url: str, goal_id: str) -> None:
    """Render execution context and pipeline status."""
    context_data = get_execution_context_remote(base_url, goal_id)
    
    if not context_data or not context_data.get("context"):
        st.info("Goal has not been executed yet.")
        return
    
    context = context_data["context"]
    
    st.subheader("🔄 Execution Pipeline")
    
    # Status indicator
    status = context.get("status", "IDLE")
    status_emoji = {
        "IDLE": "⚪",
        "PLANNING": "🔵",
        "SCHEDULING": "🟡",
        "EXECUTING": "🟢",
        "COMPLETED": "✅",
        "FAILED": "❌"
    }.get(status, "⚪")
    
    st.markdown(f"## {status_emoji} {status}")
    
    # Pipeline stages
    stages = ["Decision", "Planning", "Scheduling", "Execution"]
    current_stage = {
        "IDLE": 0,
        "PLANNING": 1,
        "SCHEDULING": 2,
        "EXECUTING": 3,
        "COMPLETED": 4,
        "FAILED": -1
    }.get(status, 0)
    
    cols = st.columns(4)
    for i, (col, stage) in enumerate(zip(cols, stages)):
        with col:
            if i < current_stage:
                st.success(f"✓ {stage}")
            elif i == current_stage:
                st.info(f"⟳ {stage}")
            else:
                st.caption(f"○ {stage}")
    
    # Timing metrics
    st.markdown("### ⏱️ Timing Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        decision_time = context.get("decision_time_ms", 0)
        st.metric("Decision", f"{decision_time:.0f} ms")
    with col2:
        planning_time = context.get("planning_time_ms", 0)
        st.metric("Planning", f"{planning_time:.0f} ms")
    with col3:
        scheduling_time = context.get("scheduling_time_ms", 0)
        st.metric("Scheduling", f"{scheduling_time:.0f} ms")
    
    # Decision result
    decision_result = context.get("decision_result")
    if decision_result:
        with st.expander("Decision Details", expanded=False):
            st.write(f"**Strategy:** {decision_result.get('strategy', 'unknown')}")
            st.write(f"**Confidence:** {decision_result.get('confidence', 0):.2f}")
            st.write(f"**Selected Option:** {decision_result.get('selected_option', 'N/A')}")
    
    # Completion info
    if status == "COMPLETED":
        st.markdown("### ✅ Completion Summary")
        col1, col2 = st.columns(2)
        with col1:
            success = context.get("actual_success", False)
            st.metric("Success", "✓" if success else "✗")
        with col2:
            score = context.get("outcome_score", 0)
            st.metric("Outcome Score", f"{score:.2f}")
        
        # Accuracy metrics
        accuracy = context.get("accuracy_metrics", {})
        if accuracy:
            with st.expander("Accuracy Metrics"):
                st.json(accuracy)
    
    # Failure info
    if status == "FAILED":
        st.error("Execution failed")
        error_msg = context.get("error_message", "Unknown error")
        st.write(error_msg)
```

**UI Location:** Third tab in goal detail view

---

### Task 3.2: System Health Dashboard
**Priority**: P1  
**Effort**: 8 hours  
**Files**: `scripts/george_streamlit_chat.py`

**New Component: `render_system_health_dashboard()`**

```python
def get_system_health_remote(base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch executive system health."""
    try:
        resp = requests.get(f"{base_url}/executive/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch system health: {exc}")
        return None


def render_system_health_dashboard(base_url: str) -> None:
    """Render system health dashboard."""
    with st.expander("🏥 System Health", expanded=False):
        health = get_system_health_remote(base_url)
        
        if not health:
            st.error("Unable to fetch system health")
            return
        
        # Core metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Goals", health.get("active_goals", 0))
        with col2:
            st.metric("Total Executions", health.get("total_executions", 0))
        with col3:
            st.metric("Success Rate", f"{health.get('success_rate', 0)*100:.1f}%")
        with col4:
            st.metric("Avg Execution Time", f"{health.get('avg_execution_time_ms', 0):.0f} ms")
        
        # Subsystem status
        st.markdown("#### Subsystem Status")
        subsystems = health.get("subsystems", {})
        for name, status in subsystems.items():
            status_icon = "✅" if status == "healthy" else "⚠️"
            st.write(f"{status_icon} **{name}**: {status}")
        
        # Recent activity
        recent = health.get("recent_activity", [])
        if recent:
            st.markdown("#### Recent Activity")
            for activity in recent[-5:]:
                st.caption(f"{activity.get('timestamp')}: {activity.get('event')}")
```

**UI Location:** Sidebar, below active goals

---

## Phase 4: Polish & Testing (Week 7-8)

### Task 4.1: Error Handling & Validation
**Priority**: P0  
**Effort**: 6 hours

- Add request timeouts for all API calls
- Implement retry logic for transient failures
- Add loading spinners for long operations
- Validate success criteria format (variable=value)
- Handle API unavailability gracefully

### Task 4.2: State Management
**Priority**: P0  
**Effort**: 4 hours

- Add `st.session_state` keys for goals, plans, schedules
- Implement cache invalidation on updates
- Add refresh buttons for data sync
- Handle race conditions (concurrent executions)

### Task 4.3: Documentation
**Priority**: P1  
**Effort**: 4 hours

- Update `docs/ui_showcase.md` with new features
- Add inline help text for complex fields
- Create video walkthrough (optional)
- Update README with executive function examples

### Task 4.4: Integration Testing
**Priority**: P0  
**Effort**: 8 hours

- Test full Goal→Decision→Plan→Schedule flow
- Verify reminder-to-goal linking
- Test edge cases (no plan, failed execution)
- Load testing with multiple concurrent goals
- Cross-browser testing (Chrome, Firefox, Safari)

---

## Implementation Checklist

### Week 1-2: Foundation ✅ 80% COMPLETE
- [x] Task 1.1: Mount executive API routes (4h) ✅ DONE
  - Added 5 new ExecutiveSystem endpoints (execute, plan, schedule, status, health)
  - All 9 endpoints tested and operational
  - See `docs/BACKEND_API_COMPLETION_SUMMARY.md`
- [x] Task 1.2: Create goal management widget (8h) ✅ DONE
  - `create_goal_remote()` function implemented
  - `render_goal_creator()` UI with priority/deadline pickers
  - Integrated into main Streamlit interface
- [x] Task 1.3: Active goals sidebar panel (6h) ✅ DONE
  - `get_goals_remote()` function implemented
  - `render_active_goals()` sidebar display
  - Priority emojis, progress bars, detail/link buttons
- [x] Task 1.4: Link reminders to goals (3h) ✅ DONE
  - Enhanced reminder creator with goal selection
  - Metadata includes goal_id and goal_title
- [ ] **NEXT**: Task 1.5: GOAP plan visualization UI (10h) 🔄
  - Backend endpoint ready: `GET /executive/goals/{id}/plan`
  - See `docs/UI_DEVELOPER_API_QUICKSTART.md` for implementation guide

### Week 3-4: Planning (Backend Ready, UI Pending)
- [ ] Task 2.1: GOAP plan viewer component (10h)
  - **Endpoint:** `GET /executive/goals/{goal_id}/plan`
  - **Returns:** {steps[], total_cost, length, planning_time_ms}
  - **UI:** Collapsible stepper showing preconditions → effects per action
- [ ] Task 2.2: Goal detail view page (6h)
  - **Endpoint:** `GET /executive/goals/{goal_id}/status`
  - **Returns:** Full ExecutionContext with all metrics
  - **UI:** Dedicated page with pipeline stages, timing, outcome
- [ ] Task 2.3: Schedule Gantt chart viewer (12h)
  - **Endpoint:** `GET /executive/goals/{goal_id}/schedule`
  - **Returns:** {tasks[], makespan, quality_metrics}
  - **UI:** Plotly timeline with resource/dependency overlays
- [ ] Add plotly to requirements.txt (if needed)
- [ ] Test: Execute goal, view plan, view schedule

### Week 5-6: Execution (Backend Ready, UI Pending)
- [ ] Task 3.1: Execution context display (8h)
  - **Endpoint:** `GET /executive/goals/{goal_id}/status`
  - **UI:** Real-time pipeline monitor (Decision → Plan → Schedule stages)
- [ ] Task 3.2: System health dashboard (8h)
  - **Endpoint:** `GET /executive/system/health`
  - **Returns:** {active_goals, success_rate, avg_time, subsystem_status}
  - **UI:** Overview metrics + subsystem health indicators
- [ ] Test: Monitor execution, check health metrics

### Week 7-8: Polish
- [ ] Task 4.1: Error handling & validation (6h)
- [ ] Task 4.2: State management (4h)
- [ ] Task 4.3: Documentation (4h)
- [ ] Task 4.4: Integration testing (8h)

---

## Backend API Status: ✅ COMPLETE

**All 9 endpoints operational and tested:**

### Original CRUD (4)
1. POST `/executive/goals` - Create goal
2. GET `/executive/goals` - List goals
3. GET `/executive/goals/{goal_id}` - Get goal details
4. GET `/executive/status` - Basic status

### New Integration Endpoints (5)
5. POST `/executive/goals/{goal_id}/execute` - Execute pipeline
6. GET `/executive/goals/{goal_id}/plan` - Get GOAP plan
7. GET `/executive/goals/{goal_id}/schedule` - Get CP-SAT schedule
8. GET `/executive/goals/{goal_id}/status` - Get execution context
9. GET `/executive/system/health` - System health metrics

**Documentation:**
- Full API reference: `docs/BACKEND_API_COMPLETION_SUMMARY.md`
- UI developer guide: `docs/UI_DEVELOPER_API_QUICKSTART.md`
- Code examples for all 5 workflows provided

---

## Success Metrics

### Coverage Improvement
- **Before:** Executive functions 0% UI coverage
- **After (Current):** Executive functions 40% UI coverage (goals + reminders)
- **After (Target):** Executive functions 80%+ UI coverage

### Functional Milestones
1. ✅ Create hierarchical goals via UI (DONE)
2. ✅ Link reminders to goals (DONE)
3. ⏳ Execute goals with one click (Backend ready, UI pending)
4. ⏳ View GOAP action plans (Backend ready, UI pending)
5. ⏳ View CP-SAT schedules (Backend ready, UI pending)
6. ⏳ Monitor execution pipeline status (Backend ready, UI pending)
7. ⏳ Track system health (Backend ready, UI pending)

### User Experience
- ✅ Users can create/view goals without API calls
- ⏳ Clear visibility into planning decisions (UI pending)
- ⏳ Real-time execution monitoring (UI pending)
- ⏳ Actionable error messages (polish phase)

---

## Risk Mitigation

### Risk 1: API Routes Not Ready ✅ RESOLVED
**Status:** All API endpoints implemented and tested  
**Actions Taken:** Added 5 new ExecutiveSystem endpoints, comprehensive documentation provided

### Risk 2: Plotly Performance
**Mitigation:** Limit Gantt chart to 50 tasks max, add pagination

### Risk 3: State Synchronization
**Mitigation:** Add explicit refresh buttons, implement polling for active executions

### Risk 4: Complex UX
**Mitigation:** Use progressive disclosure (expanders), add inline help, provide examples

---

## Next Steps (Immediate)

1. **Start with Task 1.1** - Mount executive API routes
2. **Validate** - Test `/executive/goals` endpoint
3. **Iterate** - Build goal creator widget
4. **Demo** - Show working goal creation by end of Week 1

**Let's begin with mounting the executive API routes!**
