# Quick Reference: ExecutiveSystem API for UI Developers

**Audience**: Streamlit UI developers building visualization components  
**Backend**: All endpoints operational, fully tested  
**Base URL**: `http://localhost:8000` (then use `/executive/...` paths)

This doc works with both backends:

- Main server: `python main.py api`
- Simple server: `python -c "import uvicorn; from scripts.legacy.george_api_simple import app; uvicorn.run(app, port=8001)"`

Both expose unprefixed endpoints like `/executive/*` (no `/api` needed).

---

## Common Workflows

### 1. Create and Execute a Goal

```python
import requests
import streamlit as st

# Step 1: Create goal
response = requests.post(
    "http://localhost:8000/executive/goals",
    json={
        "title": "Analyze quarterly sales data",
        "description": "Generate insights from Q4 2024 sales",
        "priority": "HIGH",  # LOW, MEDIUM, HIGH, CRITICAL
        "deadline": "2025-01-31T23:59:59",
        "success_criteria": ["data_analyzed=True", "report_created=True"]
    }
)
goal_id = response.json()["goal"]["id"]

# Step 2: Execute integrated pipeline (Decision → Plan → Schedule)
response = requests.post(
    f"http://localhost:8000/executive/goals/{goal_id}/execute"
)
context = response.json()["execution_context"]

st.write(f"Status: {context['status']}")  # COMPLETED, FAILED, EXECUTING
st.write(f"Decision: {context['decision_time_ms']}ms")
st.write(f"Planning: {context['planning_time_ms']}ms")
st.write(f"Scheduling: {context['scheduling_time_ms']}ms")
st.write(f"Total Actions: {context['total_actions']}")
st.write(f"Tasks Scheduled: {context['scheduled_tasks']}")
```

---

### 2. Visualize GOAP Plan (Action Sequence)

```python
# Get plan from execution context
response = requests.get(
    f"http://localhost:8000/executive/goals/{goal_id}/plan"
)

if response.json()["status"] == "no_plan":
    st.warning("Execute the goal first to generate a plan")
else:
    plan = response.json()["plan"]
    
    # Display plan stepper
    st.subheader(f"📋 Plan ({plan['length']} steps, cost: {plan['total_cost']:.1f})")
    
    for i, step in enumerate(plan["steps"], 1):
        with st.expander(f"Step {i}: {step['name']}"):
            st.write("**Preconditions:**")
            st.json(step["preconditions"])
            st.write("**Effects:**")
            st.json(step["effects"])
            st.metric("Cost", f"{step['cost']:.1f}")
    
    st.info(f"Planning time: {plan['planning_time_ms']}ms | Heuristic: {plan['heuristic']}")
```

---

### 3. Display Schedule Gantt Chart

```python
import pandas as pd
import plotly.express as px
from datetime import datetime

# Get schedule from execution context
response = requests.get(
    f"http://localhost:8000/executive/goals/{goal_id}/schedule"
)

if response.json()["status"] == "no_schedule":
    st.warning("Execute the goal first to generate a schedule")
else:
    schedule = response.json()["schedule"]
    
    # Convert to DataFrame for Gantt chart
    tasks = []
    for task in schedule["tasks"]:
        if task["scheduled_start"] and task["scheduled_end"]:
            tasks.append({
                "Task": task["name"],
                "Start": datetime.fromisoformat(task["scheduled_start"]),
                "Finish": datetime.fromisoformat(task["scheduled_end"]),
                "Priority": task["priority"],
                "Resources": ", ".join(task["resources"])
            })
    
    df = pd.DataFrame(tasks)
    
    # Create Gantt chart with Plotly
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Priority",
        hover_data=["Resources"],
        title=f"Schedule (Makespan: {schedule['makespan']}s)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Robustness", f"{schedule['robustness_score']:.2%}")
    col2.metric("Cognitive Smoothness", f"{schedule['cognitive_smoothness']:.2%}")
    col3.metric("Scheduling Time", f"{schedule['scheduling_time_ms']}ms")
```

---

### 4. Monitor Execution Status

```python
# Get full execution context
response = requests.get(
    f"http://localhost:8000/executive/goals/{goal_id}/status"
)

if response.json()["status"] == "not_executed":
    st.info("Goal has not been executed yet")
else:
    context = response.json()["context"]
    
    # Pipeline status header
    st.subheader(f"🎯 {context['goal_title']}")
    
    # Status indicator
    status_color = {
        "COMPLETED": "green",
        "FAILED": "red",
        "EXECUTING": "orange",
        "PLANNING": "blue",
        "SCHEDULING": "blue"
    }
    st.markdown(f"**Status:** :{status_color.get(context['status'], 'gray')}[{context['status']}]")
    
    # Pipeline stages progress
    st.write("**Pipeline Stages:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "1️⃣ Decision",
            f"{context['decision_time_ms']}ms",
            delta=None
        )
        if context["decision_result"]:
            st.caption(f"Strategy: {context['decision_result']['strategy']}")
            st.caption(f"Confidence: {context['decision_result']['confidence']:.2%}")
    
    with col2:
        st.metric(
            "2️⃣ Planning",
            f"{context['planning_time_ms']}ms",
            delta=None
        )
        st.caption(f"Actions: {context['total_actions']}")
    
    with col3:
        st.metric(
            "3️⃣ Scheduling",
            f"{context['scheduling_time_ms']}ms",
            delta=None
        )
        st.caption(f"Tasks: {context['scheduled_tasks']}")
    
    # Outcome metrics (if completed)
    if context["actual_success"] is not None:
        st.write("**Execution Outcome:**")
        col1, col2 = st.columns(2)
        col1.metric("Success", "✅" if context["actual_success"] else "❌")
        col2.metric("Outcome Score", f"{context['outcome_score']:.2%}")
```

---

### 5. System Health Dashboard

```python
# Get system-wide metrics
response = requests.get("http://localhost:8000/executive/system/health")
health = response.json()["health"]

# Overview metrics
st.header("🏥 Executive System Health")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Goals", health["active_goals"])
col2.metric("Total Executions", health["total_executions"])
col3.metric("Success Rate", f"{health['success_rate']:.1%}")
col4.metric("Avg Execution Time", f"{health['avg_execution_time_ms']:.0f}ms")

# Subsystem health
st.subheader("Subsystem Status")
subsystems = health["subsystems"]
cols = st.columns(len(subsystems))

for i, (name, status) in enumerate(subsystems.items()):
    icon = "✅" if status == "healthy" else "❌"
    cols[i].metric(name.replace("_", " ").title(), f"{icon} {status}")

# Recent activity
st.subheader("Recent Activity")
for activity in health["recent_activity"][-5:]:  # Last 5
    st.write(f"- {activity['goal_id']}: {activity['status']} at {activity['timestamp']}")

st.caption(f"Last updated: {health['timestamp']}")
```

---

## API Endpoint Reference

### Goal Management

| Method | Endpoint | Purpose | Request Body |
|--------|----------|---------|--------------|
| POST | `/executive/goals` | Create goal | `{title, description, priority, deadline, success_criteria}` |
| GET | `/executive/goals?active_only=true` | List goals | Query params |
| GET | `/executive/goals/{goal_id}` | Get goal details | None |

### Execution Pipeline

| Method | Endpoint | Purpose | Returns |
|--------|----------|---------|---------|
| POST | `/executive/goals/{goal_id}/execute` | Run Decision→Plan→Schedule | ExecutionContext |
| GET | `/executive/goals/{goal_id}/status` | Get full execution context | ExecutionContext + metadata |

### Results Retrieval

| Method | Endpoint | Purpose | Returns |
|--------|----------|---------|---------|
| GET | `/executive/goals/{goal_id}/plan` | Get GOAP plan | {steps, total_cost, length} |
| GET | `/executive/goals/{goal_id}/schedule` | Get CP-SAT schedule | {tasks, makespan, quality_metrics} |

### System Monitoring

| Method | Endpoint | Purpose | Returns |
|--------|----------|---------|---------|
| GET | `/executive/status` | Basic status | Goal counts |
| GET | `/executive/system/health` | Full health metrics | Aggregated stats + subsystem status |

---

## Response Schemas

### ExecutionContext (from `/goals/{id}/execute` or `/goals/{id}/status`)

```json
{
  "goal_id": "goal_123",
  "goal_title": "Analyze data",
  "status": "COMPLETED",
  "decision_time_ms": 45,
  "planning_time_ms": 120,
  "scheduling_time_ms": 85,
  "total_actions": 5,
  "scheduled_tasks": 8,
  "makespan": 3600,
  "failure_reason": null,
  "actual_success": true,
  "outcome_score": 0.85,
  "decision_result": {
    "strategy": "ahp",
    "confidence": 0.92,
    "selected_option": "weighted_scoring"
  },
  "accuracy_metrics": {
    "time_accuracy_ratio": 0.95,
    "plan_adherence": 0.88
  }
}
```

### Plan (from `/goals/{id}/plan`)

```json
{
  "steps": [
    {
      "name": "gather_data",
      "preconditions": {"data_source": "available"},
      "effects": {"raw_data": true},
      "cost": 10.0
    }
  ],
  "total_cost": 150.0,
  "length": 5,
  "planning_time_ms": 120,
  "heuristic": "goal_distance"
}
```

### Schedule (from `/goals/{id}/schedule`)

```json
{
  "tasks": [
    {
      "id": "task_1",
      "name": "analyze_data",
      "scheduled_start": "2025-01-15T10:00:00",
      "scheduled_end": "2025-01-15T11:30:00",
      "duration": 90,
      "priority": 8,
      "resources": ["cpu", "memory"],
      "dependencies": ["task_0"]
    }
  ],
  "makespan": 3600,
  "robustness_score": 0.85,
  "cognitive_smoothness": 0.92,
  "scheduling_time_ms": 85
}
```

---

## Error Handling

```python
try:
    response = requests.post(f"http://localhost:8000/executive/goals/{goal_id}/execute")
    response.raise_for_status()
    data = response.json()
    
    if data.get("status") == "no_plan":
        st.warning(data["message"])
    else:
        # Process successful response
        st.success("Execution complete!")
        
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        st.error("Goal not found")
    elif e.response.status_code == 500:
        st.error(f"Server error: {e.response.json()['detail']}")
except requests.exceptions.RequestException as e:
    st.error(f"Network error: {str(e)}")
```

---

## Tips for UI Development

### 1. Caching API Calls

```python
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_goal_status(goal_id):
    response = requests.get(f"http://localhost:8000/executive/goals/{goal_id}/status")
    return response.json()
```

### 2. Real-time Updates

```python
# Auto-refresh for executing goals
if context["status"] in ["PLANNING", "SCHEDULING", "EXECUTING"]:
    st.info("Execution in progress...")
    time.sleep(2)
    st.rerun()
```

### 3. Progress Indicators

```python
# Pipeline stage progress bar
stages = ["Decision", "Planning", "Scheduling", "Executing", "Complete"]
current_stage = stages.index(context["status"]) if context["status"] in stages else 0
progress = current_stage / (len(stages) - 1)
st.progress(progress)
```

### 4. Conditional Rendering

```python
# Only show plan if execution has progressed past decision stage
if context["planning_time_ms"] > 0:
    plan_response = requests.get(f"http://localhost:8000/executive/goals/{goal_id}/plan")
    # Render plan visualization
```

---

## Next Steps

Ready to build UI? Start with:

1. **Task 4**: GOAP Plan Visualization (see Workflow #2 above)
2. **Task 5**: Schedule Gantt Chart (see Workflow #3 above)
3. **Task 7**: Execution Monitor (see Workflow #4 above)
4. **Task 8**: Health Dashboard (see Workflow #5 above)

All endpoints are live and tested. Happy coding! 🚀
