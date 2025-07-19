"""
George: World-Class Human-AI Cognition Interface
==============================================

A comprehensive Streamlit interface showcasing the full Human-AI cognition system:
- Executive functioning (goals, tasks, decisions, resource management)
- Multi-modal memory systems (STM, LTM, episodic, semantic, procedural, prospective)
- Attention mechanisms with neural enhancement
- Metacognitive reflection and self-monitoring
- Dream-state consolidation and learning
- Real-time cognitive state visualization

This interface demonstrates human-like AI with transparent, explainable processes.
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "http://127.0.0.1:8000/api"
AGENT_URL = f"{BASE_URL}/agent"
EXECUTIVE_URL = f"{BASE_URL}/executive"
MEMORY_URL = f"{BASE_URL}/memory"

st.set_page_config(
    page_title="George - Human-AI Cognition Agent", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_request(method: str, url: str, **kwargs) -> Dict[str, Any]:
    """Make a safe HTTP request with error handling."""
    try:
        response = getattr(requests, method.lower())(url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status": "error"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "status": "error"}

def get_agent_status() -> Dict[str, Any]:
    """Fetch comprehensive agent cognitive state."""
    return safe_request("GET", f"{AGENT_URL}/status")

def get_executive_status() -> Dict[str, Any]:
    """Fetch executive functioning status."""
    return safe_request("GET", f"{EXECUTIVE_URL}/status")

def get_memory_status() -> Dict[str, Any]:
    """Fetch memory system status."""
    return safe_request("GET", f"{MEMORY_URL}/status")

def create_cognitive_load_chart(status: Dict) -> go.Figure:
    """Create a real-time cognitive load visualization."""
    fig = go.Figure()
    
    # Extract cognitive metrics
    attention_status = status.get("attention_status", {})
    memory_status = status.get("memory_status", {})
    executive_status = status.get("executive_status", {})
    
    # Cognitive load gauge
    cognitive_load = attention_status.get("cognitive_load", 0.0)
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = cognitive_load,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cognitive Load"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_memory_distribution_chart(status: Dict) -> go.Figure:
    """Create memory system distribution visualization."""
    memory_status = status.get("memory_status", {})
    
    # Extract memory counts
    stm_count = memory_status.get("stm", {}).get("vector_db_count", 0)
    ltm_count = memory_status.get("ltm", {}).get("memory_count", 0)
    episodic_count = memory_status.get("episodic", {}).get("total_memories", 0)
    semantic_count = memory_status.get("semantic", {}).get("fact_count", 0)
    procedural_count = memory_status.get("procedural", {}).get("procedure_count", 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=["STM", "LTM", "Episodic", "Semantic", "Procedural"],
            y=[stm_count, ltm_count, episodic_count, semantic_count, procedural_count],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
    ])
    
    fig.update_layout(
        title="Memory System Distribution",
        xaxis_title="Memory Type",
        yaxis_title="Count",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'active_goals' not in st.session_state:
    st.session_state.active_goals = []

if 'cognitive_timeline' not in st.session_state:
    st.session_state.cognitive_timeline = []

if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False

# =============================================================================
# MAIN INTERFACE
# =============================================================================

st.title("🧠 George: Human-AI Cognition Agent")
st.markdown("*Biologically-inspired AI with executive functioning, multi-modal memory, and metacognitive awareness*")

# =============================================================================
# COGNITIVE STATUS DASHBOARD
# =============================================================================

# Real-time status fetching
agent_status = get_agent_status()
executive_status = get_executive_status()
memory_status = get_memory_status()

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    cognitive_load = agent_status.get("attention_status", {}).get("cognitive_load", 0.0)
    st.metric("Cognitive Load", f"{cognitive_load:.2%}", delta=f"{cognitive_load-0.5:.2%}")

with col2:
    active_goals = len(executive_status.get("goals", {}).get("active_goals", []))
    st.metric("Active Goals", active_goals)

with col3:
    total_memories = sum([
        memory_status.get("stm", {}).get("vector_db_count", 0),
        memory_status.get("ltm", {}).get("memory_count", 0),
        memory_status.get("episodic", {}).get("total_memories", 0)
    ])
    st.metric("Total Memories", total_memories)

with col4:
    attention_focus = len(agent_status.get("attention_status", {}).get("current_focus", []))
    st.metric("Attention Items", attention_focus)

# =============================================================================
# CHAT INTERFACE
# =============================================================================

st.markdown("---")
st.subheader("💬 Cognitive Conversation")

# Enhanced chat input with context awareness
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "Talk to George:", 
            height=100, 
            placeholder="Ask me anything or give me a task to plan..."
        )
    
    with col2:
        st.markdown("**Options:**")
        include_reflection = st.checkbox("Include reflection", value=False)
        create_goal = st.checkbox("Create goal if needed", value=True)
        use_memory_context = st.checkbox("Use memory context", value=True)
        
    submit = st.form_submit_button("💭 Send", use_container_width=True)

if submit and user_input.strip():
    # Enhanced processing with multiple cognitive systems
    with st.spinner("🧠 George is thinking..."):
        payload = {
            "text": user_input.strip(),
            "include_reflection": include_reflection,
            "create_goal": create_goal,
            "use_memory_context": use_memory_context
        }
        
        # Process through cognitive agent
        response_data = safe_request("POST", f"{AGENT_URL}/process", json=payload)
        
        if "error" not in response_data:
            # Extract response components
            ai_response = response_data.get("response", "[No response]")
            memory_context = response_data.get("memory_context", [])
            memory_events = response_data.get("memory_events", [])
            rationale = response_data.get("rationale", None)
            cognitive_state = response_data.get("cognitive_state", {})
            
            # Auto-create goal if requested and appropriate
            if create_goal and ("goal" in user_input.lower() or "task" in user_input.lower()):
                goal_data = safe_request("POST", f"{EXECUTIVE_URL}/goals", json={
                    "description": user_input.strip(),
                    "priority": 0.7,
                    "metadata": {"source": "chat_interface", "auto_created": True}
                })
                if "error" not in goal_data:
                    memory_events.append(f"Created goal: {goal_data.get('goal_id', 'Unknown')}")
            
            # Store in chat history
            st.session_state.chat_history.append({
                "timestamp": datetime.now(),
                "user": user_input,
                "ai": ai_response,
                "context": memory_context,
                "memory_events": memory_events,
                "rationale": rationale,
                "cognitive_state": cognitive_state
            })
        else:
            st.session_state.chat_history.append({
                "timestamp": datetime.now(),
                "user": user_input,
                "ai": f"[Error: {response_data.get('error')}]",
                "context": [],
                "memory_events": [],
                "rationale": None,
                "cognitive_state": {}
            })

# Enhanced chat history display
st.markdown("### 🔄 Conversation History")
for i, turn in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
    with st.expander(f"💭 {turn['timestamp'].strftime('%H:%M:%S')} - {turn['user'][:50]}..."):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**George:** {turn['ai']}")
            
            if turn['context']:
                st.markdown("**Memory Context:**")
                for ctx in turn['context']:
                    st.caption(f"• {ctx.get('content', '')[:100]}... (Relevance: {ctx.get('relevance', 0):.2f})")
            
            if turn['memory_events']:
                st.info("**Learning Events:**")
                for event in turn['memory_events']:
                    st.markdown(f"• {event}")
            
            if turn['rationale']:
                st.caption(f"**Rationale:** {turn['rationale']}")
        
        with col2:
            if turn['cognitive_state']:
                st.markdown("**Cognitive State:**")
                for key, value in turn['cognitive_state'].items():
                    st.metric(key.replace("_", " ").title(), value)

# =============================================================================
# SIDEBAR: COMPREHENSIVE COGNITIVE CONTROLS
# =============================================================================

st.sidebar.header("🎛️ Cognitive Controls")

# System Controls
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("🔄 Refresh Status"):
    st.rerun()

st.sidebar.markdown("---")

# Cognitive State Visualization
st.sidebar.subheader("🧠 Real-Time Cognitive State")

if agent_status and "error" not in agent_status:
    # Cognitive metrics
    attention_status = agent_status.get("attention_status", {})
    memory_status_agent = agent_status.get("memory_status", {})
    
    st.sidebar.metric("Cognitive Mode", agent_status.get("cognitive_mode", "Unknown"))
    st.sidebar.metric("Attention Load", f"{attention_status.get('cognitive_load', 0):.1%}")
    st.sidebar.metric("Fatigue Level", f"{attention_status.get('fatigue_level', 0):.1%}")
    st.sidebar.metric("Focus Items", len(attention_status.get('current_focus', [])))
    
    # Cognitive load visualization
    if st.sidebar.checkbox("Show Cognitive Load Chart"):
        fig_load = create_cognitive_load_chart(agent_status)
        st.sidebar.plotly_chart(fig_load, use_container_width=True)

else:
    st.sidebar.warning("Agent status unavailable")

st.sidebar.markdown("---")

# Executive Functions Control
st.sidebar.subheader("🎯 Executive Functions")

# Goal management
if st.sidebar.button("📊 View Active Goals"):
    goals_data = safe_request("GET", f"{EXECUTIVE_URL}/goals", params={"active_only": True})
    if "error" not in goals_data:
        st.session_state.active_goals = goals_data.get("goals", [])

# Quick goal creation
with st.sidebar.form("quick_goal"):
    st.markdown("**Quick Goal Creation:**")
    goal_desc = st.text_input("Goal Description")
    goal_priority = st.slider("Priority", 0.0, 1.0, 0.5)
    goal_deadline = st.date_input("Deadline (optional)")
    
    if st.form_submit_button("🎯 Create Goal"):
        goal_payload = {
            "description": goal_desc,
            "priority": goal_priority,
            "deadline": goal_deadline.isoformat() if goal_deadline else None
        }
        result = safe_request("POST", f"{EXECUTIVE_URL}/goals", json=goal_payload)
        if "error" not in result:
            st.sidebar.success(f"Goal created: {result.get('goal_id', 'Unknown')}")
        else:
            st.sidebar.error(f"Error: {result.get('error')}")

# Decision making
if st.sidebar.button("🤔 Make Strategic Decision"):
    with st.sidebar.form("decision_form"):
        decision_context = st.text_area("Decision Context")
        options = st.text_area("Options (one per line)")
        criteria = st.text_area("Criteria (one per line)")
        
        if st.form_submit_button("🎲 Decide"):
            decision_payload = {
                "context": decision_context,
                "options": [opt.strip() for opt in options.split('\n') if opt.strip()],
                "criteria": [crit.strip() for crit in criteria.split('\n') if crit.strip()]
            }
            result = safe_request("POST", f"{EXECUTIVE_URL}/decisions", json=decision_payload)
            if "error" not in result:
                st.sidebar.success(f"Decision: {result.get('selected_option', 'Unknown')}")
                st.sidebar.info(f"Confidence: {result.get('confidence', 0):.2f}")
            else:
                st.sidebar.error(f"Error: {result.get('error')}")

st.sidebar.markdown("---")

# Memory System Controls
st.sidebar.subheader("🗃️ Memory Systems")

# Memory search
memory_query = st.sidebar.text_input("🔍 Search All Memory Systems")
memory_system = st.sidebar.selectbox("System", ["all", "stm", "ltm", "episodic", "semantic", "procedural"])

if st.sidebar.button("🔍 Search Memory") and memory_query.strip():
    search_results = safe_request("POST", f"{MEMORY_URL}/search", json={
        "query": memory_query.strip(),
        "system": memory_system,
        "max_results": 5
    })
    
    if "error" not in search_results:
        results = search_results.get("results", [])
        if results:
            st.sidebar.markdown("**🔍 Search Results:**")
            for result in results:
                memory_type = result.get('type', 'Unknown')
                content = result.get('content', '')[:80]
                relevance = result.get('relevance', 0)
                st.sidebar.markdown(f"**[{memory_type}]** {content}... ({relevance:.2f})")
        else:
            st.sidebar.info("No memories found")
    else:
        st.sidebar.error(f"Search failed: {search_results.get('error')}")

# Procedural memory
if st.sidebar.button("⚙️ Add Procedure"):
    with st.sidebar.form("procedure_form"):
        proc_desc = st.text_input("Procedure Description")
        proc_steps = st.text_area("Steps (one per line)")
        proc_tags = st.text_input("Tags (comma-separated)")
        proc_system = st.selectbox("Storage", ["ltm", "stm"])
        
        if st.form_submit_button("💾 Save Procedure"):
            procedure_payload = {
                "description": proc_desc,
                "steps": [step.strip() for step in proc_steps.split('\n') if step.strip()],
                "tags": [tag.strip() for tag in proc_tags.split(',') if tag.strip()],
                "memory_type": proc_system
            }
            result = safe_request("POST", f"{BASE_URL}/procedural/store", json=procedure_payload)
            if "error" not in result:
                st.sidebar.success(f"Procedure saved: {result.get('id', 'Unknown')}")
            else:
                st.sidebar.error(f"Error: {result.get('error')}")

# Prospective memory (reminders)
if st.sidebar.button("⏰ Add Reminder"):
    with st.sidebar.form("reminder_form"):
        reminder_task = st.text_input("Reminder Task")
        # Use date_input and time_input instead of datetime_input
        default_dt = datetime.now() + timedelta(hours=1)
        reminder_date = st.date_input("Date", default_dt.date())
        reminder_time_val = st.time_input("Time", default_dt.time())
        # Combine date and time into a datetime object
        reminder_datetime = datetime.combine(reminder_date, reminder_time_val)
        
        if st.form_submit_button("⏰ Set Reminder"):
            reminder_payload = {
                "task": reminder_task,
                "due_time": reminder_datetime.isoformat()
            }
            result = safe_request("POST", f"{BASE_URL}/prospective/store", json=reminder_payload)
            if "error" not in result:
                st.sidebar.success(f"Reminder set: {result.get('id', 'Unknown')}")
            else:
                st.sidebar.error(f"Error: {result.get('error')}")

st.sidebar.markdown("---")

# Metacognitive Controls
st.sidebar.subheader("🔬 Metacognitive Functions")

if st.sidebar.button("🤔 Trigger Reflection"):
    reflection_result = safe_request("POST", f"{AGENT_URL}/reflect")
    if "error" not in reflection_result:
        report = reflection_result.get("report", {})
        st.sidebar.success("Reflection completed!")
        st.sidebar.json(report)
    else:
        st.sidebar.error(f"Reflection failed: {reflection_result.get('error')}")

if st.sidebar.button("😴 Trigger Dream Consolidation"):
    dream_result = safe_request("POST", f"{AGENT_URL}/memory/consolidate")
    if "error" not in dream_result:
        st.sidebar.success("Dream consolidation triggered!")
        events = dream_result.get("consolidation_events", [])
        for event in events:
            st.sidebar.info(event)
    else:
        st.sidebar.error(f"Consolidation failed: {dream_result.get('error')}")

# Cognitive break
if st.sidebar.button("☕ Take Cognitive Break"):
    break_result = safe_request("POST", f"{AGENT_URL}/cognitive_break", json={"duration_minutes": 2.0})
    if "error" not in break_result:
        st.sidebar.success("Cognitive break completed!")
        reduction = break_result.get("cognitive_load_reduction", 0)
        st.sidebar.info(f"Cognitive load reduced by {reduction:.1%}")
    else:
        st.sidebar.error(f"Break failed: {break_result.get('error')}")

# =============================================================================
# MAIN CONTENT TABS
# =============================================================================

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Cognitive Analytics", "🎯 Executive Dashboard", "🗃️ Memory Explorer", "🔬 System Diagnostics", "⚙️ Advanced Settings"])

# =============================================================================
# TAB 1: COGNITIVE ANALYTICS
# =============================================================================

with tab1:
    st.subheader("📊 Cognitive Performance Analytics")
    
    if agent_status and "error" not in agent_status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧠 Cognitive Load Analysis")
            fig_load = create_cognitive_load_chart(agent_status)
            st.plotly_chart(fig_load, use_container_width=True)
            
            # Attention focus breakdown
            attention_focus = agent_status.get("attention_status", {}).get("current_focus", [])
            if attention_focus:
                focus_df = pd.DataFrame([
                    {"Item": item.get("content", "")[:30], "Priority": item.get("priority", 0)}
                    for item in attention_focus
                ])
                fig_focus = px.bar(focus_df, x="Item", y="Priority", title="Current Attention Focus")
                st.plotly_chart(fig_focus, use_container_width=True)
        
        with col2:
            st.markdown("#### 🗃️ Memory Distribution")
            combined_status = {**agent_status, **memory_status}
            fig_memory = create_memory_distribution_chart(combined_status)
            st.plotly_chart(fig_memory, use_container_width=True)
            
            # Performance metrics over time (simulated)
            if st.session_state.chat_history:
                timestamps = [turn["timestamp"] for turn in st.session_state.chat_history[-20:]]
                cognitive_loads = [turn.get("cognitive_state", {}).get("cognitive_load", 0.5) for turn in st.session_state.chat_history[-20:]]
                
                performance_df = pd.DataFrame({
                    "Time": timestamps,
                    "Cognitive Load": cognitive_loads
                })
                
                fig_timeline = px.line(performance_df, x="Time", y="Cognitive Load", 
                                     title="Cognitive Load Over Time")
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    else:
        st.warning("Cognitive analytics unavailable - agent not responding")

# =============================================================================
# TAB 2: EXECUTIVE DASHBOARD
# =============================================================================

with tab2:
    st.subheader("🎯 Executive Functioning Dashboard")
    
    if executive_status and "error" not in executive_status:
        # Goals overview
        goals = executive_status.get("goals", {})
        active_goals = goals.get("active_goals", [])
        completed_goals = goals.get("completed_goals", [])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Goals", len(active_goals))
        with col2:
            st.metric("Completed Goals", len(completed_goals))
        with col3:
            completion_rate = len(completed_goals) / max(len(active_goals) + len(completed_goals), 1)
            st.metric("Completion Rate", f"{completion_rate:.1%}")
        
        # Active goals table
        if active_goals:
            st.markdown("#### 🎯 Active Goals")
            goals_df = pd.DataFrame([
                {
                    "ID": goal.get("id", "")[:8],
                    "Description": goal.get("description", "")[:50],
                    "Priority": goal.get("priority", 0),
                    "Status": goal.get("status", "unknown"),
                    "Created": goal.get("created_at", "")[:10]
                }
                for goal in active_goals
            ])
            st.dataframe(goals_df, use_container_width=True)
        
        # Decision history
        decisions = executive_status.get("recent_decisions", [])
        if decisions:
            st.markdown("#### 🤔 Recent Decisions")
            decisions_df = pd.DataFrame([
                {
                    "Context": decision.get("context", "")[:40],
                    "Selected": decision.get("selected_option", ""),
                    "Confidence": decision.get("confidence", 0),
                    "Timestamp": decision.get("timestamp", "")[:19]
                }
                for decision in decisions[-10:]
            ])
            st.dataframe(decisions_df, use_container_width=True)
        
        # Resource allocation
        resources = executive_status.get("resources", {})
        if resources:
            st.markdown("#### 🔋 Resource Allocation")
            resource_df = pd.DataFrame([
                {"Resource": k.replace("_", " ").title(), "Allocation": v}
                for k, v in resources.items()
            ])
            fig_resources = px.bar(resource_df, x="Resource", y="Allocation", 
                                 title="Current Resource Allocation")
            st.plotly_chart(fig_resources, use_container_width=True)
    
    else:
        st.warning("Executive dashboard unavailable - system not responding")

# =============================================================================
# TAB 3: MEMORY EXPLORER
# =============================================================================

with tab3:
    st.subheader("🗃️ Memory System Explorer")
    
    # Memory system selector
    memory_system_tab = st.selectbox("Select Memory System", 
                                   ["Overview", "Short-Term (STM)", "Long-Term (LTM)", 
                                    "Episodic", "Semantic", "Procedural", "Prospective"])
    
    if memory_system_tab == "Overview":
        if memory_status and "error" not in memory_status:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Memory Statistics")
                for system, stats in memory_status.items():
                    st.markdown(f"**{system.upper()}:**")
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            st.text(f"  {key}: {value}")
            
            with col2:
                # Memory distribution pie chart
                memory_counts = {}
                if "stm" in memory_status and "vector_db_count" in memory_status["stm"]:
                    memory_counts["STM"] = memory_status["stm"]["vector_db_count"]
                if "ltm" in memory_status and "memory_count" in memory_status["ltm"]:
                    memory_counts["LTM"] = memory_status["ltm"]["memory_count"]
                if "episodic" in memory_status and "total_memories" in memory_status["episodic"]:
                    memory_counts["Episodic"] = memory_status["episodic"]["total_memories"]
                if "semantic" in memory_status and "fact_count" in memory_status["semantic"]:
                    memory_counts["Semantic"] = memory_status["semantic"]["fact_count"]
                
                if memory_counts:
                    fig_pie = px.pie(values=list(memory_counts.values()), 
                                   names=list(memory_counts.keys()),
                                   title="Memory Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Memory statistics unavailable")
    
    # Individual memory system exploration would be implemented here
    # This is a comprehensive framework - specific implementations would follow

# =============================================================================
# TAB 4: SYSTEM DIAGNOSTICS
# =============================================================================

with tab4:
    st.subheader("🔬 System Diagnostics & Health")
    
    # System health checks
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏥 Health Status")
        
        # Agent health
        agent_healthy = agent_status and "error" not in agent_status
        st.markdown(f"**Agent:** {'✅ Healthy' if agent_healthy else '❌ Unhealthy'}")
        
        # Executive health
        exec_healthy = executive_status and "error" not in executive_status
        st.markdown(f"**Executive:** {'✅ Healthy' if exec_healthy else '❌ Unhealthy'}")
        
        # Memory health
        memory_healthy = memory_status and "error" not in memory_status
        st.markdown(f"**Memory:** {'✅ Healthy' if memory_healthy else '❌ Unhealthy'}")
        
        # API connectivity
        api_health = safe_request("GET", f"{BASE_URL}/health")
        api_healthy = "error" not in api_health
        st.markdown(f"**API:** {'✅ Connected' if api_healthy else '❌ Disconnected'}")
    
    with col2:
        st.markdown("#### 📈 Performance Metrics")
        
        if agent_status and "error" not in agent_status:
            perf_metrics = agent_status.get("performance_metrics", {})
            for metric, value in perf_metrics.items():
                st.metric(metric.replace("_", " ").title(), value)
    
    # Raw status data (for debugging)
    if st.checkbox("Show Raw Status Data"):
        st.markdown("#### 🔧 Raw System Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Agent Status:**")
            st.json(agent_status)
        
        with col2:
            st.markdown("**Executive Status:**")
            st.json(executive_status)
        
        with col3:
            st.markdown("**Memory Status:**")
            st.json(memory_status)

# =============================================================================
# TAB 5: ADVANCED SETTINGS
# =============================================================================

with tab5:
    st.subheader("⚙️ Advanced Configuration")
    
    st.markdown("#### 🎛️ System Parameters")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Memory Settings:**")
        stm_capacity = st.slider("STM Capacity", 1, 20, 7)
        ltm_threshold = st.slider("LTM Storage Threshold", 0.0, 1.0, 0.7)
        consolidation_interval = st.slider("Auto-consolidation (minutes)", 5, 60, 15)
        
        st.markdown("**Attention Settings:**")
        attention_threshold = st.slider("Attention Threshold", 0.0, 1.0, 0.5)
        fatigue_rate = st.slider("Fatigue Accumulation Rate", 0.001, 0.1, 0.01)
        recovery_rate = st.slider("Recovery Rate", 0.01, 0.2, 0.05)
    
    with col2:
        st.markdown("**Executive Settings:**")
        goal_priority_weight = st.slider("Goal Priority Weight", 0.1, 2.0, 1.0)
        decision_confidence_threshold = st.slider("Decision Confidence Threshold", 0.0, 1.0, 0.6)
        resource_reallocation_rate = st.slider("Resource Reallocation Rate", 0.01, 0.5, 0.1)
        
        st.markdown("**Interface Settings:**")
        auto_refresh = st.checkbox("Auto-refresh status", value=False)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 30, 5)
        show_debug_info = st.checkbox("Show debug information", value=False)
    
    # Apply settings
    if st.button("💾 Apply Configuration"):
        config_payload = {
            "memory": {
                "stm_capacity": stm_capacity,
                "ltm_threshold": ltm_threshold,
                "consolidation_interval": consolidation_interval
            },
            "attention": {
                "threshold": attention_threshold,
                "fatigue_rate": fatigue_rate,
                "recovery_rate": recovery_rate
            },
            "executive": {
                "goal_priority_weight": goal_priority_weight,
                "decision_confidence_threshold": decision_confidence_threshold,
                "resource_reallocation_rate": resource_reallocation_rate
            }
        }
        
        result = safe_request("POST", f"{BASE_URL}/config/update", json=config_payload)
        if "error" not in result:
            st.success("Configuration updated successfully!")
        else:
            st.error(f"Configuration update failed: {result.get('error')}")
    
    # System reset options
    st.markdown("---")
    st.markdown("#### 🚨 System Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear All Memory"):
            if st.checkbox("Confirm memory clear"):
                clear_result = safe_request("POST", f"{MEMORY_URL}/clear_all")
                if "error" not in clear_result:
                    st.success("All memories cleared!")
                else:
                    st.error(f"Clear failed: {clear_result.get('error')}")
    
    with col2:
        if st.button("🔄 Reset Cognitive State"):
            reset_result = safe_request("POST", f"{AGENT_URL}/reset_cognitive_state")
            if "error" not in reset_result:
                st.success("Cognitive state reset!")
            else:
                st.error(f"Reset failed: {reset_result.get('error')}")
    
    with col3:
        if st.button("♻️ Full System Reset"):
            if st.checkbox("Confirm full reset"):
                reset_result = safe_request("POST", f"{BASE_URL}/system/reset")
                if "error" not in reset_result:
                    st.success("System fully reset!")
                    st.session_state.chat_history = []
                else:
                    st.error(f"Reset failed: {reset_result.get('error')}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>George: Human-AI Cognition Agent</strong><br>
    Biologically-inspired AI with executive functioning, multi-modal memory, and metacognitive awareness<br>
    <a href="http://127.0.0.1:8000/docs" target="_blank">API Documentation</a> | 
    <a href="https://github.com/doodmeister/human_ai_local" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)

# Auto-refresh functionality
if st.session_state.get("show_advanced", False) and auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
