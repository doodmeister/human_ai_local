"""
George - Production-Ready Streamlit Interface
Human-AI Cognitive Architecture Dashboard

Phase 1 Implementation:
- Enhanced Chat Interface with cognitive integration
- Memory Management Dashboard 
- Attention Monitor with real-time visualization
- Basic Executive Dashboard

Author: GitHub Copilot
Date: July 2025
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="George - Human-AI Cognitive Architecture",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-healthy { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-critical { color: #dc3545; }
    .cognitive-state {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .memory-system {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .attention-gauge {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

class GeorgeAPI:
    """API client for George's cognitive backend"""
    
    @staticmethod
    def get(endpoint: str, timeout: int = 60) -> Dict[str, Any]:
        """GET request to George API with configurable timeout"""
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out after {timeout}s. Agent may still be initializing...")
            return {"error": f"Timeout after {timeout}s", "status": "timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def post(endpoint: str, data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """POST request to George API with configurable timeout"""
        try:
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out after {timeout}s. Processing may be taking longer than expected...")
            return {"error": f"Timeout after {timeout}s", "status": "timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return {"error": str(e)}

def check_api_connection() -> bool:
    """Check if the George API is accessible and responsive"""
    try:
        # Simple health check with short timeout first
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def wait_for_agent_initialization():
    """Wait for agent to fully initialize with progress indication"""
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    if not st.session_state.agent_initialized:
        with st.spinner("🧠 Initializing George's cognitive architecture..."):
            # Check if API is responding
            if not check_api_connection():
                st.error("❌ Cannot connect to George API server. Please ensure:")
                st.markdown("""
                - The API server is running: `python start_server.py`
                - The server is accessible at http://localhost:8000
                - No firewall is blocking the connection
                """)
                st.stop()
            
            # Use initialization status endpoint for better feedback
            max_attempts = 12  # 120 seconds total
            for attempt in range(max_attempts):
                try:
                    init_status = GeorgeAPI.get("/api/agent/init-status", timeout=5)
                    status = init_status.get("status", "unknown")
                    message = init_status.get("message", "No status message")
                    
                    if status == "ready":
                        st.session_state.agent_initialized = True
                        st.success("✅ George's cognitive architecture is ready!")
                        time.sleep(1)  # Brief pause to show success message
                        return
                    elif status == "initializing":
                        st.info(f"⏳ {message} (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(10)
                    elif status == "error":
                        st.error(f"❌ Initialization failed: {message}")
                        st.stop()
                    else:
                        # Try to trigger initialization by calling agent status
                        GeorgeAPI.get("/api/agent/status", timeout=5)
                        time.sleep(5)
                except Exception as e:
                    st.warning(f"⚠️ Connection attempt {attempt + 1}/{max_attempts}: {str(e)}")
                    time.sleep(10)
            
            st.error("❌ George's cognitive architecture failed to initialize within 120 seconds")
            st.markdown("""
            **Troubleshooting steps:**
            1. Check if the API server is running: `python start_server.py`
            2. Check server logs for errors
            3. Ensure all dependencies are installed: `pip install -r requirements.txt`
            4. Try restarting the API server
            """)
            st.stop()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {}
    if 'memory_data' not in st.session_state:
        st.session_state.memory_data = {}
    if 'attention_history' not in st.session_state:
        st.session_state.attention_history = []
    if 'executive_data' not in st.session_state:
        st.session_state.executive_data = {}

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🧠 George - Human-AI Cognitive Architecture</h1>', unsafe_allow_html=True)
    st.markdown("*Production-Ready Cognitive Dashboard - Phase 1*")
    st.divider()

def render_cognitive_status_bar():
    """Render the cognitive status bar"""
    status_data = GeorgeAPI.get("/api/agent/status", timeout=30)  # Longer timeout for status
    
    if "error" not in status_data and status_data.get("status") != "timeout":
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            cognitive_mode = status_data.get("cognitive_status", {}).get("cognitive_mode", "UNKNOWN")
            st.markdown(f'<div class="cognitive-state">Mode: {cognitive_mode}</div>', unsafe_allow_html=True)
        
        with col2:
            cognitive_load = status_data.get("cognitive_status", {}).get("attention_status", {}).get("cognitive_load", 0.0)
            st.metric("Cognitive Load", f"{cognitive_load:.1%}", delta=None)
        
        with col3:
            stm_count = status_data.get("cognitive_status", {}).get("memory_status", {}).get("stm", {}).get("vector_db_count", 0)
            st.metric("STM Memories", stm_count)
        
        with col4:
            ltm_count = status_data.get("cognitive_status", {}).get("memory_status", {}).get("ltm", {}).get("memory_count", 0)
            st.metric("LTM Memories", ltm_count)
        
        with col5:
            active_processes = status_data.get("cognitive_status", {}).get("active_processes", 0)
            st.metric("Active Processes", active_processes)
        
        st.session_state.agent_status = status_data
    elif status_data.get("status") == "timeout":
        st.warning("⏱️ Cognitive status temporarily unavailable (processing request...)")
    else:
        st.error("❌ Unable to connect to George's cognitive backend")

def render_enhanced_chat():
    """Render the enhanced chat interface with cognitive integration"""
    st.subheader("💬 Enhanced Cognitive Chat")
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Chat with George:", placeholder="Ask me anything or give me a task...")
        with col2:
            include_reflection = st.checkbox("Include Reflection", value=False)
    
    if user_input and st.button("Send", type="primary"):
        # Display user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Send to George API
        chat_data = {
            "text": user_input,
            "include_reflection": include_reflection,
            "use_memory_context": True
        }
        
        with st.spinner("George is thinking..."):
            response = GeorgeAPI.post("/api/agent/chat", chat_data)
        
        if "error" not in response:
            # Display George's response
            st.session_state.chat_history.append({
                "role": "george",
                "content": response.get("response", "No response"),
                "memory_context": response.get("memory_context", []),
                "cognitive_state": response.get("cognitive_state", {}),
                "memory_events": response.get("memory_events", []),
                "rationale": response.get("rationale"),
                "timestamp": datetime.now()
            })
        else:
            st.error(f"Chat error: {response['error']}")
    
    # Display chat history
    for message in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message" style="border-left-color: #28a745;">
                    <strong>👤 You ({message['timestamp'].strftime('%H:%M:%S')}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # George's response with cognitive details
                cognitive_state = message.get("cognitive_state", {})
                memory_events = message.get("memory_events", [])
                
                st.markdown(f"""
                <div class="chat-message" style="border-left-color: #1f77b4;">
                    <strong>🧠 George ({message['timestamp'].strftime('%H:%M:%S')}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show cognitive details in expandable section
                if cognitive_state or memory_events:
                    with st.expander("🔍 Cognitive Details"):
                        if cognitive_state:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Cognitive Load", f"{cognitive_state.get('cognitive_load', 0):.1%}")
                            with col2:
                                st.metric("Attention Focus", cognitive_state.get('attention_focus', 0))
                            with col3:
                                st.metric("Memory Ops", cognitive_state.get('memory_operations', 0))
                        
                        if memory_events:
                            st.write("**Memory Events:**")
                            for event in memory_events:
                                st.write(f"• {event}")
                        
                        if message.get("rationale"):
                            st.write("**Reasoning:**")
                            st.write(message["rationale"])

def render_memory_dashboard():
    """Render the memory management dashboard"""
    st.subheader("🧠 Memory Management Dashboard")
    
    # Memory system selector
    memory_systems = ["STM", "LTM", "Episodic", "Semantic", "Procedural", "Prospective"]
    selected_system = st.selectbox("Select Memory System", memory_systems)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Memory browser
        st.write(f"**{selected_system} Memory Browser**")
        
        if selected_system in ["STM", "LTM"]:
            # Get memories from agent API
            memories_data = GeorgeAPI.get(f"/api/agent/memory/list/{selected_system.lower()}")
            
            if "error" not in memories_data and "memories" in memories_data:
                memories = memories_data["memories"]
                if memories:
                    for i, memory in enumerate(memories[:10]):  # Show first 10
                        # Handle different memory formats
                        if isinstance(memory, dict):
                            content = memory.get('content', 'N/A')
                            memory_id = memory.get('id', f'memory_{i}')
                        else:
                            content = str(memory)
                            memory_id = f'memory_{i}'
                        
                        # Truncate content for display
                        display_content = content[:50] + "..." if len(content) > 50 else content
                        
                        with st.expander(f"Memory {i+1}: {display_content}"):
                            st.json(memory)
                else:
                    st.info(f"No {selected_system} memories found")
            else:
                st.warning(f"Unable to load {selected_system} memories")
        else:
            # Placeholder for other memory systems
            st.info(f"{selected_system} memory browser coming in Phase 2")
    
    with col2:
        # Memory health metrics
        st.write("**Memory Health**")
        
        memory_status = st.session_state.agent_status.get("memory_status", {})
        
        if memory_status:
            # STM metrics
            stm_data = memory_status.get("stm", {})
            stm_count = stm_data.get("vector_db_count", 0)
            stm_utilization = stm_data.get("capacity_utilization", 0.0)
            
            st.metric("STM Count", stm_count)
            st.metric("STM Utilization", f"{stm_utilization:.1%}")
            
            # LTM metrics
            ltm_data = memory_status.get("ltm", {})
            ltm_count = ltm_data.get("memory_count", 0)
            
            st.metric("LTM Count", ltm_count)
            
            # Memory consolidation button
            if st.button("🌙 Trigger Memory Consolidation"):
                with st.spinner("Consolidating memories..."):
                    result = GeorgeAPI.post("/agent/memory/consolidate", {})
                
                if "error" not in result:
                    st.success("Memory consolidation completed!")
                    st.write("**Consolidation Events:**")
                    for event in result.get("consolidation_events", []):
                        st.write(f"• {event}")
                else:
                    st.error(f"Consolidation failed: {result['error']}")
    
    # Memory search
    st.write("**Memory Search**")
    search_query = st.text_input("Search across all memory systems:", placeholder="Enter search terms...")
    
    if search_query and st.button("🔍 Search Memories"):
        search_data = {"query": search_query}
        with st.spinner("Searching memories..."):
            results = GeorgeAPI.post("/agent/memory/search", search_data)
        
        if "error" not in results:
            memory_context = results.get("memory_context", [])
            if memory_context:
                st.write(f"**Found {len(memory_context)} relevant memories:**")
                for context in memory_context:
                    with st.expander(f"Memory: {context.get('content', 'N/A')[:100]}..."):
                        st.json(context)
            else:
                st.info("No relevant memories found")
        else:
            st.error(f"Search failed: {results['error']}")

def render_attention_monitor():
    """Render the attention monitoring interface"""
    st.subheader("🎯 Attention & Cognitive Load Monitor")
    
    col1, col2, col3 = st.columns(3)
    
    # Get current attention status
    agent_status = st.session_state.agent_status
    attention_status = agent_status.get("attention_status", {})
    
    with col1:
        # Cognitive load gauge
        cognitive_load = attention_status.get("cognitive_load", 0.0)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cognitive_load * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cognitive Load %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Fatigue level
        fatigue_level = attention_status.get("fatigue_level", 0.0)
        
        fig_fatigue = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fatigue_level * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fatigue Level %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        fig_fatigue.update_layout(height=300)
        st.plotly_chart(fig_fatigue, use_container_width=True)
    
    with col3:
        # Cognitive break controls
        st.write("**Cognitive Break Controls**")
        
        break_duration = st.slider("Break Duration (minutes)", 0.5, 5.0, 1.0, 0.5)
        
        if st.button("🧘 Take Cognitive Break", type="primary"):
            break_data = {"duration_minutes": break_duration}
            
            with st.spinner(f"Taking {break_duration} minute cognitive break..."):
                result = GeorgeAPI.post("/agent/cognitive_break", break_data)
            
            if "error" not in result:
                st.success("Cognitive break completed!")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Load Reduction", f"{result.get('cognitive_load_reduction', 0):.1%}")
                with col_b:
                    st.metric("New Load", f"{result.get('new_cognitive_load', 0):.1%}")
            else:
                st.error(f"Break failed: {result['error']}")
        
        # Attention recommendations
        if cognitive_load > 0.8:
            st.warning("⚠️ High cognitive load detected!")
            st.write("Recommendations:")
            st.write("• Take a cognitive break")
            st.write("• Reduce task complexity")
            st.write("• Focus on single task")
        elif fatigue_level > 0.7:
            st.warning("😴 High fatigue detected!")
            st.write("Recommendations:")
            st.write("• Take an extended break")
            st.write("• Switch to easier tasks")
            st.write("• Consider rest period")

def render_executive_dashboard():
    """Render the basic executive dashboard"""
    st.subheader("🎯 Executive Dashboard")
    
    # Get executive status
    exec_status = GeorgeAPI.get("/api/executive/status")
    
    if "error" not in exec_status:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Goals section
            st.write("**Goal Management**")
            
            goals = exec_status.get("goals", [])
            if goals and isinstance(goals, list):
                for goal in goals[:5]:  # Show first 5 goals
                    goal_status = goal.get("status", "unknown")
                    status_color = {
                        "active": "🟢",
                        "completed": "✅", 
                        "paused": "⏸️",
                        "cancelled": "❌"
                    }.get(goal_status, "⚪")
                    
                    st.write(f"{status_color} **{goal.get('title', 'Untitled Goal')}**")
                    st.write(f"Priority: {goal.get('priority', 'medium')} | Progress: {goal.get('progress', 0):.0%}")
                    st.write(f"Description: {goal.get('description', 'No description')[:100]}...")
                    st.divider()
            else:
                st.info("No active goals found")
            
            # Quick goal creation
            with st.expander("➕ Create New Goal"):
                goal_title = st.text_input("Goal Title")
                goal_desc = st.text_area("Goal Description")
                goal_priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
                
                if st.button("Create Goal") and goal_title:
                    goal_data = {
                        "title": goal_title,
                        "description": goal_desc,
                        "priority": goal_priority
                    }
                    
                    result = GeorgeAPI.post("/api/executive/goals", goal_data)
                    if "error" not in result:
                        st.success("Goal created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Goal creation failed: {result['error']}")
        
        with col2:
            # Performance metrics
            st.write("**Executive Performance**")
            
            performance = exec_status.get("performance", {})
            if performance:
                st.metric("Goal Completion Rate", f"{performance.get('goal_completion_rate', 0):.1%}")
                st.metric("Task Efficiency", f"{performance.get('task_efficiency', 0):.1%}")
                st.metric("Decision Accuracy", f"{performance.get('decision_accuracy', 0):.1%}")
                st.metric("Resource Utilization", f"{performance.get('resource_utilization', 0):.1%}")
            
            # Resource allocation
            st.write("**Resource Allocation**")
            resources = exec_status.get("resources", {})
            if resources:
                for resource, allocation in resources.items():
                    st.progress(allocation, text=f"{resource.title()}: {allocation:.1%}")
    else:
        st.error("Unable to load executive data")
        st.info("Executive dashboard coming in Phase 1 completion")

def render_procedural_memory():
    """Render procedural memory interface"""
    st.subheader("📋 Procedural Memory - Skills & Procedures")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Current Procedures")
        
        # Get procedural memories
        procedures_data = GeorgeAPI.get("/api/agent/memory/procedural/list")
        
        if "error" not in procedures_data:
            procedures = procedures_data.get("procedures", [])
            
            if procedures:
                for i, proc in enumerate(procedures):
                    with st.expander(f"🔧 {proc.get('description', 'Unnamed Procedure')}"):
                        st.write(f"**Steps:** {len(proc.get('steps', []))}")
                        for j, step in enumerate(proc.get('steps', []), 1):
                            st.write(f"{j}. {step}")
                        st.write(f"**Tags:** {', '.join(proc.get('tags', []))}")
                        st.write(f"**Usage Count:** {proc.get('usage_count', 0)}")
            else:
                st.info("No procedures stored yet. Create your first procedure!")
        else:
            st.warning("Unable to load procedural memories")
    
    with col2:
        st.markdown("### Create New Procedure")
        
        with st.form("create_procedure"):
            proc_description = st.text_input("Procedure Name", placeholder="e.g., Morning Coffee Routine")
            
            # Dynamic steps input
            num_steps = st.number_input("Number of Steps", min_value=1, max_value=20, value=3)
            steps = []
            for i in range(num_steps):
                step = st.text_input(f"Step {i+1}", placeholder=f"Enter step {i+1}")
                if step:
                    steps.append(step)
            
            proc_tags = st.text_input("Tags (comma-separated)", placeholder="routine, daily, morning")
            memory_type = st.selectbox("Store in", ["stm", "ltm"])
            importance = st.slider("Importance", 0.0, 1.0, 0.5)
            
            if st.form_submit_button("Create Procedure", type="primary"):
                if proc_description and steps:
                    tags_list = [tag.strip() for tag in proc_tags.split(",") if tag.strip()]
                    
                    response = GeorgeAPI.post("/api/agent/memory/procedural/create", {
                        "description": proc_description,
                        "steps": steps,
                        "tags": tags_list,
                        "memory_type": memory_type,
                        "importance": importance
                    })
                    
                    if "error" not in response:
                        st.success(f"Procedure '{proc_description}' created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to create procedure: {response.get('error')}")
                else:
                    st.error("Please provide a description and at least one step")

def render_prospective_memory():
    """Render prospective memory interface"""
    st.subheader("📅 Prospective Memory - Future Intentions & Reminders")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Active Reminders")
        
        # Get prospective memories
        reminders_data = GeorgeAPI.get("/api/agent/memory/prospective/list")
        
        if "error" not in reminders_data:
            reminders = reminders_data.get("reminders", [])
            
            if reminders:
                for reminder in reminders:
                    with st.expander(f"⏰ {reminder.get('description', 'Unnamed Reminder')}"):
                        st.write(f"**Created:** {reminder.get('created_at', 'Unknown')}")
                        if reminder.get('trigger_time'):
                            st.write(f"**Trigger Time:** {reminder.get('trigger_time')}")
                        st.write(f"**Tags:** {', '.join(reminder.get('tags', []))}")
                        st.write(f"**Status:** {'Completed' if reminder.get('completed') else 'Active'}")
            else:
                st.info("No active reminders. Create your first reminder!")
        else:
            st.warning("Unable to load prospective memories")
    
    with col2:
        st.markdown("### Create New Reminder")
        
        with st.form("create_reminder"):
            reminder_desc = st.text_area("Reminder Description", placeholder="e.g., Call dentist for appointment")
            
            # Date and time selection
            reminder_date = st.date_input("Reminder Date")
            reminder_time = st.time_input("Reminder Time")
            
            # Combine date and time
            if reminder_date and reminder_time:
                from datetime import datetime
                trigger_datetime = datetime.combine(reminder_date, reminder_time)
                trigger_time_str = trigger_datetime.isoformat()
            else:
                trigger_time_str = None
            
            reminder_tags = st.text_input("Tags (comma-separated)", placeholder="important, health, appointment")
            
            if st.form_submit_button("Create Reminder", type="primary"):
                if reminder_desc:
                    tags_list = [tag.strip() for tag in reminder_tags.split(",") if tag.strip()]
                    
                    response = GeorgeAPI.post("/api/agent/memory/prospective/create", {
                        "description": reminder_desc,
                        "trigger_time": trigger_time_str,
                        "tags": tags_list
                    })
                    
                    if "error" not in response:
                        st.success(f"Reminder created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to create reminder: {response.get('error')}")
                else:
                    st.error("Please provide a description")

def render_neural_analytics():
    """Render neural activity monitoring and performance analytics"""
    st.subheader("🔬 Neural Activity & Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Neural Network Status")
        
        neural_data = GeorgeAPI.get("/api/neural/status")
        
        if "error" not in neural_data:
            # DPAD Network Status
            st.markdown("#### DPAD Network (Attention Enhancement)")
            dpad_available = neural_data.get("dpad_available", False)
            st.write(f"Status: {'🟢 Active' if dpad_available else '🔴 Inactive'}")
            
            if dpad_available:
                activity = neural_data.get("neural_activity", {})
                attention_boost = activity.get("attention_enhancement", 0.0)
                
                # Create gauge for attention enhancement
                fig_attention = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = attention_boost * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Attention Boost %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_attention.update_layout(height=300)
                st.plotly_chart(fig_attention, use_container_width=True)
            
            # LSHN Network Status
            st.markdown("#### LSHN Network (Memory Consolidation)")
            lshn_available = neural_data.get("lshn_available", False)
            st.write(f"Status: {'🟢 Active' if lshn_available else '🔴 Inactive'}")
            
            if lshn_available:
                consolidation = neural_data.get("neural_activity", {}).get("consolidation_activity", 0.0)
                st.progress(consolidation, text=f"Consolidation Activity: {consolidation:.1%}")
        else:
            st.warning("Unable to load neural status")
    
    with col2:
        st.markdown("### Performance Analytics")
        
        analytics_data = GeorgeAPI.get("/api/analytics/performance")
        
        if "error" not in analytics_data:
            efficiency = analytics_data.get("cognitive_efficiency", {})
            usage = analytics_data.get("usage_statistics", {})
            trends = analytics_data.get("trends", {})
            
            # Cognitive Efficiency Metrics
            st.markdown("#### Cognitive Efficiency")
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric("Overall Efficiency", f"{efficiency.get('overall', 0.0):.1%}")
                st.metric("Memory Efficiency", f"{efficiency.get('memory', 0.0):.1%}")
            
            with col2b:
                st.metric("Attention Efficiency", f"{efficiency.get('attention', 0.0):.1%}")
                st.metric("Processing Efficiency", f"{efficiency.get('processing', 0.0):.1%}")
            
            # Usage Statistics
            st.markdown("#### Usage Statistics")
            session_duration = usage.get("session_duration", 0)
            hours = session_duration // 3600
            minutes = (session_duration % 3600) // 60
            
            st.write(f"**Session Duration:** {hours}h {minutes}m")
            st.write(f"**Interactions:** {usage.get('interactions', 0)}")
            st.write(f"**Error Rate:** {usage.get('error_rate', 0.0):.1%}")
            
            # Trends
            st.markdown("#### Performance Trends")
            st.write(f"**Cognitive Load:** {trends.get('cognitive_load_trend', 'Unknown')}")
            st.write(f"**Memory Usage:** {trends.get('memory_usage_trend', 'Unknown')}")
            st.write(f"**Performance:** {trends.get('performance_trend', 'Unknown')}")
            
            # Performance Chart
            if st.button("🔄 Refresh Analytics"):
                st.rerun()
        else:
            st.warning("Unable to load performance analytics")

def main():
    """Main application entry point"""
    initialize_session_state()
    wait_for_agent_initialization()  # Add initialization check
    render_header()
    render_cognitive_status_bar()
    
    # Main interface tabs - Phase 1 + Phase 2 Features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💬 Enhanced Chat", 
        "🧠 Memory Dashboard", 
        "🎯 Attention Monitor", 
        "🎯 Executive Dashboard",
        "📋 Procedural Memory",
        "📅 Prospective Memory", 
        "🔬 Neural & Analytics"
    ])
    
    with tab1:
        render_enhanced_chat()
    
    with tab2:
        render_memory_dashboard()
    
    with tab3:
        render_attention_monitor()
    
    with tab4:
        render_executive_dashboard()
    
    with tab5:
        render_procedural_memory()
    
    with tab6:
        render_prospective_memory()
    
    with tab7:
        render_neural_analytics()
    
    # Footer
    st.divider()
    st.markdown("*George - Human-AI Cognitive Architecture | Phase 1 + Phase 2 Features*")

if __name__ == "__main__":
    main()
