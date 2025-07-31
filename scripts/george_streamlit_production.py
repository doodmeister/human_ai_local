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
    def get(endpoint: str) -> Dict[str, Any]:
        """GET request to George API"""
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def post(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request to George API"""
        try:
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return {"error": str(e)}

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
    status_data = GeorgeAPI.get("/api/agent/status")
    
    if "error" not in status_data:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            cognitive_mode = status_data.get("cognitive_mode", "UNKNOWN")
            mode_color = "status-healthy" if cognitive_mode == "FOCUSED" else "status-warning"
            st.markdown(f'<div class="cognitive-state">Mode: {cognitive_mode}</div>', unsafe_allow_html=True)
        
        with col2:
            cognitive_load = status_data.get("attention_status", {}).get("cognitive_load", 0.0)
            load_color = "status-healthy" if cognitive_load < 0.7 else "status-warning" if cognitive_load < 0.9 else "status-critical"
            st.metric("Cognitive Load", f"{cognitive_load:.1%}", delta=None)
        
        with col3:
            stm_count = status_data.get("memory_status", {}).get("stm", {}).get("vector_db_count", 0)
            st.metric("STM Memories", stm_count)
        
        with col4:
            ltm_count = status_data.get("memory_status", {}).get("ltm", {}).get("memory_count", 0)
            st.metric("LTM Memories", ltm_count)
        
        with col5:
            active_processes = status_data.get("active_processes", 0)
            st.metric("Active Processes", active_processes)
        
        st.session_state.agent_status = status_data
    else:
        st.error("Unable to connect to George's cognitive backend")

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

def main():
    """Main application entry point"""
    initialize_session_state()
    render_header()
    render_cognitive_status_bar()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Enhanced Chat", "🧠 Memory Dashboard", "🎯 Attention Monitor", "🎯 Executive Dashboard"])
    
    with tab1:
        render_enhanced_chat()
    
    with tab2:
        render_memory_dashboard()
    
    with tab3:
        render_attention_monitor()
    
    with tab4:
        render_executive_dashboard()
    
    # Footer
    st.divider()
    st.markdown("*George - Human-AI Cognitive Architecture | Phase 1 Production Interface*")

if __name__ == "__main__":
    main()
