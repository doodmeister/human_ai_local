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
import plotly.graph_objects as go
from datetime import datetime
import time
from typing import Dict, Any

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
            if 'api_errors' not in st.session_state:
                st.session_state.api_errors = []
            st.session_state.api_errors.append({"method":"GET","endpoint":endpoint,"error":"timeout","ts": datetime.now().isoformat()})
            st.session_state.api_errors = st.session_state.api_errors[-50:]
            return {"error": f"Timeout after {timeout}s", "status": "timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if 'api_errors' not in st.session_state:
                st.session_state.api_errors = []
            st.session_state.api_errors.append({"method":"GET","endpoint":endpoint,"error":str(e),"ts": datetime.now().isoformat()})
            st.session_state.api_errors = st.session_state.api_errors[-50:]
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
            if 'api_errors' not in st.session_state:
                st.session_state.api_errors = []
            st.session_state.api_errors.append({"method":"POST","endpoint":endpoint,"error":"timeout","payload_keys": list(data.keys()),"ts": datetime.now().isoformat()})
            st.session_state.api_errors = st.session_state.api_errors[-50:]
            return {"error": f"Timeout after {timeout}s", "status": "timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if 'api_errors' not in st.session_state:
                st.session_state.api_errors = []
            st.session_state.api_errors.append({"method":"POST","endpoint":endpoint,"error":str(e),"payload_keys": list(data.keys()),"ts": datetime.now().isoformat()})
            st.session_state.api_errors = st.session_state.api_errors[-50:]
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
    # Aggregated captured memories cache keyed by composite key
    if 'captured_memory_cache' not in st.session_state:
        st.session_state.captured_memory_cache = {}
    if 'captured_memory_list' not in st.session_state:
        st.session_state.captured_memory_list = []  # ordered list for display

def aggregate_captured_memories(captured_list):
    """Aggregate captured memories into session_state caches.

    Ensures frequency/last_seen are updated and contradictions merged.
    """
    if not captured_list:
        return
    cache = st.session_state.captured_memory_cache
    ordered = st.session_state.captured_memory_list
    for cm in captured_list:
        key = f"{cm.get('memory_type')}|{cm.get('subject')}|{cm.get('predicate')}|{cm.get('object')}"
        existing = cache.get(key)
        if existing:
            existing['frequency'] = max(existing.get('frequency', 0), cm.get('frequency', 0))
            existing['last_seen_ts'] = cm.get('last_seen_ts', existing.get('last_seen_ts'))
            if cm.get('contradiction'):
                existing['contradiction'] = True
                existing.setdefault('contradicted_prior', [])
                for v in (cm.get('contradicted_prior', []) or []):
                    if v not in existing['contradicted_prior']:
                        existing['contradicted_prior'].append(v)
        else:
            cache[key] = dict(cm)
            ordered.append(cache[key])

def render_captured_memories_sidebar():
    """Render sidebar section listing captured memories with filters/search."""
    with st.sidebar:
        st.markdown("### 🧩 Captured Memories")
        total = len(st.session_state.captured_memory_list)
        if total == 0:
            st.caption("No captured memories yet.")
            return
        categories = sorted({cm.get('memory_type') for cm in st.session_state.captured_memory_list})
        selected_categories = st.multiselect("Categories", categories, default=categories, key='mem_cat_select')
        search = st.text_input("Search", placeholder="substring filter", key='mem_search')
        show_contradictions_only = st.checkbox("Contradictions only", value=False, key='mem_contrad_only')
        filtered = []
        for cm in st.session_state.captured_memory_list:
            if cm.get('memory_type') not in selected_categories:
                continue
            if show_contradictions_only and not cm.get('contradiction'):
                continue
            haystack = (cm.get('content','') + ' ' + str(cm.get('object',''))).lower()
            if search and search.lower() not in haystack:
                continue
            filtered.append(cm)
        contrad_count = sum(1 for cm in st.session_state.captured_memory_list if cm.get('contradiction'))
        st.metric("Stored", total)
        st.metric("Contradictions", contrad_count)
        for cm in reversed(filtered[-15:]):
            freq = cm.get('frequency', 1)
            tag = cm.get('memory_type')
            line = f"[{tag}] {cm.get('content','')[:80]} (f={freq})"
            if cm.get('contradiction'):
                line += " ⚠️"
            st.write(line)
        if len(filtered) == 0 and total > 0:
            st.caption("No matches for current filters.")

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
    # Initialize session scaffolding if missing
    if 'sessions' not in st.session_state:
        st.session_state.sessions = [{'id': 'default', 'label': 'Default Session'}]
    if 'active_session_id' not in st.session_state:
        st.session_state.active_session_id = 'default'
    if 'per_session_history' not in st.session_state:
        st.session_state.per_session_history = {}
    if 'default' not in st.session_state.per_session_history:
        st.session_state.per_session_history['default'] = []
    
    # Session management controls
    with st.expander("🗂 Session Management"):
        existing_labels = [s['label'] for s in st.session_state.sessions]
        active_idx = 0
        for i, s in enumerate(st.session_state.sessions):
            if s['id'] == st.session_state.active_session_id:
                active_idx = i
                break
        chosen_label = st.selectbox("Active Session", existing_labels, index=active_idx, key='active_session_select')
        # Map back to id
        for s in st.session_state.sessions:
            if s['label'] == chosen_label:
                st.session_state.active_session_id = s['id']
                break
        new_label = st.text_input("New Session Label", key='new_session_label_input')
        if st.button("➕ Create Session", key='create_session_btn') and new_label:
            import uuid as _uuid
            new_id = _uuid.uuid4().hex[:12]
            st.session_state.sessions.append({'id': new_id, 'label': new_label})
            st.session_state.per_session_history[new_id] = []
            st.session_state.active_session_id = new_id
            st.success(f"Created session '{new_label}'")
            st.rerun()
        if st.button("🧹 Clear Active Session Chat", key='clear_session_btn'):
            sid = st.session_state.active_session_id
            st.session_state.per_session_history[sid] = []
            if sid == 'default':
                # also clear global fallback
                st.session_state.chat_history = []
            st.success("Cleared chat for active session")

    # Chat input
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            user_input = st.text_input("Chat with George:", placeholder="Ask me anything or give me a task...", key='chat_input_box')
        with col2:
            include_reflection = st.checkbox("Reflection", value=False, key='flag_reflection')
            include_trace = st.checkbox("Trace", value=False, key='flag_trace')
        with col3:
            include_attention = st.checkbox("Attention", value=True, key='flag_attention')
            include_memory = st.checkbox("Memory", value=True, key='flag_memory')
    
    # Context preview controls
    with st.expander("🔎 Context Preview (before send)"):
        preview_msg = st.text_input("Preview message", key='preview_msg_input', placeholder="Type a message to preview context retrieval...")
        if st.button("Run Preview", key='run_preview_btn') and preview_msg:
            with st.spinner("Building context preview..."):
                preview = GeorgeAPI.get(f"/agent/chat/preview?message={preview_msg}")
            if 'items' in preview:
                st.write(f"Items ({preview.get('item_count')}):")
                for it in preview['items']:
                    st.write(f"• [{it.get('source')}] {it.get('content')}")
            else:
                st.warning("No preview data returned")

    if user_input and st.button("Send", type="primary", key='send_chat_btn'):
        # Display user message
        sid = st.session_state.active_session_id
        user_msg = {"role": "user", "content": user_input, "timestamp": datetime.now()}
        st.session_state.chat_history.append(user_msg)
        st.session_state.per_session_history[sid].append(user_msg)
        # Build payload for new /agent/chat endpoint
        flags = {
            "reflection": include_reflection,
            "include_trace": include_trace,
            "include_attention": include_attention,
            "include_memory": include_memory,
        }
        # Remove false flags to cut payload noise
        flags = {k: v for k, v in flags.items() if v}
        chat_data = {"message": user_input, "flags": flags or None, "session_id": sid}
        with st.spinner("George is thinking..."):
            response = GeorgeAPI.post("/agent/chat", chat_data)
        if "error" not in response:
            asst_msg = {
                "role": "george",
                "content": response.get("response", "No response"),
                "memory_context": response.get("memory_context", []),
                "cognitive_state": response.get("cognitive_state", {}),
                "memory_events": response.get("memory_events", []),
                "captured_memories": response.get("captured_memories", []),
                "rationale": response.get("rationale"),
                "timestamp": datetime.now()
            }
            st.session_state.chat_history.append(asst_msg)
            st.session_state.per_session_history[sid].append(asst_msg)
            # Aggregate captured memories via helper
            aggregate_captured_memories(response.get("captured_memories", []))
        else:
            st.error(f"Chat error: {response['error']}")
    
    # Display chat history
    sid = st.session_state.active_session_id
    session_history = st.session_state.per_session_history.get(sid, [])
    for message in reversed(session_history[-10:]):  # Show last 10 for active session
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
                
                # Fixed timestamp formatting: balanced braces/parentheses
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

                        captured = message.get("captured_memories") or []
                        if captured:
                            st.write("**Captured Memories (current turn):**")
                            for cm in captured:
                                subj = cm.get('subject') or ''
                                pred = cm.get('predicate') or ''
                                obj = cm.get('object') or ''
                                mtype = cm.get('memory_type')
                                freq = cm.get('frequency')
                                st.write(f"• [{mtype}] {subj} {pred} {obj} (freq={freq})")
                        
                        if message.get("rationale"):
                            st.write("**Reasoning:**")
                            st.write(message["rationale"])

def render_memory_dashboard():
    """Render the memory management dashboard"""
    st.subheader("🧠 Memory Management Dashboard")
    
    # Memory system selector
    memory_systems = ["STM", "LTM", "Episodic", "Semantic", "Procedural", "Prospective", "Semantic Facts"]
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
    elif selected_system == "Semantic Facts":
        st.write("**Semantic Facts**")
        # Counters (pull from chat metrics endpoint once per render; simple cache)
        if '_semantic_metrics_cache' not in st.session_state:
            try:
                _m = GeorgeAPI.get('/agent/chat/metrics')
                st.session_state._semantic_metrics_cache = _m
            except Exception:
                st.session_state._semantic_metrics_cache = {}
        metrics_blob = st.session_state.get('_semantic_metrics_cache', {}) or {}
        cap_counters = metrics_blob.get('captured_memory_counters') or metrics_blob.get('captured_memory', {})
        sem_promos = None
        contradictions = None
        if isinstance(cap_counters, dict):
            # Try multiple possible key names for robustness
            sem_promos = cap_counters.get('semantic_promotions_total') or cap_counters.get('captured_memory_semantic_promotions_total')
            contradictions = cap_counters.get('contradictions_total') or cap_counters.get('captured_memory_contradictions_total')
        promo_col, contrad_col, refresh_col = st.columns([1,1,1])
        with promo_col:
            st.metric("Semantic Promotions", sem_promos if sem_promos is not None else 0)
        with contrad_col:
            st.metric("Contradictions", contradictions if contradictions is not None else 0)
        with refresh_col:
            if st.button("↻ Refresh Counters", key='sem_metrics_refresh'):
                try:
                    st.session_state._semantic_metrics_cache = GeorgeAPI.get('/agent/chat/metrics')
                except Exception:
                    st.warning("Failed to refresh counters")
        # Search filters
        with st.expander("Search Facts"):
            subj = st.text_input("Subject", key="sem_search_subj")
            pred = st.text_input("Predicate", key="sem_search_pred")
            obj = st.text_input("Object Contains", key="sem_search_obj")
            if st.button("Search Facts", key="sem_search_btn"):
                params = {}
                if subj: params['subject'] = subj
                if pred: params['predicate'] = pred
                if obj: params['object_val'] = obj
                results = GeorgeAPI.post("/api/semantic/fact/search", params)
                if 'results' in results:
                    st.write(f"Found {len(results['results'])} facts")
                    for r in results['results'][:50]:
                        with st.expander(f"{r.get('subject')} {r.get('predicate')} {r.get('object_val')}"):
                            st.json(r)
                else:
                    st.warning("No results or error")
        if st.button("List All Facts", key="sem_list_all"):
            all_res = GeorgeAPI.post("/api/semantic/fact/search", {})
            facts = all_res.get('results', [])
            st.write(f"Total facts: {len(facts)}")
            for r in facts[:100]:
                st.write(f"• {r.get('subject')} {r.get('predicate')} {r.get('object_val')}")
        with st.expander("Add New Fact"):
            new_subj = st.text_input("New Subject", key="sem_new_subj")
            new_pred = st.text_input("New Predicate", key="sem_new_pred")
            new_obj = st.text_area("Object Value", key="sem_new_obj")
            if st.button("Store Fact", key="sem_store_btn") and new_subj and new_pred and new_obj:
                payload = {"subject": new_subj, "predicate": new_pred, "object_val": new_obj}
                res = GeorgeAPI.post("/api/semantic/fact/store", payload)
                if 'fact_id' in res:
                    st.success(f"Stored fact id: {res['fact_id']}")
                    st.rerun()
                else:
                    st.error("Failed to store fact")
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
    
        # Export captured memories (available regardless of selected system)
        st.write("**Captured Memories Export**")
        captured_list = st.session_state.get('captured_memory_list', [])
        if captured_list:
            import json as _json
            export_json = _json.dumps(captured_list, indent=2, default=str)
            st.download_button(
                label="⬇️ Download Captured Memories JSON",
                data=export_json.encode('utf-8'),
                file_name="captured_memories.json",
                mime="application/json",
                key='download_captured_memories'
            )
            st.caption(f"{len(captured_list)} captured memory items available")
        else:
            st.caption("No captured memories yet this session")

    # Memory search (legacy simple) + Unified Search
    st.write("**Memory Search**")
    with st.expander("Unified Search (STM, LTM, Semantic)"):
        unified_query = st.text_input("Query", key='unified_mem_query')
        col_u1, col_u2, col_u3 = st.columns([1,1,2])
        with col_u1:
            inc_stm = st.checkbox("STM", value=True, key='unified_inc_stm')
        with col_u2:
            inc_ltm = st.checkbox("LTM", value=True, key='unified_inc_ltm')
        with col_u3:
            inc_sem = st.checkbox("Semantic Facts", value=True, key='unified_inc_sem')
        limit = st.number_input("Per-System Limit", min_value=5, max_value=100, value=20, step=5, key='unified_limit')
        if unified_query and st.button("Run Unified Search", key='unified_search_btn'):
            unified_results = {}
            with st.spinner("Running unified search..."):
                try:
                    if inc_stm:
                        stm_res = GeorgeAPI.post("/agent/memory/search", {"query": unified_query, "system": "stm", "limit": limit})
                        unified_results['STM'] = stm_res.get('memory_context', stm_res.get('results', [])) if isinstance(stm_res, dict) else []
                    if inc_ltm:
                        ltm_res = GeorgeAPI.post("/agent/memory/search", {"query": unified_query, "system": "ltm", "limit": limit})
                        unified_results['LTM'] = ltm_res.get('memory_context', ltm_res.get('results', [])) if isinstance(ltm_res, dict) else []
                    if inc_sem:
                        # Semantic facts search endpoint reuses subject/predicate/object filters; we approximate with object_val contains
                        sem_res = GeorgeAPI.post("/api/semantic/fact/search", {"object_val": unified_query})
                        unified_results['Semantic Facts'] = sem_res.get('results', []) if isinstance(sem_res, dict) else []
                except Exception as e:
                    st.error(f"Unified search error: {e}")
            # Display grouped
            for sys_name, items in unified_results.items():
                st.markdown(f"**{sys_name} Results ({len(items)})**")
                if not items:
                    st.write("_None_")
                    continue
                # Show limited details
                for i, itm in enumerate(items[:limit]):
                    if isinstance(itm, dict):
                        snippet = str(itm)[:140] + ("..." if len(str(itm))>140 else "")
                        with st.expander(f"{sys_name} #{i+1}: {snippet}"):
                            st.json(itm)
                    else:
                        st.write(f"• {itm}")
    # Legacy simple search (kept for backward compatibility)
    search_query = st.text_input("Simple search (legacy across systems):", placeholder="Enter search terms...", key='legacy_mem_search')
    if search_query and st.button("🔍 Search Memories", key='legacy_mem_search_btn'):
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
                        st.success("Reminder created successfully!")
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

    # Sidebar captured memories panel
    render_captured_memories_sidebar()
    
    # Main interface tabs - Phase 1 + Phase 2 Features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "💬 Enhanced Chat", 
        "🧠 Memory Dashboard", 
        "🎯 Attention Monitor", 
        "🎯 Executive Dashboard",
        "📋 Procedural Memory",
        "📅 Prospective Memory", 
        "🔬 Neural & Analytics",
        "📊 Chat Metrics"
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

    with tab8:
        st.subheader("📊 Chat Metrics & Metacog")
        # Auto-refresh interval
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:
            refresh_seconds = st.number_input("Refresh (s)", min_value=5, max_value=120, value=15, step=5, key='chat_metrics_refresh')
        with col_b:
            do_refresh = st.checkbox("Auto Refresh", value=True, key='chat_metrics_auto')
        with col_c:
            if st.button("Manual Refresh", key='chat_metrics_manual'):
                st.session_state['_force_chat_metrics_refresh'] = True
        # Simple timer-based refresh using empty placeholder
        placeholder = st.empty()
        if 'chat_metrics_history' not in st.session_state:
            st.session_state.chat_metrics_history = []  # store (ts, p95, ema, tps)
        if 'metacog_history_cache' not in st.session_state:
            st.session_state.metacog_history_cache = []
        import time as _t
        # Decide if we fetch
        last_fetch = st.session_state.get('_chat_metrics_last_fetch')
        need = False
        now = _t.time()
        if st.session_state.get('_force_chat_metrics_refresh'):
            need = True
            st.session_state['_force_chat_metrics_refresh'] = False
        elif do_refresh and (last_fetch is None or now - last_fetch >= refresh_seconds):
            need = True
        if need:
            metrics_data = GeorgeAPI.get('/agent/chat/metrics')
            perf_data = GeorgeAPI.get('/agent/chat/performance')
            metacog_data = GeorgeAPI.get('/agent/chat/metacog/status')
            cons_data = GeorgeAPI.get('/agent/chat/consolidation/status')
            st.session_state['_chat_metrics_last_fetch'] = now
            # Record history point
            try:
                p95 = perf_data.get('latency_p95_ms') if isinstance(perf_data, dict) else None
                ema = perf_data.get('ema_turn_latency_ms') if isinstance(perf_data, dict) else None
                tps = perf_data.get('chat_turns_per_sec') if isinstance(perf_data, dict) else None
                # Derive attention/fatigue signals if present
                att_focus = None
                fatigue = None
                # Search possible locations for attention metrics
                if isinstance(metacog_data, dict):
                    snap = metacog_data.get('snapshot') or {}
                    if isinstance(snap, dict):
                        att_focus = snap.get('attention_focus') or snap.get('attention', {}).get('focus')
                        fatigue = snap.get('fatigue') or snap.get('attention', {}).get('fatigue')
                if att_focus is None and isinstance(metrics_data, dict):
                    att_focus = metrics_data.get('attention_focus')
                if fatigue is None and isinstance(metrics_data, dict):
                    fatigue = metrics_data.get('fatigue')
                st.session_state.chat_metrics_history.append({'ts': now, 'p95': p95, 'ema': ema, 'tps': tps, 'attention_focus': att_focus, 'fatigue': fatigue})
                st.session_state.chat_metrics_history = st.session_state.chat_metrics_history[-200:]
                if metacog_data.get('history_tail'):
                    st.session_state.metacog_history_cache = metacog_data['history_tail']
            except Exception:
                pass
            st.session_state['_chat_metrics_current'] = {
                'metrics': metrics_data,
                'performance': perf_data,
                'metacog': metacog_data,
                'consolidation': cons_data
            }
        current = st.session_state.get('_chat_metrics_current', {})
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            perf = current.get('performance', {}) or {}
            with col1:
                st.metric('Latency p95 (ms)', f"{perf.get('latency_p95_ms', 0):.1f}")
            with col2:
                st.metric('EMA (ms)', f"{perf.get('ema_turn_latency_ms', 0):.1f}")
            with col3:
                st.metric('Turns/sec', f"{perf.get('chat_turns_per_sec', 0):.2f}")
            # History charts
            hist = st.session_state.chat_metrics_history
            if hist:
                import pandas as pd
                df = pd.DataFrame(hist)
                df['age'] = df['ts'] - df['ts'].iloc[0]
                st.line_chart(df.set_index('age')[['p95','ema']])
                st.line_chart(df.set_index('age')[['tps']])
                # Attention & fatigue sparkline (filter non-null)
                if 'attention_focus' in df.columns and df['attention_focus'].notnull().any():
                    af_df = df[['age','attention_focus']].dropna()
                    st.line_chart(af_df.set_index('age'))
                if 'fatigue' in df.columns and df['fatigue'].notnull().any():
                    fg_df = df[['age','fatigue']].dropna()
                    st.line_chart(fg_df.set_index('age'))
            # Metacog snapshot
            metacog = current.get('metacog', {})
            if metacog.get('available'):
                st.markdown('**Last Metacog Snapshot**')
                st.json(metacog.get('snapshot', {}))
            # Metacog history tail
            if st.session_state.metacog_history_cache:
                with st.expander('Metacog History Tail'):
                    for snap in st.session_state.metacog_history_cache:
                        st.write(f"Turn {snap.get('turn_counter')} | p95={snap.get('performance',{}).get('latency_p95_ms')} | stm_util={snap.get('stm_utilization')}")
            else:
                st.caption('No metacog history yet.')
            # Consolidation status
            cons = current.get('consolidation', {})
            st.markdown('**Consolidation Status**')
            if cons.get('active'):
                st.write(cons.get('status'))
                events = cons.get('recent_events') or []
                if events:
                    with st.expander('Recent Consolidation Events'):
                        for ev in events:
                            st.write(ev)
            else:
                st.caption(cons.get('reason','inactive'))
            # API error panel
            if 'api_errors' in st.session_state and st.session_state.api_errors:
                with st.expander('⚠️ Recent API Errors'):
                    for e in reversed(st.session_state.api_errors[-25:]):
                        st.write(f"{e.get('ts')} | {e.get('method')} {e.get('endpoint')} -> {e.get('error')}")
                    if st.button('Clear API Errors', key='clear_api_err_btn'):
                        st.session_state.api_errors = []
                        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("*George - Human-AI Cognitive Architecture | Phase 1 + Phase 2 Features*")

if __name__ == "__main__":
    main()
    # QA Instructions (manual):
    # 1. Send messages like "Alice is a doctor" then "Alice is a painter" to observe contradiction badge increment.
    # 2. Send preference statements ("I like chess") multiple times; verify frequency increments in sidebar list (f=).
    # 3. Use category multiselect to isolate one memory type.
    # 4. Toggle 'Contradictions only' to filter list.
    # 5. Use Search box with substring (e.g., 'chess') to narrow results.
    # 6. Confirm counts (Stored, Contradictions) update dynamically after each turn.
