import streamlit as st
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="George - Human-AI Cognition Agent", page_icon="🤖", layout="centered")
st.title("George: Human-AI Cognition Agent")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- Meta-Cognition/Agent State Display ---
def get_agent_status():
    try:
        resp = requests.get(f"{BASE_URL}/agent/status")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}

# --- Handle Chat Submission ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("You:", height=80, key="user_input")
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    payload = {"text": user_input.strip()}
    try:
        response = requests.post(f"{BASE_URL}/agent/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        ai_response = data.get("response", "[No response]")
        # Show learning/consolidation from memory
        memory_events = data.get("memory_events", [])
        context = data.get("context", None)
        # Optionally, show rationale/trace
        rationale = data.get("rationale", None)
        st.session_state['chat_history'].append({
            "user": user_input, 
            "ai": ai_response, 
            "context": context, 
            "memory_events": memory_events, 
            "rationale": rationale
        })
    except requests.exceptions.RequestException as e:
        st.session_state['chat_history'].append({
            "user": user_input, 
            "ai": f"[Error: {e}]", 
            "context": None, 
            "memory_events": [], 
            "rationale": None
        })

# --- Chat History Display ---
for turn in reversed(st.session_state['chat_history']):
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["ai"])
        if turn["context"]:
            st.caption(f"Context: {turn['context']}")
        if turn["memory_events"]:
            st.info("Learned/Consolidated:")
            for evt in turn["memory_events"]:
                st.markdown(f"- {evt}")
        if turn["rationale"]:
            st.caption(f"Agent rationale: {turn['rationale']}")

# --- Sidebar: Controls and Memory Search ---
st.sidebar.header("George Controls")
if st.sidebar.button("Clear Chat History"):
    st.session_state['chat_history'] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Cognitive State")

# Display agent meta-cognition/status
status = get_agent_status()
if status:
    st.sidebar.markdown(f"**Mode:** {status.get('mode', '?')}")
    st.sidebar.markdown(f"**Resources:** {status.get('resources', {})}")
    st.sidebar.markdown(f"**Performance:** {status.get('performance_metrics', {})}")
    st.sidebar.markdown(f"**Active Processes:** {status.get('active_processes', 0)}")
else:
    st.sidebar.info("Agent status unavailable.")

# --- Memory Search UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("Memory Search")
memory_query = st.sidebar.text_input("Search memory for:", "")
if st.sidebar.button("Search Memory") and memory_query.strip():
    try:
        search_payload = {"query": memory_query.strip()}
        search_response = requests.post(f"{BASE_URL}/agent/memory/search", json=search_payload)
        search_response.raise_for_status()
        search_data = search_response.json()
        memory_context = search_data.get("results", [])
        if memory_context:
            st.sidebar.markdown("**Results:**")
            for mem in memory_context:
                st.sidebar.markdown(f"- **[{mem.get('type', '?')}]** {mem.get('content', '')} (Relevance: {mem.get('relevance', 0):.2f})")
        else:
            st.sidebar.info("No relevant memories found.")
    except Exception as e:
        st.sidebar.error(f"Memory search failed: {e}")

# Optional: Add manual dream/consolidation trigger for development/demo
if st.sidebar.button("Trigger Dream/Consolidate"):
    try:
        resp = requests.post(f"{BASE_URL}/agent/memory/consolidate")
        if resp.status_code == 200:
            st.sidebar.success("Dream/consolidation triggered!")
    except Exception as e:
        st.sidebar.error(f"Consolidation failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Human-AI Cognition\n**Agent:** George\n**API:** [localhost:8000](http://127.0.0.1:8000)")
