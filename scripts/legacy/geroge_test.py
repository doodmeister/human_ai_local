import asyncio
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import websockets

# -------------------------
# CONFIG
# -------------------------
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/updates"

st.set_page_config(
    page_title="George - Cognitive Architecture Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# API WRAPPER
# -------------------------
class GeorgeAPI:
    @staticmethod
    def get(endpoint: str, timeout: int = 60) -> Dict[str, Any]:
        try:
            r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def post(endpoint: str, data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        try:
            r = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

# -------------------------
# SESSION STATE INIT
# -------------------------
def init_state():
    defaults = {
        "chat_history": [],
        "agent_status": {},
        "dpad_status": {},
        "memory_analytics": {},
        "bias_settings": {"recency": 0.5, "confirmation": 0.5, "salience": 0.5},
        "planner_graph": {},
        "human_likeness": {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# -------------------------
# UTILS
# -------------------------
async def websocket_listener():
    async with websockets.connect(WS_URL) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "agent_status" in data:
                st.session_state.agent_status = data["agent_status"]
            if "dpad_status" in data:
                st.session_state.dpad_status = data["dpad_status"]
            if "memory_analytics" in data:
                st.session_state.memory_analytics = data["memory_analytics"]
            if "human_likeness" in data:
                st.session_state.human_likeness = data["human_likeness"]

# -------------------------
# TAB 1: COGNITIVE OVERVIEW
# -------------------------
def tab_cognitive_overview():
    status = st.session_state.agent_status

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mode", status.get("mode", "Unknown"))
    col2.metric("Cog. Load", f"{status.get('cognitive_load',0):.1%}")
    col3.metric("STM", status.get("stm_count", 0))
    col4.metric("LTM", status.get("ltm_count", 0))
    col5.metric("Active Processes", status.get("active_processes", 0))

    # Bias sliders
    st.subheader("Meta-Cognitive Bias Controls")
    for bias in ["recency", "confirmation", "salience"]:
        st.session_state.bias_settings[bias] = st.slider(
            f"{bias.title()} Bias",
            0.0, 1.0,
            st.session_state.bias_settings[bias],
            0.05
        )
    if st.button("Apply Bias Settings"):
        GeorgeAPI.post("/api/metacognition/bias", st.session_state.bias_settings)

    # Cognitive load & fatigue over time
    if "load_history" in status:
        df = pd.DataFrame(status["load_history"], columns=["time", "load"])
        fig = px.line(df, x="time", y="load", title="Cognitive Load Over Time")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 2: COGNITIVE CHAT
# -------------------------
def tab_cognitive_chat():
    st.subheader("Chat with George")
    user_input = st.text_input("Your message")
    reflective = st.checkbox("Reflective Mode", value=False)

    if user_input and st.button("Send"):
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "time": datetime.now()}
        )
        payload = {
            "text": user_input,
            "include_reflection": reflective,
            "use_memory_context": True
        }
        resp = GeorgeAPI.post("/api/agent/chat", payload)
        st.session_state.chat_history.append(
            {"role": "george", "content": resp.get("response", ""), "trace": resp}
        )

    for msg in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"**{msg['role'].title()}**: {msg['content']}")
        if msg["role"] == "george" and "trace" in msg:
            with st.expander("Cognitive Trace"):
                st.json(msg["trace"])

# -------------------------
# TAB 3: DPAD NEURAL MODEL
# -------------------------
def tab_dpad():
    st.subheader("DPAD Model Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Training"):
            GeorgeAPI.post("/api/dpad/train", {"action": "start"})
        if st.button("Stop Training"):
            GeorgeAPI.post("/api/dpad/train", {"action": "stop"})
    with col2:
        target_nonlin = st.multiselect(
            "Nonlinear Elements", ["A'", "K", "Cy", "Cz"], default=["A'"]
        )
        if st.button("Apply Nonlinearity Selection"):
            GeorgeAPI.post("/api/dpad/nonlinearity", {"elements": target_nonlin})

    # Performance frontier
    dpad_status = st.session_state.dpad_status
    if "frontier" in dpad_status:
        df = pd.DataFrame(dpad_status["frontier"])
        fig = px.scatter(
            df, x="self_pred_cc", y="decode_cc", color="model",
            title="Performance Frontier"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 4: MEMORY DASHBOARD
# -------------------------
def tab_memory():
    mem_analytics = st.session_state.memory_analytics
    if "ltm_salience_recency" in mem_analytics:
        df = pd.DataFrame(mem_analytics["ltm_salience_recency"])
        fig = px.scatter(
            df, x="recency", y="salience", color="emotion",
            title="LTM: Salience vs Recency"
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Search Memories"):
        q = st.text_input("Query")
        if q:
            res = GeorgeAPI.post("/api/memory/search", {"query": q})
            st.json(res)

# -------------------------
# TAB 5: DREAM STATE
# -------------------------
def tab_dream():
    if st.button("Trigger Dream Cycle"):
        GeorgeAPI.post("/api/memory/consolidate", {})
    events = GeorgeAPI.get("/api/memory/consolidation-events")
    st.json(events)

# -------------------------
# TAB 6: EXECUTIVE PLANNER
# -------------------------
def tab_planner():
    graph = st.session_state.planner_graph
    st.json(graph)  # Placeholder: could render as network chart

# -------------------------
# TAB 7: COGNITIVE ANALYTICS
# -------------------------
def tab_analytics():
    hl = st.session_state.human_likeness
    st.metric("Memory Fidelity", hl.get("memory_fidelity", 0))
    st.metric("Attentional Adaptation", hl.get("attentional_adaptation", 0))
    st.metric("Consolidation Precision", hl.get("consolidation_precision", 0))

# -------------------------
# MAIN
# -------------------------
def main():
    st.title("🧠 George - Human-AI Cognitive Architecture Dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Cognitive Overview",
        "Cognitive Chat",
        "DPAD Neural Model",
        "Memory Dashboard",
        "Dream State",
        "Executive Planner",
        "Cognitive Analytics"
    ])
    with tab1:
        tab_cognitive_overview()
    with tab2:
        tab_cognitive_chat()
    with tab3:
        tab_dpad()
    with tab4:
        tab_memory()
    with tab5:
        tab_dream()
    with tab6:
        tab_planner()
    with tab7:
        tab_analytics()

if __name__ == "__main__":
    asyncio.run(websocket_listener())  # Start WS listener
    main()
