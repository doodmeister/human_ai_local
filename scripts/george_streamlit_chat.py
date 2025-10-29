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


def render_sidebar(connection_ok: bool) -> Dict[str, Any]:
    """Render sidebar controls and return the current chat flags and salience threshold."""
    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API base URL", value=st.session_state.api_base_url)
        cleaned = (api_url or "").rstrip("/")
        st.session_state.api_base_url = cleaned or DEFAULT_API_BASE
        
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
