import streamlit as st
import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="George - Human-AI Cognition Agent", page_icon="🤖", layout="centered")
st.title("George: Human-AI Cognition Agent")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("You:", height=80, key="user_input")
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    payload = {"text": user_input.strip()}
    try:
        response = requests.post(f"{BASE_URL}/agent/process", json=payload)
        response.raise_for_status()
        data = response.json()
        ai_response = data.get("response", "[No response]")
        context = data.get("memory_context", None)
        st.session_state['chat_history'].append((user_input, ai_response, context))
    except requests.exceptions.RequestException as e:
        st.session_state['chat_history'].append((user_input, f"[Error: {e}]", None))

# Display chat history
for user, ai, context in reversed(st.session_state['chat_history']):
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(ai)
        if context:
            st.caption(f"Context: {context}")

st.sidebar.header("George Controls")
if st.sidebar.button("Clear Chat History"):
    st.session_state['chat_history'] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Human-AI Cognition\n**Agent:** George\n**API:** [localhost:8000](http://127.0.0.1:8000)")
