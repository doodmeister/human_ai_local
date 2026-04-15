"""HTTP client for the George cognitive backend.

Provides typed helpers used by the Chainlit UI.  All methods are async and
communicate with the FastAPI backend over HTTP.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

API_BASE = os.getenv("GEORGE_API_BASE_URL", "http://localhost:8000")

# Shared client – created once, reused across async calls.
_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(base_url=API_BASE, timeout=90.0)
    return _client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def check_health() -> bool:
    try:
        resp = await _get_client().get("/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

async def send_chat(
    message: str,
    session_id: str,
    *,
    flags: Optional[Dict[str, bool]] = None,
    salience_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": message, "session_id": session_id}
    if flags:
        filtered = {k: v for k, v in flags.items() if v}
        if filtered:
            payload["flags"] = filtered
    if salience_threshold is not None:
        payload["consolidation_salience_threshold"] = salience_threshold
    resp = await _get_client().post("/agent/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM Config
# ---------------------------------------------------------------------------

async def update_llm_config(
    provider: str = "openai",
    openai_model: str = "gpt-4.1-nano",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.2",
) -> Dict[str, Any]:
    payload = {
        "provider": provider,
        "openai_model": openai_model,
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
    }
    try:
        resp = await _get_client().post("/agent/config/llm", json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

async def list_reminders() -> List[Dict[str, Any]]:
    try:
        resp = await _get_client().get("/agent/reminders")
        resp.raise_for_status()
        return resp.json().get("reminders", [])
    except httpx.HTTPError:
        return []


async def create_reminder(
    content: str,
    due_in_seconds: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {"content": content, "due_in_seconds": due_in_seconds}
    if metadata:
        payload["metadata"] = metadata
    resp = await _get_client().post("/agent/reminders", json=payload)
    resp.raise_for_status()
    return resp.json().get("reminder")


async def complete_reminder(reminder_id: str) -> bool:
    try:
        resp = await _get_client().post(f"/agent/reminders/{reminder_id}/complete")
        resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


async def delete_reminder(reminder_id: str) -> bool:
    try:
        resp = await _get_client().delete(f"/agent/reminders/{reminder_id}")
        resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------------
# Dream / Reflection
# ---------------------------------------------------------------------------

async def trigger_dream(cycle_type: str = "light") -> Dict[str, Any]:
    resp = await _get_client().post("/agent/dream/start", json={"cycle_type": cycle_type})
    resp.raise_for_status()
    return resp.json()


async def trigger_reflection() -> Dict[str, Any]:
    resp = await _get_client().post("/agent/metacog/reflect")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Goals / Executive
# ---------------------------------------------------------------------------

async def list_goals(active_only: bool = True) -> List[Dict[str, Any]]:
    try:
        params = {"active_only": "true" if active_only else "false"}
        resp = await _get_client().get("/executive/goals", params=params)
        resp.raise_for_status()
        return resp.json().get("goals", [])
    except httpx.HTTPError:
        return []


async def create_goal(
    title: str,
    description: str = "",
    priority: float = 0.5,
    deadline: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "title": title,
        "description": description,
        "priority": priority,
    }
    if deadline:
        payload["deadline"] = deadline
    try:
        resp = await _get_client().post("/executive/goals", json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


async def execute_goal(goal_id: str) -> Optional[Dict[str, Any]]:
    try:
        resp = await _get_client().post(f"/executive/goals/{goal_id}/execute", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


async def get_goal_status(goal_id: str) -> Optional[Dict[str, Any]]:
    try:
        resp = await _get_client().get(f"/executive/goals/{goal_id}/status")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


async def get_goal_plan(goal_id: str) -> Optional[Dict[str, Any]]:
    try:
        resp = await _get_client().get(f"/executive/goals/{goal_id}/plan")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


async def get_goal_schedule(goal_id: str) -> Optional[Dict[str, Any]]:
    try:
        resp = await _get_client().get(f"/executive/goals/{goal_id}/schedule")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


# ---------------------------------------------------------------------------
# Memory Browser
# ---------------------------------------------------------------------------

async def fetch_memories(system: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        resp = await _get_client().get(f"/memory/{system}/list", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data.get("memories", data.get("items", []))
    except httpx.HTTPError:
        return []


async def search_memories(system: str, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    try:
        resp = await _get_client().post(
            f"/memory/{system}/search",
            json={"query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", data.get("memories", []))
    except httpx.HTTPError:
        return []


def _first_numeric_value(item: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


async def unified_memory_search(query: str, max_per_system: int = 5) -> List[Dict[str, Any]]:
    """Search STM, LTM, episodic, and semantic memory sources concurrently."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    systems = ["stm", "ltm", "episodic"]
    tasks = [search_memories(system, normalized_query, max_results=max_per_system) for system in systems]
    tasks.append(fetch_semantic_facts(normalized_query))

    results_per_system = await asyncio.gather(*tasks, return_exceptions=True)

    merged: List[Dict[str, Any]] = []
    labels = systems + ["semantic"]
    for label, result in zip(labels, results_per_system):
        if isinstance(result, Exception):
            continue
        for item in result or []:
            enriched = dict(item)
            enriched["_source_system"] = label
            enriched.setdefault("source_system", label)
            merged.append(enriched)

    merged.sort(
        key=lambda item: (
            -_first_numeric_value(item, "relevance", "similarity", "confidence", "importance", "activation"),
            -_first_numeric_value(item, "importance", "activation"),
        )
    )
    return merged[:20]


async def delete_memory(system: str, memory_id: str) -> bool:
    try:
        resp = await _get_client().delete(f"/memory/{system}/delete/{memory_id}")
        resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


async def fetch_semantic_facts(query: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        payload = {"query": query or ""}
        resp = await _get_client().post("/semantic/fact/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("facts", data.get("results", []))
    except httpx.HTTPError:
        return []


async def fetch_prospective_memories() -> List[Dict[str, Any]]:
    return await list_reminders()


async def fetch_procedural_memories() -> List[Dict[str, Any]]:
    try:
        resp = await _get_client().get("/procedure/list")
        resp.raise_for_status()
        data = resp.json()
        return data.get("procedures", data.get("items", []))
    except httpx.HTTPError:
        return []


# ---------------------------------------------------------------------------
# Learning / Experiments
# ---------------------------------------------------------------------------

async def fetch_learning_metrics() -> Dict[str, Any]:
    try:
        resp = await _get_client().get("/executive/learning/metrics")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        try:
            resp = await _get_client().get("/executive/status")
            resp.raise_for_status()
            return resp.json().get("learning_metrics", {})
        except httpx.HTTPError:
            return {}


async def fetch_experiments() -> List[Dict[str, Any]]:
    try:
        resp = await _get_client().get("/executive/experiments")
        resp.raise_for_status()
        return resp.json().get("experiments", [])
    except httpx.HTTPError:
        return []


async def fetch_execution_outcomes(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        resp = await _get_client().get("/executive/outcomes", params={"limit": limit})
        resp.raise_for_status()
        return resp.json().get("outcomes", [])
    except httpx.HTTPError:
        return []


# ---------------------------------------------------------------------------
# Metacognition
# ---------------------------------------------------------------------------

async def fetch_metacognition_dashboard(
    session_id: Optional[str] = None,
    *,
    history_limit: int = 10,
    limit: int = 50,
) -> Dict[str, Any]:
    try:
        params: Dict[str, Any] = {"history_limit": history_limit, "limit": limit}
        if session_id:
            params["session_id"] = session_id
        resp = await _get_client().get("/agent/metacog/dashboard", params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return {}


async def fetch_metacognition_tasks(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        params: Dict[str, Any] = {}
        if session_id:
            params["session_id"] = session_id
        resp = await _get_client().get("/agent/metacog/tasks", params=params)
        resp.raise_for_status()
        return resp.json().get("items", [])
    except httpx.HTTPError:
        return []


async def fetch_metacognition_reflections(
    session_id: Optional[str] = None,
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    try:
        params: Dict[str, Any] = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        resp = await _get_client().get("/agent/metacog/reflections", params=params)
        resp.raise_for_status()
        return resp.json().get("items", [])
    except httpx.HTTPError:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_due_display(reminder: Dict[str, Any]) -> str:
    """Return a short human-readable due phrase."""
    phrase = reminder.get("due_phrase")
    if phrase:
        return phrase
    raw = reminder.get("due_time")
    if not raw:
        return "no specific time"
    try:
        cleaned = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        due_dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return "unknown"
    if due_dt.tzinfo is None:
        due_dt = due_dt.replace(tzinfo=timezone.utc)
    delta = due_dt - datetime.now(timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "due now"
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24
    if minutes < 1:
        return "in <1 min"
    if minutes < 60:
        return f"in {minutes} min"
    if hours < 24:
        return f"in {hours} hr"
    return f"in {days} day{'s' if days != 1 else ''}"
