from __future__ import annotations
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator

from src.orchestration.chat.api_runtime import (
    get_agent,
    get_chat_service,
    get_prospective,
    metrics_registry,
)

_prospective = get_prospective()

router = APIRouter(prefix="/agent", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    flags: Optional[Dict[str, bool]] = None
    stream: bool = False
    consolidation_salience_threshold: Optional[float] = None


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    chat_svc = get_chat_service()
    
    # Temporarily override consolidation threshold if provided
    original_threshold = None
    if req.consolidation_salience_threshold is not None:
        try:
            adaptive_cfg = chat_svc.context_builder.cfg
            original_threshold = adaptive_cfg.get("consolidation_salience_threshold")
            adaptive_cfg["consolidation_salience_threshold"] = req.consolidation_salience_threshold
        except Exception:
            pass
    
    try:
        if req.stream:
            async def _gen() -> AsyncGenerator[bytes, None]:
                async for chunk in chat_svc.process_user_message_stream(
                    req.message,
                    session_id=req.session_id,
                    flags=req.flags,
                ):
                    yield (f"{chunk}\n").encode("utf-8")
            return StreamingResponse(_gen(), media_type="text/plain")
        result = await asyncio.to_thread(
            chat_svc.process_user_message,
            req.message,
            req.session_id,
            req.flags,
        )
        return JSONResponse(result)
    finally:
        # Restore original threshold
        if original_threshold is not None:
            try:
                adaptive_cfg = chat_svc.context_builder.cfg
                adaptive_cfg["consolidation_salience_threshold"] = original_threshold
            except Exception:
                pass


@router.get("/chat/metrics")
async def chat_metrics(light: bool = True):
    """
    Return chat metrics (lightweight by default).
    """
    snap = metrics_registry.snapshot_light() if light else metrics_registry.snapshot()
    return snap


@router.get("/chat/preview")
async def chat_context_preview(message: str, session_id: Optional[str] = None):
    """
    Return lightweight deterministic context preview (no model generation).
    """
    chat_svc = get_chat_service()
    return chat_svc.get_context_preview(message=message, session_id=session_id)


@router.get("/chat/performance")
async def chat_performance_status():
    """
    Return current chat performance status (p95 latency, target, degradation flag).
    """
    chat_svc = get_chat_service()
    return chat_svc.performance_status()


@router.get("/chat/metacog/status")
async def chat_metacog_status():
    """Return last metacognitive snapshot (if any)."""
    chat_svc = get_chat_service()
    snap = getattr(chat_svc, "_last_metacog_snapshot", None)
    if not snap:
        return {"available": False}
    history = []
    try:
        hist = getattr(chat_svc, "_metacog_history", None)
        if hist is not None:
            history = list(hist)[-10:]
    except Exception:
        history = []
    return {"available": True, "snapshot": snap, "history_tail": history}


@router.get("/chat/consolidation/status")
async def chat_consolidation_status():
    """Return consolidation system status and recent events.

    If consolidator not initialized returns inactive flag.
    """
    chat_svc = get_chat_service()
    cons = getattr(chat_svc, "consolidator", None)
    if not cons:
        return {"active": False, "reason": "consolidator_not_configured"}
    status = cons.status()
    events = cons.events_tail(10)
    return {"active": True, "status": status, "recent_events": events}


# ---------------- Prospective Memory Endpoints ----------------

class ReminderCreate(BaseModel):
    content: str
    due_in_seconds: float
    metadata: Optional[Dict[str, Any]] = None


@router.post("/reminders")
async def create_reminder(rem: ReminderCreate):
    r = _prospective.add_reminder(rem.content, rem.due_in_seconds, metadata=rem.metadata)
    return {"reminder": _prospective.to_dict(r)}


@router.get("/reminders")
async def list_reminders(include_triggered: bool = True):
    data = [_prospective.to_dict(r) for r in _prospective.list_reminders(include_triggered=include_triggered)]
    return {"reminders": data}


@router.get("/reminders/due")
async def due_reminders():
    due = _prospective.check_due()
    return {"due": [_prospective.to_dict(r) for r in due]}


@router.delete("/reminders/triggered")
async def purge_triggered_reminders():
    """Delete all triggered (already fired) reminders from in-memory store."""
    removed = _prospective.purge_triggered()
    return {"purged": removed}


@router.post("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: str):
    success = _prospective.complete_reminder(reminder_id)
    if not success:
        raise HTTPException(status_code=404, detail="reminder_not_found")
    return {"status": "completed", "reminder_id": reminder_id}


@router.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    success = _prospective.delete_reminder(reminder_id)
    if not success:
        raise HTTPException(status_code=404, detail="reminder_not_found")
    return {"status": "deleted", "reminder_id": reminder_id}


# ---------------- Dream Cycle Endpoint ----------------

class DreamRequest(BaseModel):
    cycle_type: str = "light"  # light, deep, or rem


@router.post("/dream/start")
async def start_dream_cycle(dream_req: DreamRequest):
    """
    Trigger a dream state cycle for STM → LTM consolidation.
    
    This processes memories in STM that meet promotion criteria:
    - Rehearsals >= 2 (referenced multiple times)
    - Age >= 5 seconds
    - Importance >= 0.4
    """
    agent = get_agent()
    if agent is None:
        return {"error": "agent_not_initialized", "dream_results": None}
    
    try:
        # Check if agent has dream processor
        dream_proc = getattr(agent, "dream_processor", None)
        if dream_proc is not None:
            results = await dream_proc.enter_dream_cycle(dream_req.cycle_type)
            return {"dream_results": results}
        else:
            # Fallback: use consolidator directly if available
            chat_svc = get_chat_service()
            cons = getattr(chat_svc, "consolidator", None)
            if cons is None:
                return {"error": "consolidator_not_available", "dream_results": None}
            
            # Manual promotion pass
            promoted_count = 0
            for turn_id, stats in list(cons._turn_stats.items()):
                if stats.get("stm_id") and not stats.get("ltm_id"):
                    cons._maybe_promote(turn_id, stats)
                    if stats.get("ltm_id"):
                        promoted_count += 1
            
            return {
                "dream_results": {
                    "cycle_type": dream_req.cycle_type,
                    "promoted_count": promoted_count,
                    "method": "consolidator_fallback"
                }
            }
    except Exception as e:
        return {"error": str(e), "dream_results": None}


# ---------------- LLM Configuration Endpoint ----------------

class LLMConfigUpdate(BaseModel):
    provider: str  # "openai" or "ollama"
    openai_model: Optional[str] = "gpt-4.1-nano"
    ollama_base_url: Optional[str] = "http://localhost:11434"
    ollama_model: Optional[str] = "llama3.2"


@router.post("/config/llm")
async def update_llm_config(config: LLMConfigUpdate):
    """Update LLM provider configuration and reinitialize the agent's LLM provider."""
    try:
        agent = get_agent()
        if agent is None:
            return {"status": "error", "message": "Agent not available"}
        
        # Update the config
        agent.config.llm.provider = config.provider
        if config.openai_model:
            agent.config.llm.openai_model = config.openai_model
        if config.ollama_base_url:
            agent.config.llm.ollama_base_url = config.ollama_base_url
        if config.ollama_model:
            agent.config.llm.ollama_model = config.ollama_model
        
        # Reinitialize the LLM provider
        from src.orchestration.llm_provider import LLMProviderFactory
        try:
            new_provider = LLMProviderFactory.create_from_config(agent.config.llm)
            if not new_provider.is_available():
                return {
                    "status": "warning",
                    "message": f"Provider '{config.provider}' configured but not available. Check configuration."
                }
            agent.llm_provider = new_provider
            agent.openai_client = getattr(new_provider, 'client', None) if hasattr(new_provider, 'client') else None
            
            model_name = config.openai_model if config.provider == "openai" else config.ollama_model
            return {
                "status": "ok",
                "message": f"LLM provider updated to {config.provider} ({model_name})"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to initialize provider: {str(e)}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Proactive suggestions endpoint
@router.get("/suggestions")
async def get_proactive_suggestions():
    """
    Get proactive suggestions based on due reminders, upcoming plan steps, and goal deadlines.
    
    Returns suggestions sorted by priority (highest first).
    """
    try:
        from src.orchestration.chat.proactive_agency import get_proactive_system
        
        proactive = get_proactive_system()
        suggestions = proactive.get_suggestions()
        
        return {
            "suggestions": [s.to_dict() for s in suggestions],
            "urgent_count": len([s for s in suggestions if s.priority >= 0.8]),
            "total_count": len(suggestions)
        }
    except Exception as e:
        return {"suggestions": [], "urgent_count": 0, "total_count": 0, "error": str(e)}


@router.get("/suggestions/urgent")
async def get_urgent_suggestions():
    """
    Get only high-priority (>= 0.8) proactive suggestions.
    
    Use this for notifications or alerts.
    """
    try:
        from src.orchestration.chat.proactive_agency import get_proactive_system
        
        proactive = get_proactive_system()
        suggestions = proactive.get_urgent_suggestions()
        
        return {
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions)
        }
    except Exception as e:
        return {"suggestions": [], "count": 0, "error": str(e)}

