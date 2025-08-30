from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator

from src.chat import ChatService, SessionManager, ContextBuilder
from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from src.core.config import get_chat_config
from src.chat.metrics import metrics_registry

# Singleton scaffolds (replace with DI if project uses container)
_session_manager = SessionManager()
_context_builder = ContextBuilder(chat_config=get_chat_config().__dict__)
_chat_service = ChatService(_session_manager, _context_builder)
_prospective = get_inmemory_prospective_memory()

router = APIRouter(prefix="/agent", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    flags: Optional[Dict[str, bool]] = None
    stream: bool = False


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if req.stream:
        async def _gen() -> AsyncGenerator[bytes, None]:
            async for chunk in _chat_service.process_user_message_stream(
                req.message,
                session_id=req.session_id,
                flags=req.flags,
            ):
                yield (f"{chunk}\n").encode("utf-8")
        return StreamingResponse(_gen(), media_type="text/plain")
    result = _chat_service.process_user_message(
        req.message, session_id=req.session_id, flags=req.flags
    )
    return JSONResponse(result)


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
    return _chat_service.get_context_preview(message=message, session_id=session_id)


@router.get("/chat/performance")
async def chat_performance_status():
    """
    Return current chat performance status (p95 latency, target, degradation flag).
    """
    return _chat_service.performance_status()


@router.get("/chat/metacog/status")
async def chat_metacog_status():
    """Return last metacognitive snapshot (if any)."""
    snap = getattr(_chat_service, "_last_metacog_snapshot", None)
    if not snap:
        return {"available": False}
    history = []
    try:
        hist = getattr(_chat_service, "_metacog_history", None)
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
    cons = getattr(_chat_service, "consolidator", None)
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
