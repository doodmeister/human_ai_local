from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator

from src.chat import ChatService, SessionManager, ContextBuilder
from src.core.config import get_chat_config
from src.chat.metrics import metrics_registry

# Singleton scaffolds (replace with DI if project uses container)
_session_manager = SessionManager()
_context_builder = ContextBuilder(chat_config=get_chat_config().__dict__)
_chat_service = ChatService(_session_manager, _context_builder)

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
