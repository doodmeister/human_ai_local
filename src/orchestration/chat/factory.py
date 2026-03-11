from __future__ import annotations
from typing import Optional, Any
from .chat_service import ChatService
from importlib import import_module
from src.orchestration.runtime.app_container import get_runtime
 

def _lazy_import(path: str, attr: str):
    try:
        mod = import_module(path)
        return getattr(mod, attr)
    except Exception:
        return None

def build_chat_service(
    stm: Optional[Any] = None,
    ltm: Optional[Any] = None,
    episodic: Optional[Any] = None,
    semantic: Optional[Any] = None,
    attention: Optional[Any] = None,
    executive: Optional[Any] = None,
) -> ChatService:
    """Create a ChatService with injected subsystems or runtime fallbacks."""
    return get_runtime().build_chat_service(
        stm=stm,
        ltm=ltm,
        episodic=episodic,
        semantic=semantic,
        attention=attention,
        executive=executive,
    )
