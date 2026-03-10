from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .emotion_salience import estimate_salience_and_valence
from .metrics import metrics_registry

if TYPE_CHECKING:
    from .chat_service import ChatService
    from .context_builder import ContextBuilder
    from .conversation_session import SessionManager


def __getattr__(name: str) -> Any:
    if name == "ChatService":
        from .chat_service import ChatService as _ChatService

        return _ChatService
    if name == "ContextBuilder":
        from .context_builder import ContextBuilder as _ContextBuilder

        return _ContextBuilder
    if name == "SessionManager":
        from .conversation_session import SessionManager as _SessionManager

        return _SessionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ChatService",
    "SessionManager",
    "ContextBuilder",
    "estimate_salience_and_valence",
    "metrics_registry",
]
