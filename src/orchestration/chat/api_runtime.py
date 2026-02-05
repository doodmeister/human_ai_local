"""Runtime singletons used by the FastAPI interface layer.

Phase 6: interfaces must depend on orchestration only.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.config import get_chat_config
from src.memory.consolidation.consolidator import ConsolidationPolicy, MemoryConsolidator
from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from src.memory.metrics import metrics_registry  # noqa: F401
from src.orchestration.agent_singleton import create_agent
from src.orchestration.chat import ChatService, ContextBuilder, SessionManager

_session_manager: SessionManager | None = None
_context_builder: ContextBuilder | None = None
_chat_service: ChatService | None = None
_agent_instance: Any | None = None
_prospective = get_inmemory_prospective_memory()


def get_agent() -> Any:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_agent()
    return _agent_instance


def get_prospective():
    return _prospective


def get_chat_service() -> ChatService:
    """Lazy initialization of ChatService with agent support."""
    global _session_manager, _context_builder, _chat_service

    if _chat_service is not None:
        return _chat_service

    agent = get_agent()

    _session_manager = SessionManager()

    chat_cfg = get_chat_config().to_dict()

    stm = ltm = episodic = attention = executive = None
    if agent is not None:
        memory = getattr(agent, "memory", None)
        if memory is not None:
            stm = getattr(memory, "stm", None)
            ltm = getattr(memory, "ltm", None)
            episodic = getattr(memory, "episodic", None)
        attention = getattr(agent, "attention", None)
        executive = getattr(agent, "performance_optimizer", None)

    _context_builder = ContextBuilder(
        chat_config=chat_cfg,
        stm=stm,
        ltm=ltm,
        episodic=episodic,
        attention=attention,
        executive=executive,
    )

    consolidator: Optional[MemoryConsolidator] = None
    if stm is not None or ltm is not None:
        try:
            consolidator = MemoryConsolidator(stm=stm, ltm=ltm, policy=ConsolidationPolicy())
        except Exception:
            consolidator = None

    _chat_service = ChatService(
        _session_manager,
        _context_builder,
        consolidator=consolidator,
        agent=agent,
    )

    return _chat_service
