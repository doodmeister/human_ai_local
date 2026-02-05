from __future__ import annotations
from typing import Optional, Any
from .chat_service import ChatService
from .conversation_session import SessionManager
from .context_builder import ContextBuilder
from src.core.config import get_chat_config
from importlib import import_module
 

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
    attention: Optional[Any] = None,
    executive: Optional[Any] = None,
) -> ChatService:
    """
    Create a ChatService with injected subsystems if provided.
    Real memory / attention / executive instances can be passed here.
    """
    memory_system = None
    if stm is None or ltm is None or episodic is None:
        MemorySystem = _lazy_import("src.memory", "MemorySystem")
        if MemorySystem:
            try:
                memory_system = MemorySystem()
            except Exception:
                memory_system = None

    if stm is None and memory_system is not None:
        try:
            stm = memory_system.stm
        except Exception:
            stm = None
    if ltm is None and memory_system is not None:
        try:
            ltm = memory_system.ltm
        except Exception:
            ltm = None
    if episodic is None and memory_system is not None:
        try:
            episodic = memory_system.episodic
        except Exception:
            episodic = None
    if attention is None:
        AttentionManager = _lazy_import("src.cognition.attention.attention_manager", "AttentionManager")
        if AttentionManager:
            try:
                attention = AttentionManager()
            except Exception:
                attention = None
        if attention is None:
            AttentionMechanism = _lazy_import("src.cognition.attention.attention_mechanism", "AttentionMechanism")
            if AttentionMechanism:
                try:
                    attention = AttentionMechanism()
                except Exception:
                    attention = None
    if executive is None:
        ExecutiveAgent = _lazy_import("src.executive.executive_agent", "ExecutiveAgent")
        if ExecutiveAgent:
            try:
                executive = ExecutiveAgent()
            except Exception:
                executive = None
    
    # Prospective memory for reminders
    prospective = None
    try:
        if memory_system is not None:
            prospective = memory_system.prospective
        else:
            from src.memory.prospective.prospective_memory import get_prospective_memory
            prospective = get_prospective_memory()
    except Exception:
        prospective = None
    
    cfg_obj = get_chat_config()
    cfg = cfg_obj.to_dict()
    cfg.setdefault("retrieval_timeout_ms", cfg_obj.retrieval_timeout_ms)
    builder = ContextBuilder(chat_config=cfg, stm=stm, ltm=ltm, episodic=episodic,
                             attention=attention, executive=executive, prospective=prospective)
    # Lazy create consolidator (won't fail if memory systems absent)
    consolidator = None
    try:
        from src.memory.consolidation.consolidator import MemoryConsolidator, ConsolidationPolicy  # type: ignore
        consolidator = MemoryConsolidator(stm=stm, ltm=ltm, policy=ConsolidationPolicy())
    except Exception:
        consolidator = None
    return ChatService(SessionManager(), builder, consolidator=consolidator)
