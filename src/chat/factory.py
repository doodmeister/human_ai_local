from __future__ import annotations
from typing import Optional, Any
from .chat_service import ChatService
from .conversation_session import SessionManager
from .context_builder import ContextBuilder
from src.core.config import get_chat_config
from importlib import import_module
import os

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
    if stm is None:
        VectorShortTermMemory = _lazy_import("src.memory.stm.vector_stm", "VectorShortTermMemory")
        if VectorShortTermMemory:
            try:
                stm = VectorShortTermMemory()
            except Exception:
                stm = None
    if ltm is None:
        VectorLongTermMemory = _lazy_import("src.memory.ltm.vector_ltm", "VectorLongTermMemory")
        if VectorLongTermMemory:
            try:
                ltm = VectorLongTermMemory()
            except Exception:
                ltm = None
    if episodic is None:
        EpisodicMemorySystem = _lazy_import("src.memory.episodic.episodic_memory", "EpisodicMemorySystem")
        if EpisodicMemorySystem:
            try:
                episodic = EpisodicMemorySystem()
            except Exception:
                episodic = None
    # Skip semantic memory heavy dependency in lightweight test mode
    if os.environ.get("DISABLE_SEMANTIC_MEMORY") == "1":
        if hasattr(__import__('src.memory', fromlist=['semantic']), 'semantic'):
            pass  # no-op
    if attention is None:
        AttentionMechanism = _lazy_import("src.attention.attention_mechanism", "AttentionMechanism")
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
