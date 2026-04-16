from __future__ import annotations

from importlib import import_module
import logging
from typing import Any, Optional

from src.core.config import get_chat_config
from src.memory.consolidation.consolidator import ConsolidationPolicy, MemoryConsolidator
from src.memory.metrics import metrics_registry
from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
from src.orchestration.agent_singleton import create_agent

_LOGGER = logging.getLogger(__name__)


def _lazy_import(path: str, attr: str) -> Any:
    try:
        mod = import_module(path)
        return getattr(mod, attr)
    except Exception:
        return None


class AppRuntime:
    """Central runtime container for agent and chat service composition."""

    def __init__(self) -> None:
        self._agent_instance: Any | None = None
        self._chat_service: Any | None = None
        self._metacognitive_controller: Any | None = None
        self._prospective = get_inmemory_prospective_memory()

    def get_agent(
        self,
        *,
        config: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ) -> Any:
        if self._agent_instance is None:
            self._agent_instance = create_agent(
                config=config,
                system_prompt=system_prompt,
            )
        controller = self.build_metacognitive_controller(agent=self._agent_instance)
        if controller is not None:
            self._agent_instance.set_metacognitive_controller(controller)
        return self._agent_instance

    def has_agent(self) -> bool:
        return self._agent_instance is not None

    def get_prospective(self) -> Any:
        return self._prospective

    def get_metacognitive_controller(self) -> Any:
        if self._metacognitive_controller is None:
            self._metacognitive_controller = self.build_metacognitive_controller(agent=self.get_agent())
        return self._metacognitive_controller

    def get_chat_service(self) -> Any:
        if self._chat_service is None:
            self._chat_service = self.build_chat_service(agent=self.get_agent())
        return self._chat_service

    def build_metacognitive_controller(
        self,
        *,
        agent: Optional[Any] = None,
        memory: Optional[Any] = None,
        attention: Optional[Any] = None,
        executive: Optional[Any] = None,
        prospective: Optional[Any] = None,
    ) -> Any:
        """Build a deterministic metacognitive controller using available runtime seams."""
        MetacognitiveController = _lazy_import(
            "src.orchestration.metacognition",
            "MetacognitiveController",
        )
        DefaultStateProvider = _lazy_import(
            "src.orchestration.metacognition",
            "DefaultStateProvider",
        )
        DefaultWorkspaceBuilder = _lazy_import(
            "src.orchestration.metacognition",
            "DefaultWorkspaceBuilder",
        )
        FilesystemCycleTracer = _lazy_import(
            "src.orchestration.metacognition",
            "FilesystemCycleTracer",
        )
        if not (MetacognitiveController and DefaultStateProvider and DefaultWorkspaceBuilder and FilesystemCycleTracer):
            return None

        if agent is not None:
            memory = memory or getattr(agent, "memory", None)
            attention = attention or getattr(agent, "attention", None)
            executive = executive or getattr(agent, "performance_optimizer", None)
        if prospective is None:
            prospective = self.get_prospective()

        state_provider = DefaultStateProvider(
            memory=memory,
            attention=attention,
            executive=executive,
            self_model_provider=(lambda session_id: agent.get_self_model(session_id)) if agent is not None and hasattr(agent, "get_self_model") else None,
            task_provider=(lambda session_id: agent.list_cognitive_tasks(session_id)) if agent is not None and hasattr(agent, "list_cognitive_tasks") else None,
        )

        def memory_context_provider(query: str, session_id: str) -> Any:
            if memory is not None and hasattr(memory, "get_context_for_query"):
                try:
                    return memory.get_context_for_query(query)
                except Exception as exc:
                    _LOGGER.warning(
                        "memory.get_context_for_query failed for session %s: %s",
                        session_id,
                        exc,
                        exc_info=True,
                    )
                    return {}
            return {}

        def contradiction_provider(session_id: str) -> list[dict[str, Any]]:
            if memory is not None and hasattr(memory, "get_unresolved_conflicts"):
                try:
                    conflicts = memory.get_unresolved_conflicts(session_id)
                except Exception as exc:
                    _LOGGER.warning(
                        "memory.get_unresolved_conflicts failed for session %s: %s",
                        session_id,
                        exc,
                        exc_info=True,
                    )
                    conflicts = []
                if isinstance(conflicts, list):
                    return [conflict for conflict in conflicts if isinstance(conflict, dict)]
            return []

        workspace_builder = DefaultWorkspaceBuilder(
            memory_context_provider=memory_context_provider,
            contradiction_provider=contradiction_provider,
        )
        tracer = FilesystemCycleTracer()
        controller = MetacognitiveController(
            state_provider=state_provider,
            workspace_builder=workspace_builder,
            cycle_tracer=tracer,
        )
        self._metacognitive_controller = controller
        return controller

    def build_chat_service(
        self,
        *,
        agent: Optional[Any] = None,
        stm: Optional[Any] = None,
        ltm: Optional[Any] = None,
        episodic: Optional[Any] = None,
        semantic: Optional[Any] = None,
        attention: Optional[Any] = None,
        executive: Optional[Any] = None,
        prospective: Optional[Any] = None,
        procedural: Optional[Any] = None,
        metacognitive_controller: Optional[Any] = None,
    ) -> Any:
        """Build a ChatService using explicit dependencies or runtime fallbacks."""
        from src.orchestration.chat.chat_service import ChatService
        from src.orchestration.chat.context_builder import ContextBuilder
        from src.orchestration.chat.conversation_session import SessionManager

        metrics_registry.reset()

        memory_system = None
        if agent is not None:
            memory = getattr(agent, "memory", None)
            if memory is not None:
                stm = stm or getattr(memory, "stm", None)
                ltm = ltm or getattr(memory, "ltm", None)
                episodic = episodic or getattr(memory, "episodic", None)
                semantic = semantic or getattr(memory, "semantic", None)
                prospective = prospective or getattr(memory, "prospective", None)
                procedural = procedural or getattr(memory, "procedural", None)
            attention = attention or getattr(agent, "attention", None)
            executive = executive or getattr(agent, "performance_optimizer", None)

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
        if semantic is None and memory_system is not None:
            try:
                semantic = memory_system.semantic
            except Exception:
                semantic = None
        if prospective is None and memory_system is not None:
            try:
                prospective = memory_system.prospective
            except Exception:
                prospective = None
        if procedural is None and memory_system is not None:
            try:
                procedural = memory_system.procedural
            except Exception:
                procedural = None

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

        if prospective is None:
            prospective = self.get_prospective()

        if metacognitive_controller is None:
            metacognitive_controller = self.build_metacognitive_controller(
                agent=agent,
                memory=getattr(agent, "memory", None) if agent is not None else None,
                attention=attention,
                executive=executive,
                prospective=prospective,
            )

        cfg_obj = get_chat_config()
        cfg = cfg_obj.to_dict()
        cfg.setdefault("retrieval_timeout_ms", cfg_obj.retrieval_timeout_ms)

        autobiographical_store = None
        AutobiographicalGraphStore = _lazy_import("src.memory.autobiographical", "AutobiographicalGraphStore")
        if AutobiographicalGraphStore is not None:
            try:
                autobiographical_store = AutobiographicalGraphStore()
            except Exception:
                autobiographical_store = None

        builder = ContextBuilder(
            chat_config=cfg,
            stm=stm,
            ltm=ltm,
            episodic=episodic,
            semantic=semantic,
            attention=attention,
            executive=executive,
            prospective=prospective,
            procedural=procedural,
            autobiographical_store=autobiographical_store,
        )

        consolidator: Optional[MemoryConsolidator] = None
        if stm is not None or ltm is not None:
            try:
                consolidator = MemoryConsolidator(stm=stm, ltm=ltm, policy=ConsolidationPolicy())
            except Exception:
                consolidator = None

        return ChatService(
            SessionManager(),
            builder,
            consolidator=consolidator,
            agent=agent,
            metacognitive_controller=metacognitive_controller,
        )


_runtime: AppRuntime | None = None


def get_runtime() -> AppRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AppRuntime()
    return _runtime


def reset_runtime() -> None:
    global _runtime
    _runtime = None