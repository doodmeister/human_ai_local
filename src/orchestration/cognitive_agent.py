"""
Core Cognitive Agent - Central orchestrator for the cognitive architecture
"""

from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import logging
import threading
import time

from ..core.config import CognitiveConfig
from ..memory.autobiographical import AutobiographicalGraphStore
from .cognitive_layers import ChatCognitiveLayerRuntime
from .agent import (
    CognitiveAgentLLMSession,
    CognitiveAgentRuntimeBuilder,
    CognitiveMaintenanceService,
    CognitiveReflectionService,
    CognitiveTurnProcessor,
)


def _lazy_import_llm():
    from .agent.llm_session import _lazy_import_llm as _session_lazy_import_llm

    return _session_lazy_import_llm()

# Load environment variables for LLM
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class CognitiveAgent:
    """
    Central cognitive agent that orchestrates all cognitive processes
    
    This class implements the main cognitive loop:
    1. Input processing through sensory buffer
    2. Memory retrieval and context building
    3. Attention allocation and focus management
    4. Response generation and memory consolidation
    """
    
    def __init__(self, config: Optional[CognitiveConfig] = None, system_prompt: Optional[str] = None):
        """
        Initialize the cognitive agent
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or CognitiveConfig.from_env()
        # Temporary simple session ID
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._turn_counter = 0
        self._metacognitive_controller = None
        self._last_metacognitive_cycle = None
        self._background_cognition_thread = None
        self._background_cognition_running = False
        self._background_cognition_interval_seconds = None
        self._background_cognition_limit = 5
        self._idle_reflection_interval_seconds = 60.0
        self._last_metacognitive_activity_ts = time.time()
        self._last_idle_reflection_ts = 0.0
        try:
            self._autobiographical_store = AutobiographicalGraphStore()
        except Exception:
            self._autobiographical_store = None
        
        # Initialize cognitive components
        self._initialize_components()
        self._cognitive_layers = ChatCognitiveLayerRuntime()
        
        # Cognitive state
        self.current_fatigue = 0.0
        self.attention_focus = []
        self.active_goals = []
        # Stores recent conversation history for proactive recall
        self.conversation_context: List[Dict[str, Any]] = []
        
        # LLM configuration - Initialize LLM provider
        self._llm_session = CognitiveAgentLLMSession(
            config=self.config,
            system_prompt=system_prompt,
            lazy_import_llm=_lazy_import_llm,
        )
        
        # Reflection state
        self._reflection_service = CognitiveReflectionService(
            get_memory=lambda: self.memory,
            persist_report=lambda report: self._persist_reflection_report(report),
            load_reports=lambda limit: self._load_persisted_reflection_reports(limit),
        )
        self._maintenance_service = CognitiveMaintenanceService(
            get_session_id=lambda: self.session_id,
            get_current_fatigue=lambda: self.current_fatigue,
            set_current_fatigue=self._set_current_fatigue,
            get_attention_focus=lambda: self.attention_focus,
            get_active_goals=lambda: self.active_goals,
            get_conversation_context=lambda: self.conversation_context,
            get_memory=lambda: self.memory,
            get_attention=lambda: self.attention,
            get_sensory_processor=lambda: self.sensory_processor,
            get_dream_processor=lambda: self.dream_processor,
        )
        self._turn_processor = CognitiveTurnProcessor(
            get_session=lambda: self,
            get_session_id=lambda: self.session_id,
            get_cognitive_layers=lambda: self._cognitive_layers,
            get_sensory_interface=lambda: self.sensory_interface,
            get_memory=lambda: self.memory,
            get_attention=lambda: self.attention,
            get_llm_session=lambda: self._llm_session,
            get_conversation_context=lambda: self.conversation_context,
            get_neural_integration=lambda: self.neural_integration,
            get_current_fatigue=lambda: self.current_fatigue,
            get_turn_counter=lambda: self._turn_counter,
            increment_turn_counter=self._increment_turn_counter,
            set_current_fatigue=self._set_current_fatigue,
            set_attention_focus=self._set_attention_focus,
            get_autobiographical_store=lambda: self._autobiographical_store,
            get_active_goal_ids=lambda: list(self.active_goals),
        )
        
        logger.info(f"Cognitive agent initialized with session ID: {self.session_id}")
        if self.llm_provider:
            provider_name = self.config.llm.provider
            model_name = self.config.llm.openai_model if provider_name == "openai" else self.config.llm.ollama_model
            logger.info(f"LLM provider: {provider_name} ({model_name})")
    
    def _initialize_components(self):
        """Initialize all cognitive architecture components"""
        runtime = CognitiveAgentRuntimeBuilder(self.config).build()
        self.memory = runtime.memory
        self.attention = runtime.attention
        self.sensory_processor = runtime.sensory_processor
        self.sensory_interface = runtime.sensory_interface
        self.neural_integration = runtime.neural_integration
        self.dream_processor = runtime.dream_processor
        self.performance_optimizer = runtime.performance_optimizer
        
        logger.info("Cognitive components initialized")

    @property
    def system_prompt(self) -> str:
        return self._llm_session.system_prompt

    @property
    def llm_conversation(self) -> List[Dict[str, str]]:
        return self._llm_session.conversation

    @llm_conversation.setter
    def llm_conversation(self, value: List[Dict[str, str]]) -> None:
        self._llm_session.conversation = list(value)

    @property
    def llm_provider(self) -> Any:
        return self._llm_session.provider

    @property
    def openai_client(self) -> Any:
        return self._llm_session.openai_client

    @property
    def reflection_reports(self) -> List[Dict[str, Any]]:
        return self._reflection_service.reports

    @property
    def _reflection_scheduler_running(self) -> bool:
        return bool(self._reflection_service.get_status().get("scheduler_running", False))

    @property
    def _background_scheduler_running(self) -> bool:
        return bool(self._background_cognition_running)

    def _set_current_fatigue(self, value: float) -> None:
        self.current_fatigue = value

    def _set_attention_focus(self, value: List[Any]) -> None:
        self.attention_focus = value

    def _increment_turn_counter(self) -> None:
        self._turn_counter += 1
    
    async def process_input(
        self,
        input_data: str,
        input_type: str = "text",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main cognitive processing loop
        
        Args:
            input_data: Raw input data (text, audio, etc.)
            input_type: Type of input ("text", "audio", "image")
            context: Additional context information
        
        Returns:
            Generated response
        """
        response = await self._turn_processor.process_input(input_data, input_type, context)
        self._mark_metacognitive_activity()
        controller_response = None
        try:
            cycle = self._run_metacognitive_cycle(
                input_data,
                input_type=input_type,
                session_id=(context or {}).get("session_id") if context else None,
                metadata=context,
            )
            if cycle is not None and cycle.execution_result is not None:
                controller_response = cycle.execution_result.response_text
                self._mark_metacognitive_activity()
        except Exception as exc:
            logger.debug("Metacognitive cycle skipped in process_input: %s", exc)
        return response or controller_response or ""

    def set_metacognitive_controller(self, controller: Any) -> None:
        self._metacognitive_controller = controller

    def get_metacognitive_controller(self) -> Any:
        return self._metacognitive_controller

    def get_metacognitive_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_session_id = session_id or self.session_id
        cycle = self._last_metacognitive_cycle
        if cycle is None:
            controller = self._metacognitive_controller
            if controller is not None:
                status = controller.build_status(resolved_session_id)
                status["background_scheduler_running"] = self._background_cognition_running
                status["background_tick_interval_seconds"] = self._background_cognition_interval_seconds
                status["idle_reflection_interval_seconds"] = self._idle_reflection_interval_seconds
                status["last_idle_reflection_ts"] = self._last_idle_reflection_ts or None
                status.setdefault("unresolved_contradiction_count", 0)
                return status
            return {
                "available": False,
                "session_id": resolved_session_id,
                "background_scheduler_running": self._background_cognition_running,
                "background_tick_interval_seconds": self._background_cognition_interval_seconds,
                "idle_reflection_interval_seconds": self._idle_reflection_interval_seconds,
                "last_idle_reflection_ts": self._last_idle_reflection_ts or None,
            }
        return {
            "available": True,
            "session_id": resolved_session_id,
            "last_cycle": self.get_last_cycle_summary(resolved_session_id),
            "scheduled_task_count": len(cycle.scheduled_tasks or ()),
            "active_goal_count": len(cycle.ranked_goals or ()),
            "background_scheduler_running": self._background_cognition_running,
            "background_tick_interval_seconds": self._background_cognition_interval_seconds,
            "idle_reflection_interval_seconds": self._idle_reflection_interval_seconds,
            "last_idle_reflection_ts": self._last_idle_reflection_ts or None,
            "unresolved_contradiction_count": len(cycle.workspace.contradictions if cycle.workspace is not None else ()),
        }

    def get_last_cycle_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_session_id = session_id or self.session_id
        cycle = self._last_metacognitive_cycle
        if cycle is None:
            controller = self._metacognitive_controller
            if controller is not None:
                latest_trace = controller.get_latest_trace(resolved_session_id)
                if latest_trace is not None:
                    return {
                        "available": True,
                        "session_id": resolved_session_id,
                        "cycle_id": latest_trace.get("cycle_id"),
                        "trace_id": latest_trace.get("cycle_id"),
                        "selected_goal_id": ((latest_trace.get("plan") or {}).get("selected_goal") or {}).get("goal_id"),
                        "selected_goal_kind": ((latest_trace.get("plan") or {}).get("selected_goal") or {}).get("kind"),
                        "act_types": [act.get("act_type") for act in ((latest_trace.get("plan") or {}).get("acts") or [])],
                        "success_score": (latest_trace.get("critic_report") or {}).get("success_score"),
                        "follow_up_recommended": (latest_trace.get("critic_report") or {}).get("follow_up_recommended"),
                        "response_text": (latest_trace.get("execution_result") or {}).get("response_text"),
                        "scheduled_task_count": len(latest_trace.get("scheduled_tasks") or []),
                    }
            return {"available": False, "session_id": resolved_session_id}
        return self._summarize_cycle(cycle, session_id=resolved_session_id)

    def get_self_model(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_session_id = session_id or self.session_id
        cycle = self._last_metacognitive_cycle
        if cycle is not None and cycle.updated_self_model is not None:
            return self._serialize_cycle_value(cycle.updated_self_model)
        controller = self._metacognitive_controller
        if controller is not None:
            persisted = controller.get_persisted_self_model(resolved_session_id)
            if isinstance(persisted, dict):
                return persisted
        state = self._cognitive_layers.get_self_model_state()
        return state or {"session_id": resolved_session_id, "available": False}

    def list_cognitive_tasks(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        resolved_session_id = session_id or self.session_id
        cycle = self._last_metacognitive_cycle
        if cycle is None:
            controller = self._metacognitive_controller
            if controller is not None:
                return [dict(task) for task in controller.list_tasks(resolved_session_id) if isinstance(task, dict)]
            return []
        return [self._serialize_cycle_value(task) for task in cycle.scheduled_tasks]

    def get_active_goals(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        resolved_session_id = session_id or self.session_id
        cycle = self._last_metacognitive_cycle
        if cycle is None:
            controller = self._metacognitive_controller
            if controller is not None:
                latest_trace = controller.get_latest_trace(resolved_session_id)
                if latest_trace is not None:
                    return [dict(goal) for goal in (latest_trace.get("ranked_goals") or []) if isinstance(goal, dict)]
            return []
        return [self._serialize_cycle_value(goal) for goal in cycle.ranked_goals]

    def get_metacognitive_scorecard(self, session_id: Optional[str] = None, *, limit: int = 50) -> Dict[str, Any]:
        resolved_session_id = session_id or self.session_id
        controller = self._metacognitive_controller
        if controller is not None:
            return controller.build_scorecard(resolved_session_id, limit=limit)
        return {
            "available": False,
            "session_id": resolved_session_id,
            "trace_count": 0,
            "summary": {},
            "contradictions": {},
            "self_model": {},
            "goals": {},
        }

    def _run_metacognitive_cycle(
        self,
        input_data: str,
        *,
        input_type: str = "text",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        controller = self._metacognitive_controller
        if controller is None:
            return None
        from .metacognition import Stimulus

        cycle = controller.run_cycle(
            Stimulus(
                session_id=session_id or self.session_id,
                user_input=input_data,
                input_type=input_type,
                turn_index=self._turn_counter,
                metadata=dict(metadata or {}),
            )
        )
        self._last_metacognitive_cycle = cycle
        return cycle

    def _summarize_cycle(self, cycle: Any, *, session_id: str) -> Dict[str, Any]:
        plan = cycle.plan
        critic_report = cycle.critic_report
        execution_result = cycle.execution_result
        selected_goal = plan.selected_goal if plan is not None else None
        acts = plan.acts if plan is not None else ()
        return {
            "available": True,
            "session_id": session_id,
            "cycle_id": cycle.cycle_id,
            "trace_id": cycle.trace_id,
            "selected_goal_id": selected_goal.goal_id if selected_goal is not None else None,
            "selected_goal_kind": selected_goal.kind.value if selected_goal is not None else None,
            "act_types": [act.act_type.value for act in acts],
            "success_score": critic_report.success_score if critic_report is not None else None,
            "follow_up_recommended": critic_report.follow_up_recommended if critic_report is not None else None,
            "response_text": execution_result.response_text if execution_result is not None else None,
            "scheduled_task_count": len(cycle.scheduled_tasks or ()),
        }

    def _persist_reflection_report(self, report: Dict[str, Any]) -> None:
        controller = self._metacognitive_controller
        if controller is None:
            return
        controller.persist_reflection_episode(self.session_id, report)

    def _load_persisted_reflection_reports(self, limit: int) -> List[Dict[str, Any]]:
        controller = self._metacognitive_controller
        if controller is None:
            return []
        return list(controller.list_reflection_episodes(self.session_id, limit=limit))

    def run_metacognitive_scheduler_tick(
        self,
        *,
        session_id: Optional[str] = None,
        now_ts: Optional[float] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        controller = self._metacognitive_controller
        resolved_session_id = session_id or self.session_id
        if controller is None:
            return {
                "session_id": resolved_session_id,
                "executed_count": 0,
                "executed_task_ids": [],
                "pending_task_count": 0,
            }
        result = controller.run_scheduler_tick(
            resolved_session_id,
            now_ts=now_ts,
            limit=limit,
        )
        if result.get("executed_count"):
            self._mark_metacognitive_activity(now_ts)
        return result

    def start_metacognitive_scheduler(
        self,
        *,
        interval_seconds: float = 15.0,
        limit: int = 5,
        idle_reflection_interval_seconds: float = 60.0,
    ) -> None:
        if self._background_cognition_running:
            logger.info("Metacognitive scheduler already running.")
            return

        self._background_cognition_running = True
        self._background_cognition_interval_seconds = interval_seconds
        self._background_cognition_limit = limit
        self._idle_reflection_interval_seconds = idle_reflection_interval_seconds

        def run_scheduler() -> None:
            while self._background_cognition_running:
                try:
                    tick_result = self.run_metacognitive_scheduler_tick(limit=self._background_cognition_limit)
                    if not tick_result.get("executed_count"):
                        audit_result = self.run_background_contradiction_audit()
                        if audit_result.get("enqueued_count"):
                            self._mark_metacognitive_activity()
                        else:
                            self._run_idle_reflection_if_due()
                except Exception:
                    logger.debug("Metacognitive scheduler tick failed", exc_info=True)
                time.sleep(max(0.01, float(self._background_cognition_interval_seconds or 15.0)))

        self._background_cognition_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._background_cognition_thread.start()

    def stop_metacognitive_scheduler(self) -> None:
        self._background_cognition_running = False
        self._background_cognition_thread = None

    def _mark_metacognitive_activity(self, timestamp: Optional[float] = None) -> None:
        self._last_metacognitive_activity_ts = float(timestamp if timestamp is not None else time.time())

    def run_background_contradiction_audit(
        self,
        *,
        session_id: Optional[str] = None,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        controller = self._metacognitive_controller
        resolved_session_id = session_id or self.session_id
        if controller is None:
            return {
                "session_id": resolved_session_id,
                "contradiction_count": 0,
                "enqueued_count": 0,
                "enqueued_task_ids": [],
            }
        return controller.run_contradiction_audit(resolved_session_id, now_ts=now_ts)

    def _run_idle_reflection_if_due(self, now_ts: Optional[float] = None) -> Dict[str, Any] | None:
        current_ts = float(now_ts if now_ts is not None else time.time())
        if current_ts - float(self._last_metacognitive_activity_ts) < float(self._idle_reflection_interval_seconds):
            return None
        if self._last_idle_reflection_ts and current_ts - float(self._last_idle_reflection_ts) < float(self._idle_reflection_interval_seconds):
            return None
        report = self._reflection_service.reflect(
            metadata={
                "trigger": "idle_background_scheduler",
                "idle_for_seconds": current_ts - float(self._last_metacognitive_activity_ts),
            }
        )
        self._last_idle_reflection_ts = current_ts
        return report

    @staticmethod
    def _serialize_cycle_value(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "to_dict"):
            maybe_mapping = value.to_dict()
            if isinstance(maybe_mapping, dict):
                return dict(maybe_mapping)
        return {"value": value}

    async def retrieve_memory_context(
        self,
        query: str,
        input_type: str = "text",
    ) -> List[Dict[str, Any]]:
        """Public memory-context helper for API and UI callers."""
        processed_input = {
            "raw_input": query,
            "type": input_type,
        }
        return await self._turn_processor.retrieve_memory_context(processed_input)

    async def _process_sensory_input(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """Process raw input through sensory processing module"""
        return await self._turn_processor.process_sensory_input(input_data, input_type)
    
    async def _retrieve_memory_context(self, processed_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for context building.
        Includes proactive recall based on the recent conversation context.
        """
        return await self._turn_processor.retrieve_memory_context(processed_input)

    async def _calculate_attention_allocation(
        self, 
        processed_input: Dict[str, Any], 
        memory_context: List[Dict[str, Any]]    ) -> Dict[str, float]:
        """Calculate attention scores using AttentionMechanism"""
        return await self._turn_processor.calculate_attention_allocation(processed_input, memory_context)
    
    async def _generate_response(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        attention_scores: Dict[str, float]
    ) -> str:
        """Generate response using LLM with cognitive context"""
        return await self._turn_processor.generate_response(processed_input, memory_context, attention_scores)

    async def _call_llm_chat(self, messages):
        """Call LLM provider chat completion API asynchronously."""
        return await self._llm_session.call_chat(messages)
    
    async def _call_openai_chat(self, messages):
        """Legacy method - redirects to _call_llm_chat for backward compatibility."""
        return await self._call_llm_chat(messages)

    def set_system_prompt(self, prompt: str):
        """Set a new system prompt for the agent."""
        self._llm_session.set_system_prompt(prompt)

    def reset_llm_conversation(self):
        """Clear the LLM conversation history."""
        self._llm_session.reset_conversation()

    def reconfigure_llm_provider(
        self,
        *,
        provider: str,
        openai_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._llm_session.reconfigure_provider(
            provider=provider,
            openai_model=openai_model,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )

    def get_response_policy_state(self) -> Optional[Dict[str, Any]]:
        return self._cognitive_layers.get_response_policy_state()
    
    async def _consolidate_memory(
        self,
        input_data: str,
        response: str,
        attention_scores: Dict[str, float]
    ):
        """Consolidate the current interaction into memory and update context."""
        await self._turn_processor.consolidate_memory(input_data, response, attention_scores)

    def store_fact(self, subject: str, predicate: str, object: str):
        """
        Stores a structured fact (subject-predicate-object triple) in semantic memory.

        Args:
            subject: The subject of the fact.
            predicate: The predicate of the fact.
            object: The object of the fact.
        """
        try:
            self.memory.store_fact(subject, predicate, object)
            logger.debug(f"Stored fact: ({subject}, {predicate}, {object})")
        except Exception as e:
            logger.error(f"Error storing fact: {e}")

    def find_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[tuple[str, str, str]]:
        """
        Finds facts in semantic memory matching the given components.

        Args:
            subject: The subject to search for.
            predicate: The predicate to search for.
            object: The object to search for.

        Returns:
            A list of matching facts as (subject, predicate, object) tuples.
        """
        try:
            results = self.memory.find_facts(subject, predicate, object)
            return [
                (fact["subject"], fact["predicate"], fact["object"])
                for fact in results
            ]
        except Exception as e:
            logger.error(f"Error finding facts: {e}")
            return []

    def delete_fact(self, subject: str, predicate: str, object: str) -> bool:
        """
        Deletes a specific fact from semantic memory.

        Args:
            subject: The subject of the fact to delete.
            predicate: The predicate of the fact to delete.
            object: The object of the fact to delete.

        Returns:
            True if the fact was deleted, False otherwise.
        """
        try:
            deleted = self.memory.delete_fact(subject, predicate, object)
            if deleted:
                logger.debug(f"Deleted fact: ({subject}, {predicate}, {object})")
            else:
                logger.debug(f"Fact not found for deletion: ({subject}, {predicate}, {object})")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting fact: {e}")
            return False

    def _update_cognitive_state(self, attention_scores: Dict[str, float]):
        """Update the agent's internal cognitive state"""
        self._turn_processor.update_cognitive_state(attention_scores)
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive state information with error handling"""
        return self._maintenance_service.get_cognitive_status()
    
    async def enter_dream_state(self, cycle_type: str = "deep"):
        """Enter dream-state processing for memory consolidation"""
        return await self._maintenance_service.enter_dream_state(cycle_type)
    
    def take_cognitive_break(self, duration_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Take a brief cognitive break to recover attention and reduce fatigue
        
        Args:
            duration_minutes: Duration of break in minutes
        
        Returns:
            Break recovery metrics
        """
        return self._maintenance_service.take_cognitive_break(duration_minutes)
    
    def force_dream_cycle(self, cycle_type: str = "deep"):
        """Force an immediate dream cycle"""
        self._maintenance_service.force_dream_cycle(cycle_type)
    
    def get_dream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dream processing statistics"""
        return self._maintenance_service.get_dream_statistics()
    
    def is_dreaming(self) -> bool:
        """Check if the agent is currently in a dream state"""
        return self._maintenance_service.is_dreaming()
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive agent"""
        await self._maintenance_service.shutdown()
        if self.neural_integration is not None:
            try:
                self.neural_integration.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down neural integration: %s", exc)
        self.stop_metacognitive_scheduler()
        self.stop_reflection_scheduler()
    
    def reflect(self) -> Dict[str, Any]:
        """
        Perform metacognitive reflection across all memory systems.
        Aggregates stats and health reports, stores the result.
        Returns the reflection report.
        """
        return self._reflection_service.reflect()

    def get_reflection_reports(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last n reflection reports."""
        return self._reflection_service.get_reports(n)

    def start_reflection_scheduler(self, interval_minutes: int = 10):
        """
        Start a background thread to periodically call reflect().
        Uses the 'schedule' library for timing.
        """
        self._reflection_service.start_scheduler(interval_minutes)

    def stop_reflection_scheduler(self):
        """
        Stop the background reflection scheduler.
        """
        self._reflection_service.stop_scheduler()

    def manual_reflect(self) -> Dict[str, Any]:
        """
        Manually trigger a metacognitive reflection (CLI/API hook).
        Returns the reflection report.
        """
        logger.info("Manual metacognitive reflection triggered.")
        return self.reflect()

    def get_reflection_status(self) -> dict:
        """Return current reflection scheduler status and interval."""
        return self._reflection_service.get_status()

    def get_last_reflection_report(self) -> dict:
        """Return the most recent reflection report, or None."""
        return self._reflection_service.get_last_report()

    def clear_reflection_reports(self) -> None:
        """Clear stored reflection reports."""
        self._reflection_service.clear_reports()

    async def _enhance_attention_with_neural(
        self,
        processed_input: Dict[str, Any],
        attention_result: Dict[str, Any],
        base_salience: float,
        novelty: float
    ) -> Dict[str, Any]:
        """
        Enhance attention allocation using DPAD neural network predictions
        
        Args:
            processed_input: Processed sensory input
            attention_result: Base attention allocation result
            base_salience: Base salience score
            novelty: Novelty score
        
        Returns:
            Enhanced attention result
        """
        return await self._turn_processor.enhance_attention_with_neural(
            processed_input,
            attention_result,
            base_salience,
            novelty,
        )