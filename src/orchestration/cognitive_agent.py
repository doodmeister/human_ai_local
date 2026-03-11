"""
Core Cognitive Agent - Central orchestrator for the cognitive architecture
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import logging

from ..core.config import CognitiveConfig
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
        
        # Initialize cognitive components
        self._initialize_components()
        
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
        self._reflection_service = CognitiveReflectionService(get_memory=lambda: self.memory)
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
            get_session_id=lambda: self.session_id,
            get_sensory_interface=lambda: self.sensory_interface,
            get_memory=lambda: self.memory,
            get_attention=lambda: self.attention,
            get_llm_session=lambda: self._llm_session,
            get_conversation_context=lambda: self.conversation_context,
            get_neural_integration=lambda: self.neural_integration,
            get_current_fatigue=lambda: self.current_fatigue,
            set_current_fatigue=self._set_current_fatigue,
            set_attention_focus=self._set_attention_focus,
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

    @property
    def llm_provider(self) -> Any:
        return self._llm_session.provider

    @property
    def openai_client(self) -> Any:
        return self._llm_session.openai_client

    @property
    def reflection_reports(self) -> List[Dict[str, Any]]:
        return self._reflection_service.reports

    def _set_current_fatigue(self, value: float) -> None:
        self.current_fatigue = value

    def _set_attention_focus(self, value: List[Any]) -> None:
        self.attention_focus = value
    
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
        return await self._turn_processor.process_input(input_data, input_type, context)

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