"""
Core Cognitive Agent - Central orchestrator for the cognitive architecture
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from dotenv import load_dotenv
import threading
import time
import schedule
import logging

from .config import CognitiveConfig
from ..memory.memory_system import MemorySystem, MemorySystemConfig
from ..memory.stm.vector_stm import VectorShortTermMemory
from ..memory.ltm.vector_ltm import VectorLongTermMemory
from ..attention.attention_mechanism import AttentionMechanism
from ..processing.sensory import SensoryInterface, SensoryProcessor
from ..processing.dream import DreamProcessor
from ..optimization.performance_optimizer import PerformanceOptimizer

# Lazy import LLM provider to avoid circular dependencies
def _lazy_import_llm():
    try:
        from ..model.llm_provider import LLMProviderFactory, LLMProvider
        return LLMProviderFactory, LLMProvider
    except ImportError as e:
        logger.warning(f"Failed to import LLM provider: {e}")
        return None, None

# Load environment variables for LLM
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Default guiding system prompt for the agent
DEFAULT_SYSTEM_PROMPT = (
    "You are George, a virtual AI with human-like cognition. "
    "You act as the metacognitive layer of a human mind, orchestrating short-term, long-term, episodic, and procedural memory. "
    "Your responses should reflect self-awareness, context integration, and adaptive reasoning, as if you were a thoughtful, introspective human. "
    "Always strive for clarity, empathy, and explainability in your interactions."
)

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
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm_conversation = []  # For LLM chat history
        
        try:
            LLMProviderFactory, LLMProvider = _lazy_import_llm()
            if LLMProviderFactory is None:
                self.llm_provider = None
            else:
                self.llm_provider = LLMProviderFactory.create_from_config(self.config.llm)
                if not self.llm_provider.is_available():
                    logger.warning(f"LLM provider '{self.config.llm.provider}' is not available. LLM features may not work.")
                    self.llm_provider = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider: {e}. LLM features will not work.")
            self.llm_provider = None
        
        # Backward compatibility: Keep openai_client reference for legacy code
        self.openai_client = getattr(self.llm_provider, 'client', None) if hasattr(self.llm_provider, 'client') else None
        
        # Reflection state
        self.reflection_reports: List[Dict[str, Any]] = []
        self._reflection_scheduler_thread = None
        self._reflection_scheduler_running = False
        
        logger.info(f"Cognitive agent initialized with session ID: {self.session_id}")
        if self.llm_provider:
            provider_name = self.config.llm.provider
            model_name = self.config.llm.openai_model if provider_name == "openai" else self.config.llm.ollama_model
            logger.info(f"LLM provider: {provider_name} ({model_name})")
    
    def _initialize_components(self):
        """Initialize all cognitive architecture components"""
        # Memory systems - use MemorySystemConfig
        memory_config = MemorySystemConfig(
            stm_capacity=self.config.memory.stm_capacity,
            stm_decay_threshold=self.config.memory.stm_decay_threshold,
            ltm_storage_path=self.config.memory.ltm_storage_path,
            use_vector_ltm=self.config.memory.use_vector_ltm,
            use_vector_stm=self.config.memory.use_vector_stm,
            chroma_persist_dir=self.config.memory.chroma_persist_dir,
            embedding_model=self.config.processing.embedding_model,
            semantic_storage_path=self.config.memory.semantic_storage_path
        )
        self.memory = MemorySystem(memory_config)
        
        # Attention mechanism
        self.attention = AttentionMechanism(self.config.attention)
        self.sensory_processor = SensoryProcessor()
        self.sensory_interface = SensoryInterface(self.sensory_processor)

        # Neural integration manager (DPAD)
        try:
            from ..processing.neural import NeuralIntegrationManager
            self.neural_integration = NeuralIntegrationManager(
                cognitive_config=self.config,
                model_save_path="./data/models/dpad"
            )
            logger.info("Neural integration (DPAD) initialized")
        except ImportError as e:
            logger.info(f"Neural integration disabled: {e}")
            self.neural_integration = None

        # Dream processor (initialized after memory system and neural integration)
        self.dream_processor = DreamProcessor(
            memory_system=self.memory,
            enable_scheduling=True,
            consolidation_threshold=0.6,
            neural_integration_manager=self.neural_integration
        )

        # Performance optimizer (if enabled in config)
        self.performance_optimizer = None
        if self.config.performance.enabled:
            self.performance_optimizer = PerformanceOptimizer(
                config=self.config.performance
            )
            logger.info("Performance optimizer initialized")
        
        logger.info("Cognitive components initialized")
    
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
        try:
            logger.debug(f"Processing {input_type} input: {input_data[:100]}...")
            
            # Step 1: Sensory processing and filtering
            processed_input = await self._process_sensory_input(input_data, input_type)
            
            # Step 2: Memory retrieval and context building
            memory_context = await self._retrieve_memory_context(processed_input)
              # Step 3: Attention allocation
            attention_scores = await self._calculate_attention_allocation(processed_input, memory_context)
            
            # Step 4: Response generation (placeholder for LLM integration)
            response = await self._generate_response(processed_input, memory_context, attention_scores)
            
            # Step 5: Memory consolidation
            await self._consolidate_memory(input_data, response, attention_scores)
            
            # Step 6: Update cognitive state
            self._update_cognitive_state(attention_scores)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in cognitive processing: {e}")
            return "I encountered an error while processing your request. Please try again."
    async def _process_sensory_input(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """Process raw input through sensory processing module"""
        try:
            # Use sensory interface to process the input
            processed_sensory_data = self.sensory_interface.process_user_input(input_data)
            
            # Convert to expected format for cognitive processing
            return {
                "raw_input": input_data,
                "type": input_type,
                "processed_at": datetime.now(),
                "entropy_score": processed_sensory_data.entropy_score,
                "salience_score": processed_sensory_data.salience_score,
                "relevance_score": processed_sensory_data.relevance_score,
                "embedding": processed_sensory_data.embedding,
                "filtered": processed_sensory_data.filtered,
                "processing_metadata": processed_sensory_data.processing_metadata
            }
        except Exception as e:
            logger.error(f"Error in sensory processing: {e}")
            # Fallback to basic processing
            return {
                "raw_input": input_data,
                "type": input_type,
                "processed_at": datetime.now(),
                "entropy_score": 0.5,  # Default fallback
                "salience_score": 0.5,
                "relevance_score": 0.5,
                "embedding": None,
                "filtered": False,
                "processing_metadata": {"error": str(e)}
            }
    
    async def _retrieve_memory_context(self, processed_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for context building.
        Includes proactive recall based on the recent conversation context.
        """
        try:
            # Proactive recall: Use recent conversation context to form a richer query
            if self.conversation_context:
                recent_interactions = [
                    f"User: {turn['user_input']}\nAI: {turn['ai_response']}"
                    for turn in self.conversation_context[-2:]  # Last 2 interactions
                ]
                proactive_query = "\n".join(recent_interactions)
                proactive_query += f"\nUser: {processed_input['raw_input']}"
            else:
                proactive_query = processed_input["raw_input"]

            logger.debug(f"Proactive memory search with query: '{proactive_query[:200]}...'")

            # Use memory system to search for relevant context
            memories = self.memory.search_memories(
                query=proactive_query,
                max_results=5
            )
            
            # Convert to expected format
            context_memories = []
            for memory_obj, relevance, source in memories:
                if source == "stm":
                    context_memories.append({
                        "id": memory_obj.id,
                        "content": memory_obj.content,
                        "source": "STM",
                        "relevance": relevance,
                        "timestamp": memory_obj.encoding_time
                    })
                elif source == "ltm":
                    context_memories.append({
                        "id": memory_obj.id,
                        "content": memory_obj.content,
                        "source": "LTM",
                        "relevance": relevance,
                        "timestamp": memory_obj.encoding_time                    })
            
            return context_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return []

    async def _calculate_attention_allocation(
        self, 
        processed_input: Dict[str, Any], 
        memory_context: List[Dict[str, Any]]    ) -> Dict[str, float]:
        """Calculate attention scores using AttentionMechanism"""
        # Calculate base salience factors using sensory processing results
        relevance = processed_input.get("relevance_score", 0.5)  # From sensory processing
        novelty = processed_input.get("entropy_score", 0.6)  # Use entropy as novelty proxy
        emotional_salience = processed_input.get("salience_score", 0.0)  # From sensory processing
        
        # Boost relevance if we found related memories
        if memory_context:
            avg_memory_relevance = sum(mem["relevance"] for mem in memory_context) / len(memory_context)
            relevance = min(1.0, relevance + (avg_memory_relevance * 0.2))
        
        # Base salience calculation
        base_salience = (relevance * 0.6) + (emotional_salience * 0.4)
        
        # Determine priority based on input characteristics
        priority = 0.5  # Default priority
        if processed_input.get("type") == "text":
            priority = 0.7  # Text gets higher priority for now
        
        # Estimate cognitive effort required
        effort_required = 0.5  # Default effort
        if len(processed_input.get("raw_input", "")) > 100:
            effort_required = 0.7  # Longer inputs require more effort
        
        # Allocate attention using the attention mechanism
        attention_result = self.attention.allocate_attention(
            stimulus_id=f"input_{datetime.now().strftime('%H%M%S_%f')}",
            content=processed_input["raw_input"],
            salience=base_salience,
            novelty=novelty,
            priority=priority,
            effort_required=effort_required        )
        
        # Neural attention enhancement via DPAD
        enhanced_attention = await self._enhance_attention_with_neural(
            processed_input, attention_result, base_salience, novelty
        )
        
        # Update attention state
        self.attention.update_attention_state()
        
        # Return comprehensive attention scores (using enhanced attention)
        return {
            "overall_attention": enhanced_attention.get("attention_score", 0.5),
            "relevance": relevance,
            "novelty": enhanced_attention.get("neural_novelty", novelty),
            "emotional_salience": emotional_salience,
            "allocated": enhanced_attention.get("allocated", False),
            "cognitive_load": enhanced_attention.get("current_load", 0.0),
            "fatigue_level": enhanced_attention.get("fatigue_level", 0.0),
            "items_in_focus": enhanced_attention.get("items_in_focus", 0),            "neural_enhanced": enhanced_attention.get("neural_enhanced", False),
            "neural_enhancement": enhanced_attention.get("neural_enhancement", 0.0)
        }
    
    async def _generate_response(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        attention_scores: Dict[str, float]
    ) -> str:
        """Generate response using LLM with cognitive context"""
        if not self.llm_provider:
            return "[LLM unavailable: No LLM provider configured.]"
        
        # Build LLM messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
        ]
        # Add memory context as assistant messages, including timestamps
        if memory_context:
            context_str = "\n".join(
                f"Memory ({m['source']}, {m['timestamp'] if 'timestamp' in m and m['timestamp'] else 'no time'}): {m['content']}"
                for m in memory_context
            )
            messages.append({"role": "assistant", "content": f"Relevant memories:\n{context_str}"})
        # Add conversation history
        messages += self.llm_conversation[-6:]  # Last 3 user/assistant pairs
        # Add user input
        user_msg = processed_input['raw_input']
        messages.append({"role": "user", "content": user_msg})
        try:
            response = await self._call_llm_chat(messages)
            # Update LLM conversation history
            self.llm_conversation.append({"role": "user", "content": user_msg})
            self.llm_conversation.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            return f"[ERROR] LLM call failed: {e}"

    async def _call_llm_chat(self, messages):
        """Call LLM provider chat completion API asynchronously."""
        if not self.llm_provider:
            raise Exception("LLM provider not initialized")
        
        llm_response = await self.llm_provider.chat_completion(
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        return llm_response.content
    
    async def _call_openai_chat(self, messages):
        """Legacy method - redirects to _call_llm_chat for backward compatibility."""
        return await self._call_llm_chat(messages)

    def set_system_prompt(self, prompt: str):
        """Set a new system prompt for the agent."""
        self.system_prompt = prompt

    def reset_llm_conversation(self):
        """Clear the LLM conversation history."""
        self.llm_conversation = []
    
    async def _consolidate_memory(
        self,
        input_data: str,
        response: str,
        attention_scores: Dict[str, float]
    ):
        """Consolidate the current interaction into memory and update context."""
        try:
            # Add to short-term memory
            interaction_content = f"User: {input_data}\nAI: {response}"
            self.memory.store_memory(
                memory_id=str(uuid.uuid4()),
                content=interaction_content,
                importance=attention_scores.get("overall_salience", 0.5)
            )

            # Update conversation context for proactive recall
            interaction_record = {
                "user_input": input_data,
                "ai_response": response,
                "timestamp": datetime.now()
            }
            self.conversation_context.append(interaction_record)
            
            # Keep context to a reasonable size (e.g., last 10 interactions)
            if len(self.conversation_context) > 10:
                self.conversation_context.pop(0)

            logger.debug("Interaction consolidated into memory and conversation context updated.")

        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")

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
        # The attention mechanism handles its own fatigue and state updates
        # Just synchronize our local fatigue with the attention mechanism
        self.current_fatigue = self.attention.current_fatigue
        
        # Update attention focus list from attention mechanism
        self.attention_focus = self.attention.get_attention_focus()
        
        logger.debug(f"Updated cognitive state - "
              f"Fatigue: {self.current_fatigue:.3f}, "
              f"Cognitive Load: {attention_scores.get('cognitive_load', 0.0):.3f}, "
              f"Items in Focus: {attention_scores.get('items_in_focus', 0)}")
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive state information with error handling"""
        try:
            # Get memory system status with fallback
            try:
                memory_status = self.memory.get_status()
            except Exception as e:
                logger.error(f"Error getting memory status: {e}")
                memory_status = {"error": str(e), "stm": {"vector_db_count": 0}, "ltm": {"vector_db_count": 0}}
            
            # Get attention mechanism status with fallback
            try:
                attention_status = self.attention.get_attention_status()
            except Exception as e:
                logger.error(f"Error getting attention status: {e}")
                attention_status = {"error": str(e), "available_capacity": 0.0, "cognitive_load": 0.0}
            
            # Get sensory processing statistics with fallback
            try:
                sensory_stats = self.sensory_processor.get_processing_stats()
            except Exception as e:
                logger.error(f"Error getting sensory stats: {e}")
                sensory_stats = {"error": str(e), "total_processed": 0, "filtered_count": 0}

            return {
                "session_id": self.session_id,
                "fatigue_level": self.current_fatigue,
                "attention_focus": self.attention_focus,
                "active_goals": self.active_goals,
                "conversation_length": len(self.conversation_context),
                "last_interaction": self.conversation_context[-1]["timestamp"] if self.conversation_context else None,
                "memory_status": memory_status,
                "attention_status": attention_status,
                "sensory_processing": sensory_stats,
                "cognitive_integration": {
                    "attention_memory_sync": len(self.attention_focus) > 0 and memory_status.get("stm", {}).get("vector_db_count", 0) > 0,
                    "processing_capacity": attention_status.get("available_capacity", 0.0),
                    "overall_efficiency": 1.0 - self.current_fatigue,
                    "sensory_efficiency": 1.0 - (sensory_stats.get("filtered_count", 0) / max(1, sensory_stats.get("total_processed", 1)))
                }
            }
        except Exception as e:
            logger.error(f"Critical error getting cognitive status: {e}")
            # Return minimal fallback status
            return {
                "session_id": self.session_id,
                "fatigue_level": self.current_fatigue,
                "attention_focus": [],
                "active_goals": [],
                "conversation_length": 0,
                "last_interaction": None,
                "memory_status": {"error": str(e)},
                "attention_status": {"error": str(e)},
                "sensory_processing": {"error": str(e)},
                "cognitive_integration": {"error": str(e)}
            }
    
    async def enter_dream_state(self, cycle_type: str = "deep"):
        """Enter dream-state processing for memory consolidation"""
        logger.info(f"Entering {cycle_type} dream state for memory consolidation...")
        
        # Use the advanced dream processor
        dream_results = await self.dream_processor.enter_dream_cycle(cycle_type)
        
        # Also allow attention to rest during dream state
        attention_rest = self.attention.rest_attention(duration_minutes=dream_results.get("actual_duration", 5))
        
        logger.debug(f"Dream state results: {dream_results}")
        logger.debug(f"Attention rest results: {attention_rest}")
        
        # Synchronize fatigue state
        self.current_fatigue = self.attention.current_fatigue
        
        logger.info("Advanced dream state processing completed")
        return dream_results
    
    def take_cognitive_break(self, duration_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Take a brief cognitive break to recover attention and reduce fatigue
        
        Args:
            duration_minutes: Duration of break in minutes
        
        Returns:
            Break recovery metrics
        """
        logger.info(f"Taking cognitive break for {duration_minutes} minutes...")
        
        # Use attention mechanism's rest functionality
        rest_results = self.attention.rest_attention(duration_minutes)
        
        # Synchronize fatigue state
        self.current_fatigue = self.attention.current_fatigue
        
        logger.info(f"Cognitive break completed. Fatigue reduced by {rest_results['fatigue_reduction']:.3f}")
        
        return {
            "break_duration": duration_minutes,
            "fatigue_before": rest_results["fatigue_reduction"] + self.current_fatigue,
            "fatigue_after": self.current_fatigue,
            "cognitive_load_reduction": rest_results["load_reduction"],
            "attention_items_lost": rest_results["items_lost_focus"],
            "recovery_effective": rest_results["fatigue_reduction"] > 0.05
        }
    
    def force_dream_cycle(self, cycle_type: str = "deep"):
        """Force an immediate dream cycle"""
        self.dream_processor.force_dream_cycle(cycle_type)
    
    def get_dream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dream processing statistics"""
        return self.dream_processor.get_dream_statistics()
    
    def is_dreaming(self) -> bool:
        """Check if the agent is currently in a dream state"""
        return self.dream_processor.is_dreaming
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive agent"""
        logger.info("Shutting down cognitive agent...")
        
        # Shutdown dream processor
        self.dream_processor.shutdown()
        
        # Save any pending memories
        # Close connections
        # Clean up resources
        
        logger.info("Cognitive agent shutdown complete")
    
    def reflect(self) -> Dict[str, Any]:
        """
        Perform metacognitive reflection across all memory systems.
        Aggregates stats and health reports, stores the result.
        Returns the reflection report.
        """
        timestamp = datetime.now().isoformat()
        ltm = self.memory.ltm
        stm = self.memory.stm
        ltm_metacognitive_stats = None
        ltm_health_report = None
        stm_metacognitive_stats = None
        
        # Get LTM reflection data
        if isinstance(ltm, VectorLongTermMemory):
            ltm_health_report = ltm.get_memory_health_report()
            
        # Get STM reflection data
        if isinstance(stm, VectorShortTermMemory):
            all_memories = stm.get_all_memories()
            stm_metacognitive_stats = {
                "capacity_utilization": len(all_memories) / stm.config.capacity,
                "error_rate": stm._error_count / max(1, stm._operation_count),
                "memory_count": len(all_memories),
                "avg_importance": sum(m.importance for m in all_memories) / max(1, len(all_memories)),
                "recent_activity": stm._operation_count
            }
        
        report = {
            "timestamp": timestamp,
            "ltm_metacognitive_stats": ltm_metacognitive_stats,
            "ltm_health_report": ltm_health_report,
            "stm_metacognitive_stats": stm_metacognitive_stats,
            "ltm_status": ltm.get_status() if hasattr(ltm, 'get_status') else None,
            "stm_status": stm.get_status() if hasattr(stm, 'get_status') else None,
            # Add more systems as needed
        }
        self.reflection_reports.append(report)
        return report

    def get_reflection_reports(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last n reflection reports."""
        return self.reflection_reports[-n:]

    def start_reflection_scheduler(self, interval_minutes: int = 10):
        """
        Start a background thread to periodically call reflect().
        Uses the 'schedule' library for timing.
        """
        if self._reflection_scheduler_running:
            logger.info("Reflection scheduler already running.")
            return
        self._reflection_scheduler_running = True
        schedule.clear('reflection')
        schedule.every(interval_minutes).minutes.do(self.reflect).tag('reflection')
        def run_scheduler():
            while self._reflection_scheduler_running:
                schedule.run_pending()
                time.sleep(1)
        self._reflection_scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._reflection_scheduler_thread.start()
        logger.info(f"Started metacognitive reflection scheduler (every {interval_minutes} min)")

    def stop_reflection_scheduler(self):
        """
        Stop the background reflection scheduler.
        """
        self._reflection_scheduler_running = False
        schedule.clear('reflection')
        logger.info("Stopped metacognitive reflection scheduler.")

    def manual_reflect(self) -> Dict[str, Any]:
        """
        Manually trigger a metacognitive reflection (CLI/API hook).
        Returns the reflection report.
        """
        logger.info("Manual metacognitive reflection triggered.")
        return self.reflect()

    def get_reflection_status(self) -> dict:
        """Return current reflection scheduler status and interval."""
        status = {
            "scheduler_running": getattr(self, '_reflection_scheduler_running', False),
            "interval_minutes": None
        }
        # Try to infer interval from schedule jobs
        try:
            jobs = [j for j in schedule.get_jobs('reflection')]
            if jobs:
                status["interval_minutes"] = jobs[0].interval
        except Exception:
            pass
        return status

    def get_last_reflection_report(self) -> dict:
        """Return the most recent reflection report, or None."""
        if self.reflection_reports:
            return self.reflection_reports[-1]
        return {}  # Return empty dict for type safety

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
        if not self.neural_integration:
            return attention_result  # Return original if neural integration unavailable

        try:
            # Get embedding from processed input
            if 'embedding' not in processed_input:
                return attention_result

            embedding = processed_input['embedding']

            # Import torch and numpy locally to avoid unused import warnings if neural integration is not used
            import torch
            import numpy as np

            # Convert to torch tensor if needed
            if isinstance(embedding, np.ndarray):
                embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0)  # Add batch dim
            else:
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

            # Create attention scores tensor
            attention_scores = torch.tensor([base_salience], dtype=torch.float32)

            # Process through neural network
            neural_result = await self.neural_integration.process_attention_update(
                embedding_tensor,
                attention_scores,
                salience_scores=torch.tensor([novelty], dtype=torch.float32)
            )

            if 'error' not in neural_result:
                # Extract neural predictions
                novelty_scores = neural_result.get('novelty_scores', torch.tensor([novelty]))
                processing_quality = neural_result.get('processing_quality', 1.0)

                # Enhance attention scores with neural predictions
                if len(novelty_scores) > 0:
                    enhanced_novelty = float(novelty_scores[0])

                    # Calculate enhancement factor
                    neural_enhancement = min(0.2, enhanced_novelty * 0.1)  # Cap at 20% boost

                    # Apply enhancement to attention result
                    enhanced_attention_score = min(1.0, 
                        attention_result.get("attention_score", 0.5) + neural_enhancement
                    )

                    # Update attention result
                    attention_result.update({
                        "attention_score": enhanced_attention_score,
                        "neural_enhancement": neural_enhancement,
                        "neural_novelty": enhanced_novelty,
                        "neural_processing_quality": processing_quality,
                        "neural_enhanced": True
                    })

                    logger.debug(f"Neural attention enhancement: +{neural_enhancement:.3f} "
                          f"(novelty: {enhanced_novelty:.3f})")

            return attention_result

        except Exception as e:
            logger.warning(f"Neural attention enhancement error: {e}")
            return attention_result  # Return original on error