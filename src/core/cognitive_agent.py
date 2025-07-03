"""
Core Cognitive Agent - Central orchestrator for the cognitive architecture
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
import threading
import time
import asyncio
from abc import ABC, abstractmethod

from .config import CognitiveConfig
from ..memory.memory_system import MemorySystem
from ..attention.attention_mechanism import AttentionMechanism
from ..processing.sensory import SensoryInterface, SensoryProcessor
from ..processing.dream import DreamProcessor
from ..optimization.performance_optimizer import PerformanceOptimizer


# Load environment variables for LLM
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")

# Set up logging
logger = logging.getLogger("cognitive_agent")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Default guiding system prompt for the agent
DEFAULT_SYSTEM_PROMPT = (
    "You are George, a virtual AI with human-like cognition. "
    "You act as the metacognitive layer of a human mind, orchestrating short-term, long-term, episodic, and procedural memory. "
    "Your responses should reflect self-awareness, context integration, and adaptive reasoning, as if you were a thoughtful, introspective human. "
    "Always strive for clarity, empathy, and explainability in your interactions."
)

# Constants for cognitive state management
DEFAULT_FATIGUE_INCREASE_RATE = 0.01
DEFAULT_MAX_ACTIVE_GOALS = 3

class LLMInterface(ABC):
    """
    Abstract interface for language models. Allows easy swapping of LLM backends.
    """
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        pass

class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model_name: str, system_prompt: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        full_prompt = f"{context}\n\nUser query: {prompt}" if context else prompt
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=kwargs.get("max_tokens", 1500),
            n=1,
            stop=None,
            temperature=kwargs.get("temperature", 0.7),
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

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
        
        # LLM configuration - Initialize LLM interface
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm_conversation = []  # For LLM chat history

        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set. LLM features will not work.")
            self.llm = None
        else:
            self.llm = OpenAILLM(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME, system_prompt=self.system_prompt)
        
        # Reflection state
        self.reflection_reports: List[Dict[str, Any]] = []
        self._reflection_scheduler_thread = None
        self._reflection_scheduler_running = False
        
        logger.info(f"Cognitive agent initialized with session ID: {self.session_id}")
    
    def _initialize_components(self):
        """Initialize all cognitive architecture components"""
        # Memory systems
        self.memory = MemorySystem(
            stm_capacity=self.config.memory.stm_capacity,
            stm_decay_threshold=self.config.memory.stm_decay_threshold,
            ltm_storage_path=self.config.memory.ltm_storage_path,
            use_vector_ltm=self.config.memory.use_vector_ltm,
            use_vector_stm=self.config.memory.use_vector_stm,
            chroma_persist_dir=self.config.memory.chroma_persist_dir,
            embedding_model=self.config.processing.embedding_model,
            semantic_storage_path=self.config.memory.semantic_storage_path
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(
            max_attention_items=self.config.attention.max_attention_items,
            salience_threshold=self.config.attention.salience_threshold,
            fatigue_decay_rate=self.config.attention.fatigue_decay_rate,
            attention_recovery_rate=self.config.attention.attention_recovery_rate
        )
        # Sensory processing interface
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
            logger.warning(f"Neural integration disabled: {e}")
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
            logger.info(f"Processing {input_type} input: {input_data[:100]}...")
            
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
        Retrieve relevant memories for context building using a hierarchical search.
        Search order: STM -> LTM -> Episodic -> Semantic.
        """
        try:
            if self.conversation_context:
                recent_interactions = [
                    f"User: {turn['user_input']}\nAI: {turn['ai_response']}"
                    for turn in self.conversation_context[-2:]
                ]
                proactive_query = "\n".join(recent_interactions)
                proactive_query += f"\nUser: {processed_input['raw_input']}"
            else:
                proactive_query = processed_input["raw_input"]

            logger.info(f"Hierarchical memory search with query: '{proactive_query[:200]}...'")

            # Hierarchical search
            memory_context = self.memory.hierarchical_search(
                query=proactive_query,
                max_results=5
            )

            # Convert to expected format
            context_memories = []
            for memory_obj, relevance, source in memory_context:
                memory_item = {
                    "source": source,
                    "relevance": relevance,
                    "embedding": None  # Default to None
                }
                if source in ["STM", "LTM"]:
                    memory_item.update({
                        "id": memory_obj.id,
                        "content": memory_obj.content,
                        "timestamp": memory_obj.encoding_time,
                        "embedding": memory_obj.embedding
                    })
                elif source == "Episodic":
                    memory_item.update({
                        "id": memory_obj.id,
                        "content": memory_obj.summary,
                        "timestamp": memory_obj.timestamp,
                        "embedding": memory_obj.embedding
                    })
                elif source == "Semantic":
                    # Semantic memories might not have embeddings unless explicitly created
                    memory_item.update({
                        "id": memory_obj.get('id', str(uuid.uuid4())),
                        "content": f"Fact: {memory_obj['subject']} {memory_obj['predicate']} {memory_obj['object']}",
                        "timestamp": memory_obj.get('timestamp', datetime.now()),
                        "embedding": memory_obj.get('embedding') 
                    })
                context_memories.append(memory_item)

            return context_memories

        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return []

    async def _calculate_attention_allocation(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate attention allocation scores for cognitive items based on relevance.
        
        Args:
            processed_input: Processed input data (not directly used but available for future enhancement)
            memory_context: Contextual memory items with relevance scores.
        
        Returns:
            A dictionary mapping memory item IDs to their attention scores.
        """
        try:
            if not memory_context:
                return {}

            # Use relevance scores from memory search as the basis for attention
            attention_scores = {
                item["id"]: item["relevance"]
                for item in memory_context if "id" in item and "relevance" in item
            }

            # Normalize scores to a 0-1 range if needed, though not strictly necessary here
            max_score = max(attention_scores.values()) if attention_scores else 1.0
            if max_score > 0:
                normalized_scores = {k: v / max_score for k, v in attention_scores.items()}
            else:
                normalized_scores = attention_scores

            # Select top N items for focused processing
            sorted_items = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
            top_attention_items = dict(sorted_items[:self.config.attention.max_attention_items])
            
            return top_attention_items
        
        except Exception as e:
            logger.error(f"Error calculating attention allocation: {e}")
            return {}

    async def _generate_response(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        attention_scores: Dict[str, float]
    ) -> str:
        """
        Generate response based on processed input, memory context, and attention scores
        
        Args:
            processed_input: Processed input data
            memory_context: Contextual memory items
            attention_scores: Attention allocation scores
        
        Returns:
            Generated response text
        """
        try:
            # Select top memories based on attention scores
            if attention_scores:
                # Fix: sort items by value, extract keys
                top_memory_ids = [k for k, v in sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)[:3]]
                top_memories = [
                    next((mem for mem in memory_context if mem["id"] == mid), None)
                    for mid in top_memory_ids
                ]
                # Filter out None results if a memory ID wasn't found
                top_memories = [mem for mem in top_memories if mem]
            else:
                top_memories = []

            # Format context for the LLM
            context_str = "\n".join(
                [f"- [Memory from {mem['source']}]: {mem['content']} (Relevance: {mem['relevance']:.2f})" for mem in top_memories]
            )

            # Call OpenAI API (or other LLM) for response generation
            response = await self._call_llm(processed_input["raw_input"], context_str)
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I am unable to generate a response at the moment."

    async def _call_llm(self, user_input: str, context_str: str) -> str:
        """
        Call the LLM interface to generate a response.
        """
        try:
            if not self.llm:
                logger.warning("LLM interface not initialized. Cannot generate response.")
                return "LLM is not available. Please set the OPENAI_API_KEY environment variable."
            llm = self.llm  # type: ignore
            loop = asyncio.get_event_loop()
            def call_llm():
                return llm.generate_response(user_input, context=context_str)
            response = await loop.run_in_executor(None, call_llm)
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error generating response from language model."

    async def _consolidate_memory(
        self,
        input_data: str,
        response: str,
        attention_scores: Dict[str, float]
    ):
        """
        Consolidates the current interaction into a new episodic memory.
        The significance of the memory is determined by the attention scores.
        
        Args:
            input_data: Original input data from the user.
            response: Generated response from the agent.
            attention_scores: Attention allocation scores for retrieved memories.
        """
        try:
            # Determine the significance of the interaction
            # Use the average attention score of related memories as a proxy
            significance = sum(attention_scores.values()) / len(attention_scores) if attention_scores else 0.1
            
            # Only consolidate if the interaction is significant enough
            consolidation_threshold = 0.2  # Configurable threshold
            if significance < consolidation_threshold:
                logger.info(f"Interaction significance ({significance:.2f}) below threshold. Skipping consolidation.")
                return

            # Create a new episodic memory using the correct API
            interaction_summary = f"User said: '{input_data}'. I responded: '{response}'."
            # Use the episodic memory system's API (detailed_content, importance)
            if hasattr(self.memory, 'episodic') and hasattr(self.memory.episodic, 'store_memory'):
                self.memory.episodic.store_memory(
                    detailed_content=interaction_summary,
                    importance=significance
                )
                logger.info(f"Consolidated interaction as new episodic memory with significance {significance:.2f}")
            else:
                logger.warning("Episodic memory system not available. Skipping consolidation.")
        
        except Exception as e:
            logger.error(f"Error consolidating memory: {e}")

    def _update_cognitive_state(self, attention_scores: Dict[str, float]):
        """
        Update the cognitive state based on attention allocation and processing results
        
        Args:
            attention_scores: Attention allocation scores
        """
        try:
            # Update attention focus
            self.attention_focus = list(attention_scores.keys())
            
            # Adjust fatigue based on attention demand
            self.current_fatigue += sum(attention_scores.values()) * self.config.agent.fatigue_increase_rate
            self.current_fatigue = min(self.current_fatigue, 1.0)  # Cap at 1.0
            
            # Update active goals (top N based on attention scores)
            self.active_goals = sorted(attention_scores.keys(), key=lambda x: attention_scores[x], reverse=True)[:self.config.agent.max_active_goals]
            
            logger.info(f"Updated cognitive state: fatigue={self.current_fatigue:.2f}, active_goals={self.active_goals}")
        
        except Exception as e:
            logger.error(f"Error updating cognitive state: {e}")

    def set_system_prompt(self, prompt: str):
        """
        Set a new system prompt for the agent
        
        Args:
            prompt: New system prompt text
        """
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def add_to_conversation_context(self, user_input: str, ai_response: str):
        """
        Add an interaction turn to the conversation context for proactive memory and context
        
        Args:
            user_input: User's input text
            ai_response: AI's response text
        """
        try:
            # Limit context size
            if len(self.conversation_context) > self.config.agent.max_context_turns:
                self.conversation_context.pop(0)
            
            # Add new turn
            self.conversation_context.append({
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": datetime.now()
            })
            
            logger.info("Added to conversation context")
        
        except Exception as e:
            logger.error(f"Error adding to conversation context: {e}")

    def start_reflection_scheduler(self, interval_minutes: int = 60):
        """
        Start a scheduler for periodic reflection and self-improvement
        
        Args:
            interval_minutes: Interval in minutes for reflection cycles
        """
        if self._reflection_scheduler_running:
            logger.info("Reflection scheduler is already running")
            return
        
        def reflection_task():
            while self._reflection_scheduler_running:
                try:
                    logger.info("Starting reflection cycle")
                    self.reflect_on_experience()
                    time.sleep(interval_minutes * 60)  # Convert to seconds
                except Exception as e:
                    logger.error(f"Error in reflection task: {e}")
                    time.sleep(10)  # Wait before retrying
        
        self._reflection_scheduler_running = True
        self._reflection_scheduler_thread = threading.Thread(target=reflection_task)
        self._reflection_scheduler_thread.start()
        logger.info(f"Reflection scheduler started (every {interval_minutes} minutes)")

    def stop_reflection_scheduler(self):
        """Stop the reflection scheduler"""
        self._reflection_scheduler_running = False
        if self._reflection_scheduler_thread:
            self._reflection_scheduler_thread.join()
            self._reflection_scheduler_thread = None
        logger.info("Reflection scheduler stopped")

    def reflect_on_experience(self):
        """
        Reflect on past experiences and interactions to improve future responses and behavior
        """
        try:
            logger.info("Compiling reflection report")
            report = {
                "session_id": self.session_id,
                "timestamp": datetime.now(),
                "fatigue_level": self.current_fatigue,
                "active_goals": self.active_goals,
                "conversation_context": self.conversation_context[-3:],  # Last 3 turns
            }
            
            # Add to reflection reports
            self.reflection_reports.append(report)
            
            # Upload to LLM for analysis (optional)
            if self.llm:
                llm = self.llm  # type: ignore
                loop = asyncio.get_event_loop()
                def call_llm():
                    return llm.generate_response(
                        prompt=str(report),
                        context="You are an AI analyzing the reflection report of another AI."
                    )
                future = loop.run_in_executor(None, call_llm)
                # Await the result if running in async context
                if asyncio.iscoroutine(future):
                    import nest_asyncio
                    nest_asyncio.apply()
                    analysis = loop.run_until_complete(future)
                else:
                    analysis = loop.run_until_complete(future)
                logger.info(f"Reflection analysis: {analysis}")
                self._apply_reflection_insights(analysis)
            else:
                logger.warning("No LLM interface available for reflection analysis.")
                # Optionally, skip applying insights or use a fallback
            logger.info("Reflection cycle complete")
        
        except Exception as e:
            logger.error(f"Error during reflection: {e}")

    def _apply_reflection_insights(self, insights: str):
        """
        Apply insights from reflection analysis to improve cognitive agent behavior and responses
        
        Args:
            insights: Analysis insights text
        """
        try:
            # Placeholder: Adjust parameters or strategies based on insights
            if "reduce fatigue" in insights:
                self.config.agent.fatigue_increase_rate *= 0.9  # Reduce fatigue increase rate
            if "increase focus" in insights:
                self.config.attention.max_attention_items = min(
                    self.config.attention.max_attention_items + 1,
                    12  # Set a reasonable hard limit
                )
            logger.info("Applied reflection insights")
        
        except Exception as e:
            logger.error(f"Error applying reflection insights: {e}")

    def save_state(self):
        """
        Save the current state of the cognitive agent (memories, configuration, etc.) to persistent storage
        """
        try:
            logger.warning("save_state is not implemented: memory.save_all and config serialization are not available.")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self):
        """
        Load the cognitive agent's state from persistent storage
        """
        try:
            logger.warning("load_state is not implemented: memory.load_all and config deserialization are not available.")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def clear_conversation_context(self):
        """Clear the conversation context"""
        self.conversation_context = []
        logger.info("Conversation context cleared")

    def set_fatigue_level(self, level: float):
        """
        Set the cognitive fatigue level (0.0 to 1.0)
        
        Args:
            level: Fatigue level (0.0 = no fatigue, 1.0 = fully fatigued)
        """
        self.current_fatigue = max(0.0, min(level, 1.0))
        logger.info(f"Fatigue level set to {self.current_fatigue}")

    def adjust_goal_priority(self, goal_id: str, priority: float):
        """
        Adjust the priority of an active goal
        
        Args:
            goal_id: ID of the goal
            priority: New priority level (higher values = higher priority)
        """
        try:
            if goal_id in self.active_goals:
                # Placeholder: Adjust based on priority (e.g., re-order goals)
                self.active_goals.remove(goal_id)
                self.active_goals.insert(0, goal_id)
                logger.info(f"Goal {goal_id} priority adjusted")
            else:
                logger.warning(f"Goal {goal_id} not found in active goals")
        
        except Exception as e:
            logger.error(f"Error adjusting goal priority: {e}")

    def set_memory_retrieval_strategy(self, strategy: str):
        logger.warning("set_memory_retrieval_strategy is not implemented: retrieval_strategy is not a config attribute.")

    def update_config(self, section: str, updates: dict):
        """
        Update a section of the agent's configuration.
        Args:
            section: The config section name (e.g., 'memory', 'agent', 'attention', 'processing')
            updates: Dict of key-value pairs to update in that section
        """
        config_section = getattr(self.config, section, None)
        if not config_section:
            logger.warning(f"No config section named '{section}'")
            return
        for key, value in updates.items():
            setattr(config_section, key, value)
        logger.info(f"Updated {section} config: {updates}")

    # Remove any unreachable code that still tries to set non-existent config attributes
    # (These lines are now unreachable due to the stubbed methods above)
    def set_reflection_config(self, **kwargs):
        logger.warning("set_reflection_config is not implemented: reflection config is not present.")

    def set_scheduler_config(self, **kwargs):
        logger.warning("set_scheduler_config is not implemented: scheduler config is not present.")

    def set_storage_config(self, **kwargs):
        logger.warning("set_storage_config is not implemented: storage config is not present.")

    def update_reflection_params(self, **kwargs):
        logger.warning("update_reflection_params is not implemented: reflection config is not present.")

    def update_scheduler_params(self, **kwargs):
        logger.warning("update_scheduler_params is not implemented: scheduler config is not present.")

    def update_storage_params(self, **kwargs):
        logger.warning("update_storage_params is not implemented: storage config is not present.")
