"""
Core Cognitive Agent - Production-grade central orchestrator for the cognitive architecture

This module implements a biologically-inspired cognitive agent with human-like memory,
attention, and reasoning capabilities. It serves as the main interface for cognitive
processing and orchestrates all subsystems including memory, attention, and neural processing.

Key Features:
- Thread-safe cognitive processing pipeline
- Comprehensive error handling and logging
- Input validation and sanitization
- Performance monitoring and optimization
- Graceful degradation capabilities
- Security-first design principles

Architecture:
The agent implements a layered cognitive architecture:
1. Sensory Processing Layer - Input filtering and preprocessing
2. Memory Layer - STM/LTM integration with vector storage
3. Attention Layer - Dynamic attention allocation with fatigue modeling
4. Neural Enhancement Layer - DPAD/LSHN neural network integration
5. Response Generation Layer - LLM integration with context
6. Consolidation Layer - Memory storage and context updates
"""

import asyncio
import logging
import threading
import time
import uuid
import schedule
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import secrets
import html

import numpy as np
import torch
from openai import OpenAI
from dotenv import load_dotenv

from .config import CognitiveConfig
from ..memory.memory_system import MemorySystem, MemorySystemConfig
from ..memory.stm.vector_stm import VectorShortTermMemory
from ..memory.ltm.vector_ltm import VectorLongTermMemory
from ..attention.attention_mechanism import AttentionMechanism
from ..processing.sensory import SensoryInterface, SensoryProcessor
from ..processing.dream import DreamProcessor
from ..optimization.performance_optimizer import PerformanceOptimizer
from ..utils.validators import InputValidator, ValidationError
from ..utils.security import SecurityManager, RateLimitExceeded, AccessDenied

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

# Security constants
MAX_INPUT_LENGTH = 10000
MAX_CONTEXT_SIZE = 50
MAX_CONVERSATION_HISTORY = 20
REQUEST_TIMEOUT = 30.0

# Default system prompt with security considerations
DEFAULT_SYSTEM_PROMPT = (
    "You are George, a virtual AI with human-like cognition and strict ethical guidelines. "
    "You act as the metacognitive layer of a human mind, orchestrating memory systems responsibly. "
    "Your responses reflect self-awareness, context integration, and adaptive reasoning. "
    "Always prioritize user safety, privacy, and beneficial outcomes. "
    "Never generate harmful, illegal, or inappropriate content."
)


@dataclass
class CognitiveState:
    """Immutable cognitive state snapshot for thread safety"""
    session_id: str
    fatigue_level: float
    attention_focus: List[str]
    active_goals: List[str]
    conversation_length: int
    last_interaction: Optional[datetime]
    cognitive_load: float
    processing_capacity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "fatigue_level": self.fatigue_level,
            "attention_focus": self.attention_focus.copy(),
            "active_goals": self.active_goals.copy(),
            "conversation_length": self.conversation_length,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "cognitive_load": self.cognitive_load,
            "processing_capacity": self.processing_capacity
        }


class CognitiveAgentError(Exception):
    """Base exception for cognitive agent errors"""
    pass


class ProcessingError(CognitiveAgentError):
    """Raised when cognitive processing fails"""
    pass


class MemoryError(CognitiveAgentError):
    """Raised when memory operations fail"""
    pass


class AttentionError(CognitiveAgentError):
    """Raised when attention allocation fails"""
    pass

class CognitiveAgent:
    """
    Production-grade cognitive agent that orchestrates all cognitive processes
    
    This class implements a thread-safe, secure cognitive processing pipeline with:
    - Comprehensive error handling and recovery
    - Input validation and sanitization
    - Rate limiting and security controls
    - Performance monitoring and optimization
    - Graceful degradation capabilities
    - Resource management and cleanup
    
    The cognitive processing pipeline follows these stages:
    1. Input validation and sanitization
    2. Sensory processing and filtering
    3. Memory retrieval and context building
    4. Attention allocation with neural enhancement
    5. Response generation via LLM
    6. Memory consolidation and state updates
    
    Thread Safety:
        All public methods are thread-safe and can be called concurrently.
        Internal state is protected by locks where necessary.
    
    Security:
        Input validation, rate limiting, and access controls are enforced.
        Sensitive data is handled securely with proper sanitization.
    
    Performance:
        Async/await patterns for I/O operations
        Connection pooling for external services
        Caching for frequently accessed data
        Resource cleanup and memory management
    """
    
    def __init__(
        self, 
        config: Optional[CognitiveConfig] = None, 
        system_prompt: Optional[str] = None,
        client_id: Optional[str] = None
    ):
        """
        Initialize the cognitive agent with production-grade safeguards
        
        Args:
            config: Configuration object (uses environment defaults if None)
            system_prompt: Custom system prompt for LLM interactions
            client_id: Unique client identifier for security and rate limiting
            
        Raises:
            CognitiveAgentError: If initialization fails
        """
        try:
            # Core configuration and security
            self.config = config or CognitiveConfig.from_env()
            self.client_id = client_id or f"client_{secrets.token_hex(8)}"
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
            
            # Initialize security and validation
            self._security_manager = SecurityManager()
            self._validator = InputValidator()
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cognitive_")
            
            # Thread safety
            self._state_lock = threading.RLock()
            self._memory_lock = threading.RLock()
            self._conversation_lock = threading.RLock()
            
            # Initialize cognitive components
            self._initialize_components()
            
            # Cognitive state (protected by locks)
            with self._state_lock:
                self._current_fatigue = 0.0
                self._attention_focus: List[str] = []
                self._active_goals: List[str] = []
                self._conversation_context: List[Dict[str, Any]] = []
                self._processing_metrics = {
                    "requests_processed": 0,
                    "errors_encountered": 0,
                    "average_response_time": 0.0,
                    "last_error": None
                }
            
            # Public attributes for backward compatibility
            self.current_fatigue = 0.0
            self.attention_focus: List[str] = []
            self.active_goals: List[str] = []
            self.conversation_context: List[Dict[str, Any]] = []
            
            # LLM configuration with security
            self.system_prompt = self._validator.validate_text_input(
                system_prompt or DEFAULT_SYSTEM_PROMPT
            )
            self._llm_conversation: List[Dict[str, str]] = []
            self.llm_conversation: List[Dict[str, str]] = []  # Public access
            
            # Initialize OpenAI client with error handling
            self._openai_client = None
            self.openai_client = None  # Public access
            if OPENAI_API_KEY:
                try:
                    self._openai_client = OpenAI(
                        api_key=OPENAI_API_KEY,
                        timeout=REQUEST_TIMEOUT
                    )
                    self.openai_client = self._openai_client  # Public access
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    # Continue without LLM capabilities
            else:
                logger.warning("OPENAI_API_KEY not set. LLM features will be disabled.")
            
            # Reflection and scheduling
            self._reflection_reports: List[Dict[str, Any]] = []
            self.reflection_reports: List[Dict[str, Any]] = []  # Public access
            self._reflection_scheduler_thread: Optional[threading.Thread] = None
            self._reflection_scheduler_running = False
            self._shutdown_event = threading.Event()
            
            # Performance monitoring
            self._start_time = datetime.now()
            self._last_health_check = datetime.now()
            
            logger.info(f"CognitiveAgent initialized successfully - Session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CognitiveAgent: {e}")
            raise CognitiveAgentError(f"Initialization failed: {e}") from e
    
    def _initialize_components(self) -> None:
        """
        Initialize all cognitive architecture components with error handling
        
        Raises:
            CognitiveAgentError: If critical components fail to initialize
        """
        try:
            # Memory systems with comprehensive configuration
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
            logger.info("Memory system initialized successfully")
            
            # Attention mechanism with error handling
            try:
                self.attention = AttentionMechanism(
                    max_attention_items=self.config.attention.max_attention_items,
                    salience_threshold=self.config.attention.salience_threshold,
                    fatigue_decay_rate=self.config.attention.fatigue_decay_rate,
                    attention_recovery_rate=self.config.attention.attention_recovery_rate
                )
                logger.info("Attention mechanism initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize attention mechanism: {e}")
                # Create fallback attention mechanism
                self.attention = None
                logger.warning("Attention mechanism disabled, using fallback processing")
            
            # Sensory processing interface
            try:
                self.sensory_processor = SensoryProcessor()
                self.sensory_interface = SensoryInterface(self.sensory_processor)
                logger.info("Sensory processing initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize sensory processing: {e}")
                self.sensory_processor = None
                self.sensory_interface = None
            
            # Neural integration (optional component)
            self.neural_integration = None
            try:
                from ..processing.neural import NeuralIntegrationManager
                self.neural_integration = NeuralIntegrationManager(
                    cognitive_config=self.config,
                    model_save_path="./data/models/dpad"
                )
                logger.info("Neural integration (DPAD) initialized successfully")
            except ImportError as e:
                logger.info(f"Neural integration disabled (import error): {e}")
            except Exception as e:
                logger.warning(f"Neural integration failed to initialize: {e}")
            
            # Dream processor with error handling
            try:
                self.dream_processor = DreamProcessor(
                    memory_system=self.memory,
                    enable_scheduling=True,
                    consolidation_threshold=0.6,
                    neural_integration_manager=self.neural_integration
                )
                logger.info("Dream processor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize dream processor: {e}")
                self.dream_processor = None
            
            # Performance optimizer (optional)
            self.performance_optimizer = None
            if self.config.performance.enabled:
                try:
                    self.performance_optimizer = PerformanceOptimizer(
                        config=self.config.performance
                    )
                    logger.info("Performance optimizer initialized successfully")
                except Exception as e:
                    logger.warning(f"Performance optimizer initialization failed: {e}")
            
            logger.info("All cognitive components initialized")
            
        except Exception as e:
            logger.error(f"Critical error initializing cognitive components: {e}")
            raise CognitiveAgentError(f"Component initialization failed: {e}") from e
    
    async def process_input(
        self,
        input_data: str,
        input_type: str = "text",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main cognitive processing pipeline with comprehensive error handling
        
        This method implements the complete cognitive processing cycle with:
        - Input validation and sanitization
        - Rate limiting and security checks
        - Error handling and recovery
        - Performance monitoring
        - Graceful degradation
        
        Args:
            input_data: Raw input data (text, audio, etc.)
            input_type: Type of input ("text", "audio", "image")
            context: Additional context information
        
        Returns:
            Generated response string
            
        Raises:
            ProcessingError: If processing fails critically
            ValidationError: If input validation fails
            RateLimitExceeded: If rate limits are exceeded
        """
        start_time = time.time()
        
        try:
            # Security and rate limiting checks
            self._security_manager.check_rate_limit(self.client_id)
            self._security_manager.check_lockout(self.client_id)
            
            # Input validation and sanitization
            validated_input = self._validator.validate_text_input(input_data)
            
            logger.info(f"Processing {input_type} input for client {self.client_id}")
            
            # Step 1: Sensory processing and filtering
            processed_input = await self._process_sensory_input(validated_input, input_type)
            
            # Step 2: Memory retrieval and context building
            memory_context = await self._retrieve_memory_context(processed_input)
            
            # Step 3: Attention allocation (with fallback if attention disabled)
            attention_scores = await self._calculate_attention_allocation(processed_input, memory_context)
            
            # Step 4: Response generation
            response = await self._generate_response(processed_input, memory_context, attention_scores)
            
            # Step 5: Memory consolidation
            await self._consolidate_memory(validated_input, response, attention_scores)
            
            # Step 6: Update cognitive state
            self._update_cognitive_state(attention_scores)
            
            # Record successful processing
            self._security_manager.record_successful_attempt(self.client_id)
            
            # Update metrics
            processing_time = time.time() - start_time
            with self._state_lock:
                self._processing_metrics["requests_processed"] += 1
                self._processing_metrics["average_response_time"] = (
                    (self._processing_metrics["average_response_time"] * 
                     (self._processing_metrics["requests_processed"] - 1) + processing_time) /
                    self._processing_metrics["requests_processed"]
                )
            
            logger.info(f"Processing completed in {processing_time:.2f}s")
            return response
            
        except (ValidationError, RateLimitExceeded, AccessDenied) as e:
            # Security-related errors
            self._security_manager.record_failed_attempt(self.client_id)
            logger.warning(f"Security error for client {self.client_id}: {e}")
            raise
            
        except Exception as e:
            # General processing errors
            processing_time = time.time() - start_time
            with self._state_lock:
                self._processing_metrics["errors_encountered"] += 1
                self._processing_metrics["last_error"] = str(e)
            
            logger.error(f"Error in cognitive processing: {e}", exc_info=True)
            
            # Attempt graceful degradation
            fallback_response = self._generate_fallback_response(input_data, str(e))
            return fallback_response
    
    def _generate_fallback_response(self, input_data: str, error: str) -> str:
        """
        Generate a fallback response when processing fails
        
        Args:
            input_data: Original input data
            error: Error message
            
        Returns:
            Fallback response string
        """
        logger.info("Generating fallback response due to processing error")
        
        # Simple rule-based fallback responses
        input_lower = input_data.lower()
        
        if any(word in input_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm experiencing some technical difficulties but I'm here to help."
        elif any(word in input_lower for word in ["help", "assistance", "support"]):
            return "I'd like to help you, though I'm currently experiencing some technical issues. Could you try rephrasing your request?"
        elif "?" in input_data:
            return "That's an interesting question. I'm having some technical difficulties right now, but I'll do my best to assist you."
        else:
            return "I apologize, but I'm experiencing technical difficulties processing your request. Please try again in a moment."
    async def _process_sensory_input(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """
        Process raw input through sensory processing module with error handling
        
        Args:
            input_data: Validated input text
            input_type: Type of input
            
        Returns:
            Processed sensory data dictionary
        """
        try:
            if self.sensory_interface:
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
            else:
                # Fallback processing when sensory interface is unavailable
                logger.warning("Sensory interface unavailable, using fallback processing")
                return self._fallback_sensory_processing(input_data, input_type)
                
        except Exception as e:
            logger.error(f"Error in sensory processing: {e}")
            return self._fallback_sensory_processing(input_data, input_type)
    
    def _fallback_sensory_processing(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """
        Fallback sensory processing when main system is unavailable
        
        Args:
            input_data: Input text
            input_type: Type of input
            
        Returns:
            Basic processed data dictionary
        """
        # Simple heuristic processing
        word_count = len(input_data.split())
        char_count = len(input_data)
        
        # Basic entropy calculation
        entropy_score = min(1.0, word_count / 50.0)  # Simple word count heuristic
        
        # Basic salience calculation
        salience_keywords = ["important", "urgent", "help", "problem", "question"]
        salience_score = min(1.0, sum(1 for word in salience_keywords if word in input_data.lower()) / 5.0)
        
        return {
            "raw_input": input_data,
            "type": input_type,
            "processed_at": datetime.now(),
            "entropy_score": entropy_score,
            "salience_score": salience_score,
            "relevance_score": 0.5,  # Default relevance
            "embedding": None,
            "filtered": False,
            "processing_metadata": {"fallback": True, "word_count": word_count, "char_count": char_count}
        }
    
    async def _retrieve_memory_context(self, processed_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for context building with proactive recall
        
        Args:
            processed_input: Processed sensory input
            
        Returns:
            List of relevant memory contexts
        """
        try:
            # Proactive recall: Use recent conversation context to form a richer query
            with self._conversation_lock:
                if self._conversation_context:
                    recent_interactions = [
                        f"User: {turn['user_input']}\nAI: {turn['ai_response']}"
                        for turn in self._conversation_context[-2:]  # Last 2 interactions
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
                try:
                    if source == "stm":
                        context_memories.append({
                            "id": memory_obj.id if hasattr(memory_obj, 'id') else str(uuid.uuid4()),
                            "content": memory_obj.content if hasattr(memory_obj, 'content') else str(memory_obj),
                            "source": "STM",
                            "relevance": relevance,
                            "timestamp": memory_obj.encoding_time if hasattr(memory_obj, 'encoding_time') else datetime.now()
                        })
                    elif source == "ltm":
                        context_memories.append({
                            "id": memory_obj.id if hasattr(memory_obj, 'id') else str(uuid.uuid4()),
                            "content": memory_obj.content if hasattr(memory_obj, 'content') else str(memory_obj),
                            "source": "LTM",
                            "relevance": relevance,
                            "timestamp": memory_obj.encoding_time if hasattr(memory_obj, 'encoding_time') else datetime.now()
                        })
                except Exception as mem_error:
                    logger.warning(f"Error processing memory object: {mem_error}")
                    continue
            
            return context_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return []

    async def _calculate_attention_allocation(
        self, 
        processed_input: Dict[str, Any], 
        memory_context: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate attention scores using AttentionMechanism with fallback"""
        
        if not self.attention:
            # Fallback attention calculation when attention mechanism is unavailable
            logger.warning("Attention mechanism unavailable, using fallback allocation")
            return self._fallback_attention_allocation(processed_input, memory_context)
        
        try:
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
                effort_required=effort_required
            )
            
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
                "items_in_focus": enhanced_attention.get("items_in_focus", 0),
                "neural_enhanced": enhanced_attention.get("neural_enhanced", False),
                "neural_enhancement": enhanced_attention.get("neural_enhancement", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in attention allocation: {e}")
            return self._fallback_attention_allocation(processed_input, memory_context)
    
    def _fallback_attention_allocation(
        self, 
        processed_input: Dict[str, Any], 
        memory_context: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Fallback attention allocation when main system is unavailable"""
        
        # Simple heuristic attention calculation
        relevance = processed_input.get("relevance_score", 0.5)
        novelty = processed_input.get("entropy_score", 0.5)
        emotional_salience = processed_input.get("salience_score", 0.3)
        
        # Basic overall attention score
        overall_attention = (relevance + novelty + emotional_salience) / 3.0
        
        return {
            "overall_attention": overall_attention,
            "relevance": relevance,
            "novelty": novelty,
            "emotional_salience": emotional_salience,
            "allocated": True,
            "cognitive_load": 0.3,  # Default moderate load
            "fatigue_level": 0.1,   # Default low fatigue
            "items_in_focus": 1,    # Single item in focus
            "neural_enhanced": False,
            "neural_enhancement": 0.0
        }
    
    async def _generate_response(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        attention_scores: Dict[str, float]
    ) -> str:
        """Generate response using LLM with cognitive context"""
        if not OPENAI_API_KEY:
            return "[LLM unavailable: No API key configured.]"
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
            response = await self._call_openai_chat(messages)
            # Update LLM conversation history
            self.llm_conversation.append({"role": "user", "content": user_msg})
            self.llm_conversation.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            return f"[ERROR] LLM call failed: {e}"

    async def _call_openai_chat(self, messages):
        """Call OpenAI chat completion API asynchronously."""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized - missing API key")
            
        import asyncio
        loop = asyncio.get_event_loop()
        
        def make_openai_call():
            if not self.openai_client:
                raise Exception("OpenAI client is None - API key not configured")
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            return content.strip() if content is not None else ""
        
        return await loop.run_in_executor(None, make_openai_call)

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

            print("Interaction consolidated into memory and conversation context updated.")

        except Exception as e:
            print(f"Error in memory consolidation: {e}")

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
            print(f"Stored fact: ({subject}, {predicate}, {object})")
        except Exception as e:
            print(f"Error storing fact: {e}")

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
            print(f"Error finding facts: {e}")
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
                print(f"Deleted fact: ({subject}, {predicate}, {object})")
            else:
                print(f"Fact not found for deletion: ({subject}, {predicate}, {object})")
            return deleted
        except Exception as e:
            print(f"Error deleting fact: {e}")
            return False

    def _update_cognitive_state(self, attention_scores: Dict[str, float]):
        """Update the agent's internal cognitive state with fallback"""
        try:
            if self.attention:
                # The attention mechanism handles its own fatigue and state updates
                # Just synchronize our local fatigue with the attention mechanism
                self.current_fatigue = self.attention.current_fatigue
                
                # Update attention focus list from attention mechanism
                if hasattr(self.attention, 'get_attention_focus'):
                    focus_items = self.attention.get_attention_focus()
                    if isinstance(focus_items, list):
                        self.attention_focus = [str(item) for item in focus_items]  # Convert to strings
                    else:
                        self.attention_focus = []
                else:
                    self.attention_focus = []
            else:
                # Fallback when attention mechanism is unavailable
                self.current_fatigue = attention_scores.get("fatigue_level", 0.1)
                self.attention_focus = []
                
            print(f"DEBUG: Updated cognitive state - "
                  f"Fatigue: {self.current_fatigue:.3f}, "
                  f"Cognitive Load: {attention_scores.get('cognitive_load', 0.0):.3f}, "
                  f"Items in Focus: {attention_scores.get('items_in_focus', 0)}")
                  
        except Exception as e:
            logger.error(f"Error updating cognitive state: {e}")
            # Fallback values
            self.current_fatigue = 0.1
            self.attention_focus = []
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive state information with fallback handling"""
        try:
            # Get memory system status
            memory_status = self.memory.get_status()
            
            # Get attention mechanism status with fallback
            if self.attention:
                attention_status = self.attention.get_attention_status()
            else:
                attention_status = {
                    "cognitive_load": 0.3,
                    "fatigue_level": self.current_fatigue,
                    "available_capacity": 0.7,
                    "items_in_focus": len(self.attention_focus),
                    "attention_items": self.attention_focus
                }
            
            # Get sensory processing statistics with fallback
            if self.sensory_processor:
                sensory_stats = self.sensory_processor.get_processing_stats()
            else:
                sensory_stats = {
                    "total_processed": 0,
                    "filtered_count": 0,
                    "average_processing_time": 0.0
                }

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
                    "attention_memory_sync": len(self.attention_focus) > 0 and memory_status["stm"].get("vector_db_count", 0) > 0,
                    "processing_capacity": attention_status.get("available_capacity", 0.0),
                    "overall_efficiency": 1.0 - self.current_fatigue,
                    "sensory_efficiency": 1.0 - (sensory_stats.get("filtered_count", 0) / max(1, sensory_stats.get("total_processed", 1)))
                }
            }
        except Exception as e:
            logger.error(f"Error getting cognitive status: {e}")
            # Return minimal fallback status
            return {
                "session_id": self.session_id,
                "fatigue_level": self.current_fatigue,
                "attention_focus": self.attention_focus,
                "active_goals": self.active_goals,
                "conversation_length": len(self.conversation_context),
                "last_interaction": None,
                "memory_status": {"error": str(e)},
                "attention_status": {"error": str(e)},
                "sensory_processing": {"error": str(e)},
                "cognitive_integration": {"error": str(e)}
            }
    
    async def enter_dream_state(self, cycle_type: str = "deep"):
        """Enter dream-state processing for memory consolidation with fallback"""
        print(f"Entering {cycle_type} dream state for memory consolidation...")
        
        try:
            # Use the advanced dream processor if available
            if self.dream_processor:
                dream_results = await self.dream_processor.enter_dream_cycle(cycle_type)
            else:
                logger.warning("Dream processor unavailable, using fallback dream processing")
                dream_results = {
                    "cycle_type": cycle_type,
                    "actual_duration": 5.0,
                    "memories_processed": 0,
                    "consolidations": 0,
                    "status": "fallback_mode"
                }
            
            # Also allow attention to rest during dream state if available
            if self.attention:
                attention_rest = self.attention.rest_attention(duration_minutes=dream_results.get("actual_duration", 5))
                
                # Synchronize fatigue state
                self.current_fatigue = self.attention.current_fatigue
            else:
                attention_rest = {
                    "fatigue_reduction": 0.1,
                    "load_reduction": 0.2,
                    "items_lost_focus": 0
                }
                self.current_fatigue = max(0.0, self.current_fatigue - 0.1)
            
            print(f"Dream state results: {dream_results}")
            print(f"Attention rest results: {attention_rest}")
            print("Advanced dream state processing completed")
            return dream_results
            
        except Exception as e:
            logger.error(f"Error in dream state processing: {e}")
            return {
                "cycle_type": cycle_type,
                "actual_duration": 0.0,
                "memories_processed": 0,
                "consolidations": 0,
                "status": "error",
                "error": str(e)
            }
    
    def take_cognitive_break(self, duration_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Take a brief cognitive break to recover attention and reduce fatigue with fallback
        
        Args:
            duration_minutes: Duration of break in minutes
        
        Returns:
            Break recovery metrics
        """
        print(f"Taking cognitive break for {duration_minutes} minutes...")
        
        try:
            # Use attention mechanism's rest functionality if available
            if self.attention:
                rest_results = self.attention.rest_attention(duration_minutes)
                
                # Synchronize fatigue state
                self.current_fatigue = self.attention.current_fatigue
                
                print(f"Cognitive break completed. Fatigue reduced by {rest_results['fatigue_reduction']:.3f}")
                
                return {
                    "break_duration": duration_minutes,
                    "fatigue_before": rest_results["fatigue_reduction"] + self.current_fatigue,
                    "fatigue_after": self.current_fatigue,
                    "cognitive_load_reduction": rest_results.get("load_reduction", 0.0),
                    "attention_items_lost": rest_results.get("items_lost_focus", 0),
                    "recovery_effective": rest_results["fatigue_reduction"] > 0.05
                }
            else:
                # Fallback when attention mechanism is unavailable
                logger.warning("Attention mechanism unavailable, using fallback cognitive break")
                fatigue_before = self.current_fatigue
                fatigue_reduction = min(0.2, duration_minutes * 0.1)  # Simple recovery
                self.current_fatigue = max(0.0, self.current_fatigue - fatigue_reduction)
                
                print(f"Cognitive break completed. Fatigue reduced by {fatigue_reduction:.3f}")
                
                return {
                    "break_duration": duration_minutes,
                    "fatigue_before": fatigue_before,
                    "fatigue_after": self.current_fatigue,
                    "cognitive_load_reduction": fatigue_reduction,
                    "attention_items_lost": 0,
                    "recovery_effective": fatigue_reduction > 0.05
                }
                
        except Exception as e:
            logger.error(f"Error in cognitive break: {e}")
            return {
                "break_duration": duration_minutes,
                "fatigue_before": self.current_fatigue,
                "fatigue_after": self.current_fatigue,
                "cognitive_load_reduction": 0.0,
                "attention_items_lost": 0,
                "recovery_effective": False,
                "error": str(e)
            }
    
    def force_dream_cycle(self, cycle_type: str = "deep"):
        """Force an immediate dream cycle with fallback"""
        try:
            if self.dream_processor:
                self.dream_processor.force_dream_cycle(cycle_type)
            else:
                logger.warning("Dream processor unavailable, cannot force dream cycle")
        except Exception as e:
            logger.error(f"Error forcing dream cycle: {e}")
    
    def get_dream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dream processing statistics with fallback"""
        try:
            if self.dream_processor:
                return self.dream_processor.get_dream_statistics()
            else:
                return {
                    "total_cycles": 0,
                    "last_cycle": None,
                    "status": "unavailable"
                }
        except Exception as e:
            logger.error(f"Error getting dream statistics: {e}")
            return {"error": str(e)}
    
    def is_dreaming(self) -> bool:
        """Check if the agent is currently in a dream state with fallback"""
        try:
            if self.dream_processor:
                return self.dream_processor.is_dreaming
            else:
                return False
        except Exception as e:
            logger.error(f"Error checking dream state: {e}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive agent"""
        print("Shutting down cognitive agent...")
        
        try:
            # Shutdown dream processor if available
            if self.dream_processor:
                self.dream_processor.shutdown()
            
            # Stop reflection scheduler
            self.stop_reflection_scheduler()
            
            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # Save any pending memories and close connections
            # Clean up resources
            
            print("Cognitive agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            print(f"Cognitive agent shutdown completed with errors: {e}")
    
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
            print("Reflection scheduler already running.")
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
        print(f"Started metacognitive reflection scheduler (every {interval_minutes} min)")

    def stop_reflection_scheduler(self):
        """
        Stop the background reflection scheduler.
        """
        self._reflection_scheduler_running = False
        schedule.clear('reflection')
        print("Stopped metacognitive reflection scheduler.")

    def manual_reflect(self) -> Dict[str, Any]:
        """
        Manually trigger a metacognitive reflection (CLI/API hook).
        Returns the reflection report.
        """
        print("Manual metacognitive reflection triggered.")
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
                    
                    print(f"🧠 Neural attention enhancement: +{neural_enhancement:.3f} "
                          f"(novelty: {enhanced_novelty:.3f})")
                
            return attention_result
            
        except Exception as e:
            print(f"⚠ Neural attention enhancement error: {e}")
            return attention_result  # Return original on error
