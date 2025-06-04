"""
Core Cognitive Agent - Central orchestrator for the cognitive architecture
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import torch
import numpy as np

from .config import CognitiveConfig
from ..memory.memory_system import MemorySystem
from ..attention.attention_mechanism import AttentionMechanism
from ..processing.sensory import SensoryInterface, SensoryProcessor
from ..processing.dream import DreamProcessor

class CognitiveAgent:
    """
    Central cognitive agent that orchestrates all cognitive processes
    
    This class implements the main cognitive loop:
    1. Input processing through sensory buffer
    2. Memory retrieval and context building
    3. Attention allocation and focus management
    4. Response generation and memory consolidation
    """
    
    def __init__(self, config: Optional[CognitiveConfig] = None):
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
        self.conversation_context = []
        
        print(f"Cognitive agent initialized with session ID: {self.session_id}")
    
    def _initialize_components(self):
        """Initialize all cognitive architecture components"""        # Memory systems
        self.memory = MemorySystem(
            stm_capacity=self.config.memory.stm_capacity,
            stm_decay_threshold=self.config.memory.stm_decay_threshold,
            ltm_storage_path=self.config.memory.ltm_storage_path
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(
            max_attention_items=self.config.attention.max_attention_items,
            salience_threshold=self.config.attention.salience_threshold,
            fatigue_decay_rate=self.config.attention.fatigue_decay_rate,
            attention_recovery_rate=self.config.attention.attention_recovery_rate        )        # Sensory processing interface
        self.sensory_processor = SensoryProcessor()
        self.sensory_interface = SensoryInterface(self.sensory_processor)

        # Neural integration manager (DPAD)
        try:
            from ..processing.neural import NeuralIntegrationManager
            self.neural_integration = NeuralIntegrationManager(
                cognitive_config=self.config,
                model_save_path="./data/models/dpad"
            )
            print("✓ Neural integration (DPAD) initialized")
        except ImportError as e:
            print(f"⚠ Neural integration disabled: {e}")
            self.neural_integration = None

        # Dream processor (initialized after memory system and neural integration)
        self.dream_processor = DreamProcessor(
            memory_system=self.memory,
            enable_scheduling=True,
            consolidation_threshold=0.6,
            neural_integration_manager=self.neural_integration
        )

        print("Cognitive components initialized")
    
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
            print(f"Processing {input_type} input: {input_data[:100]}...")
            
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
            print(f"Error in cognitive processing: {e}")
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
            print(f"Error in sensory processing: {e}")
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
        """Retrieve relevant memories for context building"""
        try:
            # Use memory system to search for relevant context
            memories = self.memory.search_memories(
                query=processed_input["raw_input"],
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
            print(f"Error retrieving memory context: {e}")
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
        if len(processed_input.get("raw_input", "")) > 100:            effort_required = 0.7  # Longer inputs require more effort
        
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
        # Enhanced placeholder that includes memory context
        context_info = ""
        if memory_context:
            context_info = f" (I found {len(memory_context)} related memories)"
        
        return f"I understand you said: '{processed_input['raw_input']}'. This is a placeholder response{context_info}."
    
    async def _consolidate_memory(
        self,
        input_data: str,
        response: str,
        attention_scores: Dict[str, float]
    ):
        """Consolidate the interaction into memory"""
        # Create memory ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Store conversation in memory system
        conversation_memory = {
            "interaction": {
                "input": input_data,
                "response": response,
                "timestamp": datetime.now()
            },
            "cognitive_state": {
                "attention_score": attention_scores["overall_attention"],
                "fatigue_level": self.current_fatigue,
                "relevance": attention_scores["relevance"]
            }
        }
        
        # Store in memory system with appropriate importance
        self.memory.store_memory(
            memory_id=memory_id,
            content=conversation_memory,
            importance=attention_scores["relevance"],
            attention_score=attention_scores["overall_attention"],
            emotional_valence=attention_scores["emotional_salience"],
            memory_type="episodic",
            tags=["conversation", "interaction"]
        )
        
        # Also keep in temporary conversation context for immediate access
        memory_entry = {
            "id": memory_id,
            "input": input_data,
            "response": response,
            "timestamp": datetime.now(),
            "attention_score": attention_scores["overall_attention"],
            "importance": attention_scores["relevance"]
        }
        
        self.conversation_context.append(memory_entry)
        
        # Keep only recent context (temporary until proper memory system)
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    def _update_cognitive_state(self, attention_scores: Dict[str, float]):
        """Update cognitive state based on processing"""
        # The attention mechanism handles its own fatigue and state updates
        # Just synchronize our local fatigue with the attention mechanism
        self.current_fatigue = self.attention.current_fatigue
        
        # Update attention focus list from attention mechanism
        self.attention_focus = self.attention.get_attention_focus()
        
        print(f"DEBUG: Updated cognitive state - "
              f"Fatigue: {self.current_fatigue:.3f}, "
              f"Cognitive Load: {attention_scores.get('cognitive_load', 0.0):.3f}, "              f"Items in Focus: {attention_scores.get('items_in_focus', 0)}")
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive state information"""
        # Get memory system status
        memory_status = self.memory.get_memory_status()
          # Get attention mechanism status
        attention_status = self.attention.get_attention_status()
        
        # Get sensory processing statistics
        sensory_stats = self.sensory_processor.get_processing_stats()

        return {
            "session_id": self.session_id,
            "fatigue_level": self.current_fatigue,
            "attention_focus": self.attention_focus,
            "active_goals": self.active_goals,
            "conversation_length": len(self.conversation_context),
            "last_interaction": self.conversation_context[-1]["timestamp"] if self.conversation_context else None,            "memory_status": memory_status,
            "attention_status": attention_status,
            "sensory_processing": sensory_stats,
            "cognitive_integration": {
                "attention_memory_sync": len(self.attention_focus) > 0 and memory_status["stm"]["size"] > 0,
                "processing_capacity": attention_status.get("available_capacity", 0.0),                "overall_efficiency": 1.0 - self.current_fatigue,
                "sensory_efficiency": 1.0 - (sensory_stats.get("filtered_count", 0) / max(1, sensory_stats.get("total_processed", 1)))
            }
        }
    
    async def enter_dream_state(self, cycle_type: str = "deep"):
        """Enter dream-state processing for memory consolidation"""
        print(f"Entering {cycle_type} dream state for memory consolidation...")
        
        # Use the advanced dream processor
        dream_results = await self.dream_processor.enter_dream_cycle(cycle_type)
        
        # Also allow attention to rest during dream state
        attention_rest = self.attention.rest_attention(duration_minutes=dream_results.get("actual_duration", 5))
        
        print(f"Dream state results: {dream_results}")
        print(f"Attention rest results: {attention_rest}")
        
        # Synchronize fatigue state
        self.current_fatigue = self.attention.current_fatigue
        
        print("Advanced dream state processing completed")
        return dream_results
    
    def take_cognitive_break(self, duration_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Take a brief cognitive break to recover attention and reduce fatigue
        
        Args:
            duration_minutes: Duration of break in minutes
        
        Returns:
            Break recovery metrics
        """
        print(f"Taking cognitive break for {duration_minutes} minutes...")
        
        # Use attention mechanism's rest functionality
        rest_results = self.attention.rest_attention(duration_minutes)
        
        # Synchronize fatigue state
        self.current_fatigue = self.attention.current_fatigue
        
        print(f"Cognitive break completed. Fatigue reduced by {rest_results['fatigue_reduction']:.3f}")
        
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
        print("Shutting down cognitive agent...")
        
        # Shutdown dream processor
        self.dream_processor.shutdown()
        
        # Save any pending memories
        # Close connections
        # Clean up resources
        
        print("Cognitive agent shutdown complete")
    
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
