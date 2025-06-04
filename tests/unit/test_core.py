"""
Basic tests for the cognitive framework core functionality
"""
import pytest
import asyncio
from datetime import datetime

from src.core import CognitiveAgent, CognitiveConfig
from src.utils import generate_memory_id, calculate_attention_score

class TestCognitiveFramework:
    """Test suite for basic cognitive framework functionality"""
    
    def test_config_creation(self):
        """Test that configuration can be created"""
        config = CognitiveConfig()
        assert config.memory.stm_capacity == 100
        assert config.attention.max_attention_items == 7
        assert config.processing.embedding_model == "all-MiniLM-L6-v2"
    
    def test_memory_id_generation(self):
        """Test memory ID generation"""
        content = "Test memory content"
        timestamp = datetime.now()
        
        id1 = generate_memory_id(content, timestamp)
        id2 = generate_memory_id(content, timestamp)
        
        # Same content and timestamp should generate same ID
        assert id1 == id2
        assert len(id1) == 16  # Should be 16 characters
    
    def test_attention_score_calculation(self):
        """Test attention score calculation"""
        score = calculate_attention_score(
            relevance=0.8,
            novelty=0.6,
            emotional_salience=0.0,
            current_fatigue=0.0
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high for high relevance
    
    def test_cognitive_agent_initialization(self):
        """Test that cognitive agent can be initialized"""
        agent = CognitiveAgent()
        
        assert agent.session_id is not None
        assert agent.current_fatigue == 0.0
        assert isinstance(agent.conversation_context, list)
        assert len(agent.conversation_context) == 0
    
    @pytest.mark.asyncio
    async def test_cognitive_agent_input_processing(self):
        """Test basic input processing"""
        agent = CognitiveAgent()
        
        response = await agent.process_input("Hello, how are you?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(agent.conversation_context) == 1
    
    def test_cognitive_status(self):
        """Test cognitive status reporting"""
        agent = CognitiveAgent()
        status = agent.get_cognitive_status()
        
        assert "session_id" in status
        assert "fatigue_level" in status
        assert "conversation_length" in status
        assert status["conversation_length"] == 0

if __name__ == "__main__":
    # Run basic tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
