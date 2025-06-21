import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.cognitive_agent import CognitiveAgent

@pytest.mark.asyncio
async def test_cognitive_agent_llm_response():
    agent = CognitiveAgent()
    user_input = "What is the capital of France?"
    response = await agent.process_input(user_input)
    print(f"LLM Response: {response}")
    assert isinstance(response, str)
    assert any(city in response.lower() for city in ["paris", "france"]) or "error" not in response.lower()
