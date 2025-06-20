import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
from src.core.cognitive_agent import CognitiveAgent

@pytest.mark.asyncio
async def test_cognitive_agent_stm_recall():
    agent = CognitiveAgent()
    # Store a unique fact in STM only
    unique_fact = "The STM test codeword is: ALPHA-OMEGA-42."
    agent.memory.stm.store(memory_id="stm_test", content=unique_fact)
    # Ask the agent to recall the codeword
    user_input = "What is the STM test codeword?"
    response = await agent.process_input(user_input)
    print(f"STM Recall Response: {response}")
    assert "ALPHA-OMEGA-42" in response or "stm" in response.lower()

@pytest.mark.asyncio
async def test_cognitive_agent_ltm_recall():
    agent = CognitiveAgent()
    # Store a unique fact in LTM only
    unique_fact = "The LTM test codeword is: BETA-DELTA-99."
    agent.memory.ltm.store(memory_id="ltm_test", content=unique_fact)
    # Ask the agent to recall the codeword
    user_input = "What is the LTM test codeword?"
    response = await agent.process_input(user_input)
    print(f"LTM Recall Response: {response}")
    assert "BETA-DELTA-99" in response or "ltm" in response.lower()
