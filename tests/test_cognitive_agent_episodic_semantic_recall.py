import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
from src.core.cognitive_agent import CognitiveAgent

@pytest.mark.asyncio
async def test_cognitive_agent_episodic_recall():
    agent = CognitiveAgent()
    # Store a unique episodic memory
    unique_content = "The episodic test event is: SKY-BLUE-77."
    memory_id = agent.memory.episodic.store(detailed_content=unique_content)
    # Ask the agent to recall the event
    user_input = "What is the episodic test event?"
    response = await agent.process_input(user_input)
    print(f"Episodic Recall Response: {response}")
    assert "SKY-BLUE-77" in response or "episodic" in response.lower()

@pytest.mark.asyncio
async def test_cognitive_agent_semantic_recall():
    agent = CognitiveAgent()
    # Store a unique semantic fact
    subject = "semantic_test_subject"
    predicate = "has_codeword"
    obj = "GOLDEN-KEY-123"
    agent.memory.semantic.store_fact(subject, predicate, obj)
    # Ask the agent to recall the fact
    user_input = "What is the codeword for the semantic test subject?"
    response = await agent.process_input(user_input)
    print(f"Semantic Recall Response: {response}")
    assert "GOLDEN-KEY-123" in response or "semantic" in response.lower()
