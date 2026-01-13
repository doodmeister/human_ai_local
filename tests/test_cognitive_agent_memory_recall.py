import pytest
from src.core.cognitive_agent import CognitiveAgent

@pytest.mark.asyncio
async def test_cognitive_agent_stm_recall():
    agent = CognitiveAgent()
    # Store a unique fact in STM only
    unique_fact = "The STM test codeword is: ALPHA-OMEGA-42."
    agent.memory.stm.store(memory_id="stm_test", content=unique_fact)
    # Deterministic: validate memory retrieval returns the stored fact.
    user_input = "What is the STM test codeword?"
    context = await agent._retrieve_memory_context({"raw_input": user_input})
    assert any("ALPHA-OMEGA-42" in item.get("content", "") for item in context)
    assert any(item.get("source") == "STM" for item in context)

@pytest.mark.asyncio
async def test_cognitive_agent_ltm_recall():
    agent = CognitiveAgent()
    # Store a unique fact in LTM only
    unique_fact = "The LTM test codeword is: BETA-DELTA-99."
    agent.memory.ltm.store(memory_id="ltm_test", content=unique_fact)
    # Deterministic: validate memory retrieval returns the stored fact.
    user_input = "What is the LTM test codeword?"
    context = await agent._retrieve_memory_context({"raw_input": user_input})
    assert any("BETA-DELTA-99" in item.get("content", "") for item in context)
    assert any(item.get("source") == "LTM" for item in context)
